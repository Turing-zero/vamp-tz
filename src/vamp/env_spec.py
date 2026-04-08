from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

import vamp
from vamp.pointcloud import transform_in_place
from vamp.transformations import (
    concatenate_matrices,
    euler_from_matrix,
    euler_matrix,
    quaternion_from_matrix,
    quaternion_matrix,
    translation_matrix,
)


EnvSpecFormat = Literal["json", "yaml", "yml"]


@dataclass(frozen=True)
class VisualAsset:
    """一个可延迟添加到 viser scene 的可视化资产。"""

    kind: Literal["pointcloud", "gaussian_splats", "mesh_vertices", "cuboid"]
    name: str
    payload: dict[str, Any]


def _as_T_4x4(x: Any) -> np.ndarray:
    T = np.asarray(x, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"需要 4x4 变换矩阵，但得到 {T.shape}")
    return T


def _compose_T(global_T: np.ndarray | None, local_tf: dict[str, Any] | None) -> np.ndarray:
    """
    组合规则：
    - local_tf 先作用
    - global_T 后作用
    即：T_total = global_T @ T_local
    """
    T_local = _tf_from_spec(local_tf)
    if global_T is None:
        return T_local
    return _as_T_4x4(global_T) @ T_local


def _transform_point(p_xyz: Sequence[float], T: np.ndarray) -> list[float]:
    p = _as_f64_vec(p_xyz, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    out = (R @ p + t).astype(np.float64)
    return [float(out[0]), float(out[1]), float(out[2])]


def _transform_rotation_from_euler_xyz(euler_xyz: Sequence[float], T: np.ndarray) -> list[float]:
    """
    把局部姿态（Euler XYZ, rad）在刚体变换 T 下更新：R' = R_T * R_local。
    返回新的 Euler XYZ（rad）。
    """
    e = _as_f64_vec(euler_xyz, 3)
    R_local_4 = euler_matrix(float(e[0]), float(e[1]), float(e[2]), axes="sxyz")
    R_new = (T @ R_local_4)[:3, :3]
    ex, ey, ez = euler_from_matrix(R_new, axes="sxyz")
    return [float(ex), float(ey), float(ez)]


@dataclass
class VAMPEndEnvConfig:
    """
    一个“配置驱动”的 VAMP 环境封装：

    - `init_from_config()`：读取 json/yaml 文件
    - `get_environment()`：返回已做坐标变换后的 `vamp.Environment`
    - `get_viser_visual()`：返回 visual 资产（可交给 `add_visuals_to_viser`）
    - `apply_transform()`：对点/点云/位姿矩阵应用 tf（便于你在外部复用）
    """

    spec: dict[str, Any]
    base_dir: Path
    robot_module: Any | None = None
    default_point_radius: float = 0.01
    T_world_to_robot: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    _cached_env: vamp.Environment | None = field(default=None, init=False, repr=False)
    _cached_visuals: list[VisualAsset] | None = field(default=None, init=False, repr=False)

    @classmethod
    def init_from_config(
        cls,
        path: str | Path,
        *,
        robot_module: Any | None = None,
        default_point_radius: float = 0.01,
    ) -> VAMPEndEnvConfig:
        p = Path(path)
        spec = load_env_spec(p)
        return cls(
            spec=spec,
            base_dir=p.parent,
            robot_module=robot_module,
            default_point_radius=float(default_point_radius),
        )

    def _resolve_path(self, maybe_path: str) -> str:
        mp = Path(maybe_path)
        if mp.is_absolute():
            return str(mp)
        return str((self.base_dir / mp).resolve())

    def apply_transform(self, data: Any, tf: dict[str, Any] | None) -> Any:
        """
        支持：
        - `data` 为 (N,3) 点云 ndarray：返回变换后的 ndarray
        - `data` 为 4x4 矩阵 ndarray：返回 `T @ data`
        - `data` 为 3D 点序列：返回 list[float]
        """
        T = _compose_T(self.T_world_to_robot, tf)

        if isinstance(data, np.ndarray) and data.shape == (4, 4):
            return (T @ data).astype(np.float64)

        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] == 3:
            pts = np.asarray(arr, dtype=np.float32)
            out = np.array(pts, copy=True)
            transform_in_place(out, T)
            return out

        if arr.shape == (3,):
            pts = np.asarray(arr, dtype=np.float32).reshape(1, 3)
            out = np.array(pts, copy=True)
            transform_in_place(out, T)
            return out.reshape(3).astype(np.float32).tolist()

        raise ValueError(f"apply_transform 不支持的数据形状: {arr.shape}")

    def set_world_to_robot(self, tf: dict[str, Any] | np.ndarray) -> None:
        """
        设置全局 `T_world_to_robot`。
        - 支持传入 tf spec：{"matrix4x4": ...} 或 {"translation": ..., "quat_xyzw": ...}
        - 或直接传入 4x4 ndarray
        """
        if isinstance(tf, dict):
            self.T_world_to_robot = _tf_from_spec(tf)
        else:
            self.T_world_to_robot = _as_T_4x4(tf)
        # 变换改变后，缓存失效
        self._cached_env = None
        self._cached_visuals = None

    def get_world_to_robot(self) -> np.ndarray:
        return np.asarray(self.T_world_to_robot, dtype=np.float64)

    def get_environment(self) -> vamp.Environment:
        if self._cached_env is None:
            env, visuals = build_environment_from_spec(
                self._spec_with_resolved_paths(),
                robot_module=self.robot_module,
                default_point_radius=self.default_point_radius,
                global_T_world_to_robot=self.T_world_to_robot,
            )
            self._cached_env = env
            self._cached_visuals = visuals
        return self._cached_env

    def get_viser_visual(self) -> list[VisualAsset]:
        if self._cached_visuals is None:
            _env, visuals = build_environment_from_spec(
                self._spec_with_resolved_paths(),
                robot_module=self.robot_module,
                default_point_radius=self.default_point_radius,
                global_T_world_to_robot=self.T_world_to_robot,
            )
            self._cached_visuals = visuals
        return self._cached_visuals

    def _spec_with_resolved_paths(self) -> dict[str, Any]:
        """
        将 spec 里出现的相对路径（pointcloud/mesh/gaussian/heightfield png）统一解析成绝对路径。
        """
        spec = json.loads(json.dumps(self.spec))  # 深拷贝（只处理 json/yaml 友好的类型）

        collision = spec.get("collision", {}) or {}
        visual = spec.get("visual", {}) or {}

        for pc in collision.get("pointcloud", []) or []:
            if "path" in pc:
                pc["path"] = self._resolve_path(pc["path"])

        for hf in collision.get("heightfield", []) or []:
            if "png_path" in hf:
                hf["png_path"] = self._resolve_path(hf["png_path"])

        for pcv in visual.get("pointcloud", []) or []:
            if "path" in pcv:
                pcv["path"] = self._resolve_path(pcv["path"])

        for g in visual.get("gaussian", []) or []:
            if "ply_path" in g:
                g["ply_path"] = self._resolve_path(g["ply_path"])

        for m in visual.get("mesh", []) or []:
            if "path" in m:
                m["path"] = self._resolve_path(m["path"])

        return spec


def load_env_spec(path: str | Path) -> dict[str, Any]:
    """
    从 JSON / YAML 读取环境定义。

    约定顶层包含两个字段：
    - collision: 用于构建 vamp.Environment
    - visual: 用于 viser 可视化（可选）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower().lstrip(".")
    if suffix == "json":
        return json.loads(p.read_text(encoding="utf-8"))
    if suffix in ("yaml", "yml"):
        try:
            import yaml  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError(
                "读取 YAML 需要 PyYAML。请 `pip install pyyaml`，或改用 .json 配置。"
            ) from ex
        return yaml.safe_load(p.read_text(encoding="utf-8"))

    raise ValueError(f"不支持的环境配置格式: {p.suffix}（只支持 .json/.yaml/.yml）")


def _as_f64_vec(x: Sequence[float], n: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(n)
    return a


def _tf_from_spec(tf: dict[str, Any] | None) -> np.ndarray:
    """
    支持：
    - {"matrix4x4": [[...],[...],[...],[...]]}
    - {"translation": [x,y,z], "quat_xyzw": [x,y,z,w]}
    """
    if tf is None:
        return np.eye(4, dtype=np.float64)

    if "matrix4x4" in tf:
        T = np.asarray(tf["matrix4x4"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"tf.matrix4x4 需要 4x4，但得到 {T.shape}")
        return T

    t = _as_f64_vec(tf.get("translation", [0.0, 0.0, 0.0]), 3)
    if "quat_xyzw" in tf:
        q = _as_f64_vec(tf["quat_xyzw"], 4)
        R4 = quaternion_matrix(q)
    else:
        # 允许省略旋转
        R4 = np.eye(4, dtype=np.float64)
    return concatenate_matrices(translation_matrix(t), R4)


def _load_pointcloud_points(pc: dict[str, Any]) -> np.ndarray:
    """
    pointcloud 支持以下输入：
    - {"path": "...ply", "format": "ply_mesh_vertices"}：读取三角网格顶点（与 scripts/toilet.py 一致）
    - {"path": "...npy", "format": "npy_xyz"}：读取 (N,3) float32/float64
    - {"points": [[x,y,z], ...]}：内嵌点
    """
    if "points" in pc:
        pts = np.asarray(pc["points"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pointcloud.points 需要 (N,3)，但得到 {pts.shape}")
        return pts

    path = Path(pc["path"])
    fmt = pc.get("format", None)
    if fmt is None:
        fmt = path.suffix.lower().lstrip(".")

    if fmt in ("npy_xyz", "npy"):
        pts = np.load(path)
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pointcloud npy 需要 (N,3)，但得到 {pts.shape} from {path}")
        return pts

    if fmt in ("ply_mesh_vertices", "ply"):
        try:
            import open3d as o3d  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("读取 PLY 网格顶点需要 open3d。") from ex
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_vertices() or len(mesh.vertices) == 0:
            raise RuntimeError(f"PLY 无有效顶点: {path}")
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        return pts

    raise ValueError(f"不支持的 pointcloud.format: {fmt}")


def _maybe_downsample_points(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0:
        return pts
    try:
        import open3d as o3d  # type: ignore
    except Exception as ex:  # pragma: no cover
        raise RuntimeError("点云体素下采样需要 open3d。") from ex
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pcd = pcd.voxel_down_sample(float(voxel_size))
    return np.asarray(pcd.points, dtype=np.float32)


def _maybe_subsample_points(pts: np.ndarray, max_points: int | None, seed: int = 0) -> np.ndarray:
    if max_points is None or max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    return np.asarray(pts[idx], dtype=np.float32)


def _load_gaussian_splats_ply(ply_path: str | Path) -> dict[str, Any]:
    """
    读取 3DGS PLY 为 viser 所需字段。
    返回 payload：centers, rgbs, opacities, covariances。
    """
    try:
        from plyfile import PlyData  # type: ignore
    except Exception as ex:  # pragma: no cover
        raise RuntimeError(
            "加载 gaussian splats 需要 plyfile（例如 `pip install plyfile`）。"
        ) from ex

    SH_C0 = 0.28209479177387814
    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]

    centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)).astype(np.float32)
    wxyz = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    rgbs = (0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)).astype(np.float32)
    rgbs = np.clip(rgbs, 0.0, 1.0)

    opacities = (1.0 / (1.0 + np.exp(-v["opacity"][:, None]))).astype(np.float32)

    w, x, y, z = (wxyz[:, 0], wxyz[:, 1], wxyz[:, 2], wxyz[:, 3])
    inv_norm = 1.0 / np.sqrt(np.maximum(1e-12, w * w + x * x + y * y + z * z))
    w, x, y, z = (w * inv_norm, x * inv_norm, y * inv_norm, z * inv_norm)

    Rs = np.empty((wxyz.shape[0], 3, 3), dtype=np.float32)
    Rs[:, 0, 0] = 1 - 2 * (y * y + z * z)
    Rs[:, 0, 1] = 2 * (x * y - z * w)
    Rs[:, 0, 2] = 2 * (x * z + y * w)
    Rs[:, 1, 0] = 2 * (x * y + z * w)
    Rs[:, 1, 1] = 1 - 2 * (x * x + z * z)
    Rs[:, 1, 2] = 2 * (y * z - x * w)
    Rs[:, 2, 0] = 2 * (x * z - y * w)
    Rs[:, 2, 1] = 2 * (y * z + x * w)
    Rs[:, 2, 2] = 1 - 2 * (x * x + y * y)

    covariances = np.einsum(
        "nij,njk,nlk->nil",
        Rs,
        np.eye(3, dtype=np.float32)[None, :, :] * (scales[:, None, :] ** 2),
        Rs,
        dtype=np.float32,
    )

    return {
        "centers": centers,
        "rgbs": rgbs,
        "opacities": opacities,
        "covariances": covariances,
    }


def build_environment_from_spec(
    spec: dict[str, Any],
    *,
    robot_module: Any | None = None,
    default_point_radius: float = 0.01,
    global_T_world_to_robot: np.ndarray | None = None,
) -> tuple[vamp.Environment, list[VisualAsset]]:
    """
    解析 spec（已是 dict）并初始化 `vamp.Environment`，同时返回 visual 资产列表。

    - `robot_module`: 用于在 pointcloud 未显式给出 r_min/r_max 时，提供 `min_max_radii()`。
    """
    collision = spec.get("collision", {}) or {}
    visual = spec.get("visual", {}) or {}

    env = vamp.Environment()

    global_T_world_to_robot = (
        np.eye(4, dtype=np.float64)
        if global_T_world_to_robot is None
        else _as_T_4x4(global_T_world_to_robot)
    )

    # ---- collision: pointcloud ----
    for i, pc in enumerate(collision.get("pointcloud", []) or []):
        pts = _load_pointcloud_points(pc)
        pts = _maybe_downsample_points(pts, float(pc.get("voxel_downsample", 0.0) or 0.0))
        pts = _maybe_subsample_points(pts, pc.get("max_points", None), seed=int(pc.get("seed", 0) or 0))

        T = _compose_T(global_T_world_to_robot, pc.get("tf", None))
        pts2 = np.array(pts, copy=True)
        transform_in_place(pts2, T)

        if "r_min" in pc and "r_max" in pc:
            r_min = float(pc["r_min"])
            r_max = float(pc["r_max"])
        else:
            if robot_module is None or not hasattr(robot_module, "min_max_radii"):
                raise RuntimeError(
                    "pointcloud 未提供 r_min/r_max，且未提供 robot_module（无法推断半径范围）。"
                )
            r_min, r_max = robot_module.min_max_radii()

        r_point = float(pc.get("r_point", default_point_radius))
        env.add_pointcloud(pts2.tolist(), float(r_min), float(r_max), r_point)

    # ---- collision: spheres ----
    for i, s in enumerate(collision.get("sphere", []) or []):
        T = _compose_T(global_T_world_to_robot, s.get("tf", None))
        center = _transform_point(s["center"], T)
        radius = float(s["radius"])
        obj = vamp.Sphere(center, radius)
        obj.name = str(s.get("name", f"sphere_{i}"))
        env.add_sphere(obj)

    # ---- collision: cuboids ----
    for i, b in enumerate(collision.get("cuboid", []) or []):
        T = _compose_T(global_T_world_to_robot, b.get("tf", None))
        center = _transform_point(b["center"], T)
        half_extents = _as_f64_vec(b["half_extents"], 3).tolist()
        if "euler_xyz" in b:
            euler = _transform_rotation_from_euler_xyz(b["euler_xyz"], T)
        elif "euler_xyz_deg" in b:
            euler = _transform_rotation_from_euler_xyz(np.deg2rad(_as_f64_vec(b["euler_xyz_deg"], 3)), T)
        else:
            euler = _transform_rotation_from_euler_xyz([0.0, 0.0, 0.0], T)
        obj = vamp.Cuboid(center, euler, half_extents)
        obj.name = str(b.get("name", f"cuboid_{i}"))
        env.add_cuboid(obj)

    # ---- collision: capsules (vamp 里用 Cylinder 表示) ----
    for i, c in enumerate(collision.get("capsule", []) or []):
        T = _compose_T(global_T_world_to_robot, c.get("tf", None))
        radius = float(c["radius"])
        if "endpoints" in c:
            ep = np.asarray(c["endpoints"], dtype=np.float64).reshape(2, 3)
            p1 = _transform_point(ep[0].tolist(), T)
            p2 = _transform_point(ep[1].tolist(), T)
            obj = vamp.Cylinder(p1, p2, radius)
        else:
            center = _transform_point(c["center"], T)
            if "euler_xyz" in c:
                euler = _transform_rotation_from_euler_xyz(c["euler_xyz"], T)
            elif "euler_xyz_deg" in c:
                euler = _transform_rotation_from_euler_xyz(np.deg2rad(_as_f64_vec(c["euler_xyz_deg"], 3)), T)
            else:
                euler = _transform_rotation_from_euler_xyz([0.0, 0.0, 0.0], T)
            length = float(c["length"])
            obj = vamp.Cylinder(center, euler, radius, length)
        obj.name = str(c.get("name", f"capsule_{i}"))
        env.add_capsule(obj)

    # ---- collision: heightfield ----
    for i, h in enumerate(collision.get("heightfield", []) or []):
        T = _compose_T(global_T_world_to_robot, h.get("tf", None))
        R = T[:3, :3]
        if not np.allclose(R, np.eye(3), atol=1e-9):
            raise RuntimeError(
                "heightfield 暂不支持旋转变换（只能平移）。请改用 cuboid/mesh 点云等可旋转表示，"
                "或把 world_to_robot 设置为纯平移。"
            )
        center_tf = _transform_point(h["center"], T)
        if "png_path" in h:
            # 复用 vamp.png_to_heightfield（更方便）
            hf = vamp.png_to_heightfield(
                Path(h["png_path"]),
                tuple(center_tf),
                tuple(_as_f64_vec(h["scaling"], 3).tolist()),
            )
        else:
            hf = vamp.make_heightfield(
                center_tf,
                _as_f64_vec(h["scaling"], 3).tolist(),
                [int(x) for x in _as_f64_vec(h["shape"], 2).tolist()],
                [float(x) for x in np.asarray(h["data"], dtype=np.float64).reshape(-1).tolist()],
            )
        env.add_heightfield(hf)

    # ---- visual assets ----
    visuals: list[VisualAsset] = []
    # visual.pointcloud（仅显示）
    for i, pcv in enumerate(visual.get("pointcloud", []) or []):
        pts = _load_pointcloud_points(pcv)
        pts = _maybe_downsample_points(pts, float(pcv.get("voxel_downsample", 0.0) or 0.0))
        pts = _maybe_subsample_points(pts, pcv.get("max_points", None), seed=int(pcv.get("seed", 0) or 0))
        T = _compose_T(global_T_world_to_robot, pcv.get("tf", None))
        pts2 = np.array(pts, copy=True)
        transform_in_place(pts2, T)
        visuals.append(
            VisualAsset(
                kind="pointcloud",
                name=str(pcv.get("name", f"/visual/pointcloud_{i}")),
                payload={
                    "points": pts2,
                    "colors": pcv.get("colors", []),
                    "point_size": float(pcv.get("point_size", default_point_radius)),
                },
            )
        )

    # visual.gaussian（3DGS splats PLY）
    for i, g in enumerate(visual.get("gaussian", []) or []):
        payload = _load_gaussian_splats_ply(g["ply_path"])
        T = _compose_T(global_T_world_to_robot, g.get("tf", None))
        centers = np.array(payload["centers"], copy=True)
        transform_in_place(centers, T)
        payload["centers"] = centers
        # 协方差按旋转变换（只用 R）
        R = T[:3, :3].astype(np.float32)
        payload["covariances"] = np.einsum("ij,njk,lk->nil", R, payload["covariances"], R, dtype=np.float32)
        visuals.append(
            VisualAsset(
                kind="gaussian_splats",
                name=str(g.get("name", f"/visual/gaussian_{i}")),
                payload=payload,
            )
        )

    # visual.mesh（如果 viser 没有 mesh API，就退化成“顶点点云”）
    for i, m in enumerate(visual.get("mesh", []) or []):
        path = Path(m["path"])
        try:
            import open3d as o3d  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("读取 mesh 需要 open3d。") from ex
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_vertices() or len(mesh.vertices) == 0:
            raise RuntimeError(f"mesh 无有效顶点: {path}")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        T = _compose_T(global_T_world_to_robot, m.get("tf", None))
        verts2 = np.array(verts, copy=True)
        transform_in_place(verts2, T)
        visuals.append(
            VisualAsset(
                kind="mesh_vertices",
                name=str(m.get("name", f"/visual/mesh_vertices_{i}")),
                payload={
                    "points": verts2,
                    "colors": m.get("colors", [180, 180, 180]),
                    "point_size": float(m.get("point_size", default_point_radius)),
                },
            )
        )

    # visual.cuboid（直接画 box；不参与碰撞）
    for i, b in enumerate(visual.get("cuboid", []) or []):
        T = _compose_T(global_T_world_to_robot, b.get("tf", None))
        center = _transform_point(b["center"], T)
        half_extents = _as_f64_vec(b["half_extents"], 3).tolist()
        if "euler_xyz" in b:
            euler = _transform_rotation_from_euler_xyz(b["euler_xyz"], T)
        elif "euler_xyz_deg" in b:
            euler = _transform_rotation_from_euler_xyz(np.deg2rad(_as_f64_vec(b["euler_xyz_deg"], 3)), T)
        else:
            euler = _transform_rotation_from_euler_xyz([0.0, 0.0, 0.0], T)

        visuals.append(
            VisualAsset(
                kind="cuboid",
                name=str(b.get("name", f"/visual/cuboid_{i}")),
                payload={
                    "center": center,
                    "half_extents": half_extents,
                    "euler_xyz": euler,
                    "color": b.get("color", [160, 170, 200]),
                    "opacity": b.get("opacity", 0.22),
                    "wireframe": b.get("wireframe", True),
                },
            )
        )

    return env, visuals


def add_visuals_to_viser(server: Any, visuals: Iterable[VisualAsset]) -> list[Any]:
    """
    将 `build_environment_from_spec()` 返回的 visual 资产添加到 viser。
    返回 handle 列表（按输入顺序）。
    """
    handles: list[Any] = []
    for v in visuals:
        if v.kind == "pointcloud":
            handles.append(
                server.scene.add_point_cloud(
                    name=v.name,
                    points=v.payload["points"],
                    colors=v.payload.get("colors", []),
                    point_size=float(v.payload.get("point_size", 0.01)),
                )
            )
        elif v.kind == "gaussian_splats":
            handles.append(
                server.scene.add_gaussian_splats(
                    name=v.name,
                    centers=v.payload["centers"],
                    rgbs=v.payload["rgbs"],
                    opacities=v.payload["opacities"],
                    covariances=v.payload["covariances"],
                )
            )
        elif v.kind == "mesh_vertices":
            # 兼容：如果没有 mesh API，至少能看到形状轮廓
            handles.append(
                server.scene.add_point_cloud(
                    name=v.name,
                    points=v.payload["points"],
                    colors=v.payload.get("colors", [180, 180, 180]),
                    point_size=float(v.payload.get("point_size", 0.01)),
                )
            )
        elif v.kind == "cuboid":
            c = np.asarray(v.payload["center"], dtype=np.float64).reshape(3)
            h = np.asarray(v.payload["half_extents"], dtype=np.float64).reshape(3)
            dimensions = tuple(float(x * 2.0) for x in h.tolist())

            e = np.asarray(v.payload["euler_xyz"], dtype=np.float64).reshape(3)
            R4 = euler_matrix(float(e[0]), float(e[1]), float(e[2]), axes="sxyz")
            q_xyzw = quaternion_from_matrix(R4)
            wxyz_t = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))

            col = v.payload.get("color", [160, 170, 200])
            if isinstance(col, (list, tuple)) and len(col) == 3 and all(isinstance(x, (int, float)) for x in col):
                color = tuple(int(float(x)) for x in col)
            else:
                color = (160, 170, 200)

            handles.append(
                server.scene.add_box(
                    name=v.name,
                    dimensions=dimensions,
                    position=tuple(float(x) for x in c.tolist()),
                    wxyz=wxyz_t,
                    color=color,
                    opacity=float(v.payload.get("opacity", 0.22)),
                    wireframe=bool(v.payload.get("wireframe", True)),
                )
            )
        else:  # pragma: no cover
            raise RuntimeError(f"未知 visual kind: {v.kind}")
    return handles

