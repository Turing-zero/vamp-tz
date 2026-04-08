"""
示例脚本：从环境配置文件初始化 `vamp.Environment` 并规划。

运行：
  conda activate vamp && python scripts/toilet.py --env_config resources/env_specs/toilet_example.yaml

坐标系约定：
- 环境配置（collision/visual）默认在**世界系**。
- 本脚本通过 `world_to_robot_*` 提供 `T_world_to_robot`，将环境整体转到**机械臂基座系**后再做碰撞/规划。
"""

import numpy as np
from pathlib import Path

from viser_utils import (
    setup_viser_with_robot,
    add_trajectory,
    )

import vamp
from vamp.transformations import concatenate_matrices, quaternion_matrix, translation_matrix

from fire import Fire

# Starting configuration
a = [0., 1.57, -1.57, 0., 0., 0.]

# Goal configuration
b = [0.5, 1.57, 0.0, 0., 0., 0.]

# 可选：额外球形障碍物中心（与 PLY 合并）
# problem = [
#     [0.35, -0.55, 0.25],
#     [0.35, 0.35, 0.8],
#     ]

def euler_xyz_deg_to_rotation_matrix_4x4(euler_xyz_deg: list[float]) -> np.ndarray:
    """内旋 XYZ（度）：先绕固定初始系的 X，再绕新 Y，再绕新 Z。

    与 `scipy.spatial.transform.Rotation.from_euler('xyz', ..., degrees=True)` 一致。
    """
    try:
        from scipy.spatial.transform import Rotation as Rsc  # type: ignore
    except Exception as ex:  # pragma: no cover
        raise RuntimeError(
            "将欧拉角（度）转为旋转矩阵需要 scipy（conda vamp 环境通常已安装）。"
        ) from ex

    T = np.eye(4, dtype = np.float64)
    T[:3, :3] = Rsc.from_euler(
        "xyz",
        np.asarray(euler_xyz_deg, dtype = np.float64).reshape(3),
        degrees = True,
        ).as_matrix()
    return T


def toilet_to_world_matrix(
    translation_xyz: list[float],
    *,
    euler_xyz_deg: list[float] | None = None,
    quat_xyzw: list[float] | None = None,
    ) -> np.ndarray:
    """马桶系 → 世界系：``p' = R @ p + t``。

    - **默认**用 ``euler_xyz_deg``（度，内旋 XYZ）。
    - 若传入 ``quat_xyzw``（xyzw），则**优先用四元数**，忽略欧拉角。
    """
    t = np.asarray(translation_xyz, dtype = np.float64).reshape(3)
    Tt = translation_matrix(t)
    if quat_xyzw is not None:
        q = np.asarray(quat_xyzw, dtype = np.float64).reshape(4)
        R4 = quaternion_matrix(q)
    else:
        e = euler_xyz_deg if euler_xyz_deg is not None else [0.0, 0.0, 0.0]
        R4 = euler_xyz_deg_to_rotation_matrix_4x4(e)
    return concatenate_matrices(Tt, R4)


def main(
    env_config: str,
    world_to_robot_translation: list[float] = [0.0, 0.0, 0.0],
    world_to_robot_euler_xyz_deg: list[float] = [0.0, 0.0, 0.0],
    planner: str = "rrtc",
    **kwargs,
    ):
    # 世界系 -> 机械臂基座系
    T_world_to_robot = toilet_to_world_matrix(world_to_robot_translation, euler_xyz_deg=world_to_robot_euler_xyz_deg)

    (vamp_module, planner_func, plan_settings,
     simp_settings) = (vamp.configure_robot_and_planner_with_kwargs("tz", planner, **kwargs))

    robot_dir = Path(__file__).parents[1] / "resources" / "tz"
    server, robot = setup_viser_with_robot(robot_dir, "tz_spherized.urdf")
    robot.update_cfg(a)

    # ---- 1) 用封装的类初始化环境（env_config 里不再包含 plane/ply/gs 的读取逻辑） ----
    cfg = vamp.VAMPEndEnvConfig.init_from_config(env_config, robot_module=vamp_module)

    cfg.set_world_to_robot({"matrix4x4": T_world_to_robot.tolist()})

    # ---- 2) 构建 Environment + 视觉 ----
    e = cfg.get_environment()
    visuals = cfg.get_viser_visual()
    vamp.add_visuals_to_viser(server, visuals)

    # Plan and display
    sampler = vamp_module.halton()
    result = planner_func(a, b, e, plan_settings, sampler)
    simple = vamp_module.simplify(result.path, e, simp_settings, sampler)
    simple.path.interpolate_to_resolution(vamp.tz.resolution())

    add_trajectory(server, simple.path.numpy(), robot, [], [[]])

    # display
    while True:
        continue


if __name__ == "__main__":
    Fire(main)
