from viser.extras import ViserUrdf
import viser
import yourdfpy
from typing import Sequence, Union
import numpy as np
from scipy.spatial.transform import Rotation as R


def setup_viser_with_robot(robot_dir, robot_urdf_name):
    server = viser.ViserServer()
    # change the robot here
    urdf = yourdfpy.URDF.load(str(robot_dir / robot_urdf_name))
    robot = ViserUrdf(
        server,
        urdf,
        load_meshes = True,
        load_collision_meshes = False,
        root_node_name = "/robot",
        )

    return server, robot


def add_point_cloud(
    server: viser.ViserServer,
    point_cloud: np.ndarray,
    colors: Union[Sequence[int], Sequence[Sequence[int]]] = [],
    point_size: float = 0.01,
    prefix: str = "my_point_cloud",
    ):
    point_cloud_handle = server.scene.add_point_cloud(
        name = prefix,
        points = point_cloud,
        colors = colors,
        point_size = point_size,
        )
    return point_cloud_handle


def add_cuboid(
    server: viser.ViserServer,
    center: Union[Sequence[float], np.ndarray],
    half_extents: Union[Sequence[float], np.ndarray],
    *,
    rotation_matrix: np.ndarray | None = None,
    wxyz: Union[Sequence[float], np.ndarray, None] = None,
    color: tuple[int, int, int] = (160, 170, 200),
    opacity: float | None = 0.22,
    wireframe: bool = True,
    name: str = "cuboid",
    ):
    """在世界系中绘制与 `vamp.Cuboid` 一致的轴对齐盒子（先按半轴长缩放，再旋转、平移）。

    `half_extents` 与 `vamp.Cuboid` 的 `axis_*_r` 相同；`viser` 的 `dimensions` 为全长，故内部用 `2 * half_extents`。
    姿态为 `rotation_matrix`（3×3，列为局部 x/y/z 在世界系中的方向）或 `wxyz`（viser 约定）；二者均省略则为单位旋转。
    """
    c = np.asarray(center, dtype = np.float64).reshape(3)
    h = np.asarray(half_extents, dtype = np.float64).reshape(3)
    dimensions = tuple(float(x * 2.0) for x in h.tolist())
    position = tuple(float(x) for x in c.tolist())

    if wxyz is not None:
        wq = np.asarray(wxyz, dtype = np.float64).reshape(4)
        wxyz_t = tuple(float(x) for x in wq)
    elif rotation_matrix is not None:
        R = np.asarray(rotation_matrix, dtype = np.float64).reshape(3, 3)
        T4 = np.eye(4, dtype = np.float64)
        T4[:3, :3] = R
        q_xyzw = R.from_matrix(R).as_quat()
        wxyz_t = (
            float(q_xyzw[3]),
            float(q_xyzw[0]),
            float(q_xyzw[1]),
            float(q_xyzw[2]),
            )
    else:
        wxyz_t = (1.0, 0.0, 0.0, 0.0)

    return server.scene.add_box(
        name = name,
        dimensions = dimensions,
        position = position,
        wxyz = wxyz_t,
        color = tuple(int(x) for x in color),
        opacity = opacity,
        wireframe = wireframe,
        )

def add_spheres(
    server: viser.ViserServer,
    sphere_positions: Sequence,
    sphere_radii: Sequence,
    colors: Union[Sequence[int], Sequence[Sequence[int]]] = [],
    prefix: str = "my_sphere",
    ):
    """
    Add spheres to the env/
    Sphere positions are (N,3) and sphere radii are (N)
    """
    sphere_handles = [None] * len(sphere_positions)
    if len(colors) == 0:
        colors = [[255, 0, 0]] * len(sphere_positions)
    elif len(colors) == 1:
        colors = colors * len(sphere_positions)
    else:
        assert len(colors) == len(sphere_positions)
    for i, (sphere_pos, sphere_rad) in enumerate(zip(sphere_positions, sphere_radii)):
        sphere_handles[i] = server.scene.add_icosphere(
            name = f"{prefix}_{i}",
            radius = sphere_rad,
            position = tuple(sphere_pos[:3]),
            color = tuple(colors[i]),
            )
    return sphere_handles


def add_trajectory(server, waypoints, robot, attachment_handles, attachment_positions):
    """
    Adds a slider to step through waypoints of a trajectory also allows for auto step through
    using play/pause button

    Args:
        server (ViserServer): ViserServer instance
        waypoints (numpy.array): A 2D numpy array (shape: (N,7)) with N waypoints of joint poses
        robot (ViserUrdf): ViserUrdf instance of the robot

        attachment_handles (numpy.array) - this is a P element list of attachment handles, spheres here
        attachment_positions (numpy.array) - this is a (N, P, 3) array of the position of each attachment handle at each waypoint pos.

    Returns:
        return_type: None.
    """
    if len(waypoints) < 1:
        return
    assert len(attachment_handles) == len(attachment_positions[0])
    traj_slider = server.gui.add_slider(
        "Current Waypoint", min = 0, max = len(waypoints) - 1, step = 1, initial_value = 0
        )

    @traj_slider.on_update
    def update_robot_pose(event):
        waypoint_idx = int(event.target.value)
        joint_config = waypoints[waypoint_idx]
        robot.update_cfg(joint_config)

        for attach_idx, attachment_handle in enumerate(attachment_handles):
            attachment_handle.position = attachment_positions[waypoint_idx][attach_idx]
