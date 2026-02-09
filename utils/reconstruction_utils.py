import open3d as o3d
import numpy as np


def estimate_se3_transformation(target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_xyz)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
    correspondences = np.arange(source_xyz.shape[0])
    correspondences = np.vstack([correspondences, correspondences], dtype=np.int32).T
    p2p_registration = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    source2target = p2p_registration.compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(correspondences))
    return source2target


def depth2xyz(depth_image: np.ndarray, intrinsics: np.ndarray, cam_type: str) -> np.ndarray:
    # Get the shape of the depth image
    H, W = depth_image.shape

    # Create meshgrid for pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten the grid to a 1D array of pixel coordinates
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image to a 1D array of depth values
    if depth_image.dtype == np.uint16:
        depth = depth_image.flatten() / 1000.0
    else:
        depth = depth_image.flatten()

    # Camera intrinsic matrix (3x3)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Calculate the 3D coordinates (x, y, z) from depth
    # Use the formula:
    #   X = (u - cx) * depth / fx
    #   Y = (v - cy) * depth / fy
    #   Z = depth
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack the x, y, z values into a 3D point cloud
    point_cloud = np.vstack((x, y, z)).T

    if cam_type == "opengl":
        point_cloud = point_cloud * np.array([1, -1, -1])

    # Reshape the point cloud to the original depth image shape [H, W, 3]
    point_cloud = point_cloud.reshape(H, W, 3)

    return point_cloud


def depth2xyz_world(depth_image: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, cam_type: str) -> np.ndarray:
    point_cloud = depth2xyz(depth_image, intrinsics, cam_type)
    H, W, _ = point_cloud.shape
    point_cloud_flat = point_cloud.reshape(-1, 3)
    point_cloud_homogeneous = np.hstack([point_cloud_flat, np.ones((point_cloud_flat.shape[0], 1))])
    point_cloud_world_homogeneous = (extrinsics @ point_cloud_homogeneous.T).T
    point_cloud_world = point_cloud_world_homogeneous[:, :3] / point_cloud_world_homogeneous[:, 3:]
    return point_cloud_world.reshape(H, W, 3)