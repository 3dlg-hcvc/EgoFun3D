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


def radius_filter_outliers(point_map: np.ndarray, radius: float = 0.01, nb_points: int = 15) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_map.reshape(-1, 3))
    radius_inlier_pcd, radius_inlier_idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    radius_inlier_mask = np.zeros(point_map.shape[:-1], dtype=bool)
    radius_inlier_mask.reshape(-1)[radius_inlier_idx] = True
    return radius_inlier_mask.reshape(point_map.shape[:-1])


def refine_point_mask(reconstruction_results: dict) -> dict:
    full_points_list = reconstruction_results["points"]
    full_points_mask_list = reconstruction_results["points_mask"]
    refined_points_mask_list = []
    for points, mask in zip(full_points_list, full_points_mask_list):
        radius_inlier_mask = radius_filter_outliers(points, radius=0.01, nb_points=15)
        refined_mask = np.logical_and(mask, radius_inlier_mask)
        refined_points_mask_list.append(refined_mask)
    reconstruction_results["points_mask"] = np.stack(refined_points_mask_list, axis=0)
    return reconstruction_results