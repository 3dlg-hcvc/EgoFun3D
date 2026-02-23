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


def sanitize_points_np(points: np.ndarray) -> np.ndarray:
    # points: (N,3)
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 3
    # drop NaN/Inf
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    # float32 is safest for CUDA kernels
    points = points.astype(np.float32, copy=False)
    # ensure contiguous
    points = np.ascontiguousarray(points)
    return points, finite


def radius_filter_outliers_gpu(
    point_map: np.ndarray,
    radius: float = 0.01,
    nb_points: int = 15,
    device: str = "CUDA:0",
    allow_cpu_fallback: bool = False,
) -> np.ndarray:
    """
    GPU attempt using Open3D Tensor API.
    - If your Open3D build supports radius outlier removal on CUDA in t.geometry, this will be fast.
    - If not supported, you can either raise or fall back to CPU (legacy).
    """
    assert point_map.shape[-1] == 3, f"Expected (..., 3), got {point_map.shape}"
    pts = point_map.reshape(-1, 3)

    # Optional: drop NaN/Inf
    # finite = np.isfinite(pts).all(axis=1)
    # pts_valid = pts[finite].astype(np.float32)
    pts_valid, finite = sanitize_points_np(pts)

    # Prepare output mask (invalid points remain False)
    flat_mask = np.zeros(pts.shape[0], dtype=bool)

    if pts_valid.shape[0] == 0:
        return flat_mask.reshape(point_map.shape[:-1])

    # Choose device
    dev = o3d.core.Device(device)
    if dev.get_type() == o3d.core.Device.DeviceType.CUDA and not o3d.core.cuda.is_available():
        if allow_cpu_fallback:
            dev = o3d.core.Device("CPU:0")
        else:
            raise RuntimeError("CUDA device requested but o3d.core.cuda.is_available() is False.")
    print(f"Using device: {dev}")

    # Tensor point cloud on device
    pcd_t = o3d.t.geometry.PointCloud(dev)
    pcd_t.point["positions"] = o3d.core.Tensor(pts_valid, o3d.core.float32, device=dev)

    print("after creating tensor point cloud")

    # --- Try GPU/tensor radius outlier removal ---
    # Different Open3D versions expose different method names; try a few common ones.
    inlier_mask_t = None

    # Candidate method names seen across versions / docs / builds
    candidates = [
        "remove_radius_outliers",
        "remove_radius_outlier",
        "radius_outlier_removal",
    ]

    last_err = None
    for name in candidates:
        fn = getattr(pcd_t, name, None)
        if fn is None:
            print(f"Method '{name}' not available on tensor point cloud.")
            continue
        try:
            print(f"Attempting to call '{name}' for radius outlier removal on device {dev}...")
            o3d.core.cuda.synchronize()
            print("Before radius outlier removal call")
            out = fn(nb_points=nb_points, search_radius=radius)
            print("After radius outlier removal call")
            o3d.core.cuda.synchronize()
            print(f"Successfully called '{name}' for radius outlier removal.")
            # Possible return formats:
            #   (pcd_filtered, mask)  OR (mask, pcd_filtered) OR just mask
            if isinstance(out, tuple) and len(out) == 2:
                a, b = out
                # identify which is mask
                if isinstance(a, o3d.core.Tensor):
                    inlier_mask_t = a
                    print(f"Identified inlier mask tensor in output of '{name}' as first element.")
                elif isinstance(b, o3d.core.Tensor):
                    inlier_mask_t = b
                    print(f"Identified inlier mask tensor in output of '{name}' as second element.")
                else:
                    # could be bool numpy/other
                    print(f"Neither output of '{name}' is a tensor. Output types: {type(a)}, {type(b)}. Cannot identify inlier mask.")
                    pass
            elif isinstance(out, o3d.core.Tensor):
                inlier_mask_t = out
                print(f"Output of '{name}' is a tensor, treating as inlier mask.")
            break
        except Exception as e:
            last_err = e
    print("after radius outlier removal attempt")
    if inlier_mask_t is None:
        if allow_cpu_fallback:
            # Fallback to your original CPU method
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_valid.astype(np.float32))
            _, idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
            valid_mask = np.zeros(pts_valid.shape[0], dtype=bool)
            valid_mask[np.array(idx, dtype=np.int32)] = True
        else:
            raise RuntimeError(
                "Your Open3D build does not expose a tensor/CUDA radius-outlier removal API. "
                "This operation is CPU-only in many Open3D versions.\n"
                f"Last error (if any): {repr(last_err)}"
            )
    else:
        # Bring mask back to CPU numpy
        valid_mask = inlier_mask_t.to(o3d.core.Device("CPU:0")).numpy().astype(bool).reshape(-1)
    print("after radius outlier removal and mask retrieval")

    # Write valid_mask back into full mask (including invalid points)
    flat_mask[np.where(finite)[0]] = valid_mask

    del pcd_t  # free GPU memory
    del inlier_mask_t
    print("after cleanup")
    o3d.core.cuda.synchronize()
    o3d.core.cuda.release_cache()
    print("after GPU synchronization and cache release")
    
    return flat_mask.reshape(point_map.shape[:-1])
        

def refine_point_mask(reconstruction_results: dict) -> dict:
    if "points" not in reconstruction_results.keys():
        full_points_list = []
        for i in range(len(reconstruction_results["depth"])):
            depth_i = reconstruction_results["depth"][i]
            intrinsics = reconstruction_results["intrinsics"]
            extrinsics_i = reconstruction_results["extrinsics"][i]
            base_pc_origin = depth2xyz_world(depth_i, intrinsics, extrinsics_i, cam_type="opencv")
            full_points_list.append(base_pc_origin)
        reconstruction_results["points"] = np.stack(full_points_list, axis=0)
    else:
        full_points_list = reconstruction_results["points"]
    full_points_mask_list = reconstruction_results["points_mask"]
    refined_points_mask_list = []
    for frame_id in range(len(full_points_list)):
        print(f"Refining frame {frame_id} with radius outlier removal...")
        points = full_points_list[frame_id]
        mask = full_points_mask_list[frame_id]
        radius_inlier_mask = radius_filter_outliers_gpu(points, radius=0.01, nb_points=15, allow_cpu_fallback=True)
        refined_mask = np.logical_and(mask, radius_inlier_mask)
        refined_points_mask_list.append(refined_mask)
    reconstruction_results["points_mask"] = np.stack(refined_points_mask_list, axis=0)
    return reconstruction_results