import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance
import json
import pickle
import gzip
import math
import os
from trimesh import transformations

from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.geometry import depthmap_to_world_frame
from utils.reconstruction_utils import radius_filter_outliers

from typing import Optional, Tuple, Dict

MAX_CHAMFER_DISTANCE = 100
MAX_CAMERA_ROTATION_ERROR = np.pi
MAX_CAMERA_TRANSLATION_ERROR = 10.0

def compute_part_chamfer_distance(gt_pcd: np.ndarray, pred_pcd: np.ndarray, device: str) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        gt_pcd (np.ndarray): Ground truth point cloud of shape (N, 3).
        pred_pcd (np.ndarray): Predicted point cloud of shape (M, 3).

    Returns:
        float: Chamfer Distance between the two point clouds.
    """
    gt_tensor = torch.from_numpy(gt_pcd).unsqueeze(0).to(torch.float32).to(device)  # (1, N, 3)
    pred_tensor = torch.from_numpy(pred_pcd).unsqueeze(0).to(torch.float32).to(device)  # (1, M, 3)

    chamfer_dist, _ = chamfer_distance(gt_tensor, pred_tensor)
    return chamfer_dist.item()


def compute_depth_error(gt_depth: np.ndarray, pred_depth: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute Mean Absolute Error (MAE) between two depth maps.

    Args:
        gt_depth (np.ndarray): Ground truth depth map.
        pred_depth (np.ndarray): Predicted depth map.
        valid_mask (np.ndarray): Boolean mask indicating valid pixels.

    Returns:
        float: Mean Absolute Error over valid pixels.
    """
    if np.sum(valid_mask) == 0:
        return float('inf')
    pred_error = np.abs(gt_depth[valid_mask] - pred_depth[valid_mask])
    mean_error = np.mean(pred_error)
    max_error = np.max(pred_error)
    return float(mean_error), float(max_error)


def compute_extrinsics_error(gt_extrinsics: np.ndarray, pred_extrinsics: np.ndarray) -> Tuple[float, float]:
    rotation_error_matrix = pred_extrinsics[:, :3, :3] @ gt_extrinsics[:, :3, :3].transpose(0, 2, 1)
    cos_value = (np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2
    cos_value = np.clip(cos_value, -1.0 + 1e-8, 1.0 - 1e-8)
    cam_rotation_error = np.mean(np.arccos(cos_value))
    cam_translation_error = np.mean(np.linalg.norm(pred_extrinsics[:, :3, 3] - gt_extrinsics[:, :3, 3], axis=1))
    return float(cam_rotation_error), float(cam_translation_error)


def save_pcd(pcd: np.ndarray, save_path: str):
    """
    Save point cloud to a .ply file.

    Args:
        pcd (np.ndarray): Point cloud of shape (N, 3).
        save_path (str): Path to save the .ply file.
    """
    if pcd.shape[0] > 20000:
        indices = np.random.choice(pcd.shape[0], 20000, replace=False)
        pcd = pcd[indices]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(save_path, pcd_o3d)


def _normalize_images_for_export(image_array: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image_array)
    if image_array.dtype == np.uint8:
        return image_array.astype(np.float32) / 255.0
    image_array = image_array.astype(np.float32)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    return np.clip(image_array, 0.0, 1.0)


def _sample_observation_indices(mask_list: np.ndarray, num_observations: Optional[int]) -> np.ndarray:
    mask_array = np.asarray(mask_list).astype(bool)
    valid_frame_ids = np.flatnonzero(mask_array.reshape(mask_array.shape[0], -1).any(axis=1))
    if len(valid_frame_ids) == 0:
        raise ValueError("No valid observations found in mask_list.")

    if num_observations is None or num_observations <= 0 or num_observations >= len(valid_frame_ids):
        return valid_frame_ids

    sample_positions = np.linspace(0, len(valid_frame_ids) - 1, num=num_observations, dtype=int)
    return valid_frame_ids[sample_positions]


def _sample_observation_positions(num_observations: int, requested_num_observations: Optional[int]) -> np.ndarray:
    if requested_num_observations is None or requested_num_observations <= 0 or requested_num_observations >= num_observations:
        return np.arange(num_observations, dtype=int)
    return np.linspace(0, num_observations - 1, num=requested_num_observations, dtype=int)


def save_mesh(
    reconstruction_results: Dict[str, np.ndarray],
    image_list: np.ndarray,
    mask_list: np.ndarray,
    transformation_list: np.ndarray,
    save_path: str,
    observation_indices: Optional[np.ndarray] = None,
    num_observations: Optional[int] = None,
    mesh_format: Optional[str] = None,
) -> np.ndarray:
    """
    Save a fused reconstruction mesh from evenly sampled observations.

    Args:
        reconstruction_results: Reconstruction output dictionary.
        image_list: Video frames shaped (T, H, W, 3) or a list of frames.
        mask_list: Per-frame boolean masks shaped (T, H, W).
        transformation_list: Per-observation rigid transforms aligned with the fused part.
        save_path: Output path for the exported mesh.
        observation_indices: Frame indices corresponding to transformation_list. If None,
            valid frames are derived from mask_list.
        num_observations: Number of valid observations to use. If None, use all.
        mesh_format: Export format, currently supports "glb" and "obj".

    Returns:
        np.ndarray: Selected frame indices used to build the mesh.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    image_array = np.asarray(image_list)
    mask_array = np.asarray(mask_list).astype(bool)
    transformation_array = np.asarray(transformation_list)
    if observation_indices is None:
        observation_indices = _sample_observation_indices(mask_array, None)
    observation_indices = np.asarray(observation_indices, dtype=int)
    if len(observation_indices) != len(transformation_array):
        raise ValueError("observation_indices and transformation_list must have the same length.")

    selected_positions = _sample_observation_positions(len(observation_indices), num_observations)
    selected_indices = observation_indices[selected_positions]
    selected_transformations = transformation_array[selected_positions]

    export_format = (mesh_format or os.path.splitext(save_path)[1].lstrip(".") or "glb").lower()
    if export_format not in {"glb", "obj"}:
        raise ValueError(f"Unsupported mesh format: {export_format}")

    intrinsics_torch = torch.from_numpy(reconstruction_results["intrinsics"]).to(torch.float32)
    sampled_world_points = []
    for frame_id in selected_indices:
        depthmap_torch = torch.from_numpy(reconstruction_results["depth"][frame_id]).to(torch.float32)
        camera_pose_torch = torch.from_numpy(reconstruction_results["extrinsics"][frame_id]).to(torch.float32)
        pts3d_computed, _ = depthmap_to_world_frame(depthmap_torch, intrinsics_torch, camera_pose_torch)
        sampled_world_points.append(pts3d_computed.cpu().numpy() if hasattr(pts3d_computed, "cpu") else pts3d_computed.numpy())

    sampled_images = [_normalize_images_for_export(image_array[frame_id]) for frame_id in selected_indices]

    transformed_world_points_list = []
    radius_map_list = []
    for pts3d, transformation in zip(sampled_world_points, selected_transformations):
        transformed_pts3d = (transformation[:3, :3] @ pts3d.reshape(-1, 3).T).T + transformation[:3, 3]
        transformed_pts3d = transformed_pts3d.reshape(pts3d.shape)
        transformed_world_points_list.append(transformed_pts3d)
        radius_map_list.append(radius_filter_outliers(transformed_pts3d, radius=0.01, nb_points=15))

    final_mask = np.logical_and(reconstruction_results["points_mask"][selected_indices], mask_array[selected_indices])
    final_mask = np.logical_and(final_mask, np.stack(radius_map_list, axis=0))

    predictions = {
        "extrinsic": reconstruction_results["extrinsics"][selected_indices],
        "intrinsic": np.repeat(reconstruction_results["intrinsics"][None, ...], repeats=len(selected_indices), axis=0),
        "world_points": np.stack(transformed_world_points_list, axis=0),
        "depth": reconstruction_results["depth"][selected_indices][..., None]
        if reconstruction_results["depth"][selected_indices].ndim == 3
        else reconstruction_results["depth"][selected_indices],
        "images": np.stack(sampled_images, axis=0),
        "final_mask": final_mask,
    }

    scene_mesh = predictions_to_glb(predictions, show_cam=False)

    if export_format == "obj":
        merged_mesh = scene_mesh.dump(concatenate=True)
        merged_mesh.export(save_path, file_type="obj")
    else:
        scene_mesh.export(save_path)

    return selected_indices


def evaluate_reconstruction(pred_pcd: np.ndarray, pred_extrinsics: np.ndarray,
                            gt_pcd: np.ndarray, gt_extrinsics: np.ndarray,
                            device: str = "cuda") -> Tuple[float, float, float]:
    """
    Evaluate reconstruction results using Chamfer Distance, Depth MAE, and Extrinsics error.

    Args:
        pred_pcd (np.ndarray): Predicted point cloud of shape (M, 3).
        pred_extrinsics (np.ndarray): Predicted camera extrinsics of shape (T, 4, 4).
        gt_pcd (np.ndarray): Ground truth point cloud of shape (N, 3).
        gt_extrinsics (np.ndarray): Ground truth camera extrinsics of shape (T, 4, 4).
        device (str): Device to perform computations on.

    Returns:
        Tuple[float, float, float]: Chamfer Distance, Rotation Error (radians), Translation Error.
    """
    if pred_pcd.shape[0] == 0:
        chamfer_dist = MAX_CHAMFER_DISTANCE
    else:
        chamfer_dist = compute_part_chamfer_distance(gt_pcd, pred_pcd, device)
    # depth_mean_error, depth_max_error = compute_depth_error(gt_depth, pred_depth, depth_valid_mask)
    rot_error, trans_error = compute_extrinsics_error(gt_extrinsics, pred_extrinsics)
    if np.isnan(rot_error):
        rot_error = MAX_CAMERA_ROTATION_ERROR
    if np.isnan(trans_error):
        trans_error = MAX_CAMERA_TRANSLATION_ERROR
    return chamfer_dist, rot_error, trans_error


def save_reconstruction_metrics(metrics: dict, save_path: str):
    """
    Save reconstruction metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing reconstruction metrics.
        save_path (str): Path to save the JSON file.
    """
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_reconstruction_results(reconstruction_results: Dict[str, np.ndarray], save_path: str):
    """
    Save reconstruction results including point clouds and camera parameters.

    Args:
        reconstruction_results (Dict[str, np.ndarray]): Dictionary containing reconstruction results.
        save_path (str): Path to save the results.
    """
    reconstruction_results.pop("rgb", None)
    reconstruction_results.pop("points", None)
    with gzip.open(save_path, "wb", compresslevel=5) as f:
        pickle.dump(reconstruction_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    flat_dict = {}
    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}/{k}" if prefix else str(k), v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                flatten(f"{prefix}/{i}", v)
        elif isinstance(obj, np.ndarray):
            flat_dict[prefix] = obj
        else:
            # Save non-array objects as numpy object arrays
            flat_dict[prefix] = np.array(obj, dtype=object)

    flatten("", reconstruction_results)

    output_path = save_path.replace(".pkl.gz", ".npz")
    # 3️⃣ Save as npz
    np.savez_compressed(output_path, **flat_dict)
