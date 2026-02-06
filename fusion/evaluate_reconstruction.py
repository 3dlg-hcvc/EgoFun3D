import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance
import json
import pickle
import gzip

from typing import Tuple, Dict


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
    cam_rotation_error = np.mean(np.arccos((np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2))
    cam_translation_error = np.mean(np.linalg.norm(pred_extrinsics[:, :3, 3] - gt_extrinsics[:, :3, 3], axis=1))
    return float(cam_rotation_error), float(cam_translation_error)


def save_pcd(pcd: np.ndarray, save_path: str):
    """
    Save point cloud to a .ply file.

    Args:
        pcd (np.ndarray): Point cloud of shape (N, 3).
        save_path (str): Path to save the .ply file.
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(save_path, pcd_o3d)


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
    chamfer_dist = compute_part_chamfer_distance(gt_pcd, pred_pcd, device)
    # depth_mean_error, depth_max_error = compute_depth_error(gt_depth, pred_depth, depth_valid_mask)
    rot_error, trans_error = compute_extrinsics_error(gt_extrinsics, pred_extrinsics)
    return chamfer_dist, rot_error, trans_error


def save_reconstruction_metrics(metrics: dict, save_path: str):
    """
    Save reconstruction metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing reconstruction metrics.
        save_path (str): Path to save the JSON file.
    """
    with open(f"{save_path}/reconstruction_results.json", "w") as f:
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
    with gzip.open(f"{save_path}/reconstruction_results.pkl", "wb", compresslevel=5) as f:
        pickle.dump(reconstruction_results, f, protocol=pickle.HIGHEST_PROTOCOL)