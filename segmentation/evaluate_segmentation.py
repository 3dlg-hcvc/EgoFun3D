import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import json
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import os

from typing import List, Tuple


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


def compute_part_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        gt_mask (np.ndarray): Ground truth binary mask.
        pred_mask (np.ndarray): Predicted binary mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) < 20:
            iou = 1.0
        else:
            iou = 0
    else:
        iou = intersection / union
    return iou


def compute_part_iou_video(gt_mask_list: list[np.ndarray], pred_mask_list: list[np.ndarray], valid_frame_ids: list[int]) -> Tuple[List[float], List[float]]:
    filtered_iou_scores = []
    origin_iou_scores = []
    for frame_id in range(len(gt_mask_list)):
        pred_mask = pred_mask_list[frame_id]
        iou = compute_part_iou(gt_mask_list[frame_id], pred_mask)
        origin_iou_scores.append(iou)
        if frame_id in valid_frame_ids:
            filtered_iou_scores.append(iou)
    return filtered_iou_scores, origin_iou_scores


def save_segmentation(image: np.ndarray | PILImage.Image, mask: np.ndarray, answer_dict: dict, save_dir: str, id: str):
    with open(f"{save_dir}/segmentation_answer_{id}.json", "w") as f:
        json.dump(answer_dict, f, indent=4)
    
    plt.figure(figsize=(8, 4))
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    # Image with Mask Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask] = [255, 0, 0]
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title('Image with Predicted Mask')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/segmentation_vis_{id}.png")
    plt.close()

    np.save(f"{save_dir}/segmentation_mask_{id}.npy", mask)


def save_segmentation_video(image_list: List[np.ndarray | PILImage.Image], mask_list: List[np.ndarray], answer_dict_list: List[dict], valid_frame_ids: List[int], 
                            original_iou_list: List[float], filtered_iou_list: List[float], save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for frame_id in range(len(image_list)):
        if frame_id not in valid_frame_ids:
            filtered_iou = -1
        else:
            filtered_iou = filtered_iou_list[valid_frame_ids.index(frame_id)]
        pred_mask = mask_list[frame_id]
        answer_dict = answer_dict_list[frame_id]
        answer_dict["iou"] = original_iou_list[frame_id]
        answer_dict["filtered_iou"] = filtered_iou
        save_segmentation(image_list[frame_id], pred_mask, answer_dict, save_dir, f"{frame_id:04d}")


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


def save_vlm_output(output_dict: dict, save_path: str):
    """
    Save VLM output dictionary to a JSON file.

    Args:
        output_dict (dict): VLM output dictionary.
        save_path (str): Path to save the JSON file.
    """
    with open(f"{save_path}/vlm_analysis.json", "w") as f:
        json.dump(output_dict, f, indent=4)