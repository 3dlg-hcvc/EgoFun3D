import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import utils3d
import json
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import os


def compute_part_chamfer_distance_per_frame(gt_pcd: np.ndarray, pred_pcd: np.ndarray, cam_pose: np.ndarray, cam_intrinsics: np.ndarray, image_shape: tuple, device: str) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        gt_pcd (np.ndarray): Ground truth point cloud of shape (N, 3).
        pred_pcd (np.ndarray): Predicted point cloud of shape (M, 3).

    Returns:
        float: Chamfer Distance between the two point clouds.
    """
    uv_coords, linear_depths = utils3d.np.project_cv(gt_pcd, cam_pose, cam_intrinsics)  # (N, 2), (N,)
    xy_coords = uv_coords * linear_depths[:, None]  # (N, 2)
    in_frame_points_mask = (
        (xy_coords[:, 0] >= 0) &
        (xy_coords[:, 0] < image_shape[1]) &
        (xy_coords[:, 1] >= 0) &
        (xy_coords[:, 1] < image_shape[0])
    )
    gt_pcd = gt_pcd[in_frame_points_mask]
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
        return 1.0 if intersection == 0 else 0.0
    iou = intersection / union
    return iou


def save_segmentation(image: np.ndarray | PILImage.Image, mask: np.ndarray, answer_dict: dict, save_dir: str, id: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
