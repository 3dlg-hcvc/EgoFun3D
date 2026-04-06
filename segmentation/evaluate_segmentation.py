import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from segmentation.workflow import save_segmentation_answers, save_segmentation_mask_archive


def compute_part_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) < 20:
            return 1.0
        return 0.0
    return float(intersection / union)


def compute_part_iou_video(
    gt_mask_list: list[np.ndarray],
    pred_mask_list: list[np.ndarray],
    valid_frame_ids: list[int],
    eval_frame_ids: list[int] | None = None,
) -> Tuple[List[float], List[float | None]]:
    filtered_iou_scores = []
    origin_iou_scores: List[float | None] = []
    eval_frame_set = set(eval_frame_ids) if eval_frame_ids is not None else None
    for frame_id in range(len(gt_mask_list)):
        if eval_frame_set is not None and frame_id not in eval_frame_set:
            origin_iou_scores.append(None)
            continue
        pred_mask = pred_mask_list[frame_id]
        iou = compute_part_iou(gt_mask_list[frame_id], pred_mask)
        origin_iou_scores.append(iou)
        if frame_id in valid_frame_ids:
            filtered_iou_scores.append(iou)
    return filtered_iou_scores, origin_iou_scores


def _save_segmentation_visualization(
    image: np.ndarray | PILImage.Image,
    mask: np.ndarray,
    save_dir: str,
    frame_id: int,
) -> None:
    import matplotlib.pyplot as plt

    image_np = np.asarray(image)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(image_np, alpha=0.6)
    mask_overlay = np.zeros_like(image_np)
    mask_overlay[np.asarray(mask).astype(bool)] = [255, 0, 0]
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title('Image with Predicted Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'segmentation_vis_{frame_id:04d}.png'))
    plt.close()


def save_segmentation_video(
    image_list: List[np.ndarray | PILImage.Image],
    mask_list: List[np.ndarray],
    answer_dict_list: List[dict],
    valid_frame_ids: List[int],
    original_iou_list: List[float | None],
    filtered_iou_list: List[float],
    save_dir: str,
    save_visualizations: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)

    enriched_answers: list[dict] = []
    valid_frame_set = set(valid_frame_ids)
    filtered_iou_by_frame = {frame_id: filtered_iou_list[idx] for idx, frame_id in enumerate(valid_frame_ids)}
    for frame_id in range(len(image_list)):
        answer_dict = dict(answer_dict_list[frame_id] if frame_id < len(answer_dict_list) and isinstance(answer_dict_list[frame_id], dict) else {})
        answer_dict.setdefault('frame_id', frame_id)
        answer_dict['iou'] = original_iou_list[frame_id]
        answer_dict['filtered_iou'] = filtered_iou_by_frame.get(frame_id, -1 if frame_id not in valid_frame_set else None)
        enriched_answers.append(answer_dict)
        if save_visualizations:
            _save_segmentation_visualization(image_list[frame_id], mask_list[frame_id], save_dir, frame_id)

    save_segmentation_answers(enriched_answers, save_dir)
    save_segmentation_mask_archive(mask_list, save_dir)


def save_segmentation_metrics(
    original_iou_list: List[float | None],
    filtered_iou_list: List[float],
    valid_frame_ids: List[int],
    save_dir: str,
    eval_frame_ids: List[int] | None = None,
    runtime_info: dict | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    if eval_frame_ids is None:
        eval_frame_ids = list(range(len(original_iou_list)))
    else:
        eval_frame_ids = [int(i) for i in eval_frame_ids]
    eval_iou_list = [
        float(original_iou_list[i])
        for i in eval_frame_ids
        if i < len(original_iou_list) and original_iou_list[i] is not None
    ]
    metrics = {
        'mean_iou': float(np.mean(eval_iou_list)) if len(eval_iou_list) > 0 else 0.0,
        'mean_filtered_iou': float(np.mean(filtered_iou_list)) if len(filtered_iou_list) > 0 else 0.0,
        'num_frames': int(len(original_iou_list)),
        'num_eval_frames': int(len(eval_iou_list)),
        'num_valid_frames': int(len(valid_frame_ids)),
        'eval_frame_ids': [int(i) for i in eval_frame_ids],
        'valid_frame_ids': [int(i) for i in valid_frame_ids],
        'per_frame_iou': [float(x) if x is not None else None for x in original_iou_list],
        'per_frame_filtered_iou': [float(x) for x in filtered_iou_list],
        'mask_archive': 'segmentation_masks.npz',
        'answers_archive': 'segmentation_answers.json',
    }
    if runtime_info is not None:
        metrics['runtime'] = runtime_info
    with open(os.path.join(save_dir, 'segmentation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    return metrics


def save_vlm_output(output_dict: dict, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'vlm_analysis.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)


def build_shared_vlm_output(grouped_results: Dict[str, Dict[str, str]], scene_name: str, seg_id: str, source: str) -> dict:
    receptor = grouped_results['receptor']
    effector = grouped_results['effector']
    return {
        'scene_name': scene_name,
        'seg_id': seg_id,
        'source': source,
        'parts': {
            'receptor': {
                'label': receptor['name'],
                'description': receptor['description'],
            },
            'receiver': {
                'label': receptor['name'],
                'description': receptor['description'],
            },
            'effector': {
                'label': effector['name'],
                'description': effector['description'],
            },
        },
    }


def save_shared_vlm_output(shared_output: dict, save_root: str, scene_name: str, seg_id: str) -> str:
    output_dir = os.path.join(save_root, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{seg_id}.json')
    with open(output_path, 'w') as f:
        json.dump(shared_output, f, indent=4)
    return output_path


def load_shared_vlm_output(save_root: str, scene_name: str, seg_id: str) -> Optional[dict]:
    output_path = os.path.join(save_root, scene_name, f'{seg_id}.json')
    if not os.path.exists(output_path):
        return None
    with open(output_path, 'r') as f:
        return json.load(f)
