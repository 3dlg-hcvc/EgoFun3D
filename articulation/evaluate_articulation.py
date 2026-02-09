import numpy as np
import json

from typing import List, Dict, Any, Tuple


def compute_joint_error(gt_joint_parameter: Dict[str, np.ndarray | str], pred_joint_parameter: Dict[str, np.ndarray | str]) -> Tuple[float, float, bool]:
    gt_joint_type = gt_joint_parameter["type"]
    gt_joint_axis = gt_joint_parameter["axis"]
    gt_joint_pos = gt_joint_parameter["origin"]
    # joint estimation
    pred_joint_type = pred_joint_parameter["type"]
    pred_joint_axis = pred_joint_parameter["axis"]
    pred_joint_axis = pred_joint_axis / np.linalg.norm(pred_joint_axis)
    pred_joint_pos = pred_joint_parameter["origin"]

    joint_ori_error = np.arccos(np.abs(np.dot(pred_joint_axis, gt_joint_axis)))

    n = np.cross(pred_joint_axis, gt_joint_axis)
    joint_pos_error = np.abs(np.dot(n, (pred_joint_pos - gt_joint_pos))) / np.linalg.norm(n)

    if gt_joint_type == "prismatic":
        joint_pos_error = 0

    return joint_ori_error, joint_pos_error, pred_joint_type == gt_joint_type


def save_articulation_results(pred_joint_parameter: Dict[str, Dict[str, np.ndarray | str]], save_path: str):
    for role in pred_joint_parameter:
        if isinstance(pred_joint_parameter[role], dict):
            for parameter_key in pred_joint_parameter[role]:
                if isinstance(pred_joint_parameter[role][parameter_key], np.ndarray):
                    pred_joint_parameter[role][parameter_key] = pred_joint_parameter[role][parameter_key].tolist()
    with open(save_path, "w") as f:
        json.dump({
            "type": pred_joint_parameter["type"],
            "axis": pred_joint_parameter["axis"].tolist(),
            "origin": pred_joint_parameter["origin"].tolist(),
            "state": pred_joint_parameter["state"].tolist()
        }, f, indent=4)


def save_articulation_metrics(metrics: dict, save_path: str):
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)