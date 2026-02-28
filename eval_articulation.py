import argparse
import omegaconf
from datetime import datetime
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader
import pickle
import gzip
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import loguru

from dataset.dataset import Dataset, build_dataset
from articulation.base import build_articulation_estimation_model, ArticulationEstimation
from articulation.evaluate_articulation import compute_joint_error, save_articulation_metrics, save_articulation_results

MAX_JOINT_ORI_ERROR = np.pi / 2
MAX_JOINT_POS_ERROR = 1.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_collate(batch):
    # batch is a list of dataset items
    # with batch_size=1, just return the single element
    return batch[0]


def evaluate(eval_dataloader: DataLoader, articulation_estimation_model: ArticulationEstimation, config: omegaconf.DictConfig, save_dir: str):
    # Run segmentation
    if config.debug:
        loguru.logger.debug("Debug mode enabled: Limiting evaluation dataset to 1 sample.")
        max_dataset_size = 1
    data_count = 0
    for data in eval_dataloader:
        # data = batch[0]  # batch size is 1
        loguru.logger.info(f"Evaluating data: {data['video_name']}")
        if config.debug and data_count >= max_dataset_size:
            break
        articulation_results = {}
        reconstruction_results = None
        save_articulation_dir = os.path.join(save_dir, data["video_name"], "articulation")
        if os.path.exists(f"{save_articulation_dir}/articulation_results.json") and not config.pred_mask:
            loguru.logger.info("Articulation results already exist, skipping articulation and evaluation for this sample.")
            continue
        if config.pred_mask and os.path.exists(f"{save_articulation_dir}/articulation_results_pred_mask.json"):
            loguru.logger.info("Pred mask articulation results already exist, skipping articulation and evaluation for this sample.")
            continue
        if not os.path.exists(save_articulation_dir):
            os.makedirs(save_articulation_dir)
        for role in ["receiver", "effector"]:
            gt_articulation = data[f"{role}_articulation"]
            if gt_articulation is None:
                loguru.logger.debug(f"No GT articulation for role {role}, skipping evaluation for this role.")
                articulation_results[role] = "No GT articulation, skipping evaluation for this role."
                continue
            video_frame_list = data["rgb_list"]
            if not config.pred_mask:
                mask_list = data[f"{role}_mask_list"]
            else:
                role_mask_dir = os.path.join(config.segmentation_results_dir, data["video_name"], f"00/segmentation_{role}")
                segmentation_metric_path = f"{role_mask_dir}/segmentation_metrics.json"
                with open(segmentation_metric_path, "r") as f:
                    segmentation_metrics = json.load(f)
                mean_iou = segmentation_metrics["mean_iou"]
                if mean_iou < config.pred_mask_iou_threshold:
                    loguru.logger.info(f"Mean IoU for {role} is below threshold ({mean_iou:.4f} < {config.pred_mask_iou_threshold}), skipping articulation estimation for this role.")
                    articulation_results[role] = "Pred mask IoU below threshold, skipping articulation estimation."
                    articulation_metrics = {
                        "joint axis error": MAX_JOINT_ORI_ERROR,
                        "joint position error": MAX_JOINT_POS_ERROR,
                        "joint type correct": False
                    }
                    save_articulation_metrics(articulation_metrics, f"{save_articulation_dir}/articulation_metrics_{role}_pred_mask.json")
                    continue
                mask_list = []
                for i in range(len(video_frame_list)):
                    mask = np.load(f"{role_mask_dir}/segmentation_mask_{i:04d}.npy")
                    mask_list.append(mask)
                mask_list = np.stack(mask_list, axis=0)  # (T, H, W)
                valid_frame_ids = [i for i, mask in enumerate(mask_list) if mask.sum() > 0]

            # load reconstruction
            if reconstruction_results is None:
                reconstruction_results_path = os.path.join(config.reconstruction_results_dir, data["video_name"], "reconstruction/reconstruction_results.pkl.gz")
                if os.path.exists(reconstruction_results_path):
                    loguru.logger.info(f"Loading existing reconstruction results from: {reconstruction_results_path}")
                    with gzip.open(reconstruction_results_path, "rb") as f:
                        reconstruction_results = pickle.load(f)
            if reconstruction_results is None:
                loguru.logger.info("Reconstruction failed, skipping this sample.")
                articulation_results["receiver"] = "Reconstruction failed, skipping this sample."
                articulation_results["effector"] = "Reconstruction failed, skipping this sample."
                break
            # run articulation estimation
            articulation_results[role] = articulation_estimation_model.articulation_estimation(video_frame_list, reconstruction_results, mask_list)

            # Evaluate reconstruction
            if articulation_results[role] is None:
                loguru.logger.info("Articulation estimation failed, skipping evaluation for this role.")
                articulation_results[role] = "Articulation estimation failed, skipping evaluation for this role."
                joint_ori_error = MAX_JOINT_ORI_ERROR
                joint_pos_error = MAX_JOINT_POS_ERROR
                joint_type_correct = False
            else:
                joint_ori_error, joint_pos_error, joint_type_correct = compute_joint_error(
                    gt_articulation,
                    articulation_results[role]
                )
            
            # save_pcd(fused_part_pcd, f"{save_pcd_dir}/{role}_fused.ply")
            articulation_metrics = {
                "joint axis error": joint_ori_error,
                "joint position error": joint_pos_error,
                "joint type correct": joint_type_correct
            }
            if not config.pred_mask:
                save_articulation_metrics(articulation_metrics, f"{save_articulation_dir}/articulation_metrics_{role}.json")
            else:
                save_articulation_metrics(articulation_metrics, f"{save_articulation_dir}/articulation_metrics_{role}_pred_mask.json")
        if not config.pred_mask:
            save_articulation_results(articulation_results, f"{save_articulation_dir}/articulation_results.json")
        else:
            save_articulation_results(articulation_results, f"{save_articulation_dir}/articulation_results_pred_mask.json")


@hydra.main(version_base="1.3", config_path="config", config_name="default")
def main(config: DictConfig):
    loguru.logger.info(f"Start experiment: {config.name}")
    if "save_dir" in config and config.save_dir is not None:
        loguru.logger.info(f"Resuming from: {config.save_dir}")
        save_dir = config.save_dir
    else:
        exp_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"{config.save_root_dir}/{config.name}/{exp_time}"
        # config.update({"save_dir": save_dir})
        config.save_dir = save_dir
    loguru.logger.info(f"Results will be saved to: {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}/config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)

    set_seed(config.seed)

    eval_dataset = build_dataset(config.dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=identity_collate)

    # Initialize  models
    articulation_estimation_model = build_articulation_estimation_model(config.articulation)

    evaluate(eval_dataloader, articulation_estimation_model, config, save_dir)
    # print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
