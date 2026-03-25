import argparse
import omegaconf
from datetime import datetime
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import loguru

from dataset.dataset import Dataset, build_dataset
from VLM.prompt_vlm import build_vlm_prompter, VLMPrompter
from function.evaluate_function import compute_function_error, save_function_results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_collate(batch):
    # batch is a list of dataset items
    # with batch_size=1, just return the single element
    return batch[0]


def evaluate(eval_dataloader: DataLoader, vlm: VLMPrompter, config: omegaconf.DictConfig, save_dir: str):
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
        function_results = {}
        save_function_dir = os.path.join(save_dir, data["video_name"], "function")
        if os.path.exists(f"{save_function_dir}/function_results.json") and not config.pred_mask:
            loguru.logger.info("Function results already exist, skipping function and evaluation for this sample.")
            continue
        if config.pred_mask and os.path.exists(f"{save_function_dir}/function_results_pred_mask.json"):
            loguru.logger.info("Pred mask function results already exist, skipping function and evaluation for this sample.")
            continue
        if not os.path.exists(save_function_dir):
            os.makedirs(save_function_dir)
        gt_function = data["function_annotation"]
        assert gt_function is not None, "GT function annotation is required for evaluation."
        video_frame_list = data["rgb_list"]
        if not config.pred_mask:
            receptor_mask_list = data[f"receptor_mask_list"]
            effector_mask_list = data[f"effector_mask_list"]
        else:
            receptor_mask_dir = os.path.join(config.segmentation_results_dir, data["video_name"], f"00/segmentation_receptor")
            effector_mask_dir = os.path.join(config.segmentation_results_dir, data["video_name"], f"00/segmentation_effector")
            if not os.path.exists(receptor_mask_dir) or not os.path.exists(effector_mask_dir):
                loguru.logger.info(f"Segmentation results for receptor or effector do not exist, skipping function estimation and evaluation for this sample.")
                function_error_metrics = {"physical_effect": False, "numerical_function": False}
                function_results = {"1": None, "2": None}
                save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics_pred_mask.json")
                save_function_results(function_results, f"{save_function_dir}/function_results_pred_mask.json")
                continue
            receptor_segmentation_metric_path = f"{receptor_mask_dir}/segmentation_metrics.json"
            with open(receptor_segmentation_metric_path, "r") as f:
                receptor_segmentation_metrics = json.load(f)
            receptor_mean_iou = receptor_segmentation_metrics["mean_iou"]
            effector_segmentation_metric_path = f"{effector_mask_dir}/segmentation_metrics.json"
            with open(effector_segmentation_metric_path, "r") as f:
                effector_segmentation_metrics = json.load(f)
            effector_mean_iou = effector_segmentation_metrics["mean_iou"]
            if receptor_mean_iou is None or effector_mean_iou is None or receptor_mean_iou < config.pred_mask_iou_threshold or effector_mean_iou < config.pred_mask_iou_threshold:
                loguru.logger.warning(f"Mean IoU for receptor or effector is below threshold ({config.pred_mask_iou_threshold}), skipping function and evaluation for this sample.")
                function_error_metrics = {"physical_effect": False, "numerical_function": False}
                function_results = {"1": None, "2": None}
                save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics_pred_mask.json")
                save_function_results(function_results, f"{save_function_dir}/function_results_pred_mask.json")
                continue
            receptor_mask_list = []
            for i in range(len(video_frame_list)):
                mask = np.load(f"{receptor_mask_dir}/segmentation_mask_{i:04d}.npy")
                receptor_mask_list.append(mask)
            receptor_mask_list = np.stack(receptor_mask_list, axis=0)
            receptor_mask_list = receptor_mask_list[:, data["cropped_top_left"][1]:data["cropped_bottom_right"][1], data["cropped_top_left"][0]:data["cropped_bottom_right"][0]]
            effector_mask_list = []
            for i in range(len(video_frame_list)):
                mask = np.load(f"{effector_mask_dir}/segmentation_mask_{i:04d}.npy")
                effector_mask_list.append(mask)
            effector_mask_list = np.stack(effector_mask_list, axis=0)
            effector_mask_list = effector_mask_list[:, data["cropped_top_left"][1]:data["cropped_bottom_right"][1], data["cropped_top_left"][0]:data["cropped_bottom_right"][0]]

        # run articulation estimation
        function_results = vlm.prompt_function(video_frame_list, receptor_mask_list, effector_mask_list)

        # Evaluate reconstruction
        if function_results is None:
            loguru.logger.warning("No function results from VLM, skipping evaluation for this sample.")
            function_error_metrics = {"physical_effect": False, "numerical_function": False}
            function_results = {"1": None, "2": None}
        else:
            function_error_metrics = compute_function_error(gt_function, function_results)
        if not config.pred_mask:
            save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics.json")
            save_function_results(function_results, f"{save_function_dir}/function_results.json")
        else:
            save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics_pred_mask.json")
            save_function_results(function_results, f"{save_function_dir}/function_results_pred_mask.json")


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
    VLM = build_vlm_prompter(config.vlm_function)

    evaluate(eval_dataloader, VLM, config, save_dir)
    # print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
