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
from segmentation.workflow import load_segmentation_masks_for_sample


RECOMPUTED_SEGMENTATION_IOU_PATH = os.path.join(
    "/3dlg-jupiter-project/FunGraph3D/self_capture_data/dataset/",
    "sam3agent_gemini_recomputed_iou.json",
)


def load_recomputed_segmentation_ious():
    with open(RECOMPUTED_SEGMENTATION_IOU_PATH, "r") as f:
        iou_records = json.load(f)
    if not isinstance(iou_records, list):
        raise TypeError(
            f"Expected a list in {RECOMPUTED_SEGMENTATION_IOU_PATH}"
        )

    iou_by_video = {}
    for record in iou_records:
        video_name = record.get("video_name")
        if not isinstance(video_name, str):
            raise ValueError(
                f"IoU record has no string video_name: {record}"
            )
        if video_name in iou_by_video:
            raise ValueError(f"Duplicate IoU record for video: {video_name}")
        iou_by_video[video_name] = record
    return iou_by_video


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
    segmentation_iou_by_video = (
        load_recomputed_segmentation_ious() if config.pred_mask else None
    )
    if config.debug:
        loguru.logger.debug("Debug mode enabled: Limiting evaluation dataset to 1 sample.")
        max_dataset_size = 1
    for data_count, data in enumerate(eval_dataloader):
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
        if gt_function["physics"] == "undefined" or gt_function["func"] == "undefined":
            loguru.logger.info("GT function annotation is undefined, skipping function and evaluation for this sample.")
            continue
        video_frame_list = data["rgb_list"]
        if not config.pred_mask:
            receptor_mask_list = data[f"receptor_mask_list"]
            effector_mask_list = data[f"effector_mask_list"]
        else:
            receptor_mask_dir = os.path.join(config.segmentation_results_dir, data["video_name"], "segmentation/segmentation_receptor")
            effector_mask_dir = os.path.join(config.segmentation_results_dir, data["video_name"], "segmentation/segmentation_effector")
            if not os.path.exists(receptor_mask_dir) or not os.path.exists(effector_mask_dir):
                loguru.logger.info(f"Segmentation results for receptor or effector do not exist, skipping function estimation and evaluation for this sample.")
                function_error_metrics = {"physical_effect": False, "numerical_function": False}
                function_results = {"1": None, "2": None}
                save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics_pred_mask.json")
                save_function_results(function_results, f"{save_function_dir}/function_results_pred_mask.json")
                continue
            segmentation_metrics = segmentation_iou_by_video.get(data["video_name"])
            if segmentation_metrics is None:
                loguru.logger.warning(
                    f"No recomputed IoU record for {data['video_name']}, "
                    "skipping function estimation and evaluation for this sample."
                )
                function_error_metrics = {"physical_effect": False, "numerical_function": False}
                function_results = {"1": None, "2": None}
                save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics_pred_mask.json")
                save_function_results(function_results, f"{save_function_dir}/function_results_pred_mask.json")
                continue
            receptor_mean_iou = segmentation_metrics.get("receptor_mean_iou")
            effector_mean_iou = segmentation_metrics.get("effector_mean_iou")
            if receptor_mean_iou is None or effector_mean_iou is None or receptor_mean_iou < config.pred_mask_iou_threshold or effector_mean_iou < config.pred_mask_iou_threshold:
                loguru.logger.warning(f"Mean IoU for receptor or effector is below threshold ({config.pred_mask_iou_threshold}), skipping function and evaluation for this sample.")
                function_error_metrics = {"physical_effect": False, "numerical_function": False}
                function_results = {"1": None, "2": None}
                save_function_results(function_error_metrics, f"{save_function_dir}/function_metrics_pred_mask.json")
                save_function_results(function_results, f"{save_function_dir}/function_results_pred_mask.json")
                continue
            receptor_mask_list = load_segmentation_masks_for_sample(data, receptor_mask_dir)
            effector_mask_list = load_segmentation_masks_for_sample(data, effector_mask_dir)

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
