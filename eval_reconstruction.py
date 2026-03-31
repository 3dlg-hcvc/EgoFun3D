import argparse
import gzip
import pickle
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
import time
from dataset.dataset import Dataset, build_dataset
from fusion.fusion import build_fusion_model, BaseFusion, FeatureMatchingFusion, TrackingFusion
from fusion.reconstruction import build_reconstruction_model, BaseReconstruction, ViPEReconstruction
from fusion.evaluate_reconstruction import save_reconstruction_metrics, evaluate_reconstruction, save_pcd, save_reconstruction_results
from utils.reconstruction_utils import refine_point_mask


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_collate(batch):
    # batch is a list of dataset items
    # with batch_size=1, just return the single element
    return batch[0]


def evaluate(input_modality: str, eval_dataloader: DataLoader, fusion_model: BaseFusion, reconstruction_model: BaseReconstruction, config: DictConfig, save_dir: str):
    # Run segmentation
    if config.debug:
        print("Debug mode enabled: Limiting evaluation dataset to 1 sample.")
        max_dataset_size = 1
    data_count = 0
    for data in eval_dataloader:
        start_time = time.time()
        # data = batch[0]  # batch size is 1
        print("Evaluating data:", data["video_name"])
        if config.debug and data_count >= max_dataset_size:
            break
        reconstruction_results = None
        tracks3d = None
        kptsA_origin_dict = {}
        kptsB_origin_dict = {}
        save_pcd_dir = os.path.join(save_dir, data["video_name"], "reconstruction")
        if os.path.exists(f"{save_pcd_dir}/reconstruction_results.pkl.gz") and not config.pred_mask:
            print("Reconstruction results already exist, skipping reconstruction and evaluation for this sample.")
            continue
        if config.pred_mask and os.path.exists(f"{save_pcd_dir}/reconstruction_metrics_receptor_pred_mask.json") and os.path.exists(f"{save_pcd_dir}/reconstruction_metrics_effector_pred_mask.json"):
            print("Pred mask reconstruction metrics already exist, skipping refinement and evaluation for this sample.")
            continue
        if not os.path.exists(save_pcd_dir):
            os.makedirs(save_pcd_dir)
        project = False
        for role in ["receptor", "effector"]:
            role_start = time.time()
            video_frame_list = data["rgb_list"]
            if not config.pred_mask:
                mask_list = data[f"{role}_mask_list"]
                valid_frame_ids = [i for i, mask in enumerate(mask_list) if mask.sum() > 0]
            else:
                role_mask_dir = os.path.join(config.segmentation_results_dir, data["video_name"], f"00/segmentation_{role}")
                if not os.path.exists(role_mask_dir):
                    reconstruction_metrics = {
                        "chamfer_distance": 200,
                        "rotation_error_radians": 0,
                        "translation_error": 0
                    }
                    save_reconstruction_metrics(reconstruction_metrics, f"{save_pcd_dir}/reconstruction_metrics_{role}_pred_mask.json")
                    continue
                segmentation_metric_path = f"{role_mask_dir}/segmentation_metrics.json"
                with open(segmentation_metric_path, "r") as f:
                    segmentation_metrics = json.load(f)
                mean_iou = segmentation_metrics["mean_iou"]
                if mean_iou is None or mean_iou < config.pred_mask_iou_threshold:
                    print(f"Mean IoU for {role} is below threshold ({config.pred_mask_iou_threshold}), skipping reconstruction and evaluation for this role.")
                    reconstruction_metrics = {
                        "chamfer_distance": 200,
                        "rotation_error_radians": 0,
                        "translation_error": 0
                    }
                    save_reconstruction_metrics(reconstruction_metrics, f"{save_pcd_dir}/reconstruction_metrics_{role}_pred_mask.json")
                    continue
                mask_list = []
                for i in range(len(video_frame_list)):
                    mask = np.load(f"{role_mask_dir}/segmentation_mask_{i:04d}.npy")
                    mask_list.append(mask)
                mask_list = np.stack(mask_list, axis=0)
                mask_list = mask_list[:, data["cropped_top_left"][1]:data["cropped_bottom_right"][1], data["cropped_top_left"][0]:data["cropped_bottom_right"][0]]
                valid_frame_ids = [i for i, mask in enumerate(mask_list) if mask.sum() > 0]

            # run reconstruction
            load_results_start = time.time()
            if reconstruction_results is None:
                if config.pred_mask and os.path.exists(f"{save_pcd_dir}/reconstruction_results.pkl.gz"):
                    print("Existing reconstruction results found, loading for pred mask.")
                    with gzip.open(f"{save_pcd_dir}/reconstruction_results.pkl.gz", "rb") as f:
                        reconstruction_results = pickle.load(f)
                else:
                    init_extrinsics = data["camera_extrinsics"][0]
                    if isinstance(reconstruction_model, ViPEReconstruction):
                        # video_dir = os.path.dirname(data["video_path"])
                        reconstruction_results = reconstruction_model.reconstruct(data["video_path"], init_extrinsics, data["sample_indices"])
                    else:
                        input_intrinsics = None
                        input_extrinsics = None
                        input_depth = None
                        if input_modality.find("intrinsics") != -1:
                            input_intrinsics = data["camera_intrinsics"]
                        if input_modality.find("extrinsics") != -1:
                            input_extrinsics = data["camera_extrinsics"]
                        reconstruction_results = reconstruction_model.reconstruct(video_frame_list, init_extrinsics, input_intrinsics, input_extrinsics, input_depth)
            load_results_end = time.time()
            print(f"Initial reconstruction time: {load_results_end - load_results_start:.2f} seconds")
            if reconstruction_results is None:
                print("Reconstruction failed, skipping this sample.")
                break
            refine_time_start = time.time()
            if config.pred_mask and not project:
                reconstruction_results = refine_point_mask(reconstruction_results)
                project = True
            refine_time_end = time.time()
            print(f"Refinement time: {refine_time_end - refine_time_start:.2f} seconds")
            # run fusion
            if config.debug:
                full_points = reconstruction_results["points"]
                full_masks = reconstruction_results["points_mask"]
                points_list = []
                for i, (points, mask) in enumerate(zip(full_points, full_masks)):
                    if i == len(full_points) // 2:
                        part_points = points[mask]
                        save_pcd(points.reshape(-1, 3), f"{save_pcd_dir}/frame_{i}_full_points.ply")
                    points_list.append(points.reshape(-1, 3))
                save_pcd_dir_debug = os.path.join(save_dir, data["video_name"], "debug_full_reconstruction")
                if not os.path.exists(save_pcd_dir_debug):
                    os.makedirs(save_pcd_dir_debug)
                save_pcd(np.concatenate(points_list, axis=0), f"{save_pcd_dir_debug}/{role}_full_reconstruction.ply")
            points_mask_list = reconstruction_results["points_mask"]
            valid_mask_list = []
            for i in valid_frame_ids:
                points_mask = points_mask_list[i]
                combined_mask = np.logical_and(points_mask, mask_list[i])
                valid_mask_list.append(combined_mask)
            # valid_image_path_list = [data["rgb_path_list"][i] for i in valid_frame_ids]
            intrinsics = reconstruction_results["intrinsics"]
            valid_points_map_list = [reconstruction_results["points"][i] for i in valid_frame_ids]
            
            fuse_start = time.time()
            if isinstance(fusion_model, FeatureMatchingFusion):
                fused_part_pcd, transformation_list, kptsA_origin_dict, kptsB_origin_dict = fusion_model.fuse_part_pcds(video_frame_list, valid_mask_list, valid_points_map_list, kptsA_origin_dict, kptsB_origin_dict)
                # print("kpts len:", len(kptsA_origin_dict), len(kptsB_origin_dict))
            elif isinstance(fusion_model, TrackingFusion):
                if tracks3d is None:
                    tracks3d = fusion_model.tracking_video(video_frame_list, reconstruction_results["depth"], reconstruction_results["extrinsics"], intrinsics, reconstruction_results["points_mask"])
                valid_tracks3d = [tracks3d[i] for i in valid_frame_ids]
                fused_part_pcd, transformation_list = fusion_model.fuse_part_pcds(video_frame_list, valid_mask_list, valid_points_map_list, valid_tracks3d)
            fuse_end = time.time()
            print(f"Fusion time: {fuse_end - fuse_start:.2f} seconds")
            # Evaluate reconstruction
            chamfer_dist, rot_error, trans_error = evaluate_reconstruction(
                pred_pcd=fused_part_pcd,
                pred_extrinsics=reconstruction_results["extrinsics"],
                gt_pcd=data["geometry_data"][role]["part_pcd"],
                gt_extrinsics=data["camera_extrinsics"],
            )
            if not config.pred_mask:
                save_pcd(fused_part_pcd, f"{save_pcd_dir}/{role}_fused.ply")
            else:
                save_pcd(fused_part_pcd, f"{save_pcd_dir}/{role}_fused_pred_mask.ply")
            reconstruction_metrics = {
                "chamfer_distance": chamfer_dist,
                "rotation_error_radians": rot_error,
                "translation_error": trans_error
            }
            if not config.pred_mask:
                save_reconstruction_metrics(reconstruction_metrics, f"{save_pcd_dir}/reconstruction_metrics_{role}.json")
            else:
                save_reconstruction_metrics(reconstruction_metrics, f"{save_pcd_dir}/reconstruction_metrics_{role}_pred_mask.json")
            role_end = time.time()
            print(f"Total time for {role} (including fusion and evaluation): {role_end - role_start:.2f} seconds")
        if reconstruction_results is not None and not config.pred_mask:
            save_reconstruction_results(reconstruction_results, f"{save_pcd_dir}/reconstruction_results.pkl.gz")
        data_count += 1
        end_time = time.time()
        print(f"Total evaluation time for this sample: {end_time - start_time:.2f} seconds")


@hydra.main(version_base="1.3", config_path="config", config_name="default")
def main(config: DictConfig):
    print("Start experiment:", config.name)
    if "save_dir" in config and config.save_dir is not None:
        print("Resuming from:", config.save_dir)
        save_dir = config.save_dir
    else:
        exp_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"{config.save_root_dir}/{config.name}/{exp_time}"
        # config.update({"save_dir": save_dir})
        config.save_dir = save_dir
    print("Results will be saved to:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}/config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)

    set_seed(config.seed)

    eval_dataset = build_dataset(config.dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=identity_collate)

    # Initialize segmentation models
    fusion_model = build_fusion_model(config.fusion)
    input_modality = config["input_modality"]
    reconstruction_model = build_reconstruction_model(input_modality=input_modality, **config.reconstruction)

    evaluate(input_modality, eval_dataloader, fusion_model, reconstruction_model, config, save_dir)


if __name__ == "__main__":
    main()
