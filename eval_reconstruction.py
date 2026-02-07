import argparse
import omegaconf
from datetime import datetime
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader
from dataset.dataset import Dataset, build_dataset
from fusion.fusion import build_fusion_model, BaseFusion, FeatureMatchingFusion, TrackingFusion
from fusion.reconstruction import build_reconstruction_model, BaseReconstruction, ViPEReconstruction
from fusion.evaluate_reconstruction import save_reconstruction_metrics, evaluate_reconstruction, save_pcd, save_reconstruction_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_collate(batch):
    # batch is a list of dataset items
    # with batch_size=1, just return the single element
    return batch[0]


def evaluate(input_modality: str, eval_dataloader: DataLoader, fusion_model: BaseFusion, reconstruction_model: BaseReconstruction, config: omegaconf.DictConfig, save_dir: str):
    # Run segmentation
    if config.debug:
        print("Debug mode enabled: Limiting evaluation dataset to 1 sample.")
        max_dataset_size = 1
    data_count = 0
    for data in eval_dataloader:
        # data = batch[0]  # batch size is 1
        print("Evaluating data:", data["video_name"])
        if config.debug and data_count >= max_dataset_size:
            break
        reconstruction_results = None
        tracks3d = None
        save_pcd_dir = os.path.join(save_dir, data["video_name"], "reconstruction")
        if not os.path.exists(save_pcd_dir):
            os.makedirs(save_pcd_dir)
        for role in ["receiver", "effector"]:
            video_frame_list = data["rgb_list"]
            mask_list = data[f"{role}_mask_list"]
            valid_frame_ids = [i for i, mask in enumerate(mask_list) if mask.sum() > 0]

            # run reconstruction
            if reconstruction_results is None:
                init_extrinsics = data["camera_extrinsics"][0]
                if isinstance(reconstruction_model, ViPEReconstruction):
                    video_dir = os.path.dirname(data["rgb_path_list"][0])
                    reconstruction_results = reconstruction_model.reconstruct(video_dir, init_extrinsics, data["sample_indices"])
                else:
                    input_intrinsics = None
                    input_extrinsics = None
                    input_depth = None
                    if input_modality.find("intrinsics") != -1:
                        input_intrinsics = data["camera_intrinsics"]
                    if input_modality.find("extrinsics") != -1:
                        input_extrinsics = data["camera_extrinsics"]
                    reconstruction_results = reconstruction_model.reconstruct(video_frame_list, init_extrinsics, input_intrinsics, input_extrinsics, input_depth)
            if reconstruction_results is None:
                print("Reconstruction failed, skipping this sample.")
                break
            # run fusion
            if config.debug:
                full_points = reconstruction_results["points"]
                full_masks = reconstruction_results["points_mask"]
                points_list = []
                for i, (points, mask) in enumerate(zip(full_points, full_masks)):
                    # part_points = points[mask]
                    # save_pcd(points.reshape(-1, 3), f"{save_pcd_dir}/frame_{i}_full_points.ply")
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
            valid_image_path_list = [data["rgb_path_list"][i] for i in valid_frame_ids]
            intrinsics = reconstruction_results["intrinsics"]
            valid_points_map_list = [reconstruction_results["points"][i] for i in valid_frame_ids]
            if isinstance(fusion_model, FeatureMatchingFusion):
                fused_part_pcd, transformation_list = fusion_model.fuse_part_pcds(valid_image_path_list, valid_mask_list, valid_points_map_list)
            elif isinstance(fusion_model, TrackingFusion):
                if tracks3d is None:
                    tracks3d = fusion_model.tracking_video(video_frame_list, reconstruction_results["depth"], reconstruction_results["extrinsics"], intrinsics, reconstruction_results["points_mask"])
                valid_tracks3d = [tracks3d[i] for i in valid_frame_ids]
                fused_part_pcd, transformation_list = fusion_model.fuse_part_pcds(valid_image_path_list, valid_mask_list, valid_points_map_list, valid_tracks3d)

            # Evaluate reconstruction
            chamfer_dist, rot_error, trans_error = evaluate_reconstruction(
                pred_pcd=fused_part_pcd,
                pred_extrinsics=reconstruction_results["extrinsics"],
                gt_pcd=data["geometry_data"][role]["part_pcd"],
                gt_extrinsics=data["camera_extrinsics"],
            )
            
            save_pcd(fused_part_pcd, f"{save_pcd_dir}/{role}_fused.ply")
            reconstruction_metrics = {
                "chamfer_distance": chamfer_dist,
                "rotation_error_radians": rot_error,
                "translation_error": trans_error
            }
            save_reconstruction_metrics(reconstruction_metrics, f"{save_pcd_dir}/reconstruction_metrics_{role}.json")
        save_reconstruction_results(reconstruction_results, f"{save_pcd_dir}/reconstruction_results.pkl.gz")
        data_count += 1


def main(config: omegaconf.DictConfig):
    print("Start experiment:", config.name)
    exp_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config.save_root_dir}/{config.name}/{exp_time}"
    print("Results will be saved to:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}/config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)

    set_seed(config.seed)

    eval_dataset = build_dataset(config.dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=identity_collate)

    # Initialize segmentation models
    fusion_model = build_fusion_model(**config.fusion)
    input_modality = config["input_modality"]
    reconstruction_model = build_reconstruction_model(input_modality=input_modality, **config.reconstruction)

    evaluate(input_modality, eval_dataloader, fusion_model, reconstruction_model, config, save_dir)


if __name__ == "__main__":
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    config.update({"debug": args.debug})
    main(config)
