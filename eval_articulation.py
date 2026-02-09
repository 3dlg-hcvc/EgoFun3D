import argparse
import omegaconf
from datetime import datetime
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader
from dataset.dataset import Dataset, build_dataset
from articulation.base import build_articulation_estimation_model, ArticulationEstimation
from articulation.evaluate_articulation import compute_joint_error, save_articulation_metrics, save_articulation_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--resume', type=str, help='Path to the experiment directory to resume from')
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


def evaluate(eval_dataloader: DataLoader, articulation_estimation_model: ArticulationEstimation, config: omegaconf.DictConfig, save_dir: str):
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
        articulation_results = {}
        reconstruction_results = None
        save_articulation_dir = os.path.join(save_dir, data["video_name"], "articulation")
        if os.path.exists(f"{save_articulation_dir}/articulation_results.pkl.gz"):
            print("Articulation results already exist, skipping articulation and evaluation for this sample.")
            continue
        if not os.path.exists(save_articulation_dir):
            os.makedirs(save_articulation_dir)
        for role in ["receiver", "effector"]:
            gt_articulation = data[f"{role}_articulation"]
            if gt_articulation is None:
                articulation_results[role] = "No GT articulation, skipping evaluation for this role."
                continue
            video_frame_list = data["rgb_list"]
            mask_list = data[f"{role}_mask_list"]
            # valid_frame_ids = [i for i, mask in enumerate(mask_list) if mask.sum() > 0]

            # load reconstruction
            if reconstruction_results is None:
                reconstruction_results_path = os.path.join(config.reconstruction_results_dir, data["video_name"], "reconstruction/reconstruction_results.pkl.gz")
                if os.path.exists(reconstruction_results_path):
                    print("Loading existing reconstruction results from:", reconstruction_results_path)
                    with open(reconstruction_results_path, "rb") as f:
                        reconstruction_results = torch.load(f)
            if reconstruction_results is None:
                print("Reconstruction failed, skipping this sample.")
                break
            # run articulation estimation
            # if config.debug:
            #     full_points = reconstruction_results["points"]
            #     full_masks = reconstruction_results["points_mask"]
            #     points_list = []
            #     for i, (points, mask) in enumerate(zip(full_points, full_masks)):
            #         # part_points = points[mask]
            #         # save_pcd(points.reshape(-1, 3), f"{save_pcd_dir}/frame_{i}_full_points.ply")
            #         points_list.append(points.reshape(-1, 3))
            #     save_pcd_dir_debug = os.path.join(save_dir, data["video_name"], "debug_full_reconstruction")
            #     if not os.path.exists(save_pcd_dir_debug):
            #         os.makedirs(save_pcd_dir_debug)
            #     save_pcd(np.concatenate(points_list, axis=0), f"{save_pcd_dir_debug}/{role}_full_reconstruction.ply")
            articulation_results[role] = articulation_estimation_model.articulation_estimation(video_frame_list, reconstruction_results, mask_list)

            # Evaluate reconstruction
            joint_ori_error, joint_pos_error, joint_type_correct = compute_joint_error(
                gt_articulation,
                articulation_results[role]
            )
            
            # save_pcd(fused_part_pcd, f"{save_pcd_dir}/{role}_fused.ply")
            articulation_metrics = {
                "joint axis error": joint_ori_error,
                "joint position error": joint_pos_error,
                "joint type error": joint_type_correct
            }
            save_articulation_metrics(articulation_metrics, f"{save_articulation_dir}/articulation_metrics_{role}.json")
        save_articulation_results(articulation_results, f"{save_articulation_dir}/articulation_results.json")
        data_count += 1


def main(config: omegaconf.DictConfig):
    print("Start experiment:", config.name)
    if "save_dir" in config and config.save_dir is not None:
        print("Resuming from:", config.save_dir)
        save_dir = config.save_dir
    else:
        exp_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"{config.save_root_dir}/{config.name}/{exp_time}"
        config.update({"save_dir": save_dir})
    print("Results will be saved to:", save_dir)
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


if __name__ == "__main__":
    args = parse_args()
    if args.resume is not None:
        config_path = os.path.join(args.resume, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found in resume directory: {config_path}")
        print("Loading config from resume directory:", config_path)
        config = omegaconf.OmegaConf.load(config_path)
        config.update({"save_dir": args.resume})
    else:
        config = omegaconf.OmegaConf.load(args.config)
    config.update({"debug": args.debug})
    main(config)
