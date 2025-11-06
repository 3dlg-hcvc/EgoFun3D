import argparse
import omegaconf
from datetime import datetime
import numpy as np
import os
from dataset.dataset import BaseDataset, build_dataset
from segmentation.prompt_vlm import VLMPrompter, build_vlm_prompter
from segmentation.seg_zero import SegZero, build_refseg_model
from segmentation.fusion import FeatureMatchingFusion
from segmentation.evaluate_segmentation import compute_part_iou_video, compute_part_chamfer_distance, save_segmentation_video, save_pcd, save_vlm_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()


def evaluate(eval_dataset:BaseDataset, vlm_prompter: VLMPrompter, refseg_model: SegZero, fusion_model: FeatureMatchingFusion, config: omegaconf.DictConfig, save_dir: str):
    # Run segmentation
    for data in eval_dataset:
        ego_video_path = data["ego_video_path"]
        grouped_results = vlm_prompter.prompt(ego_video_path)
        if len(grouped_results.keys()) != 2:
            print(f"Warning: VLM did not return two parts for video {ego_video_path}. Skipping this sample.")
            continue
        else:
            save_vlm_dir = f"{save_dir}/{data['scene_name']}/{data['seg_id']}/vlm_narrator"
            if not os.path.exists(save_vlm_dir):
                os.makedirs(save_vlm_dir)
            save_vlm_output(grouped_results, save_vlm_dir)
        for role in grouped_results.keys():
            print(f"Role: {role}, Details: {grouped_results[role]}")
            part_description = grouped_results[role]["description"]
            video_frame_list = data["ego_video_rgb_list"]
            pred_mask_list, answer_dict_list, valid_frame_ids = refseg_model.segment_video(video_frame_list, part_description)
            valid_image_path_list = [data["ego_video_rgb_path_list"][i] for i in valid_frame_ids]
            valid_cam_pose_list = [np.array(data["ego_video_camera_list"][i]["extrinsics"]) for i in valid_frame_ids]
            if config.segmentation.use_gt_depth:
                gt_depth_list = [data["ego_video_depth_list"][i] for i in valid_frame_ids]
                gt_intrinsics_list = [data["ego_video_camera_list"][i]["intrinsics"] for i in valid_frame_ids]
            else:
                gt_depth_list = None
                gt_intrinsics_list = None
            fused_part_pcd = fusion_model.fuse_part_pcds(valid_image_path_list, pred_mask_list, valid_cam_pose_list, gt_depth_list, gt_intrinsics_list)

            gt_mask_list = data[f"gt_{role}_mask_list"]
            iou_list = compute_part_iou_video(gt_mask_list, pred_mask_list, valid_frame_ids)
            save_segmentation_video(video_frame_list, pred_mask_list, answer_dict_list, valid_frame_ids, iou_list, save_dir)

            # Save fused point cloud
            gt_part_pcd = data["gt_pcd_annotation"][role]["part_pcd"]
            chamfer_dist = compute_part_chamfer_distance(gt_part_pcd, fused_part_pcd, refseg_model.device)
            print(f"Fused {role} part Chamfer Distance: {chamfer_dist}")
            save_pcd_dir = f"{save_dir}/{data['scene_name']}/{data['seg_id']}/fused_pcd"
            if not os.path.exists(save_pcd_dir):
                os.makedirs(save_pcd_dir)
            save_pcd(fused_part_pcd, f"{save_pcd_dir}/{role}_fused.ply")
            with open(f"{save_pcd_dir}/{role}_fused_metrics.txt", "w") as f:
                f.write(f"Chamfer Distance: {chamfer_dist}\n")


def main(config: omegaconf.DictConfig):
    print("Start experiment:", config.name)
    exp_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config.save_root_dir}/{config.name}_{exp_time}"
    print("Results will be saved to:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}/config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)

    eval_dataset = build_dataset(config.dataset)

    # Initialize segmentation models
    vlm_prompter = build_vlm_prompter(config.vlm)
    refseg_model = build_refseg_model(config.segmentation)
    fusion_model = FeatureMatchingFusion(config.fusion.moge_model_path, device=config.fusion.device)

    evaluate(eval_dataset, vlm_prompter, refseg_model, fusion_model, config, save_dir)


if __name__ == "__main__":
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    main(config)
