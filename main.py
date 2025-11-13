import argparse
import omegaconf
from datetime import datetime
import numpy as np
import os
from dataset.dataset import BaseDataset, build_dataset
from segmentation.prompt_vlm import VLMPrompter, build_vlm_prompter
from segmentation.seg_zero import SegZero, build_refseg_model
from fusion.fusion import build_fusion_model, BaseFusion, FeatureMatchingFusion, TrackingFusion
from fusion.reconstruction import build_reconstruction_model, BaseReconstruction, ViPEReconstruction
from segmentation.evaluate_segmentation import compute_part_iou_video, compute_part_chamfer_distance, save_segmentation_video, save_pcd, save_vlm_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()


def evaluate(input_modality: str, eval_dataset:BaseDataset, vlm_prompter: VLMPrompter, refseg_model: SegZero, fusion_model: BaseFusion, reconstruction_model: BaseReconstruction, config: omegaconf.DictConfig, save_dir: str):
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
            # run segmentation
            part_description = grouped_results[role]["description"]
            video_frame_list = data["ego_video_rgb_list"]
            pred_mask_list, answer_dict_list, valid_frame_ids = refseg_model.segment_video(video_frame_list, part_description)
            if len(valid_frame_ids) == 0:
                print(f"Warning: No valid frames found for part {role} in video {ego_video_path}. Skipping this part.")
                continue
            # run reconstruction
            if isinstance(reconstruction_model, ViPEReconstruction):
                video_dir = os.path.dirname(data["ego_video_rgb_path_list"][0])
                reconstruction_results = reconstruction_model.reconstruct(video_dir)
            else:
                input_intrinsics = None
                input_extrinsics = None
                input_depth = None
                if input_modality.find("intrinsics") != -1:
                    input_intrinsics = data["ego_video_camera_list"][0]["intrinsics"]
                if input_modality.find("extrinsics") != -1:
                    input_extrinsics = [np.array(data["ego_video_camera_list"][i]["extrinsics"]) for i in range(len(data["ego_video_rgb_list"]))]
                if input_modality.find("depth") != -1:
                    input_depth = data["ego_video_depth_list"]
                reconstruction_results = reconstruction_model.reconstruct(video_frame_list, input_intrinsics, input_extrinsics, input_depth)
            # run fusion
            points_mask_list = reconstruction_results["points_mask"]
            valid_mask_list = []
            for i in valid_frame_ids:
                points_mask = points_mask_list[i]
                pred_mask = pred_mask_list[i]
                combined_mask = np.logical_and(points_mask, pred_mask)
                valid_mask_list.append(combined_mask)
            valid_image_path_list = [data["ego_video_rgb_path_list"][i] for i in valid_frame_ids]
            valid_cam_pose_list = [reconstruction_results["extrinsics"][i] for i in valid_frame_ids]
            intrinsics = reconstruction_results["intrinsics"]
            valid_points_map_list = [reconstruction_results["points"][i] for i in valid_frame_ids]
            valid_depth_list = [reconstruction_results["depth"][i] for i in valid_frame_ids]
            valid_depth_mask_list = [reconstruction_results["points_mask"][i] for i in valid_frame_ids]
            if isinstance(fusion_model, FeatureMatchingFusion):
                fused_part_pcd, transformation_list = fusion_model.fuse_part_pcds(valid_image_path_list, valid_mask_list, valid_points_map_list)
            elif isinstance(fusion_model, TrackingFusion):
                fused_part_pcd, transformation_list = fusion_model.fuse_part_pcds(valid_image_path_list, valid_mask_list, points_mask_list, valid_depth_list, valid_cam_pose_list, intrinsics, valid_depth_mask_list)

            # Evaluate segmentation
            gt_mask_list = data[f"gt_{role}_mask_list"]
            vlm_filtered_iou_list, original_iou_list = compute_part_iou_video(gt_mask_list, pred_mask_list, valid_frame_ids)
            save_2d_segmentation_dir = f"{save_dir}/{data['scene_name']}/{data['seg_id']}/segmentation_{role}"
            if not os.path.exists(save_2d_segmentation_dir):
                os.makedirs(save_2d_segmentation_dir)
            save_segmentation_video(video_frame_list, pred_mask_list, answer_dict_list, valid_frame_ids, original_iou_list, vlm_filtered_iou_list, save_2d_segmentation_dir)

            # Evaluate fused point cloud
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
    fusion_model = build_fusion_model(**config.fusion)
    input_modality = config["input_modality"]
    reconstruction_model = build_reconstruction_model(input_modality=input_modality, **config.reconstruction)

    evaluate(input_modality, eval_dataset, vlm_prompter, refseg_model, fusion_model, reconstruction_model, config, save_dir)


if __name__ == "__main__":
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    main(config)
