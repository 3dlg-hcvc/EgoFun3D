import argparse
import omegaconf
from datetime import datetime
import numpy as np
import os
from dataset.dataset import BaseDataset, build_dataset
from segmentation.prompt_vlm import VLM_Prompter, build_vlm_prompter
from segmentation.seg_zero import SegZero, build_refseg_model
from segmentation.evaluate_segmentation import compute_part_iou, compute_part_chamfer_distance, save_segmentation, save_pcd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()


def evaluate(eval_dataset:BaseDataset, vlm_prompter: VLM_Prompter, refseg_model: SegZero, config: omegaconf.DictConfig, save_dir: str):
    # Run segmentation
    for data in eval_dataset:
        ego_video_path = data["ego_video_path"]
        grouped_results = {}
        while len(grouped_results.keys()) != 2:
            grouped_results = vlm_prompter.prompt(ego_video_path)
        for role in grouped_results.keys():
            print(f"Role: {role}, Details: {grouped_results[role]}")
            part_description = grouped_results[role]["description"]
            video_frame_list = data["ego_video_rgb_list"]
            part_pcd_list = []
            transformation_list = []
            anchor_image_path = data["ego_video_rgb_path_list"][0]
            anchor_point_map = None  # To be computed when needed
            anchor_part_mask = None  # To be computed when needed
            for frame_id, video_frame in enumerate(video_frame_list):
                mask, answer_dict = refseg_model.segment(video_frame, part_description)
                gt_mask = data[f"gt_{role}_mask_list"][frame_id]
                iou = compute_part_iou(gt_mask, mask)
                answer_dict["iou"] = iou
                save_segmentation(
                    video_frame, mask, answer_dict,
                    save_dir=f"{save_dir}/{data['scene_name']}/{data['seg_id']}/segmentation/{role}",
                    id=f"{frame_id:04d}"
                )

                cam_pose = np.array(data["ego_video_camera_list"][frame_id]["extrinsics"])
                if config.use_gt_depth:
                    gt_depth = data["ego_video_depth_list"][frame_id]
                    gt_intrinsics = data["ego_video_camera_list"][frame_id]["intrinsics"]
                else:
                    gt_depth = None
                    gt_intrinsics = None
                part_pcd_world, current_point_map = refseg_model.get_part_pcd(video_frame, mask, cam_pose, gt_depth, gt_intrinsics)
                part_pcd_list.append(part_pcd_world)
                if frame_id == 0:
                    transformation_list.append(np.eye(4))
                    anchor_point_map = current_point_map
                    anchor_part_mask = mask
                else:
                    current_image_path = data["ego_video_rgb_path_list"][frame_id]
                    transformation = refseg_model.compute_part_transformation(
                        current_image_path, current_point_map, mask,
                        anchor_image_path, anchor_point_map, anchor_part_mask
                    )
                    transformation_list.append(transformation)
            fused_part_pcd = refseg_model.fuse_part_pcds(part_pcd_list, transformation_list)
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

    eval_dataset = build_dataset(config.dataset)

    # Initialize segmentation models
    vlm_prompter = build_vlm_prompter(config.vlm)
    refseg_model = build_refseg_model(config.segmentation)

    evaluate(eval_dataset, vlm_prompter, refseg_model, config, save_dir)


if __name__ == "__main__":
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    main(config)
