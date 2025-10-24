import argparse
import omegaconf
from datetime import datetime
import os
from dataset.dataset import build_dataset
from segmentation.prompt_vlm import build_vlm_prompter
from segmentation.seg_zero import build_refseg_model
from segmentation.evaluate_segmentation import compute_part_chamfer_distance, save_segmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()


def evaluate(eval_dataset, vlm_prompter, refseg_model, config, save_dir):
    # Run segmentation
    for data in eval_dataset:
        gt_annotation = data["gt_annotation"]
        ego_video_path = data["ego_video_path"]
        grouped_results = vlm_prompter.prompt(ego_video_path)
        for role in grouped_results.keys():
            print(f"Role: {role}, Details: {grouped_results[role]}")
            part_description = grouped_results[role]["description"]
            video_frame_list = data["ego_video_frame_path_list"]
            for frame_id, video_frame in enumerate(video_frame_list):
                mask, answer_dict = refseg_model.segment(video_frame, part_description)
                if config.segmentation.use_gt_depth:
                    camera = data["ego_video_camera_list"][frame_id]
                    gt_depth = data["ego_video_depth_list"][frame_id]
                    gt_intrinsics = camera["intrinsics"]
                    cam_pose = camera["extrinsics"]
                    part_pcd = refseg_model.get_part_pcd(video_frame, mask, cam_pose, gt_depth, gt_intrinsics)
                else:
                    camera = data["ego_video_camera_list"][frame_id]
                    cam_pose = camera["extrinsics"]
                    part_pcd = refseg_model.get_part_pcd(video_frame, mask, cam_pose)
                gt_part_pcd = gt_annotation[role]["part_pcd"]
                chamfer_dist = compute_part_chamfer_distance(gt_part_pcd, part_pcd, config.segmentation.device)
                print(f"Frame {frame_id}: Chamfer Distance for role {role}: {chamfer_dist}")
                answer_dict["chamfer_distance"] = chamfer_dist
                save_segmentation(
                    video_frame, mask, answer_dict,
                    save_dir=f"{save_dir}/{data['scene_name']}/{data['seg_id']}/segmentation/{role}",
                    id=f"{frame_id:04d}"
                )


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
