import argparse
import gzip
import omegaconf
from datetime import datetime
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader
import pickle

from PIL import Image as PILImage

from dataset.dataset import build_dataset
from segmentation.ref_seg import RefSeg, build_refseg_model
from fusion.fusion import build_fusion_model, BaseFusion, FeatureMatchingFusion, TrackingFusion
from fusion.reconstruction import build_reconstruction_model, BaseReconstruction, ViPEReconstruction
from fusion.evaluate_reconstruction import save_mesh, save_reconstruction_metrics, evaluate_reconstruction, save_pcd, save_reconstruction_results
from articulation.base import build_articulation_estimation_model, ArticulationEstimation
from articulation.evaluate_articulation import compute_joint_error, save_articulation_metrics, save_articulation_results
from VLM.prompt_vlm import build_vlm_prompter, VLMPrompter
from function.evaluate_function import compute_function_error, save_function_results
from utils.reconstruction_utils import refine_point_mask

MAX_JOINT_ORI_ERROR = np.pi / 2
MAX_JOINT_POS_ERROR = 1.0


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
    return batch[0]


def _get_mesh_save_path(save_dir: str, role: str, mesh_format: str = "glb") -> str:
    return os.path.join(save_dir, f"{role}_mesh.{mesh_format}")


def evaluate(
    input_modality: str,
    eval_dataloader: DataLoader,
    refseg_model: RefSeg,
    fusion_model: BaseFusion,
    reconstruction_model: BaseReconstruction,
    articulation_model: ArticulationEstimation,
    function_vlm: VLMPrompter,
    config: omegaconf.DictConfig,
    save_dir: str,
):
    if config.debug:
        print("Debug mode enabled: Limiting evaluation dataset to 1 sample.")
        max_dataset_size = 1
    data_count = 0

    for data in eval_dataloader:
        print("Evaluating data:", data["video_name"])
        if config.debug and data_count >= max_dataset_size:
            break

        video_name = data["video_name"]
        video_frame_list = data["rgb_list"]  # list of numpy arrays (H, W, 3)
        pil_frame_list = [PILImage.fromarray(frame) for frame in video_frame_list]

        reconstruction_results = None
        tracks3d = None
        kptsA_origin_dict = {}
        kptsB_origin_dict = {}
        pred_masks_by_role = {}
        transformation_list_by_role = {}

        # ── Segmentation + Reconstruction + Fusion per role ────────────────
        for role in ["receptor", "effector"]:
            part_description = data[f"{role}_name"]
            print(f"Segmenting role '{role}': {part_description}")

            pred_mask_list, answer_dict_list, valid_frame_ids = refseg_model.segment_video(
                pil_frame_list, part_description
            )
            pred_masks_by_role[role] = np.stack(pred_mask_list, axis=0)  # (T, H, W)

            save_seg_dir = os.path.join(save_dir, video_name, f"segmentation_{role}")
            os.makedirs(save_seg_dir, exist_ok=True)
            for i, mask in enumerate(pred_mask_list):
                np.save(os.path.join(save_seg_dir, f"segmentation_mask_{i:04d}.npy"), mask)

            if len(valid_frame_ids) == 0:
                print(f"Warning: No valid frames for role '{role}'. Skipping reconstruction for this role.")
                continue

            # ── Reconstruction (shared across roles) ───────────────────────
            if reconstruction_results is None:
                init_extrinsics = data["camera_extrinsics"][0]
                if isinstance(reconstruction_model, ViPEReconstruction):
                    reconstruction_results = reconstruction_model.reconstruct(
                        data["video_path"], init_extrinsics, data["sample_indices"]
                    )
                else:
                    input_intrinsics = None
                    input_extrinsics = None
                    input_depth = None
                    if input_modality.find("intrinsics") != -1:
                        input_intrinsics = data["camera_intrinsics"]
                    if input_modality.find("extrinsics") != -1:
                        input_extrinsics = data["camera_extrinsics"]
                    reconstruction_results = reconstruction_model.reconstruct(
                        video_frame_list, init_extrinsics, input_intrinsics, input_extrinsics, input_depth
                    )
                if reconstruction_results is None:
                    print("Reconstruction failed. Skipping this sample.")
                    break

                reconstruction_results = refine_point_mask(reconstruction_results)

            # ── Fusion ─────────────────────────────────────────────────────
            points_mask_list = reconstruction_results["points_mask"]
            valid_mask_list = []
            for i in valid_frame_ids:
                combined_mask = np.logical_and(points_mask_list[i], pred_mask_list[i])
                valid_mask_list.append(combined_mask)

            valid_video_frame_list = [video_frame_list[i] for i in valid_frame_ids]
            valid_points_map_list = [reconstruction_results["points"][i] for i in valid_frame_ids]

            if isinstance(fusion_model, FeatureMatchingFusion):
                fused_part_pcd, transformation_list, kptsA_origin_dict, kptsB_origin_dict = (
                    fusion_model.fuse_part_pcds(
                        valid_video_frame_list,
                        valid_mask_list,
                        valid_points_map_list,
                        kptsA_origin_dict,
                        kptsB_origin_dict,
                    )
                )
            elif isinstance(fusion_model, TrackingFusion):
                if tracks3d is None:
                    intrinsics = reconstruction_results["intrinsics"]
                    tracks3d = fusion_model.tracking_video(
                        video_frame_list,
                        reconstruction_results["depth"],
                        reconstruction_results["extrinsics"],
                        intrinsics,
                        reconstruction_results["points_mask"],
                    )
                valid_tracks3d = [tracks3d[i] for i in valid_frame_ids]
                fused_part_pcd, transformation_list = fusion_model.fuse_part_pcds(
                    valid_video_frame_list, valid_mask_list, valid_points_map_list, valid_tracks3d
                )
            transformation_list_by_role[role] = transformation_list

            # ── Evaluate reconstruction ────────────────────────────────────
            chamfer_dist, rot_error, trans_error = evaluate_reconstruction(
                pred_pcd=fused_part_pcd,
                pred_extrinsics=reconstruction_results["extrinsics"],
                gt_pcd=data["geometry_data"][role]["part_pcd"],
                gt_extrinsics=data["camera_extrinsics"],
            )
            reconstruction_metrics = {
                "chamfer_distance": chamfer_dist,
                "rotation_error_radians": rot_error,
                "translation_error": trans_error,
            }
            save_recon_dir = os.path.join(save_dir, video_name, "reconstruction")
            os.makedirs(save_recon_dir, exist_ok=True)
            save_pcd(fused_part_pcd, os.path.join(save_recon_dir, f"{role}_fused.ply"))
            save_reconstruction_metrics(
                reconstruction_metrics,
                os.path.join(save_recon_dir, f"reconstruction_metrics_{role}.json"),
            )
            print(f"[{role}] chamfer={chamfer_dist:.4f}, rot_err={rot_error:.4f}, trans_err={trans_error:.4f}")

            if config.get("save_mesh", False):
                save_mesh(
                    reconstruction_results=reconstruction_results,
                    image_list=video_frame_list,
                    mask_list=pred_masks_by_role[role],
                    transformation_list=transformation_list,
                    save_path=_get_mesh_save_path(save_recon_dir, role),
                    observation_indices=np.asarray(valid_frame_ids, dtype=int),
                    num_observations=3,
                )

        # Save shared reconstruction results once
        if reconstruction_results is not None:
            save_recon_dir = os.path.join(save_dir, video_name, "reconstruction")
            os.makedirs(save_recon_dir, exist_ok=True)
            save_reconstruction_results(
                reconstruction_results,
                os.path.join(save_recon_dir, "reconstruction_results.pkl.gz"),
            )

            if config.get("save_mesh", False):
                receptor_masks = pred_masks_by_role.get("receptor")
                effector_masks = pred_masks_by_role.get("effector")
                if receptor_masks is not None and effector_masks is not None:
                    base_mask_list = np.logical_and(
                        data["object_mask_list"],
                        np.logical_not(np.logical_or(receptor_masks, effector_masks)),
                    )
                    base_valid_frame_ids = [i for i, mask in enumerate(base_mask_list) if mask.sum() > 0]
                    if len(base_valid_frame_ids) > 0:
                        base_valid_mask_list = [
                            np.logical_and(reconstruction_results["points_mask"][i], base_mask_list[i])
                            for i in base_valid_frame_ids
                        ]
                        base_valid_points_map_list = [reconstruction_results["points"][i] for i in base_valid_frame_ids]
                        base_video_frame_list = [video_frame_list[i] for i in base_valid_frame_ids]
                        if isinstance(fusion_model, FeatureMatchingFusion):
                            base_feature_fusion_model = fusion_model
                        else:
                            base_feature_fusion_model = FeatureMatchingFusion(device=config.fusion.device)
                        _, base_transformation_list, _, _ = base_feature_fusion_model.fuse_part_pcds(
                            base_video_frame_list,
                            base_valid_mask_list,
                            base_valid_points_map_list,
                            kptsA_origin_dict,
                            kptsB_origin_dict,
                        )
                        save_mesh(
                            reconstruction_results=reconstruction_results,
                            image_list=video_frame_list,
                            mask_list=base_mask_list,
                            transformation_list=base_transformation_list,
                            save_path=_get_mesh_save_path(save_recon_dir, "base"),
                            observation_indices=np.asarray(base_valid_frame_ids, dtype=int),
                            num_observations=3,
                        )

        # ── Articulation estimation ─────────────────────────────────────────
        if articulation_model is not None and reconstruction_results is not None:
            articulation_results = {}
            save_articulation_dir = os.path.join(save_dir, video_name, "articulation")
            os.makedirs(save_articulation_dir, exist_ok=True)

            for role in ["receptor", "effector"]:
                gt_articulation = data[f"{role}_articulation"]
                if gt_articulation is None:
                    print(f"No GT articulation for role '{role}', skipping.")
                    articulation_results[role] = "No GT articulation, skipping evaluation for this role."
                    continue

                mask_list_role = pred_masks_by_role.get(role)
                if mask_list_role is None:
                    articulation_results[role] = "No segmentation results for this role."
                    continue

                articulation_result = articulation_model.articulation_estimation(
                    video_frame_list, reconstruction_results, mask_list_role
                )
                articulation_results[role] = articulation_result

                if articulation_result is None:
                    print(f"Articulation estimation failed for role '{role}'.")
                    joint_ori_error = MAX_JOINT_ORI_ERROR
                    joint_pos_error = MAX_JOINT_POS_ERROR
                    joint_type_correct = False
                else:
                    joint_ori_error, joint_pos_error, joint_type_correct = compute_joint_error(
                        gt_articulation, articulation_result
                    )
                    print(
                        f"[{role}] joint_ori_err={joint_ori_error:.4f}, "
                        f"joint_pos_err={joint_pos_error:.4f}, type_correct={joint_type_correct}"
                    )

                articulation_metrics = {
                    "joint axis error": joint_ori_error,
                    "joint position error": joint_pos_error,
                    "joint type correct": joint_type_correct,
                }
                save_articulation_metrics(
                    articulation_metrics,
                    os.path.join(save_articulation_dir, f"articulation_metrics_{role}.json"),
                )

            save_articulation_results(
                articulation_results,
                os.path.join(save_articulation_dir, "articulation_results.json"),
            )

        # ── Function prediction ─────────────────────────────────────────────
        if function_vlm is not None:
            save_function_dir = os.path.join(save_dir, video_name, "function")
            os.makedirs(save_function_dir, exist_ok=True)

            receptor_mask_list = pred_masks_by_role.get("receptor", data["receptor_mask_list"])
            effector_mask_list = pred_masks_by_role.get("effector", data["effector_mask_list"])

            function_results = function_vlm.prompt_function(
                video_frame_list, receptor_mask_list, effector_mask_list
            )

            gt_function = data["function_annotation"]
            if function_results is None:
                print("No function results from VLM.")
                function_error_metrics = {"physical_effect": False, "numerical_function": False}
                function_results = {"1": None, "2": None}
            else:
                function_error_metrics = compute_function_error(gt_function, function_results)
                print(f"function metrics: {function_error_metrics}")

            save_function_results(function_error_metrics, os.path.join(save_function_dir, "function_metrics.json"))
            save_function_results(function_results, os.path.join(save_function_dir, "function_results.json"))

        data_count += 1


def main(config: omegaconf.DictConfig):
    print("Start experiment:", config.name)
    exp_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config.save_root_dir}/{config.name}/{exp_time}"
    print("Results will be saved to:", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/config.yaml", "w") as f:
        omegaconf.OmegaConf.save(config, f)

    if "seed" in config:
        set_seed(config.seed)

    eval_dataset = build_dataset(config.dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=identity_collate)

    refseg_model = build_refseg_model(config.segmentation)
    fusion_model = build_fusion_model(config.fusion)
    input_modality = config["input_modality"]
    reconstruction_model = build_reconstruction_model(input_modality=input_modality, **config.reconstruction)

    articulation_model = None
    if "articulation" in config and config.articulation is not None:
        articulation_model = build_articulation_estimation_model(config.articulation)

    function_vlm = None
    if "vlm_function" in config and config.vlm_function is not None:
        function_vlm = build_vlm_prompter(config.vlm_function)

    evaluate(input_modality, eval_dataloader, refseg_model, fusion_model, reconstruction_model, articulation_model, function_vlm, config, save_dir)


if __name__ == "__main__":
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    config.update({"debug": args.debug})
    main(config)
