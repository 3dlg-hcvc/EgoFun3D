import numpy as np
import PIL.Image as PILImage
from torchvision.transforms.functional import pil_to_tensor
import torch
from moge.model.v2 import MoGeModel # Let's try MoGe-2
from romatch import roma_indoor
from utils.segment_utils import estimate_se3_transformation, depth2xyz

from typing import Tuple, List


class FeatureMatchingFusion:
    def __init__(self, moge_model_path: str, device: str):
        self.feature_matching_model = roma_indoor(device=device)
        self.monocular_model = MoGeModel.from_pretrained(moge_model_path).to(device)
        self.device = device


    def get_part_pcd(self, image: PILImage.Image, part_mask: np.ndarray, cam_pose: np.ndarray, gt_depth: np.ndarray = None, gt_intrinsics: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        if gt_depth is not None and gt_intrinsics is not None:
            points_map = depth2xyz(gt_depth, gt_intrinsics, cam_type="opencv")
            valid_mask = gt_depth > 0
            valid_part_mask = np.logical_and(valid_mask, part_mask) # H, W
            valid_part_points = points_map[valid_part_mask]
        else:
            image_tensor = pil_to_tensor(image).to(self.device).to(torch.float32) / 255.0
            output = self.monocular_model.infer(image_tensor)
            valid_mask = output["mask"].cpu().numpy()
            valid_part_mask = np.logical_and(valid_mask, part_mask) # H, W
            points_map = output["points"].cpu().numpy() # H, W, 3
            valid_part_points = points_map[valid_part_mask]
        valid_part_points = valid_part_points @ cam_pose[:3, :3].T + cam_pose[:3, 3:].T # point cloud in world coord but not in rest state
        return valid_part_points, points_map


    def compute_part_transformation(self, current_image_path: str, current_point_map: np.ndarray, current_part_mask: np.ndarray, anchor_image_path: str, anchor_point_map: np.ndarray, anchor_part_mask: np.ndarray) -> np.ndarray:
        warp, certainty = self.feature_matching_model.match(anchor_image_path, current_image_path, device=self.device)
        # Sample matches for estimation
        matches, certainty = self.feature_matching_model.sample(warp, certainty)
        # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
        certainty_mask = certainty > 0.95
        matches = matches[certainty_mask]
        certainty = certainty[certainty_mask]
        
        H, W = current_part_mask.shape
        kptsA, kptsB = self.feature_matching_model.to_pixel_coordinates(matches, H, W, H, W)
        kptsA = kptsA.cpu().numpy().astype(np.int32)
        kptsB = kptsB.cpu().numpy().astype(np.int32)
        # Filter keypoints with part masks
        kptsA_index = current_part_mask[kptsA[:,1], kptsA[:,0]]
        kptsB_index = anchor_part_mask[kptsB[:,1], kptsB[:,0]]
        valid_index = np.logical_and(kptsA_index, kptsB_index)
        kptsA = kptsA[valid_index]
        kptsB = kptsB[valid_index]
        # current_part_kpts = kptsA[current_part_mask[kptsA[:,1], kptsA[:,0]]]
        # anchor_part_kpts = kptsB[anchor_part_mask[kptsB[:, 1], kptsB[:,0]]]
        anchor_part_3dkpts = anchor_point_map[kptsA[:,1], kptsA[:,0]]
        current_part_3dkpts = current_point_map[kptsB[:,1], kptsB[:,0]]
        if len(current_part_3dkpts) < 10 or len(anchor_part_3dkpts) < 10:
            print("Not enough keypoints for transformation estimation.")
            return np.eye(4)
        # Estimate transformation
        current2anchor = estimate_se3_transformation(current_part_3dkpts, anchor_part_3dkpts)
        return current2anchor
    

    def fuse_part_pcds(self, image_path_list: List[str], part_mask_list: List[np.ndarray], cam_pose_list: List[np.ndarray], gt_depth_list: List[np.ndarray] = None, gt_intrinsics_list: List[np.ndarray] = None) -> np.ndarray:
        part_pcd_list = []
        transformation_list = []
        anchor_image_path = None
        anchor_point_map = None
        anchor_part_mask = None
        for frame_id, image_path in enumerate(image_path_list):
            image = PILImage.open(image_path).convert("RGB")
            part_mask = part_mask_list[frame_id]
            cam_pose = cam_pose_list[frame_id]
            if gt_depth_list is not None and gt_intrinsics_list is not None:
                gt_depth = gt_depth_list[frame_id]
                gt_intrinsics = gt_intrinsics_list[frame_id]
            else:
                gt_depth = None
                gt_intrinsics = None
            part_pcd_world, current_point_map = self.get_part_pcd(image, part_mask, cam_pose, gt_depth, gt_intrinsics)
            part_pcd_list.append(part_pcd_world)
            
            if frame_id == 0:
                transformation_list.append(np.eye(4))
                anchor_image_path = image_path
                anchor_point_map = current_point_map
                anchor_part_mask = part_mask
            else:
                transformation = self.compute_part_transformation(
                    image_path, current_point_map, part_mask,
                    anchor_image_path, anchor_point_map, anchor_part_mask
                )
                transformation_list.append(transformation)
        
        fused_part_pcd = []
        for part_pcd, transformation in zip(part_pcd_list, transformation_list):
            part_pcd_anchored = part_pcd @ transformation[:3, :3].T + transformation[:3, 3:].T
            fused_part_pcd.append(part_pcd_anchored)
        fused_part_pcd = np.concatenate(fused_part_pcd, axis=0)
        return fused_part_pcd
        