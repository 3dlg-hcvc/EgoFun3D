import numpy as np
import PIL.Image as PILImage
from torchvision.transforms.functional import pil_to_tensor
import torch
from romatch import roma_indoor
from SpaTrackerV2.models.SpaTrackV2.models.predictor import Predictor
from SpaTrackerV2.models.SpaTrackV2.models.utils import get_points_on_a_grid

from utils.segment_utils import estimate_se3_transformation

from typing import Tuple, List


class FeatureMatchingFusion:
    def __init__(self, device: str):
        self.feature_matching_model = roma_indoor(device=device)
        # self.monocular_model = MoGeModel.from_pretrained(moge_model_path).to(device)
        self.device = device

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

    def fuse_part_pcds(self, image_path_list: List[PILImage.Image], part_mask_list: List[np.ndarray], points_map_list: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        part_pcd_list = []
        transformation_list = []
        anchor_image_path = None
        anchor_point_map = None
        anchor_part_mask = None
        for frame_id, image_path in enumerate(image_path_list):
            part_mask = part_mask_list[frame_id]
            current_point_map = points_map_list[frame_id]
            part_pcd_world = current_point_map[part_mask]
            # part_pcd_world, current_point_map = self.get_part_pcd(image, part_mask, cam_pose, gt_depth, gt_intrinsics)
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
        return fused_part_pcd, transformation_list
        

class TrackingFusion:
    def __init__(self, track_mode: str = "offline", vo_points: int = 756, grid_size: int = 10, device: str = "cuda"):
        if track_mode == "offline":
            self.model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        else:
            self.model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

        # config the model; the track_num is the number of points in the grid
        self.model.spatrack.track_num = vo_points
        
        self.model.eval()
        self.model.to(device)

        self.grid_size = grid_size
        self.device = device

    def tracking_video(self, video_frame_list: List[PILImage.Image], depth_frame_list: List[np.ndarray], cam_pose_list: List[np.ndarray], intrinsics: np.ndarray, depth_mask_list: List[np.ndarray]):
        video_tensor_list = []
        for video_frame in video_frame_list:
            video_tensor = pil_to_tensor(video_frame).to(torch.float32).to(self.device)
            video_tensor_list.append(video_tensor)
        video_tensor = torch.stack(video_tensor_list) # N, C, H, W
        
        if type(depth_mask_list) is list:
            unc_metric = np.stack(depth_mask_list)
        else:
            unc_metric = depth_mask_list
        if type(depth_frame_list) is list:
            depth_tensor = np.stack(depth_frame_list)
        else:
            depth_tensor = depth_frame_list
        if type(cam_pose_list) is list:
            extrs = np.stack(cam_pose_list)
        else:
            extrs = cam_pose_list
        
        mask = np.ones_like(video_tensor[0,0].cpu().numpy(), dtype=bool)
        # get frame H W
        frame_H, frame_W = video_tensor.shape[2:]
        grid_pts = get_points_on_a_grid(self.grid_size, (frame_H, frame_W), device="cpu")
        # Sample mask values at grid points and filter out points where mask=0
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
        query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

        # Run model inference
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                c2w_traj, intrs, point_map, conf_depth,
                track3d_pred, track2d_pred, vis_pred, conf_pred, video
            ) = self.model.forward(video_tensor, depth=depth_tensor,
                                   intrs=intrinsics, extrs=extrs, 
                                   queries=query_xyt,
                                   fps=1, full_point=False, iters_track=4,
                                   query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                                   support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        tracks3d = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        return tracks3d
    
    def fuse_part_pcds(self, video_frame_list: List[PILImage.Image], part_mask_list: List[np.ndarray], points_map_list: List[np.ndarray],
                       depth_frame_list: List[np.ndarray], cam_pose_list: List[np.ndarray], intrinsics: np.ndarray, depth_mask_list: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        tracks3d = self.tracking_video(video_frame_list, depth_frame_list, cam_pose_list, intrinsics, depth_mask_list)
        fused_part_pcd = []
        transformation_list = []
        anchor_track_points = None
        for frame_id in range(len(video_frame_list)):
            part_mask = part_mask_list[frame_id]
            if frame_id == 0:
                anchor_track_points = tracks3d[frame_id]
                transformation = np.eye(4)
            else:
                current_track_points = tracks3d[frame_id]
                transformation = estimate_se3_transformation(current_track_points[part_mask], anchor_track_points[part_mask])
            transformation_list.append(transformation)
            points_map = points_map_list[frame_id]
            part_pcd = points_map[part_mask]
            part_pcd = part_pcd @ transformation[:3, :3].T + transformation[:3, 3:].T
            fused_part_pcd.append(part_pcd)
        fused_part_pcd = np.concatenate(fused_part_pcd, axis=0)
        return fused_part_pcd, transformation_list