import numpy as np
import PIL.Image as PILImage
from torchvision.transforms.functional import pil_to_tensor
import torch
import os
import zipfile

from moge.model.v2 import MoGeModel # Let's try MoGe-2

from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

from vipe.vipe.utils.io import read_depth_artifacts, read_intrinsics_artifacts, read_pose_artifacts
from vipe.vipe.utils.depth import reliable_depth_mask_range

from utils.segment_utils import depth2xyz

from typing import List, Tuple, Dict


class BaseReconstruction:
    def __init__(self):
        pass

    def reconstruct(self, video_frame_list: List[PILImage.Image], intrinsics: np.ndarray = None, cam_pose_list: List[np.ndarray] = None, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class NaiveReconstruction(BaseReconstruction):
    def __init__(self):
        pass

    def reconstruct(self, video_frame_list: List[PILImage.Image], intrinsics: np.ndarray, cam_pose_list: List[np.ndarray], depth_frame_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        point_map_list = []
        valid_mask_list = []
        for frame_idx in range(len(video_frame_list)):
            gt_depth = depth_frame_list[frame_idx]
            gt_intrinsics = intrinsics[frame_idx]
            points_map = depth2xyz(gt_depth, gt_intrinsics, cam_type="opencv")
            cam_pose = cam_pose_list[frame_idx]
            ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
            points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
            valid_mask = np.ones_like(gt_depth, dtype=bool)
            valid_mask_list.append(valid_mask)
        return {"rgb": video_frame_list, "intrinsics": intrinsics, "extrinsics": np.stack(cam_pose_list), "depth": np.stack(depth_frame_list), "points": np.stack(point_map_list), "points_mask": np.stack(valid_mask_list)}
    

class MoGeReconstruction(BaseReconstruction):
    def __init__(self, model_path: str = "Ruicheng/moge-2-vitl-normal", device: str = "cuda"):
        self.model = MoGeModel.from_pretrained(model_path).to(device)
        self.device = device

    def reconstruct(self, video_frame_list: List[PILImage.Image], intrinsics: np.ndarray, cam_pose_list: List[np.ndarray], depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        w = video_frame_list[0].width
        fov_x = 2 * np.arctan(w / (2 * intrinsics[0, 0]))
        point_map_list = []
        valid_mask_list = []
        for frame_idx, video_frame in enumerate(video_frame_list):
            if depth_frame_list is not None:
                depth_frame = depth_frame_list[frame_idx]
                points_map = depth2xyz(depth_frame, intrinsics, cam_type="opencv")
                valid_mask = np.ones_like(depth_frame, dtype=bool)
            else:
                video_frame_tensor = pil_to_tensor(video_frame).to(self.device).to(torch.float32) / 255.0  # (3, H, W)
                output = self.model.infer(video_frame_tensor, fov_x=fov_x)
                points_map = output["points"].cpu().numpy() # H, W, 3
                valid_mask = output["mask"].cpu().numpy().astype(bool)
            cam_pose = cam_pose_list[frame_idx]
            ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
            points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
            valid_mask_list.append(valid_mask)
        return {"rgb": video_frame_list, "intrinsics": intrinsics, "extrinsics": cam_pose_list, "depth": depth_frame_list, "points": point_map_list, "points_mask": valid_mask_list}
    

class SpatrackerReconstruction(BaseReconstruction):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = VGGT4Track.from_pretrained(model_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    def reconstruct(self, video_frame_list: List[PILImage.Image], intrinsics: np.ndarray = None, cam_pose_list: List[np.ndarray] = None, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        video_tensor_list = []
        for video_frame in video_frame_list:
            video_tensor = pil_to_tensor(video_frame).to(torch.float32).to(self.device)
            video_tensor_list.append(video_tensor)
        video_tensor = torch.stack(video_tensor_list) # N, C, H, W

        if depth_frame_list is None or cam_pose_list is None or intrinsics is None:
             # process the image tensor
            video_tensor = preprocess_image(video_tensor)[None]
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Predict attributes including cameras, depth maps, and point maps.
                    predictions = self.model(video_tensor / 255)
            if depth_frame_list is None:
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
                depth_tensor = depth_map.squeeze().cpu().numpy()
                unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
            else:
                depth_tensor = np.stack(depth_frame_list)
                unc_metric = np.ones_like(depth_tensor, dtype=bool)
                
            if cam_pose_list is None:
                extrinsic = predictions["poses_pred"]
                extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
                extrs = extrinsic.squeeze().cpu().numpy()
            else:
                extrs = np.stack(cam_pose_list)
            if intrinsics is None:
                intrinsic = predictions["intrs"]
                intrs = intrinsic.squeeze().cpu().numpy()
            else:
                intrs = intrinsics.copy()
            video_tensor = video_tensor.squeeze()

        point_map_list = []
        for frame_idx in range(len(depth_tensor)):
            depth = depth_tensor[frame_idx]
            points_map = depth2xyz(depth, intrs, cam_type="opencv")
            cam_pose = extrs[frame_idx]
            ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
            points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
        return {"rgb": video_frame_list, "intrinsics": intrs, "extrinsics": extrs, "depth": depth_tensor, "points": np.stack(point_map_list), "points_mask": unc_metric}
    

class ViPEReconstruction(BaseReconstruction):
    def reconstruct(self, video_dir: str) -> Dict[str, np.ndarray]:
        tmp_dir = "./tmp_vipe_reconstruction"
        os.system(f"vipe infer --image-dir {video_dir} --output {tmp_dir}")
        
        base_name = os.path.basename(video_dir)
        depth_zip_file = os.path.join(tmp_dir, "depth", f"{base_name}.zip")
        depth_frame_list = []
        depth_mask_list = []
        for frame_id, depth_tensor in read_depth_artifacts(depth_zip_file):
            depth_mask_tensor = reliable_depth_mask_range(depth_tensor)
            depth_np = depth_tensor.numpy()
            depth_frame_list.append(depth_np)
            depth_mask_list.append(depth_mask_tensor.numpy())
        intrinsics_file = os.path.join(tmp_dir, "intrinsics", f"{base_name}.npz")
        inds, intrinsics_tensor, camera_types = read_intrinsics_artifacts(intrinsics_file)
        fx, fy, cx, cy = intrinsics_tensor[0].cpu().numpy()
        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        cam_pose_file = os.path.join(tmp_dir, "pose", f"{base_name}.npz")
        inds, pose_tensor = read_pose_artifacts(cam_pose_file)
        point_map_list = []
        cam_pose_list = []
        for frame_id, depth in enumerate(depth_frame_list):
            points_map = depth2xyz(depth, intrinsics, cam_type="opencv")
            cam_pose = pose_tensor[frame_id].matrix().numpy()
            points_map_homogeneous = np.concatenate([points_map, np.ones((points_map.shape[0], points_map.shape[1], 1))], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
            cam_pose_list.append(cam_pose)
        return {"intrinsics": intrinsics, "extrinsics": np.stack(cam_pose_list), "depth": np.stack(depth_frame_list), "points": np.stack(point_map_list), "points_mask": np.stack(depth_mask_list)}

