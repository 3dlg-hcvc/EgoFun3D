import numpy as np
import PIL.Image as PILImage
from torchvision.transforms.functional import pil_to_tensor
import torch
import os
import shutil
from scipy.ndimage import zoom
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import sys
sys.path.append("third_party/SpaTrackerV2/")
from third_party.SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from third_party.SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
sys.path.append("third_party/vipe/")
from third_party.vipe.vipe.utils.io import read_depth_artifacts, read_intrinsics_artifacts, read_pose_artifacts
from third_party.vipe.vipe.utils.depth import reliable_depth_mask_range
sys.path.append("third_party/Depth-Anything-3/")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.hf_utils.viz import predictions_to_glb, image_mesh
from mapanything.utils.geometry import depthmap_to_world_frame

from utils.reconstruction_utils import depth2xyz

from typing import List, Tuple, Dict


class BaseReconstruction:
    def __init__(self):
        pass

    def reconstruct(self, video_frame_list: List[PILImage.Image], init_extrinsics: np.ndarray, intrinsics: np.ndarray = None, cam_pose_list: np.ndarray = None, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class NaiveReconstruction(BaseReconstruction):
    def __init__(self):
        pass

    def reconstruct(self, video_frame_list: List[PILImage.Image], init_extrinsics: np.ndarray, intrinsics: np.ndarray, cam_pose_list: np.ndarray, depth_frame_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        point_map_list = []
        valid_mask_list = []
        cam2init = init_extrinsics @ np.linalg.inv(cam_pose_list[0])
        for frame_idx in range(len(video_frame_list)):
            gt_depth = depth_frame_list[frame_idx]
            points_map = depth2xyz(gt_depth, intrinsics, cam_type="opencv")
            cam_pose = cam2init @ cam_pose_list[frame_idx]
            ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
            points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
            valid_mask = np.ones_like(gt_depth, dtype=bool)
            valid_mask_list.append(valid_mask)
        return {"rgb": video_frame_list, 
                "intrinsics": intrinsics, 
                "extrinsics": np.stack(cam_pose_list), 
                "depth": np.stack(depth_frame_list), 
                "points": np.stack(point_map_list), 
                "points_mask": np.stack(valid_mask_list)}
    

class MoGeReconstruction(BaseReconstruction):
    def __init__(self, model_path: str = "Ruicheng/moge-2-vitl-normal", device: str = "cuda"):
        self.model = MoGeModel.from_pretrained(model_path).to(device)
        self.device = device

    def reconstruct(self, video_frame_list: List[PILImage.Image], init_extrinsics: np.ndarray, intrinsics: np.ndarray, cam_pose_list: np.ndarray, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        w = video_frame_list[0].width
        fov_x = 2 * np.arctan(w / (2 * intrinsics[0, 0]))
        point_map_list = []
        valid_mask_list = []
        cam2init = init_extrinsics @ np.linalg.inv(cam_pose_list[0])
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
            cam_pose = cam2init @ cam_pose_list[frame_idx]
            ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
            points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
            valid_mask_list.append(valid_mask)
        return {"rgb": video_frame_list, 
                "intrinsics": intrinsics, 
                "extrinsics": cam_pose_list, 
                "depth": depth_frame_list, 
                "points": point_map_list, 
                "points_mask": valid_mask_list}
    

class SpatrackerReconstruction(BaseReconstruction):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = VGGT4Track.from_pretrained(model_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    def reconstruct(self, video_frame_list: List[PILImage.Image], init_extrinsics: np.ndarray, intrinsics: np.ndarray = None, cam_pose_list: np.ndarray = None, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
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
                extrs = extrinsic.squeeze().cpu().numpy()
            else:
                extrs = np.stack(cam_pose_list)
            if intrinsics is None:
                intrinsic = predictions["intrs"]
                intrs = intrinsic.squeeze().cpu().numpy()
            else:
                intrs = intrinsics.copy()
                intrs = np.repeat(intrs[None, :, :], len(video_frame_list), axis=0)
            video_tensor = video_tensor.squeeze()
        cam2init = init_extrinsics @ np.linalg.inv(extrs[0])
        point_map_list = []
        for frame_idx in range(len(depth_tensor)):
            depth = depth_tensor[frame_idx]
            points_map = depth2xyz(depth, intrs[frame_idx], cam_type="opencv")
            cam_pose = cam2init @ extrs[frame_idx]
            ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
            points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
        return {"rgb": video_frame_list, 
                "intrinsics": intrs[0], 
                "extrinsics": extrs, 
                "depth": depth_tensor, 
                "points": np.stack(point_map_list), 
                "points_mask": unc_metric}
    

class ViPEReconstruction(BaseReconstruction):
    def reconstruct(self, video_dir: str, init_extrinsics: np.ndarray, sample_indices: List[int]) -> Dict[str, np.ndarray]:
        tmp_dir = "./tmp_vipe_reconstruction"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        try:
            os.system(f"vipe infer --image-dir {video_dir} --output {tmp_dir} --pipeline dav3")
        except Exception as e:
            print(f"Error during vipe inference: {e}")
            return None
        
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
        cam2init = None
        for frame_id, depth in enumerate(depth_frame_list):
            points_map = depth2xyz(depth, intrinsics, cam_type="opencv")
            cam_pose = pose_tensor[frame_id].matrix().numpy()
            if cam2init is None:
                cam2init = init_extrinsics @ np.linalg.inv(cam_pose)
            cam_pose = cam2init @ cam_pose
            points_map_homogeneous = np.concatenate([points_map, np.ones((points_map.shape[0], points_map.shape[1], 1))], axis=-1)
            points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
            point_map_list.append(points_map)
            cam_pose_list.append(cam_pose)
        sample_cam_pose_list = [cam_pose_list[i] for i in sample_indices]
        sample_depth_frame_list = [depth_frame_list[i] for i in sample_indices]
        sample_point_map_list = [point_map_list[i] for i in sample_indices]
        sample_depth_mask_list = [depth_mask_list[i] for i in sample_indices]
        return {"intrinsics": intrinsics, 
                "extrinsics": np.stack(sample_cam_pose_list), 
                "depth": np.stack(sample_depth_frame_list), 
                "points": np.stack(sample_point_map_list), 
                "points_mask": np.stack(sample_depth_mask_list)}
    

class DA3DReconstruction(BaseReconstruction):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = DepthAnything3.from_pretrained(model_path)
        self.model = self.model.to(device=device)
        self.device = device

    def _resize_ixt(
            self,
            intrinsic: np.ndarray | None,
            orig_w: int,
            orig_h: int,
            w: int,
            h: int,
        ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        # scale fx, cx by w ratio; fy, cy by h ratio
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K
    
    def get_conf_thresh(
        self,
        prediction: Prediction,
        sky_mask: np.ndarray,
        conf_thresh: float = 1.05,
        conf_thresh_percentile: float = 40.0,
        ensure_thresh_percentile: float = 90.0,
    ):
        if sky_mask is not None and (~sky_mask).sum() > 10:
            conf_pixels = prediction.conf[~sky_mask]
        else:
            conf_pixels = prediction.conf
        lower = np.percentile(conf_pixels, conf_thresh_percentile)
        upper = np.percentile(conf_pixels, ensure_thresh_percentile)
        conf_thresh = min(max(conf_thresh, lower), upper)
        return conf_thresh
    
    def reconstruct(self, video_frame_list: List[PILImage.Image], init_extrinsics: np.ndarray, intrinsics: np.ndarray = None, cam_pose_list: np.ndarray = None, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if intrinsics is not None and cam_pose_list is not None and depth_frame_list is not None:
            naive_recon = NaiveReconstruction()
            return naive_recon.reconstruct(video_frame_list, init_extrinsics, intrinsics, cam_pose_list, depth_frame_list)
        else:
            inv_cam_pose_list = [np.linalg.inv(cam_pose) for cam_pose in cam_pose_list] if cam_pose_list is not None else None
            inv_extrinsics = np.stack(inv_cam_pose_list) if inv_cam_pose_list is not None else None
            intrinsics_list = np.repeat(intrinsics[None, :, :], len(video_frame_list), axis=0) if intrinsics is not None else None
            outputs = self.model.inference(video_frame_list, intrinsics=intrinsics_list, extrinsics=inv_extrinsics)
            depth = outputs.depth # N, H, W
            depth_conf = outputs.conf # N, H, W depth confidence map
            original_height, original_width = video_frame_list[0].height, video_frame_list[0].width
            new_height, new_width = depth.shape[1], depth.shape[2]
            zoom_factors = (original_height / new_height, original_width / new_width)
            if inv_extrinsics is None:
                inv_extrinsics = outputs.extrinsics
                inv_extrinsics_4x4 = []
                for i in range(len(video_frame_list)):
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :3] = inv_extrinsics[i][:3, :3]
                    pose_4x4[:3, 3] = inv_extrinsics[i][:3, 3]
                    inv_extrinsics_4x4.append(pose_4x4)
                inv_extrinsics = np.stack(inv_extrinsics_4x4)
            if intrinsics is None:
                intrinsics = self._resize_ixt(outputs.intrinsics[0], new_width, new_height, original_width, original_height)
            point_map_list = []
            extrinsics = []
            cam2init = init_extrinsics @ inv_extrinsics[0]
            depth_conf_list = []
            depth_frame_list = []
            conf_thresh = self.get_conf_thresh(outputs, getattr(outputs, "sky_mask", None),)
            for frame_idx in range(len(video_frame_list)):
                depth_frame = zoom(depth[frame_idx], zoom_factors)
                depth_conf_frame = zoom(depth_conf[frame_idx], zoom_factors)
                depth_conf_list.append(depth_conf_frame)
                depth_frame_list.append(depth_frame)
                points_map = depth2xyz(depth_frame, intrinsics, cam_type="opencv")
                cam_pose = cam2init @ np.linalg.inv(inv_extrinsics[frame_idx])
                extrinsics.append(cam_pose)
                ones = np.ones((points_map.shape[0], points_map.shape[1], 1))
                points_map_homogeneous = np.concatenate([points_map, ones], axis=-1)
                points_map = (cam_pose @ points_map_homogeneous.reshape(-1, 4).T).T[:, :3].reshape(points_map.shape)
                point_map_list.append(points_map)
            return {"rgb": video_frame_list, 
                    "intrinsics": intrinsics[0], 
                    "extrinsics": np.stack(extrinsics), 
                    "depth": np.stack(depth_frame_list), 
                    "points": np.stack(point_map_list), 
                    "points_mask": np.stack(depth_conf_list) > conf_thresh}


class MapAnythingReconstruction(BaseReconstruction):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = MapAnything.from_pretrained(model_path)
        self.model = self.model.to(device=device)
        self.device = device

    def _resize_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        # scale fx, cx by w ratio; fy, cy by h ratio
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K

    def reconstruct(self, video_frame_list: List[PILImage.Image], init_extrinsics: np.ndarray, intrinsics: np.ndarray = None, cam_pose_list: np.ndarray = None, depth_frame_list: List[np.ndarray] = None) -> Dict[str, np.ndarray]:
        input_views = []
        # origin_extrinsics_list = []
        original_height, original_width = video_frame_list[0].height, video_frame_list[0].width
        for frame_id in range(len(video_frame_list)):
            input_view = {
                "img": video_frame_list[frame_id],
            }
            if intrinsics is not None:
                input_view["intrinsics"] = intrinsics.astype(np.float32)
            if cam_pose_list is not None:
                input_view["camera_poses"] = cam_pose_list[frame_id].astype(np.float32)
                input_view["is_metric_scale"] = torch.tensor([True], device=self.device)
            input_views.append(input_view)
            # origin_extrinsics_list.append(extrinsic)
        processed_views = preprocess_inputs(input_views, resize_mode="square", size=700)
        # Run inference with any combination of inputs
        outputs = self.model.infer(
            processed_views,                  # Any combination of input views
            memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
            use_amp=True,                     # Use mixed precision inference (recommended)
            amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
            apply_mask=True,                  # Apply masking to dense geometry outputs
            mask_edges=True,                  # Remove edge artifacts by using normals and depth
            apply_confidence_mask=False,      # Filter low-confidence regions
            confidence_percentile=5,         # Remove bottom 5 percentile confidence pixels
            # Control which inputs to use/ignore
            # By default, all inputs are used when provided
            # If is_metric_scale flag is not provided, all inputs are assumed to be in metric scale
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=False,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=False,
        )

        pred_intrinsics_list = []
        pred_extrinsics_list = []
        depth_list = []
        point_map_list = []
        point_mask_list = []
        cam2init = None
        # Access results for each view - Complete list of metric outputs
        for i, pred in enumerate(outputs):
            # Geometry outputs
            pts3d = pred["pts3d"].cpu().numpy()                     # 3D points in world coordinates (B, H, W, 3)
            pts3d_h, pts3d_w = pts3d.shape[1], pts3d.shape[2]
            zoom_factors = (original_height / pts3d_h, original_width / pts3d_w)
            # pts3d_cam = pred["pts3d_cam"]             # 3D points in camera coordinates (B, H, W, 3)
            depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)
            depth_origin_size = zoom(depth_z[0, :, :, 0].cpu().numpy(), zoom_factors)
            depth_list.append(depth_origin_size)
            # depth_along_ray = pred["depth_along_ray"] # Depth along ray in camera frame (B, H, W, 1)

            # Camera outputs
            # ray_directions = pred["ray_directions"]   # Ray directions in camera frame (B, H, W, 3)
            pred_intrinsics = pred["intrinsics"]           # Recovered pinhole camera intrinsics (B, 3, 3)
            pred_intrinsics_origin_size = self._resize_ixt(pred_intrinsics[0].cpu().numpy(), pts3d_w, pts3d_h, original_width, original_height)
            pred_intrinsics_list.append(pred_intrinsics_origin_size)
            pred_camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
            if cam2init is None:
                cam2init = init_extrinsics @ pred_camera_poses[0].cpu().numpy()
            pred_extrinsics = cam2init @ np.linalg.inv(pred_camera_poses[0].cpu().numpy())
            pred_extrinsics_list.append(pred_extrinsics)
            # print("pts3d shape:", pts3d.shape)
            pts3d_origin_size = depth2xyz(depth_origin_size, pred_intrinsics_origin_size if intrinsics is None else intrinsics, cam_type="opencv")
            pts3d_homo = (cam2init @ np.concatenate([pts3d_origin_size.reshape(-1, 3), np.ones((original_height * original_width, 1))], axis=-1).T).T
            print("pts3d_homo shape:", pts3d_homo.shape)
            pts3d = pts3d_homo[:, :3].reshape(original_height, original_width, 3)
            point_map_list.append(pts3d)
            # cam_trans = pred["cam_trans"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
            # cam_quats = pred["cam_quats"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

            # Quality and masking
            confidence = pred["conf"]                 # Per-pixel confidence scores (B, H, W)
            mask = pred["mask"]                       # Combined validity mask (B, H, W, 1)
            non_ambiguous_mask = pred["non_ambiguous_mask"]                # Non-ambiguous regions (B, H, W)
            non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # Mask logits (B, H, W)
            point_mask_list.append(mask[0, :, :, 0].cpu().numpy())
            # Scaling
            metric_scaling_factor = pred["metric_scaling_factor"]  # Applied metric scaling (B,)

            # Original input
            img_no_norm = pred["img_no_norm"]         # Denormalized input images for visualization (B, H, W, 3)
        return {"rgb": video_frame_list, 
                "intrinsics": pred_intrinsics_list[0] if intrinsics is None else intrinsics, 
                "extrinsics": np.stack(pred_extrinsics_list) if cam_pose_list is None else cam_pose_list, 
                "depth": np.stack(depth_list), 
                "points": np.stack(point_map_list), 
                "points_mask": np.stack(point_mask_list)}


def build_reconstruction_model(input_modality: str, recon_method: str, model_path: str = None, device: str = "cuda") -> BaseReconstruction:
    if recon_method == "naive":
        assert input_modality == "rgb+extrinsics+intrinsics+depth", "Naive reconstruction requires depth input."
        recon_model = NaiveReconstruction()
    elif recon_method == "moge":
        assert input_modality in ["rgb+extrinsics+intrinsics", "rgb+extrinsics+intrinsics+depth"], "MoGe reconstruction requires at least rgb, intrinsics and extrinsics input."
        recon_model = MoGeReconstruction(model_path=model_path, device=device)
    elif recon_method == "spatracker":
        recon_model = SpatrackerReconstruction(model_path=model_path, device=device)
    elif recon_method == "vipe":
        recon_model = ViPEReconstruction()
    elif recon_method == "da3":
        recon_model = DA3DReconstruction(model_path=model_path, device=device)
    elif recon_method == "mapanything":
        recon_model = MapAnythingReconstruction(model_path=model_path, device=device)
    else:
        raise NotImplementedError(f"Reconstruction method {recon_method} not implemented.")
    return recon_model