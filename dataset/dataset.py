import json
import copy
from contextlib import contextmanager
import time
import re
import h5py
import numpy as np
import os
from pathlib import Path
import tempfile
import omegaconf
import point_cloud_utils as pcu
import open3d as o3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio

from typing import Tuple, List, Dict, Any, Mapping, Optional, Sequence


def _load_hdf5_mask_groups(filepath: str) -> dict:
    """Read normalized or legacy H5 groups using the existing in-memory keys.

    Keep group names as keys (including numbered additional parts) and expose
    the original annotation label separately through ``name``.
    """
    data = {}
    with h5py.File(filepath, "r") as file:
        for group_name, group in file.items():
            if not isinstance(group, h5py.Group):
                continue
            mask_id = group.attrs.get("id", group.attrs.get("mask_idx"))
            if mask_id is None:
                raise KeyError(f"Mask group '{group_name}' in {filepath} has no id.")
            key = "mask" if "mask" in group else "masks" if "masks" in group else None
            if key is None:
                raise KeyError(
                    f"Mask group '{group_name}' in {filepath} has no 'mask' or 'masks' dataset."
                )
            name = group.attrs.get("name", group_name)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            data[group_name] = {
                "mask_idx": int(mask_id),
                "masks": group[key][:],
                "name": str(name),
            }
    return data


@contextmanager
def temporary_video_from_frames(
    rgb_list: Sequence[np.ndarray],
    source_video_path: Optional[str] = None,
    sample_indices: Optional[Sequence[int]] = None,
):
    """Write sampled frames to a temporary MP4 in the repository root.

    This is used when a consumer such as ViPE needs a file path but the
    in-memory frames have been cropped. The temporary video contains exactly
    the frames in ``rgb_list`` and is removed when the context exits.
    """
    if len(rgb_list) == 0:
        raise ValueError("Cannot create a temporary video from an empty frame list.")

    frames = np.stack([np.asarray(frame) for frame in rgb_list])
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected video frames shaped (T, H, W, 3), got {frames.shape}."
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    fps = 30.0
    if source_video_path is not None:
        try:
            metadata = imageio.v3.immeta(source_video_path)
            fps = float(metadata.get("fps", fps))
        except (OSError, TypeError, ValueError):
            pass
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    if sample_indices is not None and len(sample_indices) > 1:
        frame_steps = np.diff(np.asarray(sample_indices, dtype=float))
        positive_steps = frame_steps[frame_steps > 0]
        if len(positive_steps) > 0:
            fps /= float(np.median(positive_steps))

    repository_root = Path(__file__).resolve().parent.parent
    file_descriptor, temp_video_path = tempfile.mkstemp(
        prefix="tmp_vipe_", suffix=".mp4", dir=repository_root
    )
    os.close(file_descriptor)
    try:
        imageio.mimwrite(
            temp_video_path,
            frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv444p",
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        yield temp_video_path
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


class UniformDataset(Dataset):
    def __init__(self, root_path: str, meta_file_path: str, image_type: str = "undistorted", sample_strategy: str = "fix_size", sample_num = 20):
        self.root_path = root_path
        with open(meta_file_path, "r") as f:
            self.meta_info = json.load(f)
        self.image_type = image_type
        self.sample_strategy = sample_strategy
        self.sample_num = sample_num

    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_dict = self.meta_info[idx]
        video_name = video_dict["video_name"]
        print(f"Loading video: {video_name}")
        
        if video_dict["source"] == "egoexo4d" and self.image_type == "cropped":
            video_path = video_dict["cropped_video_path"]
            crop = True
        else:
            video_path = video_dict["original_video_path"]
            crop = False
        # rgb
        # rgb_list, rgb_path_list = self.load_video(video_path)
        rgb_list, full_video_path = self.load_video(video_path)
        full_num_frames = len(rgb_list)
        sample_indices = self.get_sample_indices(full_num_frames)
        rgb_list = [rgb_list[i] for i in sample_indices]
        # rgb_path_list = [rgb_path_list[i] for i in sample_indices]

        camera_extrinsics, camera_intrinsics, cropped_top_left, cropped_bottom_right = self.load_camera(
            video_dict["camera_extrinsics_path"],
            video_dict["camera_intrinsics_path"],
            crop
        )
        camera_extrinsics = camera_extrinsics[sample_indices]

        receptor_mask_list, effector_mask_list, object_mask_list, receptor_name, effector_name, object_name = self.load_2d_masks(
            video_dict["video_mask_path"]
        )
        receptor_mask_list = receptor_mask_list[sample_indices, cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]]
        effector_mask_list = effector_mask_list[sample_indices, cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]]
        object_mask_list = object_mask_list[sample_indices, cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]]

        geometry_data = self.load_part_point_cloud(
            video_dict["geometry_type"],
            video_dict["geometry_path"],
            video_dict["part_annotation_path"],
            video_dict["function_instance_id"]
        )

        receptor_articulation, effector_articulation = self.load_articulation(
            video_dict["articulation_path"],
            geometry_data
        )

        function_annotation = self.load_function_annotation(
            video_dict["function_annotation_path"],
            video_dict["function_instance_id"]
        )

        data_dict = {
            "video_name": video_name,
            "video_path": full_video_path,
            "video_mask_path": os.path.join(self.root_path, video_dict["video_mask_path"]),
            "rgb_list": rgb_list,
            # "rgb_path_list": rgb_path_list,
            "camera_extrinsics": camera_extrinsics,
            "camera_intrinsics": camera_intrinsics,
            "receptor_mask_list": receptor_mask_list,
            "effector_mask_list": effector_mask_list,
            "object_mask_list": object_mask_list,
            "cropped_top_left": cropped_top_left,
            "cropped_bottom_right": cropped_bottom_right,
            "receptor_name": receptor_name,
            "effector_name": effector_name,
            "object_name": object_name,
            "geometry_data": geometry_data,
            "receptor_articulation": receptor_articulation,
            "effector_articulation": effector_articulation,
            "function_annotation": function_annotation,
            "initial_state": video_dict.get("initial_state", "close"),
            "sample_indices": sample_indices,
            "num_total_frames": int(full_num_frames),
            "crop": crop,
        }
        return data_dict
    
    def get_sample_indices(self, total_frames: int) -> List[int]:
        if self.sample_strategy == "fix_size":
            if total_frames > self.sample_num:
                sample_indices = np.linspace(0, total_frames-1, self.sample_num, dtype=int)
            else:
                sample_indices = list(range(total_frames))
        elif self.sample_strategy == "fix_step":
            sample_indices = list(range(0, total_frames, self.sample_num))
        else:
            sample_indices = list(range(total_frames))
        return sample_indices

    def load_video(self, video_path: str) -> Tuple[np.ndarray, str]:
        # video_path = os.path.join(self.root_path, video_dict["video_path"])
        full_video_path = os.path.join(self.root_path, video_path)
        rgb_list = imageio.v3.imread(full_video_path)  # (T, H, W, 3)
        return rgb_list, full_video_path
        # rgb_path_list = glob.glob(os.path.join(video_frame_dir, "*.jpg"))
        # rgb_path_list.sort()
        # rgb_list = []
        # for frame_path in rgb_path_list:
        #     image = PILImage.open(frame_path)
        #     image = image.convert("RGB")
        #     rgb_list.append(image)
        # return rgb_list, rgb_path_list
    
    def load_camera(self, extrinsics_path: str, intrinsics_path: str, crop: bool) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        camera_extrinsics_path = os.path.join(self.root_path, extrinsics_path)
        camera_extrinsics = np.load(camera_extrinsics_path)  # (N, 4, 4)
        camera_intrinsics_path = os.path.join(self.root_path, intrinsics_path)
        with open(camera_intrinsics_path, "r") as f:
            camera_intrinsics_data = json.load(f)
        if crop and "cropped_intrinsics" in camera_intrinsics_data.keys():
            camera_intrinsics = np.array(camera_intrinsics_data["cropped_intrinsics"])
            cropped_top_left = camera_intrinsics_data["cropped_top_left"]
            cropped_bottom_right = camera_intrinsics_data["cropped_bottom_right"]
        else:
            camera_intrinsics = np.array(camera_intrinsics_data["undistorted_intrinsics"])
            cropped_top_left = [0, 0]
            cropped_bottom_right = camera_intrinsics_data["original_frame_size"]
        return camera_extrinsics, camera_intrinsics, cropped_top_left, cropped_bottom_right
    
    def load_from_hdf5(self, filepath: str) -> dict:
        """Load all mask groups, accepting both normalized and legacy files."""
        return _load_hdf5_mask_groups(filepath)
    
    def load_2d_masks(self, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        # 2d masks
        mask_path = os.path.join(self.root_path, mask_path)
        mask_data = self.load_from_hdf5(mask_path)
        receptor_mask = None
        effector_mask = None
        object_mask = None
        receptor_name = None
        effector_name = None
        object_name = None
        for mask_name in mask_data.keys():
            if mask_data[mask_name]["mask_idx"] == 3:
                receptor_mask = mask_data[mask_name]["masks"]
                receptor_name = mask_data[mask_name]["name"]
            elif mask_data[mask_name]["mask_idx"] == 4:
                effector_mask = mask_data[mask_name]["masks"]
                effector_name = mask_data[mask_name]["name"]
            elif mask_data[mask_name]["mask_idx"] == 5:
                object_mask = mask_data[mask_name]["masks"]
                object_name = mask_data[mask_name]["name"]
        if object_mask is None:
            object_mask = np.logical_or(receptor_mask, effector_mask)
            object_name = f"{receptor_name} and {effector_name}"
        return receptor_mask, effector_mask, object_mask, receptor_name, effector_name, object_name

    def load_part_point_cloud(self, geometry_type: str, geometry_path: str, part_annotation_path: str, function_instance_id: int) -> dict:
        if geometry_type == "point cloud":
            return self.load_point_cloud_data(geometry_path, part_annotation_path, function_instance_id)
        elif geometry_type == "mesh":
            return self.load_mesh_data(geometry_path, part_annotation_path, function_instance_id)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")

    def load_point_cloud_data(self, geometry_path: str, part_annotation_path: str, function_instance_id: int) -> dict:
        full_pcd = pcu.load_mesh_v(os.path.join(self.root_path, geometry_path), np.float32)  # (N, 3)
        with open(os.path.join(self.root_path, part_annotation_path), "r") as f:
            annotations = json.load(f)
        annotations_dict = {item["function_instance_id"]: item for item in annotations}
        if function_instance_id not in annotations_dict.keys():
            raise ValueError(f"Segment ID {function_instance_id} not found in annotations.")
        geometry_annotations = {}
        for role in ["receptor", "effector"]:
            # print(annotations_dict[function_instance_id].keys())
            part_name = annotations_dict[function_instance_id][role]["label"]
            part_indices = annotations_dict[function_instance_id][role]["indices"]
            pid = annotations_dict[function_instance_id][role]["pid"]
            if not part_indices:
                raise ValueError(f"No indices found for role {role} in segment {function_instance_id}.")
            part_pcd = full_pcd[part_indices]
            geometry_annotations[role] = {
                "part_pcd": part_pcd,
                "pid": pid
            }
        geometry_annotations["relation"] = annotations_dict[function_instance_id]["description"]
        return geometry_annotations

    def load_mesh_data(self, geometry_path: str, part_annotation_path: str, function_instance_id: int) -> dict:
        full_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_path, geometry_path))
        with open(os.path.join(self.root_path, part_annotation_path), "r") as f:
            annotations = json.load(f)
        annotations_dict = {item["function_instance_id"]: item for item in annotations}
        if function_instance_id not in annotations_dict.keys():
            raise ValueError(f"Segment ID {function_instance_id} not found in annotations.")
        geometry_annotations = {}

        for role in ["receptor", "effector"]:
            # print(annotations_dict[function_instance_id].keys())
            part_name = annotations_dict[function_instance_id][role]["label"]
            part_indices = annotations_dict[function_instance_id][role]["indices"]
            pid = annotations_dict[function_instance_id][role]["pid"]
            if not part_indices:
                raise ValueError(f"No indices found for role {role} in segment {function_instance_id}.")
            part_mesh = full_mesh.select_by_index(part_indices)
            part_pcd = part_mesh.sample_points_uniformly(number_of_points=10000)
            part_pcd_np = np.asarray(part_pcd.points)
            geometry_annotations[role] = {
                "part_pcd": part_pcd_np,
                "part_mesh": {
                    "vertices": np.asarray(part_mesh.vertices).copy(),
                    "triangles": np.asarray(part_mesh.triangles).copy(),
                },
                "pid": pid
            }
        geometry_annotations["relation"] = annotations_dict[function_instance_id]["description"]
        return geometry_annotations
    
    def load_articulation(self, articulation_path: str, geometry_data: dict) -> Tuple[dict, dict]:
        receptor_articulation = None
        effector_articulation = None
        articulation_path = os.path.join(self.root_path, articulation_path)
        if os.path.exists(articulation_path):
            with open(articulation_path, "r") as f:
                articulation_data = json.load(f)
            for role in ["receptor", "effector"]:
                pid = geometry_data[role]["pid"]
                for joint_data in articulation_data:
                    if joint_data["pid"] == pid:
                        if role == "receptor":
                            receptor_articulation = joint_data
                        elif role == "effector":
                            effector_articulation = joint_data
        return receptor_articulation, effector_articulation
    
    def load_function_annotation(self, function_annotation_path: str, function_instance_id: str) -> dict:
        function_annotation_path = os.path.join(self.root_path, function_annotation_path)
        with open(function_annotation_path, "r") as f:
            function_data = json.load(f)
        function_data_dict = {item["function_instance_id"]: item for item in function_data}
        function_annotation = function_data_dict[function_instance_id]
        return function_annotation


class NewDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        meta_file_path: str,
        image_type: str = "undistorted",
        sample_strategy: str = "fix_size",
        sample_num: int = 20,
        load_2d_masks: bool = True,
        load_mesh_data: bool = True,
        load_articulation: bool = True,
        load_function_annotation: bool = True,
    ):
        self.root_path = root_path
        with open(meta_file_path, "r") as f:
            self.meta_info = json.load(f)
        self.image_type = image_type
        self.sample_strategy = sample_strategy
        self.sample_num = sample_num
        self.load_2d_masks_enabled = load_2d_masks
        self.load_mesh_data_enabled = load_mesh_data
        self.load_articulation_enabled = load_articulation
        self.load_function_annotation_enabled = load_function_annotation

    def __len__(self):
        return len(self.meta_info)

    def _select_video_path(
        self, video_dict: Mapping[str, Any]
    ) -> Tuple[str, bool]:
        """Return the source video path and whether it is already cropped."""
        video_path = video_dict.get("video_path", video_dict.get("video"))
        if video_path is None:
            raise KeyError("Metadata entry is missing 'video'.")
        return str(video_path), False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_dict = self.meta_info[idx]
        video_path, use_cropped_video = self._select_video_path(video_dict)
        video_name = video_dict.get(
            "video_name", os.path.basename(video_path).split(".")[0]
        )
        print(f"Loading video: {video_name}")

        # 2D data
        rgb_list, full_video_path = self.load_video(video_path)
        full_num_frames = len(rgb_list)
        sample_indices = self.get_sample_indices(full_num_frames)
        rgb_list = [rgb_list[i].copy() for i in sample_indices]

        (
            camera_extrinsics,
            camera_intrinsics,
            stored_cropped_intrinsics,
            stored_cropped_top_left,
            stored_cropped_bottom_right,
        ) = self.load_camera_details(
            video_dict.get("camera_path", video_dict.get("camera"))
        )
        camera_extrinsics = camera_extrinsics[sample_indices]

        receptor_mask_list = None
        effector_mask_list = None
        object_mask_list = None
        receptor_name = None
        effector_name = None
        object_name = None
        has_object = True
        if self.load_2d_masks_enabled:
            (
                receptor_mask_list,
                effector_mask_list,
                object_mask_list,
                receptor_name,
                effector_name,
                object_name,
                has_object,
            ) = self.load_2d_masks(
                video_dict.get("video_mask_path", video_dict.get("video_mask"))
            )
            receptor_mask_list = receptor_mask_list[sample_indices]
            effector_mask_list = effector_mask_list[sample_indices]
            object_mask_list = object_mask_list[sample_indices]

        if self.image_type == "cropped":
            if use_cropped_video and stored_cropped_top_left is None:
                raise ValueError(
                    f"Metadata selects pre-cropped video {video_path}, but its "
                    "camera file has no saved crop calibration."
                )
            if stored_cropped_top_left is not None:
                cropped_top_left = stored_cropped_top_left
                cropped_bottom_right = stored_cropped_bottom_right
                camera_intrinsics = stored_cropped_intrinsics
            else:
                cropped_top_left, cropped_bottom_right = self.compute_crop_coordinates(
                    rgb_list[0]
                )
                camera_intrinsics = self.compute_cropped_intrinsics(
                    camera_intrinsics, cropped_top_left
                )
            if not use_cropped_video:
                rgb_list = [
                    frame[
                        cropped_top_left[1]:cropped_bottom_right[1],
                        cropped_top_left[0]:cropped_bottom_right[0],
                    ]
                    for frame in rgb_list
                ]
            if self.load_2d_masks_enabled:
                receptor_mask_list = receptor_mask_list[
                    :,
                    cropped_top_left[1]:cropped_bottom_right[1],
                    cropped_top_left[0]:cropped_bottom_right[0],
                ]
                effector_mask_list = effector_mask_list[
                    :,
                    cropped_top_left[1]:cropped_bottom_right[1],
                    cropped_top_left[0]:cropped_bottom_right[0],
                ]
                object_mask_list = object_mask_list[
                    :,
                    cropped_top_left[1]:cropped_bottom_right[1],
                    cropped_top_left[0]:cropped_bottom_right[0],
                ]
        else:
            cropped_top_left = [0, 0]
            cropped_bottom_right = [rgb_list[0].shape[1], rgb_list[0].shape[0]]

        # 3D data
        geometry_path = video_dict.get("geometry_path", video_dict.get("mesh"))
        transformation_path = video_dict.get(
            "transformation_path", video_dict.get("object_transformation")
        )
        part_annotation_path = video_dict.get(
            "part_annotation_path", video_dict.get("object_mask")
        )
        function_instance_id = video_dict.get(
            "function_instance_id", video_dict.get("function_id")
        )
        articulation_path = video_dict.get(
            "articulation_path", video_dict.get("articulation")
        )
        geometry_data = None
        if self.load_mesh_data_enabled:
            geometry_data = self.load_geometry_data(
                geometry_path,
                video_dict.get("geometry_type", "mesh"),
                transformation_path,
                part_annotation_path,
                function_instance_id,
                has_object,
                articulation_path=(
                    articulation_path if self.load_articulation_enabled else None
                ),
                normalize_articulation_geometry=video_dict.get(
                    "normalize_articulation_geometry", True
                ),
            )

        receptor_articulation = None
        effector_articulation = None
        if self.load_articulation_enabled:
            receptor_articulation, effector_articulation = self.load_articulation(
                articulation_path,
                geometry_data,
                part_annotation_path=part_annotation_path,
                function_instance_id=function_instance_id,
                transformation_path=transformation_path,
                transform_to_world=video_dict.get(
                    "articulation_in_canonical_frame", True
                ),
            )

        function_annotation = None
        if self.load_function_annotation_enabled:
            function_annotation = self.load_function_annotation(
                video_dict.get("function_annotation_path", video_dict.get("function")),
                function_instance_id,
            )

        video_mask_path = video_dict.get(
            "video_mask_path", video_dict.get("video_mask")
        )

        data_dict = {
            "video_name": video_name,
            "video_path": full_video_path,
            "video_mask_path": (
                os.path.join(self.root_path, video_mask_path)
                if video_mask_path is not None else None
            ),
            "rgb_list": rgb_list,
            # "rgb_path_list": rgb_path_list,
            "camera_extrinsics": camera_extrinsics,
            "camera_intrinsics": camera_intrinsics,
            "receptor_mask_list": receptor_mask_list,
            "effector_mask_list": effector_mask_list,
            "object_mask_list": object_mask_list,
            "cropped_top_left": cropped_top_left,
            "cropped_bottom_right": cropped_bottom_right,
            "receptor_name": receptor_name,
            "effector_name": effector_name,
            "object_name": object_name,
            "geometry_data": geometry_data,
            "receptor_articulation": receptor_articulation,
            "effector_articulation": effector_articulation,
            "function_annotation": function_annotation,
            "initial_state": video_dict.get("initial_state", "close"),
            "sample_indices": sample_indices,
            "num_total_frames": int(full_num_frames),
            "crop": self.image_type == "cropped",
        }
        return data_dict

    def compute_crop_coordinates(self, rgb_img: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find an axis-aligned crop whose corners are inside the valid image.

        Aria frames have rounded black corners.  Each image corner is scanned
        diagonally toward the centre; the most restrictive hit on each side
        defines the crop.  The returned bottom-right coordinate is exclusive,
        matching NumPy slicing conventions.
        """
        rgb_img = np.asarray(rgb_img)
        if rgb_img.ndim == 2:
            non_black = rgb_img != 0
        elif rgb_img.ndim == 3:
            non_black = np.any(rgb_img != 0, axis=-1)
        else:
            raise ValueError(
                f"Expected an HxW or HxWxC image, got shape {rgb_img.shape}."
            )

        height, width = non_black.shape
        max_inset = min(height, width)

        def first_valid_from_corner(corner: str) -> Tuple[int, int]:
            for inset in range(max_inset):
                x = inset if corner in {"top_left", "bottom_left"} else width - 1 - inset
                y = inset if corner in {"top_left", "top_right"} else height - 1 - inset
                if non_black[y, x]:
                    return x, y
            raise ValueError("Cannot crop an image containing only black pixels.")

        top_left_hit = first_valid_from_corner("top_left")
        top_right_hit = first_valid_from_corner("top_right")
        bottom_left_hit = first_valid_from_corner("bottom_left")
        bottom_right_hit = first_valid_from_corner("bottom_right")

        cropped_top_left = [
            max(top_left_hit[0], bottom_left_hit[0]),
            max(top_left_hit[1], top_right_hit[1]),
        ]
        cropped_bottom_right = [
            min(top_right_hit[0], bottom_right_hit[0]) + 1,
            min(bottom_left_hit[1], bottom_right_hit[1]) + 1,
        ]
        if (cropped_top_left[0] >= cropped_bottom_right[0]
                or cropped_top_left[1] >= cropped_bottom_right[1]):
            raise ValueError("The non-black image region does not form a valid crop.")
        return cropped_top_left, cropped_bottom_right

    def compute_cropped_intrinsics(self, original_intrinsics: np.ndarray, cropped_top_left: List[int]) -> np.ndarray:
        original_intrinsics = np.asarray(original_intrinsics)
        if original_intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected a 3x3 camera intrinsic matrix, got {original_intrinsics.shape}."
            )
        if len(cropped_top_left) != 2:
            raise ValueError("cropped_top_left must contain [x, y].")

        cropped_intrinsics = original_intrinsics.copy()
        cropped_intrinsics[0, 2] -= cropped_top_left[0]
        cropped_intrinsics[1, 2] -= cropped_top_left[1]
        return cropped_intrinsics

    def crop_data(self, original_rgb_list: List[np.ndarray], original_mask_list: np.ndarray, original_camera_intrinsics: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[int], List[int]]:
        if not original_rgb_list:
            raise ValueError("Cannot crop an empty video.")
        cropped_top_left, cropped_bottom_right = self.compute_crop_coordinates(original_rgb_list[0])
        cropped_intrinsics = self.compute_cropped_intrinsics(original_camera_intrinsics, cropped_top_left)
        cropped_rgb_list = [frame[cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]] for frame in original_rgb_list]
        cropped_mask_list = original_mask_list[
            ..., cropped_top_left[1]:cropped_bottom_right[1],
            cropped_top_left[0]:cropped_bottom_right[0]
        ]
        return cropped_rgb_list, cropped_mask_list, cropped_intrinsics, cropped_top_left, cropped_bottom_right

    def get_sample_indices(self, total_frames: int) -> List[int]:
        if self.sample_strategy == "fix_size":
            if total_frames > self.sample_num:
                sample_indices = np.linspace(0, total_frames-1, self.sample_num, dtype=int)
            else:
                sample_indices = list(range(total_frames))
        elif self.sample_strategy == "fix_step":
            sample_indices = list(range(0, total_frames, self.sample_num))
        else:
            sample_indices = list(range(total_frames))
        return sample_indices

    def load_video(self, video_path: str) -> Tuple[np.ndarray, str]:
        full_video_path = os.path.join(self.root_path, video_path)
        rgb_list = imageio.v3.imread(full_video_path)  # (T, H, W, 3)
        return rgb_list, full_video_path

    def load_camera_details(
        self, camera_path: str
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[int]],
    ]:
        if camera_path is None:
            raise KeyError("Metadata entry is missing 'camera'.")
        full_camera_path = os.path.join(self.root_path, camera_path)
        with h5py.File(full_camera_path, "r") as camera_file:
            if "T_world_camera" not in camera_file:
                raise KeyError(f"{full_camera_path} does not contain 'T_world_camera'.")
            if "K" not in camera_file:
                raise KeyError(f"{full_camera_path} does not contain 'K'.")
            camera_extrinsics = camera_file["T_world_camera"][:]
            camera_intrinsics = camera_file["K"][:]
            cropped_camera_intrinsics = (
                camera_file["K_cropped"][:] if "K_cropped" in camera_file else None
            )
            cropped_top_left = (
                camera_file["cropped_top_left"][:].astype(int).tolist()
                if "cropped_top_left" in camera_file else None
            )
            cropped_bottom_right = (
                camera_file["cropped_bottom_right"][:].astype(int).tolist()
                if "cropped_bottom_right" in camera_file else None
            )

        if camera_extrinsics.ndim != 3 or camera_extrinsics.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected camera extrinsics shaped (N, 4, 4), got {camera_extrinsics.shape}."
            )
        if camera_intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected camera intrinsics shaped (3, 3), got {camera_intrinsics.shape}."
            )
        crop_values = (
            cropped_camera_intrinsics,
            cropped_top_left,
            cropped_bottom_right,
        )
        if any(value is None for value in crop_values) and not all(
            value is None for value in crop_values
        ):
            raise ValueError(
                f"{full_camera_path} must contain all or none of K_cropped, "
                "cropped_top_left, and cropped_bottom_right."
            )
        if cropped_camera_intrinsics is not None and cropped_camera_intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected cropped camera intrinsics shaped (3, 3), "
                f"got {cropped_camera_intrinsics.shape}."
            )
        return (
            camera_extrinsics,
            camera_intrinsics,
            cropped_camera_intrinsics,
            cropped_top_left,
            cropped_bottom_right,
        )

    def load_camera(self, camera_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load undistorted parameters while preserving the public two-value API."""
        camera_extrinsics, camera_intrinsics, _, _, _ = self.load_camera_details(
            camera_path
        )
        return camera_extrinsics, camera_intrinsics

    def load_from_hdf5(self, filepath: str) -> dict:
        """Load all mask groups, accepting both normalized and legacy files."""
        return _load_hdf5_mask_groups(filepath)

    def load_2d_masks(self, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str, bool]:
        if mask_path is None:
            raise KeyError("Metadata entry is missing 'video_mask'.")
        full_mask_path = os.path.join(self.root_path, mask_path)
        role_by_id = {3: "receptor", 4: "effector", 5: "object"}
        masks = {}
        names = {}

        with h5py.File(full_mask_path, "r") as mask_file:
            for group_name, group in mask_file.items():
                if not isinstance(group, h5py.Group):
                    continue
                mask_id = group.attrs.get("id", group.attrs.get("mask_idx"))
                role = role_by_id.get(int(mask_id)) if mask_id is not None else None
                normalized_group_name = group_name.lower()
                if role is None and normalized_group_name in role_by_id.values():
                    role = normalized_group_name
                if role is None:
                    continue

                dataset_name = "mask" if "mask" in group else "masks" if "masks" in group else None
                if dataset_name is None:
                    raise KeyError(
                        f"Mask group '{group_name}' in {full_mask_path} has no 'mask' or 'masks' dataset."
                    )
                masks[role] = group[dataset_name][:].astype(bool, copy=False)
                role_name = group.attrs.get("name", group_name)
                if isinstance(role_name, bytes):
                    role_name = role_name.decode("utf-8")
                names[role] = str(role_name)

        missing_roles = [role for role in ("receptor", "effector") if role not in masks]
        if missing_roles:
            raise ValueError(
                f"Missing required mask role(s) {missing_roles} in {full_mask_path}."
            )

        receptor_mask = masks["receptor"]
        effector_mask = masks["effector"]
        if receptor_mask.shape != effector_mask.shape:
            raise ValueError("Receptor and effector masks must have the same shape.")

        has_object = "object" in masks
        if has_object:
            object_mask = masks["object"]
            if object_mask.shape != receptor_mask.shape:
                raise ValueError("Object and part masks must have the same shape.")
            object_name = names["object"]
        else:
            object_mask = np.logical_or(receptor_mask, effector_mask)
            object_name = f"{names['receptor']} and {names['effector']}"

        receptor_name = names["receptor"]
        effector_name = names["effector"]
        return receptor_mask, effector_mask, object_mask, receptor_name, effector_name, object_name, has_object

    def load_role_part_annotations(
        self, part_annotation_path: str, function_instance_id: int
    ) -> Tuple[Any, Dict[str, List[dict]]]:
        """Load role records, including annotations shared by multiple roles.

        Annotation labels begin with one or more role-instance tokens. For
        example, ``r1_handle`` belongs to receptor 1, while
        ``r1+r2_shared_handle`` belongs to receptors 1 and 2 and
        ``r1+e1_shared_part`` belongs to both roles of function instance 1.
        A role can therefore match multiple annotations. A missing receptor is
        invalid; a missing effector is represented by an empty list and is
        later interpreted as the whole object mesh.
        """
        if part_annotation_path is None:
            raise KeyError("Metadata entry is missing the part annotation path.")
        if function_instance_id is None:
            raise KeyError("Metadata entry is missing 'function_id'.")

        with open(os.path.join(self.root_path, part_annotation_path), "r") as f:
            annotation_data = json.load(f)
        if isinstance(annotation_data, dict):
            annotations = annotation_data.get("data", {}).get(
                "annotations", annotation_data.get("annotations", [])
            )
        else:
            annotations = annotation_data
        if not isinstance(annotations, list):
            raise ValueError(f"Invalid part annotations in {part_annotation_path}.")

        instance_id = str(function_instance_id)
        if isinstance(annotation_data, list) and annotations and all(
            "function_instance_id" in annotation for annotation in annotations
        ):
            matching_records = [
                annotation for annotation in annotations
                if str(annotation["function_instance_id"]) == instance_id
            ]
            if len(matching_records) != 1:
                raise ValueError(
                    f"Expected one record for function {instance_id!r} in "
                    f"{part_annotation_path}, found {len(matching_records)}."
                )
            annotation_data = matching_records[0]
            return annotation_data, {
                "receptor": [annotation_data["receptor"]],
                "effector": (
                    [annotation_data["effector"]]
                    if annotation_data.get("effector") is not None else []
                ),
            }

        role_annotations = {}
        for role, role_prefix in (("receptor", "r"), ("effector", "e")):
            role_token = f"{role_prefix}{instance_id}"
            matches = [
                annotation for annotation in annotations
                if role_token in self.parse_part_role_tokens(
                    str(annotation.get("label", ""))
                )
            ]
            if not matches and role == "receptor":
                raise ValueError(
                    f"No receptor part assigned to '{role_token}' in "
                    f"{part_annotation_path}."
                )
            role_annotations[role] = matches
        return annotation_data, role_annotations

    @staticmethod
    def parse_part_role_tokens(part_label: str) -> List[str]:
        """Return leading role tokens such as ``r1`` and ``e2`` from a label."""
        prefix_match = re.match(r"^([re]\d+(?:\+[re]\d+)*)", part_label)
        return prefix_match.group(1).split("+") if prefix_match else []

    def get_ordered_role_pids(
        self, annotations: List[dict], role: str, function_instance_id: int
    ) -> List[Any]:
        """Return unique PIDs, preferring a part assigned only to this role."""
        role_token = f"{'r' if role == 'receptor' else 'e'}{function_instance_id}"
        primary_annotation = next(
            (
                annotation for annotation in annotations
                if self.parse_part_role_tokens(str(annotation.get("label", "")))
                == [role_token]
            ),
            annotations[0] if annotations else None,
        )
        ordered_annotations = (
            [primary_annotation]
            + [
                annotation for annotation in annotations
                if annotation is not primary_annotation
            ]
            if primary_annotation is not None else []
        )
        pids = []
        for annotation in ordered_annotations:
            annotation_pid = annotation.get(
                "pid", annotation.get("partId", annotation.get("objectId"))
            )
            if annotation_pid is not None and annotation_pid not in pids:
                pids.append(annotation_pid)
        return pids

    @staticmethod
    def get_open3d_triangle_index_remap(
        full_mesh_path: str,
        full_mesh: o3d.geometry.TriangleMesh,
    ) -> np.ndarray:
        """Map GLTF node-order triangle indices to Open3D's loaded order.

        The part annotations index triangles in GLTF node/primitive order.
        Open3D's legacy GLB reader instead groups primitives by material while
        flattening them into one ``TriangleMesh``. When a material is reused by
        non-adjacent nodes, directly applying annotation indices can therefore
        select triangles from the wrong node.
        """
        try:
            from pygltflib import GLTF2
        except ImportError as exc:
            raise ImportError(
                "Loading triangle-index annotations from GLTF/GLB requires "
                "pygltflib so their indices can be aligned with Open3D."
            ) from exc

        gltf = GLTF2().load(full_mesh_path)
        primitive_records = []
        source_offset = 0
        for node_index, node in enumerate(gltf.nodes or []):
            if node.mesh is None:
                continue
            mesh = gltf.meshes[node.mesh]
            for primitive_index, primitive in enumerate(mesh.primitives):
                if primitive.mode not in (None, 4):
                    raise ValueError(
                        f"Unsupported GLTF primitive mode {primitive.mode} in "
                        f"{full_mesh_path}; triangle annotations require mode 4."
                    )
                accessor_index = primitive.indices
                if accessor_index is None:
                    accessor_index = primitive.attributes.POSITION
                index_count = gltf.accessors[accessor_index].count
                if index_count % 3:
                    raise ValueError(
                        f"Primitive {(node_index, primitive_index)} in "
                        f"{full_mesh_path} has {index_count} indices."
                    )
                face_count = index_count // 3
                material = (
                    primitive.material if primitive.material is not None else -1
                )
                primitive_records.append(
                    {
                        "key": (node_index, primitive_index),
                        "material": material,
                        "face_count": face_count,
                        "source_start": source_offset,
                    }
                )
                source_offset += face_count

        loaded_face_count = len(full_mesh.triangles)
        if source_offset != loaded_face_count:
            raise ValueError(
                f"GLTF contains {source_offset} triangle faces but Open3D loaded "
                f"{loaded_face_count} from {full_mesh_path}."
            )

        primitives_by_material = {}
        for record in primitive_records:
            primitives_by_material.setdefault(record["material"], []).append(record)
        open3d_records = [
            record
            for material_records in primitives_by_material.values()
            for record in material_records
        ]

        target_offset = 0
        target_start_by_key = {}
        expected_material_ids = []
        for record in open3d_records:
            target_start_by_key[record["key"]] = target_offset
            expected_material_ids.append(
                np.full(
                    record["face_count"], record["material"], dtype=np.int32
                )
            )
            target_offset += record["face_count"]

        source_keys = [record["key"] for record in primitive_records]
        target_keys = [record["key"] for record in open3d_records]
        if source_keys != target_keys:
            loaded_material_ids = np.asarray(full_mesh.triangle_material_ids)
            expected_material_ids = np.concatenate(expected_material_ids)
            if (len(loaded_material_ids) != loaded_face_count
                    or not np.array_equal(
                        loaded_material_ids.astype(np.int32, copy=False),
                        expected_material_ids,
                    )):
                raise ValueError(
                    "Could not confirm Open3D's material-grouped triangle "
                    f"ordering for {full_mesh_path}."
                )

        remap = np.empty(loaded_face_count, dtype=np.int64)
        for record in primitive_records:
            source_start = record["source_start"]
            source_stop = source_start + record["face_count"]
            target_start = target_start_by_key[record["key"]]
            remap[source_start:source_stop] = (
                target_start + np.arange(record["face_count"], dtype=np.int64)
            )
        return remap

    def load_geometry_data(
        self,
        geometry_path: str,
        geometry_type: str,
        transformation_path: Optional[str],
        part_annotation_path: str,
        function_instance_id: int,
        has_object: bool,
        articulation_path: Optional[str] = None,
        normalize_articulation_geometry: bool = True,
    ) -> dict:
        """Load either aligned mesh or point-cloud geometry."""
        normalized_type = geometry_type.lower().replace("_", " ")
        if normalized_type == "mesh":
            return self.load_mesh_data(
                geometry_path,
                transformation_path,
                part_annotation_path,
                function_instance_id,
                has_object,
                # Old meshes are already expressed in their observed world
                # frame.  Only canonical meshes with an explicit transform
                # need to be normalized to articulation minima first.
                articulation_path=(
                    articulation_path if normalize_articulation_geometry else None
                ),
            )
        if normalized_type == "point cloud":
            return self.load_point_cloud_data(
                geometry_path,
                transformation_path,
                part_annotation_path,
                function_instance_id,
            )
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

    def load_point_cloud_data(
        self,
        geometry_path: str,
        transformation_path: Optional[str],
        part_annotation_path: str,
        function_instance_id: int,
    ) -> dict:
        """Load old point-cloud geometry through the aligned metadata schema."""
        full_pcd = pcu.load_mesh_v(
            os.path.join(self.root_path, geometry_path), np.float32
        )
        annotation_data, role_annotations = self.load_role_part_annotations(
            part_annotation_path, function_instance_id
        )
        canonical2world = (
            self.load_object_transformation(transformation_path)
            if transformation_path is not None else np.eye(4)
        )
        if canonical2world.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 canonical-to-world transform, got "
                f"{canonical2world.shape}."
            )
        full_pcd = (
            full_pcd @ canonical2world[:3, :3].T + canonical2world[:3, 3]
        )
        geometry_annotations = {"canonical_to_world": canonical2world}
        selected_labels = {}
        for role, annotations in role_annotations.items():
            if not annotations:
                if role != "effector":
                    raise ValueError(f"No point-cloud annotation found for role '{role}'.")
                part_pcd = full_pcd
                pid = None
                pids = []
                selected_labels[role] = "whole point cloud"
            else:
                vertex_indices = np.unique(
                    np.concatenate(
                        [
                            np.asarray(
                                annotation.get("vertexIndices", annotation.get("indices")),
                                dtype=np.int64,
                            )
                            for annotation in annotations
                        ]
                    )
                )
                if (vertex_indices.size == 0 or vertex_indices.min() < 0
                        or vertex_indices.max() >= len(full_pcd)):
                    raise ValueError(
                        f"Invalid point indices in {part_annotation_path}."
                    )
                part_pcd = full_pcd[vertex_indices]
                pids = self.get_ordered_role_pids(
                    annotations, role, function_instance_id
                )
                pid = pids[0] if pids else None
                selected_labels[role] = " + ".join(
                    str(annotation["label"]) for annotation in annotations
                )
            geometry_annotations[role] = {
                "part_pcd": part_pcd,
                "pid": pid,
                "pids": pids,
            }
        geometry_annotations["object"] = {
            "part_pcd": full_pcd,
            "pid": None,
        }
        relation = (
            annotation_data.get("relation", annotation_data.get("description"))
            if isinstance(annotation_data, dict) else None
        )
        geometry_annotations["relation"] = relation or (
            f"{selected_labels['receptor']} and {selected_labels['effector']}"
        )
        return geometry_annotations

    def load_mesh_data(
        self,
        geometry_path: str,
        transformation_path: Optional[str],
        part_annotation_path: str,
        function_instance_id: int,
        has_object: bool,
        articulation_path: Optional[str] = None,
    ) -> dict:
        required_paths = {
            "mesh": geometry_path,
            "part annotation": part_annotation_path,
        }
        for description, path in required_paths.items():
            if path is None:
                raise KeyError(f"Metadata entry is missing the {description} path.")

        full_mesh_path = os.path.join(self.root_path, geometry_path)
        full_mesh = o3d.io.read_triangle_mesh(full_mesh_path)
        if full_mesh.is_empty() or len(full_mesh.triangles) == 0:
            raise ValueError(f"Could not load a triangle mesh from {full_mesh_path}.")

        annotation_data, role_annotations = self.load_role_part_annotations(
            part_annotation_path, function_instance_id
        )
        articulation_records = self.load_articulation_records(articulation_path)
        articulations_by_pid = {}
        for articulation in articulation_records:
            pid = articulation.get("pid")
            if pid is not None:
                articulations_by_pid.setdefault(str(pid), []).append(articulation)

        canonical2world = (
            self.load_object_transformation(transformation_path)
            if transformation_path is not None else np.eye(4)
        )
        if canonical2world.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 canonical-to-world transform, got {canonical2world.shape}."
            )

        full_vertices = np.asarray(full_mesh.vertices)
        full_triangles = np.asarray(full_mesh.triangles)
        has_triangle_annotations = any(
            annotation.get("triIndices") is not None
            for annotations in role_annotations.values()
            for annotation in annotations
        )
        triangle_index_remap = (
            self.get_open3d_triangle_index_remap(full_mesh_path, full_mesh)
            if has_triangle_annotations else None
        )

        def annotated_part_mesh(annotation: dict) -> o3d.geometry.TriangleMesh:
            triangle_indices = annotation.get("triIndices")
            if triangle_indices is not None:
                triangle_indices = np.asarray(triangle_indices, dtype=np.int64)
                if (triangle_indices.size == 0 or triangle_indices.min() < 0
                        or triangle_indices.max() >= len(full_triangles)):
                    raise ValueError(f"Invalid triangle indices for part '{annotation.get('label')}'.")
                triangle_indices = triangle_index_remap[triangle_indices]
                selected_triangles = full_triangles[triangle_indices]
                selected_vertices, remapped = np.unique(
                    selected_triangles.reshape(-1), return_inverse=True
                )
                part_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(full_vertices[selected_vertices]),
                    o3d.utility.Vector3iVector(remapped.reshape(-1, 3)),
                )
                return part_mesh

            vertex_indices = annotation.get("vertexIndices", annotation.get("indices"))
            if vertex_indices is None or len(vertex_indices) == 0:
                raise ValueError(f"No mesh indices found for part '{annotation.get('label')}'.")
            return full_mesh.select_by_index(vertex_indices)

        def normalized_part_mesh(annotation: dict) -> o3d.geometry.TriangleMesh:
            part_mesh = annotated_part_mesh(annotation)
            pid = annotation.get(
                "pid", annotation.get("partId", annotation.get("objectId"))
            )
            return self.transform_part_mesh_to_min_joint_values(
                part_mesh, articulations_by_pid.get(str(pid), [])
            )

        def world_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError("Cannot transform an empty annotated part mesh.")
            transformed_mesh = copy.deepcopy(mesh)
            transformed_mesh.transform(canonical2world)
            return transformed_mesh

        def sample_points(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError("Cannot sample an empty annotated part mesh.")
            return np.asarray(
                mesh.sample_points_uniformly(number_of_points=10000).points
            )

        def mesh_data(mesh: o3d.geometry.TriangleMesh) -> dict:
            return {
                "vertices": np.asarray(mesh.vertices).copy(),
                "triangles": np.asarray(mesh.triangles).copy(),
            }

        geometry_annotations = {"canonical_to_world": canonical2world}
        selected_labels = {}
        loaded_role_parts = {}
        full_mesh_points = None
        full_world_mesh = None
        for role, annotations in role_annotations.items():
            if not annotations:
                if role != "effector":
                    raise ValueError(f"No mesh annotation found for role '{role}'.")
                if full_world_mesh is None:
                    full_world_mesh = world_mesh(full_mesh)
                if full_mesh_points is None:
                    full_mesh_points = sample_points(full_world_mesh)
                part_points = full_mesh_points
                part_mesh = full_world_mesh
                pid = None
                pids = []
                selected_labels[role] = "whole mesh"
            else:
                annotation_key = tuple(id(annotation) for annotation in annotations)
                if annotation_key not in loaded_role_parts:
                    canonical_part_mesh = normalized_part_mesh(annotations[0])
                    for annotation in annotations[1:]:
                        canonical_part_mesh += normalized_part_mesh(annotation)
                    transformed_part_mesh = world_mesh(canonical_part_mesh)
                    loaded_role_parts[annotation_key] = (
                        transformed_part_mesh,
                        sample_points(transformed_part_mesh),
                    )
                part_mesh, part_points = loaded_role_parts[annotation_key]

                pids = self.get_ordered_role_pids(
                    annotations, role, function_instance_id
                )
                pid = pids[0] if pids else None
                selected_labels[role] = " + ".join(
                    str(annotation["label"]) for annotation in annotations
                )
            geometry_annotations[role] = {
                "part_pcd": part_points,
                "part_mesh": mesh_data(part_mesh),
                "pid": pid,
                "pids": pids,
            }

        if has_object or not role_annotations["effector"]:
            if full_world_mesh is None:
                full_world_mesh = world_mesh(full_mesh)
            if full_mesh_points is None:
                full_mesh_points = sample_points(full_world_mesh)
            object_points = full_mesh_points
            object_mesh = full_world_mesh
        else:
            object_points = np.concatenate(
                [geometry_annotations[role]["part_pcd"] for role in ("receptor", "effector")],
                axis=0,
            )
            receptor_key = tuple(
                id(annotation) for annotation in role_annotations["receptor"]
            )
            effector_key = tuple(
                id(annotation) for annotation in role_annotations["effector"]
            )
            object_mesh = copy.deepcopy(loaded_role_parts[receptor_key][0])
            object_mesh += loaded_role_parts[effector_key][0]
        geometry_annotations["object"] = {
            "part_pcd": object_points,
            "part_mesh": mesh_data(object_mesh),
            "pid": None,
        }
        relation = None
        if isinstance(annotation_data, dict):
            relation = annotation_data.get("relation", annotation_data.get("description"))
            if relation is None and isinstance(annotation_data.get("data"), dict):
                relation = annotation_data["data"].get(
                    "relation", annotation_data["data"].get("description")
                )
        geometry_annotations["relation"] = relation or (
            f"{selected_labels['receptor']} and {selected_labels['effector']}"
        )
        return geometry_annotations

    def load_object_transformation(self, transformation_path: str) -> np.ndarray:
        with open(os.path.join(self.root_path, transformation_path), "r") as f:
            transformation_data = json.load(f)
        canonical2world = np.array(transformation_data["canonical_to_world"])
        return canonical2world

    def load_articulation_records(
        self, articulation_path: Optional[str]
    ) -> List[dict]:
        """Load all articulation records, preserving their file order."""
        if articulation_path is None:
            return []
        full_articulation_path = os.path.join(self.root_path, articulation_path)
        if not os.path.exists(full_articulation_path):
            return []
        with open(full_articulation_path, "r") as f:
            articulation_data = json.load(f)
        records = (
            articulation_data.get("articulations", [])
            if isinstance(articulation_data, dict)
            else articulation_data
        )
        if not isinstance(records, list):
            raise ValueError(f"Invalid articulation data in {articulation_path}.")
        return records

    @staticmethod
    def transform_part_mesh_to_min_joint_values(
        part_mesh: o3d.geometry.TriangleMesh,
        articulation_records: Sequence[dict],
    ) -> o3d.geometry.TriangleMesh:
        """Move a part from joint value zero to every matching minimum value.

        Revolute ranges are interpreted in radians and prismatic ranges in the
        mesh's canonical distance unit. Records without a finite minimum (for
        example continuous joints) do not transform the mesh.
        """
        transformed_mesh = copy.deepcopy(part_mesh)
        for articulation in articulation_records:
            joint_type = str(articulation.get("type", "")).lower()
            range_min = articulation.get(
                "rangeMin", articulation.get("range_min")
            )
            if range_min is None or joint_type == "continuous":
                continue
            range_min = float(range_min)
            if not np.isfinite(range_min):
                raise ValueError("Articulation rangeMin must be finite.")

            # The annotated mesh is assumed to be at joint value zero.
            joint_delta = range_min
            if np.isclose(joint_delta, 0.0):
                continue

            axis = np.asarray(articulation.get("axis"), dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError("Articulation axis must contain three finite values.")
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 0:
                raise ValueError("Articulation axis must be non-zero.")
            axis = axis / axis_norm

            if joint_type == "revolute":
                origin = articulation.get("origin")
                if origin is None:
                    raise ValueError("A revolute articulation requires an origin.")
                origin = np.asarray(origin, dtype=float)
                if origin.shape != (3,) or not np.isfinite(origin).all():
                    raise ValueError(
                        "Articulation origin must contain three finite values."
                    )
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    axis * joint_delta
                )
                transformed_mesh.rotate(rotation, center=origin)
            elif joint_type == "prismatic":
                transformed_mesh.translate(axis * joint_delta)
            else:
                raise ValueError(
                    f"Unsupported articulation type '{articulation.get('type')}'."
                )
        return transformed_mesh

    def load_articulation(
        self,
        articulation_path: str,
        geometry_data: Optional[dict],
        part_annotation_path: Optional[str] = None,
        function_instance_id: Optional[int] = None,
        transformation_path: Optional[str] = None,
        transform_to_world: bool = True,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        receptor_articulation = None
        effector_articulation = None
        if articulation_path is None:
            return receptor_articulation, effector_articulation

        joints = self.load_articulation_records(articulation_path)
        joints_by_pid = {str(joint.get("pid")): joint for joint in joints}
        if geometry_data is None:
            _, role_annotations = self.load_role_part_annotations(
                part_annotation_path, function_instance_id
            )
            role_geometry_data = {
                role: {
                    "pid": None,
                    "pids": self.get_ordered_role_pids(
                        annotations, role, function_instance_id
                    ),
                }
                for role, annotations in role_annotations.items()
            }
            for role_data in role_geometry_data.values():
                role_data["pid"] = (
                    role_data["pids"][0] if role_data["pids"] else None
                )
            canonical2world = (
                self.load_object_transformation(transformation_path)
                if transformation_path is not None else None
            )
        else:
            role_geometry_data = geometry_data
            canonical2world = geometry_data.get("canonical_to_world")

        def joint_for_role(role: str):
            role_pids = role_geometry_data[role].get(
                "pids", [role_geometry_data[role].get("pid")]
            )
            joint = next(
                (
                    joints_by_pid[str(pid)] for pid in role_pids
                    if pid is not None and str(pid) in joints_by_pid
                ),
                None,
            )
            if joint is None:
                return None
            joint = copy.deepcopy(joint)
            if transform_to_world and canonical2world is not None:
                canonical2world_array = np.asarray(canonical2world)
                if "axis" in joint:
                    axis = canonical2world_array[:3, :3] @ np.asarray(joint["axis"])
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 0:
                        joint["axis"] = (axis / axis_norm).tolist()
                if joint.get("ref") is not None:
                    reference = (
                        canonical2world_array[:3, :3] @ np.asarray(joint["ref"])
                    )
                    joint["ref"] = reference.tolist()
                if joint.get("origin") is not None:
                    origin = np.append(np.asarray(joint["origin"]), 1.0)
                    joint["origin"] = (canonical2world_array @ origin)[:3].tolist()
            return joint

        receptor_articulation = joint_for_role("receptor")
        effector_articulation = joint_for_role("effector")
        return receptor_articulation, effector_articulation

    def load_function_annotation(self, function_annotation_path: str, function_instance_id: str) -> dict:
        if function_annotation_path is None:
            raise KeyError("Metadata entry is missing 'function'.")
        with open(os.path.join(self.root_path, function_annotation_path), "r") as f:
            function_data = json.load(f)
        function_annotation = function_data[str(function_instance_id)]
        return function_annotation


class CombinedDataset(NewDataset):
    """Load the unified old/new dataset from its combined metadata.

    Combined metadata always declares both undistorted_video and
    cropped_video. When cropped input is requested, a stored cropped video
    is preferred; records without one load and crop the undistorted video.
    """

    def __init__(
        self,
        root_path: str,
        meta_file_path: str,
        image_type: str = "undistorted",
        sample_strategy: str = "fix_size",
        sample_num: int = 20,
        load_2d_masks: bool = True,
        load_mesh_data: bool = True,
        load_articulation: bool = True,
        load_function_annotation: bool = True,
    ):
        super().__init__(
            root_path=root_path,
            meta_file_path=meta_file_path,
            image_type=image_type,
            sample_strategy=sample_strategy,
            sample_num=sample_num,
            load_2d_masks=load_2d_masks,
            load_mesh_data=load_mesh_data,
            load_articulation=load_articulation,
            load_function_annotation=load_function_annotation,
        )
        if self.image_type not in {"undistorted", "cropped"}:
            raise ValueError(
                "CombinedDataset image_type must be 'undistorted' or 'cropped', "
                f"got {self.image_type!r}."
            )
        self._validate_combined_video_metadata()

    def _validate_combined_video_metadata(self) -> None:
        seen_video_names = set()
        for index, item in enumerate(self.meta_info):
            missing_keys = {
                key
                for key in ("video_name", "undistorted_video", "cropped_video")
                if key not in item
            }
            if missing_keys:
                raise KeyError(
                    f"Combined metadata entry {index} is missing keys "
                    f"{sorted(missing_keys)}."
                )

            video_name = item["video_name"]
            if video_name in seen_video_names:
                raise ValueError(
                    f"Duplicate combined metadata video_name: {video_name}"
                )
            seen_video_names.add(video_name)

            undistorted_video = item["undistorted_video"]
            if not isinstance(undistorted_video, str) or not undistorted_video:
                raise ValueError(
                    f"Combined metadata entry {index} has an invalid "
                    f"undistorted_video: {undistorted_video!r}."
                )
            cropped_video = item["cropped_video"]
            if cropped_video is not None and (
                not isinstance(cropped_video, str) or not cropped_video
            ):
                raise ValueError(
                    f"Combined metadata entry {index} has an invalid "
                    f"cropped_video: {cropped_video!r}."
                )

    def _select_video_path(
        self, video_dict: Mapping[str, Any]
    ) -> Tuple[str, bool]:
        cropped_video = video_dict["cropped_video"]
        if self.image_type == "cropped" and cropped_video is not None:
            return cropped_video, True
        return video_dict["undistorted_video"], False


def _create_articulation_arrow(
    axis: np.ndarray,
    shaft_center: np.ndarray,
    length: float,
    color: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Create an Open3D arrow whose shaft follows an axis through a point."""
    cylinder_height = length * 0.75
    cone_height = length - cylinder_height
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * 0.025,
        cone_radius=length * 0.06,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=24,
        cylinder_split=4,
        cone_split=1,
    )

    source_axis = np.asarray([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(source_axis, axis), -1.0, 1.0))
    if np.isclose(cosine, 1.0):
        rotation = np.eye(3)
    elif np.isclose(cosine, -1.0):
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.asarray([np.pi, 0.0, 0.0])
        )
    else:
        rotation_axis = np.cross(source_axis, axis)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * np.arccos(cosine)
        )

    arrow.rotate(rotation, center=[0.0, 0.0, 0.0])
    arrow.translate(shaft_center - axis * (cylinder_height * 0.5))
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()
    return arrow


def visualize_loaded_geometry(
    data: Dict[str, Any],
    roles: Optional[Sequence[str]] = None,
    role_colors: Optional[Mapping[str, Sequence[float]]] = None,
    point_size: float = 3.0,
    show_mesh_wireframe: bool = True,
    show_articulations: bool = True,
    articulations: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    articulation_arrow_scale: float = 0.3,
    show_coordinate_frame: bool = True,
    window_name: Optional[str] = None,
    window_width: int = 1280,
    window_height: int = 720,
    geometry_type: str = "auto",
) -> None:
    """Visualize loaded meshes or point clouds in an Open3D window.

    ``data`` may be either a dataset item returned by ``__getitem__`` or its
    ``geometry_data`` dictionary. Mesh triangles are colored by annotated
    role. By default, receptor and effector are displayed without the full
    object mesh so the object does not obscure overlapping annotated parts.
    Loaded revolute axes are red and pass through their joint origins. Loaded
    prismatic axes are green and pass through their associated part centers.

    Args:
        data: Loaded dataset item or geometry-data dictionary.
        roles: Role names to show, such as ``("receptor", "effector")``.
        role_colors: Optional RGB colors in the range [0, 1], keyed by role.
        point_size: Open3D render point size in pixels.
        show_mesh_wireframe: Whether to draw triangle edges in mesh mode.
        show_articulations: Whether to draw loaded articulation axes.
        articulations: Optional role-to-joint mapping when ``data`` is a bare
            geometry dictionary. Dataset items provide this automatically.
        articulation_arrow_scale: Arrow length relative to the displayed
            geometry's bounding-box diagonal.
        show_coordinate_frame: Whether to draw the world-coordinate axes.
        window_name: Title of the Open3D window.
        window_width: Initial window width in pixels.
        window_height: Initial window height in pixels.
        geometry_type: ``"auto"`` (default), ``"mesh"``, or
            ``"point_cloud"``. Auto selects mesh when all requested roles
            contain triangle data and otherwise selects point cloud.

    Example:
        >>> sample = dataset[0]
        >>> visualize_loaded_geometry(sample)
        >>> visualize_loaded_geometry(
        ...     sample, roles=("receptor", "effector"), geometry_type="mesh"
        ... )
    """
    dataset_item = data if "geometry_data" in data else None
    if dataset_item is not None:
        geometry_data = data["geometry_data"]
    else:
        geometry_data = data
    if not isinstance(geometry_data, dict):
        raise TypeError("data must be a dataset item or geometry-data dictionary.")

    if articulations is None and dataset_item is not None:
        articulations = {
            "receptor": dataset_item.get(
                "receptor_articulation",
                dataset_item.get("receiver_articulation"),
            ),
            "effector": dataset_item.get("effector_articulation"),
        }
    role_articulations = dict(articulations or {})
    if "receptor" not in role_articulations and "receiver" in role_articulations:
        role_articulations["receptor"] = role_articulations["receiver"]

    geometry_keys = {"point_cloud": "part_pcd", "mesh": "part_mesh"}
    if geometry_type not in {"auto", *geometry_keys}:
        raise ValueError(
            "geometry_type must be 'auto', 'mesh', or 'point_cloud'; "
            f"got {geometry_type!r}."
        )

    if isinstance(roles, str):
        requested_roles = [roles]
    else:
        requested_roles = list(roles) if roles is not None else None
    if geometry_type == "auto":
        candidate_roles = requested_roles or [
            role
            for role, role_data in geometry_data.items()
            if isinstance(role_data, dict) and role != "object"
        ]
        geometry_type = (
            "mesh"
            if candidate_roles
            and all(
                isinstance(geometry_data.get(role), dict)
                and "part_mesh" in geometry_data[role]
                for role in candidate_roles
            )
            else "point_cloud"
        )
    geometry_key = geometry_keys[geometry_type]

    available_roles = [
        role for role, role_data in geometry_data.items()
        if isinstance(role_data, dict) and geometry_key in role_data
    ]
    if requested_roles is None:
        selected_roles = [
            role for role in ("receptor", "effector") if role in available_roles
        ]
        if not selected_roles:
            selected_roles = available_roles
    else:
        selected_roles = requested_roles
    if not selected_roles:
        raise ValueError(f"No role {geometry_type} data were found to visualize.")

    missing_roles = [role for role in selected_roles if role not in available_roles]
    if missing_roles:
        raise ValueError(
            f"No {geometry_type} found for role(s) {missing_roles}. "
            f"Available roles: {available_roles}."
        )
    if point_size <= 0:
        raise ValueError("point_size must be positive.")
    if articulation_arrow_scale <= 0 or not np.isfinite(articulation_arrow_scale):
        raise ValueError("articulation_arrow_scale must be positive and finite.")
    if window_width <= 0 or window_height <= 0:
        raise ValueError("Window dimensions must be positive.")
    if window_name is None:
        window_name = (
            "Loaded annotated mesh triangles"
            if geometry_type == "mesh"
            else "Loaded role point clouds"
        )

    default_colors = {
        "receptor": (1.0, 0.25, 0.10),
        "effector": (0.10, 0.45, 1.0),
        "object": (0.65, 0.65, 0.65),
    }
    fallback_colors = (
        (0.20, 0.80, 0.35),
        (0.85, 0.25, 0.75),
        (1.00, 0.75, 0.10),
        (0.15, 0.80, 0.85),
    )
    requested_colors = dict(role_colors or {})
    role_geometries = []
    all_points = []
    role_centers = {}

    print(f"{geometry_type.replace('_', ' ').title()} visualization legend:")
    for role_index, role in enumerate(selected_roles):
        color = requested_colors.get(
            role,
            default_colors.get(role, fallback_colors[role_index % len(fallback_colors)]),
        )
        color = np.asarray(color, dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError(f"Color for role '{role}' must contain three finite values.")
        if np.any(color < 0) or np.any(color > 1):
            raise ValueError(f"Color for role '{role}' must be in the range [0, 1].")

        if geometry_type == "point_cloud":
            points = np.asarray(geometry_data[role][geometry_key])
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"Role '{role}' must have a point cloud shaped (N, 3); "
                    f"got {points.shape}."
                )
            if len(points) == 0:
                raise ValueError(f"Role '{role}' has an empty point cloud.")
            if not np.isfinite(points).all():
                raise ValueError(
                    f"Role '{role}' contains non-finite point coordinates."
                )

            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(
                points.astype(float, copy=False)
            )
            count_description = f"{len(points):,} points"
        else:
            role_mesh = geometry_data[role][geometry_key]
            if not isinstance(role_mesh, Mapping):
                raise ValueError(
                    f"Role '{role}' mesh must contain 'vertices' and 'triangles'."
                )
            vertices = np.asarray(role_mesh.get("vertices"))
            triangles = np.asarray(role_mesh.get("triangles"))
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
                raise ValueError(
                    f"Role '{role}' mesh vertices must have shape (N, 3); "
                    f"got {vertices.shape}."
                )
            if not np.isfinite(vertices).all():
                raise ValueError(
                    f"Role '{role}' contains non-finite mesh vertices."
                )
            if (triangles.ndim != 2 or triangles.shape[1] != 3
                    or len(triangles) == 0):
                raise ValueError(
                    f"Role '{role}' mesh triangles must have shape (M, 3); "
                    f"got {triangles.shape}."
                )
            if not np.issubdtype(triangles.dtype, np.integer):
                raise ValueError(f"Role '{role}' mesh triangle indices must be integers.")
            if triangles.min() < 0 or triangles.max() >= len(vertices):
                raise ValueError(
                    f"Role '{role}' mesh contains out-of-range triangle indices."
                )

            points = vertices
            # Legacy Open3D releases do not expose per-triangle colors. Make
            # each triangle own its three vertices so vertex colors act as
            # unambiguous face colors without interpolation across parts.
            colored_vertices = vertices[triangles].reshape(-1, 3)
            colored_triangles = np.arange(
                len(colored_vertices), dtype=np.int32
            ).reshape(-1, 3)
            colored_vertices_rgb = np.repeat(
                color[None, :], len(colored_vertices), axis=0
            )
            geometry = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(
                    colored_vertices.astype(float, copy=False)
                ),
                o3d.utility.Vector3iVector(colored_triangles),
            )
            geometry.vertex_colors = o3d.utility.Vector3dVector(
                colored_vertices_rgb
            )
            geometry.compute_triangle_normals()
            geometry.compute_vertex_normals()
            count_description = (
                f"{len(vertices):,} vertices, {len(triangles):,} triangles"
            )

        if geometry_type == "point_cloud":
            geometry.paint_uniform_color(color)
        role_geometries.append(geometry)
        all_points.append(points)
        role_centers[role] = points.mean(axis=0)
        pids = geometry_data[role].get("pids")
        if pids is None:
            pid = geometry_data[role].get("pid")
            pids = [] if pid is None else [pid]
        pid_description = f", part IDs={list(pids)}" if pids else ""
        print(
            f"  {role}: {count_description}{pid_description}, "
            f"RGB={color.tolist()}, "
            f"min={points.min(axis=0).tolist()}, max={points.max(axis=0).tolist()}"
        )

    combined_points = np.concatenate(all_points, axis=0)
    scene_extent = float(
        np.linalg.norm(combined_points.max(axis=0) - combined_points.min(axis=0))
    )
    scene_scale = max(scene_extent, 1e-3)
    geometries = list(role_geometries)

    articulation_colors = {
        "revolute": np.asarray([1.0, 0.0, 0.0]),
        "prismatic": np.asarray([0.0, 1.0, 0.0]),
    }
    if show_articulations:
        displayed_joint_count = 0
        for role in selected_roles:
            joint = role_articulations.get(role)
            if joint is None:
                continue
            if not isinstance(joint, Mapping):
                raise ValueError(
                    f"Articulation for role '{role}' must be a mapping."
                )

            raw_joint_type = str(
                joint.get("type", joint.get("joint_type", ""))
            ).lower()
            joint_type = (
                "revolute" if raw_joint_type == "continuous" else raw_joint_type
            )
            if joint_type not in articulation_colors:
                raise ValueError(
                    f"Unsupported articulation type {raw_joint_type!r} "
                    f"for role '{role}'."
                )

            axis = np.asarray(joint.get("axis"), dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError(
                    f"Articulation axis for role '{role}' must contain "
                    "three finite values."
                )
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 0:
                raise ValueError(
                    f"Articulation axis for role '{role}' must be non-zero."
                )
            axis = axis / axis_norm

            if joint_type == "revolute":
                origin = joint.get("origin", joint.get("position", joint.get("pos")))
                if origin is None:
                    raise ValueError(
                        f"Revolute articulation for role '{role}' has no origin."
                    )
                shaft_center = np.asarray(origin, dtype=float)
                if shaft_center.shape != (3,) or not np.isfinite(shaft_center).all():
                    raise ValueError(
                        f"Revolute origin for role '{role}' must contain "
                        "three finite values."
                    )
                anchor_description = f"origin={shaft_center.tolist()}"
            else:
                shaft_center = role_centers[role]
                anchor_description = f"part center={shaft_center.tolist()}"

            color = articulation_colors[joint_type]
            geometries.append(
                _create_articulation_arrow(
                    axis=axis,
                    shaft_center=shaft_center,
                    length=scene_scale * articulation_arrow_scale,
                    color=color,
                )
            )
            displayed_joint_count += 1
            print(
                f"  {role} articulation: {raw_joint_type}, axis={axis.tolist()}, "
                f"{anchor_description}, RGB={color.tolist()}"
            )
        if displayed_joint_count:
            print("Articulation legend: revolute/continuous=red, prismatic=green")

    if show_coordinate_frame:
        frame_size = scene_scale * 0.15
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size, origin=[0.0, 0.0, 0.0]
            )
        )

    visualizer = o3d.visualization.Visualizer()
    window_created = visualizer.create_window(
        window_name=window_name,
        width=window_width,
        height=window_height,
    )
    if not window_created:
        raise RuntimeError(
            "Open3D could not create a window. Check that a graphical display is available."
        )
    try:
        for geometry in geometries:
            visualizer.add_geometry(geometry)
        render_options = visualizer.get_render_option()
        render_options.point_size = float(point_size)
        render_options.background_color = np.asarray([0.03, 0.03, 0.03])
        if geometry_type == "mesh":
            render_options.mesh_show_wireframe = bool(show_mesh_wireframe)
            render_options.mesh_show_back_face = True
        visualizer.run()
    finally:
        visualizer.destroy_window()


# Backward-compatible import for existing scripts.
visualize_loaded_point_clouds = visualize_loaded_geometry


def visualize_loaded_video_with_masks(
    data: Dict[str, Any],
    roles: Optional[Sequence[str]] = None,
    role_colors: Optional[Mapping[str, Sequence[float]]] = None,
    alpha: float = 0.45,
    fps: float = 10.0,
    show_legend: bool = True,
    window_name: str = "Loaded video with mask overlays",
) -> None:
    """Play loaded RGB frames with role masks overlaid in one OpenCV window.

    The function expects a dataset item returned by ``__getitem__``. By
    default it overlays receptor and effector masks; pass ``roles`` to include
    other loaded masks such as ``object``. Overlapping masks are shown using
    the mean of their role colors.

    Playback controls:
        - ``Space``: pause or resume
        - ``Left``/``A`` and ``Right``/``D``: step backward or forward
        - ``R``: restart from the first loaded frame
        - ``Q``/``Esc``: close the viewer

    Args:
        data: Loaded dataset item containing ``rgb_list`` and role mask lists.
        roles: Roles to overlay. Defaults to receptor and effector when present.
        role_colors: Optional RGB colors in the range [0, 1], keyed by role.
        alpha: Mask opacity in the range [0, 1].
        fps: Playback speed for the loaded (possibly sampled) frames.
        show_legend: Whether to draw role colors and frame position on each frame.
        window_name: Title of the OpenCV window.

    Example:
        >>> sample = dataset[0]
        >>> visualize_loaded_video_with_masks(sample)
        >>> visualize_loaded_video_with_masks(
        ...     sample, roles=("object", "receptor", "effector"), fps=5
        ... )
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "visualize_loaded_video_with_masks requires OpenCV (cv2)."
        ) from exc

    if not isinstance(data, dict):
        raise TypeError("data must be a dataset item dictionary.")
    if "rgb_list" not in data:
        raise KeyError("Dataset item is missing 'rgb_list'.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0, 1].")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    rgb_list = data["rgb_list"]
    if len(rgb_list) == 0:
        raise ValueError("Cannot visualize an empty video.")

    available_roles = [
        key[:-len("_mask_list")]
        for key in data
        if key.endswith("_mask_list")
    ]
    if isinstance(roles, str):
        selected_roles = [roles]
    elif roles is None:
        selected_roles = [
            role for role in ("receptor", "effector")
            if role in available_roles
        ]
        if not selected_roles:
            selected_roles = available_roles
    else:
        selected_roles = list(roles)
    if not selected_roles:
        raise ValueError("No role masks were found to visualize.")

    missing_roles = [role for role in selected_roles if role not in available_roles]
    if missing_roles:
        raise ValueError(
            f"No mask found for role(s) {missing_roles}. "
            f"Available roles: {available_roles}."
        )

    first_frame = np.asarray(rgb_list[0])
    if first_frame.ndim == 2:
        frame_height, frame_width = first_frame.shape
    elif first_frame.ndim == 3 and first_frame.shape[2] in (1, 3, 4):
        frame_height, frame_width = first_frame.shape[:2]
    else:
        raise ValueError(
            f"RGB frames must be HxW, HxWx1, HxWx3, or HxWx4; "
            f"got {first_frame.shape}."
        )

    masks_by_role = {}
    for role in selected_roles:
        role_masks = np.asarray(data[f"{role}_mask_list"])
        expected_shape = (len(rgb_list), frame_height, frame_width)
        if role_masks.shape != expected_shape:
            raise ValueError(
                f"Mask list for role '{role}' must have shape {expected_shape}; "
                f"got {role_masks.shape}."
            )
        if not np.issubdtype(role_masks.dtype, np.bool_):
            if not np.issubdtype(role_masks.dtype, np.number):
                raise ValueError(f"Masks for role '{role}' must be numeric or boolean.")
            if not np.isfinite(role_masks).all():
                raise ValueError(f"Masks for role '{role}' contain non-finite values.")
        masks_by_role[role] = role_masks.astype(bool, copy=False)

    default_colors = {
        "receptor": (1.0, 0.25, 0.10),
        "effector": (0.10, 0.45, 1.0),
        "object": (0.65, 0.65, 0.65),
    }
    fallback_colors = (
        (0.20, 0.80, 0.35),
        (0.85, 0.25, 0.75),
        (1.00, 0.75, 0.10),
        (0.15, 0.80, 0.85),
    )
    requested_colors = dict(role_colors or {})
    colors = {}
    for role_index, role in enumerate(selected_roles):
        color = requested_colors.get(
            role,
            default_colors.get(role, fallback_colors[role_index % len(fallback_colors)]),
        )
        color = np.asarray(color, dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError(f"Color for role '{role}' must contain three finite values.")
        if np.any(color < 0) or np.any(color > 1):
            raise ValueError(f"Color for role '{role}' must be in the range [0, 1].")
        colors[role] = color

    sample_indices = data.get("sample_indices")
    if sample_indices is not None and len(sample_indices) != len(rgb_list):
        sample_indices = None

    def to_uint8_rgb(frame: np.ndarray, frame_index: int) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.shape[:2] != (frame_height, frame_width):
            raise ValueError(
                f"Frame {frame_index} has spatial shape {frame.shape[:2]}; expected "
                f"{(frame_height, frame_width)}."
            )
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame {frame_index} has unsupported shape {frame.shape}.")
        if not np.issubdtype(frame.dtype, np.number):
            raise ValueError(f"Frame {frame_index} must contain numeric pixel values.")
        if not np.isfinite(frame).all():
            raise ValueError(f"Frame {frame_index} contains non-finite pixel values.")
        if frame.dtype == np.uint8:
            return frame.copy()
        frame = frame.astype(np.float32)
        if frame.size and frame.min() >= 0 and frame.max() <= 1:
            frame *= 255.0
        return np.clip(frame, 0, 255).astype(np.uint8)

    def render_frame(frame_index: int) -> np.ndarray:
        frame = to_uint8_rgb(rgb_list[frame_index], frame_index)
        color_sum = np.zeros(frame.shape, dtype=np.float32)
        mask_count = np.zeros((frame_height, frame_width), dtype=np.float32)
        for role in selected_roles:
            role_mask = masks_by_role[role][frame_index]
            color_sum[role_mask] += colors[role] * 255.0
            mask_count[role_mask] += 1.0

        covered = mask_count > 0
        if np.any(covered):
            mean_mask_color = color_sum[covered] / mask_count[covered, None]
            blended = frame.astype(np.float32)
            blended[covered] = (
                (1.0 - alpha) * blended[covered] + alpha * mean_mask_color
            )
            frame = np.clip(blended, 0, 255).astype(np.uint8)

        if show_legend:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.45, min(frame_height, frame_width) / 1400.0)
            thickness = max(1, int(round(font_scale * 2)))
            box_size = max(12, int(round(font_scale * 20)))
            line_height = max(20, int(round(font_scale * 30)))
            x = 12
            for role_index, role in enumerate(selected_roles):
                y = 12 + role_index * line_height
                rgb_color = tuple(int(round(value * 255)) for value in colors[role])
                cv2.rectangle(
                    frame, (x, y), (x + box_size, y + box_size), rgb_color, -1
                )
                role_name = str(data.get(f"{role}_name", role))
                cv2.putText(
                    frame,
                    f"{role}: {role_name}",
                    (x + box_size + 8, y + box_size),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            frame_label = f"frame {frame_index + 1}/{len(rgb_list)}"
            if sample_indices is not None:
                frame_label += f" (source {int(sample_indices[frame_index])})"
            cv2.putText(
                frame,
                frame_label,
                (12, frame_height - 14),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return frame

    print("Video-mask visualization legend:")
    for role in selected_roles:
        print(
            f"  {role}: RGB={colors[role].tolist()}, "
            f"masked pixels={int(masks_by_role[role].sum()):,}"
        )
    print("Controls: Space pause/resume | A/Left previous | D/Right next | R restart | Q/Esc close")

    frame_delay_ms = max(1, int(round(1000.0 / fps)))
    frame_index = 0
    paused = False
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame_rgb = render_frame(frame_index)
            cv2.imshow(window_name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKeyEx(50 if paused else frame_delay_ms)

            if key in (27, ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key in (ord("a"), ord("A"), 81, 2424832):
                frame_index = (frame_index - 1) % len(rgb_list)
                paused = True
                continue
            if key in (ord("d"), ord("D"), 83, 2555904):
                frame_index = (frame_index + 1) % len(rgb_list)
                paused = True
                continue
            if key in (ord("r"), ord("R")):
                frame_index = 0
                continue
            if not paused:
                frame_index = (frame_index + 1) % len(rgb_list)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def build_dataset(dataset_config: dict) -> Dataset:
    dataset_name = dataset_config["name"]
    if dataset_name == "Uniform":
        return UniformDataset(
            root_path=dataset_config["root_path"],
            meta_file_path=dataset_config["meta_file"],
            image_type=dataset_config.get("image_type", "undistorted"),
            sample_strategy=dataset_config.get("sample_strategy", "fix_size"),
            sample_num=dataset_config.get("sample_num", 20)
        )
    elif dataset_name == "NewDataset":
        return NewDataset(
            root_path=dataset_config["root_path"],
            meta_file_path=dataset_config["meta_file"],
            image_type=dataset_config.get("image_type", "undistorted"),
            sample_strategy=dataset_config.get("sample_strategy", "fix_size"),
            sample_num=dataset_config.get("sample_num", 20),
            load_2d_masks=dataset_config.get("load_2d_masks", True),
            load_mesh_data=dataset_config.get("load_mesh_data", True),
            load_articulation=dataset_config.get("load_articulation", True),
            load_function_annotation=dataset_config.get(
                "load_function_annotation", True
            ),
        )
    elif dataset_name == "CombinedDataset":
        return CombinedDataset(
            root_path=dataset_config["root_path"],
            meta_file_path=dataset_config["meta_file"],
            image_type=dataset_config.get("image_type", "undistorted"),
            sample_strategy=dataset_config.get("sample_strategy", "fix_size"),
            sample_num=dataset_config.get("sample_num", 20),
            load_2d_masks=dataset_config.get("load_2d_masks", True),
            load_mesh_data=dataset_config.get("load_mesh_data", True),
            load_articulation=dataset_config.get("load_articulation", True),
            load_function_annotation=dataset_config.get(
                "load_function_annotation", True
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")


# def identity_collate(batch):
#     # batch is a list of dataset items
#     # with batch_size=1, just return the single element
#     return batch[0]


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Visualize a dataset item.")
#     parser.add_argument("dataset_config", type=str, help="Path to the dataset config JSON file.")
#     parser.add_argument("item_index", type=int, help="Index of the dataset item to visualize.")
#     parser.add_argument(
#         "--geometry-type",
#         choices=("auto", "point_cloud", "mesh"),
#         default="auto",
#         help="Render meshes when available, or explicitly select a geometry type.",
#     )
#     parser.add_argument(
#         "--roles",
#         nargs="+",
#         choices=("receptor", "effector", "object"),
#         default=None,
#         help=(
#             "Render only the selected role(s). For example, "
#             "'--roles receptor' shows the receptor alone."
#         ),
#     )
#     args = parser.parse_args()

#     with open(args.dataset_config, "r") as f:
#         dataset_config = omegaconf.OmegaConf.load(f)

#     dataset = build_dataset(dataset_config)
#     eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=identity_collate)

#     for data_count, data in enumerate(eval_dataloader):
#         if data_count != args.item_index:
#             continue
#         # visualize_loaded_video_with_masks(data)
#         if "geometry_data" in data:
#             visualize_loaded_geometry(
#                 data,
#                 roles=args.roles,
#                 geometry_type=args.geometry_type,
#             )
#         break
