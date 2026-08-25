import json
import copy
import time
import h5py
import numpy as np
import os
import point_cloud_utils as pcu
import open3d as o3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio

from typing import Tuple, List, Dict, Any, Mapping, Optional, Sequence


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
            "sample_indices": sample_indices,
            "num_total_frames": int(full_num_frames)
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
        """Load the HDF5 file back into the original dict format."""
        data = {}
        with h5py.File(filepath, 'r') as f:
            for name in f:
                grp = f[name]
                data[name] = {
                    'mask_idx': int(grp.attrs['mask_idx']),
                    'masks': grp['masks'][:]
                }
        return data
    
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
                receptor_name = mask_name
            elif mask_data[mask_name]["mask_idx"] == 4:
                effector_mask = mask_data[mask_name]["masks"]
                effector_name = mask_name
            elif mask_data[mask_name]["mask_idx"] == 5:
                object_mask = mask_data[mask_name]["masks"]
                object_name = mask_name
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_dict = self.meta_info[idx]
        video_path = video_dict.get("video_path", video_dict.get("video"))
        if video_path is None:
            raise KeyError("Metadata entry is missing 'video'.")
        video_name = os.path.basename(video_path).split(".")[0]
        print(f"Loading video: {video_name}")

        # 2D data
        rgb_list, full_video_path = self.load_video(video_path)
        full_num_frames = len(rgb_list)
        sample_indices = self.get_sample_indices(full_num_frames)
        rgb_list = [rgb_list[i].copy() for i in sample_indices]

        camera_extrinsics, camera_intrinsics = self.load_camera(
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
            cropped_top_left, cropped_bottom_right = self.compute_crop_coordinates(
                rgb_list[0]
            )
            camera_intrinsics = self.compute_cropped_intrinsics(
                camera_intrinsics, cropped_top_left
            )
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
        geometry_data = None
        if self.load_mesh_data_enabled:
            geometry_data = self.load_mesh_data(
                geometry_path,
                transformation_path,
                part_annotation_path,
                function_instance_id,
                has_object,
            )

        receptor_articulation = None
        effector_articulation = None
        if self.load_articulation_enabled:
            receptor_articulation, effector_articulation = self.load_articulation(
                video_dict.get("articulation_path", video_dict.get("articulation")),
                geometry_data,
                part_annotation_path=part_annotation_path,
                function_instance_id=function_instance_id,
                transformation_path=transformation_path,
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
            "sample_indices": sample_indices,
            "num_total_frames": int(full_num_frames)
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

    def load_camera(self, camera_path: str) -> Tuple[np.ndarray, np.ndarray]:
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

        if camera_extrinsics.ndim != 3 or camera_extrinsics.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected camera extrinsics shaped (N, 4, 4), got {camera_extrinsics.shape}."
            )
        if camera_intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected camera intrinsics shaped (3, 3), got {camera_intrinsics.shape}."
            )
        return camera_extrinsics, camera_intrinsics

    def load_from_hdf5(self, filepath: str) -> dict:
        """Load the HDF5 file back into the original dict format."""
        data = {}
        with h5py.File(filepath, 'r') as f:
            for name in f:
                grp = f[name]
                data[name] = {
                    'mask_idx': int(grp.attrs['mask_idx']),
                    'masks': grp['masks'][:]
                }
        return data

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
                mask_id = group.attrs.get("mask_idx", group.attrs.get("id"))
                role = role_by_id.get(int(mask_id)) if mask_id is not None else None
                normalized_group_name = group_name.lower()
                if role is None and normalized_group_name in role_by_id.values():
                    role = normalized_group_name
                if role is None:
                    continue

                dataset_name = "masks" if "masks" in group else "mask" if "mask" in group else None
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
    ) -> Tuple[Any, Dict[str, dict]]:
        """Load the receptor/effector records without loading the mesh itself."""
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
        role_annotations = {}
        for role, role_prefix in (("receptor", "r"), ("effector", "e")):
            label_prefix = f"{role_prefix}{instance_id}_"
            matches = [
                annotation for annotation in annotations
                if str(annotation.get("label", "")).startswith(label_prefix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one part with prefix '{label_prefix}' in "
                    f"{part_annotation_path}, found {len(matches)}."
                )
            role_annotations[role] = matches[0]
        return annotation_data, role_annotations

    def load_mesh_data(self, geometry_path: str, transformation_path: str, part_annotation_path: str, function_instance_id: int, has_object: bool) -> dict:
        required_paths = {
            "mesh": geometry_path,
            "object transformation": transformation_path,
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

        canonical2world = self.load_object_transformation(transformation_path)
        if canonical2world.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 canonical-to-world transform, got {canonical2world.shape}."
            )

        full_vertices = np.asarray(full_mesh.vertices)
        full_triangles = np.asarray(full_mesh.triangles)

        def annotated_part_mesh(annotation: dict) -> o3d.geometry.TriangleMesh:
            triangle_indices = annotation.get("triIndices")
            if triangle_indices is not None:
                triangle_indices = np.asarray(triangle_indices, dtype=np.int64)
                if (triangle_indices.size == 0 or triangle_indices.min() < 0
                        or triangle_indices.max() >= len(full_triangles)):
                    raise ValueError(f"Invalid triangle indices for part '{annotation.get('label')}'.")
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

        def sample_world_points(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError("Cannot sample an empty annotated part mesh.")
            points = np.asarray(
                mesh.sample_points_uniformly(number_of_points=10000).points
            )
            homogeneous_points = np.concatenate(
                [points, np.ones((len(points), 1), dtype=points.dtype)], axis=1
            )
            return (canonical2world @ homogeneous_points.T).T[:, :3]

        geometry_annotations = {"canonical_to_world": canonical2world}
        selected_labels = {}
        for role, annotation in role_annotations.items():
            part_mesh = annotated_part_mesh(annotation)
            pid = annotation.get("pid", annotation.get("partId", annotation.get("objectId")))
            geometry_annotations[role] = {
                "part_pcd": sample_world_points(part_mesh),
                "pid": pid,
            }
            selected_labels[role] = str(annotation["label"])

        if has_object:
            object_points = sample_world_points(full_mesh)
        else:
            object_points = np.concatenate(
                [geometry_annotations[role]["part_pcd"] for role in ("receptor", "effector")],
                axis=0,
            )
        geometry_annotations["object"] = {"part_pcd": object_points, "pid": None}
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

    def load_articulation(
        self,
        articulation_path: str,
        geometry_data: Optional[dict],
        part_annotation_path: Optional[str] = None,
        function_instance_id: Optional[int] = None,
        transformation_path: Optional[str] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        receptor_articulation = None
        effector_articulation = None
        if articulation_path is None:
            return receptor_articulation, effector_articulation

        full_articulation_path = os.path.join(self.root_path, articulation_path)
        if not os.path.exists(full_articulation_path):
            return receptor_articulation, effector_articulation
        with open(full_articulation_path, "r") as f:
            articulation_data = json.load(f)
        joints = (
            articulation_data.get("articulations", [])
            if isinstance(articulation_data, dict)
            else articulation_data
        )
        joints_by_pid = {str(joint.get("pid")): joint for joint in joints}
        if geometry_data is None:
            _, role_annotations = self.load_role_part_annotations(
                part_annotation_path, function_instance_id
            )
            role_geometry_data = {
                role: {
                    "pid": annotation.get(
                        "pid", annotation.get("partId", annotation.get("objectId"))
                    )
                }
                for role, annotation in role_annotations.items()
            }
            canonical2world = (
                self.load_object_transformation(transformation_path)
                if transformation_path is not None else None
            )
        else:
            role_geometry_data = geometry_data
            canonical2world = geometry_data.get("canonical_to_world")

        def joint_for_role(role: str):
            joint = joints_by_pid.get(str(role_geometry_data[role]["pid"]))
            if joint is None:
                return None
            joint = copy.deepcopy(joint)
            if canonical2world is not None:
                canonical2world_array = np.asarray(canonical2world)
                if "axis" in joint:
                    axis = canonical2world_array[:3, :3] @ np.asarray(joint["axis"])
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 0:
                        joint["axis"] = (axis / axis_norm).tolist()
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


def visualize_loaded_point_clouds(
    data: Dict[str, Any],
    roles: Optional[Sequence[str]] = None,
    role_colors: Optional[Mapping[str, Sequence[float]]] = None,
    point_size: float = 3.0,
    show_coordinate_frame: bool = True,
    window_name: str = "Loaded role point clouds",
    window_width: int = 1280,
    window_height: int = 720,
) -> None:
    """Visualize loaded role point clouds together in an Open3D window.

    ``data`` may be either a dataset item returned by ``__getitem__`` or its
    ``geometry_data`` dictionary. By default, every entry containing a
    ``part_pcd`` array is shown. Pass ``roles`` to display a subset.

    Args:
        data: Loaded dataset item or geometry-data dictionary.
        roles: Role names to show, such as ``("receptor", "effector")``.
        role_colors: Optional RGB colors in the range [0, 1], keyed by role.
        point_size: Open3D render point size in pixels.
        show_coordinate_frame: Whether to draw the world-coordinate axes.
        window_name: Title of the Open3D window.
        window_width: Initial window width in pixels.
        window_height: Initial window height in pixels.

    Example:
        >>> sample = dataset[0]
        >>> visualize_loaded_point_clouds(sample)
        >>> visualize_loaded_point_clouds(sample, roles=("receptor", "effector"))
    """
    if "geometry_data" in data:
        geometry_data = data["geometry_data"]
    else:
        geometry_data = data
    if not isinstance(geometry_data, dict):
        raise TypeError("data must be a dataset item or geometry-data dictionary.")

    available_roles = [
        role for role, role_data in geometry_data.items()
        if isinstance(role_data, dict) and "part_pcd" in role_data
    ]
    if isinstance(roles, str):
        selected_roles = [roles]
    else:
        selected_roles = list(roles) if roles is not None else available_roles
    if not selected_roles:
        raise ValueError("No role point clouds were found to visualize.")

    missing_roles = [role for role in selected_roles if role not in available_roles]
    if missing_roles:
        raise ValueError(
            f"No point cloud found for role(s) {missing_roles}. "
            f"Available roles: {available_roles}."
        )
    if point_size <= 0:
        raise ValueError("point_size must be positive.")
    if window_width <= 0 or window_height <= 0:
        raise ValueError("Window dimensions must be positive.")

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
    point_clouds = []
    all_points = []

    print("Point-cloud visualization legend:")
    for role_index, role in enumerate(selected_roles):
        points = np.asarray(geometry_data[role]["part_pcd"])
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Role '{role}' must have a point cloud shaped (N, 3); "
                f"got {points.shape}."
            )
        if len(points) == 0:
            raise ValueError(f"Role '{role}' has an empty point cloud.")
        if not np.isfinite(points).all():
            raise ValueError(f"Role '{role}' contains non-finite point coordinates.")

        color = requested_colors.get(
            role,
            default_colors.get(role, fallback_colors[role_index % len(fallback_colors)]),
        )
        color = np.asarray(color, dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError(f"Color for role '{role}' must contain three finite values.")
        if np.any(color < 0) or np.any(color > 1):
            raise ValueError(f"Color for role '{role}' must be in the range [0, 1].")

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(float, copy=False))
        point_cloud.paint_uniform_color(color)
        point_clouds.append(point_cloud)
        all_points.append(points)
        print(
            f"  {role}: {len(points):,} points, RGB={color.tolist()}, "
            f"min={points.min(axis=0).tolist()}, max={points.max(axis=0).tolist()}"
        )

    geometries = list(point_clouds)
    if show_coordinate_frame:
        combined_points = np.concatenate(all_points, axis=0)
        scene_extent = np.linalg.norm(
            combined_points.max(axis=0) - combined_points.min(axis=0)
        )
        frame_size = max(float(scene_extent) * 0.15, 1e-3)
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
        visualizer.run()
    finally:
        visualizer.destroy_window()


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
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")
