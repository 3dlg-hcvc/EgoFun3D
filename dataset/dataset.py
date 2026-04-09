import json
import h5py
import numpy as np
import os
import point_cloud_utils as pcu
import open3d as o3d
from torch.utils.data import Dataset
import imageio

from typing import Tuple, List, Dict, Any


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
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    