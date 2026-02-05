import json
import numpy as np
import os
import point_cloud_utils as pcu
import open3d as o3d
from PIL import Image as PILImage
import gzip
import pickle
import glob
from torch.utils.data import Dataset

from typing import Tuple, List, Dict, Any


# class BaseDataset:
#     def __init__(self, root_path: str, meta_file_path: str):
#         self.root_path = root_path
#         with open(meta_file_path, "r") as f:
#             self.metadata = json.load(f)
#         # self.test_clips = self.metadata["geometry"] + self.metadata["illumination"] + self.metadata["thermal"] + self.metadata["liquid"]
#         self.data = []
    
#     def __iter__(self):
#         """Return an iterator object."""
#         self._index = 0
#         return self

#     def __next__(self):
#         """Return the next element in the iteration."""
#         if self._index < len(self.data):
#             item = self.data[self._index]
#             self._index += 1
#             return item
#         else:
#             raise StopIteration

#     def __len__(self):
#         """Optional: define length to support len(dataset)."""
#         return len(self.data)


# class OpenFunGraphDataset(BaseDataset):
#     def __init__(self, root_path: str, meta_file_path: str):
#         super().__init__(root_path, meta_file_path)
#         for func_type in self.metadata.keys():
#             clip_list = self.metadata[func_type]
#             for clip in clip_list:
#                 print(f"Loading clip: {clip}")
#                 scene_name, seg_id = clip.split("-")
#                 ego_video_path = os.path.join(self.root_path, scene_name, f"seg{seg_id}.MP4")
#                 ego_video_frame_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}")
#                 ego_video_path_list = glob.glob(os.path.join(ego_video_frame_dir, "*.jpg"))
#                 ego_video_path_list.sort()
#                 ego_video_depth_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}_complete_depth")
#                 ego_depth_path_list = glob.glob(os.path.join(ego_video_depth_dir, "*.npy"))
#                 ego_depth_path_list.sort()
#                 ego_video_camera_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}_camera")
#                 ego_camera_path_list = glob.glob(os.path.join(ego_video_camera_dir, "*.json"))
#                 ego_camera_path_list.sort()
#                 ego_video_seg_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}_seg_annotation")
#                 ego_seg_path_list = glob.glob(os.path.join(ego_video_seg_dir, "*.npy"))
#                 ego_seg_path_list.sort()
#                 rgb_list = []
#                 rgb_path_list = []
#                 depth_list = []
#                 camera_list = []
#                 gt_receiver_mask_list = []
#                 gt_effector_mask_list = []
#                 for depth_id, frame_path in enumerate(ego_depth_path_list):
#                     frame_name = os.path.basename(frame_path)
#                     frame_idx = int(os.path.splitext(frame_name)[0])
#                     image = PILImage.open(ego_video_path_list[frame_idx])
#                     rgb_path_list.append(ego_video_path_list[frame_idx])
#                     image = image.convert("RGB")
#                     rgb_list.append(image)
#                     depth = np.load(frame_path)
#                     depth_list.append(depth)
#                     with open(ego_camera_path_list[depth_id], "r") as f:
#                         camera = json.load(f)
#                     camera["intrinsics"] = np.array(camera["intrinsics"])
#                     camera["extrinsics"] = np.array(camera["extrinsics"])
#                     camera_list.append(camera)
#                     seg_mask = np.load(ego_seg_path_list[frame_idx])
#                     gt_receiver_mask = (seg_mask == 2)
#                     gt_effector_mask = (seg_mask == 3)
#                     gt_receiver_mask_list.append(gt_receiver_mask)
#                     gt_effector_mask_list.append(gt_effector_mask)

#                 part_annotation_path = os.path.join(self.root_path, scene_name, "refined_annotations.json")
#                 full_pcd_path = os.path.join(self.root_path, scene_name, f"{scene_name}.ply")
#                 gt_pcd_annotation = self.get_gt_annotation_openfungraph(full_pcd_path, part_annotation_path, seg_id)
#                 data_dict = {
#                     "func_type": func_type,
#                     "scene_name": scene_name,
#                     "seg_id": seg_id,
#                     "ego_video_path": ego_video_path,
#                     "ego_video_rgb_list": rgb_list,
#                     "ego_video_rgb_path_list": rgb_path_list,
#                     "ego_video_depth_list": depth_list,
#                     "ego_video_camera_list": camera_list,
#                     "gt_receiver_mask_list": gt_receiver_mask_list,
#                     "gt_effector_mask_list": gt_effector_mask_list,
#                     "gt_pcd_annotation": gt_pcd_annotation
#                 }
#                 self.data.append(data_dict)


#     def get_gt_annotation_openfungraph(self, full_pcd_path: str, part_annotation_path: str, seg_id: str) -> dict:
#         """
#         Extract ground truth part annotation.

#         Args:
#             full_pcd_path (str): Path to the full point cloud .npy file.
#             part_annotation_path (str): Path to the part annotation .json file.
#             seg_id (str): Segment ID to extract the part from.

#         Returns:
#             dict: Extracted ground truth annotations for the specified segment ID.
#         """
#         full_pcd = pcu.load_mesh_v(full_pcd_path, np.float32)  # (N, 3)
#         with open(part_annotation_path, "r") as f:
#             annotations = json.load(f)
#         if seg_id not in annotations:
#             raise ValueError(f"Segment ID {seg_id} not found in annotations.")
#         gt_annotations = {}
#         for role in ["receiver", "effector"]:
#             print(annotations[seg_id].keys())
#             part_name = annotations[seg_id][role]["label"]
#             part_indices = annotations[seg_id][role]["indices"]
#             if not part_indices:
#                 raise ValueError(f"No indices found for role {role} in segment {seg_id}.")
#             part_pcd = full_pcd[part_indices]
#             gt_annotations[role] = {
#                 "part_name": part_name,
#                 "part_pcd": part_pcd
#             }
#         gt_annotations["relation"] = annotations[seg_id]["description"]
#         return gt_annotations
    

# class iPhoneDataset(BaseDataset):
#     def __init__(self, root_path: str, meta_file_path: str):
#         super().__init__(root_path, meta_file_path)
#         for func_type in self.metadata.keys():
#             clip_list = self.metadata[func_type]
#             for clip in clip_list:
#                 print(f"Loading clip: {clip}")
#                 scene_name, seg_id = clip.split("-")
#                 ego_video_path = os.path.join(self.root_path, scene_name, f"seg{seg_id}", f"seg{seg_id}.mp4")
#                 ego_video_frame_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}", "valid_rgbd_sampled")
#                 ego_video_path_list = glob.glob(os.path.join(ego_video_frame_dir, "*.jpg"))
#                 ego_video_path_list.sort()
#                 ego_video_depth_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}", f"valid_prompt_depth")
#                 ego_depth_path_list = glob.glob(os.path.join(ego_video_depth_dir, "*.npy"))
#                 ego_depth_path_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
#                 ego_video_camera_file = os.path.join(self.root_path, scene_name, f"seg{seg_id}", "valid_ego_camera2world.json")
#                 with open(ego_video_camera_file, "r") as f:
#                     ego_camera_data = json.load(f)
#                 ego_camera_intrinsics_file = os.path.join(self.root_path, scene_name, f"seg{seg_id}", "intrinsics.json")
#                 with open(ego_camera_intrinsics_file, "r") as f:
#                     ego_camera_intrinsics = json.load(f)
#                 intrinsics = np.array(ego_camera_intrinsics["K"])
#                 ego_video_seg_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}", f"valid_seg{seg_id}_seg_annotation")
#                 ego_seg_path_list = glob.glob(os.path.join(ego_video_seg_dir, "*.npy"))
#                 ego_seg_path_list.sort()
#                 rgb_list = []
#                 rgb_path_list = []
#                 depth_list = []
#                 camera_list = []
#                 gt_receiver_mask_list = []
#                 gt_effector_mask_list = []
#                 for frame_id, frame_path in enumerate(ego_video_path_list):
#                     frame_name = os.path.basename(frame_path)
#                     frame_idx = int(os.path.splitext(frame_name)[0])
#                     image = PILImage.open(frame_path)
#                     rgb_path_list.append(frame_path)
#                     image = image.convert("RGB")
#                     rgb_list.append(image)
#                     depth = np.load(ego_depth_path_list[frame_idx])
#                     depth_list.append(depth)
#                     camera = {"intrinsics": intrinsics, "extrinsics": np.array(ego_camera_data[str(frame_idx)])}
#                     camera_list.append(camera)
#                     seg_mask = np.load(ego_seg_path_list[frame_idx])
#                     gt_receiver_mask = (seg_mask == 2)
#                     gt_effector_mask = (seg_mask == 3)
#                     gt_receiver_mask_list.append(gt_receiver_mask)
#                     gt_effector_mask_list.append(gt_effector_mask)
                
#                 part_annotation_path = os.path.join(self.root_path, scene_name, "part_annotation_singleobj.json")
#                 full_pcd_path = os.path.join(self.root_path, scene_name, "mesh_aligned.ply")
#                 gt_pcd_annotation = self.get_gt_annotation_openfungraph(full_pcd_path, part_annotation_path, seg_id)
#                 data_dict = {
#                     "func_type": func_type,
#                     "scene_name": scene_name,
#                     "seg_id": seg_id,
#                     "ego_video_path": ego_video_path,
#                     "ego_video_rgb_list": rgb_list,
#                     "ego_video_rgb_path_list": rgb_path_list,
#                     "ego_video_depth_list": depth_list,
#                     "ego_video_camera_list": camera_list,
#                     "gt_receiver_mask_list": gt_receiver_mask_list,
#                     "gt_effector_mask_list": gt_effector_mask_list,
#                     "gt_pcd_annotation": gt_pcd_annotation
#                 }
#                 self.data.append(data_dict)


#     def get_gt_annotation_openfungraph(self, full_mesh_path: str, part_annotation_path: str, seg_id: str) -> dict:
#         """
#         Extract ground truth part annotation.

#         Args:
#             full_mesh_path (str): Path to the full mesh .ply file.
#             part_annotation_path (str): Path to the part annotation .json file.
#             seg_id (str): Segment ID to extract the part from.

#         Returns:
#             dict: Extracted ground truth annotations for the specified segment ID.
#         """
#         full_mesh = o3d.io.read_triangle_mesh(full_mesh_path)
#         with open(part_annotation_path, "r") as f:
#             annotations = json.load(f)
#         if seg_id not in annotations:
#             raise ValueError(f"Segment ID {seg_id} not found in annotations.")
#         gt_annotations = {}
#         for role in ["receiver", "effector"]:
#             print(annotations[seg_id].keys())
#             part_name = annotations[seg_id][role]["label"]
#             part_indices = annotations[seg_id][role]["indices"]
#             if not part_indices:
#                 raise ValueError(f"No indices found for role {role} in segment {seg_id}.")
#             part_mesh = full_mesh.select_by_index(part_indices)
#             part_pcd = part_mesh.sample_points_uniformly(number_of_points=10000)
#             part_pcd_np = np.asarray(part_pcd.points)
#             gt_annotations[role] = {
#                 "part_name": part_name,
#                 "part_pcd": part_pcd_np
#             }
#         gt_annotations["relation"] = annotations[seg_id]["description"]
#         return gt_annotations
    

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
            frame_dir = video_dict["cropped_frame_dir"]
            crop = True
        else:
            frame_dir = video_dict["original_frame_dir"]
            crop = False
        # rgb
        rgb_list, rgb_path_list = self.load_video(frame_dir)
        sample_indices = self.get_sample_indices(len(rgb_list))
        rgb_list = [rgb_list[i] for i in sample_indices]
        rgb_path_list = [rgb_path_list[i] for i in sample_indices]

        camera_extrinsics, camera_intrinsics, cropped_top_left, cropped_bottom_right = self.load_camera(
            video_dict["camera_extrinsics_path"],
            video_dict["camera_intrinsics_path"],
            crop
        )
        camera_extrinsics = camera_extrinsics[sample_indices]

        receiver_mask_list, effector_mask_list, object_mask_list, receiver_name, effector_name, object_name = self.load_2d_masks(
            video_dict["video_mask_path"]
        )
        receiver_mask_list = receiver_mask_list[sample_indices, cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]]
        effector_mask_list = effector_mask_list[sample_indices, cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]]
        object_mask_list = object_mask_list[sample_indices, cropped_top_left[1]:cropped_bottom_right[1], cropped_top_left[0]:cropped_bottom_right[0]]

        geometry_data = self.load_part_point_cloud(
            video_dict["geometry_type"],
            video_dict["geometry_path"],
            video_dict["part_annotation_path"],
            video_dict["function_instance_id"]
        )

        receiver_articulation, effector_articulation = self.load_articulation(
            video_dict["articulation_path"],
            geometry_data
        )

        function_annotation = self.load_function_annotation(
            video_dict["function_annotation_path"],
            video_dict["function_instance_id"]
        )

        data_dict = {
            "video_name": video_name,
            "rgb_list": rgb_list,
            "rgb_path_list": rgb_path_list,
            "camera_extrinsics": camera_extrinsics,
            "camera_intrinsics": camera_intrinsics,
            "receiver_mask_list": receiver_mask_list,
            "effector_mask_list": effector_mask_list,
            "object_mask_list": object_mask_list,
            "receiver_name": receiver_name,
            "effector_name": effector_name,
            "object_name": object_name,
            "geometry_data": geometry_data,
            "receiver_articulation": receiver_articulation,
            "effector_articulation": effector_articulation,
            "function_annotation": function_annotation,
            "sample_indices": sample_indices
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

    def load_video(self, frame_dir: str) -> Tuple[List[PILImage.Image], List[str]]:
        # video_path = os.path.join(self.root_path, video_dict["video_path"])
        video_frame_dir = os.path.join(self.root_path, frame_dir)
        rgb_path_list = glob.glob(os.path.join(video_frame_dir, "*.jpg"))
        rgb_path_list.sort()
        rgb_list = []
        for frame_path in rgb_path_list:
            image = PILImage.open(frame_path)
            image = image.convert("RGB")
            rgb_list.append(image)
        return rgb_list, rgb_path_list
    
    def load_camera(self, extrinsics_path: str, intrinsics_path: str, crop: bool) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        camera_extrinsics_path = os.path.join(self.root_path, extrinsics_path)
        camera_extrinsics = np.load(camera_extrinsics_path)  # (N, 4, 4)
        camera_intrinsics_path = os.path.join(self.root_path, intrinsics_path)
        with open(camera_intrinsics_path, "r") as f:
            camera_intrinsics_data = json.load(f)
        if crop:
            camera_intrinsics = np.array(camera_intrinsics_data["cropped_intrinsics"])
            cropped_top_left = camera_intrinsics_data["cropped_top_left"]
            cropped_bottom_right = camera_intrinsics_data["cropped_bottom_right"]
        else:
            camera_intrinsics = np.array(camera_intrinsics_data["undistorted_intrinsics"])
            cropped_top_left = [0, 0]
            cropped_bottom_right = camera_intrinsics_data["origin_frame_size"]
        return camera_extrinsics, camera_intrinsics, cropped_top_left, cropped_bottom_right
    
    def load_2d_masks(self, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        # 2d masks
        mask_path = os.path.join(self.root_path, mask_path)
        with gzip.open(mask_path, "rb") as f:
            mask_data = pickle.load(f)
        receiver_mask = None
        effector_mask = None
        object_mask = None
        receiver_name = None
        effector_name = None
        object_name = None
        for mask_name in mask_data.keys():
            if mask_data[mask_name]["mask_idx"] == 3:
                receiver_mask = mask_data[mask_name]["masks"]
                receiver_name = mask_name
            elif mask_data[mask_name]["mask_idx"] == 4:
                effector_mask = mask_data[mask_name]["masks"]
                effector_name = mask_name
            elif mask_data[mask_name]["mask_idx"] == 5:
                object_mask = mask_data[mask_name]["masks"]
                object_name = mask_name
        return receiver_mask, effector_mask, object_mask, receiver_name, effector_name, object_name
    
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
        if function_instance_id not in annotations:
            raise ValueError(f"Segment ID {function_instance_id} not found in annotations.")
        geometry_annotations = {}
        for role in ["receiver", "effector"]:
            # print(annotations[function_instance_id].keys())
            part_name = annotations[function_instance_id][role]["label"]
            part_indices = annotations[function_instance_id][role]["indices"]
            pid = annotations[function_instance_id][role]["pid"]
            if not part_indices:
                raise ValueError(f"No indices found for role {role} in segment {function_instance_id}.")
            part_pcd = full_pcd[part_indices]
            geometry_annotations[role] = {
                "part_pcd": part_pcd,
                "pid": pid
            }
        geometry_annotations["relation"] = annotations[function_instance_id]["description"]
        return geometry_annotations

    def load_mesh_data(self, geometry_path: str, part_annotation_path: str, function_instance_id: int) -> dict:
        full_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_path, geometry_path))
        with open(os.path.join(self.root_path, part_annotation_path), "r") as f:
            annotations = json.load(f)
        if function_instance_id not in annotations.keys():
            raise ValueError(f"Segment ID {function_instance_id} not found in annotations.")
        geometry_annotations = {}

        for role in ["receiver", "effector"]:
            # print(annotations[function_instance_id].keys())
            part_name = annotations[function_instance_id][role]["label"]
            part_indices = annotations[function_instance_id][role]["indices"]
            pid = annotations[function_instance_id][role]["pid"]
            if not part_indices:
                raise ValueError(f"No indices found for role {role} in segment {function_instance_id}.")
            part_mesh = full_mesh.select_by_index(part_indices)
            part_pcd = part_mesh.sample_points_uniformly(number_of_points=10000)
            part_pcd_np = np.asarray(part_pcd.points)
            geometry_annotations[role] = {
                "part_pcd": part_pcd_np,
                "pid": pid
            }
        geometry_annotations["relation"] = annotations[function_instance_id]["description"]
        return geometry_annotations
    
    def load_articulation(self, articulation_path: str, geometry_data: dict) -> Tuple[dict, dict]:
        receiver_articulation = None
        effector_articulation = None
        articulation_path = os.path.join(self.root_path, articulation_path)
        if os.path.exists(articulation_path):
            with open(articulation_path, "r") as f:
                articulation_data = json.load(f)
            for role in ["receiver", "effector"]:
                pid = geometry_data[role]["pid"]
                for joint_data in articulation_data:
                    if joint_data["pid"] == pid:
                        if role == "receiver":
                            receiver_articulation = joint_data
                        elif role == "effector":
                            effector_articulation = joint_data
        return receiver_articulation, effector_articulation
    
    def load_function_annotation(self, function_annotation_path: str, function_instance_id: str) -> dict:
        function_annotation_path = os.path.join(self.root_path, function_annotation_path)
        with open(function_annotation_path, "r") as f:
            function_data = json.load(f)
        function_annotation = function_data[function_instance_id]
        return function_annotation


def build_dataset(dataset_config: dict) -> Dataset:
    dataset_name = dataset_config["name"]
    # if dataset_name == "OpenFunGraph":
    #     return OpenFunGraphDataset(
    #         root_path=dataset_config["root_path"],
    #         meta_file_path=dataset_config["meta_file"]
    #     )
    # elif dataset_name == "iPhone":
    #     return iPhoneDataset(
    #         root_path=dataset_config["root_path"],
    #         meta_file_path=dataset_config["meta_file"]
    #     )
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

    