import json
import numpy as np
import os
import point_cloud_utils as pcu
from PIL import Image as PILImage
import glob


class BaseDataset:
    def __init__(self, root_path: str, meta_file_path: str):
        self.root_path = root_path
        with open(meta_file_path, "r") as f:
            self.metadata = json.load(f)
        # self.test_clips = self.metadata["geometry"] + self.metadata["illumination"] + self.metadata["thermal"] + self.metadata["liquid"]
        self.data = []
    
    def __iter__(self):
        """Return an iterator object."""
        self._index = 0
        return self

    def __next__(self):
        """Return the next element in the iteration."""
        if self._index < len(self.data):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __len__(self):
        """Optional: define length to support len(dataset)."""
        return len(self.data)


class OpenFunGraphDataset(BaseDataset):
    def __init__(self, root_path: str, meta_file_path: str):
        super().__init__(root_path, meta_file_path)
        for func_type in self.metadata.keys():
            clip_list = self.metadata[func_type]
            for clip in clip_list:
                scene_name, seg_id = clip.split("-")
                ego_video_path = os.path.join(self.root_path, scene_name, f"seg{seg_id}.MP4")
                ego_video_frame_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}")
                ego_video_path_list = glob.glob(os.path.join(ego_video_frame_dir, "*.jpg"))
                ego_video_path_list.sort()
                ego_video_depth_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}_depth")
                ego_depth_path_list = glob.glob(os.path.join(ego_video_depth_dir, "*.npy"))
                ego_depth_path_list.sort()
                ego_video_camera_dir = os.path.join(self.root_path, scene_name, f"seg{seg_id}_camera")
                ego_camera_path_list = glob.glob(os.path.join(ego_video_camera_dir, "*.json"))
                ego_camera_path_list.sort()
                rgb_list = []
                depth_list = []
                camera_list = []
                for depth_id, frame_path in enumerate(ego_depth_path_list):
                    frame_name = os.path.basename(frame_path)
                    frame_idx = int(os.path.splitext(frame_name)[0])
                    image = PILImage.open(ego_video_path_list[frame_idx])
                    image = image.convert("RGB")
                    rgb_list.append(image)
                    depth = np.load(frame_path)
                    depth_list.append(depth)
                    with open(ego_camera_path_list[depth_id], "r") as f:
                        camera = json.load(f)
                    camera["intrinsics"] = np.array(camera["intrinsics"])
                    camera["extrinsics"] = np.array(camera["extrinsics"])
                    camera_list.append(camera)

                part_annotation_path = os.path.join(self.root_path, scene_name, "refined_annotations.json")
                full_pcd_path = os.path.join(self.root_path, scene_name, f"{scene_name}.ply")
                gt_annotation = self.get_gt_annotation_openfungraph(full_pcd_path, part_annotation_path, seg_id)
                data_dict = {
                    "func_type": func_type,
                    "scene_name": scene_name,
                    "seg_id": seg_id,
                    "ego_video_path": ego_video_path,
                    "ego_video_rgb_list": rgb_list,
                    "ego_video_depth_list": depth_list,
                    "ego_video_camera_list": camera_list,
                    "gt_annotation": gt_annotation
                }
                self.data.append(data_dict)


    def get_gt_annotation_openfungraph(self, full_pcd_path: str, part_annotation_path: str, seg_id: str) -> dict:
        """
        Extract ground truth part annotation.

        Args:
            full_pcd_path (str): Path to the full point cloud .npy file.
            part_annotation_path (str): Path to the part annotation .json file.
            seg_id (str): Segment ID to extract the part from.

        Returns:
            dict: Extracted ground truth annotations for the specified segment ID.
        """
        full_pcd = pcu.load_mesh_v(full_pcd_path, np.float32)  # (N, 3)
        with open(part_annotation_path, "r") as f:
            annotations = json.load(f)
        if seg_id not in annotations:
            raise ValueError(f"Segment ID {seg_id} not found in annotations.")
        gt_annotations = {}
        for role in ["receiver", "effector"]:
            part_name = annotations[seg_id][role]["label"]
            part_indices = annotations[seg_id][role]["indices"]
            if not part_indices:
                raise ValueError(f"No indices found for role {role} in segment {seg_id}.")
            part_pcd = full_pcd[part_indices]
            gt_annotations[role] = {
                "part_name": part_name,
                "part_pcd": part_pcd
            }
        gt_annotations["relation"] = annotations[seg_id]["description"]
        return gt_annotations


def build_dataset(dataset_config: dict) -> BaseDataset:
    dataset_name = dataset_config["name"]
    if dataset_name == "OpenFunGraph":
        return OpenFunGraphDataset(
            root_path=dataset_config["root_path"],
            meta_file_path=dataset_config["meta_file"]
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    