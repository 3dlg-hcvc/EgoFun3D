import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from torchvision.transforms.functional import pil_to_tensor
import json
from PIL import Image as PILImage
import open3d as o3d
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
from moge.model.v2 import MoGeModel # Let's try MoGe-2
from romatch import roma_indoor
import utils3d

from typing import Tuple


class SegZero:
    def __init__(self, reasoning_model_path: str, segmentation_model_path: str, moge_model_path: str, device: str = "cuda"):
        self.device = device
        #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.segmentation_model = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
        self.reasoning_model.eval()
        # default processor
        self.processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

        self.question_template = \
            "Please find \"{Question}\" with bboxs and points." \
            "Compare the difference between object(s) and find the most closely matched object(s)." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"

        self.monocular_model = MoGeModel.from_pretrained(moge_model_path).to(device)
        self.feature_matching_model = roma_indoor(device=device)


    def extract_bbox_points_think(self, output_text, x_factor, y_factor):
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [[
                int(item['bbox_2d'][0] * x_factor + 0.5),
                int(item['bbox_2d'][1] * y_factor + 0.5),
                int(item['bbox_2d'][2] * x_factor + 0.5),
                int(item['bbox_2d'][3] * y_factor + 0.5)
            ] for item in data]
            pred_points = [[
                int(item['point_2d'][0] * x_factor + 0.5),
                int(item['point_2d'][1] * y_factor + 0.5)
            ] for item in data]
        else:
            pred_bboxes = None
            pred_points = None
        
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        think_text = ""
        if think_match:
            think_text = think_match.group(1)
        
        return pred_bboxes, pred_points, think_text
    

    def segment(self, image: PILImage.Image, part_description: str) -> Tuple[np.ndarray, dict]:
        original_width, original_height = image.size
        resize_size = 1080
        x_factor, y_factor = original_width/resize_size, original_height/resize_size
        
        messages = []
        message = [{
            "role": "user",
            "content": [
            {
                "type": "image", 
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
            },
            {   
                "type": "text",
                "text": self.question_template.format(
                    Question=part_description.lower().strip("."),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )    
            }
        ]
        }]
        messages.append(message)

        # Preparation for inference
        text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        
        #pdb.set_trace()
        image_inputs, video_inputs = process_vision_info(messages)
        #pdb.set_trace()
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Inference: Generation of the output
        generated_ids = self.reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(output_text[0])
        # pdb.set_trace()
        bboxes, points, think = self.extract_bbox_points_think(output_text[0], x_factor, y_factor)
        if bboxes is None or points is None:
            print("Error in parsing segmentation output.")
            return np.zeros((original_height, original_width), dtype=bool), None
        answer_dict = {"points": points, "thinking": think}
        print(points, len(points))
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            mask_all = np.zeros((image.height, image.width), dtype=bool)
            self.segmentation_model.set_image(image)
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = self.segmentation_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                mask = masks[0].astype(bool)
                mask_all = np.logical_or(mask_all, mask)

        return mask_all, answer_dict


    def get_part_pcd(self, image: PILImage.Image, part_mask: np.ndarray, cam_pose: np.ndarray, gt_depth: np.ndarray = None, gt_intrinsics: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        if gt_depth is not None and gt_intrinsics is not None:
            # points_map = utils3d.np.depth_map_to_point_map(gt_depth, gt_intrinsics, np.eye(4))
            points_map = self.depth2xyz(gt_depth, gt_intrinsics, cam_type="opencv")
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
        warp, certainty = self.feature_matching_model.match(current_image_path, anchor_image_path, device=self.device)
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
        current_part_3dkpts = current_point_map[kptsA[:,1], kptsA[:,0]]
        anchor_part_3dkpts = anchor_point_map[kptsB[:,1], kptsB[:,0]]
        if len(current_part_3dkpts) < 10 or len(anchor_part_3dkpts) < 10:
            print("Not enough keypoints for transformation estimation.")
            return np.eye(4)
        # Estimate transformation
        current2anchor = self.estimate_se3_transformation(anchor_part_3dkpts, current_part_3dkpts)
        return current2anchor
    

    def estimate_se3_transformation(self, target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_xyz)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
        correspondences = np.arange(source_xyz.shape[0])
        correspondences = np.vstack([correspondences, correspondences], dtype=np.int32).T
        p2p_registration = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
        source2target = p2p_registration.compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(correspondences))
        return source2target
    

    def depth2xyz(self, depth_image: np.ndarray, intrinsics: np.ndarray, cam_type: str) -> np.ndarray:
        # Get the shape of the depth image
        H, W = depth_image.shape

        # Create meshgrid for pixel coordinates (u, v)
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Flatten the grid to a 1D array of pixel coordinates
        u = u.flatten()
        v = v.flatten()

        # Flatten the depth image to a 1D array of depth values
        if depth_image.dtype == np.uint16:
            depth = depth_image.flatten() / 1000.0
        else:
            depth = depth_image.flatten()

        # Camera intrinsic matrix (3x3)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Calculate the 3D coordinates (x, y, z) from depth
        # Use the formula:
        #   X = (u - cx) * depth / fx
        #   Y = (v - cy) * depth / fy
        #   Z = depth
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Stack the x, y, z values into a 3D point cloud
        point_cloud = np.vstack((x, y, z)).T

        if cam_type == "opengl":
            point_cloud = point_cloud * np.array([1, -1, -1])

        # Reshape the point cloud to the original depth image shape [H, W, 3]
        point_cloud = point_cloud.reshape(H, W, 3)

        return point_cloud


    def fuse_part_pcds(self, part_pcd_list: list[np.ndarray], transformation_list: list[np.ndarray]) -> np.ndarray:
        fused_part_pcd = []
        for part_pcd, transformation in zip(part_pcd_list, transformation_list):
            part_pcd_anchored = part_pcd @ transformation[:3, :3].T + transformation[:3, 3:].T
            fused_part_pcd.append(part_pcd_anchored)
        fused_part_pcd = np.concatenate(fused_part_pcd, axis=0)
        return fused_part_pcd
    

def build_refseg_model(segmentation_config: dict) -> SegZero:
    if segmentation_config.model == "SegZero":
        return SegZero(
            reasoning_model_path=segmentation_config["reasoning_model_path"],
            segmentation_model_path=segmentation_config["segmentation_model_path"],
            moge_model_path=segmentation_config["moge_model_path"],
            device=segmentation_config.get("device", "cuda")
        )
    else:
        raise ValueError(f"Unsupported segmentation model: {segmentation_config.model}")