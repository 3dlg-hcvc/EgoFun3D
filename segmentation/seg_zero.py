import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from segmentation.prompt_vlm import build_vlm_prompter

from typing import List, Tuple


class SegZero:
    def __init__(self, reasoning_model_path: str, segmentation_model_path: str, segment_judge_config: dict, max_query: int = 10, device: str = "cuda"):
        self.device = device
        #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)
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
        self.max_query = max_query

        self.seg_judge = build_vlm_prompter(segment_judge_config)


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
    

    def segment_image(self, image: PILImage.Image, part_description: str) -> Tuple[np.ndarray, dict]:
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


    def segment_video(self, video_frame_list: list[PILImage.Image], part_description: str) -> Tuple[List[np.ndarray], List[dict], List[int]]:
        mask_list = []
        answer_dict_list = []
        valid_frame_ids = []
        for frame_id, frame in enumerate(video_frame_list):
            print(f"Segmenting frame {frame_id} ...")
            seg_query_count = 0
            answer_dict = None
            while answer_dict is None and seg_query_count < self.max_query:
                mask, answer_dict = self.segment_image(frame, part_description)
                seg_query_count += 1
            if answer_dict is None:
                print(f"Warning: Segmentation model failed for frame {frame_id} in video. Skipping this frame.")
                continue
            else:
                valid_frame_ids.append(frame_id)
                answer_dict_list.append(answer_dict)
                mask_list.append(mask)
        return mask_list, answer_dict_list, valid_frame_ids


def build_refseg_model(segmentation_config: dict) -> SegZero:
    if segmentation_config.model == "SegZero":
        return SegZero(
            reasoning_model_path=segmentation_config["reasoning_model_path"],
            segmentation_model_path=segmentation_config["segmentation_model_path"],
            segment_judge_config=segmentation_config["vlm_judge"],
            moge_model_path=segmentation_config["moge_model_path"],
            device=segmentation_config.get("device", "cuda")
        )
    else:
        raise ValueError(f"Unsupported segmentation model: {segmentation_config.model}")