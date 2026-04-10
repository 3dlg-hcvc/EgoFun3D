import os
import json
import re
import shutil
import sys
import tempfile
from functools import partial
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from VLM.prompt_vlm import build_vlm_prompter


def _build_sam2_image_predictor(model_id: str, device: str):
    import sam2
    from sam2.build_sam import _hf_download, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config_name, ckpt_path = _hf_download(model_id)
    if config_name.startswith("configs/"):
        config_name = config_name[len("configs/") :]

    config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        sam2_model = build_sam2(config_name, ckpt_path, device=device)
    return SAM2ImagePredictor(sam2_model)


class RefSeg:
    def __init__(self):
        pass

    def segment_video(self, video_frame_list: List[PILImage.Image] | List[str], part_description: str) -> Tuple[List[np.ndarray], List[dict], List[int]]:
        pass

    def set_debug_context(
        self,
        save_dir: str | None,
        scene_name: str | None,
        seg_id: str | None,
        role: str | None,
        prompt: str | None,
    ) -> None:
        return


class SegZero(RefSeg):
    def __init__(
        self,
        reasoning_model_path: str,
        segmentation_model_path: str,
        segment_judge_config: dict | None,
        max_query: int = 10,
        device: str = "cuda",
        sam_backend: str = "sam2",
        sam3_confidence_threshold: float = 0.5,
    ):
        self.device = device
        self.sam_backend = str(sam_backend).lower()
        if self.sam_backend not in {"sam2", "sam3"}:
            raise ValueError(
                f"Unsupported SegZero sam_backend={sam_backend}. Expected one of: sam2, sam3."
            )
        #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.segmentation_model = None
        self.sam3_processor = None
        if self.sam_backend == "sam2":
            self.segmentation_model = _build_sam2_image_predictor(segmentation_model_path, device)
        else:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bpe_path = os.path.join(repo_root, "bpe_simple_vocab_16e6.txt.gz")
            if not os.path.exists(bpe_path):
                raise FileNotFoundError(
                    f"SAM3 BPE vocab not found at {bpe_path}. "
                    "Download it from https://github.com/facebookresearch/sam3/raw/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                )
            sam3_model = build_sam3_image_model(bpe_path=bpe_path, device=device)
            self.sam3_processor = Sam3Processor(
                sam3_model,
                confidence_threshold=float(sam3_confidence_threshold),
                device=device,
            )
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

        self.seg_judge = build_vlm_prompter(segment_judge_config) if segment_judge_config is not None else None


    def extract_bbox_points_think(self, output_text, x_factor, y_factor):
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        data = None
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                data = None
        if data is not None:
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
            pred_bboxes = []
            pred_points = []
            fallback_pattern = r'\{"bbox_2d"\s*:\s*\[([^\]]+)\]\s*,\s*"point_2d"\s*:\s*\[([^\]]+)\]\s*\}'
            for bbox_str, point_str in re.findall(fallback_pattern, output_text):
                bbox_vals = [v.strip() for v in bbox_str.split(",") if v.strip()]
                point_vals = [v.strip() for v in point_str.split(",") if v.strip()]
                if len(bbox_vals) < 4 or len(point_vals) < 2:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox_vals[:4]]
                    px, py = [float(v) for v in point_vals[:2]]
                except ValueError:
                    continue
                pred_bboxes.append([
                    int(x1 * x_factor + 0.5),
                    int(y1 * y_factor + 0.5),
                    int(x2 * x_factor + 0.5),
                    int(y2 * y_factor + 0.5),
                ])
                pred_points.append([
                    int(px * x_factor + 0.5),
                    int(py * y_factor + 0.5),
                ])
            if len(pred_bboxes) == 0 or len(pred_points) == 0:
                pred_bboxes = None
                pred_points = None
        
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        think_text = ""
        if think_match:
            think_text = think_match.group(1)
        
        return pred_bboxes, pred_points, think_text
    
    def _segment_with_sam3(
        self, image: PILImage.Image, points: list[list[int]]
    ) -> np.ndarray:
        width, height = image.size
        if len(points) == 0:
            return np.zeros((height, width), dtype=bool)

        state = self.sam3_processor.set_image(image)
        if "language_features" not in state["backbone_out"]:
            dummy_text = self.sam3_processor.model.backbone.forward_text(
                ["visual"], device=self.sam3_processor.device
            )
            state["backbone_out"].update(dummy_text)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.sam3_processor.model._get_dummy_prompt()

        norm_points = []
        for x, y in points:
            x_n = min(max(float(x) / width, 0.0), 1.0)
            y_n = min(max(float(y) / height, 0.0), 1.0)
            norm_points.append([x_n, y_n])

        points_tensor = torch.tensor(
            norm_points, device=self.sam3_processor.device, dtype=torch.float32
        ).view(-1, 1, 2)
        labels = torch.ones(
            (points_tensor.shape[0], 1),
            device=self.sam3_processor.device,
            dtype=torch.long,
        )
        state["geometric_prompt"].append_points(points_tensor, labels)
        state = self.sam3_processor._forward_grounding(state)

        masks = state.get("masks")
        if masks is None or masks.numel() == 0:
            return np.zeros((height, width), dtype=bool)
        return masks.squeeze(1).any(dim=0).detach().cpu().numpy().astype(bool)


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
        answer_dict = {"points": points, "thinking": think, "sam_backend": self.sam_backend}
        print(points, len(points))

        if self.sam_backend == "sam2":
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
        else:
            with torch.inference_mode():
                mask_all = self._segment_with_sam3(image, points)

        return mask_all, answer_dict


    def segment_video(self, video_frame_list: List[PILImage.Image], part_description: str) -> Tuple[List[np.ndarray], List[dict], List[int]]:
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
                answer_dict_list.append({"points": [], "thinking": "", "vlm_judge": {}})
                pred_mask = np.zeros((frame.height, frame.width), dtype=bool)
                mask_list.append(pred_mask)
            else:
                if self.seg_judge is None:
                    valid_frame_ids.append(frame_id)
                    answer_dict["vlm_judge"] = {}
                else:
                    vlm_judge_response = self.seg_judge.prompt(frame, mask, part_description)
                    answer_dict["vlm_judge"] = vlm_judge_response if len(vlm_judge_response.keys()) == 2 else {}
                    if len(vlm_judge_response.keys()) == 2 and vlm_judge_response["answer"] == "yes":
                        valid_frame_ids.append(frame_id)
                answer_dict_list.append(answer_dict)
                mask_list.append(mask)
        return mask_list, answer_dict_list, valid_frame_ids


def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def _resolve_torch_dtype(dtype_name: Any) -> torch.dtype | str:
    if dtype_name is None:
        return torch.bfloat16
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    dtype_str = str(dtype_name).lower().strip()
    mapping = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_str in mapping:
        return mapping[dtype_str]
    raise ValueError(
        f"Unsupported Sa2VA torch_dtype={dtype_name}. "
        "Expected one of: auto, bf16, bfloat16, fp16, float16, fp32, float32."
    )


def _infer_sa2va_num_layers(model_name_or_path: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    for attr in ("num_hidden_layers",):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    for nested_attr in ("llm_config", "text_config", "language_config"):
        nested = getattr(cfg, nested_attr, None)
        if nested is not None and hasattr(nested, "num_hidden_layers"):
            return int(getattr(nested, "num_hidden_layers"))
    raise ValueError(
        f"Unable to infer Sa2VA layer count from config at {model_name_or_path}. "
        "Set segmentation.sa2va.use_custom_device_map=false to use device_map='auto'."
    )


def split_model(model_name_or_path: str):
    import math

    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = max(num_gpus // max(world_size, 1), 1)
    num_layers = _infer_sa2va_num_layers(model_name_or_path)
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    denom = max(num_gpus - 0.2, 1.0)
    num_layers_per_gpu = math.ceil(num_layers / denom)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    if num_layers_per_gpu:
        num_layers_per_gpu[0] = max(math.ceil(num_layers_per_gpu[0] * 0.8), 1)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
        if layer_cnt >= num_layers:
            break
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    device_map['grounding_encoder'] = rank
    device_map['text_hidden_fcs'] = rank
    return device_map


class Sa2VA(RefSeg):
    def __init__(
        self,
        model_name: str,
        segment_judge_config: dict | None,
        sa2va_config: dict | None = None,
        max_query: int = 1,
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self.sa2va_config = sa2va_config or {}
        self.max_query = int(self.sa2va_config.get("max_query", max_query))
        self.max_new_tokens = int(self.sa2va_config.get("max_new_tokens", 1024))
        self.union_multi_seg_tokens = bool(self.sa2va_config.get("union_multi_seg_tokens", True))
        self.prompt_template = self.sa2va_config.get(
            "prompt_template",
            "<image>Please segment the part in the following description: {PartDescription}.",
        )
        self.torch_dtype = _resolve_torch_dtype(self.sa2va_config.get("torch_dtype", "bfloat16"))
        self.use_custom_device_map = bool(self.sa2va_config.get("use_custom_device_map", False))
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": self.torch_dtype,
        }
        if str(self.device).startswith("cuda"):
            if self.use_custom_device_map and torch.cuda.device_count() > 1:
                model_kwargs["device_map"] = split_model(model_name)
            else:
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": self.device}

        self.model = None
        model_errors: list[str] = []
        for model_cls in (AutoModelForCausalLM, AutoModel):
            try:
                self.model = model_cls.from_pretrained(model_name, **model_kwargs).eval()
                break
            except Exception as exc:
                model_errors.append(f"{model_cls.__name__}: {type(exc).__name__}: {exc}")
        if self.model is None:
            joined = "\n".join(model_errors)
            raise RuntimeError(f"Failed to load Sa2VA model from {model_name}.\n{joined}")

        self.tokenizer = None
        self.processor = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        except Exception:
            self.tokenizer = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.processor = None

        if hasattr(self.model, "generation_config") and self.max_new_tokens > 0:
            try:
                self.model.generation_config.max_new_tokens = self.max_new_tokens
            except Exception:
                pass
        self.seg_judge = build_vlm_prompter(segment_judge_config) if segment_judge_config is not None else None

    def _format_prompt(self, part_description: str) -> str:
        return self.prompt_template.replace("{PartDescription}", part_description)

    @staticmethod
    def _prediction_masks_to_binary_mask(
        prediction_masks: Any,
        height: int,
        width: int,
        union_multi_seg_tokens: bool = True,
    ) -> np.ndarray:
        zero_mask = np.zeros((height, width), dtype=bool)
        if prediction_masks is None:
            return zero_mask

        candidates: list[np.ndarray] = []

        def _collect(node: Any) -> None:
            if node is None:
                return
            if torch.is_tensor(node):
                candidates.append(node.detach().cpu().numpy())
                return
            if isinstance(node, np.ndarray):
                candidates.append(node)
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    _collect(item)

        _collect(prediction_masks)
        if len(candidates) == 0:
            return zero_mask

        merged = zero_mask.copy()
        found = False
        for candidate in candidates:
            arr = np.asarray(candidate)
            if arr.size == 0:
                continue
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                if arr.shape != (height, width):
                    continue
                current = arr > 0
            elif arr.ndim == 3:
                if arr.shape[-2:] != (height, width):
                    continue
                if union_multi_seg_tokens:
                    current = np.any(arr > 0, axis=0)
                else:
                    current = arr[0] > 0
            else:
                continue

            if not found:
                merged = current.astype(bool)
                found = True
            elif union_multi_seg_tokens:
                merged = np.logical_or(merged, current.astype(bool))
        return merged if found else zero_mask

    def _predict_forward(self, frame: PILImage.Image, prompt: str) -> dict:
        payload_base = {
            "image": frame,
            "text": prompt,
            "past_text": "",
            "mask_prompts": None,
        }
        payloads = []
        if self.tokenizer is not None and self.processor is not None:
            payloads.append({**payload_base, "tokenizer": self.tokenizer, "processor": self.processor})
        if self.tokenizer is not None:
            payloads.append({**payload_base, "tokenizer": self.tokenizer})
        if self.processor is not None:
            payloads.append({**payload_base, "processor": self.processor})
        payloads.append(payload_base)

        last_error = None
        for payload in payloads:
            try:
                out = self.model.predict_forward(**payload)
                if isinstance(out, dict):
                    return out
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Sa2VA predict_forward did not return a dictionary output.")

    def _segment_frame(self, frame: PILImage.Image, part_description: str) -> Tuple[np.ndarray, dict]:
        prompt = self._format_prompt(part_description)
        return_dict = self._predict_forward(frame, prompt)
        prediction = str(return_dict.get("prediction", ""))
        prediction_masks = return_dict.get("prediction_masks", None)
        mask = self._prediction_masks_to_binary_mask(
            prediction_masks=prediction_masks,
            height=frame.height,
            width=frame.width,
            union_multi_seg_tokens=self.union_multi_seg_tokens,
        )
        answer_dict = {
            "answer": prediction,
            "prompt": prompt,
            "model": self.model_name,
            "num_mask_tokens": len(prediction_masks) if isinstance(prediction_masks, list) else None,
        }
        if prediction_masks is None:
            answer_dict["error"] = "no_prediction_masks"
        return mask, answer_dict

    def segment_video(
        self, video_frame_list: List[PILImage.Image] | List[str], part_description: str
    ) -> Tuple[List[np.ndarray], List[dict], List[int]]:
        mask_list: list[np.ndarray] = []
        answer_dict_list: list[dict] = []
        valid_frame_ids: list[int] = []

        for frame_id, frame_or_path in enumerate(video_frame_list):
            if isinstance(frame_or_path, str):
                with PILImage.open(frame_or_path) as frame_img:
                    frame = frame_img.convert("RGB")
            else:
                frame = frame_or_path
            mask = np.zeros((frame.height, frame.width), dtype=bool)
            answer_dict = {
                "answer": "",
                "error": "sa2va_failed",
                "prompt": self._format_prompt(part_description),
                "model": self.model_name,
            }

            for _ in range(max(self.max_query, 1)):
                try:
                    mask, answer_dict = self._segment_frame(frame, part_description)
                    break
                except Exception as exc:
                    answer_dict = {
                        "answer": "",
                        "error": f"{type(exc).__name__}: {exc}",
                        "prompt": self._format_prompt(part_description),
                        "model": self.model_name,
                    }

            if self.seg_judge is None:
                if not answer_dict.get("error"):
                    valid_frame_ids.append(frame_id)
                answer_dict["vlm_judge"] = {}
            else:
                if answer_dict.get("error"):
                    answer_dict["vlm_judge"] = {}
                else:
                    vlm_judge_response = self.seg_judge.prompt(frame, mask, part_description)
                    answer_dict["vlm_judge"] = (
                        vlm_judge_response if len(vlm_judge_response.keys()) == 2 else {}
                    )
                    if (
                        len(vlm_judge_response.keys()) == 2
                        and vlm_judge_response.get("answer", "").lower() == "yes"
                    ):
                        valid_frame_ids.append(frame_id)
            answer_dict_list.append(answer_dict)
            mask_list.append(mask.astype(bool))
        return mask_list, answer_dict_list, valid_frame_ids


class XSam(RefSeg):
    def __init__(
        self,
        xsam_config: dict,
        segment_judge_config: dict | None,
        max_query: int = 1,
        device: str = "cuda",
    ):
        self.device = device
        self.max_query = int(xsam_config.get("max_query", max_query))
        self.xsam_code_root = str(
            xsam_config.get(
                "code_root",
                "third_party/X-SAM/xsam",
            )
        )
        self.xsam_config_path = str(
            xsam_config.get(
                "config_path",
                "xsam/configs/xsam/s3_mixed_finetune/"
                "xsam_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune.py",
            )
        )
        self.xsam_checkpoint_path = xsam_config.get("checkpoint_path", None)
        self.score_thr = float(xsam_config.get("score_thr", 0.5))
        self.prompt_template = xsam_config.get(
            "prompt_template",
            "Can you segment <p>{PartDescription}</p> in this image? "
            "Please output the corresponding segmentation mask.",
        )
        self._xsam_module = None
        self._demo = None
        self._metadata_catalog = None
        self._init_xsam()
        self.seg_judge = build_vlm_prompter(segment_judge_config) if segment_judge_config is not None else None

    def _resolve_path(self, raw_path: str | None, base_dir: str | None = None) -> str | None:
        if raw_path is None:
            return None
        if os.path.isabs(raw_path):
            return raw_path
        if base_dir is not None:
            return os.path.abspath(os.path.join(base_dir, raw_path))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.abspath(os.path.join(repo_root, raw_path))

    def _resolve_existing_path(
        self,
        raw_path: str | None,
        *,
        candidate_base_dirs: list[str | None],
    ) -> str | None:
        if raw_path is None:
            return None
        tried: list[str] = []
        for base_dir in candidate_base_dirs:
            resolved = self._resolve_path(raw_path, base_dir=base_dir)
            if resolved is None:
                continue
            if resolved in tried:
                continue
            tried.append(resolved)
            if os.path.exists(resolved):
                return resolved
        # Return the first resolved candidate for clearer downstream error messages.
        if len(tried) > 0:
            return tried[0]
        return None

    def _init_xsam(self) -> None:
        code_root = self._resolve_path(self.xsam_code_root)
        if code_root is None or not os.path.isdir(code_root):
            raise RuntimeError(
                f"X-SAM code root does not exist: {self.xsam_code_root}. "
                "Set segmentation.xsam.code_root to your X-SAM `xsam` directory."
            )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        try:
            from mmengine.config import Config
            from xtuner.tools.utils import set_model_resource

            from xsam.dataset.utils.catalog import MetadataCatalog
            from xsam.demo.demo import XSamDemo
            from xsam.utils.utils import register_function
        except Exception as exc:
            raise RuntimeError(
                "Failed to import X-SAM runtime dependencies. "
                "Install X-SAM requirements and ensure they are importable."
            ) from exc

        config_path = self._resolve_existing_path(
            self.xsam_config_path,
            candidate_base_dirs=[code_root, None],
        )
        if config_path is None or not os.path.isfile(config_path):
            raise RuntimeError(
                f"X-SAM config file not found: {self.xsam_config_path}. "
                "Set segmentation.xsam.config_path to a valid config file."
            )

        checkpoint_path = self._resolve_existing_path(
            self.xsam_checkpoint_path,
            candidate_base_dirs=[None, code_root],
        )
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                f"X-SAM checkpoint file not found: {self.xsam_checkpoint_path}. "
                "Set segmentation.xsam.checkpoint_path to a valid `pytorch_model.bin`."
            )
        xsam_repo_root = os.path.dirname(code_root)
        env_overrides = {
            "CODE_DIR": code_root.rstrip("/") + "/",
            "DATA_DIR": os.path.join(xsam_repo_root, "datas").rstrip("/") + "/",
            "INIT_DIR": os.path.join(xsam_repo_root, "inits").rstrip("/") + "/",
            "WORK_DIR": os.path.join(xsam_repo_root, "wkdrs").rstrip("/") + "/",
        }
        for key, value in env_overrides.items():
            os.environ.setdefault(key, value)

        # X-SAM config files call helper functions like getenv()/deepcopy(), which break
        # under mmengine's lazy config parser. Force eager parsing when available.
        try:
            cfg = Config.fromfile(config_path, lazy_import=False)
        except TypeError:
            cfg = Config.fromfile(config_path)
        set_model_resource(cfg)
        register_function(cfg._cfg_dict)
        cfg.model.s1_pretrained_pth = None
        cfg.model.s2_pretrained_pth = None

        self._demo = XSamDemo(cfg, pth_model=checkpoint_path, output_ids_with_output=False)
        self._normalize_xsam_demo_processors()
        self._metadata_catalog = MetadataCatalog
        self._xsam_module = {
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
        }

    def _normalize_xsam_demo_processors(self) -> None:
        """Compat shim for newer transformers returning SiglipProcessor wrappers."""
        if self._demo is None:
            return
        image_processor = getattr(self._demo, "image_processor", None)
        if image_processor is None:
            return
        nested_image_processor = getattr(image_processor, "image_processor", None)
        if nested_image_processor is None:
            return

        missing_expected_api = (
            not hasattr(image_processor, "image_mean")
            or not hasattr(image_processor, "preprocess")
        )
        has_nested_expected_api = (
            hasattr(nested_image_processor, "image_mean")
            and hasattr(nested_image_processor, "preprocess")
        )
        if missing_expected_api and has_nested_expected_api:
            self._demo.image_processor = nested_image_processor

    def _format_prompt(self, part_description: str) -> str:
        return self.prompt_template.replace("{PartDescription}", part_description)

    @staticmethod
    def _segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
        zero_mask = np.zeros((height, width), dtype=bool)
        if segmentation is None:
            return zero_mask
        if torch.is_tensor(segmentation):
            seg = segmentation.detach().cpu().numpy()
        else:
            seg = np.asarray(segmentation)
        seg = np.squeeze(seg)
        if seg.ndim != 2:
            return zero_mask
        if seg.shape != (height, width):
            return zero_mask
        # For refseg postprocess, foreground is encoded as 1 and background/ignore uses other ids.
        return seg == 1

    @staticmethod
    def _segments_score(segments_info: Any) -> float | None:
        if isinstance(segments_info, dict):
            score = segments_info.get("score", None)
            return float(score) if score is not None else None
        if isinstance(segments_info, list):
            scores = [
                float(item.get("score"))
                for item in segments_info
                if isinstance(item, dict) and item.get("score") is not None
            ]
            if len(scores) > 0:
                return float(max(scores))
        return None

    def _segment_frame(self, frame: PILImage.Image, part_description: str) -> Tuple[np.ndarray, dict]:
        prompt = self._format_prompt(part_description)
        task_name = "refseg"
        data_dict = {"pil_image": frame, "vprompt_masks": None, "task_name": task_name}
        classes, task_name_postprocess = self._demo._get_classes_from_prompt(prompt, task_name)
        self._demo.model.postprocess_fn = self._demo.postprocess_fns[task_name_postprocess]
        self._demo._set_metadata(task_name, classes)
        data_dict.update(self._demo._process_prompt(prompt, task_name, classes))
        data_dict.update(self._demo._process_image(frame))
        data_dict.update(self._demo._process_data_dict(data_dict))
        data_dict, data_samples = self._demo._process_input_dict(data_dict)

        metadata = (
            self._metadata_catalog.get(task_name)
            if task_name in self._metadata_catalog.list()
            else self._demo.metadata
        )
        with torch.no_grad():
            llm_outputs, seg_outputs = self._demo.model(
                data_dict,
                data_samples,
                mode="predict",
                metadata=metadata,
                generation_config=self._demo.generation_config,
                stopping_criteria=self._demo.stop_criteria,
                do_postprocess=True,
                do_loss=False,
                mask_threshold=self.score_thr,
            )

        generation_output = ""
        if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
            generation_output = self._demo.tokenizer.decode(llm_outputs.sequences[0]).strip()
            generation_output = generation_output.replace("<|end|>", "").strip()

        if seg_outputs is None or len(seg_outputs) == 0:
            return np.zeros((frame.height, frame.width), dtype=bool), {
                "answer": generation_output,
                "prompt": prompt,
                "error": "xsam_no_segmentation_output",
            }

        seg_item = seg_outputs[0]
        segmentation = seg_item.get("segmentation", None) if isinstance(seg_item, dict) else None
        segments_info = seg_item.get("segments_info", None) if isinstance(seg_item, dict) else None
        mask = self._segmentation_to_mask(segmentation, frame.height, frame.width)
        answer_dict = {
            "answer": generation_output,
            "prompt": prompt,
            "score": self._segments_score(segments_info),
            "task_name": task_name,
            "model": "X-SAM",
        }
        if not np.any(mask):
            answer_dict["error"] = "xsam_empty_mask"
        return mask.astype(bool), answer_dict

    def segment_video(
        self, video_frame_list: List[PILImage.Image] | List[str], part_description: str
    ) -> Tuple[List[np.ndarray], List[dict], List[int]]:
        mask_list: list[np.ndarray] = []
        answer_dict_list: list[dict] = []
        valid_frame_ids: list[int] = []

        for frame_id, frame_or_path in enumerate(video_frame_list):
            if isinstance(frame_or_path, str):
                with PILImage.open(frame_or_path) as frame_img:
                    frame = frame_img.convert("RGB")
            else:
                frame = frame_or_path
            mask = np.zeros((frame.height, frame.width), dtype=bool)
            answer_dict = {
                "answer": "",
                "prompt": self._format_prompt(part_description),
                "error": "xsam_failed",
                "task_name": "refseg",
                "model": "X-SAM",
            }
            for _ in range(max(self.max_query, 1)):
                try:
                    mask, answer_dict = self._segment_frame(frame, part_description)
                    break
                except Exception as exc:
                    answer_dict = {
                        "answer": "",
                        "prompt": self._format_prompt(part_description),
                        "error": f"{type(exc).__name__}: {exc}",
                        "task_name": "refseg",
                        "model": "X-SAM",
                    }
            if self.seg_judge is None:
                if not answer_dict.get("error"):
                    valid_frame_ids.append(frame_id)
                answer_dict["vlm_judge"] = {}
            else:
                if answer_dict.get("error"):
                    answer_dict["vlm_judge"] = {}
                else:
                    vlm_judge_response = self.seg_judge.prompt(frame, mask, part_description)
                    answer_dict["vlm_judge"] = (
                        vlm_judge_response if len(vlm_judge_response.keys()) == 2 else {}
                    )
                    if (
                        len(vlm_judge_response.keys()) == 2
                        and vlm_judge_response.get("answer", "").lower() == "yes"
                    ):
                        valid_frame_ids.append(frame_id)
            answer_dict_list.append(answer_dict)
            mask_list.append(mask.astype(bool))
        return mask_list, answer_dict_list, valid_frame_ids


class MolmoSAM(RefSeg):
    def __init__(
        self,
        molmo_config: dict,
        segment_judge_config: dict | None,
        device: str = "cuda",
    ):
        self.device = device
        self.molmo_model = molmo_config.get("vlm_model", "allenai/Molmo-7B-D-0924")
        backend = str(molmo_config.get("backend", "auto")).lower()
        if backend not in {"auto", "molmo1", "molmo2"}:
            raise ValueError(
                f"Unsupported molmo.backend={backend}. Expected one of: auto, molmo1, molmo2."
            )
        if backend == "auto":
            self.molmo_backend = "molmo2" if "molmo2" in self.molmo_model.lower() else "molmo1"
        else:
            self.molmo_backend = backend

        default_point_scale = 1000.0 if self.molmo_backend == "molmo2" else 100.0
        point_scale_cfg = molmo_config.get("point_scale", None)
        self.point_scale = default_point_scale if point_scale_cfg is None else float(point_scale_cfg)
        if self.point_scale <= 0:
            raise ValueError(f"molmo.point_scale must be > 0, got {self.point_scale}.")

        scale_str = str(int(self.point_scale)) if self.point_scale.is_integer() else str(self.point_scale)
        default_prompt_template = (
            "Point to the {PartDescription} in each frame. "
            "Return points in the format: <points coords=\"frame_id point_id x y; ...\"/> "
            f"where x,y are in [0,{scale_str}]."
        )
        self.prompt_template = molmo_config.get("prompt_template", default_prompt_template)
        self.max_query = int(molmo_config.get("max_query", 1))
        self.max_new_tokens = int(molmo_config.get("max_new_tokens", 1024))
        self.max_points_per_frame = molmo_config.get("max_points_per_frame", None)
        self.max_points_per_frame = (
            int(self.max_points_per_frame) if self.max_points_per_frame is not None else None
        )
        self.prefer_scaled_points = bool(
            molmo_config.get("prefer_scaled_points", self.molmo_backend == "molmo2")
        )
        self.strict_point_parsing = bool(
            molmo_config.get("strict_point_parsing", self.molmo_backend == "molmo2")
        )

        # Load Molmo model
        self.molmo_processor = AutoProcessor.from_pretrained(
            self.molmo_model, trust_remote_code=True, dtype="auto", device_map="auto"
        )
        self.molmo = None
        try:
            self.molmo = AutoModelForImageTextToText.from_pretrained(
                self.molmo_model, trust_remote_code=True, dtype="auto", device_map="auto"
            )
        except ValueError:
            try:
                self.molmo = AutoModelForCausalLM.from_pretrained(
                    self.molmo_model, trust_remote_code=True, dtype="auto", device_map="auto"
                )
            except Exception:
                self.molmo = AutoModel.from_pretrained(
                    self.molmo_model, trust_remote_code=True, dtype="auto", device_map="auto"
                )
        if not hasattr(self.molmo, "generate"):
            raise ValueError(
                f"Molmo model {self.molmo_model} does not support generate(). "
                "Please use a causal LM-compatible model."
            )
        # Force HF generation to use legacy KV cache (avoid DynamicCache incompatibility).
        if hasattr(self.molmo.__class__, "_is_stateful"):
            self.molmo.__class__._is_stateful = True
        self.molmo.eval()

        # Build SAM3 processor
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bpe_path = os.path.join(repo_root, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(
                f"SAM3 BPE vocab not found at {bpe_path}. "
                "Download it from https://github.com/facebookresearch/sam3/raw/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            )
        confidence_threshold = float(molmo_config.get("sam3_confidence_threshold", 0.5))
        sam3_model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        self.sam3_processor = Sam3Processor(
            sam3_model, confidence_threshold=confidence_threshold, device=device
        )

        self.seg_judge = build_vlm_prompter(segment_judge_config) if segment_judge_config is not None else None
        self.debug_dir: str | None = None
        self.debug_records: list[dict] = []
        self._last_points_debug: dict | None = None
        self._debug_frame_id: int | None = None

    def set_debug_context(
        self,
        save_dir: str | None,
        scene_name: str | None,
        seg_id: str | None,
        role: str | None,
        prompt: str | None,
    ) -> None:
        if not save_dir or not scene_name or not seg_id:
            self.debug_dir = None
            self.debug_records = []
            return

        prompt_str = prompt or "prompt"
        prompt_str = re.sub(r"[^a-zA-Z0-9._-]+", "_", prompt_str).strip("_")
        if not prompt_str:
            prompt_str = "prompt"
        if len(prompt_str) > 48:
            prompt_str = prompt_str[:48]
        role_str = role or "role"
        role_str = re.sub(r"[^a-zA-Z0-9._-]+", "_", role_str).strip("_") or "role"

        self.debug_dir = os.path.join(
            save_dir, scene_name, seg_id, "debug", "molmo_sam", f"{role_str}_{prompt_str}"
        )
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_records = []

    def _debug_print(self, message: str) -> None:
        if self.debug_dir is not None:
            print(f"[molmo_sam] {message}", flush=True)

    def _format_prompt(self, part_description: str) -> str:
        scale_str = str(int(self.point_scale)) if self.point_scale.is_integer() else str(self.point_scale)
        formatted = self.prompt_template
        formatted = formatted.replace("{PartDescription}", part_description)
        formatted = formatted.replace("{PartLabel}", part_description)
        formatted = formatted.replace("{PointScale}", scale_str)
        if formatted != self.prompt_template:
            return formatted
        return f"{self.prompt_template.strip()} {part_description}".strip()

    def _points_from_num_str(self, text, image_w, image_h, points_regex):
        for points in points_regex.finditer(text):
            idx, x, y = points.group(1), points.group(2), points.group(3)
            raw_x, raw_y = float(x), float(y)
            scaled = self._scale_points([(raw_x, raw_y)], image_w=image_w, image_h=image_h)
            if len(scaled) > 0:
                x_s, y_s = scaled[0]
                yield idx, x_s, y_s

    def _extract_video_points(self, text: str, image_w: int, image_h: int) -> list[tuple[float, float, float]]:
        coord_regex = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
        frame_regex = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
        points_regex = re.compile(r"([0-9]+) ([0-9]{1,4}) ([0-9]{1,4})")

        all_points: list[tuple[float, float, float]] = []
        for coord in coord_regex.finditer(text):
            for point_grp in frame_regex.finditer(coord.group(1)):
                frame_id = float(point_grp.group(1))
                for _, x, y in self._points_from_num_str(
                    point_grp.group(2), image_w, image_h, points_regex
                ):
                    all_points.append((frame_id, x, y))
        return all_points

    def _extract_points_fallback(
        self, text: str, image_w: int, image_h: int
    ) -> list[tuple[float, float, float]]:
        points = self._extract_video_points(text, image_w=image_w, image_h=image_h)
        if len(points) > 0:
            return points
        single_points = self._extract_points_single(text, image_w=image_w, image_h=image_h)
        return [(0.0, x, y) for x, y in single_points]

    def _scale_points(
        self, points: list[tuple[float, float]], image_w: int, image_h: int
    ) -> list[tuple[float, float]]:
        if len(points) == 0:
            return points
        all_unit_coords = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)

        def _try_scaled(raw_x: float, raw_y: float) -> tuple[float, float] | None:
            # Handle explicit [0,1] normalized outputs when all returned points are unit-range.
            if all_unit_coords and 0 <= raw_x <= 1 and 0 <= raw_y <= 1:
                return raw_x * image_w, raw_y * image_h
            # Primary configured scale (Molmo2 defaults to [0,1000]).
            if 0 <= raw_x <= self.point_scale and 0 <= raw_y <= self.point_scale:
                return raw_x / self.point_scale * image_w, raw_y / self.point_scale * image_h
            # Backward compatibility for common Molmo ranges.
            for fallback_scale in (1000.0, 100.0):
                if fallback_scale == self.point_scale:
                    continue
                if 0 <= raw_x <= fallback_scale and 0 <= raw_y <= fallback_scale:
                    return raw_x / fallback_scale * image_w, raw_y / fallback_scale * image_h
            return None

        scaled = []
        for x, y in points:
            if self.prefer_scaled_points:
                maybe_scaled = _try_scaled(x, y)
                if maybe_scaled is not None:
                    scaled.append(maybe_scaled)
                    continue
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    scaled.append((x, y))
                    continue
            else:
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    scaled.append((x, y))
                    continue
                maybe_scaled = _try_scaled(x, y)
                if maybe_scaled is not None:
                    scaled.append(maybe_scaled)
                    continue
        return scaled

    def _extract_points_single(
        self, text: str, image_w: int, image_h: int
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        parse_mode = "none"
        tag_matches = re.findall(r"<points\s+([^>]+)>", text)
        coord_matches: list[str] = []
        pair_candidates_comma: list[tuple[str, str]] = []
        pair_candidates_space: list[tuple[str, str]] = []
        # Parse Molmo-style <points x1=".." y1=".." ...> tags.
        for tag_attrs in tag_matches:
            xs = dict(re.findall(r"x(\d+)=\"([0-9]*\.?[0-9]+)\"", tag_attrs))
            ys = dict(re.findall(r"y(\d+)=\"([0-9]*\.?[0-9]+)\"", tag_attrs))
            for idx in sorted(set(xs.keys()) & set(ys.keys()), key=lambda v: int(v)):
                try:
                    points.append((float(xs[idx]), float(ys[idx])))
                except ValueError:
                    continue
        if len(points) > 0:
            parse_mode = "points_tag"

        if len(points) == 0:
            coord_matches = re.findall(r"coords\s*=\s*\"([^\"]+)\"", text)
            for coords in coord_matches:
                cleaned = re.sub(r"[;,]", " ", coords)
                tokens = [t for t in cleaned.split() if t]
                nums: list[float] = []
                for tok in tokens:
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        continue
                if len(nums) >= 3:
                    if len(nums) % 4 == 0:
                        for i in range(0, len(nums), 4):
                            points.append((nums[i + 2], nums[i + 3]))
                    elif len(nums) % 2 == 0:
                        for i in range(0, len(nums), 2):
                            points.append((nums[i], nums[i + 1]))
                    elif len(nums) % 3 == 0:
                        for i in range(0, len(nums), 3):
                            points.append((nums[i + 1], nums[i + 2]))
                    else:
                        for i in range(0, len(nums) - 1, 2):
                            points.append((nums[i], nums[i + 1]))
            if len(points) > 0:
                parse_mode = "coords"

        if len(points) == 0:
            pair_candidates_comma = re.findall(
                r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text
            )
            pair_candidates = pair_candidates_comma
            if len(pair_candidates) == 0 and not self.strict_point_parsing:
                pair_candidates_space = re.findall(
                    r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)", text
                )
                pair_candidates = pair_candidates_space
            elif len(pair_candidates) == 0:
                # In strict mode, only accept whitespace-separated pairs when
                # they are explicitly delimited, to avoid parsing frame/point ids.
                pair_candidates_space = re.findall(
                    r"(?:^|[;\n\t\(\[])\s*([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*(?=$|[;\n\t\)\]])",
                    text,
                )
                pair_candidates = pair_candidates_space
            for x_str, y_str in pair_candidates:
                try:
                    points.append((float(x_str), float(y_str)))
                except ValueError:
                    continue
            if len(points) > 0:
                parse_mode = "pair"

        pre_scale_points = list(points)
        points = self._scale_points(points, image_w=image_w, image_h=image_h)
        if self.debug_dir is not None:
            raw_preview = re.sub(r"\s+", " ", text).strip()
            if len(raw_preview) > 200:
                raw_preview = raw_preview[:200] + "..."
            raw_preview = raw_preview.replace("\"", "\\\"")
            self._last_points_debug = {
                "frame_id": self._debug_frame_id,
                "text_len": len(text),
                "points_tag_matches": len(tag_matches),
                "coords_attr_matches": len(coord_matches),
                "pair_candidates_comma": len(pair_candidates_comma),
                "pair_candidates_space": len(pair_candidates_space),
                "parse_mode": parse_mode,
                "pre_scale_points": len(pre_scale_points),
                "post_scale_points": len(points),
                "filtered_points": len(pre_scale_points) - len(points),
                "raw_preview": raw_preview,
            }
        return points

    def _draw_points(self, frame: PILImage.Image, points: list[tuple[float, float]]) -> PILImage.Image:
        out = frame.copy()
        draw = ImageDraw.Draw(out)
        for idx, (x, y) in enumerate(points):
            xi, yi = int(round(x)), int(round(y))
            r = 6
            draw.ellipse((xi - r, yi - r, xi + r, yi + r), outline="red", width=2)
            draw.text((xi + r + 1, yi - r - 1), str(idx), fill="red")
        return out

    def _prompt_points_single(
        self, frame: PILImage.Image, part_description: str
    ) -> tuple[list[tuple[float, float, float]], str]:
        prompt_text = self._format_prompt(part_description)
        if "<points" not in prompt_text:
            scale_str = str(int(self.point_scale)) if self.point_scale.is_integer() else str(self.point_scale)
            prompt_text = (
                f"{prompt_text}\n"
                "Return points in the format: <points coords=\"0 0 x y; ...\"/> "
                f"where x,y are in [0,{scale_str}]."
            )

        if self.molmo_backend == "molmo2":
            # Molmo2 expects chat-template formatting and processor __call__.
            messages = [
                {
                    "role": "user",
                    "content": [
                        dict(type="image"),
                        dict(type="text", text=prompt_text),
                    ],
                }
            ]
            text = self.molmo_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.molmo_processor(
                images=[frame],
                text=[text],
                return_tensors="pt",
            )
        else:
            inputs = self.molmo_processor.process(
                text=prompt_text,
                images=frame,
                return_tensors="pt",
            )
            # Ensure batch dimension for legacy Molmo processor output.
            if inputs["input_ids"].dim() == 1:
                inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
            if "images" in inputs and inputs["images"].dim() == 3:
                inputs["images"] = inputs["images"].unsqueeze(0)
            if "image_input_idx" in inputs and inputs["image_input_idx"].dim() == 2:
                inputs["image_input_idx"] = inputs["image_input_idx"].unsqueeze(0)
            if "image_masks" in inputs and inputs["image_masks"].dim() == 2:
                inputs["image_masks"] = inputs["image_masks"].unsqueeze(0)

        inputs = {
            k: v.to(self.molmo.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            if self.molmo_backend == "molmo2":
                generated_ids = self.molmo.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                input_ids = inputs["input_ids"]
                attention_mask = None
                position_ids = None
                append_last_valid_logits = None
                if getattr(self.molmo.config, "use_position_ids", False):
                    attention_mask = input_ids != -1
                    position_ids = torch.clamp(
                        torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1, min=0
                    )
                    append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            attention_mask.new_ones((input_ids.shape[0], self.max_new_tokens)),
                        ],
                        dim=1,
                    )

                generated_ids = self.molmo.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    append_last_valid_logits=append_last_valid_logits,
                )

        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)
        prompt_token_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated_tokens = generated_ids[0, prompt_token_len:]
        generated_text = self.molmo_processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        points = self._extract_points_single(
            generated_text, image_w=frame.width, image_h=frame.height
        )
        return [(0.0, x, y) for x, y in points], generated_text

    def _prompt_points(self, frames: List[PILImage.Image], part_description: str) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        if len(frames) == 0:
            return points
        for frame_id, frame in enumerate(frames):
            if self.debug_dir is not None:
                self._debug_frame_id = frame_id
            frame_points: list[tuple[float, float, float]] = []
            raw_text = ""
            attempts = 0
            for _ in range(max(self.max_query, 1)):
                frame_points, raw_text = self._prompt_points_single(frame, part_description)
                attempts += 1
                if len(frame_points) > 0:
                    break
            if self.debug_dir is not None:
                dbg = self._last_points_debug or {}
                self._debug_print(
                    "frame="
                    f"{frame_id} attempts={attempts} points={len(frame_points)} "
                    f"parse_mode={dbg.get('parse_mode')} "
                    f"tags={dbg.get('points_tag_matches')} "
                    f"coords={dbg.get('coords_attr_matches')} "
                    f"pair_comma={dbg.get('pair_candidates_comma')} "
                    f"pair_space={dbg.get('pair_candidates_space')} "
                    f"pre_scale={dbg.get('pre_scale_points')} "
                    f"post_scale={dbg.get('post_scale_points')} "
                    f"filtered={dbg.get('filtered_points')} "
                    f"raw_len={dbg.get('text_len')} "
                    f"raw_preview=\"{dbg.get('raw_preview')}\""
                )
            for _, x, y in frame_points:
                points.append((float(frame_id), float(x), float(y)))
            if self.debug_dir is not None:
                overlay = self._draw_points(frame, [(x, y) for _, x, y in frame_points])
                overlay.save(os.path.join(self.debug_dir, f"frame_{frame_id:04d}.png"))
                self.debug_records.append(
                    {
                        "frame_id": int(frame_id),
                        "points": [[float(x), float(y)] for _, x, y in frame_points],
                        "raw_text": raw_text,
                    }
                )
        return points

    def _group_points_by_frame(
        self, points: list[tuple[float, float, float]], num_frames: int
    ) -> list[list[tuple[float, float]]]:
        grouped: list[list[tuple[float, float]]] = [[] for _ in range(num_frames)]
        for frame_id, x, y in points:
            frame_idx = int(round(frame_id))
            if 0 <= frame_idx < num_frames:
                grouped[frame_idx].append((float(x), float(y)))
        if self.max_points_per_frame is not None and self.max_points_per_frame > 0:
            grouped = [pts[: self.max_points_per_frame] for pts in grouped]
        return grouped

    def _sam3_from_points(
        self, image: PILImage.Image, points: list[tuple[float, float]]
    ) -> Tuple[np.ndarray, dict]:
        width, height = image.size
        if len(points) == 0:
            return np.zeros((height, width), dtype=bool), {"error": "no_points"}

        state = self.sam3_processor.set_image(image)
        if "language_features" not in state["backbone_out"]:
            dummy_text = self.sam3_processor.model.backbone.forward_text(
                ["visual"], device=self.sam3_processor.device
            )
            state["backbone_out"].update(dummy_text)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.sam3_processor.model._get_dummy_prompt()

        norm_points = []
        for x, y in points:
            x_n = min(max(x / width, 0.0), 1.0)
            y_n = min(max(y / height, 0.0), 1.0)
            norm_points.append([x_n, y_n])

        points_tensor = torch.tensor(
            norm_points, device=self.sam3_processor.device, dtype=torch.float32
        ).view(-1, 1, 2)
        labels = torch.ones(
            (points_tensor.shape[0], 1),
            device=self.sam3_processor.device,
            dtype=torch.long,
        )
        state["geometric_prompt"].append_points(points_tensor, labels)
        state = self.sam3_processor._forward_grounding(state)

        masks = state.get("masks")
        if masks is None or masks.numel() == 0:
            return np.zeros((height, width), dtype=bool), {"error": "sam3_no_masks"}
        mask = masks.squeeze(1).any(dim=0).detach().cpu().numpy().astype(bool)

        answer_dict = {
            "molmo_points": points,
            "sam3_pred_boxes": state["boxes"].detach().cpu().numpy().tolist()
            if "boxes" in state
            else [],
            "sam3_pred_scores": state["scores"].detach().to(torch.float32).cpu().numpy().tolist()
            if "scores" in state
            else [],
        }
        return mask, answer_dict

    def segment_video(
        self, video_frame_list: List[PILImage.Image] | List[str], part_description: str
    ) -> Tuple[List[np.ndarray], List[dict], List[int]]:
        if len(video_frame_list) == 0:
            return [], [], []

        frames: List[PILImage.Image] = []
        for frame in video_frame_list:
            if isinstance(frame, str):
                frames.append(PILImage.open(frame).convert("RGB"))
            else:
                frames.append(frame)

        points = []
        for _ in range(max(self.max_query, 1)):
            points = self._prompt_points(frames, part_description)
            if len(points) > 0:
                break

        points_by_frame = self._group_points_by_frame(points, len(frames))

        mask_list = []
        answer_dict_list = []
        valid_frame_ids = []

        for frame_id, frame in enumerate(frames):
            frame_points = points_by_frame[frame_id]
            mask, answer_dict = self._sam3_from_points(frame, frame_points)
            if isinstance(answer_dict, dict) and answer_dict.get("error"):
                answer_dict.setdefault("molmo_points", frame_points)
                answer_dict["vlm_judge"] = {}
            else:
                if self.seg_judge is None:
                    valid_frame_ids.append(frame_id)
                    answer_dict["vlm_judge"] = {}
                else:
                    vlm_judge_response = self.seg_judge.prompt(
                        frame, mask, part_description
                    )
                    answer_dict["vlm_judge"] = (
                        vlm_judge_response if len(vlm_judge_response.keys()) == 2 else {}
                    )
                    if (
                        len(vlm_judge_response.keys()) == 2
                        and vlm_judge_response["answer"] == "yes"
                    ):
                        valid_frame_ids.append(frame_id)
            answer_dict_list.append(answer_dict)
            mask_list.append(mask)

        if self.debug_dir is not None:
            with open(os.path.join(self.debug_dir, "molmo_points.json"), "w") as f:
                json.dump(
                    {
                        "prompt": part_description,
                        "frames": self.debug_records,
                    },
                    f,
                    indent=2,
                )

        return mask_list, answer_dict_list, valid_frame_ids


class SAM3Agent(RefSeg):
    def __init__(
        self,
        llm_config: dict,
        segment_judge_config: dict,
        confidence_threshold: float = 0.5,
        max_generations: int = 100,
        device: str = "cuda",
        output_dir: str = "sam3_agent_out",
    ):
        self.device = device
        self.output_dir = output_dir
        self.max_generations = max_generations

        from sam3 import build_sam3_image_model
        from sam3.agent.agent_core import agent_inference
        from sam3.agent.client_llm import (
            send_generate_request as send_generate_request_orig,
        )
        from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
        from sam3.model.sam3_image_processor import Sam3Processor

        # Use cached BPE vocab file
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bpe_path = os.path.join(repo_root, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(
                f"SAM3 BPE vocab not found at {bpe_path}. "
                "Download it from https://github.com/facebookresearch/sam3/raw/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            )
        model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        self.agent_inference = agent_inference

        self.send_generate_request = partial(
            send_generate_request_orig,
            server_url=llm_config["llm_server_url"],
            model=llm_config["llm_model"],
            api_key=llm_config["llm_api_key"],
        )
        self.call_sam_service = partial(call_sam_service_orig, sam3_processor=self.processor)

        self.seg_judge = build_vlm_prompter(segment_judge_config) if segment_judge_config is not None else None

    def _decode_rle_masks(self, rle_counts_list: List[str], height: int, width: int) -> np.ndarray:
        from pycocotools import mask as mask_utils

        if len(rle_counts_list) == 0:
            return np.zeros((height, width), dtype=bool)
        mask_all = np.zeros((height, width), dtype=bool)
        for rle_counts in rle_counts_list:
            rle = {"counts": rle_counts, "size": [height, width]}
            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            mask_all = np.logical_or(mask_all, decoded.astype(bool))
        return mask_all

    def segment_image(self, image: PILImage.Image | str, part_description: str) -> Tuple[np.ndarray, dict]:
        temp_path = None
        if isinstance(image, str):
            image_path = image
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name
            image_path = temp_path

        # Inject local system prompts into agent_inference via monkeypatch
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_dir = os.path.join(repo_root, "sam3_system_prompts")
        
        try:
            # Temporarily set environment variable that we'll patch agent_core to use
            original_prompt_path = os.environ.get('SAM3_SYSTEM_PROMPT_DIR')
            os.environ['SAM3_SYSTEM_PROMPT_DIR'] = prompt_dir
            
            # Directly pass system prompts by reading them ourselves
            with open(os.path.join(prompt_dir, "system_prompt.txt")) as f:
                system_prompt = f.read().strip()
            with open(os.path.join(prompt_dir, "system_prompt_iterative_checking.txt")) as f:
                iter_prompt = f.read().strip()
            
            # Unfortunately agent_inference doesn't accept custom prompts, so overwrite package prompts.
            import sam3.agent.agent_core as agent_module
            agent_dir = os.path.dirname(agent_module.__file__)
            prompt_target = os.path.join(agent_dir, "system_prompts")
            os.makedirs(prompt_target, exist_ok=True)

            def _copy_if_different(src: str, dst: str):
                if os.path.exists(dst) and os.path.samefile(src, dst):
                    return
                shutil.copyfile(src, dst)

            _copy_if_different(
                os.path.join(prompt_dir, "system_prompt.txt"),
                os.path.join(prompt_target, "system_prompt.txt"),
            )
            _copy_if_different(
                os.path.join(prompt_dir, "system_prompt_iterative_checking.txt"),
                os.path.join(prompt_target, "system_prompt_iterative_checking.txt"),
            )
            
            raw_send_generate_request = self.send_generate_request

            def safe_send_generate_request(messages):
                generated_text = raw_send_generate_request(messages)
                if generated_text is None:
                    generated_text = ""
                # If this is an iterative mask check, force a strict Accept/Reject verdict.
                def _message_text(m: dict) -> str:
                    content = m.get("content")
                    # SAM3 agent_core uses string system prompts for iterative checking.
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts: list[str] = []
                        for c in content:
                            if isinstance(c, dict):
                                parts.append(c.get("text", ""))
                            elif isinstance(c, str):
                                parts.append(c)
                        return "\n".join(parts)
                    return ""

                joined_text = "\n".join(_message_text(m) for m in messages if isinstance(m, dict))
                if "visual mask verifier" in joined_text or "<verdict>" in joined_text:
                    if "<verdict>Accept</verdict>" in generated_text:
                        return "<verdict>Accept</verdict>"
                    if "<verdict>Reject</verdict>" in generated_text:
                        return "<verdict>Reject</verdict>"
                    # Sometimes models emit bare words; normalize into the expected tag.
                    if "Accept" in generated_text and "Reject" not in generated_text:
                        return "<verdict>Accept</verdict>"
                    if "Reject" in generated_text and "Accept" not in generated_text:
                        return "<verdict>Reject</verdict>"
                    return "<verdict>Reject</verdict>"
                if "<tool>" in generated_text:
                    tool_block = generated_text.split("</tool>", 1)[0].split("<tool>")[-1].strip()
                    try:
                        json.loads(tool_block)
                        return f"<tool>{tool_block}</tool>"
                    except Exception:
                        pass
                # Fallback: force a valid tool call
                tool_call = {
                    "name": "segment_phrase",
                    "parameters": {"text_prompt": part_description},
                }
                return f"<tool>{json.dumps(tool_call)}</tool>"

            try:
                _, final_output, _ = self.agent_inference(
                    image_path,
                    part_description,
                    send_generate_request=safe_send_generate_request,
                    call_sam_service=self.call_sam_service,
                    max_generations=self.max_generations,
                    output_dir=self.output_dir,
                )
            except Exception as exc:
                # Fail per-frame without killing the full run.
                print(f"Warning: SAM3 agent inference failed ({type(exc).__name__}): {exc}")
                final_output = None
        finally:
            if original_prompt_path is not None:
                os.environ['SAM3_SYSTEM_PROMPT_DIR'] = original_prompt_path
            elif 'SAM3_SYSTEM_PROMPT_DIR' in os.environ:
                del os.environ['SAM3_SYSTEM_PROMPT_DIR']
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

        if final_output is None:
            if isinstance(image, str):
                with PILImage.open(image) as img:
                    img = img.convert("RGB")
                    height, width = img.height, img.width
            else:
                height, width = image.height, image.width
            mask = np.zeros((height, width), dtype=bool)
            answer_dict = {
                "prompt": part_description,
                "sam3_pred_boxes": [],
                "sam3_pred_scores": [],
                "error": "sam3_agent_inference_failed",
            }
            return mask, answer_dict

        mask = self._decode_rle_masks(final_output["pred_masks"], final_output["orig_img_h"], final_output["orig_img_w"])
        answer_dict = {
            "prompt": part_description,
            "sam3_pred_boxes": final_output["pred_boxes"],
            "sam3_pred_scores": final_output["pred_scores"],
        }
        return mask, answer_dict

    def segment_video(self, video_frame_list: List[PILImage.Image] | List[str], part_description: str) -> Tuple[List[np.ndarray], List[dict], List[int]]:
        mask_list = []
        answer_dict_list = []
        valid_frame_ids = []
        for frame_id, frame in enumerate(video_frame_list):
            print(f"Segmenting frame {frame_id} ...")
            mask, answer_dict = self.segment_image(frame, part_description)
            if isinstance(frame, str):
                frame_image = PILImage.open(frame).convert("RGB")
            else:
                frame_image = frame
            if self.seg_judge is None:
                if isinstance(answer_dict, dict) and answer_dict.get("error"):
                    answer_dict["vlm_judge"] = {}
                else:
                    valid_frame_ids.append(frame_id)
                    answer_dict["vlm_judge"] = {}
            else:
                if isinstance(answer_dict, dict) and answer_dict.get("error"):
                    answer_dict["vlm_judge"] = {}
                else:
                    vlm_judge_response = self.seg_judge.prompt(frame_image, mask, part_description)
                    answer_dict["vlm_judge"] = vlm_judge_response if len(vlm_judge_response.keys()) == 2 else {}
                    if len(vlm_judge_response.keys()) == 2 and vlm_judge_response["answer"] == "yes":
                        valid_frame_ids.append(frame_id)
            answer_dict_list.append(answer_dict)
            mask_list.append(mask)
        return mask_list, answer_dict_list, valid_frame_ids
    

def build_refseg_model(segmentation_config: dict) -> RefSeg:
    disable_vlm_judge = segmentation_config.get("disable_vlm_judge", False)
    segment_judge_config = None if disable_vlm_judge else segmentation_config["vlm_judge"]
    if segmentation_config.model == "SegZero":
        return SegZero(
            reasoning_model_path=segmentation_config["reasoning_model_path"],
            segmentation_model_path=segmentation_config["segmentation_model_path"],
            segment_judge_config=segment_judge_config,
            max_query=segmentation_config["max_query"],
            device=segmentation_config["device"],
            sam_backend=segmentation_config.get("sam_backend", "sam2"),
            sam3_confidence_threshold=segmentation_config.get("sam3_confidence_threshold", 0.5),
        )
    elif segmentation_config.model == "Sa2VA":
        sa2va_config = segmentation_config.get("sa2va", {})
        model_name = sa2va_config.get("model_path", segmentation_config.get("reasoning_model_path"))
        if model_name is None:
            raise ValueError(
                "Sa2VA requires either segmentation.sa2va.model_path "
                "or segmentation.reasoning_model_path to be set."
            )
        return Sa2VA(
            model_name=model_name,
            segment_judge_config=segment_judge_config,
            sa2va_config=sa2va_config,
            max_query=segmentation_config.get("max_query", 1),
            device=segmentation_config.get("device", "cuda"),
        )
    elif segmentation_config.model == "XSam":
        xsam_config = segmentation_config.get("xsam", {})
        return XSam(
            xsam_config=xsam_config,
            segment_judge_config=segment_judge_config,
            max_query=segmentation_config.get("max_query", 1),
            device=segmentation_config.get("device", "cuda"),
        )
    elif segmentation_config.model == "SAM3Agent":
        return SAM3Agent(
            llm_config=segmentation_config["sam3"],
            segment_judge_config=segment_judge_config,
            confidence_threshold=segmentation_config["sam3"]["confidence_threshold"],
            max_generations=segmentation_config["sam3"]["max_generations"],
            device=segmentation_config["device"],
            output_dir=segmentation_config["sam3"]["output_dir"],
        )
    elif segmentation_config.model == "MolmoSAM":
        molmo_config = segmentation_config.get("molmo", {})
        return MolmoSAM(
            molmo_config=molmo_config,
            segment_judge_config=segment_judge_config,
            device=segmentation_config["device"],
        )
    else:
        raise ValueError(f"Unsupported segmentation model: {segmentation_config.model}")
