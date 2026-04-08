from google import genai
from openai import OpenAI
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from molmo_utils import process_vision_info
from vllm import LLM, SamplingParams
# from qwen_vl_utils import process_vision_info
import re
import cv2
import numpy as np
from PIL import Image as PILImage
import base64
import os
import shutil
import tempfile
try:
    # MoviePy >=2
    from moviepy import ImageSequenceClip
except ImportError:
    try:
        # MoviePy 1.x
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        ImageSequenceClip = None

from typing import List, Tuple, Dict, Optional

from omegaconf import OmegaConf


def compose_video_from_numpy_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = "libx264",
) -> None:
    """Compose an MP4 video from a list of HxWx3 numpy frames."""
    if ImageSequenceClip is None:
        raise ImportError(
            "moviepy is required to compose videos. Install with `pip install moviepy`."
        )
    if len(frames) == 0:
        raise ValueError("frames must contain at least one frame")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    normalized_frames = []
    first_shape = frames[0].shape
    for i, frame in enumerate(frames):
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] != 3:
            raise ValueError(f"frame {i} must have shape (H, W, 3), got {frame_np.shape}")
        if frame_np.shape != first_shape:
            raise ValueError(
                f"all frames must have the same shape; frame 0 is {first_shape}, frame {i} is {frame_np.shape}"
            )
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        normalized_frames.append(frame_np)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    clip = ImageSequenceClip(normalized_frames, fps=fps)
    clip.write_videofile(output_path, codec=codec)
    clip.close()

def build_video_metadata(num_frames: int, fps: float):
    return {
        "fps": float(fps),
        "total_num_frames": int(num_frames),
        "duration": float(num_frames) / float(fps),
        # This key is commonly used by vLLM's video loader outputs;
        # providing it makes timestamp logic consistent.
        "frames_indices": list(range(int(num_frames))),
    }


class VLMPrompter:
    def __init__(self, vlm_model: str = "gemini-2.5-flash", prompt_template: str = "", max_query: int = 10):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = None
        self.vlm_model = vlm_model
        self.prompt_template = prompt_template
        self.max_query = max_query


class GeminiVideoNarrator(VLMPrompter):
    def __init__(self, vlm_model: str = "gemini-2.5-flash", prompt_template: str = "", max_query: int = 10):
        super().__init__(vlm_model, prompt_template, max_query)
        self.client = genai.Client()
    
    def prompt_description(self, video_path: str) -> dict:
        grouped_results = {}
        query_count = 0
        video_bytes = open(video_path, 'rb').read()
        while len(grouped_results.keys()) != 2 and query_count < self.max_query:
            response = self.client.models.generate_content(
                model=self.vlm_model, contents=genai.types.Content(
                    parts=[
                        genai.types.Part(
                            inline_data=genai.types.Blob(data=video_bytes, mime_type='video/mp4')
                        ),
                        genai.types.Part(text=self.prompt_template)
                    ]
                )
            )
            query_count += 1
            grouped_results = self.post_process_description_output(response.text)
        return grouped_results
    
    def post_process_description_output(self, output_text: str) -> dict:
        pairs = re.findall(r'\{\s*name:\s*(.*?)\s*,\s*description:\s*(.*?)\s*\}', output_text, flags=re.S)

        def clean(t: str) -> str:
            return re.sub(r'\s+', ' ', t).strip()

        grouped = {}
        if len(pairs) == 2:
            for i, (name, desc) in enumerate(pairs):
                name_c = clean(name)
                desc_c = clean(desc)
                role = "receptor" if i == 0 else "effector"
                grouped[role] = {"name": name_c, "description": desc_c}
        else:
            print("Warning: Unexpected number of parts found in VLM output.")
        return grouped
    
    def prompt_function(self, rgb_frame_list: List[np.ndarray], receptor_part_masks: np.ndarray, effector_part_masks: np.ndarray) -> Dict[str, str]:
        rendered_frames = []
        for i in range(len(rgb_frame_list)):
            rgb_frame = rgb_frame_list[i]
            receptor_mask = receptor_part_masks[i]
            effector_mask = effector_part_masks[i]

            # Create color masks
            receptor_color_mask = np.zeros_like(rgb_frame)
            receptor_color_mask[:, :, 1] = receptor_mask * 255  # Green for receptor

            effector_color_mask = np.zeros_like(rgb_frame)
            effector_color_mask[:, :, 0] = effector_mask * 255  # Red for effector

            # Blend the original frame with the masks
            alpha = 0.5
            blended_frame = cv2.addWeighted(rgb_frame, 1 - alpha, receptor_color_mask, alpha, 0)
            blended_frame = cv2.addWeighted(blended_frame, 1 - alpha, effector_color_mask, alpha, 0)

            rendered_frames.append(blended_frame)
        tmp_dir = "tmp_masked_video"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        compose_video_from_numpy_frames(
            frames=rendered_frames,
            output_path=f"{tmp_dir}/rendered_video.mp4",
            fps=15,
            codec="libx264",
        )

        grouped_results = {}
        query_count = 0
        video_bytes = open(f"{tmp_dir}/rendered_video.mp4", 'rb').read()
        while len(grouped_results.keys()) != 3 and query_count < self.max_query:
            response = self.client.models.generate_content(
                model=self.vlm_model, contents=genai.types.Content(
                    parts=[
                        genai.types.Part(
                            inline_data=genai.types.Blob(data=video_bytes, mime_type='video/mp4')
                        ),
                        genai.types.Part(text=self.prompt_template)
                    ]
                )
            )
            query_count += 1
            grouped_results = self.post_process_function_output(response.text)

        return grouped_results
    
    def post_process_function_output(self, output_text: str) -> dict:
        try:
            grouped_results = eval(output_text)
        except (SyntaxError, NameError):
            print(f"Failed to evaluate output text: {output_text}")
            grouped_results = {}
        return grouped_results


class GPTVideoNarrator(VLMPrompter):
    def __init__(self, vlm_model = "gpt-5-mini", prompt_template = "", max_query = 10):
        super().__init__(vlm_model, prompt_template, max_query)
        self.client = OpenAI()
    
    def prompt_description(self, video_path: str) -> dict:
        file = self.client.files.create(
            file=open(video_path, "rb"),
            purpose="raw video"
        )
        grouped_results = {}
        query_count = 0
        while len(grouped_results.keys()) != 2 and query_count < self.max_query:
            response = self.client.responses.create(
                model=self.vlm_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file.id,
                            },
                            {
                                "type": "input_text",
                                "text": self.prompt_template,
                            },
                        ]
                    }
                ]
            )
            query_count += 1
            grouped_results = self.post_process_description_output(response.output_text)
        return grouped_results

    def post_process_description_output(self, output_text: str) -> dict:
        pairs = re.findall(r'\{\s*name:\s*(.*?)\s*,\s*description:\s*(.*?)\s*\}', output_text, flags=re.S)

        def clean(t: str) -> str:
            return re.sub(r'\s+', ' ', t).strip()

        grouped = {}
        if len(pairs) == 2:
            for i, (name, desc) in enumerate(pairs):
                name_c = clean(name)
                desc_c = clean(desc)
                role = "receptor" if i == 0 else "effector"
                grouped[role] = {"name": name_c, "description": desc_c}
        else:
            print("Warning: Unexpected number of parts found in VLM output.")
        return grouped

    def prompt_function(self, rgb_frame_list: List[np.ndarray], receptor_part_masks: np.ndarray, effector_part_masks: np.ndarray) -> Dict[str, str]:
        rendered_base64_frames = []
        for i in range(len(rgb_frame_list)):
            rgb_frame = rgb_frame_list[i]
            receptor_mask = receptor_part_masks[i]
            effector_mask = effector_part_masks[i]

            # Create color masks
            receptor_color_mask = np.zeros_like(rgb_frame)
            receptor_color_mask[:, :, 1] = receptor_mask * 255  # Green for receptor

            effector_color_mask = np.zeros_like(rgb_frame)
            effector_color_mask[:, :, 0] = effector_mask * 255  # Red for effector

            # Blend the original frame with the masks
            alpha = 0.5
            blended_frame = cv2.addWeighted(rgb_frame, 1 - alpha, receptor_color_mask, alpha, 0)
            blended_frame = cv2.addWeighted(blended_frame, 1 - alpha, effector_color_mask, alpha, 0)

            blended_frame = cv2.cvtColor(blended_frame, cv2.COLOR_RGB2BGR)  # Convert to RGB for PIL compatibility
            _, buffer = cv2.imencode(".jpg", blended_frame)

            rendered_base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        # tmp_dir = "tmp_masked_video_gpt"
        # if os.path.exists(tmp_dir):
        #     shutil.rmtree(tmp_dir)
        # compose_video_from_numpy_frames(
        #     frames=rendered_frames,
        #     output_path=f"{tmp_dir}/rendered_video.mp4",
        #     fps=15,
        #     codec="libx264",
        # )
        if len(rendered_base64_frames) > 50:
            sample_interval = len(rendered_base64_frames) // 50 + 1
            rendered_base64_frames = rendered_base64_frames[::sample_interval]

        grouped_results = {}
        query_count = 0
        # file = self.client.files.create(
        #     file=open(f"{tmp_dir}/rendered_video.mp4", "rb"),
        #     purpose="user_data"
        # )
        while len(grouped_results.keys()) != 3 and query_count < self.max_query:
            response = self.client.responses.create(
                model=self.vlm_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{frame}"
                                }
                                for frame in rendered_base64_frames
                            ],
                            {
                                "type": "input_text",
                                "text": self.prompt_template,
                            },
                        ]
                    }
                ]
            )
            query_count += 1
            grouped_results = self.post_process_function_output(response.output_text)

        return grouped_results
    
    def post_process_function_output(self, output_text: str) -> dict:
        try:
            grouped_results = eval(output_text)
        except (SyntaxError, NameError):
            print(f"Failed to evaluate output text: {output_text}")
            grouped_results = {}
        return grouped_results
    

class MolmoVideoNarrator(VLMPrompter):
    def __init__(self, vlm_model = "allenai/Molmo2-8B", prompt_template = "", max_query = 10):
        super().__init__(vlm_model, prompt_template, max_query)
        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            vlm_model,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        # load the model
        self.model = AutoModelForImageTextToText.from_pretrained(
            vlm_model,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

    def prompt(self, video_path: str, prompt_type: str = "text") -> dict:
        if prompt_type == "text":
            return self.prompt_text(video_path)
        elif prompt_type == "points":
            return self.prompt_points(video_path)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def prompt_text(self, video_path: str) -> dict:
        # process the video and text
        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=self.prompt_template),
                    dict(type="video", video=video_path),
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        grouped_results = {}
        query_count = 0
        while len(grouped_results.keys()) != 2 and query_count < self.max_query:
            # generate output
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            # only get generated tokens; decode them to text
            generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            query_count += 1
            grouped_results = self.post_process_output(generated_text)
        return grouped_results
    
    def post_process_output(self, output_text: str) -> dict:
        pairs = re.findall(r'\{\s*name:\s*(.*?)\s*,\s*description:\s*(.*?)\s*\}', output_text, flags=re.S)

        def clean(t: str) -> str:
            return re.sub(r'\s+', ' ', t).strip()

        grouped = {}
        if len(pairs) == 2:
            for i, (name, desc) in enumerate(pairs):
                name_c = clean(name)
                desc_c = clean(desc)
                role = "receptor" if i == 0 else "effector"
                grouped[role] = {"name": name_c, "description": desc_c}
        else:
            print("Warning: Unexpected number of parts found in VLM output.")
        return grouped
    
    def prompt_points(self, video_path: str) -> list:
        COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
        FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
        POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=self.prompt_template),
                    dict(type="video", video=video_path, max_fps=8),
                ],
            }
        ]

        # process the video using `molmo_utils.process_vision_info`
        _, videos, video_kwargs = process_vision_info(messages)
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)

        # apply the chat template to the input messages
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # process the video and text
        inputs = self.processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # generate output
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)

        # only get generated tokens; decode them to text
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # decode video pointing outputs
        points = self.extract_video_points(generated_text, image_w=video_metadatas[0]["width"], image_h=video_metadatas[0]["height"], 
                                           COORD_REGEX=COORD_REGEX, FRAME_REGEX=FRAME_REGEX, POINTS_REGEX=POINTS_REGEX)
        return points

    def _points_from_num_str(self, text, image_w, image_h, POINTS_REGEX, extract_ids=False):
        all_points = []
        for points in POINTS_REGEX.finditer(text):
            ix, x, y = points.group(1), points.group(2), points.group(3)
            # our points format assume coordinates are scaled by 1000
            x, y = float(x)/1000*image_w, float(y)/1000*image_h
            if 0 <= x <= image_w and 0 <= y <= image_h:
                yield ix, x, y

    def extract_video_points(self, text, image_w, image_h, COORD_REGEX, FRAME_REGEX, POINTS_REGEX, extract_ids=False):
        """Extract video pointing coordinates as a flattened list of (t, x, y) triplets from model output text."""
        all_points = []
        for coord in COORD_REGEX.finditer(text):
            for point_grp in FRAME_REGEX.finditer(coord.group(1)):
                frame_id = float(point_grp.group(1))
                w, h = (image_w, image_h)
                for idx, x, y in self._points_from_num_str(point_grp.group(2), w, h, POINTS_REGEX):
                    if extract_ids:
                        all_points.append((frame_id, idx, x, y))
                    else:
                        all_points.append((frame_id, x, y))
        return all_points

    def prompt_function(self, rgb_frame_list: List[PILImage.Image], receptor_part_masks: np.ndarray, effector_part_masks: np.ndarray) -> Dict[str, str]:
        rendered_frames = []
        for i in range(len(rgb_frame_list)):
            rgb_frame = np.array(rgb_frame_list[i])
            receptor_mask = receptor_part_masks[i]
            effector_mask = effector_part_masks[i]

            # Create color masks
            receptor_color_mask = np.zeros_like(rgb_frame)
            receptor_color_mask[:, :, 1] = receptor_mask * 255  # Green for receptor

            effector_color_mask = np.zeros_like(rgb_frame)
            effector_color_mask[:, :, 0] = effector_mask * 255  # Red for effector

            # Blend the original frame with the masks
            alpha = 0.5
            blended_frame = cv2.addWeighted(rgb_frame, 1 - alpha, receptor_color_mask, alpha, 0)
            blended_frame = cv2.addWeighted(blended_frame, 1 - alpha, effector_color_mask, alpha, 0)

            rendered_frames.append(blended_frame)
        tmp_dir = "tmp_masked_video_molmo"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        compose_video_from_numpy_frames(
            frames=rendered_frames,
            output_path=f"{tmp_dir}/rendered_video.mp4",
            fps=15,
            codec="libx264",
        )

        grouped_results = {}
        query_count = 0
        # process the video and text
        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=self.prompt_template),
                    dict(type="video", video=f"{tmp_dir}/rendered_video.mp4"),
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        while len(grouped_results.keys()) != 2 and query_count < self.max_query:
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            # only get generated tokens; decode them to text
            generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            query_count += 1
            grouped_results = self.post_process_function_output(generated_text)

        return grouped_results
    
    def post_process_function_output(self, output_text: str) -> dict:
        try:
            grouped_results = eval(output_text)
        except (SyntaxError, NameError):
            print(f"Failed to evaluate output text: {output_text}")
            grouped_results = {}
        return grouped_results
    

class MolmovllmVideoNarrator(VLMPrompter):
    def __init__(self, vlm_model = "allenai/Molmo2-8B", prompt_template = "", max_query = 10):
        super().__init__(vlm_model, prompt_template, max_query)
        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            vlm_model,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        # load the model
        self.model = LLM(
            model=vlm_model,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 1},
            tensor_parallel_size=2,
            trust_remote_code=True,
            max_num_batched_tokens=31872
        )

        self.sampling_params = SamplingParams(max_tokens=1024)
    

    def prompt_function(self, rgb_frame_list: List[np.ndarray], receptor_part_masks: np.ndarray, effector_part_masks: np.ndarray) -> Dict[str, str]:
        rendered_frames = []
        for i in range(len(rgb_frame_list)):
            rgb_frame = rgb_frame_list[i]
            receptor_mask = receptor_part_masks[i]
            effector_mask = effector_part_masks[i]

            # Create color masks
            receptor_color_mask = np.zeros_like(rgb_frame)
            receptor_color_mask[:, :, 1] = receptor_mask * 255  # Green for receptor

            effector_color_mask = np.zeros_like(rgb_frame)
            effector_color_mask[:, :, 0] = effector_mask * 255  # Red for effector

            # Blend the original frame with the masks
            alpha = 0.5
            blended_frame = cv2.addWeighted(rgb_frame, 1 - alpha, receptor_color_mask, alpha, 0)
            blended_frame = cv2.addWeighted(blended_frame, 1 - alpha, effector_color_mask, alpha, 0)

            rendered_frames.append(blended_frame.astype(np.uint8))
        # tmp_dir = "tmp_masked_video_qwen"
        # if os.path.exists(tmp_dir):
        #     shutil.rmtree(tmp_dir)
        # compose_video_from_numpy_frames(
        #     frames=rendered_frames,
        #     output_path=f"{tmp_dir}/rendered_video.mp4",
        #     fps=15,
        #     codec="libx264",
        # )
        # rendered_video = np.stack(rendered_frames)  # shape: (num_frames, H, W, 3)
        if len(rendered_frames) < 600:
            rendered_frames = rendered_frames[::2]
        elif len(rendered_frames) < 900:
            rendered_frames = rendered_frames[::3]
        else:
            rendered_frames = rendered_frames[::4]

        grouped_results = {}
        query_count = 0
        # process the video and text
        # 1) Build chat messages ONLY for the prompt template (keep your current structure)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template},
                    {"type": "video", "video": "in memory"},  # used for templating
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        metadata = build_video_metadata(num_frames=len(rendered_frames), fps=15)

        llm_inputs = {
            "prompt": inputs,
            "multi_modal_data": {
                "video": (rendered_frames, metadata),  # <-- key: include metadata
            },
        }
        while len(grouped_results.keys()) != 3 and query_count < self.max_query:
            outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
            query_count += 1
            grouped_results = self.post_process_function_output(outputs[0].outputs[0].text)

        return grouped_results
    
    def post_process_function_output(self, output_text: str) -> dict:
        try:
            grouped_results = eval(output_text)
        except (SyntaxError, NameError):
            print(f"Failed to evaluate output text: {output_text}")
            grouped_results = {}
        return grouped_results


class QwenTransformersVideoNarrator(VLMPrompter):
    """Qwen3-VL (or compatible) function prediction via Hugging Face Transformers only (no vLLM)."""

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        prompt_template: str = "",
        max_query: int = 10,
        device: Optional[str] = None,
    ):
        super().__init__(vlm_model, prompt_template, max_query)
        dev = (device or "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(dev)
        self.processor = AutoProcessor.from_pretrained(vlm_model, trust_remote_code=True)
        if dev == "cpu":
            self.model = AutoModelForImageTextToText.from_pretrained(
                vlm_model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model.to(self._torch_device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                vlm_model,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
            )

    def _infer_device(self):
        return next(self.model.parameters()).device

    def prompt_function(
        self,
        rgb_frame_list: List[np.ndarray],
        receptor_part_masks: np.ndarray,
        effector_part_masks: np.ndarray,
    ) -> Dict[str, str]:
        rendered_frames: list[np.ndarray] = []
        for i in range(len(rgb_frame_list)):
            rgb_frame = rgb_frame_list[i]
            receptor_mask = receptor_part_masks[i]
            effector_mask = effector_part_masks[i]
            receptor_color_mask = np.zeros_like(rgb_frame)
            receptor_color_mask[:, :, 1] = receptor_mask * 255
            effector_color_mask = np.zeros_like(rgb_frame)
            effector_color_mask[:, :, 0] = effector_mask * 255
            alpha = 0.5
            blended_frame = cv2.addWeighted(rgb_frame, 1 - alpha, receptor_color_mask, alpha, 0)
            blended_frame = cv2.addWeighted(blended_frame, 1 - alpha, effector_color_mask, alpha, 0)
            rendered_frames.append(blended_frame.astype(np.uint8))

        if len(rendered_frames) < 600:
            rendered_frames = rendered_frames[::2]
        elif len(rendered_frames) < 900:
            rendered_frames = rendered_frames[::3]
        else:
            rendered_frames = rendered_frames[::4]

        td = tempfile.mkdtemp(prefix="qwen_tf_masked_")
        video_path = os.path.join(td, "rendered_video.mp4")
        try:
            compose_video_from_numpy_frames(
                frames=rendered_frames,
                output_path=video_path,
                fps=15,
                codec="libx264",
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        dict(type="text", text=self.prompt_template),
                        dict(type="video", video=video_path),
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            dev = self._infer_device()
            inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
            grouped_results: dict = {}
            query_count = 0
            while len(grouped_results.keys()) != 3 and query_count < self.max_query:
                with torch.inference_mode():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
                generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
                generated_text = self.processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                query_count += 1
                grouped_results = self.post_process_function_output(generated_text)
            return grouped_results
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def post_process_function_output(self, output_text: str) -> dict:
        try:
            grouped_results = eval(output_text)
        except (SyntaxError, NameError):
            print(f"Failed to evaluate output text: {output_text}")
            grouped_results = {}
        return grouped_results


class QwenVideoNarrator(VLMPrompter):
    def __init__(self, vlm_model = "Qwen/Qwen3-VL-8B-Instruct", prompt_template = "", max_query = 10):
        super().__init__(vlm_model, prompt_template, max_query)
        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            vlm_model,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        # load the model
        self.model = LLM(
            model=vlm_model,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 1},
            tensor_parallel_size=2,
        )

        self.sampling_params = SamplingParams(max_tokens=1024)
    

    def prompt_function(self, rgb_frame_list: List[np.ndarray], receptor_part_masks: np.ndarray, effector_part_masks: np.ndarray) -> Dict[str, str]:
        rendered_frames = []
        for i in range(len(rgb_frame_list)):
            rgb_frame = rgb_frame_list[i]
            receptor_mask = receptor_part_masks[i]
            effector_mask = effector_part_masks[i]

            # Create color masks
            receptor_color_mask = np.zeros_like(rgb_frame)
            receptor_color_mask[:, :, 1] = receptor_mask * 255  # Green for receptor

            effector_color_mask = np.zeros_like(rgb_frame)
            effector_color_mask[:, :, 0] = effector_mask * 255  # Red for effector

            # Blend the original frame with the masks
            alpha = 0.5
            blended_frame = cv2.addWeighted(rgb_frame, 1 - alpha, receptor_color_mask, alpha, 0)
            blended_frame = cv2.addWeighted(blended_frame, 1 - alpha, effector_color_mask, alpha, 0)

            rendered_frames.append(blended_frame.astype(np.uint8))
        # tmp_dir = "tmp_masked_video_qwen"
        # if os.path.exists(tmp_dir):
        #     shutil.rmtree(tmp_dir)
        # compose_video_from_numpy_frames(
        #     frames=rendered_frames,
        #     output_path=f"{tmp_dir}/rendered_video.mp4",
        #     fps=15,
        #     codec="libx264",
        # )
        # rendered_video = np.stack(rendered_frames)  # shape: (num_frames, H, W, 3)

        grouped_results = {}
        query_count = 0
        # process the video and text
        # 1) Build chat messages ONLY for the prompt template (keep your current structure)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template},
                    {"type": "video", "video": "in memory"},  # used for templating
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        metadata = build_video_metadata(num_frames=len(rendered_frames), fps=15)

        llm_inputs = {
            "prompt": inputs,
            "multi_modal_data": {
                "video": (rendered_frames, metadata),  # <-- key: include metadata
            },
        }
        while len(grouped_results.keys()) != 3 and query_count < self.max_query:
            outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
            query_count += 1
            grouped_results = self.post_process_function_output(outputs[0].outputs[0].text)

        return grouped_results
    
    def post_process_function_output(self, output_text: str) -> dict:
        try:
            grouped_results = eval(output_text)
        except (SyntaxError, NameError):
            print(f"Failed to evaluate output text: {output_text}")
            grouped_results = {}
        return grouped_results
    

class VLMSegJudge(VLMPrompter):
    def __init__(self, vlm_model: str = "gemini-2.5-flash", prompt_template: str = "", max_query: int = 10):
        super().__init__(vlm_model, prompt_template, max_query)

    def overlay_mask(self, image: PILImage.Image, mask: np.ndarray) -> np.ndarray:
        image = np.array(image)
        # Convert mask to 3-channel color
        mask_color = np.zeros_like(image)
        mask_color[:, :, 1] = mask * 255  # color mask in green channel (you can pick R=0,G=1,B=2)

        # Blend image and mask
        alpha = 0.5  # transparency factor
        overlay = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)
        return overlay

    def prompt(self, image: PILImage.Image, mask: np.ndarray, part_description: str) -> dict:
        grouped_results = {}
        query_count = 0
        origin_image = np.array(image)
        overlay_image = self.overlay_mask(image, mask)
        _, encoded_overlay_buffer = cv2.imencode('.jpg', overlay_image)
        overlay_image_bytes = encoded_overlay_buffer.tobytes()
        _, encoded_origin_buffer = cv2.imencode('.jpg', origin_image)
        origin_image_bytes = encoded_origin_buffer.tobytes()
        while len(grouped_results.keys()) != 2 and query_count < self.max_query:

            response = self.client.models.generate_content(
                model=self.vlm_model, contents=genai.types.Content(
                    parts=[
                        genai.types.Part.from_bytes(
                            data=origin_image_bytes, mime_type='image/jpeg'
                        ),
                        genai.types.Part.from_bytes(
                            data=overlay_image_bytes, mime_type='image/jpeg'
                        ),
                        genai.types.Part(text=self.prompt_template.format(PartDescription=part_description))
                    ]
                )
            )
            grouped_results = self.post_process_output(response.text)
            query_count += 1
        return grouped_results


    def post_process_output(self, output_text: str) -> dict:
        pairs = re.findall(r'\s*<answer>\s*(.*?)\s*</answer>\s*<reason>\s*(.*?)\s*</reason>', output_text, flags=re.S)

        def clean(t: str) -> str:
            return re.sub(r'\s+', ' ', t).strip()

        grouped = {}
        if len(pairs) == 1:
            answer_c = clean(pairs[0][0])
            reason_c = clean(pairs[0][1])
            grouped = {"answer": answer_c, "reason": reason_c}
        else:
            print("Warning: Unexpected content found in VLM output.")
        return grouped


def build_vlm_prompter(vlm_config: dict) -> VLMPrompter:
    if vlm_config["role"] == "video_narrator":
        if vlm_config["vlm_type"] == "gemini":
            return GeminiVideoNarrator(
                vlm_model=vlm_config.vlm_model,
                prompt_template=vlm_config.prompt_template,
                max_query=vlm_config.max_query
            )
        elif vlm_config["vlm_type"] == "gpt":
            return GPTVideoNarrator(
                vlm_model=vlm_config.vlm_model,
                prompt_template=vlm_config.prompt_template,
                max_query=vlm_config.max_query
            )
        elif vlm_config["vlm_type"] == "molmo":
            return MolmovllmVideoNarrator(
                vlm_model=vlm_config.vlm_model,
                prompt_template=vlm_config.prompt_template,
                max_query=vlm_config.max_query
            )
        elif vlm_config["vlm_type"] == "qwen":
            return QwenVideoNarrator(
                vlm_model=vlm_config.vlm_model,
                prompt_template=vlm_config.prompt_template,
                max_query=vlm_config.max_query
            )
        elif vlm_config["vlm_type"] == "qwen_transformers":
            dev = OmegaConf.select(vlm_config, "device", default=None)
            return QwenTransformersVideoNarrator(
                vlm_model=vlm_config.vlm_model,
                prompt_template=vlm_config.prompt_template,
                max_query=vlm_config.max_query,
                device=dev,
            )
        else:
            raise ValueError(f"Unknown VLM type: {vlm_config['vlm_type']}")
    elif vlm_config["role"] == "segmentation_judge":
        return VLMSegJudge(
            vlm_model=vlm_config.vlm_model,
            prompt_template=vlm_config.prompt_template,
            max_query=vlm_config.max_query
        )
    else:
        raise ValueError(f"Unknown VLM role: {vlm_config['role']}")
