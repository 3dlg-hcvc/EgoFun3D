from google import genai
import re
import cv2
import numpy as np
from PIL import Image as PILImage
import base64


class VLMPrompter:
    def __init__(self, vlm_model: str = "gemini-2.5-flash", prompt_template: str = "", max_query: int = 10):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.vlm_model = vlm_model
        self.prompt_template = prompt_template
        self.max_query = max_query

class VLMVideoNarrator(VLMPrompter):
    def __init__(self, vlm_model: str = "gemini-2.5-flash", prompt_template: str = "", max_query: int = 10):
        super().__init__(vlm_model, prompt_template, max_query)
    

    def prompt(self, video_path: str) -> dict:
        grouped_results = {}
        query_count = 0
        while len(grouped_results.keys()) != 2 and query_count < self.max_query:
            video_bytes = open(video_path, 'rb').read()
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
            grouped_results = self.post_process_output(response.text)
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
                role = "receiver" if i == 0 else "effector"
                grouped[role] = {"name": name_c, "description": desc_c}
        else:
            print("Warning: Unexpected number of parts found in VLM output.")
        return grouped


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
        return VLMVideoNarrator(
            vlm_model=vlm_config.vlm_model,
            prompt_template=vlm_config.prompt_template,
            max_query=vlm_config.max_query
        )
    elif vlm_config["role"] == "segmentation_judge":
        return VLMSegJudge(
            vlm_model=vlm_config.vlm_model,
            prompt_template=vlm_config.prompt_template,
            max_query=vlm_config.max_query
        )
    else:
        raise ValueError(f"Unknown VLM role: {vlm_config['role']}")