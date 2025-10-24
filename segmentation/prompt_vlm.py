from google import genai
import re


class VLM_Prompter:
    def __init__(self, vlm_model: str = "gemini-2.5-flash", prompt_template: str = ""):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.vlm_model = vlm_model
        self.prompt_template = prompt_template

    def prompt(self, video_path: str) -> dict:
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

        assert len(pairs) == 2, "Expected exactly two parts in the output."
        grouped = {}
        for i, (name, desc) in enumerate(pairs):
            name_c = clean(name)
            desc_c = clean(desc)
            role = "receiver" if i == 0 else "effector"
            grouped[role] = {"name": name_c, "description": desc_c}
        return grouped


def build_vlm_prompter(vlm_config: dict) -> VLM_Prompter:
    return VLM_Prompter(
        vlm_model=vlm_config.vlm_model,
        prompt_template=vlm_config.prompt_template
    )