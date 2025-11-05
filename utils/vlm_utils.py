import json


def save_vlm_results(vlm_results: dict, save_path: str):
    """
    Save VLM results to a JSON file.

    Args:
        vlm_results (dict): VLM results dictionary.
        save_path (str): Path to save the JSON file.
    """
    with open(save_path, "w") as f:
        json.dump(vlm_results, f, indent=4)