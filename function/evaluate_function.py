import json


PHYSICAL_EFFECT_MAP = {"a": "geometry", "b": "illumination", "c": "temperature", "d": "fluid"}
NUMERICAL_FUNCTION_MAP = {"a": "binary", "b": "step", "c": "linear", "d": "cumulative"}


def compute_function_error(gt_function: dict, pred_function: dict) -> dict:
    error_metrics = {}
    error_metrics["physical_effect"] = gt_function["physics"] == PHYSICAL_EFFECT_MAP[pred_function["1"]]
    error_metrics["numerical_function"] = gt_function["func"] == NUMERICAL_FUNCTION_MAP[pred_function["2"]]
    return error_metrics


def save_function_results(function_results: dict, save_path: str):
    with open(save_path, "w") as f:
        json.dump(function_results, f, indent=4)