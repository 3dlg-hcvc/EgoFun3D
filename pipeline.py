"""
Gradio demo: video upload → segmentation, DA3, fusion, iTACO, optional function VLM (Gemini API or Qwen via Transformers).

Segmentation preview MP4: receptor mask = green-teal tint, effector = orange tint (seaborn Set2 swatches);
areas outside both masks darkened; mask boundaries outlined in light gray. Requires ``ffmpeg`` and ``seaborn``.

``DA3DReconstruction`` in ``fusion/reconstruction.py`` expects uint8 ``HxWx3`` numpy frames; this script uses
``video_frames_as_numpy_hwc_uint8`` before DA3 / SAM3 / overlay.

3D viewer: ``trimesh`` exports one temporary ``.glb`` (part points as small icospheres for visible size, plus iTACO joint axes as arrows).
``origin`` / ``axis`` from iTACO live in the same DA3 world frame as ``recon_results["points"]``. Requires
``trimesh`` and ``matplotlib`` (colormap for point colors). ``gr.Model3D`` displays the GLB.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import cv2
import gradio as gr
import numpy as np
import omegaconf
import torch
import seaborn as sns
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation

from articulation.base import build_articulation_estimation_model
from fusion.fusion import build_fusion_model
from fusion.reconstruction import build_reconstruction_model
from segmentation.ref_seg import RefSeg, build_refseg_model
from segmentation.workflow import (
    build_sam3_tracker,
    evenly_spaced_indices,
    propagate_full_video_from_masks,
    to_pil_rgb,
)
from VLM.prompt_vlm import build_vlm_prompter

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_GRADIO_PORT = 7868

# Fixed pipeline (matches eval defaults): all frames loaded; SegZero on 20 seeds; SAM3 → full clip.
SEGMENTATION_SEED_FRAMES = 20
SEGMENTATION_OVERLAY_FALLBACK_FPS = 10

# GLB viewers draw glTF POINTS as 1-pixel; use small icospheres so the cloud stays visible without dominating the scene.
GLB_POINT_PROXY_RADIUS_FRAC = 0.0065
GLB_POINT_PROXY_MAX_SPHERES = 16_000
GLB_POINT_ICOSPHERE_SUBDIV = 1

# Arrow shaft + cone (trimesh ``cone`` has base at z=0, tip at z=height; meet shaft top at z=h_shaft).
ARROW_MESH_SECTIONS = 40

# Fixed model configurations (not exposed in UI).
_DEVICE = "cuda"
_REASONING_MODEL_PATH = "Ricky06662/VisionReasoner-7B"
_SEGMENTATION_MODEL_PATH = "facebook/sam3"
_SAM_BACKEND = "sam3"
_DA3_MODEL_PATH = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
_QWEN_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

_MODEL_CACHE: dict[str, Any] = {}
_da3_chunk_cfg: Any = None
_itaco_yaml_cfg_cache: Any = None


def _clear_model_cache_and_empty_cuda() -> None:
    """Clear the demo model cache and CUDA allocator. Call only after dropping local handles to large models."""
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_seg_sam3_from_cache(
    device: str,
    reasoning_model_path: str,
    segmentation_model_path: str,
    sam_backend: str,
    sam3_checkpoint_path: str,
    sam3_bpe_path: str,
) -> None:
    ck = (sam3_checkpoint_path or "").strip() or ""
    bp = (sam3_bpe_path or "").strip() or ""
    for k in (
        f"seg|{device}|{reasoning_model_path}|{segmentation_model_path}|{sam_backend}",
        f"sam3_vid|{device}|{ck}|{bp}",
    ):
        _MODEL_CACHE.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_da3_from_cache(device: str, da3_model_path: str) -> None:
    k = f"da3|{device}|{da3_model_path}|{_da3_max_pred_frame()}"
    _MODEL_CACHE.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_fusion_from_cache(device: str) -> None:
    _MODEL_CACHE.pop(f"fusion|{device}", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _config_path(*parts: str) -> str:
    return os.path.join(_ROOT, "config", *parts)


def _da3_max_pred_frame() -> int:
    global _da3_chunk_cfg
    if _da3_chunk_cfg is None:
        p = _config_path("reconstruction", "da3.yaml")
        _da3_chunk_cfg = (
            omegaconf.OmegaConf.load(p)
            if os.path.isfile(p)
            else omegaconf.OmegaConf.create({"max_pred_frame": 20})
        )
    return int(_da3_chunk_cfg.get("max_pred_frame", 20))


def _expand_seed_outputs_to_full_video(
    full_frame_list: list[np.ndarray],
    seed_frame_ids: list[int],
    seed_pred_masks: list[np.ndarray],
    seed_answer_dicts: list[dict],
    seed_valid_frame_ids: list[int],
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    total_frames = len(full_frame_list)
    full_pred_mask_list: list[np.ndarray | None] = [None] * total_frames
    full_answer_dict_list: list[dict | None] = [None] * total_frames
    local_to_full = {local_idx: frame_id for local_idx, frame_id in enumerate(seed_frame_ids)}

    for local_idx, frame_id in local_to_full.items():
        full_pred_mask_list[frame_id] = np.asarray(seed_pred_masks[local_idx]).astype(bool)
        answer_dict = dict(seed_answer_dicts[local_idx] if isinstance(seed_answer_dicts[local_idx], dict) else {})
        answer_dict.setdefault("frame_id", frame_id)
        answer_dict["seed_frame"] = True
        answer_dict["propagated"] = False
        answer_dict["subsampled"] = len(seed_frame_ids) != total_frames
        full_answer_dict_list[frame_id] = answer_dict

    for frame_id in range(total_frames):
        if full_pred_mask_list[frame_id] is None:
            frame = full_frame_list[frame_id]
            full_pred_mask_list[frame_id] = np.zeros(frame.shape[:2], dtype=bool)
            full_answer_dict_list[frame_id] = {
                "frame_id": frame_id,
                "seed_frame": False,
                "propagated": False,
                "subsampled": True,
                "skipped": True,
                "skip_reason": "frame_subsample",
            }

    valid_frame_ids = [local_to_full[local_idx] for local_idx in seed_valid_frame_ids if local_idx in local_to_full]
    return [np.asarray(mask).astype(bool) for mask in full_pred_mask_list], [dict(answer) for answer in full_answer_dict_list], valid_frame_ids


def _propagate_seed_masks_to_full_video(
    full_frame_list: list[np.ndarray],
    seed_frame_ids: list[int],
    seed_pred_masks: list[np.ndarray],
    seed_answer_dicts: list[dict],
    seed_valid_frame_ids: list[int],
    sam3_tracker,
    offload_video_to_cpu: bool = False,
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    total_frames = len(full_frame_list)
    valid_seed_mask_entries: list[tuple[int, np.ndarray]] = []
    valid_seed_frame_ids: list[int] = []
    for local_idx in seed_valid_frame_ids:
        if local_idx < 0 or local_idx >= len(seed_frame_ids):
            continue
        full_frame_id = int(seed_frame_ids[local_idx])
        mask = np.asarray(seed_pred_masks[local_idx]).astype(bool)
        if int(mask.sum()) <= 0:
            continue
        valid_seed_mask_entries.append((full_frame_id, mask))
        valid_seed_frame_ids.append(full_frame_id)

    if len(valid_seed_mask_entries) > 0:
        propagated_masks = propagate_full_video_from_masks(
            sam3_tracker,
            full_frame_list,
            valid_seed_mask_entries,
            offload_video_to_cpu=offload_video_to_cpu,
        )
    else:
        propagated_masks = [np.zeros(frame.shape[:2], dtype=bool) for frame in full_frame_list]

    seed_answer_map = {}
    for local_idx, frame_id in enumerate(seed_frame_ids):
        answer_dict = dict(seed_answer_dicts[local_idx] if isinstance(seed_answer_dicts[local_idx], dict) else {})
        answer_dict.setdefault("frame_id", int(frame_id))
        answer_dict["seed_frame"] = True
        answer_dict["propagated"] = False
        answer_dict["subsampled"] = len(seed_frame_ids) != total_frames
        seed_answer_map[int(frame_id)] = answer_dict

    full_answer_dicts = []
    seed_frame_id_set = set(int(frame_id) for frame_id in seed_frame_ids)
    for frame_id in range(total_frames):
        if frame_id in seed_answer_map:
            answer_dict = dict(seed_answer_map[frame_id])
        else:
            answer_dict = {
                "frame_id": frame_id,
                "seed_frame": False,
                "propagated": True,
                "subsampled": len(seed_frame_ids) != total_frames,
                "propagation_source": "sam3_mask_prompt_propagation",
            }
        answer_dict["upsampled"] = True
        answer_dict["upsample_method"] = "sam3_mask_prompt_propagation"
        answer_dict["upsample_seed_frame_ids"] = [int(i) for i in valid_seed_frame_ids]
        answer_dict["is_seed_frame"] = frame_id in seed_frame_id_set
        full_answer_dicts.append(answer_dict)

    valid_frame_ids = [frame_id for frame_id, mask in enumerate(propagated_masks) if int(np.asarray(mask).astype(bool).sum()) > 0]
    return [np.asarray(mask).astype(bool) for mask in propagated_masks], full_answer_dicts, valid_frame_ids


def _segment_one_role(
    refseg: RefSeg,
    full_rgb_np: list[np.ndarray],
    part_description: str,
    sam3_tracker,
    log,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    total = len(full_rgb_np)
    seed_frame_ids = evenly_spaced_indices(total, SEGMENTATION_SEED_FRAMES)
    log(f"  Seed frame indices ({len(seed_frame_ids)} of {total}): {seed_frame_ids[:8]}{'…' if len(seed_frame_ids) > 8 else ''}")
    seed_input_frames = [to_pil_rgb(full_rgb_np[i]) for i in seed_frame_ids]
    seed_pred_masks, seed_answer_dicts, seed_valid_frame_ids = refseg.segment_video(seed_input_frames, part_description)

    should_propagate = len(seed_frame_ids) < total
    if should_propagate:
        masks, _answers, valid_ids = _propagate_seed_masks_to_full_video(
            full_rgb_np,
            seed_frame_ids,
            seed_pred_masks,
            seed_answer_dicts,
            seed_valid_frame_ids,
            sam3_tracker,
        )
        log(f"  SAM3 propagation applied ({total} frames); non-empty mask frames: {len(valid_ids)}.")
    else:
        masks, _answers, valid_ids = _expand_seed_outputs_to_full_video(
            full_rgb_np,
            seed_frame_ids,
            seed_pred_masks,
            seed_answer_dicts,
            seed_valid_frame_ids,
        )
        log(f"  No SAM3 propagation (subsample covers all frames or disabled); non-empty: {len(valid_ids)}.")

    return masks, valid_ids, seed_frame_ids


def _numpy_json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


# Function VLM template (gemini_function / qwen_function): Q1 = physical effect, Q2 = numerical relationship.
_FUNCTION_VLM_PHYSICAL = {
    "a": "geometry change",
    "b": "illumination change",
    "c": "temperature change",
    "d": "fluid change",
}
_FUNCTION_VLM_NUMERICAL = {
    "a": "binary function",
    "b": "step function",
    "c": "linear function",
    "d": "cumulative function",
}


def _function_vlm_choice_letter(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    for ch in ("a", "b", "c", "d"):
        if s == ch or s.startswith(f"{ch})") or s.startswith(f"({ch}"):
            return ch
    if len(s) >= 1 and s[0] in _FUNCTION_VLM_PHYSICAL:
        return s[0]
    return None


def _enrich_function_vlm_result(raw: Any) -> dict[str, Any]:
    """Map option letters to labels; keep raw model dict for traceability."""
    if raw is None:
        return {"result": None, "note": "VLM returned None"}
    if isinstance(raw, str):
        t = raw.strip()
        try:
            raw = json.loads(t)
        except json.JSONDecodeError:
            return {"raw_text": t}
    if not isinstance(raw, dict):
        return {"raw": raw}

    q1 = raw.get("1", raw.get("question_1"))
    q2 = raw.get("2", raw.get("question_2"))
    rsn = raw.get("reason", raw.get("Reason"))
    l1 = _function_vlm_choice_letter(q1)
    l2 = _function_vlm_choice_letter(q2)

    return {
        "summary": {
            "physical_effect": _FUNCTION_VLM_PHYSICAL.get(l1) if l1 else None,
            "numerical_relationship": _FUNCTION_VLM_NUMERICAL.get(l2) if l2 else None,
        },
        "choices": {
            "1_physical_effect": {
                "letter": l1,
                "label": _FUNCTION_VLM_PHYSICAL.get(l1),
                "raw": q1,
            },
            "2_numerical_relationship": {
                "letter": l2,
                "label": _FUNCTION_VLM_NUMERICAL.get(l2),
                "raw": q2,
            },
        },
        "reason": rsn,
        "raw_model_output": dict(raw),
    }


def _function_vlm_display_json(raw: Any) -> str:
    return json.dumps(_enrich_function_vlm_result(raw), indent=2, default=_numpy_json_default)


def _function_vlm_summary_html(function_json_text: str) -> str:
    """Render a compact card row with physical-effect and numerical-relationship badges."""
    try:
        data = json.loads(function_json_text)
    except (json.JSONDecodeError, TypeError):
        return ""
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    physical = (summary.get("physical_effect") or "—").replace("<", "&lt;")
    numerical = (summary.get("numerical_relationship") or "—").replace("<", "&lt;")
    reason_raw = data.get("reason") or "" if isinstance(data, dict) else ""
    reason = str(reason_raw).replace("<", "&lt;") if reason_raw else ""
    reason_block = (
        f'<div style="flex:2;background:#fff;border:1px solid #d9e3e3;border-radius:10px;'
        f'padding:14px 18px;">'
        f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:4px;">Reason</div>'
        f'<div style="color:#333;font-size:0.92rem;line-height:1.4;">{reason}</div>'
        f"</div>"
    ) if reason else ""
    return (
        '<div style="display:flex;gap:14px;padding:14px 0 4px;font-family:system-ui,sans-serif;align-items:stretch;">'
        '<div style="flex:1;background:rgb(100,175,220);color:#fff;'
        'padding:16px 20px;border-radius:10px;text-align:center;">'
        '<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;opacity:0.85;margin-bottom:6px;">Physical Effect</div>'
        f'<div style="font-size:1.15rem;font-weight:700;">{physical}</div>'
        "</div>"
        '<div style="flex:1;background:rgb(110,39,107);color:#fff;'
        'padding:16px 20px;border-radius:10px;text-align:center;">'
        '<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;opacity:0.85;margin-bottom:6px;">Numerical Relationship</div>'
        f'<div style="font-size:1.15rem;font-weight:700;">{numerical}</div>'
        "</div>"
        + reason_block
        + "</div>"
    )


def load_full_video(video_path: str) -> tuple[list[PILImage.Image], list[np.ndarray], float]:
    """Decode every frame in order (no subsampling, no frame cap).

    Third return value is container FPS from OpenCV (``CAP_PROP_FPS``), or
    ``SEGMENTATION_OVERLAY_FALLBACK_FPS`` when missing or invalid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS))
    if not (fps_raw > 0.0) or fps_raw != fps_raw:
        fps_raw = float(SEGMENTATION_OVERLAY_FALLBACK_FPS)
    frames_bgr: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames_bgr.append(fr)
    cap.release()
    if not frames_bgr:
        raise RuntimeError("No frames read from video.")
    pil_list: list[PILImage.Image] = []
    rgb_list: list[np.ndarray] = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_list.append(PILImage.fromarray(rgb))
        rgb_list.append(rgb.astype(np.uint8))
    return pil_list, rgb_list, fps_raw


def video_frames_as_numpy_hwc_uint8(frames: list) -> list[np.ndarray]:
    """``fusion/reconstruction.DA3DReconstruction`` uses ``frame.shape[0/1]`` — requires ndarray (H,W,3) uint8, not PIL."""
    out: list[np.ndarray] = []
    for f in frames:
        if isinstance(f, PILImage.Image):
            out.append(np.asarray(f.convert("RGB"), dtype=np.uint8))
            continue
        arr = np.asarray(f)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB frame, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out.append(arr)
    return out


def get_seg_model(
    device: str,
    reasoning_model_path: str,
    segmentation_model_path: str,
    sam_backend: str,
) -> RefSeg:
    key = f"seg|{device}|{reasoning_model_path}|{segmentation_model_path}|{sam_backend}"
    if key not in _MODEL_CACHE:
        cfg = omegaconf.OmegaConf.create(
            {
                "model": "SegZero",
                "sam_backend": sam_backend,
                "reasoning_model_path": reasoning_model_path,
                "segmentation_model_path": segmentation_model_path,
                "sam3_confidence_threshold": 0.5,
                "device": device,
                "max_query": 10,
                "disable_vlm_judge": True,
                "vlm_judge": {
                    "role": "segmentation_judge",
                    "vlm_model": "gemini-2.5-flash",
                    "prompt_template": "noop {PartDescription}",
                    "max_query": 1,
                },
            }
        )
        _MODEL_CACHE[key] = build_refseg_model(cfg)
    return _MODEL_CACHE[key]


def get_sam3_tracker(device: str, checkpoint_path: str | None, bpe_path: str | None):
    ck = checkpoint_path or ""
    bp = bpe_path or ""
    key = f"sam3_vid|{device}|{ck}|{bp}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_sam3_tracker(
            device=device,
            checkpoint_path=checkpoint_path if checkpoint_path else None,
            bpe_path=bpe_path if bpe_path else None,
        )
    return _MODEL_CACHE[key]


def get_recon_model(device: str, da3_model_path: str):
    key = f"da3|{device}|{da3_model_path}|{_da3_max_pred_frame()}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_reconstruction_model(
            input_modality="rgb",
            recon_method="da3",
            model_path=da3_model_path,
            device=device,
            max_pred_frame=_da3_max_pred_frame(),
        )
    return _MODEL_CACHE[key]


def get_fusion_model(device: str):
    key = f"fusion|{device}"
    if key not in _MODEL_CACHE:
        cfg = omegaconf.OmegaConf.create(
            {"fusion_method": "feature_matching", "device": device}
        )
        _MODEL_CACHE[key] = build_fusion_model(cfg)
    return _MODEL_CACHE[key]


def _itaco_yaml_cfg():
    global _itaco_yaml_cfg_cache
    if _itaco_yaml_cfg_cache is None:
        _itaco_yaml_cfg_cache = omegaconf.OmegaConf.load(_config_path("articulation", "iTACO.yaml"))
    return _itaco_yaml_cfg_cache


def _subsample_for_itaco(
    rgb_frame_list: list[np.ndarray],
    recon_results: dict[str, Any],
    part_masks: np.ndarray,
    sampling_cfg: omegaconf.DictConfig,
) -> tuple[list[np.ndarray], dict[str, Any], np.ndarray]:
    """Match ``iTACO.articulation_estimation`` frame subsampling, then force C-contiguous arrays.

    ``iTACORefine.optimize_joint`` uses ``torch.from_numpy(points).view(-1, 3)``, which requires
    contiguous storage; ``recon_results["points"][::step]`` is a strided (non-contiguous) view.
    """
    strat = str(sampling_cfg.get("sample_strategy", "fix_step"))
    sn = int(sampling_cfg.get("sample_num", 10))
    n = len(rgb_frame_list)
    if strat == "fix_num":
        if n > sn:
            step = max(n // sn, 1)
            sl = slice(None, None, step)
        else:
            sl = slice(None)
    elif strat == "fix_step":
        sl = slice(None, None, sn)
    else:
        sl = slice(None)
    rgb_s = rgb_frame_list[sl]
    recon_out = {
        "depth": np.ascontiguousarray(np.asarray(recon_results["depth"])[sl]),
        "extrinsics": np.ascontiguousarray(np.asarray(recon_results["extrinsics"])[sl]),
        "points": np.ascontiguousarray(np.asarray(recon_results["points"])[sl]),
    }
    masks_s = np.asarray(part_masks)[sl]
    return rgb_s, recon_out, masks_s


def get_itaco(device: str):
    """iTACO with internal subsampling disabled; ``run_pipeline`` subsamples via ``_subsample_for_itaco``."""
    key = f"itaco|{device}|demo_presubsampled"
    if key not in _MODEL_CACHE:
        cfg = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(_itaco_yaml_cfg(), resolve=True)
        )
        cfg.device = device
        cfg.sample_strategy = "fix_num"
        cfg.sample_num = 2**30
        _MODEL_CACHE[key] = build_articulation_estimation_model(cfg)
    return _MODEL_CACHE[key]


def _align_mask_to_hw(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.shape[:2] == (height, width):
        return m
    u8 = m.astype(np.uint8) * 255
    resized = cv2.resize(u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 127


def _apply_color_tint(out: np.ndarray, mask, color: np.ndarray, alpha: float):
    if mask is not None and mask.any():
        m = mask.astype(bool)
        for c in range(3):
            out[m, c] = (1 - alpha) * out[m, c] + alpha * color[c]


def _apply_outlines(
    out: np.ndarray,
    receiver_mask,
    effector_mask,
    thickness: int = 6,
    color=(215, 215, 215),
):
    struct = np.ones((3, 3), dtype=bool)
    for mask in (receiver_mask, effector_mask):
        if mask is None or not mask.any():
            continue
        m = mask.astype(bool)
        dilated = binary_dilation(m, structure=struct, iterations=thickness)
        outline = dilated & ~m
        for c, val in enumerate(color):
            out[outline, c] = val


def overlay_masks_pop(
    rgb: np.ndarray,
    receiver_mask,
    effector_mask,
    alpha: float = 0.65,
    darken: float = 0.7,
    outline_thickness: int = 6,
    outline_color=(215, 215, 215),
) -> np.ndarray:
    """Receptor = Set2[0] green-teal tint, effector = Set2[1] orange; background darkened; light gray mask outlines."""
    palette = sns.color_palette("Set2")
    recv_color = np.array(palette[0]) * 255
    eff_color = np.array(palette[1]) * 255

    out = rgb.astype(float).copy()

    highlighted = np.zeros(rgb.shape[:2], dtype=bool)
    if receiver_mask is not None:
        highlighted |= receiver_mask.astype(bool)
    if effector_mask is not None:
        highlighted |= effector_mask.astype(bool)
    out[~highlighted] *= darken

    _apply_color_tint(out, receiver_mask, recv_color, alpha)
    _apply_color_tint(out, effector_mask, eff_color, alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    _apply_outlines(out, receiver_mask, effector_mask, outline_thickness, outline_color)
    return out


def frames_to_overlay_mp4(
    rgb_frames: list[np.ndarray],
    receiver_masks: list[np.ndarray],
    effector_masks: list[np.ndarray],
    output_mp4_path: str,
    fps: float = 10.0,
) -> bool:
    """Encode overlay frames (see ``overlay_masks_pop``) to MP4 via ffmpeg."""
    n = len(rgb_frames)
    if n <= 0:
        return False
    with tempfile.TemporaryDirectory(prefix="pfr_seg_vis_") as tmpdir:
        for i in range(n):
            h, w = rgb_frames[i].shape[:2]
            recv = _align_mask_to_hw(receiver_masks[i], h, w)
            eff = _align_mask_to_hw(effector_masks[i], h, w)
            overlay = overlay_masks_pop(rgb_frames[i], recv, eff)
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            PILImage.fromarray(overlay).save(frame_path)
        pattern = os.path.join(os.path.abspath(tmpdir), "frame_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            pattern,
            "-frames:v",
            str(n),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_mp4_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            err = getattr(e, "stderr", None)
            if err:
                sys.stderr.write(err.decode() if isinstance(err, bytes) else str(err))
            return False


def _itaco_joint_arrow_mesh(
    origin: np.ndarray,
    direction: np.ndarray,
    height: float,
    shaft_radius: float,
    head_radius: float,
):
    """Arrow along local +Z with base at origin; rotate +Z → ``direction``, then place base at ``origin``.

    Uses ``trimesh.transformations`` + separate rotate / translate so row-vector vertices match trimesh.
    """
    import trimesh
    import trimesh.transformations as tft

    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    head_frac = 0.28
    h_shaft = height * (1.0 - head_frac)
    h_head = height * head_frac
    sec = ARROW_MESH_SECTIONS
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=h_shaft, sections=sec)
    shaft.apply_translation([0.0, 0.0, h_shaft * 0.5])
    head = trimesh.creation.cone(radius=head_radius, height=h_head, sections=sec)
    # Cone base (z=0) flush with cylinder top (z=h_shaft); tip at z=h_shaft+h_head.
    head.apply_translation([0.0, 0.0, h_shaft])
    arrow = trimesh.util.concatenate([shaft, head])

    z = np.array([0.0, 0.0, 1.0])
    d = direction
    axis = np.cross(z, d)
    la = float(np.linalg.norm(axis))
    if la < 1e-12:
        r44 = np.eye(4) if d[2] > 0 else tft.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        axis = axis / la
        ang = float(np.arctan2(la, float(np.dot(z, d))))
        r44 = tft.rotation_matrix(ang, axis)
    arrow.apply_transform(r44)
    arrow.apply_translation(origin)
    return arrow


def _masked_world_points_xyz(recon_results: dict[str, Any]) -> np.ndarray | None:
    chunks: list[np.ndarray] = []
    pm = recon_results["points_mask"]
    plist = recon_results["points"]
    for i in range(len(plist)):
        p = np.asarray(plist[i])
        m = np.asarray(pm[i], dtype=bool)
        if m.any():
            chunks.append(p[m].reshape(-1, 3))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _masked_world_points_xyz_for_masks(
    recon_results: dict[str, Any], masks_per_frame: list[np.ndarray]
) -> np.ndarray | None:
    """DA3 world points where ``points_mask`` ∧ resized segmentation mask (per frame)."""
    chunks: list[np.ndarray] = []
    pm = recon_results["points_mask"]
    plist = recon_results["points"]
    for i in range(len(plist)):
        p = np.asarray(plist[i])
        h, w = int(p.shape[0]), int(p.shape[1])
        m = np.asarray(pm[i], dtype=bool)
        if i < len(masks_per_frame):
            rm = _align_mask_to_hw(masks_per_frame[i], h, w).astype(bool)
            m = np.logical_and(m, rm)
        if m.any():
            chunks.append(p[m].reshape(-1, 3))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _point_cloud_proxy_spheres(
    vertices: np.ndarray,
    colors_rgba: np.ndarray,
    sphere_radius: float,
    rng: np.random.Generator,
    max_spheres: int,
):
    """One icosphere per point (subsampled to ``max_spheres``) for visible GLB point size."""
    import trimesh

    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(colors_rgba, dtype=np.uint8).reshape(-1, 4)
    n = int(v.shape[0])
    if n == 0:
        return None
    if n > max_spheres:
        sel = rng.choice(n, size=max_spheres, replace=False)
        v, c = v[sel], c[sel]
        n = max_spheres
    template = trimesh.creation.icosphere(
        subdivisions=GLB_POINT_ICOSPHERE_SUBDIV, radius=float(sphere_radius)
    )
    pieces: list = []
    for i in range(n):
        s = template.copy()
        s.apply_translation(v[i])
        s.visual.face_colors = np.tile(c[i : i + 1], (len(s.faces), 1))
        pieces.append(s)
    return trimesh.util.concatenate(pieces)


def _export_recon_articulation_glb(
    recon_results: dict[str, Any],
    art_out: dict[str, Any],
    fused_by_role: dict[str, np.ndarray | None] | None,
    segmentation_masks_by_role: dict[str, list[np.ndarray]] | None,
    max_pts: int = 120_000,
) -> tuple[str | None, np.ndarray | None]:
    """GLB: per role, RoMa fused cloud if available else DA3 points under that role's mask; else one viridis DA3 cloud; + iTACO arrows.

    Returns ``(glb_path, scene_center_world)`` for aligning the viewer with frame-0 camera; center is mean of extent points.
    """
    import trimesh
    from matplotlib import cm
    from matplotlib.colors import Normalize

    role_rgb_pt = {
        "receptor": np.array([102, 194, 165, 255], dtype=np.uint8),
        "effector": np.array([252, 141, 98, 255], dtype=np.uint8),
    }
    geometries: list = []
    rng = np.random.default_rng(0)
    extent_basis: list[np.ndarray] = []
    point_layers: list[tuple[np.ndarray, np.ndarray]] = []

    fused = fused_by_role or {}
    masks_by_role = segmentation_masks_by_role or {}
    per_role_cap = max(2000, max_pts // 2)

    # Bake the OpenCV → glTF coordinate flip into every vertex.
    # DA3 uses OpenCV convention (Y-down, Z-into-scene); glTF / three.js expects Y-up,
    # Z-toward-viewer.  Flipping Y and Z is a 180° rotation around X (det=+1, winding
    # preserved).  Baking into vertex data (rather than a scene-level node transform) is
    # required so that three.js's OrbitControls bounding-sphere fit — and therefore the
    # viewer's built-in reset button — land in the correct position.
    _cv2gl = np.array([1.0, -1.0, -1.0], dtype=np.float64)

    for role in ("receptor", "effector"):
        pts_src: np.ndarray | None = None
        arr = fused.get(role)
        if arr is not None and np.asarray(arr).size > 0:
            pts_src = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        else:
            ml = masks_by_role.get(role)
            if ml:
                pts_src = _masked_world_points_xyz_for_masks(recon_results, ml)
        if pts_src is None or pts_src.shape[0] == 0:
            continue
        extent_basis.append(pts_src)
        n_i = min(per_role_cap, pts_src.shape[0])
        idx = rng.choice(pts_src.shape[0], size=n_i, replace=False)
        pts_i = pts_src[idx] * _cv2gl
        c = role_rgb_pt[role]
        rgba_i = np.tile(c, (n_i, 1))
        point_layers.append((pts_i, rgba_i))

    if not point_layers:
        all_pts = _masked_world_points_xyz(recon_results)
        if all_pts is None or all_pts.shape[0] == 0:
            return None, None
        extent_basis.append(all_pts)
        n = min(max_pts, all_pts.shape[0])
        idx = rng.choice(all_pts.shape[0], size=n, replace=False)
        pts = all_pts[idx].astype(np.float64)
        z = pts[:, 2]  # depth in original OpenCV frame for colormap, before flip
        norm = Normalize(vmin=float(z.min()), vmax=float(z.max()))
        rgb = (cm.viridis(norm(z))[:, :3] * 255.0).astype(np.uint8)
        rgba = np.concatenate([rgb, np.full((n, 1), 255, dtype=np.uint8)], axis=1)
        point_layers.append((pts * _cv2gl, rgba))

    all_ext = np.concatenate(extent_basis, axis=0)
    extent_pre = float(np.linalg.norm(all_ext.max(axis=0) - all_ext.min(axis=0))) + 1e-6
    sphere_r = max(extent_pre * GLB_POINT_PROXY_RADIUS_FRAC, extent_pre * 1e-4)
    n_layers = len(point_layers)
    per_layer_cap = max(1, GLB_POINT_PROXY_MAX_SPHERES // max(n_layers, 1))
    for pts_i, rgba_i in point_layers:
        proxy = _point_cloud_proxy_spheres(pts_i, rgba_i, sphere_r, rng, per_layer_cap)
        if proxy is not None:
            geometries.append(proxy)

    extent = extent_pre
    arrow_len = extent * 0.2
    shaft_d = max(arrow_len * 0.04, extent * 0.006)
    head_d = shaft_d * 2.8
    role_rgba_mesh = {
        "receptor": np.array([102, 194, 165, 255], dtype=np.uint8),
        "effector": np.array([252, 141, 98, 255], dtype=np.uint8),
    }

    for role in ("receptor", "effector"):
        est = art_out.get(role)
        if est is None or not isinstance(est, dict):
            continue
        if "axis" not in est or "origin" not in est:
            continue
        axis = np.asarray(est["axis"], dtype=np.float64).reshape(-1)[:3]
        origin = np.asarray(est["origin"], dtype=np.float64).reshape(-1)[:3]
        an = float(np.linalg.norm(axis))
        if an < 1e-12:
            continue
        direction = (axis / an) * _cv2gl   # flip direction vector to glTF frame
        origin_t = origin * _cv2gl          # flip origin to glTF frame
        arrow = _itaco_joint_arrow_mesh(
            origin_t, direction, arrow_len, shaft_radius=shaft_d, head_radius=head_d
        )
        c = role_rgba_mesh[role]
        arrow.visual.face_colors = np.tile(c, (len(arrow.faces), 1))
        geometries.append(arrow)

    scene_center = all_ext.mean(axis=0).astype(np.float64)
    scene = trimesh.Scene(geometries)
    fd, out_path = tempfile.mkstemp(suffix=".glb", prefix="pfr_scene_")
    os.close(fd)
    scene.export(out_path)
    return out_path, scene_center


def run_pipeline(
    video_file: str | None,
    receptor_label: str,
    effector_label: str,
    progress=gr.Progress(track_tqdm=True),
):
    device = _DEVICE
    reasoning_model_path = _REASONING_MODEL_PATH
    segmentation_model_path = _SEGMENTATION_MODEL_PATH
    sam_backend = _SAM_BACKEND
    da3_model_path = _DA3_MODEL_PATH
    sam3_checkpoint_path: str | None = None
    sam3_bpe_path: str | None = None

    log_lines: list[str] = []
    seg_video_path: str | None = None
    recon_scene_glb_path: str | None = None
    articulation_text = ""
    function_text = ""
    function_summary = ""

    def log(msg: str):
        log_lines.append(msg)

    def emit(
        recon_3d=None,
        articulation_val: str | None = None,
        function_val: str | None = None,
        summary_val: str | None = None,
    ):
        return (
            "\n".join(log_lines),
            seg_video_path,
            recon_3d,
            summary_val if summary_val is not None else function_summary,
            articulation_text if articulation_val is None else articulation_val,
            function_text if function_val is None else function_val,
        )

    try:
        if not video_file:
            log("Error: upload a video.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return

        if isinstance(video_file, str):
            path = video_file
        elif isinstance(video_file, dict):
            path = video_file.get("path") or video_file.get("name")
        else:
            path = getattr(video_file, "name", None)
        if not path or not os.path.isfile(path):
            log("Error: invalid upload.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return

        receptor_label = (receptor_label or "").strip()
        effector_label = (effector_label or "").strip()
        if not receptor_label or not effector_label:
            log("Error: provide receptor and effector part descriptions.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return

        log("Loading full video (every frame, in order)…")
        pil_frames, _rgb_unused, input_video_fps = load_full_video(path)
        rgb_np = video_frames_as_numpy_hwc_uint8(pil_frames)
        total_f = len(pil_frames)
        log(f"Loaded {total_f} frames. Segmentation: {SEGMENTATION_SEED_FRAMES} evenly spaced seeds, SAM3 propagates to all frames when N > seeds.")

        init_extrinsic = np.eye(4, dtype=np.float64)

        log("Loading / caching SegZero (VisionReasoner)…")
        refseg = get_seg_model(device, reasoning_model_path, segmentation_model_path, sam_backend)

        seed_ids_preview = evenly_spaced_indices(total_f, SEGMENTATION_SEED_FRAMES)
        should_propagate = len(seed_ids_preview) < total_f
        sam3_tracker = None
        if should_propagate:
            log("Loading / caching SAM3 video tracker (mask propagation)…")
            sam3_tracker = get_sam3_tracker(device, sam3_checkpoint_path, sam3_bpe_path)

        receptor_desc = f"{receptor_label}. {receptor_label}"
        effector_desc = f"{effector_label}. {effector_label}"

        log("Segmenting receptor (SegZero on seed frames; SAM3 to full video when N > seeds)…")
        r_masks, r_valid, _r_seeds = _segment_one_role(
            refseg,
            rgb_np,
            receptor_desc,
            sam3_tracker,
            log,
        )
        log(f"Receptor: {len(r_valid)} frames with non-empty mask after propagation.")

        log("Segmenting effector…")
        e_masks, e_valid, _e_seeds = _segment_one_role(
            refseg,
            rgb_np,
            effector_desc,
            sam3_tracker,
            log,
        )
        log(f"Effector: {len(e_valid)} frames with non-empty mask after propagation.")

        r_stack = np.stack([m.astype(bool) for m in r_masks], axis=0)
        e_stack = np.stack([m.astype(bool) for m in e_masks], axis=0)

        fd_mp4, seg_video_path = tempfile.mkstemp(suffix="_segmentation_overlay.mp4", prefix="pfr_demo_")
        os.close(fd_mp4)
        log(f"Rendering segmentation overlay video ({input_video_fps:g} fps, same as input)…")
        if frames_to_overlay_mp4(
            rgb_np,
            r_masks,
            e_masks,
            seg_video_path,
            fps=input_video_fps,
        ):
            log(f"Segmentation overlay video: {seg_video_path}")
        else:
            log("Warning: ffmpeg failed or missing; segmentation video not created. Install ffmpeg and retry.")
            try:
                os.unlink(seg_video_path)
            except OSError:
                pass
            seg_video_path = None

        # Stream segmentation output immediately; later stages continue and will send a final update.
        yield emit(recon_3d=gr.update(), articulation_val="", function_val="", summary_val="")

        log("Unloading SegZero / SAM3 from GPU cache before DA3 (frees VRAM for depth + RoMa)…")
        refseg = None
        sam3_tracker = None
        _release_seg_sam3_from_cache(
            device,
            reasoning_model_path,
            segmentation_model_path,
            sam_backend,
            sam3_checkpoint_path or "",
            sam3_bpe_path or "",
        )

        log("Depth Anything 3 reconstruction (HxWx3 uint8 numpy; required by fusion/reconstruction.py DA3)…")
        recon = get_recon_model(device, da3_model_path)
        recon_results = recon.reconstruct(rgb_np, init_extrinsic, None, None, None)
        if recon_results is None:
            log("Error: reconstruction failed.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return

        log("Unloading DA3 from GPU cache before RoMa fusion…")
        recon = None
        _release_da3_from_cache(device, da3_model_path)

        log("Feature-matching fusion (per part)…")
        fusion = get_fusion_model(device)
        fused_by_role: dict[str, np.ndarray | None] = {"receptor": None, "effector": None}

        # RoMa (romatch) fails with BFloat16 vs Float in cholesky_solve if CUDA autocast is on; fusion.py must stay untouched.
        _fusion_no_amp = (
            torch.autocast(device_type="cuda", enabled=False)
            if device == "cuda" and torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with _fusion_no_amp:
            for role, masks, label in (
                ("receptor", r_masks, receptor_label),
                ("effector", e_masks, effector_label),
            ):
                valid_ids = [i for i, m in enumerate(masks) if np.asarray(m).sum() > 0]
                if not valid_ids:
                    log(f"[{role}] no non-empty masks; skip fusion.")
                    continue
                pm = recon_results["points_mask"]
                vm = [np.logical_and(pm[i], np.asarray(masks[i], dtype=bool)) for i in valid_ids]
                pts = [recon_results["points"][i] for i in valid_ids]
                frames_hwc = [rgb_np[i] for i in valid_ids]
                kpts_a: dict[str, np.ndarray] = {}
                kpts_b: dict[str, np.ndarray] = {}
                fused_pcd, _trans, kpts_a, kpts_b = fusion.fuse_part_pcds(
                    frames_hwc, vm, pts, kpts_a, kpts_b
                )
                fused_by_role[role] = np.asarray(fused_pcd, dtype=np.float64).reshape(-1, 3)
                log(f"[{role}] fused PCD points: {int(fused_pcd.shape[0])} ({label})")

        log("Unloading RoMa fusion from GPU cache before iTACO…")
        fusion = None
        _release_fusion_from_cache(device)

        log("iTACO articulation (receptor, then effector)…")
        itaco = get_itaco(device)
        itaco_samp = _itaco_yaml_cfg()
        recon_for_itaco = {
            "depth": recon_results["depth"],
            "extrinsics": recon_results["extrinsics"],
            "points": recon_results["points"],
        }
        art_out: dict[str, Any] = {}
        # PyTorch3D chamfer/kNN requires matching dtypes; CUDA autocast can mix BF16 (xyz) with FP32 (surface_xyz).
        _itaco_no_amp = (
            torch.autocast(device_type="cuda", enabled=False)
            if device == "cuda" and torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with _itaco_no_amp:
            for role, stack in (("receptor", r_stack), ("effector", e_stack)):
                rgb_s, recon_s, masks_s = _subsample_for_itaco(rgb_np, recon_for_itaco, stack, itaco_samp)
                est = itaco.articulation_estimation(rgb_s, recon_s, masks_s)
                art_out[role] = est
        log(
            "3D scene: RoMa fused part clouds when available, else DA3 points under each mask; iTACO joint arrows (shared world frame)…"
        )
        recon_scene_glb_path, _scene_center = _export_recon_articulation_glb(
            recon_results,
            art_out,
            fused_by_role,
            {"receptor": r_masks, "effector": e_masks},
        )
        if recon_scene_glb_path:
            log(f"3D viewer GLB: {recon_scene_glb_path}")
        recon_3d_update = gr.update(value=recon_scene_glb_path)
        articulation_text = json.dumps(art_out, indent=2, default=_numpy_json_default)

        log(
            "Releasing SegZero / SAM3 / DA3 / fusion / iTACO from memory before loading Qwen (Transformers)…"
        )
        refseg = None
        sam3_tracker = None
        recon = None
        fusion = None
        itaco = None
        _clear_model_cache_and_empty_cuda()
        log("Qwen function VLM (Hugging Face Transformers, local weights)…")
        vlm_cfg = omegaconf.OmegaConf.load(
            _config_path("vlm_function", "qwen_function_transformers.yaml")
        )
        vlm_cfg.vlm_model = _QWEN_VLM_MODEL
        vlm_cfg.device = device
        fn_vlm = build_vlm_prompter(vlm_cfg)
        fn_res = fn_vlm.prompt_function(rgb_np, r_stack, e_stack)
        function_text = _function_vlm_display_json(fn_res)
        function_summary = _function_vlm_summary_html(function_text)

        log("Done.")
        yield emit(
            recon_3d=recon_3d_update,
            articulation_val=articulation_text,
            function_val=function_text,
            summary_val=function_summary,
        )

    except Exception:
        log(traceback.format_exc())
        err_3d = (
            gr.update(value=recon_scene_glb_path)
            if recon_scene_glb_path
            else gr.update(value=None)
        )
        yield emit(
            recon_3d=err_3d,
            articulation_val=articulation_text,
            function_val=function_text,
            summary_val=function_summary,
        )


def build_ui():
    with gr.Blocks(
        title="EgoFun3D",
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange"),
        css="""
            .egofun-header { margin-bottom: 0 !important; }
            .egofun-header h1 { margin: 0 !important; }
            .egofun-subtitle { margin-top: 6px !important; padding-top: 0 !important; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown("# EgoFun3D", elem_classes=["egofun-header"])
        gr.Markdown(
            "Upload a short video and label the **receptor** and **effector** parts in natural language. "
            "The pipeline segments both parts, reconstructs the 3-D scene (DA3 + RoMa fusion), "
            "estimates articulation axes (iTACO), and predicts part function with Qwen-3-VL 8B.",
            elem_classes=["egofun-subtitle"],
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                vid = gr.Video(label="Input video", format="mp4")
            with gr.Column(scale=1):
                seg_out = gr.Video(
                    label="Segmentation preview  ·  receptor: green-teal  ·  effector: orange",
                    format="mp4",
                    interactive=False,
                )

        with gr.Row():
            receptor = gr.Textbox(
                label="Receptor part",
                placeholder="e.g. faucet switch",
                scale=1,
            )
            effector = gr.Textbox(
                label="Effector part",
                placeholder="e.g. faucet spout",
                scale=1,
            )

        btn = gr.Button("Run pipeline", variant="primary")

        # Intermediate state for the segmentation video path.  seg_out is NOT in the main
        # generator's outputs, so Gradio never shows its "still running" loading overlay on
        # the video player between the first yield (seg ready) and the final yield (all done).
        # Instead, when _seg_path_state changes the tiny .change() handler immediately paints
        # the video with no further spinner.
        _seg_path_state = gr.State(value=None)
        _seg_path_state.change(
            fn=lambda p: p,
            inputs=[_seg_path_state],
            outputs=[seg_out],
            queue=False,
        )

        recon_3d = gr.Model3D(
            label="3D reconstruction  ·  receptor: green-teal  ·  effector: orange  ·  iTACO joint axes",
            height=540,
            clear_color=[0.06, 0.06, 0.08, 1.0],
        )

        function_summary = gr.HTML(label="Function prediction", value="")

        with gr.Row():
            with gr.Column(scale=1):
                log_box = gr.Textbox(label="Log", lines=12, max_lines=30)
            with gr.Column(scale=1):
                art_box = gr.Textbox(label="Articulation (JSON)", lines=12, max_lines=40)

        fn_box = gr.Textbox(
            label="Function VLM — full output (JSON)",
            lines=8,
            max_lines=30,
        )

        btn.click(
            run_pipeline,
            inputs=[vid, receptor, effector],
            outputs=[log_box, _seg_path_state, recon_3d, function_summary, art_box, fn_box],
            queue=True,
        )
    return demo


if __name__ == "__main__":
    if os.path.abspath(os.getcwd()) != os.path.abspath(_ROOT):
        print(
            f"Warning: cwd is {os.getcwd()!r}; recommended: cd {_ROOT} before running "
            "so third_party paths in fusion/reconstruction resolve correctly.",
            file=sys.stderr,
        )
    parser = argparse.ArgumentParser(description="Gradio demo for part-function reconstruction pipeline.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public gradio.live URL (Gradio share tunnel).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help=f"Listen port (default: GRADIO_SERVER_PORT env or {_DEFAULT_GRADIO_PORT}).",
    )
    args = parser.parse_args()
    server_port = args.port if args.port is not None else int(
        os.environ.get("GRADIO_SERVER_PORT", str(_DEFAULT_GRADIO_PORT))
    )
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=args.share,
    )
