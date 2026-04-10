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
import base64
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from io import BytesIO
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


def _resolve_uploaded_video_path(video_file: Any) -> str | None:
    if not video_file:
        return None
    if isinstance(video_file, str):
        path = video_file
    elif isinstance(video_file, dict):
        path = video_file.get("path") or video_file.get("name")
    else:
        path = getattr(video_file, "name", None)
    return path or None


INTERACTIVE_MIN_SEED_FRAMES = 1


def _is_interactive_mode(segmentation_mode: str | None) -> bool:
    return str(segmentation_mode or "").strip().lower().startswith("interactive")


def _empty_interactive_seg_state() -> dict[str, Any]:
    return {
        "video_path": None,
        "frames": [],
        "fps": float(SEGMENTATION_OVERLAY_FALLBACK_FPS),
        "active_frame": 0,
        "active_role": "receptor",
        "active_click_label": 1,
        "annotations": {"receptor": {}, "effector": {}},
    }


def _interactive_role_color(role: str) -> tuple[int, int, int]:
    palette = sns.color_palette("Set2")
    idx = 0 if role == "receptor" else 1
    return tuple(int(round(v * 255.0)) for v in palette[idx])


def _interactive_button_updates(active_role: str, active_click_label: int = 1):
    return (
        gr.update(value="Receptor", variant="primary" if active_role == "receptor" else "secondary"),
        gr.update(value="Effector", variant="primary" if active_role == "effector" else "secondary"),
        gr.update(value="Positive clicks (+)", variant="primary" if int(active_click_label) == 1 else "secondary"),
        gr.update(value="Negative clicks (-)", variant="primary" if int(active_click_label) == 0 else "secondary"),
    )


def _interactive_entry(
    state: dict[str, Any], role: str, frame_idx: int, create: bool = False
) -> dict[str, Any] | None:
    annotations = state.setdefault("annotations", {})
    role_map = annotations.setdefault(role, {})
    key = int(frame_idx)
    entry = role_map.get(key)
    if entry is None and create:
        entry = {"points": [], "mask": None}
        role_map[key] = entry
    return entry


def _interactive_points_for_frame(state: dict[str, Any], role: str, frame_idx: int) -> list[dict[str, int]]:
    entry = _interactive_entry(state, role, frame_idx, create=False)
    if not entry:
        return []
    out: list[dict[str, int]] = []
    for point in entry.get("points", []):
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
            label = point.get("label", 1)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
            label = 1
        else:
            continue
        if x is None or y is None:
            continue
        out.append({"x": int(x), "y": int(y), "label": 1 if int(label) != 0 else 0})
    return out


def _interactive_mask_for_frame(state: dict[str, Any], role: str, frame_idx: int) -> np.ndarray | None:
    entry = _interactive_entry(state, role, frame_idx, create=False)
    if not entry or entry.get("mask") is None:
        return None
    return np.asarray(entry["mask"]).astype(bool)


def _interactive_seed_frame_ids(state: dict[str, Any], role: str) -> list[int]:
    out: list[int] = []
    for frame_idx, entry in state.get("annotations", {}).get(role, {}).items():
        mask = entry.get("mask")
        if mask is not None and int(np.asarray(mask).astype(bool).sum()) > 0:
            out.append(int(frame_idx))
    return sorted(set(out))


def _interactive_union_seed_frame_ids(state: dict[str, Any]) -> list[int]:
    return sorted(
        set(_interactive_seed_frame_ids(state, "receptor"))
        | set(_interactive_seed_frame_ids(state, "effector"))
    )


def _interactive_status_html(state: dict[str, Any]) -> str:
    base_style = (
        "<style>"
        ".egofun-interactive-status, .egofun-interactive-status * { color: #243035 !important; }"
        ".egofun-interactive-status .muted { color: #5a6c72 !important; }"
        ".egofun-interactive-status .counts { color: #44535a !important; }"
        "</style>"
    )
    frames = state.get("frames", [])
    if not frames:
        return (
            base_style
            + '<div class="egofun-interactive-status" style="padding:12px 14px;border:1px solid #d9e3e3;border-radius:10px;background:#fbfdfd;font-size:0.95rem;font-family:system-ui,sans-serif;">'
            'Choose <b>Interactive SAM3</b> and upload a video to start annotating.</div>'
        )

    active_role = str(state.get("active_role", "receptor"))
    active_click_label = 1 if int(state.get("active_click_label", 1)) != 0 else 0
    frame_idx = int(state.get("active_frame", 0))
    receptor_ids = _interactive_seed_frame_ids(state, "receptor")
    effector_ids = _interactive_seed_frame_ids(state, "effector")
    union_ids = _interactive_union_seed_frame_ids(state)
    current_points = _interactive_points_for_frame(state, active_role, frame_idx)
    pos_count = sum(1 for point in current_points if int(point.get("label", 1)) == 1)
    neg_count = sum(1 for point in current_points if int(point.get("label", 1)) == 0)

    def _fmt(ids: list[int]) -> str:
        if not ids:
            return "none"
        shown = ", ".join(str(i) for i in ids[:8])
        if len(ids) > 8:
            shown += ", ..."
        return shown

    active_color = "rgb(102,194,165)" if active_role == "receptor" else "rgb(252,141,98)"
    click_mode = "positive (+)" if active_click_label == 1 else "negative (-)"
    return (
        base_style
        + '<div class="egofun-interactive-status" style="padding:12px 14px;border:1px solid #d9e3e3;border-radius:10px;background:#fbfdfd;font-family:system-ui,sans-serif;">'
        f'<div style="font-size:0.95rem;margin-bottom:8px;">Current frame: <b>{frame_idx}</b> / {max(len(frames) - 1, 0)} '
        f'&nbsp; <span style="color:{active_color} !important;font-weight:700;">Active part: {active_role}</span> '
        f'&nbsp; <span class="counts">Click mode: <b>{click_mode}</b></span> '
        f'&nbsp; <span class="counts">Frame clicks for active part: {len(current_points)} total, {pos_count} positive, {neg_count} negative</span></div>'
        '<div style="font-size:0.9rem;line-height:1.45;">'
        f'<b>Receptor seeds:</b> {len(receptor_ids)} frame(s) [{_fmt(receptor_ids)}]<br>'
        f'<b>Effector seeds:</b> {len(effector_ids)} frame(s) [{_fmt(effector_ids)}]<br>'
        f'<b>Frames annotated overall:</b> {len(union_ids)} frame(s) [{_fmt(union_ids)}]<br>'
        f'<span class="muted">Use positive clicks to grow the mask and negative clicks to suppress spill. You can use the same frame for both parts by switching the Receptor/Effector button before clicking.</span>'
        '</div></div>'
    )

def _render_interactive_frame(state: dict[str, Any]) -> np.ndarray | None:
    frames = state.get("frames", [])
    if not frames:
        return None
    frame_idx = int(np.clip(state.get("active_frame", 0), 0, len(frames) - 1))
    state["active_frame"] = frame_idx
    active_role = str(state.get("active_role", "receptor"))

    rgb = np.asarray(frames[frame_idx], dtype=np.uint8)
    recv_mask = _interactive_mask_for_frame(state, "receptor", frame_idx)
    eff_mask = _interactive_mask_for_frame(state, "effector", frame_idx)
    overlay = overlay_masks_pop(
        rgb,
        recv_mask,
        eff_mask,
        alpha=0.55,
        darken=0.82,
        outline_thickness=4,
    )

    for role in ("receptor", "effector"):
        color = _interactive_role_color(role)
        radius = 7 if role == active_role else 5
        for point_idx, point in enumerate(_interactive_points_for_frame(state, role, frame_idx), start=1):
            x = int(point["x"])
            y = int(point["y"])
            label = 1 if int(point.get("label", 1)) != 0 else 0
            marker_text = "+" if label == 1 else "-"
            cv2.circle(overlay, (x, y), radius + 3, (245, 245, 245), 2, lineType=cv2.LINE_AA)
            if label == 1:
                cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(overlay, (x, y), radius, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(overlay, (x, y), max(radius - 2, 1), (22, 26, 29), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                marker_text,
                (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                str(point_idx),
                (x + 10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        f"Frame {frame_idx + 1}/{len(frames)}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Active: {active_role}",
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        _interactive_role_color(active_role),
        2,
        cv2.LINE_AA,
    )
    return overlay

def _sam3_bpe_path() -> str:
    candidate = os.path.join(_ROOT, "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"SAM3 BPE vocab not found at {candidate}. Download it from "
            "https://github.com/facebookresearch/sam3/raw/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )
    return candidate


def get_sam3_image_processor(
    device: str,
    checkpoint_path: str | None = None,
    bpe_path: str | None = None,
    confidence_threshold: float = 0.5,
):
    ck = checkpoint_path or ""
    bp = bpe_path or ""
    key = f"sam3_img|{device}|{ck}|{bp}|{float(confidence_threshold):.4f}"
    if key not in _MODEL_CACHE:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        resolved_bpe_path = bpe_path or _sam3_bpe_path()
        model = build_sam3_image_model(
            bpe_path=resolved_bpe_path,
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            # Enable the SAM2-style interactive predictor so that click-based
            # segmentation in _sam3_mask_from_points works correctly.
            enable_inst_interactivity=True,
        )
        _MODEL_CACHE[key] = Sam3Processor(
            model,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )
    return _MODEL_CACHE[key]


def _sam3_mask_from_points(
    image: np.ndarray,
    points: list[dict[str, int]],
    device: str,
    checkpoint_path: str | None = None,
    bpe_path: str | None = None,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    height, width = image.shape[:2]
    if len(points) == 0:
        return np.zeros((height, width), dtype=bool)

    point_labels = [1 if int(p.get("label", 1)) != 0 else 0 for p in points]
    if not any(lbl == 1 for lbl in point_labels):
        return np.zeros((height, width), dtype=bool)

    processor = get_sam3_image_processor(
        device=device,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        confidence_threshold=confidence_threshold,
    )

    # Use SAM3's SAM2-style interactive predictor (inst_interactive_predictor).
    # The grounding API (geometric_prompt.append_points + _forward_grounding) is a
    # detector that ignores click positions; the interactive predictor is specifically
    # trained for click-based interactive segmentation.
    #
    # The tracker backbone is None — it shares image features with the main SAM3
    # backbone. processor.set_image() extracts "sam2_backbone_out" from the main
    # backbone and applies conv_s0/conv_s1 to adapt the FPN features for the
    # interactive predictor's SAM2 decoder. We then manually populate the predictor's
    # _features from those adapted features, bypassing its standalone set_image().
    interactive_pred = processor.model.inst_interactive_predictor
    if interactive_pred is None:
        return np.zeros((height, width), dtype=bool)

    image_pil = to_pil_rgb(image)
    state = processor.set_image(image_pil)

    sam2_out = state["backbone_out"].get("sam2_backbone_out")
    if sam2_out is None:
        return np.zeros((height, width), dtype=bool)

    # Mirror SAM3InteractiveImagePredictor.set_image() but using the pre-computed
    # (and already conv_s0/conv_s1-adapted) FPN features from processor.set_image().
    _, vision_feats, _, _ = interactive_pred.model._prepare_backbone_features(sam2_out)
    vision_feats[-1] = vision_feats[-1] + interactive_pred.model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], interactive_pred._bb_feat_sizes[::-1])
    ][::-1]
    interactive_pred._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    interactive_pred._orig_hw = [(height, width)]
    interactive_pred._is_image_set = True
    interactive_pred._is_batch = False

    point_coords = np.array(
        [[int(p["x"]), int(p["y"])] for p in points], dtype=np.float32
    )
    point_labels_arr = np.array(point_labels, dtype=np.int32)

    # multimask_output=True for a single ambiguous click (returns 3 candidates
    # and picks the best by IOU score). False for multiple clicks.
    multimask_output = len(points) == 1

    masks_np, iou_predictions_np, _ = interactive_pred.predict(
        point_coords=point_coords,
        point_labels=point_labels_arr,
        multimask_output=multimask_output,
        normalize_coords=True,  # point_coords are in pixel space
    )
    # masks_np: (C, H, W) float binary (thresholded at mask_threshold=0.0)
    best_idx = int(np.argmax(iou_predictions_np))
    return (masks_np[best_idx] > 0.5).astype(bool)

def _prepare_interactive_state(video_file: Any, state: dict[str, Any] | None) -> dict[str, Any]:
    path = _resolve_uploaded_video_path(video_file)
    if not path or not os.path.isfile(path):
        return _empty_interactive_seg_state()

    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    if prepared.get("video_path") != path or not prepared.get("frames"):
        _pil_frames_unused, rgb_np, fps = load_full_video(path)
        prepared = _empty_interactive_seg_state()
        prepared["video_path"] = path
        prepared["frames"] = rgb_np
        prepared["fps"] = float(fps)

    if prepared.get("active_role") not in {"receptor", "effector"}:
        prepared["active_role"] = "receptor"
    prepared["active_click_label"] = 1 if int(prepared.get("active_click_label", 1)) != 0 else 0
    num_frames = len(prepared.get("frames", []))
    prepared["active_frame"] = int(np.clip(prepared.get("active_frame", 0), 0, max(num_frames - 1, 0)))
    return prepared


def _interactive_ui_updates(segmentation_mode: str, video_file: Any, state: dict[str, Any] | None):
    if not _is_interactive_mode(segmentation_mode):
        receptor_btn, effector_btn, positive_btn, negative_btn = _interactive_button_updates("receptor", 1)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            state if isinstance(state, dict) else _empty_interactive_seg_state(),
            gr.update(value=None),
            gr.update(minimum=0, maximum=0, value=0),
            _interactive_status_html(_empty_interactive_seg_state()),
            receptor_btn,
            effector_btn,
            positive_btn,
            negative_btn,
        )

    prepared = _prepare_interactive_state(video_file, state)
    receptor_btn, effector_btn, positive_btn, negative_btn = _interactive_button_updates(
        prepared.get("active_role", "receptor"), prepared.get("active_click_label", 1)
    )
    max_frame = max(len(prepared.get("frames", [])) - 1, 0)
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        prepared,
        gr.update(value=_render_interactive_frame(prepared)),
        gr.update(minimum=0, maximum=max_frame, value=int(prepared.get("active_frame", 0))),
        _interactive_status_html(prepared),
        receptor_btn,
        effector_btn,
        positive_btn,
        negative_btn,
    )

def _interactive_refresh_outputs(state: dict[str, Any]):
    receptor_btn, effector_btn, positive_btn, negative_btn = _interactive_button_updates(
        state.get("active_role", "receptor"), state.get("active_click_label", 1)
    )
    return (
        state,
        gr.update(value=_render_interactive_frame(state)),
        _interactive_status_html(state),
        receptor_btn,
        effector_btn,
        positive_btn,
        negative_btn,
    )

def _interactive_set_active_role(state: dict[str, Any] | None, role: str):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    prepared["active_role"] = role if role in {"receptor", "effector"} else "receptor"
    return _interactive_refresh_outputs(prepared)


def _interactive_set_frame(state: dict[str, Any] | None, frame_idx: float | int):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    num_frames = len(prepared.get("frames", []))
    prepared["active_frame"] = int(np.clip(int(frame_idx), 0, max(num_frames - 1, 0))) if num_frames > 0 else 0
    return _interactive_refresh_outputs(prepared)


def _interactive_set_click_label(state: dict[str, Any] | None, label: int):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    prepared["active_click_label"] = 1 if int(label) != 0 else 0
    return _interactive_refresh_outputs(prepared)


def _interactive_recompute_mask(state: dict[str, Any], role: str, frame_idx: int) -> None:
    entry = _interactive_entry(state, role, frame_idx, create=False)
    if entry is None:
        return
    points = _interactive_points_for_frame(state, role, frame_idx)
    role_map = state.setdefault("annotations", {}).setdefault(role, {})
    if len(points) == 0:
        role_map.pop(int(frame_idx), None)
        return
    frame = np.asarray(state["frames"][int(frame_idx)], dtype=np.uint8)
    entry["points"] = points
    entry["mask"] = _sam3_mask_from_points(frame, points, _DEVICE)


def _interactive_add_click(state: dict[str, Any] | None, evt: gr.SelectData):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    frames = prepared.get("frames", [])
    if not frames:
        return _interactive_refresh_outputs(prepared)

    idx = getattr(evt, "index", None)
    if isinstance(idx, (list, tuple)) and len(idx) >= 2:
        x, y = idx[0], idx[1]
    elif isinstance(idx, dict):
        x, y = idx.get("x"), idx.get("y")
    else:
        return _interactive_refresh_outputs(prepared)
    if x is None or y is None:
        return _interactive_refresh_outputs(prepared)

    frame_idx = int(prepared.get("active_frame", 0))
    h, w = np.asarray(frames[frame_idx]).shape[:2]
    x_i = int(np.clip(int(round(float(x))), 0, w - 1))
    y_i = int(np.clip(int(round(float(y))), 0, h - 1))
    role = str(prepared.get("active_role", "receptor"))
    label = 1 if int(prepared.get("active_click_label", 1)) != 0 else 0
    entry = _interactive_entry(prepared, role, frame_idx, create=True)
    entry.setdefault("points", []).append({"x": x_i, "y": y_i, "label": label})
    _interactive_recompute_mask(prepared, role, frame_idx)
    return _interactive_refresh_outputs(prepared)


def _interactive_undo_click(state: dict[str, Any] | None):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    frame_idx = int(prepared.get("active_frame", 0))
    role = str(prepared.get("active_role", "receptor"))
    entry = _interactive_entry(prepared, role, frame_idx, create=False)
    if entry and entry.get("points"):
        entry["points"].pop()
        _interactive_recompute_mask(prepared, role, frame_idx)
    return _interactive_refresh_outputs(prepared)


def _interactive_clear_current_role(state: dict[str, Any] | None):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    frame_idx = int(prepared.get("active_frame", 0))
    role = str(prepared.get("active_role", "receptor"))
    prepared.setdefault("annotations", {}).setdefault(role, {}).pop(frame_idx, None)
    return _interactive_refresh_outputs(prepared)


def _interactive_clear_all(state: dict[str, Any] | None):
    prepared = state if isinstance(state, dict) else _empty_interactive_seg_state()
    prepared["annotations"] = {"receptor": {}, "effector": {}}
    return _interactive_refresh_outputs(prepared)


def _segment_one_role_interactive(
    interactive_state: dict[str, Any],
    full_rgb_np: list[np.ndarray],
    role: str,
    sam3_tracker,
    log,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    seed_frame_ids = _interactive_seed_frame_ids(interactive_state, role)
    seed_pred_masks = [
        np.asarray(_interactive_mask_for_frame(interactive_state, role, frame_id)).astype(bool)
        for frame_id in seed_frame_ids
    ]
    seed_answer_dicts = []
    seed_valid_frame_ids = []
    for local_idx, frame_id in enumerate(seed_frame_ids):
        points = _interactive_points_for_frame(interactive_state, role, frame_id)
        mask = seed_pred_masks[local_idx]
        seed_answer_dicts.append(
            {
                "frame_id": int(frame_id),
                "seed_frame": True,
                "propagated": False,
                "annotation_source": "interactive_sam3_clicks",
                "num_clicks": len(points),
            }
        )
        if int(mask.sum()) > 0:
            seed_valid_frame_ids.append(local_idx)

    total = len(full_rgb_np)
    log(
        f"  User-provided {role} seed frames ({len(seed_frame_ids)}): "
        f"{seed_frame_ids[:8]}{'...' if len(seed_frame_ids) > 8 else ''}"
    )
    if len(seed_frame_ids) < total:
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
        log(f"  No SAM3 propagation needed; non-empty mask frames: {len(valid_ids)}.")
    return masks, valid_ids, seed_frame_ids


def run_pipeline(
    video_file: str | None,
    receptor_label: str,
    effector_label: str,
    segmentation_mode: str,
    interactive_state: dict[str, Any] | None,
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
        path = _resolve_uploaded_video_path(video_file)
        if not path or not os.path.isfile(path):
            log("Error: invalid upload.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return

        receptor_label = (receptor_label or "").strip()
        effector_label = (effector_label or "").strip()

        interactive_mode = _is_interactive_mode(segmentation_mode)
        if not interactive_mode and (not receptor_label or not effector_label):
            log("Error: provide receptor and effector part descriptions.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return
        if interactive_mode:
            receptor_label = receptor_label or "receptor"
            effector_label = effector_label or "effector"
        if interactive_mode:
            log("Interactive SAM3 segmentation selected.")
            interactive_state = _prepare_interactive_state(video_file, interactive_state)
            rgb_np = [np.asarray(frame, dtype=np.uint8) for frame in interactive_state.get("frames", [])]
            if not rgb_np:
                log("Error: upload a video before annotating interactive masks.")
                yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
                return
            input_video_fps = float(interactive_state.get("fps", SEGMENTATION_OVERLAY_FALLBACK_FPS))
            total_f = len(rgb_np)
            r_seed_ids = _interactive_seed_frame_ids(interactive_state, "receptor")
            e_seed_ids = _interactive_seed_frame_ids(interactive_state, "effector")
            if len(r_seed_ids) < INTERACTIVE_MIN_SEED_FRAMES or len(e_seed_ids) < INTERACTIVE_MIN_SEED_FRAMES:
                log(
                    f"Error: interactive mode needs at least {INTERACTIVE_MIN_SEED_FRAMES} annotated frames per part. "
                    f"Current counts - receptor: {len(r_seed_ids)}, effector: {len(e_seed_ids)}."
                )
                yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
                return
            log(
                f"Loaded {total_f} frames from interactive cache. "
                f"Receptor seeds: {len(r_seed_ids)}. Effector seeds: {len(e_seed_ids)}."
            )
        else:
            log("Loading full video (every frame, in order)...")
            _pil_frames, _rgb_unused, input_video_fps = load_full_video(path)
            rgb_np = video_frames_as_numpy_hwc_uint8(_pil_frames)
            total_f = len(rgb_np)
            log(
                f"Loaded {total_f} frames. Segmentation: {SEGMENTATION_SEED_FRAMES} evenly spaced seeds, "
                "SAM3 propagates to all frames when N > seeds."
            )

        init_extrinsic = np.eye(4, dtype=np.float64)

        if interactive_mode:
            should_propagate = (
                len(_interactive_seed_frame_ids(interactive_state, "receptor")) < total_f
                or len(_interactive_seed_frame_ids(interactive_state, "effector")) < total_f
            )
            sam3_tracker = None
            if should_propagate:
                log("Loading / caching SAM3 video tracker (propagation from user-annotated frames)...")
                sam3_tracker = get_sam3_tracker(device, sam3_checkpoint_path, sam3_bpe_path)

            log("Propagating receptor masks from interactive SAM3 clicks...")
            r_masks, r_valid, _r_seeds = _segment_one_role_interactive(
                interactive_state,
                rgb_np,
                "receptor",
                sam3_tracker,
                log,
            )
            log(f"Receptor: {len(r_valid)} frames with non-empty mask after propagation.")

            log("Propagating effector masks from interactive SAM3 clicks...")
            e_masks, e_valid, _e_seeds = _segment_one_role_interactive(
                interactive_state,
                rgb_np,
                "effector",
                sam3_tracker,
                log,
            )
            log(f"Effector: {len(e_valid)} frames with non-empty mask after propagation.")
        else:
            log("Loading / caching SegZero (VisionReasoner)...")
            refseg = get_seg_model(device, reasoning_model_path, segmentation_model_path, sam_backend)

            seed_ids_preview = evenly_spaced_indices(total_f, SEGMENTATION_SEED_FRAMES)
            should_propagate = len(seed_ids_preview) < total_f
            sam3_tracker = None
            if should_propagate:
                log("Loading / caching SAM3 video tracker (mask propagation)...")
                sam3_tracker = get_sam3_tracker(device, sam3_checkpoint_path, sam3_bpe_path)

            receptor_desc = f"{receptor_label}. {receptor_label}"
            effector_desc = f"{effector_label}. {effector_label}"

            log("Segmenting receptor (SegZero on seed frames; SAM3 to full video when N > seeds)...")
            r_masks, r_valid, _r_seeds = _segment_one_role(
                refseg,
                rgb_np,
                receptor_desc,
                sam3_tracker,
                log,
            )
            log(f"Receptor: {len(r_valid)} frames with non-empty mask after propagation.")

            log("Segmenting effector...")
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
        log(f"Rendering segmentation overlay video ({input_video_fps:g} fps, same as input)...")
        if frames_to_overlay_mp4(rgb_np, r_masks, e_masks, seg_video_path, fps=input_video_fps):
            log(f"Segmentation overlay video: {seg_video_path}")
        else:
            log("Warning: ffmpeg failed or missing; segmentation video not created. Install ffmpeg and retry.")
            try:
                os.unlink(seg_video_path)
            except OSError:
                pass
            seg_video_path = None

        yield emit(recon_3d=gr.update(), articulation_val="", function_val="", summary_val="")

        log("Unloading segmentation models from GPU cache before DA3 (frees VRAM for depth + RoMa)...")
        if not interactive_mode:
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
        _MODEL_CACHE.pop(f"sam3_img|{device}|||0.5000", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log("Depth Anything 3 reconstruction (HxWx3 uint8 numpy; required by fusion/reconstruction.py DA3)...")
        recon = get_recon_model(device, da3_model_path)
        recon_results = recon.reconstruct(rgb_np, init_extrinsic, None, None, None)
        if recon_results is None:
            log("Error: reconstruction failed.")
            yield emit(recon_3d=gr.update(value=None), articulation_val="", function_val="", summary_val="")
            return

        log("Unloading DA3 from GPU cache before RoMa fusion...")
        recon = None
        _release_da3_from_cache(device, da3_model_path)

        log("Feature-matching fusion (per part)...")
        fusion = get_fusion_model(device)
        fused_by_role: dict[str, np.ndarray | None] = {"receptor": None, "effector": None}

        _fusion_no_amp = (
            torch.autocast(device_type="cuda", enabled=False)
            if device == "cuda" and torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with _fusion_no_amp:
            for role, masks, label in (("receptor", r_masks, receptor_label), ("effector", e_masks, effector_label)):
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
                fused_pcd, _trans, kpts_a, kpts_b = fusion.fuse_part_pcds(frames_hwc, vm, pts, kpts_a, kpts_b)
                fused_by_role[role] = np.asarray(fused_pcd, dtype=np.float64).reshape(-1, 3)
                log(f"[{role}] fused PCD points: {int(fused_pcd.shape[0])} ({label})")

        log("Unloading RoMa fusion from GPU cache before iTACO...")
        fusion = None
        _release_fusion_from_cache(device)

        log("iTACO articulation (receptor, then effector)...")
        itaco = get_itaco(device)
        itaco_samp = _itaco_yaml_cfg()
        recon_for_itaco = {
            "depth": recon_results["depth"],
            "extrinsics": recon_results["extrinsics"],
            "points": recon_results["points"],
        }
        art_out: dict[str, Any] = {}
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
            "3D scene: RoMa fused part clouds when available, else DA3 points under each mask; iTACO joint arrows (shared world frame)..."
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

        log("Releasing segmentation / DA3 / fusion / iTACO from memory before loading Qwen (Transformers)...")
        recon = None
        fusion = None
        itaco = None
        _clear_model_cache_and_empty_cuda()
        log("Qwen function VLM (Hugging Face Transformers, local weights)...")
        vlm_cfg = omegaconf.OmegaConf.load(_config_path("vlm_function", "qwen_function_transformers.yaml"))
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
        err_3d = gr.update(value=recon_scene_glb_path) if recon_scene_glb_path else gr.update(value=None)
        yield emit(
            recon_3d=err_3d,
            articulation_val=articulation_text,
            function_val=function_text,
            summary_val=function_summary,
        )


def build_ui():
    with gr.Blocks(
        title="EgoFun3D",
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="slate"),
        css="""
            .egofun-header { margin-bottom: 0 !important; }
            .egofun-header h1 { margin: 0 !important; }
            .egofun-subtitle { margin-top: 6px !important; padding-top: 0 !important; }
            /* Receptor button: teal — CSS vars control fill/text per variant;
               direct border override ensures the outline is always visible. */
            #sam3-receptor-btn {
                --button-primary-background-fill: #3d9e7d;
                --button-primary-background-fill-hover: #2d8e6d;
                --button-primary-text-color: white;
                --button-primary-border-color: #3d9e7d;
                --button-secondary-background-fill: rgba(61,158,125,0.07);
                --button-secondary-background-fill-hover: rgba(61,158,125,0.15);
                --button-secondary-text-color: #3d9e7d;
                --button-secondary-border-color: #3d9e7d;
            }
            #sam3-receptor-btn button { border: 2px solid #3d9e7d !important; color: #3d9e7d !important; }
            #sam3-receptor-btn button.primary { background: #3d9e7d !important; color: white !important; }
            /* Effector button: always orange. */
            #sam3-effector-btn {
                --button-primary-background-fill: #d46a2a;
                --button-primary-background-fill-hover: #c45a1a;
                --button-primary-text-color: white;
                --button-primary-border-color: #d46a2a;
                --button-secondary-background-fill: rgba(212,106,42,0.07);
                --button-secondary-background-fill-hover: rgba(212,106,42,0.15);
                --button-secondary-text-color: #d46a2a;
                --button-secondary-border-color: #d46a2a;
            }
            #sam3-effector-btn button { border: 2px solid #d46a2a !important; color: #d46a2a !important; }
            #sam3-effector-btn button.primary { background: #d46a2a !important; color: white !important; }
            /* Positive clicks button: green. */
            #sam3-positive-btn {
                --button-primary-background-fill: #2e7d32;
                --button-primary-background-fill-hover: #1b5e20;
                --button-primary-text-color: white;
                --button-secondary-background-fill: rgba(46,125,50,0.07);
                --button-secondary-background-fill-hover: rgba(46,125,50,0.15);
                --button-secondary-text-color: #2e7d32;
                --button-secondary-border-color: #2e7d32;
            }
            #sam3-positive-btn button { border: 2px solid #2e7d32 !important; color: #2e7d32 !important; }
            #sam3-positive-btn button.primary { background: #2e7d32 !important; color: white !important; }
            /* Negative clicks button: red. */
            #sam3-negative-btn {
                --button-primary-background-fill: #c62828;
                --button-primary-background-fill-hover: #b71c1c;
                --button-primary-text-color: white;
                --button-secondary-background-fill: rgba(198,40,40,0.07);
                --button-secondary-background-fill-hover: rgba(198,40,40,0.15);
                --button-secondary-text-color: #c62828;
                --button-secondary-border-color: #c62828;
            }
            #sam3-negative-btn button { border: 2px solid #c62828 !important; color: #c62828 !important; }
            #sam3-negative-btn button.primary { background: #c62828 !important; color: white !important; }
            .egofun-interactive-note {
                border: 1px solid #d9e3e3;
                border-radius: 12px;
                padding: 14px 16px;
                background: #f7f9fb;
                color: #2d3748 !important;
            }
            .egofun-interactive-note p,
            .egofun-interactive-note li,
            .egofun-interactive-note strong {
                color: #2d3748 !important;
            }
            #interactive-sam3-image .image-frame {
                height: 72vh !important;
                max-height: 720px !important;
                min-height: 320px !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                overflow: hidden !important;
            }
            #interactive-sam3-image .image-frame img {
                width: auto !important;
                height: auto !important;
                max-width: 100% !important;
                max-height: 100% !important;
                object-fit: contain !important;
                display: block !important;
            }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown("# EgoFun3D", elem_classes=["egofun-header"])
        gr.Markdown(
            "Upload a short video and label the **receptor** and **effector** parts in natural language. "
            "The pipeline can segment automatically with Vision Reasoner + SAM3, or let you annotate SAM3 seed masks interactively; "
            "then it reconstructs the 3-D scene (DA3 + RoMa fusion), estimates articulation axes (iTACO), and predicts part function with Qwen-3-VL 8B.",
            elem_classes=["egofun-subtitle"],
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                vid = gr.Video(label="Input video", format="mp4")
            with gr.Column(scale=1):
                seg_out = gr.Video(
                    label="Segmentation preview",
                    format="mp4",
                    interactive=False,
                )

        with gr.Row() as part_label_row:
            receptor = gr.Textbox(label="Receptor part", placeholder="e.g. faucet switch", scale=1)
            effector = gr.Textbox(label="Effector part", placeholder="e.g. faucet spout", scale=1)

        segmentation_mode = gr.Radio(
            choices=["Vision Reasoner", "Interactive SAM3"],
            value="Vision Reasoner",
            label="Segmentation mode",
            info=None,
        )

        interactive_state = gr.State(value=_empty_interactive_seg_state())

        with gr.Column(visible=False) as interactive_panel:
            gr.Markdown(
                "**Interactive SAM3 annotation**  \n"
                "1. Pick the **Receptor** or **Effector** button to set which part you are annotating.  \n"
                "2. Move to a frame with the slider and click on the preview image. Use positive clicks to include the part and negative clicks to suppress spill.  \n"
                "3. The SAM3 mask updates immediately on that frame. You can annotate both parts on the same frame by switching between the Receptor/Effector buttons before clicking.  \n"
                "4. When you run the pipeline, the seed masks are propagated to the full clip with SAM3.",
                elem_classes=["egofun-interactive-note"],
            )
            with gr.Row():
                receptor_btn = gr.Button("Receptor", variant="primary", elem_id="sam3-receptor-btn")
                effector_btn = gr.Button("Effector", variant="secondary", elem_id="sam3-effector-btn")
                positive_btn = gr.Button("Positive clicks (+)", variant="primary", elem_id="sam3-positive-btn")
                negative_btn = gr.Button("Negative clicks (-)", variant="secondary", elem_id="sam3-negative-btn")
                undo_btn = gr.Button("Undo last click")
                clear_role_btn = gr.Button("Clear current part on this frame")
                clear_all_btn = gr.Button("Clear all interactive masks")
            frame_slider = gr.Slider(minimum=0, maximum=0, value=0, step=1, label="Annotation frame")
            interactive_img = gr.Image(
                label="Click to add annotation points",
                format="png",
                type="numpy",
                interactive=False,
                show_download_button=False,
                elem_id="interactive-sam3-image",
            )
            interactive_status = gr.HTML(value=_interactive_status_html(_empty_interactive_seg_state()))

        btn = gr.Button("Run pipeline", variant="primary")

        _seg_path_state = gr.State(value=None)
        _seg_path_state.change(fn=lambda p: p, inputs=[_seg_path_state], outputs=[seg_out], queue=False)

        recon_3d = gr.Model3D(
            label="3D reconstruction + articulation axes",
            height=540,
            clear_color=[0.06, 0.06, 0.08, 1.0],
        )

        function_summary = gr.HTML(label="Function prediction", value="")
        _fn_summary_state = gr.State(value="")
        _fn_summary_state.change(fn=lambda h: h, inputs=[_fn_summary_state], outputs=[function_summary], queue=False)

        with gr.Row():
            with gr.Column(scale=1):
                log_box = gr.Textbox(label="Log", lines=12, max_lines=30)
            with gr.Column(scale=1):
                art_box = gr.Textbox(label="Articulation (JSON)", lines=12, max_lines=40)

        fn_box = gr.Textbox(label="Function VLM — full output (JSON)", lines=8, max_lines=30)

        segmentation_mode.change(
            _interactive_ui_updates,
            inputs=[segmentation_mode, vid, interactive_state],
            outputs=[interactive_panel, part_label_row, interactive_state, interactive_img, frame_slider, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        vid.change(
            _interactive_ui_updates,
            inputs=[segmentation_mode, vid, interactive_state],
            outputs=[interactive_panel, part_label_row, interactive_state, interactive_img, frame_slider, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        receptor_btn.click(
            lambda state: _interactive_set_active_role(state, "receptor"),
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        effector_btn.click(
            lambda state: _interactive_set_active_role(state, "effector"),
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        positive_btn.click(
            lambda state: _interactive_set_click_label(state, 1),
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        negative_btn.click(
            lambda state: _interactive_set_click_label(state, 0),
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        frame_slider.change(
            _interactive_set_frame,
            inputs=[interactive_state, frame_slider],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        interactive_img.select(
            _interactive_add_click,
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        undo_btn.click(
            _interactive_undo_click,
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        clear_role_btn.click(
            _interactive_clear_current_role,
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )
        clear_all_btn.click(
            _interactive_clear_all,
            inputs=[interactive_state],
            outputs=[interactive_state, interactive_img, interactive_status, receptor_btn, effector_btn, positive_btn, negative_btn],
            queue=False,
        )

        btn.click(
            run_pipeline,
            inputs=[vid, receptor, effector, segmentation_mode, interactive_state],
            outputs=[log_box, _seg_path_state, recon_3d, _fn_summary_state, art_box, fn_box],
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
