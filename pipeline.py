from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import cv2
import numpy as np
import omegaconf
import torch
import seaborn as sns
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation

from articulation.base import build_articulation_estimation_model
from articulation.evaluate_articulation import save_articulation_results
from fusion.fusion import build_fusion_model, FeatureMatchingFusion
from fusion.reconstruction import build_reconstruction_model
from fusion.evaluate_reconstruction import save_mesh, save_pcd, save_reconstruction_results
from function.evaluate_function import save_function_results
from segmentation.ref_seg import build_refseg_model
from segmentation.workflow import (
    build_sam3_tracker,
    evenly_spaced_indices,
    propagate_full_video_from_masks,
    to_pil_rgb,
)
from utils.reconstruction_utils import refine_point_mask
from VLM.prompt_vlm import build_vlm_prompter

try:
    import vllm as _vllm

    _orig_llm_init = _vllm.LLM.__init__

    def _patched_llm_init(self, *args, **kwargs):
        if "tensor_parallel_size" in kwargs:
            n_gpus = max(1, torch.cuda.device_count())
            kwargs["tensor_parallel_size"] = min(kwargs["tensor_parallel_size"], n_gpus)
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            safe_util = (free_bytes / total_bytes) * 0.90
            requested = kwargs.get("gpu_memory_utilization", 0.8)
            if requested > safe_util:
                kwargs["gpu_memory_utilization"] = round(safe_util, 3)
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 32_768
        _orig_llm_init(self, *args, **kwargs)

    _vllm.LLM.__init__ = _patched_llm_init
except ImportError:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))

SEGMENTATION_SEED_FRAMES = 20
SEGMENTATION_OVERLAY_FALLBACK_FPS = 10

GLB_POINT_PROXY_RADIUS_FRAC = 0.0065
GLB_POINT_PROXY_MAX_SPHERES = 16_000
GLB_POINT_ICOSPHERE_SUBDIV = 1
ARROW_MESH_SECTIONS = 40

_DEVICE = "cuda"
_REASONING_MODEL_PATH = "Ricky06662/VisionReasoner-7B"
_SEGMENTATION_MODEL_PATH = "facebook/sam3"
_SAM_BACKEND = "sam3"
_DA3_MODEL_PATH = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

_MODEL_CACHE: dict[str, Any] = {}
_da3_chunk_cfg: Any = None
_itaco_yaml_cfg_cache: Any = None

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

def _numpy_json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))

def _clear_model_cache_and_empty_cuda() -> None:
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _drop_cache_keys_starting_with(*prefixes: str) -> None:
    for k in list(_MODEL_CACHE.keys()):
        if any(k.startswith(p) for p in prefixes):
            _MODEL_CACHE.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_full_video(video_path: str) -> tuple[list[PILImage.Image], list[np.ndarray], float]:
    """Decode every frame in order.  Returns (PIL list, RGB-uint8 list, fps)."""
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
    """Ensure every frame is an HxWx3 uint8 ndarray (required by DA3 / SAM3)."""
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
    local_to_full = {li: fid for li, fid in enumerate(seed_frame_ids)}

    for local_idx, frame_id in local_to_full.items():
        full_pred_mask_list[frame_id] = np.asarray(seed_pred_masks[local_idx]).astype(bool)
        ad = dict(seed_answer_dicts[local_idx] if isinstance(seed_answer_dicts[local_idx], dict) else {})
        ad.setdefault("frame_id", frame_id)
        ad["seed_frame"] = True
        ad["propagated"] = False
        ad["subsampled"] = len(seed_frame_ids) != total_frames
        full_answer_dict_list[frame_id] = ad

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

    valid_frame_ids = [local_to_full[li] for li in seed_valid_frame_ids if li in local_to_full]
    return (
        [np.asarray(m).astype(bool) for m in full_pred_mask_list],
        [dict(a) for a in full_answer_dict_list],
        valid_frame_ids,
    )

def _propagate_seed_masks_to_full_video(
    full_frame_list: list[np.ndarray],
    seed_frame_ids: list[int],
    seed_pred_masks: list[np.ndarray],
    seed_answer_dicts: list[dict],
    seed_valid_frame_ids: list[int],
    sam3_tracker: Any,
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    total_frames = len(full_frame_list)
    valid_seed_mask_entries: list[tuple[int, np.ndarray]] = []
    valid_seed_frame_ids_full: list[int] = []
    for local_idx in seed_valid_frame_ids:
        if local_idx < 0 or local_idx >= len(seed_frame_ids):
            continue
        full_frame_id = int(seed_frame_ids[local_idx])
        mask = np.asarray(seed_pred_masks[local_idx]).astype(bool)
        if int(mask.sum()) <= 0:
            continue
        valid_seed_mask_entries.append((full_frame_id, mask))
        valid_seed_frame_ids_full.append(full_frame_id)

    if valid_seed_mask_entries:
        propagated_masks = propagate_full_video_from_masks(
            sam3_tracker, full_frame_list, valid_seed_mask_entries
        )
    else:
        propagated_masks = [np.zeros(frame.shape[:2], dtype=bool) for frame in full_frame_list]

    seed_answer_map: dict[int, dict] = {}
    for local_idx, frame_id in enumerate(seed_frame_ids):
        ad = dict(seed_answer_dicts[local_idx] if isinstance(seed_answer_dicts[local_idx], dict) else {})
        ad.setdefault("frame_id", int(frame_id))
        ad["seed_frame"] = True
        ad["propagated"] = False
        ad["subsampled"] = len(seed_frame_ids) != total_frames
        seed_answer_map[int(frame_id)] = ad

    seed_frame_id_set = set(int(fid) for fid in seed_frame_ids)
    full_answer_dicts: list[dict] = []
    for frame_id in range(total_frames):
        if frame_id in seed_answer_map:
            ad = dict(seed_answer_map[frame_id])
        else:
            ad = {
                "frame_id": frame_id,
                "seed_frame": False,
                "propagated": True,
                "subsampled": len(seed_frame_ids) != total_frames,
                "propagation_source": "sam3_mask_prompt_propagation",
            }
        ad["upsampled"] = True
        ad["upsample_method"] = "sam3_mask_prompt_propagation"
        ad["upsample_seed_frame_ids"] = [int(i) for i in valid_seed_frame_ids_full]
        ad["is_seed_frame"] = frame_id in seed_frame_id_set
        full_answer_dicts.append(ad)

    valid_frame_ids = [
        fid for fid, m in enumerate(propagated_masks)
        if int(np.asarray(m).astype(bool).sum()) > 0
    ]
    return (
        [np.asarray(m).astype(bool) for m in propagated_masks],
        full_answer_dicts,
        valid_frame_ids,
    )

def _segment_one_role(
    refseg: Any,
    full_rgb_np: list[np.ndarray],
    part_description: str,
    sam3_tracker: Any,
    log: Any,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    total = len(full_rgb_np)
    seed_frame_ids = evenly_spaced_indices(total, SEGMENTATION_SEED_FRAMES)
    log(
        f"  Seed frame indices ({len(seed_frame_ids)} of {total}): "
        f"{seed_frame_ids[:8]}{'…' if len(seed_frame_ids) > 8 else ''}"
    )
    seed_input_frames = [to_pil_rgb(full_rgb_np[i]) for i in seed_frame_ids]
    seed_pred_masks, seed_answer_dicts, seed_valid_frame_ids = refseg.segment_video(
        seed_input_frames, part_description
    )

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
        log(f"  No SAM3 propagation (subsample covers all frames); non-empty: {len(valid_ids)}.")

    return masks, valid_ids, seed_frame_ids

def _align_mask_to_hw(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.shape[:2] == (height, width):
        return m
    u8 = m.astype(np.uint8) * 255
    resized = cv2.resize(u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 127

def _apply_color_tint(out: np.ndarray, mask: np.ndarray | None, color: np.ndarray, alpha: float) -> None:
    if mask is not None and mask.any():
        m = mask.astype(bool)
        for c in range(3):
            out[m, c] = (1 - alpha) * out[m, c] + alpha * color[c]

def _apply_outlines(
    out: np.ndarray,
    receiver_mask: np.ndarray | None,
    effector_mask: np.ndarray | None,
    thickness: int = 6,
    color: tuple = (215, 215, 215),
) -> None:
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
    receiver_mask: np.ndarray | None,
    effector_mask: np.ndarray | None,
    alpha: float = 0.65,
    darken: float = 0.7,
    outline_thickness: int = 6,
    outline_color: tuple = (215, 215, 215),
) -> np.ndarray:
    """Receptor = Set2[0] green-teal tint, effector = Set2[1] orange; background darkened."""
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
    """Encode overlay frames to MP4 via ffmpeg.  Returns True on success."""
    n = len(rgb_frames)
    if n <= 0:
        return False
    with tempfile.TemporaryDirectory(prefix="pfr_seg_vis_") as tmpdir:
        for i in range(n):
            h, w = rgb_frames[i].shape[:2]
            recv = _align_mask_to_hw(receiver_masks[i], h, w)
            eff = _align_mask_to_hw(effector_masks[i], h, w)
            overlay = overlay_masks_pop(rgb_frames[i], recv, eff)
            PILImage.fromarray(overlay).save(os.path.join(tmpdir, f"frame_{i:04d}.png"))
        pattern = os.path.join(os.path.abspath(tmpdir), "frame_%04d.png")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-start_number", "0", "-i", pattern,
            "-frames:v", str(n),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            output_mp4_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            err = getattr(exc, "stderr", None)
            if err:
                sys.stderr.write(err.decode() if isinstance(err, bytes) else str(err))
            return False

def _itaco_joint_arrow_mesh(
    origin: np.ndarray,
    direction: np.ndarray,
    height: float,
    shaft_radius: float,
    head_radius: float,
) -> Any:
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
    return np.concatenate(chunks, axis=0) if chunks else None

def _masked_world_points_xyz_for_masks(
    recon_results: dict[str, Any], masks_per_frame: list[np.ndarray]
) -> np.ndarray | None:
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
    return np.concatenate(chunks, axis=0) if chunks else None

def _point_cloud_proxy_spheres(
    vertices: np.ndarray,
    colors_rgba: np.ndarray,
    sphere_radius: float,
    rng: np.random.Generator,
    max_spheres: int,
) -> Any | None:
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
        s.visual.face_colors = np.tile(c[i: i + 1], (len(s.faces), 1))
        pieces.append(s)
    return trimesh.util.concatenate(pieces)

def _export_recon_articulation_glb(
    recon_results: dict[str, Any],
    art_out: dict[str, Any],
    fused_by_role: dict[str, np.ndarray | None],
    segmentation_masks_by_role: dict[str, list[np.ndarray]],
    max_pts: int = 120_000,
) -> tuple[str | None, np.ndarray | None]:
    """Export a GLB: per-role fused point clouds + iTACO joint-axis arrows.

    Applies the OpenCV → glTF coordinate flip (Y-down, Z-into-scene →
    Y-up, Z-toward-viewer) baked into vertex positions so that the
    three.js OrbitControls bounding-sphere fit lands in the right place.

    Returns ``(glb_path, scene_center_world)`` or ``(None, None)`` on failure.
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
        rgba_i = np.tile(role_rgb_pt[role], (n_i, 1))
        point_layers.append((pts_i, rgba_i))

    if not point_layers:
        all_pts = _masked_world_points_xyz(recon_results)
        if all_pts is None or all_pts.shape[0] == 0:
            return None, None
        extent_basis.append(all_pts)
        n = min(max_pts, all_pts.shape[0])
        idx = rng.choice(all_pts.shape[0], size=n, replace=False)
        pts = all_pts[idx].astype(np.float64)
        z = pts[:, 2]
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
        direction = (axis / an) * _cv2gl
        origin_t = origin * _cv2gl
        arrow = _itaco_joint_arrow_mesh(origin_t, direction, arrow_len, shaft_d, head_d)
        c = role_rgba_mesh[role]
        arrow.visual.face_colors = np.tile(c, (len(arrow.faces), 1))
        geometries.append(arrow)

    if not geometries:
        return None, None

    scene_center = all_ext.mean(axis=0).astype(np.float64)
    scene = trimesh.Scene(geometries)
    fd, out_path = tempfile.mkstemp(suffix=".glb", prefix="pfr_scene_")
    os.close(fd)
    scene.export(out_path)
    return out_path, scene_center

def get_seg_model(seg_cfg: omegaconf.DictConfig, device: str) -> Any:
    key = (
        f"seg|{device}|{seg_cfg.get('model', 'SegZero')}"
        f"|{seg_cfg.get('reasoning_model_path', '')}"
        f"|{seg_cfg.get('segmentation_model_path', '')}"
    )
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_refseg_model(seg_cfg)
    return _MODEL_CACHE[key]

def get_sam3_tracker_cached(
    device: str,
    checkpoint_path: str | None = None,
    bpe_path: str | None = None,
) -> Any:
    ck = checkpoint_path or ""
    bp = bpe_path or ""
    key = f"sam3_vid|{device}|{ck}|{bp}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_sam3_tracker(
            device=device,
            checkpoint_path=checkpoint_path or None,
            bpe_path=bpe_path or None,
        )
    return _MODEL_CACHE[key]

def get_recon_model_cached(device: str, da3_model_path: str) -> Any:
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

def get_fusion_model_cached(device: str) -> Any:
    key = f"fusion|{device}"
    if key not in _MODEL_CACHE:
        cfg = omegaconf.OmegaConf.create({"fusion_method": "feature_matching", "device": device})
        _MODEL_CACHE[key] = build_fusion_model(cfg)
    return _MODEL_CACHE[key]

def _itaco_yaml_cfg() -> omegaconf.DictConfig:
    global _itaco_yaml_cfg_cache
    if _itaco_yaml_cfg_cache is None:
        _itaco_yaml_cfg_cache = omegaconf.OmegaConf.load(_config_path("articulation", "iTACO.yaml"))
    return _itaco_yaml_cfg_cache

def get_itaco_cached(device: str) -> Any:
    """iTACO with internal frame-subsampling effectively disabled; ``run_demo``
    pre-subsamples via ``_subsample_for_itaco`` (mirrors pipeline.py behaviour)."""
    key = f"itaco|{device}|demo_presubsampled"
    if key not in _MODEL_CACHE:
        cfg = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(_itaco_yaml_cfg(), resolve=True)
        )
        cfg.device = device
        cfg.sample_strategy = "fix_num"
        cfg.sample_num = 2 ** 30
        _MODEL_CACHE[key] = build_articulation_estimation_model(cfg)
    return _MODEL_CACHE[key]

def _subsample_for_itaco(
    rgb_frame_list: list[np.ndarray],
    recon_results: dict[str, Any],
    part_masks: np.ndarray,
    sampling_cfg: omegaconf.DictConfig,
) -> tuple[list[np.ndarray], dict[str, Any], np.ndarray]:
    """Subsample frames and force C-contiguous arrays before passing to iTACO."""
    strat = str(sampling_cfg.get("sample_strategy", "fix_step"))
    sn = int(sampling_cfg.get("sample_num", 10))
    n = len(rgb_frame_list)
    if strat == "fix_num":
        sl = slice(None, None, max(n // sn, 1)) if n > sn else slice(None)
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

def _load_compile_module() -> Any:
    """Import compile/compile.py as a module without conflicting with the
    Python builtin ``compile``.  Adds compile/ to sys.path first so that the
    relative ``from build_urdf import ...`` inside compile.py resolves."""
    compile_dir = os.path.join(_ROOT, "compile")
    if compile_dir not in sys.path:
        sys.path.insert(0, compile_dir)
    spec = importlib.util.spec_from_file_location(
        "_egofun3d_compile",
        os.path.join(compile_dir, "compile.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_DESCRIPTION_PROMPT = (
    "This is a video. Analyze this video and answer the following questions: "
    "1. Which part of the object receives human action? "
    "2. Which part of the object reacts to human action? "
    "Please describe the name and features of the part as well as the spatial relationship with surrounding objects. "
    "Please only answer in this template: "
    "{1: {name: xxx, description: aaa}, 2: {name: yyy, description: bbb}} "
    'Substitue "xxx" and "yyy" with the name of the part of the object, "aaa" and "bbb" with the description of the part. '
    "DO NOT answer any other information."
)

def _parse_description_output(output_text: str) -> dict:
    """Extract receptor/effector {name, description} pairs from VLM output text."""
    import re
    pairs = re.findall(r'\{\s*name:\s*(.*?)\s*,\s*description:\s*(.*?)\s*\}', output_text, flags=re.S)

    def clean(t: str) -> str:
        return re.sub(r'\s+', ' ', t).strip()

    grouped: dict = {}
    if len(pairs) == 2:
        for i, (name, desc) in enumerate(pairs):
            role = "receptor" if i == 0 else "effector"
            grouped[role] = {"name": clean(name), "description": clean(desc)}
    return grouped

def _load_video_frames_for_narrator(video_path: str, max_frames: int = 32) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) > max_frames:
        step = max(len(frames) // max_frames, 1)
        frames = frames[::step][:max_frames]
    return [f.astype(np.uint8) for f in frames]

def _prompt_description_qwen(video_path: str, vlm_model: str, max_query: int = 10) -> dict:
    """Run Qwen via HuggingFace Transformers (single-process, no vLLM/multiprocessing).

    vLLM forces ``multiprocessing.spawn`` when CUDA is already initialized, which
    causes its EngineCore subprocesses to fail with ``CUDA driver initialization
    failed`` in nested subprocess environments.  Using the Transformers backend
    directly avoids all spawn/multiprocessing complexity.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(vlm_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm_model, trust_remote_code=True, dtype="auto", device_map="auto"
    )

    frames = _load_video_frames_for_narrator(video_path)
    td = tempfile.mkdtemp(prefix="qwen_desc_")
    tmp_video = os.path.join(td, "input.mp4")
    try:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": _DESCRIPTION_PROMPT},
                {"type": "video", "video": tmp_video},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

        result: dict = {}
        for _ in range(max_query):
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
            text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            result = _parse_description_output(text)
            if len(result) == 2:
                break
    finally:
        shutil.rmtree(td, ignore_errors=True)

    del model, processor
    return result

def _prompt_description_gemini(video_path: str, vlm_model: str, max_query: int = 10) -> dict:
    """Run Gemini to detect part labels.  Reads GEMINI_API_KEY from environment."""
    from google import genai

    client = genai.Client()
    video_bytes = open(video_path, "rb").read()
    result: dict = {}
    for _ in range(max_query):
        response = client.models.generate_content(
            model=vlm_model,
            contents=genai.types.Content(parts=[
                genai.types.Part(inline_data=genai.types.Blob(data=video_bytes, mime_type="video/mp4")),
                genai.types.Part(text=_DESCRIPTION_PROMPT),
            ]),
        )
        result = _parse_description_output(response.text)
        if len(result) == 2:
            break
    return result

def _prompt_part_labels(
    video_path: str,
    gemini_key: str | None = None,
    log: Any = print,
) -> tuple[str, str]:
    """Ask a VLM to watch the video and return (receptor_label, effector_label).

    Uses Qwen (vLLM) by default with tensor_parallel_size auto-set to GPU count.
    If ``gemini_key`` is provided, sets ``GEMINI_API_KEY`` and uses Gemini instead.
    """
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        vlm_model = omegaconf.OmegaConf.load(
            _config_path("vlm_segmentation", "gemini_segmentation.yaml")
        ).vlm_model
        log(f"  Using Gemini narrator ({vlm_model}) for label detection...")
        result = _prompt_description_gemini(video_path, vlm_model)
    else:
        vlm_model = omegaconf.OmegaConf.load(
            _config_path("vlm_segmentation", "qwen_segmentation.yaml")
        ).vlm_model
        log(f"  Using Qwen narrator ({vlm_model}, {max(1, torch.cuda.device_count())} GPU(s)) for label detection...")
        result = _prompt_description_qwen(video_path, vlm_model)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    receptor_info = result.get("receptor", {})
    effector_info = result.get("effector", {})
    receptor_label = receptor_info.get("description") or receptor_info.get("name") or "receptor part"
    effector_label = effector_info.get("description") or effector_info.get("name") or "effector part"
    return receptor_label, effector_label

def build_default_seg_config(device: str) -> omegaconf.DictConfig:
    """VisionReasoner + SAM3 backend — mirrors the Gradio demo defaults."""
    return omegaconf.OmegaConf.create(
        {
            "model": "SegZero",
            "sam_backend": _SAM_BACKEND,
            "reasoning_model_path": _REASONING_MODEL_PATH,
            "segmentation_model_path": _SEGMENTATION_MODEL_PATH,
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

def run_pipeline(
    video_path: str,
    output_dir: str,
    receptor_label: str | None = None,
    effector_label: str | None = None,
    gemini_key: str | None = None,
    seg_config: omegaconf.DictConfig | None = None,
    device: str = "cuda",
    da3_model_path: str = _DA3_MODEL_PATH,
    robot_name: str = "articulated_object",
    delta: float | None = None,
    emitter_position: tuple | None = None,
    save_mesh_outputs: bool = True,
) -> dict[str, Any]:
    """Run the full EgoFun3D pipeline on *video_path* and write results under
    *output_dir* in the evaluation suite's directory layout.

    If *receptor_label* / *effector_label* are ``None``, a VLM narrator
    (Qwen by default, Gemini if *gemini_key* is provided) watches the video
    and auto-detects the part labels before segmentation.

    Returns a summary dict with paths to key outputs and raw result dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "demo_log.txt")
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)
        with open(log_path, "a") as _f:
            _f.write(msg + "\n")

    log(f"EgoFun3D headless demo")
    log(f"  video      : {video_path}")
    log(f"  output_dir : {output_dir}")
    log(f"  device     : {device}")

    if seg_config is None:
        seg_config = build_default_seg_config(device)
        log("Segmentation: default VisionReasoner + SAM3.")
    else:
        log(f"Segmentation: custom config (model={seg_config.get('model', '?')}).")

    if receptor_label is None or effector_label is None:
        narrator = "Gemini" if gemini_key else "Qwen"
        log(f"No part labels provided — using {narrator} to auto-detect receptor/effector...")
        receptor_label, effector_label = _prompt_part_labels(video_path, gemini_key, log)
        log(f"  Auto-detected receptor : {receptor_label!r}")
        log(f"  Auto-detected effector : {effector_label!r}")
    else:
        log(f"  receptor   : {receptor_label!r}")
        log(f"  effector   : {effector_label!r}")

    log("Loading full video (every frame, in order)...")
    pil_frames, _rgb_unused, input_video_fps = load_full_video(video_path)
    rgb_np = video_frames_as_numpy_hwc_uint8(pil_frames)
    total_f = len(rgb_np)
    log(f"Loaded {total_f} frames at {input_video_fps:.2f} fps.")

    init_extrinsic = np.eye(4, dtype=np.float64)

    seed_ids_preview = evenly_spaced_indices(total_f, SEGMENTATION_SEED_FRAMES)
    should_propagate = len(seed_ids_preview) < total_f

    log("Loading VisionReasoner segmentation model...")
    refseg = get_seg_model(seg_config, device)
    sam3_tracker = None
    if should_propagate:
        log("Loading SAM3 video tracker for mask propagation...")
        sam3_tracker = get_sam3_tracker_cached(device)

    receptor_desc = f"{receptor_label}. {receptor_label}"
    effector_desc = f"{effector_label}. {effector_label}"

    log(f"Segmenting receptor ('{receptor_label}')...")
    r_masks, r_valid, _r_seeds = _segment_one_role(
        refseg, rgb_np, receptor_desc, sam3_tracker, log
    )
    log(f"Receptor: {len(r_valid)} non-empty mask frames.")

    log(f"Segmenting effector ('{effector_label}')...")
    e_masks, e_valid, _e_seeds = _segment_one_role(
        refseg, rgb_np, effector_desc, sam3_tracker, log
    )
    log(f"Effector: {len(e_valid)} non-empty mask frames.")

    r_stack = np.stack([m.astype(bool) for m in r_masks], axis=0)
    e_stack = np.stack([m.astype(bool) for m in e_masks], axis=0)

    seg_mp4_path: str | None = os.path.join(output_dir, "segmentation_overlay.mp4")
    log(f"Rendering segmentation overlay video ({input_video_fps:g} fps)...")
    if frames_to_overlay_mp4(rgb_np, r_masks, e_masks, seg_mp4_path, fps=input_video_fps):
        log(f"  → {seg_mp4_path}")
    else:
        log("  Warning: ffmpeg not available; segmentation overlay video skipped.")
        seg_mp4_path = None

    log("Releasing segmentation/SAM3 models from GPU cache before DA3...")
    refseg = None
    sam3_tracker = None
    _drop_cache_keys_starting_with("seg|", "sam3_vid|")

    log("Running Depth Anything 3 reconstruction...")
    recon_model = get_recon_model_cached(device, da3_model_path)
    recon_results = recon_model.reconstruct(rgb_np, init_extrinsic, None, None, None)
    if recon_results is None:
        raise RuntimeError("DA3 reconstruction failed.")
    recon_results = refine_point_mask(recon_results)
    log("Reconstruction done.")

    log("Releasing DA3 before feature-matching (RoMa) fusion...")
    recon_model = None
    _drop_cache_keys_starting_with("da3|")

    log("Running feature-matching (RoMa) fusion per part...")
    fusion = get_fusion_model_cached(device)
    fused_by_role: dict[str, np.ndarray | None] = {"receptor": None, "effector": None}
    transformation_by_role: dict[str, tuple[Any, list[int]]] = {}
    kpts_a: dict = {}
    kpts_b: dict = {}

    _no_amp = (
        torch.autocast(device_type="cuda", enabled=False)
        if device == "cuda" and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with _no_amp:
        for role, masks in (("receptor", r_masks), ("effector", e_masks)):
            valid_ids = [i for i, m in enumerate(masks) if np.asarray(m).sum() > 0]
            if not valid_ids:
                log(f"  [{role}] no non-empty masks; skipping fusion.")
                continue
            pm = recon_results["points_mask"]
            vm = [np.logical_and(pm[i], np.asarray(masks[i], dtype=bool)) for i in valid_ids]
            pts = [recon_results["points"][i] for i in valid_ids]
            frames_hwc = [rgb_np[i] for i in valid_ids]
            fused_pcd, trans_list, kpts_a, kpts_b = fusion.fuse_part_pcds(
                frames_hwc, vm, pts, kpts_a, kpts_b
            )
            fused_by_role[role] = np.asarray(fused_pcd, dtype=np.float64).reshape(-1, 3)
            transformation_by_role[role] = (trans_list, valid_ids)
            log(f"  [{role}] fused PCD: {int(fused_pcd.shape[0])} points.")

        if save_mesh_outputs:
            base_mask_arr = np.logical_not(np.logical_or(r_stack, e_stack))
            base_valid_ids = [i for i, m in enumerate(base_mask_arr) if m.sum() > 0]
            if base_valid_ids:
                base_fusion = FeatureMatchingFusion(device=device)
                base_vm = [
                    np.logical_and(recon_results["points_mask"][i], base_mask_arr[i])
                    for i in base_valid_ids
                ]
                base_pts = [recon_results["points"][i] for i in base_valid_ids]
                base_frames = [rgb_np[i] for i in base_valid_ids]
                _, base_trans, _, _ = base_fusion.fuse_part_pcds(
                    base_frames, base_vm, base_pts, {}, {}
                )
                transformation_by_role["base"] = (base_trans, base_valid_ids)
                log(f"  [base] fused (for base mesh), {len(base_valid_ids)} valid frames.")
            else:
                log("  [base] no non-base mask frames; base_mesh.glb will be skipped.")

    log("Releasing RoMa fusion before iTACO articulation...")
    fusion = None
    _drop_cache_keys_starting_with("fusion|")

    log("Running iTACO articulation estimation (receptor + effector)...")
    itaco = get_itaco_cached(device)
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
            rgb_s, recon_s, masks_s = _subsample_for_itaco(
                rgb_np, recon_for_itaco, stack, itaco_samp
            )
            est = itaco.articulation_estimation(rgb_s, recon_s, masks_s)
            art_out[role] = est
            brief = json.dumps(est, default=_numpy_json_default)[:160] if est else "None"
            log(f"  [{role}] articulation: {brief}…")

    log("Exporting 3-D scene GLB (reconstruction + articulation axes)...")
    tmp_glb, _ = _export_recon_articulation_glb(
        recon_results,
        art_out,
        fused_by_role,
        {"receptor": r_masks, "effector": e_masks},
    )
    glb_out_path: str | None = None
    if tmp_glb:
        glb_out_path = os.path.join(output_dir, "scene_3d.glb")
        shutil.move(tmp_glb, glb_out_path)
        log(f"  → {glb_out_path}")
    else:
        log("  Warning: 3-D GLB export failed (trimesh unavailable or no point data).")

    recon_dir = os.path.join(output_dir, "reconstruction")
    os.makedirs(recon_dir, exist_ok=True)

    for role, pcd in fused_by_role.items():
        if pcd is not None:
            ply_path = os.path.join(recon_dir, f"{role}_fused.ply")
            save_pcd(pcd, ply_path)
            log(f"  [{role}] fused PLY → {ply_path}")

    if save_mesh_outputs:
        image_arr = np.array(rgb_np)
        for role, masks_list in (("receptor", r_masks), ("effector", e_masks)):
            if role not in transformation_by_role:
                log(f"  [{role}] no transformation; skipping {role}_mesh.glb.")
                continue
            trans_list, valid_ids_role = transformation_by_role[role]
            mask_arr = np.array([m.astype(bool) for m in masks_list])
            mesh_path = os.path.join(recon_dir, f"{role}_mesh.glb")
            save_mesh(
                reconstruction_results=recon_results,
                image_list=image_arr,
                mask_list=mask_arr,
                transformation_list=trans_list,
                save_path=mesh_path,
                observation_indices=np.asarray(valid_ids_role, dtype=int),
                num_observations=3,
            )
            log(f"  [{role}] mesh GLB → {mesh_path}")

        if "base" in transformation_by_role:
            base_trans, base_valid_ids = transformation_by_role["base"]
            base_mask_arr = np.logical_not(np.logical_or(r_stack, e_stack))
            base_mesh_path = os.path.join(recon_dir, "base_mesh.glb")
            save_mesh(
                reconstruction_results=recon_results,
                image_list=image_arr,
                mask_list=base_mask_arr,
                transformation_list=base_trans,
                save_path=base_mesh_path,
                observation_indices=np.asarray(base_valid_ids, dtype=int),
                num_observations=3,
            )
            log(f"  [base] mesh GLB → {base_mesh_path}")

    recon_pkl_path = os.path.join(recon_dir, "reconstruction_results.pkl.gz")
    save_reconstruction_results(recon_results, recon_pkl_path)
    log(f"  Reconstruction results → {recon_pkl_path}")

    art_dir = os.path.join(output_dir, "articulation")
    os.makedirs(art_dir, exist_ok=True)
    art_results_path = os.path.join(art_dir, "articulation_results.json")
    save_articulation_results(art_out, art_results_path)
    log(f"  Articulation results → {art_results_path}")

    log("Releasing all remaining models before Qwen VLM...")
    itaco = None
    _clear_model_cache_and_empty_cuda()

    log("Running Qwen function VLM (QwenVideoNarrator)...")
    vlm_cfg = omegaconf.OmegaConf.load(_config_path("vlm_function", "qwen_function.yaml"))
    fn_vlm = build_vlm_prompter(vlm_cfg)
    fn_res = fn_vlm.prompt_function(rgb_np, r_stack, e_stack)
    log(f"  Function VLM result: {json.dumps(fn_res, default=_numpy_json_default)}")

    fn_dir = os.path.join(output_dir, "function")
    os.makedirs(fn_dir, exist_ok=True)
    fn_results_path = os.path.join(fn_dir, "function_results.json")
    save_function_results(fn_res, fn_results_path)
    log(f"  Function results → {fn_results_path}")

    compile_output_dir = os.path.join(output_dir, "compile")
    log(f"Running compile step → {compile_output_dir} ...")
    compile_result: dict | None = None

    try:
        compile_mod = _load_compile_module()
        compile_kwargs: dict[str, Any] = {}
        if emitter_position is not None:
            compile_kwargs["emitter_position"] = tuple(float(v) for v in emitter_position)
        compile_result = compile_mod.compile_function_instance(
            reconstruction_dir=output_dir,
            articulation_dir=output_dir,
            function_dir=output_dir,
            output_dir=compile_output_dir,
            robot_name=robot_name,
            delta=delta,
            **compile_kwargs,
        )
        log(
            f"  Compile complete: "
            f"physical_effect={compile_result.get('physical_effect')}, "
            f"mapping_type={compile_result.get('mapping_type')}"
        )
        log(f"  URDF → {compile_result.get('urdf_result', {}).get('urdf_path')}")
        log(f"  Script → {compile_result.get('function_script_path')}")
    except Exception:
        log(f"  Warning: compile step failed:\n{traceback.format_exc()}")

    log("Done. Full EgoFun3D pipeline complete.")

    return {
        "output_dir": output_dir,
        "segmentation_overlay_mp4": seg_mp4_path,
        "scene_3d_glb": glb_out_path,
        "reconstruction_dir": recon_dir,
        "articulation_results_json": art_results_path,
        "function_results_json": fn_results_path,
        "compile_output_dir": compile_output_dir,
        "articulation_results": art_out,
        "function_results": fn_res,
        "compile_result": compile_result,
    }

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EgoFun3D headless demo: run the full pipeline on a single video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where all outputs are saved.",
    )
    parser.add_argument(
        "--receptor", default=None, metavar="LABEL",
        help=(
            "Natural-language description of the receptor part (e.g. 'faucet handle'). "
            "When omitted, a VLM narrator auto-detects the label from the video "
            "(Qwen by default; Gemini if --gemini_key is provided)."
        ),
    )
    parser.add_argument(
        "--effector", default=None, metavar="LABEL",
        help=(
            "Natural-language description of the effector part (e.g. 'faucet spout'). "
            "When omitted, a VLM narrator auto-detects the label from the video."
        ),
    )
    parser.add_argument(
        "--gemini_key", default=None, metavar="API_KEY",
        help=(
            "Gemini API key.  When provided, Gemini (gemini_segmentation.yaml) is used "
            "for part-label auto-detection instead of Qwen.  Also sets the "
            "GEMINI_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--seg_config", default=None, metavar="YAML",
        help=(
            "Path to an OmegaConf segmentation config YAML "
            "(e.g. config/segmentation/VisionReasoner.yaml). "
            "Defaults to VisionReasoner + SAM3 — same settings as the Gradio demo."
        ),
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Compute device passed to all models.",
    )
    parser.add_argument(
        "--da3_model", default=_DA3_MODEL_PATH, metavar="HF_ID",
        help="HuggingFace model ID or local path for Depth Anything 3.",
    )
    parser.add_argument(
        "--robot_name", default="articulated_object",
        help="Robot name embedded in the URDF <robot> tag.",
    )
    parser.add_argument(
        "--delta", type=float, default=None,
        help="Delta value for cumulative numerical mapping (compile step only).",
    )
    parser.add_argument(
        "--emitter_position", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Fluid emitter position XYZ for the compile step. "
            "Auto-estimated from the effector mesh OBB if omitted."
        ),
    )
    parser.add_argument(
        "--no_mesh", action="store_true",
        help=(
            "Skip saving part/base GLB meshes (faster, but the compile step "
            "will be unavailable)."
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    if os.path.abspath(os.getcwd()) != os.path.abspath(_ROOT):
        print(
            f"Warning: cwd is {os.getcwd()!r}; recommended to run from {_ROOT!r} "
            "so third_party paths in fusion/reconstruction resolve correctly.",
            file=sys.stderr,
        )

    args = _parse_args()

    seg_cfg: omegaconf.DictConfig | None = None
    if args.seg_config:
        seg_cfg = omegaconf.OmegaConf.load(args.seg_config)

    result = run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        receptor_label=args.receptor,
        effector_label=args.effector,
        gemini_key=args.gemini_key,
        seg_config=seg_cfg,
        device=args.device,
        da3_model_path=args.da3_model,
        robot_name=args.robot_name,
        delta=args.delta,
        emitter_position=tuple(args.emitter_position) if args.emitter_position else None,
        save_mesh_outputs=not args.no_mesh,
    )

    print("\nPipeline complete.  Key outputs:")
    for k, v in result.items():
        if k not in ("articulation_results", "function_results", "compile_result"):
            print(f"  {k}: {v}")
