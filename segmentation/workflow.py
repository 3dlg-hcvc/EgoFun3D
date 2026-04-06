import gzip
import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image as PILImage


MASK_ARCHIVE_NAME = "segmentation_masks.npz"
ANSWER_ARCHIVE_NAME = "segmentation_answers.json"


def evenly_spaced_indices(total_frames: int, num_samples: int | None) -> list[int]:
    if total_frames <= 0:
        return []
    if num_samples is None:
        return list(range(total_frames))
    try:
        num_samples_int = int(num_samples)
    except (TypeError, ValueError):
        return list(range(total_frames))
    if num_samples_int <= 0 or num_samples_int >= total_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num_samples_int)
    indices = np.round(indices).astype(int).tolist()
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return unique_indices


def to_pil_rgb(frame: Any) -> PILImage.Image:
    if isinstance(frame, PILImage.Image):
        return frame.convert("RGB")
    if isinstance(frame, np.ndarray):
        frame_np = frame
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        return PILImage.fromarray(frame_np).convert("RGB")
    raise TypeError(f"Unsupported frame type: {type(frame)}")


def to_numpy_rgb(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        frame_np = frame
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        return frame_np
    return np.asarray(to_pil_rgb(frame), dtype=np.uint8)


def segmentation_mask_archive_path(role_dir: str) -> str:
    return os.path.join(role_dir, MASK_ARCHIVE_NAME)


def segmentation_answers_path(role_dir: str) -> str:
    return os.path.join(role_dir, ANSWER_ARCHIVE_NAME)


def save_segmentation_mask_archive(mask_list: list[np.ndarray], role_dir: str) -> str:
    os.makedirs(role_dir, exist_ok=True)
    masks = np.stack([np.asarray(mask).astype(bool) for mask in mask_list], axis=0)
    archive_path = segmentation_mask_archive_path(role_dir)
    np.savez_compressed(archive_path, masks=masks.astype(np.uint8))
    return archive_path


def save_segmentation_answers(answer_dict_list: list[dict], role_dir: str) -> str:
    os.makedirs(role_dir, exist_ok=True)
    answers_path = segmentation_answers_path(role_dir)
    with open(answers_path, "w") as f:
        json.dump(answer_dict_list, f, indent=4)
    return answers_path


def load_segmentation_mask_archive(role_dir: str) -> np.ndarray | None:
    archive_path = segmentation_mask_archive_path(role_dir)
    if not os.path.exists(archive_path):
        return None
    archive = np.load(archive_path)
    if "masks" in archive:
        masks = archive["masks"]
    else:
        first_key = next(iter(archive.files), None)
        if first_key is None:
            return None
        masks = archive[first_key]
    return np.asarray(masks).astype(bool)


def load_segmentation_answers(role_dir: str, total_frames: int | None = None) -> list[dict]:
    answers_path = segmentation_answers_path(role_dir)
    if os.path.exists(answers_path):
        try:
            with open(answers_path, "r") as f:
                answers = json.load(f)
            if isinstance(answers, list):
                parsed_answers = [dict(answer) if isinstance(answer, dict) else {} for answer in answers]
                if total_frames is None or len(parsed_answers) == total_frames:
                    return parsed_answers
        except (OSError, json.JSONDecodeError):
            pass

    if total_frames is None:
        return []
    answers = []
    for frame_id in range(total_frames):
        answer_path = os.path.join(role_dir, f"segmentation_answer_{frame_id:04d}.json")
        if os.path.exists(answer_path):
            try:
                with open(answer_path, "r") as f:
                    answer = json.load(f)
            except (OSError, json.JSONDecodeError):
                answer = {"frame_id": frame_id, "skipped": True, "skip_reason": "invalid_existing_answer"}
        else:
            answer = {"frame_id": frame_id, "skipped": True, "skip_reason": "missing_existing_answer"}
        if isinstance(answer, dict):
            answer.setdefault("frame_id", frame_id)
        answers.append(answer)
    return answers


def _load_legacy_mask_frames(role_dir: str, num_frames: int) -> np.ndarray:
    masks = []
    for frame_id in range(num_frames):
        mask_path = os.path.join(role_dir, f"segmentation_mask_{frame_id:04d}.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing legacy segmentation mask: {mask_path}")
        masks.append(np.load(mask_path).astype(bool))
    return np.stack(masks, axis=0)


def align_masks_to_sampled_frames(data: dict, masks: np.ndarray) -> np.ndarray:
    sampled_video_frame_list = list(data["rgb_list"])
    sampled_num_frames = len(sampled_video_frame_list)
    sample_indices = [int(i) for i in data.get("sample_indices", list(range(sampled_num_frames)))]

    masks = np.asarray(masks).astype(bool)
    if masks.shape[0] == sampled_num_frames:
        sampled_masks = masks
    elif len(sample_indices) > 0 and masks.shape[0] > max(sample_indices):
        sampled_masks = masks[sample_indices]
    else:
        raise ValueError(
            f"Cannot align segmentation masks of shape {masks.shape} with sample indices {sample_indices[:10]}"
        )

    target_hw = tuple(np.asarray(sampled_video_frame_list[0]).shape[:2]) if sampled_num_frames > 0 else None
    if target_hw is None or sampled_masks.shape[1:] == target_hw:
        return sampled_masks

    crop_top_left = data.get("cropped_top_left", [0, 0])
    crop_bottom_right = data.get("cropped_bottom_right", [sampled_masks.shape[2], sampled_masks.shape[1]])
    x0, y0 = int(crop_top_left[0]), int(crop_top_left[1])
    x1, y1 = int(crop_bottom_right[0]), int(crop_bottom_right[1])
    if (
        0 <= y0 < y1 <= sampled_masks.shape[1]
        and 0 <= x0 < x1 <= sampled_masks.shape[2]
        and (y1 - y0, x1 - x0) == target_hw
    ):
        return sampled_masks[:, y0:y1, x0:x1]

    raise ValueError(
        f"Segmentation masks have spatial shape {sampled_masks.shape[1:]}, "
        f"but sampled frames expect {target_hw}."
    )


def load_segmentation_masks_for_sample(data: dict, role_dir: str) -> np.ndarray:
    masks = load_segmentation_mask_archive(role_dir)
    if masks is None:
        sampled_num_frames = len(data["rgb_list"])
        sample_indices = [int(i) for i in data.get("sample_indices", list(range(sampled_num_frames)))]
        try:
            masks = _load_legacy_mask_frames(role_dir, sampled_num_frames)
        except FileNotFoundError:
            if len(sample_indices) > 0:
                masks = _load_legacy_mask_frames(role_dir, max(sample_indices) + 1)
            else:
                raise
    return align_masks_to_sampled_frames(data, masks)


def _is_identity_sample(sample_indices: list[int], num_frames: int) -> bool:
    return len(sample_indices) == num_frames and sample_indices == list(range(num_frames))


def load_full_video_frames(data: dict) -> list[np.ndarray]:
    sampled_frames = [to_numpy_rgb(frame) for frame in data["rgb_list"]]
    sample_indices = [int(i) for i in data.get("sample_indices", list(range(len(sampled_frames))))]
    if _is_identity_sample(sample_indices, len(sampled_frames)):
        return sampled_frames
    full_frames = imageio.v3.imread(data["video_path"])
    return [to_numpy_rgb(frame) for frame in full_frames]


def _select_mask_array(mask_data: dict, role: str) -> np.ndarray:
    target_idx = {"receptor": 3, "effector": 4, "object": 5}[role]
    for mask_info in mask_data.values():
        if int(mask_info["mask_idx"]) == target_idx:
            return np.asarray(mask_info["masks"]).astype(bool)
    raise KeyError(f"Role mask {role} not found in annotation file.")


def load_full_role_masks(data: dict, role: str) -> np.ndarray:
    sampled_masks = np.asarray(data[f"{role}_mask_list"]).astype(bool)
    sample_indices = [int(i) for i in data.get("sample_indices", list(range(len(sampled_masks))))]
    if _is_identity_sample(sample_indices, len(sampled_masks)):
        return sampled_masks

    video_mask_path = data.get("video_mask_path")
    if video_mask_path is None:
        return sampled_masks

    with gzip.open(video_mask_path, "rb") as f:
        mask_data = pickle.load(f)
    full_masks = _select_mask_array(mask_data, role)

    crop_top_left = data.get("cropped_top_left", [0, 0])
    crop_bottom_right = data.get("cropped_bottom_right", [full_masks.shape[2], full_masks.shape[1]])
    x0, y0 = int(crop_top_left[0]), int(crop_top_left[1])
    x1, y1 = int(crop_bottom_right[0]), int(crop_bottom_right[1])
    return full_masks[:, y0:y1, x0:x1]


@contextmanager
def _make_sam3_compatible_jpeg_dir(frame_list: list[np.ndarray]):
    with tempfile.TemporaryDirectory(prefix="sam3_frames_") as tmp_dir_str:
        for idx, frame in enumerate(frame_list):
            frame_path = os.path.join(tmp_dir_str, f"{idx:06d}.jpg")
            to_pil_rgb(frame).save(frame_path, format="JPEG", quality=95)
        yield tmp_dir_str


def build_sam3_tracker(device: str, checkpoint_path: str | None = None, bpe_path: str | None = None):
    from sam3.model_builder import build_sam3_video_model

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_bpe_path = bpe_path
    if resolved_bpe_path is None:
        candidate_bpe_path = os.path.join(repo_root, "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(candidate_bpe_path):
            resolved_bpe_path = candidate_bpe_path

    model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        bpe_path=resolved_bpe_path,
        device=device,
        load_from_HF=checkpoint_path is None,
    )
    tracker = model.tracker
    if getattr(tracker, "backbone", None) is None:
        tracker.backbone = model.detector.backbone
    tracker.to(device=device)
    tracker.eval()
    return tracker


def _extract_single_mask(video_res_masks: torch.Tensor, obj_ids: list[int], obj_id: int) -> np.ndarray:
    if len(obj_ids) == 0:
        h = int(video_res_masks.shape[-2])
        w = int(video_res_masks.shape[-1])
        return np.zeros((h, w), dtype=bool)
    obj_idx = obj_ids.index(obj_id) if obj_id in obj_ids else 0
    mask_logits = video_res_masks[obj_idx]
    while mask_logits.ndim > 2:
        mask_logits = mask_logits[0]
    return (mask_logits > 0).detach().cpu().numpy().astype(bool)


def propagate_full_video_from_masks(
    tracker,
    frame_list: list[np.ndarray],
    seed_masks: list[tuple[int, np.ndarray]],
    *,
    offload_video_to_cpu: bool = False,
) -> list[np.ndarray]:
    total_frames = len(frame_list)
    if total_frames == 0:
        return []
    if len(seed_masks) == 0:
        height, width = frame_list[0].shape[:2]
        zero_mask = np.zeros((height, width), dtype=bool)
        return [zero_mask.copy() for _ in range(total_frames)]

    seed_masks = sorted(
        [(int(frame_id), np.asarray(mask).astype(bool)) for frame_id, mask in seed_masks],
        key=lambda item: item[0],
    )
    start_frame_idx = int(seed_masks[0][0])
    obj_id = 1

    with _make_sam3_compatible_jpeg_dir(frame_list) as sam3_frame_dir:
        inference_state = tracker.init_state(
            video_path=sam3_frame_dir,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        video_h = int(inference_state["video_height"])
        video_w = int(inference_state["video_width"])
        pred_by_frame: dict[int, np.ndarray] = {}
        try:
            for frame_idx, mask in seed_masks:
                tracker.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(frame_idx),
                    obj_id=obj_id,
                    mask=torch.from_numpy(mask.astype(np.float32)),
                    add_mask_to_memory=False,
                )

            for reverse, preflight in ((False, True), (True, False)):
                for frame_idx, obj_ids, _, video_res_masks, _ in tracker.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=total_frames,
                    reverse=reverse,
                    tqdm_disable=False,
                    propagate_preflight=preflight,
                ):
                    if int(frame_idx) in pred_by_frame:
                        continue
                    pred_by_frame[int(frame_idx)] = _extract_single_mask(
                        video_res_masks=video_res_masks,
                        obj_ids=[int(x) for x in obj_ids],
                        obj_id=obj_id,
                    )
        finally:
            if hasattr(tracker, "reset_state"):
                tracker.reset_state(inference_state)
            elif hasattr(tracker, "clear_all_points_in_video"):
                tracker.clear_all_points_in_video(inference_state)

    fallback = np.zeros((video_h, video_w), dtype=bool)
    return [pred_by_frame.get(frame_id, fallback.copy()) for frame_id in range(total_frames)]
