import json
import os
import random
import time
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.dataset import build_dataset
from segmentation.evaluate_segmentation import (
    build_shared_vlm_output,
    compute_part_iou_video,
    load_shared_vlm_output,
    save_segmentation_metrics,
    save_segmentation_video,
    save_shared_vlm_output,
    save_vlm_output,
)
from segmentation.ref_seg import build_refseg_model
from segmentation.workflow import (
    align_masks_to_sampled_frames,
    build_sam3_tracker,
    evenly_spaced_indices,
    load_full_role_masks,
    load_full_video_frames,
    load_segmentation_answers,
    load_segmentation_mask_archive,
    propagate_full_video_from_masks,
    to_pil_rgb,
)
from VLM.prompt_vlm import build_vlm_prompter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def identity_collate(batch):
    return batch[0]



def _get_optional_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    if key not in config:
        return default
    if OmegaConf.is_missing(config, key):
        return default
    value = config[key]
    return default if value is None else value



def _normalize_grouped_results(grouped_results: dict | None) -> dict | None:
    if not isinstance(grouped_results, dict):
        return None
    if 'receptor' in grouped_results and 'effector' in grouped_results:
        return grouped_results
    if 'receiver' in grouped_results and 'effector' in grouped_results:
        return {
            'receptor': grouped_results['receiver'],
            'effector': grouped_results['effector'],
        }
    return None



def _normalize_shared_vlm_results(shared_vlm_output: dict | None) -> dict | None:
    if not isinstance(shared_vlm_output, dict):
        return None
    parts = shared_vlm_output.get('parts', {})
    if not isinstance(parts, dict):
        return None
    receptor = parts.get('receptor', parts.get('receiver'))
    effector = parts.get('effector')
    if not isinstance(receptor, dict) or not isinstance(effector, dict):
        return None
    return {
        'receptor': {
            'name': receptor.get('label', 'unknown'),
            'description': receptor.get('description', receptor.get('label', 'unknown')),
        },
        'effector': {
            'name': effector.get('label', 'unknown'),
            'description': effector.get('description', effector.get('label', 'unknown')),
        },
    }



def _prompt_description(vlm_prompter: Any, video_path: str) -> dict:
    if hasattr(vlm_prompter, 'prompt_description'):
        return vlm_prompter.prompt_description(video_path)
    if hasattr(vlm_prompter, 'prompt'):
        return vlm_prompter.prompt(video_path)
    raise AttributeError('VLM prompter does not expose prompt_description() or prompt().')



def _save_segmentation_runtime(runtime_info: dict, save_dir: str, scene_name: str, seg_id: str, role: str) -> str:
    runtime_dir = os.path.join(save_dir, scene_name, seg_id, f'segmentation_{role}')
    os.makedirs(runtime_dir, exist_ok=True)
    runtime_path = os.path.join(runtime_dir, 'segmentation_runtime.json')
    with open(runtime_path, 'w') as f:
        json.dump(runtime_info, f, indent=4)
    return runtime_path



def _load_existing_vlm_output(save_dir: str, scene_name: str, seg_id: str) -> dict | None:
    vlm_path = os.path.join(save_dir, scene_name, seg_id, 'vlm_narrator', 'vlm_analysis.json')
    if not os.path.exists(vlm_path):
        return None
    try:
        with open(vlm_path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _normalize_grouped_results(data)



def _default_shared_vlm_root(config: DictConfig) -> str:
    override = _get_optional_config_value(config, 'shared_vlm_root', None)
    if override is not None:
        return str(override)
    return os.path.join(str(config.save_root_dir), 'shared_vlm')



def _resolve_grouped_results(
    data: dict,
    config: DictConfig,
    save_dir: str,
    scene_name: str,
    seg_id: str,
    vlm_prompter: Any | None,
) -> tuple[dict | None, str, Any | None]:
    grouped_results = None
    vlm_source = 'vlm'
    use_from_existing = bool(_get_optional_config_value(config, 'from_existing', False))
    use_from_shared_vlm = bool(_get_optional_config_value(config, 'from_shared_vlm', False))
    save_shared_vlm = bool(_get_optional_config_value(config, 'save_shared_vlm', False))
    disable_vlm_calls = bool(_get_optional_config_value(config, 'disable_vlm_calls', False))
    shared_vlm_root = _default_shared_vlm_root(config)

    if bool(_get_optional_config_value(config, 'gt_labels', False)):
        config.segmentation.disable_vlm_judge = True
        grouped_results = {
            'receptor': {
                'name': str(data.get('receptor_name', 'unknown')),
                'description': str(data.get('receptor_name', 'unknown')),
            },
            'effector': {
                'name': str(data.get('effector_name', 'unknown')),
                'description': str(data.get('effector_name', 'unknown')),
            },
        }
        vlm_source = 'gt_labels'
    elif use_from_existing:
        grouped_results = _load_existing_vlm_output(save_dir, scene_name, seg_id)
        if grouped_results is not None:
            vlm_source = 'from_existing'

    if grouped_results is None and use_from_shared_vlm:
        shared_vlm_output = load_shared_vlm_output(shared_vlm_root, scene_name, seg_id)
        grouped_results = _normalize_shared_vlm_results(shared_vlm_output)
        if grouped_results is not None:
            vlm_source = shared_vlm_output.get('source', 'vlm')

    if grouped_results is None:
        if disable_vlm_calls:
            return None, 'disabled', vlm_prompter
        if vlm_prompter is None:
            vlm_prompter = build_vlm_prompter(config.vlm_segmentation)
        grouped_results = _normalize_grouped_results(_prompt_description(vlm_prompter, data['video_path']))
        if grouped_results is None or len(grouped_results.keys()) != 2:
            return None, 'invalid_prompt_result', vlm_prompter

    grouped_results = _normalize_grouped_results(grouped_results)
    if grouped_results is None:
        return None, 'invalid_prompt_result', vlm_prompter

    save_vlm_dir = os.path.join(save_dir, scene_name, seg_id, 'vlm_narrator')
    save_vlm_output(grouped_results, save_vlm_dir)
    if save_shared_vlm:
        shared_output = build_shared_vlm_output(grouped_results, scene_name, seg_id, vlm_source)
        save_shared_vlm_output(shared_output, shared_vlm_root, scene_name, seg_id)
    return grouped_results, vlm_source, vlm_prompter



def _load_existing_segmentation_outputs(
    save_dir: str,
    scene_name: str,
    seg_id: str,
    role: str,
    total_frames: int,
) -> tuple[list[np.ndarray], list[dict], list[int]] | None:
    role_dir = os.path.join(save_dir, scene_name, seg_id, f'segmentation_{role}')
    if not os.path.isdir(role_dir):
        return None
    masks = load_segmentation_mask_archive(role_dir)
    if masks is None or int(masks.shape[0]) != int(total_frames):
        return None
    answers = load_segmentation_answers(role_dir, total_frames=total_frames)
    metrics_path = os.path.join(role_dir, 'segmentation_metrics.json')
    valid_frame_ids = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            if isinstance(metrics, dict) and 'valid_frame_ids' in metrics:
                valid_frame_ids = [int(i) for i in metrics['valid_frame_ids']]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            valid_frame_ids = None
    if valid_frame_ids is None:
        valid_frame_ids = [i for i, mask in enumerate(masks) if np.asarray(mask).astype(bool).sum() > 0]
    return [np.asarray(mask).astype(bool) for mask in masks], answers, valid_frame_ids



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
        answer_dict.setdefault('frame_id', frame_id)
        answer_dict['seed_frame'] = True
        answer_dict['propagated'] = False
        answer_dict['subsampled'] = len(seed_frame_ids) != total_frames
        full_answer_dict_list[frame_id] = answer_dict

    for frame_id in range(total_frames):
        if full_pred_mask_list[frame_id] is None:
            frame = full_frame_list[frame_id]
            full_pred_mask_list[frame_id] = np.zeros(frame.shape[:2], dtype=bool)
            full_answer_dict_list[frame_id] = {
                'frame_id': frame_id,
                'seed_frame': False,
                'propagated': False,
                'subsampled': True,
                'skipped': True,
                'skip_reason': 'frame_subsample',
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
        propagated_masks = propagate_full_video_from_masks(sam3_tracker, full_frame_list, valid_seed_mask_entries)
    else:
        propagated_masks = [np.zeros(frame.shape[:2], dtype=bool) for frame in full_frame_list]

    seed_answer_map = {}
    for local_idx, frame_id in enumerate(seed_frame_ids):
        answer_dict = dict(seed_answer_dicts[local_idx] if isinstance(seed_answer_dicts[local_idx], dict) else {})
        answer_dict.setdefault('frame_id', int(frame_id))
        answer_dict['seed_frame'] = True
        answer_dict['propagated'] = False
        answer_dict['subsampled'] = len(seed_frame_ids) != total_frames
        seed_answer_map[int(frame_id)] = answer_dict

    full_answer_dicts = []
    seed_frame_id_set = set(int(frame_id) for frame_id in seed_frame_ids)
    for frame_id in range(total_frames):
        if frame_id in seed_answer_map:
            answer_dict = dict(seed_answer_map[frame_id])
        else:
            answer_dict = {
                'frame_id': frame_id,
                'seed_frame': False,
                'propagated': True,
                'subsampled': len(seed_frame_ids) != total_frames,
                'propagation_source': 'sam3_mask_prompt_propagation',
            }
        answer_dict['upsampled'] = True
        answer_dict['upsample_method'] = 'sam3_mask_prompt_propagation'
        answer_dict['upsample_seed_frame_ids'] = [int(i) for i in valid_seed_frame_ids]
        answer_dict['is_seed_frame'] = frame_id in seed_frame_id_set
        full_answer_dicts.append(answer_dict)

    valid_frame_ids = [frame_id for frame_id, mask in enumerate(propagated_masks) if int(np.asarray(mask).astype(bool).sum()) > 0]
    return [np.asarray(mask).astype(bool) for mask in propagated_masks], full_answer_dicts, valid_frame_ids



def _sam3_tracker_from_config(config: DictConfig, sam3_tracker):
    propagate_with_sam3 = bool(config.segmentation.get('propagate_with_sam3', False))
    if not propagate_with_sam3:
        return sam3_tracker
    propagation_cfg = config.segmentation.get('sam3_propagation', {})
    if sam3_tracker is None:
        sam3_tracker = build_sam3_tracker(
            device=str(propagation_cfg.get('device', config.segmentation.get('device', 'cuda'))),
            checkpoint_path=propagation_cfg.get('checkpoint_path', None),
            bpe_path=propagation_cfg.get('bpe_path', None),
        )
    return sam3_tracker



def segment_scene(
    data: dict,
    config: DictConfig,
    save_dir: str,
    refseg_model=None,
    vlm_prompter: Any | None = None,
    sam3_tracker=None,
):
    scene_name = str(data['video_name'])
    seg_id = str(data.get('seg_id', '00'))

    grouped_results, _vlm_source, vlm_prompter = _resolve_grouped_results(
        data=data,
        config=config,
        save_dir=save_dir,
        scene_name=scene_name,
        seg_id=seg_id,
        vlm_prompter=vlm_prompter,
    )
    if grouped_results is None:
        return None, refseg_model, vlm_prompter, sam3_tracker
    if bool(_get_optional_config_value(config, 'vlm_only', False)):
        return {'grouped_results': grouped_results, 'roles': {}}, refseg_model, vlm_prompter, sam3_tracker

    if refseg_model is None:
        refseg_model = build_refseg_model(config.segmentation)

    full_video_frame_list = load_full_video_frames(data)
    total_frames = len(full_video_frame_list)
    seed_frame_ids = evenly_spaced_indices(total_frames, config.segmentation.get('frame_subsample', None))
    if len(seed_frame_ids) == 0:
        return None, refseg_model, vlm_prompter, sam3_tracker

    should_propagate = bool(config.segmentation.get('propagate_with_sam3', False)) and len(seed_frame_ids) < total_frames
    sam3_tracker = _sam3_tracker_from_config(config, sam3_tracker) if should_propagate else sam3_tracker
    propagation_cfg = config.segmentation.get('sam3_propagation', {})
    role_results = {}

    for role in ('receptor', 'effector'):
        role_start = time.time()
        role_dir = os.path.join(save_dir, scene_name, seg_id, f'segmentation_{role}')
        existing_outputs = None
        if bool(_get_optional_config_value(config, 'from_existing', False)):
            existing_outputs = _load_existing_segmentation_outputs(save_dir, scene_name, seg_id, role, total_frames)

        used_existing_masks = existing_outputs is not None
        if existing_outputs is not None:
            pred_mask_list, answer_dict_list, valid_frame_ids = existing_outputs
        else:
            part_description = str(grouped_results[role]['description'])
            seed_input_frames = [to_pil_rgb(full_video_frame_list[frame_id]) for frame_id in seed_frame_ids]
            seed_pred_masks, seed_answer_dicts, seed_valid_frame_ids = refseg_model.segment_video(seed_input_frames, part_description)
            if should_propagate:
                pred_mask_list, answer_dict_list, valid_frame_ids = _propagate_seed_masks_to_full_video(
                    full_video_frame_list,
                    seed_frame_ids,
                    seed_pred_masks,
                    seed_answer_dicts,
                    seed_valid_frame_ids,
                    sam3_tracker,
                )
            else:
                pred_mask_list, answer_dict_list, valid_frame_ids = _expand_seed_outputs_to_full_video(
                    full_video_frame_list,
                    seed_frame_ids,
                    seed_pred_masks,
                    seed_answer_dicts,
                    seed_valid_frame_ids,
                )

        gt_mask_list = load_full_role_masks(data, role)
        eval_frame_ids = list(range(total_frames))
        filtered_iou_list, original_iou_list = compute_part_iou_video(
            list(gt_mask_list),
            pred_mask_list,
            valid_frame_ids,
            eval_frame_ids=eval_frame_ids,
        )
        runtime_info = {
            'scene_name': scene_name,
            'seg_id': seg_id,
            'role': role,
            'model': str(config.segmentation.model),
            'frame_subsample': config.segmentation.get('frame_subsample', None),
            'num_total_frames': total_frames,
            'num_sampled_frames': len(seed_frame_ids),
            'sampled_frame_ids': [int(i) for i in seed_frame_ids],
            'subsample_applied': len(seed_frame_ids) != total_frames,
            'runtime_seconds': float(time.time() - role_start),
            'used_existing_masks': used_existing_masks,
            'propagated_with_sam3': should_propagate and not used_existing_masks,
            'propagation_method': 'sam3_mask_prompt_propagation' if should_propagate else None,
            'propagation_device': propagation_cfg.get('device', config.segmentation.get('device', 'cuda')) if should_propagate else None,
        }
        save_segmentation_video(
            full_video_frame_list,
            pred_mask_list,
            answer_dict_list,
            valid_frame_ids,
            original_iou_list,
            filtered_iou_list,
            role_dir,
            save_visualizations=bool(config.segmentation.get('save_visualizations', False)),
        )
        save_segmentation_metrics(
            original_iou_list,
            filtered_iou_list,
            valid_frame_ids,
            role_dir,
            eval_frame_ids=eval_frame_ids,
            runtime_info=runtime_info,
        )
        _save_segmentation_runtime(runtime_info, save_dir, scene_name, seg_id, role)
        role_results[role] = {
            'full_masks': np.stack(pred_mask_list, axis=0).astype(bool),
            'valid_frame_ids': [int(i) for i in valid_frame_ids],
            'role_dir': role_dir,
        }

    return {'grouped_results': grouped_results, 'roles': role_results}, refseg_model, vlm_prompter, sam3_tracker



def evaluate(eval_dataloader: DataLoader, config: DictConfig, save_dir: str):
    if config.debug:
        print('Debug mode enabled: Limiting evaluation dataset to 1 sample.')
        max_dataset_size = 1
    refseg_model = None
    vlm_prompter = None
    sam3_tracker = None
    data_count = 0
    for data in eval_dataloader:
        scene_name = str(data['video_name'])
        print(f'Evaluating data: {scene_name}')
        if config.debug and data_count >= max_dataset_size:
            break
        scene_result, refseg_model, vlm_prompter, sam3_tracker = segment_scene(
            data,
            config,
            save_dir,
            refseg_model=refseg_model,
            vlm_prompter=vlm_prompter,
            sam3_tracker=sam3_tracker,
        )
        if scene_result is None:
            print(f'Warning: no segmentation results for {scene_name}; skipping.')
            continue
        if bool(_get_optional_config_value(config, 'vlm_only', False)):
            print(f'Cached VLM labels only for {scene_name}.')
        data_count += 1


@hydra.main(version_base='1.3', config_path='config', config_name='default')
def main(config: DictConfig):
    print(f'Start experiment: {config.name}')
    save_dir = _get_optional_config_value(config, 'save_dir', None)
    if save_dir is not None:
        print(f'Resuming from: {save_dir}')
    else:
        exp_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        save_dir = f"{config.save_root_dir}/{config.name}/{exp_time}"
        config.save_dir = save_dir
    print(f'Results will be saved to: {save_dir}')
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)

    set_seed(int(config.seed))
    eval_dataset = build_dataset(config.dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=identity_collate)
    evaluate(eval_dataloader, config, save_dir)


if __name__ == '__main__':
    main()
