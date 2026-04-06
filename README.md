# Part Function Modeling from Egocentric Videos

## Installation
1. Install `torch` and `torchvision`. The current setup here uses `torch 2.7.0+cu126`.
2. Install the Gemini and OpenAI Python SDKs and export API keys.
   ```bash
   export GEMINI_API_KEY=$YOUR_GEMINI_API_KEY
   export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
   ```
3. Install the remaining dependencies.
   ```text
   pillow
   open3d
   opencv-python
   transformers==4.48.3
   point_cloud_utils
   qwen_vl_utils
   molmo_utils
   pytorch3d
   sam2
   sam3
   ```
4. Install the third-party modules used by reconstruction and tracking.
   1. Go to `third_party/vipe` and install ViPE. You may need to adjust its `requirements.txt` for your CUDA / torch version.
   2. Go to `third_party/SpaTrackerV2` and run `python setup.py install`.
   3. Go to `third_party/Depth-Anything-3` and install Depth Anything 3.

## Segmentation
The segmentation flow is:
1. Run the selected segmentation model on 20 seed frames.
2. Propagate those masks to all frames with SAM3.
3. Save one mask archive per role in `segmentation_masks.npz`.
4. Use those saved results in the downstream evaluation scripts.

Use `eval_segmentation.py` for the staged release workflow. By default the release segmentation configs use `segmentation.frame_subsample=20` and `segmentation.propagate_with_sam3=true`.

Example:
```bash
python eval_segmentation.py segmentation=VisionReasoner vlm_segmentation=gemini_segmentation
```

If you want to run segmentation with ground-truth part labels instead of VLM-predicted labels:
```bash
python eval_segmentation.py gt_labels=true segmentation=VisionReasoner
```

If you want Gemini part labels precomputed instead of queried during segmentation, first cache the VLM output:
```bash
python eval_segmentation.py \
  vlm_only=true \
  save_shared_vlm=true \
  segmentation=VisionReasoner \
  vlm_segmentation=gemini_segmentation
```

Then run segmentation from the cached labels:
```bash
python eval_segmentation.py \
  from_shared_vlm=true \
  disable_vlm_calls=true \
  segmentation=VisionReasoner \
  vlm_segmentation=gemini_segmentation
```
