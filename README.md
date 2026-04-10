<div align="center">
<h1>EgoFun3D: Modeling Interactive Objects from Egocentric Videos using Function Templates</h1>
<a href="https://map-anything.github.io/assets/MapAnything.pdf"><img src="https://img.shields.io/badge/Paper-blue" alt="Paper"></a>
<a href="https://arxiv.org/abs/2509.13414"><img src="https://img.shields.io/badge/arXiv-2509.13414-b31b1b" alt="arXiv"></a>
<a href="https://map-anything.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://x.com/Nik__V__/status/1968316841618518371"><img src="https://img.shields.io/badge/X_Thread-1DA1F2" alt="X Thread"></a>
<a href="https://huggingface.co/spaces/facebook/map-anything"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<br>
<br>
<strong>
<a href="https://willipwk.github.io/">Weikun Peng</a>
&nbsp;&nbsp;
<a href="https://diliash.github.io/">Denys Iliash</a>
&nbsp;&nbsp;
<a href="https://msavva.github.io/">Manolis Savva</a>

Simon Fraser University
</strong>

</div>

![image-0](docs/static/images/teaser.png)



## Installation
1. Clone this repo
   ```bash
   git clone --recursive git@github.com:willipwk/part_function_reconstruction.git
   ```
2. Prepare a conda environment
   ```bash
   cd part_function_reconstruction
   conda create -n test_env python=3.11
   conda activate test_env
   pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
   pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
   pip install transformers==4.57.6
   pip install vllm==0.15.1
   pip install -U flash-attn --no-build-isolation
   pip install accelerate open3d point_cloud_utils qwen_vl_utils molmo_utils hydra-core google-genai openai pycocotools scikit-image
   pip install "git+https://github.com/facebookresearch/sam3.git"
   pip install "git+https://github.com/facebookresearch/sam2.git"

   # prepare environment for reconstruction
   cd third_party/map-anything
   pip install -e .
   cd ../Depth-Anything-3
   pip install xformers==0.0.33.post2
   pip install -e .
   cd ../vipe
   pip install --no-build-isolation -e .
   # if you run into error like "<eigen3/Eigen/Sparse> No such file or directory" when building vipe, install eigen in conda env
   conda install anaconda::eigen==3.4.0
   pip install romatch
   pip install fused-local-corr==0.2.3
   cd ../..

   # prepare environment for artipoint
   cd third_party/artipoint
   pip install lightning gdown ultralytics yacs loguru pycryptodomex gnupg rospkg flow_vis tensorboard imageio[ffmpeg]
   pip install git+https://github.com/ChaoningZhang/MobileSAM.git
   pip install git+https://github.com/facebookresearch/co-tracker.git
   pip install -e .
   mkdir -p checkpoints && cd checkpoints
   # Mobile-SAM
   gdown --fuzzy "https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view?usp=sharing"
   unzip weight.zip
   # CoTracker (offline and/or online)
   wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
   wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth || true
   cd ../..
   ```
3. Download EgoFun3D dataset.
   ```bash
   hf download 3dlg-hcvc/EgoFun3D --repo-type dataset --local-dir full_dataset
   ```
4. Setup environment variable.
   ```bash
   export GEMINI_API_KEY=$YOUR_GEMINI_API_KEY
   export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
   ```

## Run the Benchmark
### Segmentation
The segmentation flow is:
1. Run the selected segmentation model on 20 seed frames.
2. Propagate those masks to all frames with SAM3.
3. Save one mask archive per role in `segmentation_masks.h5`.
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

Results are saved under `outputs/{exp_name}/{time}/{video_name}/segmentation/`.

### Reconstruction
The reconstruction flow is:
1. Load 2D segmentation results from the previous step or from the dataset.
2. Running reconstruction on the input videos.
3. Aligning moving parts to the initial state using RoMa.
4. Build meshes.

To run reconstruction on the ground truth 2D segmentation
```bash
python eval_reconstruction.py reconstruction=da3
```
To run reconstruction on the predicted 2D segmentation
```bash
python eval_reconstruction.py reconstruction=da3 pred_mask=True segmentation_results_dir={YOUR SEGMENTATION RESULTS PATH}
```
You can switch reconstruction method to `mapanything` or `vipe`.

Results are saved under `outputs/{exp_name}/{time}/{video_name}/reconstruction/`.

### Articulation Estimation
The articulation estimation will take the reconstruction results as input and estimate articulation parameters. Thus, please run reconstruction before running articulation estimation.

To run articulation estimation on the ground truth 2D segmentation
```bash
python eval_articulation.py articulation=iTACO reconstruction_results_dir={YOUR RECONSTRUCTION RESULTS PATH}
```
Similarly, to run articulation estimation on the predicted 2D segmentation
```bash
python eval_articulation.py articulation=iTACO reconstruction_results_dir={YOUR RECONSTRUCTION RESULTS PATH} pred_mask=True segmentation_results_dir={YOUR SEGMENTATION RESULTS PATH}
```
You can switch articulation method to `Artipoint`

Results are saved under `outputs/{exp_name}/{time}/{video_name}/articulation/`.

### Function Prediction
The function template prediction first marks the receptor and effector in different colors on the original video and query VLMs for function template prediction.
```bash
python eval_function.py vlm_function=gemini_function
```
You can switch articulation method to `gpt_function`, `qwen_function`, or `molmo_function`.

Results are saved under `outputs/{exp_name}/{time}/{video_name}/function/`.

## Run the Full Pipeline
We also provide a gradio interface to run the full pipeline at once.
```bash
python pipeline.py
```

## Acknowledgment
This work was funded in part by a Canada Research Chair, NSERC Discovery Grant, and enabled by support from the Digital Research Alliance of Canada. The authors would like to thank Tianrun Hu from National University of Singapore for collecting data, Jiayi Liu, Xingguang Yan, Austin T. Wang, and Morteza Badali for valuable discussions and proofreading.

This codebase is built on top of [VisionReasoner](https://github.com/JIA-Lab-research/VisionReasoner), [Sa2VA](https://github.com/bytedance/Sa2VA), [X-SAM](https://github.com/wanghao9610/X-SAM), [Molmo2](https://github.com/allenai/molmo2), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [SAM2](https://github.com/facebookresearch/sam2), [sam3](https://github.com/facebookresearch/sam3), [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Map-Anything](https://github.com/facebookresearch/map-anything), [ViPE](https://github.com/nv-tlabs/vipe), [Artipoint](https://github.com/robot-learning-freiburg/artipoint), [iTACO](https://github.com/3dlg-hcvc/video2articulation). We thank the authors for open sourcing these invaluable projects.

## Citation
If you find our project to be useful, please cite our paper
```bibtex
@article{YourPaperKey2024,
  title={Your Paper Title Here},
  author={First Author and Second Author and Third Author},
  journal={Conference/Journal Name},
  year={2024},
  url={https://your-domain.com/your-project-page}
}
```