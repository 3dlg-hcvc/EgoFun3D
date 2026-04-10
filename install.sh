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