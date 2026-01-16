# Part Function Modeling from Egocentric Videos

## Installation
1. install torch and torchvision. Here I use torch 2.7.0+cu126
2. install Gemini and GPT python API. Export API key 
   ```bash
   export GEMINI_API_KEY=$YOUR_GEMINI_API_KEY
   export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
   ```
3. install other dependency
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
   ```
4. install third party modules
   1. go to `third_party/vipe` to install ViPE, you may need to change its `requirements.txt` according to your torch and cuda version.
   2. go to `third_party/SpaTrackerV2` to install spatrackerv2 package. run `python setup.py install`
   3. go to `thid_party/Depth-Anything-3` to install depth anything 3.