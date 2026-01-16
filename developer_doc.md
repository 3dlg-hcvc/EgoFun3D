# Developer Document
This document defines the interface of different modules in this repo.

## Repo Structure
This repo contains different modules on loading datasets, VLM, reconstruction, etc. The file structure is like this:

```
root
  |--config
  |     |--default.yaml # define configuration and parameters for different modules
  |--dataset
  |     |--dataset.py # define how to load dataset
  |--fusion
  |     |--evaluate_reconstruction.py # evaluation script for reconstruction of parts
  |     |--fusion.py # fuse reconstructed point cloud at different steps together
  |     |--reconstruction.py # reconstruct object point cloud from video
  |--segmentation
  |     |--evaluate_segmentation.py # evaluation script of 2D part segmentation
  |     |--prompt_vlm.py # define different vlm models and how to prompt them
  |     |--ref_seg.py # given either text description or point description, segment the part in the video
  |--utils
  |     |--reconstruction_utils.py # tool functions for reconstruction
  |     |--vlm_utils.py # tool functions for vlm prompt
  |--main.py # entrypoint
```

## Dataset
Currently I have implemented two datasets: OpenFunGraph and iPhone dataset. All datasets should load into the same format.

- [x] OpenFunGraph dataset
- [x] iPhone dataset (data collected by ourselves)
- [ ] Ego-Exo4D dataset

### input:
* root_path (str): path to the root folder of the dataset
* meta_file_path (str): path to the dataset meta file

### output data sample format:
```python
{
    "func_type": func_type, # str, physics function type like geometry, illumination, etc.
    "scene_name": scene_name, # str, scene name
    "seg_id": seg_id, # str, video clip id in the scene
    "ego_video_path": ego_video_path, # str, mp4 file to the video clip
    "ego_video_rgb_list": rgb_list, # List[PIL.Image], list of video frames in PIL format
    "ego_video_rgb_path_list": rgb_path_list, # List[str], list of video frame paths
    "ego_video_depth_list": depth_list, # List[numpy.ndarray], list of depth maps
    "ego_video_camera_list": camera_list, # List[Dict[str, numpy.ndarray]], list of camera parameters, {"intrinsics": 3x3 intrinsic matrix, "extrinsics": 4x4 extrinsic matrix}
    "gt_receiver_mask_list": gt_receiver_mask_list, # List[numpy.ndarray] 2D masks for receiver
    "gt_effector_mask_list": gt_effector_mask_list, # List[numpy.ndarray] 2D masks for effector
    "gt_pcd_annotation": gt_pcd_annotation # Dict[str, str | Dict[str, str, numpy.ndarray]], point cloud annotation {"receiver": {"part_name": name, "part_pcd": point cloud}, "effector": {"part_name": name, "part_pcd": point cloud}, "relation": language description of relation}
}
```

## VLM
I plan to use VLM to analize videos. Currently VLM needs to generate part description. Potentially I will use VLM to generate point description, and choose part function.

- [x] Gemini
- [x] GPT
- [x] Molmo

### input
* video_path (str): path to mp4 video

### text description output
* part description (Dict[str, Dict[str, str]]): a dict storing description of receiver and effector in the following format
```python
{
    "receiver": {"name": name, "description": description},
    "effector": {"name": name, "description": description}
}
```

### point description output (WIP)
* part description (Dict[str, List[Tuple[int, int, int]]]): a dict storing point coordinates of receiver and effector throughout the whole video. The coordinate is stored in this formate `(frame_id, x, y)`

## Referring Segmentation
Referring segmentation is doing segmentation based on some forms of promts. Here I have implemented text-based referring segmentation. I will continue add point-based referring segmentation.

- [x] VisionReasoner (text)
- [x] Sa2VA (text)
- [ ] X-SAM (text)
- [ ] UniPixel (text)
- [ ] SAM3 (text/point)

### text prompt input
* video_frame_list (List[PIL.Image] | List[str]): list of video frames in PIL format or list of video frame paths
* part description (str): text description of interested part

### output
* mask_list (List[numpy.ndarray]): list of 2D masks 
* answer_dict_list (List[Dict[str, str]]): list of text answer, as some models will output text answers
* valid_frame_ids (List[int]): list of frame indices that have valid results. Currently there is a VLM judge to decide whether the segmentation is correct or not. If VLM judge decides a mask is not accurate, we will discard results of this frame. (May discard VLM judge for benchmarking)

## 4D Reconstruction
Reconstructing point cloud given the video. **This part no part mask is applied.** I have implemented Naive reconstruction (knowing everything), MoGe (only depth is unknown), Spatracker, ViPE, and Depth-Anything 3.

- [x] Naive (knowing everything)
- [x] MoGe-2 (only depth is unknown)
- [x] Spatracker
- [x] ViPE
- [x] Depth-Anything 3
- [ ] Map-Anything
- [ ] Any4D

### input
* video_frame_list (List[PIL.Image] | str): list of video frames in PIL format. For ViPE, the input is the part to video frames folder.
* init_extrinsics (numpy.ndarray): in order to compare against ground truth point cloud, the extrinsics of the first frame is provided to align all predicted camera poses and point cloud to the same coordinate.
* intrinsics (numpy.ndarray): video intrinsic matrix. **Optional**
* cam_pose_list (List[numpy.ndarray]): list of camera pose matrices for each frame. **Optional**
* depth_frame_list (List[numpy.ndarray]): list of depth maps. **Optional**

### output
Return reconstruction results in a dict in this format:
```python
{
    "rgb": video_frame_list, # List[PIL.Image], list of video frames in PIL format
    "intrinsics": intrinsics, # numpy.ndarray, 3x3 video intrinsic matrix
    "extrinsics": extrinsics, # numpy.ndarray, Nx4x4 extrinsic matrices, N is the number of frames
    "depth": depth, # numpy.ndarray, NxHxW depth maps, N is the number of frames
    "points": point_map, # numpy.ndarray, NxHxWx3 point maps, N is the number of frames
    "points_mask": point_mask # numpy.ndarray, NxHxW point masks, indicating which points are reliable
}
```

## Fusion
After reconstruct point maps of the video, we need to fuse point cloud at different frames together. Since the object is moving in the video, the key idea of fusion is to find out the part transformation between different states. I have implemented two strategies: feature matching based method and tracking based method. Feature matching method is built on Roma. Tracking method is built on Spatracker. There might be better strategies I haven't come up with.

- [x] feature matching
- [x] points tracking

### input
* video_frame_list (List[PIL.Image | str]): list of video frames in PIL format or path to video frames
* part_mask_list (List[numpy.ndarray]): list of 2D part masks
* points_map_list (List[numpy.ndarray]): list of 3D point maps
* tracks3d_list (List[numpy.ndarray]): list of 3D tracking points positions per frame (Only for tracking based method)
* tracks2d_list (List[numpy.ndarray]): list of 2D tracking pixels coordinates per frame (Only for tracking based method)

### output
* fused_part_pcd (numpy.ndarray): part point cloud
* transformation_list (List[numpy.ndarray]): list of part rigid transformation matrics

## Third Party Module
I put all third party module under `third_party/` folder and add them in the `.gitmodules`. Please do so for other third party repos.
