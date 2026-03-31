from copy import deepcopy
import PIL
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
import cv2
import torch
import torch.nn.functional as F
from kornia.feature import LoFTR
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix
import omegaconf
from tqdm import tqdm
import yaml

from articulation.base import ArticulationEstimation
from utils.reconstruction_utils import estimate_se3_transformation, depth2xyz_world
from typing import List, Dict, Tuple


class iTACOCoarse:
    def __init__(self, matcher: str = "loftr", device: str = "cuda"):
        self.matcher = LoFTR(pretrained="indoor").to(device)

    def filter_match(self, kp1: np.ndarray, kp2: np.ndarray, thresh: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        dist = np.linalg.norm((kp1 - kp2), axis=1)
        kp1 = kp1[dist < thresh]
        kp2 = kp2[dist < thresh]
        return kp1, kp2

    def compute_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1_torch = torch.Tensor(img1_raw).cuda() / 255.
        img1_torch = torch.reshape(img1_torch, (1, 1, img1_torch.shape[0], img1_torch.shape[1]))
        # img2_raw = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2_raw = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img2_torch = torch.Tensor(img2_raw).cuda() / 255.
        img2_torch = torch.reshape(img2_torch, (1, 1, img2_torch.shape[0], img2_torch.shape[1]))

        input = {"image0": img1_torch, "image1": img2_torch}
        correspondences_dict = self.matcher(input)

        mkpts0 = correspondences_dict['keypoints0'].cpu().numpy()
        mkpts1 = correspondences_dict['keypoints1'].cpu().numpy()
        mconf = correspondences_dict['confidence'].cpu().numpy()
        torch.cuda.empty_cache()
        return mkpts0, mkpts1, mconf

    def estimate_joint_transformation(self, base_kp: np.ndarray, curr_kp: np.ndarray, type: str, RANSAC: bool) -> Tuple[np.ndarray, np.ndarray]:
        curr2base = None
        inlier = None
        if RANSAC:
            k = 50
            inlier_thresh = 1e-2
            d = int(base_kp.shape[0] * 0.4)
            best_se3 = None
            best_error = 10000
            best_inlier = None
            inlier_list = []
            se3_list = []
            inlier_index_list = []
            for _ in range(k):
                init_sample = np.random.choice(base_kp.shape[0], 10, replace=False)
                init_kp1 = base_kp[init_sample]
                init_kp2 = curr_kp[init_sample]
                
                if type == "revolute":
                    se3 = estimate_se3_transformation(init_kp1, init_kp2)
                elif type == "prismatic":
                    only_translation = np.mean((init_kp1 - init_kp2), axis=0)
                    se3 = np.eye(4)
                    se3[:3, 3] = only_translation
                se3_list.append(se3)
                rotation = se3[:3, :3]
                translation = se3[:3, 3]

                transform_kp2 = curr_kp @ rotation.T + translation
                dist = np.linalg.norm((base_kp - transform_kp2), axis=1)
                inlier = np.nonzero(dist < inlier_thresh)[0]
                inlier_list.append(inlier.shape[0])
                inlier_index_list.append(inlier)
                if inlier.shape[0] > d:
                    if type == "revolute":
                        se3 = estimate_se3_transformation(base_kp[inlier], curr_kp[inlier])
                    elif type == "prismatic":
                        only_translation = np.mean((base_kp[inlier] - curr_kp[inlier]), axis=0)
                        se3 = np.eye(4)
                        se3[:3, 3] = only_translation
                    se3_list[-1] = se3
                    rotation = se3[:3, :3]
                    translation = se3[:3, 3]

                    transform_inlier_kp2 = curr_kp[inlier] @ rotation.T + translation
                    this_error = np.mean((base_kp[inlier] - transform_inlier_kp2) ** 2)
                    if this_error < best_error:
                        best_se3 = se3
                        best_error = this_error
                        best_inlier = inlier
            if best_se3 is None:
                print("RANSAC fail!")
                max_inlier_index = inlier_list.index(max(inlier_list))
                best_se3 = se3_list[max_inlier_index]
                best_inlier = inlier_index_list[max_inlier_index]
            else:
                print("RANSAC success!")
            curr2base = best_se3
            inlier = best_inlier
        else:
            if type == "revolute":
                curr2base = estimate_se3_transformation(base_kp, curr_kp)
            elif type == "prismatic":
                only_translation = np.mean((base_kp - curr_kp), axis=0)
                curr2base = np.eye(4)
                curr2base[:3, 3] = only_translation
            inlier = np.arange(base_kp.shape[0])
        return curr2base, inlier

    def estimate_joint_single(self, base_kp: np.ndarray, curr_kp: np.ndarray, RANSAC: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        curr2base, revolute_inlier = self.estimate_joint_transformation(base_kp, curr_kp, "revolute", RANSAC)
        rotation = curr2base[:3, :3]
        translation = curr2base[:3, 3]

        result = {}
        joint_rotvec = R.from_matrix(rotation.T).as_rotvec()
        revolute_joint_axis = joint_rotvec / np.linalg.norm(joint_rotvec)
        det = np.linalg.det(np.eye(3) - rotation)
        valid = True
        try:
            revolute_joint_pos = np.linalg.inv(np.eye(3) - rotation) @ translation
            revolute_joint_pos = revolute_joint_pos - np.dot(revolute_joint_pos, revolute_joint_axis) * revolute_joint_axis
        except:
            det = 0
            revolute_joint_pos = np.zeros(3)
            valid = False
        if abs(det) < 1e-17:
            print("angle too small!")
            valid = False
        rotate_curr_kp = (curr_kp[revolute_inlier] - revolute_joint_pos) @ rotation.T + revolute_joint_pos
        rotation_error = np.mean((base_kp[revolute_inlier] - rotate_curr_kp) ** 2)
        result["revolute"] = {"X": curr_kp[revolute_inlier], "Y": base_kp[revolute_inlier], "axis": revolute_joint_axis, "pos": revolute_joint_pos, "error": rotation_error, "det": det,  "valid": valid}

        only_translation_se3, prismatic_inlier = self.estimate_joint_transformation(base_kp, curr_kp, "prismatic", RANSAC)
        only_translation = only_translation_se3[:3, 3]
        prismatic_joint_axis = only_translation / np.linalg.norm(only_translation)
        prismatic_joint_pos = base_kp[0]
        translate_curr_kp = curr_kp[prismatic_inlier] + only_translation
        translation_error = np.mean((base_kp[prismatic_inlier] - translate_curr_kp) ** 2)
        result["prismatic"] = {"X": curr_kp[prismatic_inlier], "Y": base_kp[prismatic_inlier], "axis": prismatic_joint_axis, "pos": prismatic_joint_pos, "error": translation_error, "valid": True}

        return result

    def estimate_joint_all(self, result_list: List[Dict[str, Dict[str, np.ndarray]]]) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
        revolute_error = 0
        prismatic_error = 0
        revolute_joint_axis = 0
        prismatic_joint_axis = 0
        revolute_joint_pos = 0
        revolute_count = 0
        revolute_max_det = -1
        revolute_max_det_index = -1
        joint_type_vote = 0
        for index, result in enumerate(result_list):
            if result["revolute"]["valid"]:
                revolute_error += result["revolute"]["error"]
                revolute_joint_axis += result["revolute"]["axis"]
                revolute_joint_pos += result["revolute"]["pos"]
                revolute_count += 1
            if result["revolute"]["det"] > revolute_max_det:
                revolute_max_det_index = index
                revolute_max_det = result["revolute"]["det"]

            prismatic_error += result["prismatic"]["error"]
            prismatic_joint_axis += result["prismatic"]["axis"]
            if result["revolute"]["valid"]:
                if result["revolute"]["error"] < result["prismatic"]["error"]:
                    joint_type_vote += 1
                else:
                    joint_type_vote -= 1
        if joint_type_vote > 0:
            pred_joint_type = "revolute"
        elif joint_type_vote < 0:
            pred_joint_type = "prismatic"
        else:
            if revolute_count == 0:
                revolute_error = result_list[revolute_max_det_index]["revolute"]["error"]
            else:
                revolute_error = revolute_error / revolute_count
            prismatic_error = prismatic_error / len(result_list)
            pred_joint_type = "revolute" if revolute_error < prismatic_error else "prismatic"
        if revolute_count == 0:
            revolute_joint_axis = result_list[revolute_max_det_index]["revolute"]["axis"]
            revolute_joint_pos = result_list[revolute_max_det_index]["revolute"]["pos"]
        else:
            revolute_joint_axis = revolute_joint_axis / np.linalg.norm(revolute_joint_axis)
            revolute_joint_pos = revolute_joint_pos / revolute_count
        prismatic_joint_axis = prismatic_joint_axis / np.linalg.norm(prismatic_joint_axis)
        prismatic_joint_pos = np.zeros(3)
        pred_joint_metrics = {"revolute": {"axis": revolute_joint_axis, "pos": revolute_joint_pos}, 
                              "prismatic": {"axis": prismatic_joint_axis, "pos": prismatic_joint_pos},}
        
        return pred_joint_metrics, pred_joint_type

    def compute_average_rotation_angle(self, X: np.ndarray, Y: np.ndarray, joint_axis: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        # Normalize the rotation axis
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        
        # Initialize sums for sine and cosine components
        sin_sum = 0
        cos_sum = 0
        
        for x, y in zip(X, Y):
            x = x - joint_pos
            y = y - joint_pos
            # Project points onto the plane perpendicular to the axis
            x_perp = x - np.dot(x, joint_axis) * joint_axis
            y_perp = y - np.dot(y, joint_axis) * joint_axis
            
            # Normalize the projected vectors
            x_perp = x_perp / np.linalg.norm(x_perp)
            y_perp = y_perp / np.linalg.norm(y_perp)
            
            # Compute cosine and sine for this pair
            cos_theta = np.dot(x_perp, y_perp)
            sin_theta = np.dot(joint_axis, np.cross(x_perp, y_perp))
            
            # Accumulate sine and cosine
            cos_sum += cos_theta
            sin_sum += sin_theta
        
        # Compute the average angle
        angle = np.arctan2(sin_sum, cos_sum)
        return angle

    def compute_average_translation_distance(self, X: np.ndarray, Y: np.ndarray, joint_axis: np.ndarray) -> np.ndarray:
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        translation_vec = Y - X
        translation_dist = np.mean(np.dot(translation_vec, joint_axis))
        return translation_dist

    def estimate_joint(self, rgb_frame_list: List[np.ndarray], reconstruction_results: Dict, part_masks: np.ndarray) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
        result_list = []
        pair_list = []
        for interval in [1, 2, 3]:
            for i in range(0, len(rgb_frame_list) - interval, 1):
                try:
                    mkpts0, mkpts1, conf = self.compute_match(rgb_frame_list[i], rgb_frame_list[i + interval])
                except Exception as e:
                    print("Matcher fail between frame {} and frame {}! Error message: {}".format(i, i + interval, e))
                    continue
                match_mask = conf > 0.9
                mkpts0 = mkpts0[match_mask].astype(np.uint32)
                mkpts1 = mkpts1[match_mask].astype(np.uint32)

                part_mask_i = part_masks[i]
                dynamic_index = np.nonzero(part_mask_i[mkpts0[:, 1], mkpts0[:, 0]])[0]
                dynamic_pts0 = mkpts0[dynamic_index]
                dynamic_pts1 = mkpts1[dynamic_index]

                if "points" in reconstruction_results.keys():
                    base_pc_origin = reconstruction_results["points"][i]
                    next_pc_origin = reconstruction_results["points"][min(i + interval, len(reconstruction_results["points"]) - 1)]
                else:
                    depth_i = reconstruction_results["depth"][i]
                    depth_next = reconstruction_results["depth"][min(i + interval, len(reconstruction_results["depth"]) - 1)]
                    intrinsics = reconstruction_results["intrinsics"]
                    extrinsics_i = reconstruction_results["extrinsics"][i]
                    extrinsics_next = reconstruction_results["extrinsics"][min(i + interval, len(reconstruction_results["extrinsics"]) - 1)]
                    base_pc_origin = depth2xyz_world(depth_i, intrinsics, extrinsics_i, cam_type="opencv")
                    next_pc_origin = depth2xyz_world(depth_next, intrinsics, extrinsics_next, cam_type="opencv")

                dynamic_kp0 = base_pc_origin[dynamic_pts0[:, 1], dynamic_pts0[:, 0]]
                dynamic_kp1 = next_pc_origin[dynamic_pts1[:, 1], dynamic_pts1[:, 0]]
                dynamic_kp0, dynamic_kp1 = self.filter_match(dynamic_kp0, dynamic_kp1)

                if dynamic_kp0.shape[0] > 10:
                    result_i = self.estimate_joint_single(dynamic_kp0, dynamic_kp1, RANSAC=True)
                    result_list.append(result_i)
                    pair_list.append((i, i + interval))
        if len(result_list) > 0:
            pred_joint_metrics, pred_joint_type = self.estimate_joint_all(result_list)
            for joint_type in pred_joint_metrics.keys():
                joint_value_per_frame = 0
                for pair, result in zip(pair_list, result_list):
                    if joint_type == "revolute":
                        angle = self.compute_average_rotation_angle(result[joint_type]["X"], result[joint_type]["Y"], pred_joint_metrics[joint_type]["axis"], pred_joint_metrics[joint_type]["pos"])
                        angle_per_frame = angle / (pair[1] - pair[0])
                        joint_value_per_frame += angle_per_frame
                    elif joint_type == "prismatic":
                        distance = self.compute_average_translation_distance(result[joint_type]["X"], result[joint_type]["Y"], pred_joint_metrics[joint_type]["axis"])
                        distance_per_frame = distance / (pair[1] - pair[0])
                        joint_value_per_frame += distance_per_frame
                joint_value_per_frame = joint_value_per_frame / (len(result_list))
                pred_joint_metrics[joint_type]["average_value"] = joint_value_per_frame
        else:
            pred_joint_metrics = None
            pred_joint_type = None
        self.prediction_joint_metrics = pred_joint_metrics
        self.prediction_joint_type = pred_joint_type
        return pred_joint_metrics, pred_joint_type


class iTACORefine:
    def __init__(self, lr: float, opt_steps: int, device: torch.device):
        self.device = device
        self.opt_steps = opt_steps
        self.current_step = 0
        self.lr = lr

    def distances_to_line(self, points: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray, return_min: bool = False) -> float:
        """
        Compute perpendicular distances from a set of 3‑D points to a 3‑D line.

        Parameters
        ----------
        points : (N, 3) array_like
            Coordinates of the N points.
        line_point : (3,) array_like
            A point on the line (the vector **a** in the formula).
        line_dir : (3,) array_like
            Direction vector of the line (the vector **v**).  Need not be unit‑length.
        return_min : bool, default False
            If True, also return the index of the closest point and its distance.

        Returns
        -------
        dists : (N,) ndarray
            Perpendicular distances for every point.
        (optional) min_idx, min_dist : int, float
            Index of the closest point and its distance (only if return_min=True).
        """
        p = np.asarray(points, dtype=float)          # (N, 3)
        a = np.asarray(line_point, dtype=float)      # (3,)
        v = np.asarray(line_dir, dtype=float)        # (3,)

        # Vector from line point to every point
        r = p - a                                    # (N, 3)

        # Norm of cross‑product gives numerator; norm of v gives denominator
        cross = np.cross(r, v)                       # (N, 3)
        num = np.linalg.norm(cross, axis=1)          # (N,)
        denom = np.linalg.norm(v)                    # scalar
        dists = num / denom

        min_dist = np.min(dists)
        if return_min:
            min_idx = np.argmin(dists)
            return dists, min_idx, dists[min_idx]
        return min_dist

    def chamfer_loss(self, xyz: torch.Tensor, joint_axis: torch.Tensor, joint_pos: torch.Tensor, joint_state: torch.Tensor, joint_type: str, moving_map: torch.Tensor, surface_xyz: torch.Tensor) -> torch.Tensor:
        N, H, W, C = xyz.shape
        # dynamic chamfer distance
        joint_axis_norm = F.normalize(joint_axis.reshape(1, 3))
        if joint_type == "revolute":
            rot_vec = joint_axis_norm.repeat(N, 1) * joint_state.reshape(N, 1) # N, 3
            rotations = axis_angle_to_matrix(rot_vec) # N, 3, 3
            translations = torch.matmul((torch.eye(3, dtype=torch.float32).repeat(N, 1, 1).to(self.device) - rotations), joint_pos) # N, 3
        elif joint_type == "prismatic":
            rotations = torch.eye(3, dtype=torch.float32, device=self.device).repeat(N, 1, 1) # N, 3, 3
            translations = joint_axis_norm.repeat(N, 1) * joint_state.reshape(N, 1) # N, 3

        joint_transformed_xyz = torch.matmul(xyz.reshape(N, H*W, C), rotations.permute(0, 2, 1)) + translations.reshape(N, 1, 3) # N, H*W, 3
        dynamic_chamfer_loss = torch.zeros(N, device=self.device)
        for b in range(joint_transformed_xyz.shape[0]):
            filter_joint_transformed_xyz = joint_transformed_xyz[b, moving_map[b].reshape(H*W), :] # compute_mask_num, 3
            random_sample_num = filter_joint_transformed_xyz.shape[0] // 2
            random_sample_index = torch.randint(0, max(filter_joint_transformed_xyz.shape[0], 1), (random_sample_num,), device=self.device)
            random_filter_joint_transformed_xyz = filter_joint_transformed_xyz[random_sample_index, :] # random_sample_num, 3
            dynamic_chamfer_dist_b, _ = chamfer_distance(random_filter_joint_transformed_xyz[None, ...], surface_xyz[None, ...], 
                                                            batch_reduction=None, point_reduction=None, single_directional=True, ) # 1, mask_num
            weighted_dynamic_chamfer_dist_b = dynamic_chamfer_dist_b.mean() # * random_filter_norm_moving_map.mean()
            if torch.isnan(weighted_dynamic_chamfer_dist_b):
                dynamic_chamfer_loss[b] = 0
            else:
                dynamic_chamfer_loss[b] = weighted_dynamic_chamfer_dist_b

        return dynamic_chamfer_loss

    def optimize_joint(self, rgb_frame_list: List[np.ndarray], reconstruction_results: Dict, part_masks: np.ndarray, coarse_prediction_results: Dict, joint_type: str):
        ## surface point cloud
        if "points" in reconstruction_results.keys():
            surface_xyz = torch.from_numpy(reconstruction_results["points"]).to(self.device).to(torch.float32) # H * W * 3
        else:
            depth = reconstruction_results["depth"][0]
            intrinsics = reconstruction_results["intrinsics"]
            extrinsics = reconstruction_results["extrinsics"][0]
            surface_xyz_np = depth2xyz_world(depth, intrinsics, extrinsics, cam_type="opencv")
            surface_xyz = torch.from_numpy(surface_xyz_np).to(self.device).to(torch.float32) # H * W * 3
        sample_num = 10000
        random_sample_index = torch.randperm(surface_xyz.shape[0] * surface_xyz.shape[1])[:sample_num]
        surface_xyz = surface_xyz.view(-1, 3)[random_sample_index]

        ## video pcd
        xyz_list = []
        for i in range(len(rgb_frame_list)):
            if "points" in reconstruction_results.keys():
                frame_xyz = reconstruction_results["points"][i]
            else:
                depth = reconstruction_results["depth"][i]
                intrinsics = reconstruction_results["intrinsics"]
                extrinsics = reconstruction_results["extrinsics"][i]
                frame_xyz = depth2xyz_world(depth, intrinsics, extrinsics, cam_type="opencv")
            xyz_list.append(frame_xyz)
        xyz = torch.from_numpy(np.stack(xyz_list)).to(self.device).to(torch.float32) # N, H, W, 3
        N, H, W, C = xyz.shape

        ## moving map
        moving_map = torch.from_numpy(part_masks).to(self.device) # N, H, W, bool
        
        ## joint axis, joint position, and joint state
        estimation_results = {"revolute": {"best_loss": None, "axis": None, "pos": None, "state": None}, 
                              "prismatic": {"best_loss": None, "axis": None, "pos": None, "state": None}}
        for joint_type in ["revolute", "prismatic"]:
            joint_axis = torch.nn.Parameter(torch.from_numpy(coarse_prediction_results[joint_type]["axis"]).to(self.device).to(torch.float32), requires_grad=True) # 3,
            joint_pos = torch.nn.Parameter(torch.from_numpy(coarse_prediction_results[joint_type]["pos"]).to(self.device).to(torch.float32), requires_grad=True) # 3,
            joint_state = np.arange(N) * coarse_prediction_results[joint_type]["average_value"]
            joint_state = torch.nn.Parameter(torch.from_numpy(joint_state).to(self.device).to(torch.float32), requires_grad=True) # N,
            if torch.any(torch.isnan(joint_state)):
                del joint_state
                if joint_type == "revolute":
                    joint_state = torch.nn.Parameter(torch.linspace(0, torch.pi / 2, N, device=self.device, dtype=torch.float32), requires_grad=True) # N,
                elif joint_type == "prismatic":
                    joint_state = torch.nn.Parameter(torch.linspace(0, 0.1, N, device=self.device, dtype=torch.float32), requires_grad=True) # N,
            
            ## optimizer
            optimize_params = [{"params": (joint_axis, joint_pos, joint_state)}]
            optimizer = torch.optim.Adam(optimize_params, lr=self.lr)

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.opt_steps)
            best_joint_axis = joint_axis.detach().cpu().numpy()
            best_joint_pos = joint_pos.detach().cpu().numpy()
            best_joint_state = joint_state.detach().cpu().numpy()
            best_loss = 100
            estimation_results[joint_type]["best_loss"] = best_loss
            estimation_results[joint_type]["axis"] = best_joint_axis
            estimation_results[joint_type]["pos"] = best_joint_pos
            estimation_results[joint_type]["state"] = best_joint_state

            tbar = tqdm(range(self.opt_steps))
            for i, _ in enumerate(tbar):
                self.current_step = i

                # start_time = time.time()
                optimizer.zero_grad()
                
                chamfer_loss = self.chamfer_loss(xyz, joint_axis, joint_pos, joint_state, joint_type, moving_map, surface_xyz)
                loss = torch.mean(chamfer_loss)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                # end_time = time.time()

                eval_loss = loss.detach()
                if eval_loss.detach().cpu().item() < best_loss:
                    best_joint_axis = joint_axis.detach().cpu().numpy()
                    best_joint_pos = joint_pos.detach().cpu().numpy()
                    best_joint_state = joint_state.detach().cpu().numpy()
                    
                    best_loss = eval_loss.detach().cpu().item()

                tbar.set_description("Loss: {:.6f}".format(loss.item()))
            estimation_results[joint_type]["best_loss"] = best_loss
            estimation_results[joint_type]["axis"] = best_joint_axis
            estimation_results[joint_type]["pos"] = best_joint_pos
            estimation_results[joint_type]["state"] = best_joint_state

            del joint_axis
            del joint_pos
            del joint_state
            torch.cuda.empty_cache()
            
        min_dist = self.distances_to_line(surface_xyz.cpu().numpy(), estimation_results["revolute"]["pos"], estimation_results["revolute"]["axis"])
        if min_dist < 0.15:
            pred_joint_type = "revolute" if estimation_results["revolute"]["best_loss"] < estimation_results["prismatic"]["best_loss"] else "prismatic"
        else:
            pred_joint_type = "prismatic"
        
        return {"axis": estimation_results[pred_joint_type]["axis"], 
                "origin": estimation_results[pred_joint_type]["pos"], 
                "state": estimation_results[pred_joint_type]["state"], 
                "type": pred_joint_type, 
                "loss": estimation_results[pred_joint_type]["best_loss"]}


class iTACO(ArticulationEstimation):
    def __init__(self, config: omegaconf.DictConfig):
        self.coarse_prediction = iTACOCoarse(config.coarse.matcher, config.device)
        self.refinement = iTACORefine(config.refine.lr, config.refine.opt_steps, config.device)
        self.sample_strategy = config.sample_strategy
        self.sample_num = config.sample_num

    def articulation_estimation(self, rgb_frame_list: List[np.ndarray], reconstruction_results: Dict, part_masks: np.ndarray) -> Dict[str, np.ndarray]:
        sample_rgb_frame_list = deepcopy(rgb_frame_list)
        sample_reconstruction_results = deepcopy(reconstruction_results)
        sample_part_masks = deepcopy(part_masks)
        if self.sample_strategy == "fix_num":
            total_frames = len(rgb_frame_list)
            if total_frames > self.sample_num:
                sample_interval = total_frames // self.sample_num
                sample_rgb_frame_list = sample_rgb_frame_list[::sample_interval]
                sample_reconstruction_results["depth"] = sample_reconstruction_results["depth"][::sample_interval]
                sample_reconstruction_results["extrinsics"] = sample_reconstruction_results["extrinsics"][::sample_interval]
                if "points" in sample_reconstruction_results.keys():
                    sample_reconstruction_results["points"] = sample_reconstruction_results["points"][::sample_interval]
                sample_part_masks = sample_part_masks[::sample_interval]
        elif self.sample_strategy == "fix_step":
            sample_rgb_frame_list = sample_rgb_frame_list[::self.sample_num]
            sample_reconstruction_results["depth"] = sample_reconstruction_results["depth"][::self.sample_num]
            sample_reconstruction_results["extrinsics"] = sample_reconstruction_results["extrinsics"][::self.sample_num]
            if "points" in sample_reconstruction_results.keys():
                sample_reconstruction_results["points"] = sample_reconstruction_results["points"][::self.sample_num]
            sample_part_masks = sample_part_masks[::self.sample_num]
        coarse_prediction_results, coarse_predicted_joint_type = self.coarse_prediction.estimate_joint(sample_rgb_frame_list, sample_reconstruction_results, sample_part_masks)
        if coarse_prediction_results is not None:
            refine_prediction_results = self.refinement.optimize_joint(sample_rgb_frame_list, sample_reconstruction_results, sample_part_masks, coarse_prediction_results, coarse_predicted_joint_type)
            return refine_prediction_results
        else:
            return None

    