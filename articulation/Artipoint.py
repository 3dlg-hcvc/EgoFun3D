import os
import json
import numpy as np
from PIL import Image as PILImage
import cv2
import torch
import loguru
import time
from pathlib import Path
import omegaconf
import gtsam

from ripl_articulation import solve_articulation_from_poses
from third_party.artipoint.artipoint.factor_graph.pose_est import PoseEstFactorGraph
from third_party.artipoint.artipoint.track.arti_estimator import ArtiEstimator
from third_party.artipoint.artipoint.segmentor.articulated_object_segmentor import ArticulatedObjectSegmentor
from third_party.artipoint.artipoint.track.arti_estimator import smooth_trajectory_optimization
from third_party.artipoint.artipoint.utils.articulation_helper import estimate_motion_point_bisectors

from articulation.base import ArticulationEstimation
from typing import List, Tuple, Dict


class Artipoint(ArticulationEstimation):
    def __init__(self, config: omegaconf.DictConfig):
        self.cfg = config
        config_segmentor = {
            "hand_model_repo": self.cfg.segmentation.hand_model_repo,
            "hand_model_name": self.cfg.segmentation.hand_model_name,
            "hand_model_checkpoint_path": self.cfg.segmentation.hand_model_checkpoint_path,
            "sam_checkpoint": self.cfg.segmentation.sam_checkpoint,
            "use_cuda": True if self.cfg.device == "cuda" else False,
            "hand_resize": tuple(self.cfg.segmentation.hand_resize),
            "arti4d_dataset": None,
        }
        self.arti_segmentor = ArticulatedObjectSegmentor(config_segmentor)
        motion_cfg = {
            "model_path": self.cfg.tracking.model_path,
            "cotracker2": self.cfg.tracking.cotracker2,
            "yolo_path": self.cfg.tracking.yolo_path,
            "arti4d_dataset": None,
            "device": "cuda" if self.cfg.device else "cpu",
        }
        self.arti_estimator = ArtiEstimator(motion_cfg)

    def extract_hand_segments(self, rgb_frame_list: List[PILImage.Image]) -> List[Tuple[int, int]]:
        """Extract hand action segments from RGB frames."""
        segments = self.arti_segmentor.extract_hand_action_segments_smoothed(
            rgb_frame_list,
            smoothing_window_size=self.cfg.segmentation.smoothing_window_size,
            min_window_size=self.cfg.segmentation.min_window_size,
            max_window_size=self.cfg.segmentation.max_window_size,
        )
        loguru.logger.info(f"Found {len(segments)} hand action segments.")
        return segments

    def compute_human_masks(
        self, rgb_frame_list: List[PILImage.Image], segments: List[Tuple[int, int]]
    ) -> List[List[np.ndarray]]:
        """Compute human masks for each segment."""
        masks_per_segment = []
        for seg_start, seg_end in segments:
            segment_masks = []
            for i in range(seg_end - seg_start):
                img = np.array(rgb_frame_list[i + seg_start])
                human_masks = self.arti_estimator._segment_human(img)
                if human_masks:
                    # Use first detected human mask and process it
                    mask = cv2.dilate(
                        human_masks[0],
                        np.ones(
                            (
                                self.cfg.segmentation.dilate_kernel_size,
                                self.cfg.segmentation.dilate_kernel_size,
                            ),
                            np.uint8,
                        ),
                        iterations=self.cfg.segmentation.dilate_iterations,
                    )
                    mask = cv2.resize(
                        mask,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    mask = np.zeros((img.shape[0], img.shape[1]))
                segment_masks.append(mask)
            masks_per_segment.append(segment_masks)
        # free GPU memory
        torch.cuda.empty_cache()
        return masks_per_segment

    def extract_queries_per_segment(
        self, rgb_frame_list: List[PILImage.Image], part_masks: np.ndarray, segments: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[torch.Tensor]]:
        """Extract query points for each segment using articulated segmentation."""
        queries_segments = []
        valid_segments = []
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            if self.cfg.queries.manual_queries:
                points = self.arti_estimator._select_points(np.array(rgb_frame_list[seg_start]))
                queries = self.arti_estimator._create_queries_points(
                    points, frames=[0], device="cpu"
                )
                queries_segments.append(queries)
                valid_segments.append((seg_start, seg_end))
                continue
            queries = None
            for j in range(0, seg_end - seg_start, self.cfg.queries.frame_step):
                part_mask = part_masks[j + seg_start]
                if part_mask.sum() == 0:
                    continue
                orb_points = self.arti_segmentor.sample_fetures(
                    np.array(rgb_frame_list[j + seg_start]), [part_mask], self.cfg.queries.max_feat_points, feat_type=self.cfg.queries.feat_type
                )
                # rgb = rgb_frame_list[j + seg_start]
                # depth = depths[j + seg_start]
                # camera_pose = camera_poses[j + seg_start]
                # arti_results = self.arti_segmentor.segment_articulated_object(
                #     rgb,
                #     depth,
                #     camera_pose=camera_pose,
                #     num_points=self.cfg.queries.num_points,
                #     dist_thresh=self.cfg.queries.dist_thresh,
                #     max_feat_points=self.cfg.queries.max_feat_points,
                #     eps_pixels=self.cfg.queries.eps_pixels,
                #     feat_type=self.cfg.queries.feat_type,
                # )
                # if not arti_results.get("orb_points"):
                #     continue
                query = self.arti_estimator._create_queries_points(
                    orb_points, frames=[j], device="cpu"
                )
                queries = (
                    query if queries is None else torch.cat((queries, query), dim=0)
                )
                queries = queries[: self.cfg.queries.max_queries]
                # # Visualize results
                # if self.cfg.visualization.show_segmented_image:
                #     masks = arti_results.get("filtered_masks")
                #     rgb_vis = self.arti_segmentor.sam_segmenter.visualize_segmentation(
                #         rgb, masks
                #     )
                #     all_points = arti_results.get("orb_points", [])
                #     if all_points:
                #         rgb_vis = self.arti_segmentor.sam_segmenter.draw_points(
                #             rgb_vis, all_points, [1] * len(all_points)
                #         )
                #     cv2.imshow(
                #         "Segmented Image", cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
                #     )
                #     cv2.waitKey(1)

            cv2.destroyAllWindows()
            if queries is not None:
                queries_segments.append(queries)
                valid_segments.append((seg_start, seg_end))
                loguru.logger.info(f"Queries for segment {seg_idx}: {queries.shape}")
            else:
                loguru.logger.warning(f"No queries found for segment {seg_idx}")

        return valid_segments, queries_segments

    def track_and_project_queries(
        self,
        rgb_frame_list: List[PILImage.Image],
        depths: np.ndarray,
        intrinsics: np.ndarray,
        segments: List[Tuple[int, int]],
        queries_segments: List[torch.Tensor],
        human_masks_per_segment: List[List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        For each segment:
          - Track queries using the CoTracker.
          - Project 2D tracks to 3D using depth information.
          - Apply human mask and valid depth filtering.
        """
        # save 2D tracks the results:
        # save_dir = "2d_tracks"
        # os.makedirs(save_dir, exist_ok=True)
        # vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
        rgb_frame_list_np = [np.array(frame) for frame in rgb_frame_list]

        pred_3d_tracks_segments = []
        pred_visibility_segments = []
        for idx, (seg_start, seg_end) in enumerate(segments):
            st = time.time()
            pred_2d_tracks, pred_visibility = self.arti_estimator._cotracker_process(
                rgb_frame_list_np[seg_start:seg_end],
                queries_segments[idx].to("cuda"),
                backward_tracking=self.cfg.tracking.backward_tracking,
            )
            loguru.logger.info(
                f"Tracking for segment {idx} with {seg_end - seg_start} frames took {time.time() - st} seconds."
            )
            # window_frames = self.rgb_frames[seg_start:seg_end]
            # video = (
            #     torch.tensor(np.stack(window_frames))
            #     .permute(0, 3, 1, 2)[None]
            #     .float()
            #     .to("cuda")
            # )
            # vis.visualize(
            #     video,
            #     pred_2d_tracks,
            #     pred_visibility,
            #     filename=f"segment_{idx}.mp4",
            # )
            pred_2d_tracks = pred_2d_tracks.cpu().numpy().squeeze()
            pred_visibility = pred_visibility.cpu().numpy().squeeze()
            torch.cuda.empty_cache()
            pred_3d_tracks, valid_depth_masks = (
                self.arti_estimator._project_2d_tracks_to_3d(
                    depths[seg_start:seg_end], intrinsics, pred_2d_tracks
                )
            )
            # Filter out trackers falling on human regions
            human_masks_stack = np.stack(human_masks_per_segment[idx]).astype(bool)
            x_loc = np.clip(
                pred_2d_tracks[:, :, 0], 0, human_masks_stack.shape[2] - 1
            ).astype(int)
            y_loc = np.clip(
                pred_2d_tracks[:, :, 1], 0, human_masks_stack.shape[1] - 1
            ).astype(int)
            pred_visibility &= ~human_masks_stack[
                np.arange(human_masks_stack.shape[0])[:, None], y_loc, x_loc
            ]
            # Combine with valid depth masks
            pred_visibility &= valid_depth_masks
            pred_3d_tracks_segments.append(pred_3d_tracks)
            pred_visibility_segments.append(pred_visibility)
        loguru.logger.info(
            f"Processed tracking for {len(pred_3d_tracks_segments)} segments."
        )
        return pred_3d_tracks_segments, pred_visibility_segments

    def compensate_cam_motion(
        self,
        camera_poses: np.ndarray,
        segments: List[Tuple[int, int]],
        pred_3d_tracks_segments: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Compensate for camera motion by transform all point tracks to the global frame.
        """
        for i, (start_idx, end_idx) in enumerate(segments):
            for j in range(len(pred_3d_tracks_segments[i])):
                pred_3d_tracks_segments[i][j] = (
                    camera_poses[start_idx + j][:3, :3]
                    @ pred_3d_tracks_segments[i][j].T
                    + camera_poses[start_idx + j][:3, 3][:, None]
                ).T
        return pred_3d_tracks_segments

    def filter(
        self,
        pred_3d_tracks_segments: List[np.ndarray],
        pred_visibility_segments: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        filter out static points and jerky points
        """
        for i in range(len(pred_3d_tracks_segments)):
            pred_3d_tracks_segments[i], pred_visibility_segments[i] = (
                self.arti_estimator.filter_non_smooth_tracks(
                    pred_3d_tracks_segments[i],
                    pred_visibility_segments[i],
                    thrsh=self.cfg.filtering.smoothness_threshold,
                )
            )
            pred_3d_tracks_segments[i], pred_visibility_segments[i] = (
                self.arti_estimator.median_filter(
                    pred_3d_tracks_segments[i],
                    pred_visibility_segments[i],
                    variance_type=self.cfg.filtering.variance_type,
                    percentile=self.cfg.filtering.percentile,
                )
            )
        return pred_3d_tracks_segments, pred_visibility_segments

    def estimate_motion_for_segments(
        self,
        pred_3d_tracks_segments: List[np.ndarray],
        pred_visibility_segments: List[np.ndarray],
        segments: List[Tuple[int, int]],
    ):
        """
        Estimate motion for each segment.
        """
        segments_results = []
        traj_segments = []
        free_traj_segments = []
        axis_segments = []
        motion_center_segments = []
        successful_segments = []

        assert len(pred_3d_tracks_segments) == len(segments)

        for i in range(len(pred_3d_tracks_segments)):
            res = {}
            pairs = self.arti_estimator.create_points_pairs(
                pred_3d_tracks_segments[i], pred_visibility_segments[i]
            )
            # Estimate motion with factor graph
            traj, free_traj, results = self.estimate_motion(pairs, config=self.cfg)
            if results:  # Only add if estimation was successful

                # Run RIPL FG (Buchanan et al. https://github.com/ripl-lab/ripl_articulation)
                w_T_a = [free_traj[0]] * len(free_traj)
                w_T_b = free_traj
                w_T_a = np.array(w_T_a)
                w_T_b = np.array(w_T_b)
                xi, thetas = solve_articulation_from_poses(
                    w_T_a, w_T_b, prior_theta=0.0, prior_xi=None
                )
                w, v = xi[:3], xi[3:]
                w_norm = np.linalg.norm(w)
                v_norm = np.linalg.norm(v)
                w_axis = xi[:3] / w_norm
                v_axis = xi[3:] / v_norm
                joint_type = "revolute" if w_norm > v_norm else "prismatic"
                center = (
                    free_traj[0][:3, 3].tolist()
                    if joint_type == "prismatic"
                    else estimate_motion_point_bisectors(free_traj, w_axis).tolist()
                )
                res["twist"] = xi.tolist()
                res["thetas"] = [float(theta) for theta in thetas]
                res["axis"] = (
                    w_axis.tolist() if joint_type == "revolute" else v_axis.tolist()
                )
                res["joint_type"] = joint_type
                res["pitch"] = 0
                res["center"] = center
                res["start_frame"] = segments[i][0]
                res["end_frame"] = segments[i][1]

                segments_results.append(res)
                axis = res["axis"]
                motion_center = res["center"]
                traj_segments.append(traj)
                free_traj_segments.append(free_traj)
                axis_segments.append(axis)
                motion_center_segments.append(motion_center)
                successful_segments.append(
                    i
                )  # Store the index of the successful segment
            else:
                loguru.logger.warning(f"Motion estimation failed for segment {i}")

        return (
            traj_segments,
            free_traj_segments,
            axis_segments,
            motion_center_segments,
            segments_results,
            successful_segments,
        )

    def filter_unreliable_tracks(
        self,
        pred_3d_tracks_segments: List[np.ndarray],
        pred_visibility_segments: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Filter out unreliable tracks based on visibility and other criteria.
        """
        filtered_tracks_segments = []
        filtered_visibility_segments = []
        for i in range(len(pred_3d_tracks_segments)):
            tracks, visibility = self.arti_estimator.filter_occluded_tracks(
                pred_3d_tracks_segments[i],
                pred_visibility_segments[i],
                occlusion_threshold=self.cfg.filtering.occlusion_threshold,
            )
            filtered_tracks_segments.append(tracks)
            filtered_visibility_segments.append(visibility)
        return filtered_tracks_segments, filtered_visibility_segments

    def filter_outlier_tracks(
        self,
        pred_3d_tracks_segments: List[np.ndarray],
        pred_visibility_segments: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Filter out outlier tracks based on clustering.
        """
        filtered_tracks_segments = []
        filtered_visibility_segments = []
        for i in range(len(pred_3d_tracks_segments)):
            tracks, visibility = self.arti_estimator.filter_outlier_tracks(
                pred_3d_tracks_segments[i],
                pred_visibility_segments[i],
                eps=self.cfg.filtering.eps,
                min_samples=self.cfg.filtering.min_samples,
            )
            filtered_tracks_segments.append(tracks)
            filtered_visibility_segments.append(visibility)
        return filtered_tracks_segments, filtered_visibility_segments

    def write_results(self, save_path, segments, results, free_traj_segments=None):
        """
        Write the results to a JSON file.
        """
        for i, res in enumerate(results):
            seg_path = os.path.join(save_path, f"segment_{i}")
            os.makedirs(seg_path, exist_ok=True)
            # Write axis info
            with open(os.path.join(seg_path, "axis_info.json"), "w") as f:
                json.dump(res, f, indent=4)
            # Write metadata
            with open(os.path.join(seg_path, "metadata.json"), "w") as f:
                metadata = {
                    "start_frame": results[i]["start_frame"],
                    "end_frame": results[i]["end_frame"],
                    "id": i,
                    "scene_name": Path(self.cfg.dataset.root_path).stem,
                }
                json.dump(metadata, f, indent=4)
            # Write free trajectory if available
            if free_traj_segments is not None:
                poses = np.array(free_traj_segments[i])
                np.save(os.path.join(seg_path, "poses.npy"), poses)
        loguru.logger.info(f"Results saved to {save_path}")

    def estimate_motion(self, pairs, world_frame=None, config=None):
        """
        Estimate motion from pairs of 3D points using a factor graph.

        Args:
            pairs: List of pairs of 3D points. (T, P, 2, 3)
            world_frame: Optional world frame transformation matrix.
            config: Configuration parameters.

        Returns:
            Tuple (trajectory, free_trajectory, results)
        """
        # Create factor graph to solve for the transformations
        graph = PoseEstFactorGraph(len(pairs))
        for i in range(len(pairs)):
            graph.add_point_pair_factor(pairs[i], i)
            graph.add_articulation_factor(
                i, theta_prior=config.motion_estimation.theta_prior
            )
            graph.add_world_frame_factor(i)

        # Solve
        if graph.solve_factor_graph():
            transformations = graph.get_optimized_transformations()
            twist = graph.get_optimized_twist()
            T_w = graph.get_optimized_world_frame()
            thetas = graph.get_optimized_thetas()
            Ts = []
            for i, T in enumerate(transformations):
                Ts.append(T.matrix())

            results = graph.estimate_joint_type_and_parameters()
            loguru.logger.info(f"Estimated w norm: {np.linalg.norm(twist[:3])}")
            loguru.logger.info(f"Estimated v norm: {np.linalg.norm(twist[3:])}")
            loguru.logger.info(f"Estimated pitch: {results['pitch']}")

            motion_center = results["q"] if "q" in results else results["center"]
            loguru.logger.info(f"Joint type: {results['type']}")

            traj = [T_w]
            free_traj = [T_w]
            for i in range(len(Ts)):
                T = gtsam.Pose3.Expmap(twist * thetas[i]).matrix()
                traj.append(T @ traj[-1])
                free_traj.append(Ts[i] @ free_traj[-1])

            # using the world frame transform all trajectories to the same frame
            if world_frame is not None:
                traj = [world_frame @ T for T in traj]
                free_traj = [world_frame @ T for T in free_traj]

            if world_frame is not None:
                results["center"] = (
                    world_frame[:3, :3] @ motion_center + world_frame[:3, 3]
                )

            return traj, free_traj, results
        else:
            loguru.logger.error("Factor graph optimization failed.")
            return None, None, None

    def articulation_estimation(self, rgb_frame_list: List[PILImage.Image], reconstruction_results: Dict, part_masks: np.ndarray) -> Dict[str, np.ndarray | str]:
        segments = self.extract_hand_segments(rgb_frame_list)

        # Extract query points per segment
        segments, queries_segments = self.extract_queries_per_segment(rgb_frame_list, part_masks, segments)
        # Compute human masks per segment
        human_masks_per_segment = self.compute_human_masks(rgb_frame_list, segments)

        pred_3d_tracks_segments, pred_visibility_segments = (
            self.track_and_project_queries(
                rgb_frame_list, reconstruction_results["depth"], reconstruction_results["intrinsics"], segments, queries_segments, human_masks_per_segment
            )
        )

        # Compensate for camera motion
        pred_3d_tracks_segments = self.compensate_cam_motion(
            reconstruction_results["extrinsics"], segments, pred_3d_tracks_segments
        )

        # Filter out static and jerky points
        pred_3d_tracks_segments, pred_visibility_segments = self.filter(
            pred_3d_tracks_segments, pred_visibility_segments
        )

        # Filter out unreliable tracks
        pred_3d_tracks_segments, pred_visibility_segments = (
            self.filter_unreliable_tracks(
                pred_3d_tracks_segments, pred_visibility_segments
            )
        )

        # Filter out outlier tracks
        pred_3d_tracks_segments, pred_visibility_segments = (
            self.filter_outlier_tracks(
                pred_3d_tracks_segments, pred_visibility_segments
            )
        )

        # Smooth out the tracks
        pred_3d_tracks_segments_smooth = []
        for i in range(len(pred_3d_tracks_segments)):
            track = smooth_trajectory_optimization(
                pred_3d_tracks_segments[i],
                pred_visibility_segments[i],
                lambda_vel=self.cfg.filtering.lambda_vel,
                lambda_jerk=self.cfg.filtering.lambda_jerk,
            )
            pred_3d_tracks_segments_smooth.append(track)
        pred_3d_tracks_segments = pred_3d_tracks_segments_smooth

        # Estimate motion for each segment
        (
            traj_segments,
            free_traj_segments,
            axis_segments,
            motion_center_segments,
            segments_results,
            successful_segments,
        ) = self.estimate_motion_for_segments(
            pred_3d_tracks_segments, pred_visibility_segments, segments
        )

        results_list = [{"axis": np.array(results["axis"]), "origin": np.array(results["center"]), "state": np.array(results["thetas"]), "type": results["joint_type"]} for results in segments_results]

        return results_list[0]
        