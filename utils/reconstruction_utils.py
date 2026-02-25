import time

import open3d as o3d
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import torch
from pytorch3d.ops import utils as oputil
from pytorch3d.ops import knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def estimate_se3_transformation(target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_xyz)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
    correspondences = np.arange(source_xyz.shape[0])
    correspondences = np.vstack([correspondences, correspondences], dtype=np.int32).T
    p2p_registration = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    source2target = p2p_registration.compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(correspondences))
    return source2target


def depth2xyz(depth_image: np.ndarray, intrinsics: np.ndarray, cam_type: str) -> np.ndarray:
    # Get the shape of the depth image
    H, W = depth_image.shape

    # Create meshgrid for pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten the grid to a 1D array of pixel coordinates
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image to a 1D array of depth values
    if depth_image.dtype == np.uint16:
        depth = depth_image.flatten() / 1000.0
    else:
        depth = depth_image.flatten()

    # Camera intrinsic matrix (3x3)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Calculate the 3D coordinates (x, y, z) from depth
    # Use the formula:
    #   X = (u - cx) * depth / fx
    #   Y = (v - cy) * depth / fy
    #   Z = depth
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack the x, y, z values into a 3D point cloud
    point_cloud = np.vstack((x, y, z)).T

    if cam_type == "opengl":
        point_cloud = point_cloud * np.array([1, -1, -1])

    # Reshape the point cloud to the original depth image shape [H, W, 3]
    point_cloud = point_cloud.reshape(H, W, 3)

    return point_cloud


def depth2xyz_world(depth_image: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, cam_type: str) -> np.ndarray:
    point_cloud = depth2xyz(depth_image, intrinsics, cam_type)
    H, W, _ = point_cloud.shape
    point_cloud_flat = point_cloud.reshape(-1, 3)
    point_cloud_homogeneous = np.hstack([point_cloud_flat, np.ones((point_cloud_flat.shape[0], 1))])
    point_cloud_world_homogeneous = (extrinsics @ point_cloud_homogeneous.T).T
    point_cloud_world = point_cloud_world_homogeneous[:, :3] / point_cloud_world_homogeneous[:, 3:]
    return point_cloud_world.reshape(H, W, 3)


def radius_filter_outliers(point_map: np.ndarray, radius: float = 0.01, nb_points: int = 15) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_map.reshape(-1, 3))
    radius_inlier_pcd, radius_inlier_idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    radius_inlier_mask = np.zeros(point_map.shape[:-1], dtype=bool)
    radius_inlier_mask.reshape(-1)[radius_inlier_idx] = True
    return radius_inlier_mask.reshape(point_map.shape[:-1])


def sanitize_points_np(points: np.ndarray) -> np.ndarray:
    # points: (N,3)
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 3
    # drop NaN/Inf
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    # float32 is safest for CUDA kernels
    points = points.astype(np.float32, copy=False)
    # ensure contiguous
    points = np.ascontiguousarray(points)
    return points, finite


# def radius_filter_outliers_gpu(
#     point_map: np.ndarray,
#     radius: float = 0.01,
#     nb_points: int = 15,
#     device: str = "CUDA:0",
#     allow_cpu_fallback: bool = False,
# ) -> np.ndarray:
#     """
#     GPU attempt using Open3D Tensor API.
#     - If your Open3D build supports radius outlier removal on CUDA in t.geometry, this will be fast.
#     - If not supported, you can either raise or fall back to CPU (legacy).
#     """
#     assert point_map.shape[-1] == 3, f"Expected (..., 3), got {point_map.shape}"
#     pts = point_map.reshape(-1, 3)

#     # Optional: drop NaN/Inf
#     # finite = np.isfinite(pts).all(axis=1)
#     # pts_valid = pts[finite].astype(np.float32)
#     pts_valid, finite = sanitize_points_np(pts)

#     # Prepare output mask (invalid points remain False)
#     flat_mask = np.zeros(pts.shape[0], dtype=bool)

#     if pts_valid.shape[0] == 0:
#         return flat_mask.reshape(point_map.shape[:-1])

#     # Choose device
#     dev = o3d.core.Device(device)
#     if dev.get_type() == o3d.core.Device.DeviceType.CUDA and not o3d.core.cuda.is_available():
#         if allow_cpu_fallback:
#             dev = o3d.core.Device("CPU:0")
#         else:
#             raise RuntimeError("CUDA device requested but o3d.core.cuda.is_available() is False.")
#     print(f"Using device: {dev}")

#     # Tensor point cloud on device
#     pcd_t = o3d.t.geometry.PointCloud(dev)
#     pcd_t.point["positions"] = o3d.core.Tensor(pts_valid, o3d.core.float32, device=dev)

#     print("after creating tensor point cloud")

#     # --- Try GPU/tensor radius outlier removal ---
#     # Different Open3D versions expose different method names; try a few common ones.
#     inlier_mask_t = None

#     # Candidate method names seen across versions / docs / builds
#     candidates = [
#         "remove_radius_outliers",
#         "remove_radius_outlier",
#         "radius_outlier_removal",
#     ]

#     last_err = None
#     for name in candidates:
#         fn = getattr(pcd_t, name, None)
#         if fn is None:
#             print(f"Method '{name}' not available on tensor point cloud.")
#             continue
#         try:
#             print(f"Attempting to call '{name}' for radius outlier removal on device {dev}...")
#             o3d.core.cuda.synchronize()
#             print("Before radius outlier removal call")
#             out = fn(nb_points=nb_points, search_radius=radius)
#             print("After radius outlier removal call")
#             o3d.core.cuda.synchronize()
#             print(f"Successfully called '{name}' for radius outlier removal.")
#             # Possible return formats:
#             #   (pcd_filtered, mask)  OR (mask, pcd_filtered) OR just mask
#             if isinstance(out, tuple) and len(out) == 2:
#                 a, b = out
#                 # identify which is mask
#                 if isinstance(a, o3d.core.Tensor):
#                     inlier_mask_t = a
#                     print(f"Identified inlier mask tensor in output of '{name}' as first element.")
#                 elif isinstance(b, o3d.core.Tensor):
#                     inlier_mask_t = b
#                     print(f"Identified inlier mask tensor in output of '{name}' as second element.")
#                 else:
#                     # could be bool numpy/other
#                     print(f"Neither output of '{name}' is a tensor. Output types: {type(a)}, {type(b)}. Cannot identify inlier mask.")
#                     pass
#             elif isinstance(out, o3d.core.Tensor):
#                 inlier_mask_t = out
#                 print(f"Output of '{name}' is a tensor, treating as inlier mask.")
#             break
#         except Exception as e:
#             last_err = e
#     print("after radius outlier removal attempt")
#     if inlier_mask_t is None:
#         if allow_cpu_fallback:
#             print("GPU radius outlier removal not available, falling back to CPU method.")
#             # Fallback to your original CPU method
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(pts_valid.astype(np.float32))
#             _, idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
#             valid_mask = np.zeros(pts_valid.shape[0], dtype=bool)
#             valid_mask[np.array(idx, dtype=np.int32)] = True
#             print("after CPU radius outlier removal")
#         else:
#             raise RuntimeError(
#                 "Your Open3D build does not expose a tensor/CUDA radius-outlier removal API. "
#                 "This operation is CPU-only in many Open3D versions.\n"
#                 f"Last error (if any): {repr(last_err)}"
#             )
#     else:
#         # Bring mask back to CPU numpy
#         valid_mask = inlier_mask_t.to(o3d.core.Device("CPU:0")).numpy().astype(bool).reshape(-1)
#     print("after radius outlier removal and mask retrieval")

#     # Write valid_mask back into full mask (including invalid points)
#     flat_mask[np.where(finite)[0]] = valid_mask

#     del pcd_t  # free GPU memory
#     del inlier_mask_t
#     print("after cleanup")
#     try:
#         o3d.core.cuda.synchronize()
#         o3d.core.cuda.release_cache()
#     except Exception:
#         print("Failed to synchronize GPU")
#     print("after GPU synchronization and cache release")
    
#     return flat_mask.reshape(point_map.shape[:-1])


def _gpu_worker_shm(points_f32, radius, nb_points, shm_name, n, err_q):
    """
    Writes uint8 mask (0/1) into shared memory of length n.
    """
    try:
        dev = o3d.core.Device("CUDA:0")
        pcd_t = o3d.t.geometry.PointCloud(dev)
        pcd_t.point["positions"] = o3d.core.Tensor(points_f32, device=dev)
        print("GPU worker: Created tensor point cloud on CUDA.")
        out = pcd_t.remove_radius_outliers(nb_points=nb_points, search_radius=radius)
        print("GPU worker: Completed radius outlier removal on CUDA.")

        # Force CUDA error to surface inside worker
        o3d.core.cuda.synchronize()
        print("GPU worker: Synchronized CUDA after radius outlier removal.")

        # Unpack mask (Open3D version dependent)
        if isinstance(out, tuple) and len(out) == 2:
            a, b = out
            mask_t = a if isinstance(a, o3d.core.Tensor) else b
        else:
            mask_t = out

        mask = mask_t.cpu().numpy().astype(bool).reshape(-1)  # 0/1
        print("GPU worker: Retrieved mask from CUDA and converted to numpy.")
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = np.ndarray((n,), dtype=bool, buffer=shm.buf)
        buf[:] = mask  # write result
        shm.close()
        print("GPU worker: Wrote mask to shared memory and closed it.")
    except Exception as e:
        # report error to parent
        try:
            err_q.put(repr(e))
        except Exception:
            pass


def remove_radius_outliers_mask_robust_shm(point_map, radius=0.01, nb_points=15, timeout_s=300):
    assert point_map.shape[-1] == 3, f"Expected (..., 3), got {point_map.shape}"
    pts = point_map.reshape(-1, 3)
    # points_np = np.asarray(points_np)
    # assert points_np.ndim == 2 and points_np.shape[1] == 3
    pts_valid, finite = sanitize_points_np(pts)

    # finite = np.isfinite(points_np).all(axis=1)
    pts_valid = np.ascontiguousarray(pts_valid.astype(np.float32))
    n = pts_valid.shape[0]

    flat_mask = np.zeros(pts.shape[0], dtype=bool)

    if n == 0:
        return flat_mask.reshape(point_map.shape[:-1])

    ctx = mp.get_context("spawn")
    err_q = ctx.Queue(maxsize=1)

    # Allocate shared memory for bool mask (0/1)
    shm = shared_memory.SharedMemory(create=True, size=n * np.dtype(bool).itemsize)
    shm_name = shm.name
    shm_buf = np.ndarray((n,), dtype=bool, buffer=shm.buf)
    shm_buf[:] = False

    p = ctx.Process(
        target=_gpu_worker_shm,
        args=(pts_valid, radius, nb_points, shm_name, n, err_q),
    )
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        shm.close()
        shm.unlink()
        raise TimeoutError(f"GPU worker did not finish within {timeout_s}s (likely stuck).")

    # If worker reported an error, do CPU fallback
    err = None
    if not err_q.empty():
        err = err_q.get()

    if p.exitcode != 0 or err is not None:
        print(f"GPU worker failed with exit code {p.exitcode}. Error: {err}. Falling back to CPU method.")
        # CPU fallback in parent (safe)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        _, idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        valid_mask = np.zeros(n, dtype=bool)
        valid_mask[np.asarray(idx, dtype=np.int32)] = True
        print("CPU fallback: Completed radius outlier removal on CPU.")
    else:
        valid_mask = shm_buf.astype(bool)

    # Cleanup shared memory
    shm.close()
    shm.unlink()
    print("Cleaned up shared memory and GPU worker process.")

    flat_mask[np.where(finite)[0]] = valid_mask
    return flat_mask.reshape(point_map.shape[:-1])


# ---------------- Worker ----------------
def _gpu_worker_loop(ctrl_conn):
    import numpy as np
    import open3d as o3d
    from multiprocessing import shared_memory

    dev = o3d.core.Device("CUDA:0")

    while True:
        msg = ctrl_conn.recv()
        if msg["cmd"] == "stop":
            ctrl_conn.send({"ok": True})
            return

        shm_in_name = msg["shm_in"]
        shm_out_name = msg["shm_out"]
        n = int(msg["n"])
        radius = float(msg["radius"])
        nb_points = int(msg["nb_points"])

        shm_in = shm_out = None
        try:
            shm_in = shared_memory.SharedMemory(name=shm_in_name)
            shm_out = shared_memory.SharedMemory(name=shm_out_name)

            pts = np.ndarray((n, 3), dtype=np.float32, buffer=shm_in.buf)
            out_buf = np.ndarray((n,), dtype=bool, buffer=shm_out.buf)

            # Tensor point cloud on CUDA
            pcd_t = o3d.t.geometry.PointCloud(dev)
            pcd_t.point["positions"] = o3d.core.Tensor(pts, device=dev)
            # print("GPU worker: Created tensor point cloud on CUDA.")

            out = pcd_t.remove_radius_outliers(nb_points=nb_points, search_radius=radius)
            # print("GPU worker: Completed radius outlier removal on CUDA.")

            # Force async CUDA errors to surface inside worker
            # o3d.core.cuda.synchronize()
            # print("GPU worker: Synchronized CUDA after radius outlier removal.")
            # Unpack mask tensor (Open3D version dependent)
            if isinstance(out, tuple) and len(out) == 2:
                a, b = out
                mask_t = a if isinstance(a, o3d.core.Tensor) else b
            else:
                mask_t = out

            mask = mask_t.cpu().numpy().astype(bool).reshape(-1)
            out_buf[:] = mask.astype(bool)
            # o3d.core.cuda.synchronize()
            o3d.core.cuda.release_cache()
            # Optional: release cached CUDA memory (you said this helps)
            # del pcd_t, mask_t, out
            # o3d.core.cuda.release_cache()

            ctrl_conn.send({"ok": True})
        except Exception as e:
            # NOTE: if this is cudaErrorIllegalAddress, this worker is likely poisoned.
            # Parent will restart worker.
            ctrl_conn.send({"ok": False, "err": repr(e)})
        finally:
            try:
                if shm_in is not None:
                    shm_in.close()
            except Exception:
                pass
            try:
                if shm_out is not None:
                    shm_out.close()
            except Exception:
                pass


# ---------------- Client wrapper ----------------
class Open3DRadiusOutlierGPUWorker:
    """
    Persistent GPU worker with CPU fallback and auto-restart-on-GPU-failure.
    """

    def __init__(self, device: str = "CUDA:0"):
        self.device = device
        self.ctx = mp.get_context("spawn")
        self._conn = None
        self._proc = None
        self._start_worker()

    def _start_worker(self):
        parent_conn, child_conn = self.ctx.Pipe()
        self._conn = parent_conn
        self._proc = self.ctx.Process(target=_gpu_worker_loop, args=(child_conn,), daemon=True)
        self._proc.start()

    def _stop_worker(self):
        if self._proc is None:
            return
        try:
            if self._conn is not None:
                self._conn.send({"cmd": "stop"})
                if self._conn.poll(1.0):
                    _ = self._conn.recv()
        except Exception:
            pass
        try:
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join()
        except Exception:
            pass
        self._proc = None
        self._conn = None

    def restart_worker(self):
        self._stop_worker()
        self._start_worker()

    def close(self):
        self._stop_worker()

    @staticmethod
    def _cpu_fallback_mask(pts_valid_f32: np.ndarray, radius: float, nb_points: int) -> np.ndarray:
        """
        CPU remove_radius_outlier on valid points only. Returns bool mask of shape (N_valid,).
        """
        print("Running CPU fallback for radius outlier removal...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_valid_f32.astype(np.float32, copy=False))
        _, idx = pcd.remove_radius_outlier(nb_points=int(nb_points), radius=float(radius))
        mask = np.zeros(pts_valid_f32.shape[0], dtype=bool)
        mask[np.asarray(idx, dtype=np.int32)] = True
        print("CPU fallback: Completed radius outlier removal on CPU.")
        return mask

    def run(
        self,
        point_map: np.ndarray,
        radius: float = 0.01,
        nb_points: int = 15,
        timeout_s: float = 60.0,
        fallback_to_cpu: bool = True,
        restart_on_gpu_fail: bool = True,
    ) -> np.ndarray:
        """
        Returns a boolean mask with shape (N,) for input points_np (N,3).
        Any NaN/Inf points are marked False.
        """
        assert point_map.shape[-1] == 3, f"Expected (..., 3), got {point_map.shape}"
        pts = point_map.reshape(-1, 3)
        # points_np = np.asarray(points_np)
        # assert points_np.ndim == 2 and points_np.shape[1] == 3
        pts_valid, finite = sanitize_points_np(pts)

        # finite = np.isfinite(points_np).all(axis=1)
        pts_valid = np.ascontiguousarray(pts_valid.astype(np.float32))
        n = pts_valid.shape[0]

        flat_mask = np.zeros(pts.shape[0], dtype=bool)

        if n == 0:
            return flat_mask.reshape(point_map.shape[:-1])
        # points_np = np.asarray(points_np)
        # assert points_np.ndim == 2 and points_np.shape[1] == 3

        # finite = np.isfinite(points_np).all(axis=1)
        # pts_valid = np.ascontiguousarray(points_np[finite].astype(np.float32))
        # n = pts_valid.shape[0]

        # full_mask = np.zeros(points_np.shape[0], dtype=bool)
        # if n == 0:
        #     return full_mask

        # Allocate shared memory for input and output
        shm_in = shared_memory.SharedMemory(create=True, size=pts_valid.nbytes)
        shm_out = shared_memory.SharedMemory(create=True, size=n * np.dtype(bool).itemsize)

        gpu_ok = False
        gpu_err = None

        try:
            in_buf = np.ndarray((n, 3), dtype=np.float32, buffer=shm_in.buf)
            out_buf = np.ndarray((n,), dtype=bool, buffer=shm_out.buf)
            in_buf[:] = pts_valid
            out_buf[:] = False

            # Send GPU job
            self._conn.send({
                "cmd": "run",
                "shm_in": shm_in.name,
                "shm_out": shm_out.name,
                "n": n,
                "radius": float(radius),
                "nb_points": int(nb_points),
            })

            # Wait for response with timeout
            t0 = time.time()
            while True:
                if self._conn.poll(0.01):
                    try:
                        resp = self._conn.recv()
                    except EOFError:
                        gpu_ok = False
                        gpu_err = "EOFError: GPU worker process ended unexpectedly."
                        break
                    except (BrokenPipeError, OSError) as e:
                        gpu_ok = False
                        gpu_err = f"Pipe connection error: {repr(e)}"
                        break
                    gpu_ok = bool(resp.get("ok", False))
                    gpu_err = resp.get("err")
                    break
                if (time.time() - t0) > timeout_s:
                    gpu_ok = False
                    gpu_err = f"Timeout after {timeout_s}s"
                    break

            if gpu_ok:
                valid_mask = out_buf.astype(bool)
                flat_mask[np.where(finite)[0]] = valid_mask
                return flat_mask.reshape(point_map.shape[:-1])

            # GPU failed or timed out
            if restart_on_gpu_fail:
                # Worker may be poisoned; restart it
                print(f"GPU worker failed with error: {gpu_err}. Restarting worker and falling back to CPU method.")
                self.restart_worker()

            if not fallback_to_cpu:
                raise RuntimeError(f"GPU remove_radius_outliers failed: {gpu_err}")

            # CPU fallback in parent
            valid_mask = self._cpu_fallback_mask(pts_valid, radius=radius, nb_points=nb_points)
            flat_mask[np.where(finite)[0]] = valid_mask
            return flat_mask.reshape(point_map.shape[:-1])

        finally:
            # Clean up shared memory
            try:
                shm_in.close(); shm_in.unlink()
            except Exception:
                pass
            try:
                shm_out.close(); shm_out.unlink()
            except Exception:
                pass
        

def pytorch3d_remove_outlier(point_map: np.ndarray, radius: float = 0.01, nb_points: int = 15) -> np.ndarray:
    pcd = Pointclouds(torch.from_numpy(point_map.reshape(-1, 3)).cuda().to(torch.float32)[None,...])
    nn_dists, nn_idx, nn = knn_points(oputil.convert_pointclouds_to_tensor(pcd)[0],
                                    oputil.convert_pointclouds_to_tensor(pcd)[0],
                                    K=nb_points)

    # threshold = 0.1 # you could estimate this based on assuming a Gaussian over K-nn distances as in PCL
    radius_inlier_mask = nn_dists[0,:,1:].mean(1) < radius
    # pcd_filtered = Pointclouds(pcd[nn_dists[0,:,1:].mean(1) < radius][None,...])
    return radius_inlier_mask.reshape(point_map.shape[:-1]).cpu().numpy().astype(bool)


def refine_point_mask(reconstruction_results: dict) -> dict:
    if "points" not in reconstruction_results.keys():
        full_points_list = []
        for i in range(len(reconstruction_results["depth"])):
            depth_i = reconstruction_results["depth"][i]
            intrinsics = reconstruction_results["intrinsics"]
            extrinsics_i = reconstruction_results["extrinsics"][i]
            base_pc_origin = depth2xyz_world(depth_i, intrinsics, extrinsics_i, cam_type="opencv")
            full_points_list.append(base_pc_origin)
        reconstruction_results["points"] = np.stack(full_points_list, axis=0)
    else:
        full_points_list = reconstruction_results["points"]
    full_points_mask_list = reconstruction_results["points_mask"]
    refined_points_mask_list = []
    # worker = Open3DRadiusOutlierGPUWorker()
    # try:
    for frame_id in range(len(full_points_list)):
        print(f"Refining frame {frame_id} with radius outlier removal...")
        points = full_points_list[frame_id]
        mask = full_points_mask_list[frame_id]
        # radius_inlier_mask = remove_radius_outliers_mask_robust_shm(points, radius=0.01, nb_points=15)
        # radius_inlier_mask = worker.run(points, radius=0.01, nb_points=15, timeout_s=60.0, fallback_to_cpu=True, restart_on_gpu_fail=True)
        # radius_inlier_mask = radius_filter_outliers(points, radius=0.01, nb_points=15)
        radius_inlier_mask = pytorch3d_remove_outlier(points, radius=0.01, nb_points=15)
        refined_mask = np.logical_and(mask, radius_inlier_mask)
        refined_points_mask_list.append(refined_mask)
    reconstruction_results["points_mask"] = np.stack(refined_points_mask_list, axis=0)
    # finally:
    #     worker.close()
    return reconstruction_results