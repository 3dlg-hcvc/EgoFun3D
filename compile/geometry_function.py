# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.replicator.core as rep

import numpy as np
import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend
import os


INSERT_DEFINITIONS_HERE

OBJECT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            INITIALIZE_JOINT
        },
    ),
    # actuators={"door_acts": ImplicitActuatorCfg(joint_names_expr=["effector_joint"], damping=5, stiffness=20)},
    actuators=ACTUATORS_CONFIG,
)


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    sim_utils.create_prim("/World/Origin_00", "Xform")
    sim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=1080,
        width=1920,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Object = OBJECT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Object")

    camera = CameraCfg(
        prim_path="/World/CameraSensor",
        update_period=0,
        height=1080,
        width=1920,
        data_types=[
            "rgb",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24, focus_distance=400, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    camera: Camera = scene["camera"]
    camera_index = 0  # since we have multiple cameras, we need to specify which camera's data we want to save

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[-0.003964 + 1.06, 1.75081 + 0.61, 0.258466 + 1.2]], device=sim.device)
    camera_targets = torch.tensor([[-1.042195 + 1.06, -0.594510 + 0.61, -0.580865 + 1.2]], device=sim.device)

    camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_object_state = scene["Object"].data.default_root_state.clone()
            root_object_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Object"].write_root_pose_to_sim(root_object_state[:, :7])
            scene["Object"].write_root_velocity_to_sim(root_object_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Object"].data.default_joint_pos.clone(),
                scene["Object"].data.default_joint_vel.clone(),
            )
            scene["Object"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Object state...")

        joint_state = scene["Object"].data.joint_pos
        if len(joint_state[0]) > 1: # set receptor target state
            if count % 500 < 250:
                receptor_target_state = RECEPTOR_STATE_MAX
            else:
                receptor_target_state = RECEPTOR_STATE_MIN
        effector_target_state = MAPPING_FUNCTION
        # if joint_state[0][0] > 0.015:
        #     scene["Object"].set_joint_position_target(torch.Tensor([[np.pi / 2, receiver_target]]))
        if len(joint_state[0]) > 1:
            scene["Object"].set_joint_position_target(torch.Tensor([[effector_target_state, receptor_target_state]]))
        else:
            scene["Object"].set_joint_position_target(torch.Tensor([[effector_target_state]]))

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        camera.update(dt=sim.get_physics_dt())

        # Save images from camera at camera_index
        # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )

        # Extract the other information
        single_cam_info = camera.data.info[camera_index]

        # Pack data back into replicator format to save them using its writer
        rep_output = {"annotators": {}}
        for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
            if info is not None:
                rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
            else:
                rep_output["annotators"][key] = {"render_product": {"data": data}}
        # Save images
        # Note: We need to provide On-time data for Replicator to save the images.
        rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
        rep_writer.write(rep_output)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
