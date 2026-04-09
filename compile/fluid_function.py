import numpy as np
import genesis as gs

INSERT_DEFINITIONS_HERE

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt = 4e-3,
        substeps=10,
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sph_options=gs.options.SPHOptions(
        # lower_bound=(-0.5, -0.5, 0.0),
        # upper_bound=(0.5, 0.5, 1),
        particle_size=0.005,
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary=True,
        ambient_light=(0.3, 0.3, 0.3),
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane_light.urdf", fixed=True))

# when loading an entity, you can specify its pose in the morph.
faucet = scene.add_entity(
    gs.morphs.URDF(file=URDF_PATH, fixed=True, convexify=False, coacd_options=gs.options.CoacdOptions(
                        threshold=0.01,
                        max_convex_hull=1600,
                        resolution=2000,
                        preprocess_mode="auto",
    )),
    surface=gs.surfaces.Smooth(roughness=1.0)
)

emitter = scene.add_emitter(
    material=gs.materials.SPH.Liquid(),
    surface=gs.surfaces.Water(vis_mode="recon"),
    max_particles=50000,
)

########################## build ##########################
scene.build()

jnt_names = [
    'receptor_joint',
]
dofs_idx = [faucet.get_joint(name).dof_idx_local for name in jnt_names]
joint = faucet.get_joint('receptor_joint')
joint_limits = joint.dofs_limit
print("Joint limits:", joint_limits)
joint_step = (joint_limits[dofs_idx[0], 1] - joint_limits[dofs_idx[0], 0]) / 1000
faucet.set_dofs_position(np.array([joint_limits[dofs_idx[0], 0]]), dofs_idx)
scene.step()

for i in range(200):
    handle_position = min(joint_limits[dofs_idx[0], 0] + 5 * joint_step * i, joint_limits[dofs_idx[0], 1])
    faucet.set_dofs_position(np.array([handle_position]), dofs_idx)
    droplet_size = MAPPING_FUNCTION
    emitter.emit(
        pos=EMITTER_POSITION,
        direction=np.array([0.0, 0.0, -1.0]),
        speed=5,
        droplet_shape="circle",
        droplet_size=droplet_size,
    )