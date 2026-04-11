import json
import yaml
import os
import numpy as np
import open3d as o3d
import argparse
from build_urdf import generate_urdf_from_open3d_meshes


dir_abs_path = os.path.dirname(os.path.abspath(__file__))

PHYSICAL_EFFECT_MAP = {"a": "geometry", "b": "illumination", "c": "temperature", "d": "fluid"}
NUMERICAL_FUNCTION_MAP = {"a": "binary", "b": "step", "c": "linear", "d": "cumulative"}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compile function instance from reconstruction and prediction results.")
    parser.add_argument("--reconstruction_dir", type=str, required=True, help="Path to the reconstruction results folder.")
    parser.add_argument("--articulation_dir", type=str, required=True, help="Path to the articulation estimation results folder.")
    parser.add_argument("--function_dir", type=str, required=True, help="Path to the function prediction results folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for URDF and mesh outputs.")
    parser.add_argument("--robot_name", type=str, default="articulated_object", help="Robot name embedded in the URDF <robot> tag.")
    # Delta is only required for cumulative mapping, but we accept it as an optional argument here and will raise an error if it's missing when needed.
    parser.add_argument("--delta", type=float, default=None, help="Delta value for cumulative mapping (required only when mapping type is 'cumulative').")
    # emitter_position is only relevant for fluid effects, but we accept it as an optional argument here and will compute a default OBB-based estimate if it's missing when needed.
    parser.add_argument("--emitter_position", type=float, nargs=3, default=None, help="Emitter xyz for fluid effects; defaults to OBB-based estimate if not provided.")
    # rigid_connected is only relevant for geometry effects, but we accept it as an optional argument here and will default to False when needed.
    parser.add_argument("--rigid_connected", action="store_true", help="Whether the receptor and effector are rigidly connected (only relevant for geometry effects).")
    return parser.parse_args()


def load_parameters(physical_effect: str) -> dict:
    parameter_config_file = os.path.join(dir_abs_path, f"{physical_effect}_parameters.yaml")
    with open(parameter_config_file, 'r') as f:
        parameters = yaml.safe_load(f)
    return parameters


def load_mapping_template(mapping: str):
    mapping_template_file = os.path.join(dir_abs_path, f"{mapping}_mapping.yaml")
    with open(mapping_template_file, 'r') as f:
        mapping_config = yaml.safe_load(f)
    return mapping_config


def compute_mapping_parameters(mapping: str, receptor_state_min: float|bool, receptor_state_max: float|bool, effector_state_min: float|bool, effector_state_max: float|bool) -> dict:
    if mapping == "step":
        mapping_parameters = {"RECEPTOR_STATE": (receptor_state_min + receptor_state_max) / 2, 
                              "EFFECTOR_STATE1": effector_state_min, 
                              "EFFECTOR_STATE2": effector_state_max}
    elif mapping == "linear":
        if isinstance(receptor_state_min, bool) or isinstance(receptor_state_max, bool):
            raise ValueError("For linear mapping, receptor states must be numeric.")
        if isinstance(effector_state_min, bool) or isinstance(effector_state_max, bool):
            raise ValueError("For linear mapping, effector states must be numeric.")
        coefficient = (effector_state_max - effector_state_min) / (receptor_state_max - receptor_state_min)
        mapping_parameters = {"COEFFICIENT": coefficient}
    elif mapping == "binary":
        mapping_parameters = {"RECEPTOR_STATE": receptor_state_max,
                              "EFFECTOR_STATE1": effector_state_min,
                              "EFFECTOR_STATE2": effector_state_max}
    elif mapping == "cumulative":
        # manually specify delta
        mapping_parameters = {"RECEPTOR_STATE": receptor_state_max}
    else:
        raise ValueError(f"Unsupported mapping type: {mapping}")
    
    return mapping_parameters


def instantiate_mapping_function(mapping_config: dict, receptor_state_min: float|bool, receptor_state_max: float|bool, effector_state_min: float|bool, effector_state_max: float|bool, delta: float = None) -> str:
    template = mapping_config["template"]
    mapping_parameters_dict = compute_mapping_parameters(mapping_config["type"], receptor_state_min, receptor_state_max, effector_state_min, effector_state_max)
    if mapping_config["type"] == "cumulative":
        if delta is None:
            raise ValueError("Delta must be provided for cumulative mapping.")
        mapping_parameters_dict["DELTA"] = delta
    for key, value in mapping_parameters_dict.items():
        template = template.replace(key, str(value))
    return template


def generate_definition(parameters_dict: dict) -> str:
    definition_list = []
    for key, value in parameters_dict.items():
        if key == "MAPPING_FUNCTION":
            definition_list.append(value + "\n")
        else:
            definition_list.append(f"{key} = {value}\n")
    return ''.join(definition_list)


def generate_function_call(mapping: str, receptor_state_name: str, effector_state_name: str) -> str:
    if mapping == "binary":
        function_call = f"binary_mapping({receptor_state_name})\n"
    elif mapping == "linear":
        function_call = f"linear_mapping({receptor_state_name})\n"
    elif mapping == "step":
        function_call = f"step_mapping({receptor_state_name})\n"
    elif mapping == "cumulative":
        function_call = f"cumulative_mapping({receptor_state_name}, {effector_state_name})\n"
    else:
        raise ValueError(f"Unsupported mapping type: {mapping}")
    return function_call
        

def build_fluid_function(physics_parameters: dict, mapping_config: dict, urdf_path: str, emitter_position: tuple,
                         receptor_state_min: float|bool, receptor_state_max: float|bool, effector_state_min: float|bool, effector_state_max: float|bool, output_dir: str,
                         delta: float = None):
    parameters_dict = {"MAX_DROPLET_SIZE": physics_parameters["MAX_DROPLET_SIZE"], 
                       "MIN_DROPLET_SIZE": physics_parameters["MIN_DROPLET_SIZE"],
                       "URDF_PATH": urdf_path,
                       "EMITTER_POSITION": emitter_position}
    
    mapping_instance = instantiate_mapping_function(mapping_config, receptor_state_min, receptor_state_max, physics_parameters["MIN_DROPLET_SIZE"], physics_parameters["MAX_DROPLET_SIZE"], delta)
    parameters_dict["MAPPING_FUNCTION"] = mapping_instance

    with open(os.path.join(dir_abs_path, "fluid_function.py"), 'r') as f:
        fluid_function_template = f.read()
    
    definitions = generate_definition(parameters_dict)
    function_call = generate_function_call(mapping_config["type"], physics_parameters["RECETPOR_STATE_NAME"], physics_parameters["EFFECTOR_STATE_NAME"])
    fluid_function_code = fluid_function_template.replace("INSERT_DEFINITIONS_HERE", definitions).replace("MAPPING_FUNCTION", function_call)

    with open(os.path.join(output_dir, "fluid_function_instanced.py"), 'w') as f:
        f.write(fluid_function_code)


def generate_joint_init(rigid_connected: bool, receptor_state_min: float, effector_state_min: float) -> str:
    joint_init_list = [f"\"effector_joint\": {effector_state_min},\n"]
    if not rigid_connected:
        joint_init_list.append(f"\"receptor_joint\": {receptor_state_min}\n")
    return ''.join(joint_init_list)


def generate_actuator_config(rigid_connected: bool) -> str:
    actuator_config_list = ["\"effector_act\": ImplicitActuatorCfg(joint_names_expr=[\"effector_joint\"], damping=5, stiffness=20),\n"]
    if not rigid_connected:
        actuator_config_list.append("\"receptor_act\": ImplicitActuatorCfg(joint_names_expr=[\"receptor_joint\"], damping=5, stiffness=20)")
    actuator_config = ''.join(actuator_config_list)
    actuator_config = "{" + actuator_config + "}"
    return actuator_config


def build_geometry_function(physics_parameters: dict, mapping_config: dict, usd_path: str, rigid_connected: bool,
                            receptor_state_min: float|bool, receptor_state_max: float|bool, effector_state_min: float, effector_state_max: float, output_dir: str,
                            delta: float = None):
    parameters_dict = {"USD_PATH": usd_path, "RIGID_CONNECTED": rigid_connected}
    
    if not rigid_connected:
        mapping_instance = instantiate_mapping_function(mapping_config, receptor_state_min, receptor_state_max, effector_state_min, effector_state_max, delta)
        parameters_dict["MAPPING_FUNCTION"] = mapping_instance
        parameters_dict["RECEPTOR_STATE_MIN"] = receptor_state_min
        parameters_dict["RECEPTOR_STATE_MAX"] = receptor_state_max
    else:
        parameters_dict["RECEPTOR_STATE_MIN"] = effector_state_min
        parameters_dict["RECEPTOR_STATE_MAX"] = effector_state_max

    with open(os.path.join(dir_abs_path, "geometry_function.py"), 'r') as f:
        geometry_function_template = f.read()
    
    definitions = generate_definition(parameters_dict)
    joint_initialize = generate_joint_init(rigid_connected, receptor_state_min, effector_state_min)
    actuators_config = generate_actuator_config(rigid_connected)
    if rigid_connected:
        function_call = "receptor_target_state"
    else:
        function_call = generate_function_call(mapping_config["type"], physics_parameters["RECETPOR_STATE_NAME"], physics_parameters["EFFECTOR_STATE_NAME"])
    geometry_function_code = geometry_function_template.replace("INSERT_DEFINITIONS_HERE", definitions)\
                                                       .replace("INITIALIZE_JOINT", joint_initialize)\
                                                       .replace("ACTUATORS_CONFIG", actuators_config)\
                                                       .replace("MAPPING_FUNCTION", function_call)

    with open(os.path.join(output_dir, "geometry_function_instanced.py"), 'w') as f:
        f.write(geometry_function_code)


def _normalize_physical_effect(physical_effect: str) -> str:
    normalized_effect = physical_effect.strip().lower()
    effect_aliases = {
        "fluid": "fluid",
        "geometry": "geometry",
        "geometric": "geometry",
    }
    if normalized_effect not in effect_aliases:
        supported_effects = ", ".join(sorted(effect_aliases))
        raise ValueError(
            f"Unsupported physical effect: {physical_effect}. Supported values: {supported_effects}"
        )
    return effect_aliases[normalized_effect]


def build_function(
    physical_effect: str,
    mapping_config: dict,
    receptor_state_min: float | bool,
    receptor_state_max: float | bool,
    effector_state_min: float | bool,
    effector_state_max: float | bool,
    delta: float = None,
    output_dir: str = None,
    **kwargs,
):
    normalized_effect = _normalize_physical_effect(physical_effect)
    physics_parameters = load_parameters(normalized_effect)

    if normalized_effect == "fluid":
        required_keys = ("urdf_path", "emitter_position")
        missing_keys = [key for key in required_keys if key not in kwargs]
        if missing_keys:
            raise ValueError(
                f"Missing required arguments for fluid effect: {', '.join(missing_keys)}"
            )
        return build_fluid_function(
            physics_parameters=physics_parameters,
            mapping_config=mapping_config,
            urdf_path=kwargs["urdf_path"],
            emitter_position=kwargs["emitter_position"],
            receptor_state_min=receptor_state_min,
            receptor_state_max=receptor_state_max,
            effector_state_min=effector_state_min,
            effector_state_max=effector_state_max,
            output_dir=output_dir,
            delta=delta,
        )

    required_keys = ("usd_path", "rigid_connected")
    missing_keys = [key for key in required_keys if key not in kwargs]
    if missing_keys:
        raise ValueError(
            f"Missing required arguments for geometry effect: {', '.join(missing_keys)}"
        )
    return build_geometry_function(
        physics_parameters=physics_parameters,
        mapping_config=mapping_config,
        usd_path=kwargs["usd_path"],
        rigid_connected=kwargs["rigid_connected"],
        receptor_state_min=receptor_state_min,
        receptor_state_max=receptor_state_max,
        effector_state_min=effector_state_min,
        effector_state_max=effector_state_max,
        output_dir=output_dir,
        delta=delta,
    )


def build_urdf_from_reconstruction(
    receptor_mesh_path: str,
    effector_mesh_path: str,
    base_mesh_path: str,
    articulation_results: dict,
    output_dir: str,
    robot_name: str = "articulated_object",
    urdf_filename: str = "object.urdf",
    rigid_connected: bool = False,
) -> dict:
    """Build a URDF from reconstructed part meshes and articulation estimation results.

    Args:
        receptor_mesh_path: Path to the receptor part mesh (e.g. receptor_mesh.glb).
        effector_mesh_path: Path to the effector part mesh (e.g. effector_mesh.glb).
        base_mesh_path: Path to the base mesh (e.g. base_mesh.glb).
        articulation_results: Parsed articulation JSON dict with keys "receiver"/"effector".
            Each value is either a skip-message string or a dict with keys:
            "type", "axis", "origin", "state".
        output_dir: Directory where the URDF and mesh files will be written.
        robot_name: Name embedded in the URDF <robot> tag.
        urdf_filename: Output URDF filename.
        rigid_connected: Whether the receptor and effector are rigidly connected.
    Returns:
        Dict with keys "urdf_path", "root_link", "virtual_root_name",
        "virtual_root_to_base_xyz" (from generate_urdf_from_open3d_meshes).
    """
    # RGBA in uint8
    effector_rgba = np.array([252, 141, 98, 255], dtype=np.uint8)  # red, effector
    receptor_rgba = np.array([102, 194, 165, 255], dtype=np.uint8)  # blue-green, receiver
    base_rgba = np.array([179, 179, 179, 255], dtype=np.uint8)  # gray, background
    meshes = {
        "base": o3d.io.read_triangle_mesh(base_mesh_path),
        "receptor": o3d.io.read_triangle_mesh(receptor_mesh_path),
        "effector": o3d.io.read_triangle_mesh(effector_mesh_path),
    }
    meshes["base"].paint_uniform_color(base_rgba[:3] / 255)
    meshes["receptor"].paint_uniform_color(receptor_rgba[:3] / 255)
    meshes["effector"].paint_uniform_color(effector_rgba[:3] / 255)

    articulations = []
    for role in ["receptor", "effector"]:
        # Support both "receptor" (main.py) and "receiver" (pred_mask variant) keys.
        result = articulation_results.get(role)
        if isinstance(result, str) or result is None:
            # Estimation was skipped or missing — treat as fixed joint.
            if rigid_connected:
                parent = "effector"
                print("rigid_connected is True, treating receptor as child of effector in URDF.")
            else:
                parent = "base"
            articulation = {
                "parent": parent,
                "child": role,
                "joint_type": "fixed",
                "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                "name": f"{role}_joint",
            }
        else:
            joint_type = result["type"]
            axis = result.get("axis", [0.0, 0.0, 1.0])
            origin_xyz = result.get("origin", [0.0, 0.0, 0.0])
            states = result.get("state", [0.0])
            lower = float(max(min(states), 0.0))
            upper = float(max(max(states), 0.0))
            articulation = {
                "parent": "base",
                "child": role,
                "joint_type": joint_type,
                "axis": axis,
                "origin": {"xyz": origin_xyz, "rpy": [0.0, 0.0, 0.0]},
                "limit": {"lower": lower, "upper": upper, "effort": 1.0, "velocity": 1.0},
                "name": f"{role}_joint",
            }
        articulations.append(articulation)

    return generate_urdf_from_open3d_meshes(
        meshes=meshes,
        articulations=articulations,
        output_dir=output_dir,
        robot_name=robot_name,
        root_link="base",
        urdf_filename=urdf_filename,
        insert_virtual_root=True,
        recenter_by_base=True,
    )


def _get_articulation_state_range(articulation_results: dict, role: str) -> tuple:
    """Return (min_state, max_state) for a role; (False, False) if skipped/unavailable."""
    result = articulation_results.get(role)
    if result is None and role == "receptor":
        result = articulation_results.get("receiver")
    if isinstance(result, str) or result is None:
        return False, False
    states = result.get("state", [0.0])
    return float(min(states)), float(max(states))


def compute_emitter_position(receptor_mesh_path: str, effector_mesh_path: str) -> tuple:
    receptor_mesh = o3d.io.read_triangle_mesh(receptor_mesh_path)
    effector_mesh = o3d.io.read_triangle_mesh(effector_mesh_path)
    receptor_center = np.asarray(receptor_mesh.get_center(), dtype=float)

    effector_obb = effector_mesh.get_oriented_bounding_box()
    effector_vertices = np.asarray(effector_obb.get_box_points(), dtype=float)

    if effector_vertices.shape != (8, 3):
        raise ValueError(
            f"Expected 8 oriented bounding box vertices, got shape {effector_vertices.shape}."
        )

    lower_vertex_indices = np.argsort(effector_vertices[:, 2])[:4]
    lower_vertices = effector_vertices[lower_vertex_indices]

    distances_to_receptor = np.linalg.norm(lower_vertices - receptor_center, axis=1)
    farthest_vertex_indices = np.argsort(distances_to_receptor)[-2:]
    farthest_vertices = lower_vertices[farthest_vertex_indices]

    emitter_position = farthest_vertices.mean(axis=0)
    return tuple(float(v) for v in emitter_position)


def translate_position(position: tuple | list, translation: tuple | list) -> tuple:
    position_np = np.asarray(position, dtype=float)
    translation_np = np.asarray(translation, dtype=float)
    return tuple(float(v) for v in position_np + translation_np)


def compile_function_instance(
    reconstruction_dir: str,
    articulation_dir: str,
    function_dir: str,
    output_dir: str,
    robot_name: str = "articulated_object",
    delta: float = None,
    **kwargs,
) -> dict:
    """Full compilation pipeline: meshes + articulation + function prediction → instanced script.

    Reads articulation and function prediction results from disk, builds a URDF from the
    reconstructed part meshes, then generates the physics function instance script.

    Args:
        receptor_mesh_path: Path to the receptor part mesh (e.g. receptor_mesh.glb).
        effector_mesh_path: Path to the effector part mesh (e.g. effector_mesh.glb).
        base_mesh_path: Path to the base mesh (e.g. base_mesh.glb).
        articulation_results_path: Path to the articulation estimation JSON.
            Keys are "receptor"/"effector" (main.py) or "receiver"/"effector" (pred_mask variant).
        function_results_path: Path to the function prediction JSON.
            Expected keys: "1" (physical effect letter), "2" (numerical function letter).
        output_dir: Directory for URDF and mesh outputs.
        robot_name: Robot name embedded in the URDF <robot> tag.
        delta: Delta value for cumulative mapping (required only when mapping type is "cumulative").
        **kwargs:
            emitter_position (tuple): Emitter xyz for fluid effects.
                Defaults to the OBB-based faucet emitter estimate if not provided.
            usd_path (str): USD asset path; required for geometry effects.

    Returns:
        Dict with keys:
            "urdf_result": output of generate_urdf_from_open3d_meshes (contains "urdf_path", etc.)
            "function_script_path": absolute path to the generated instanced function script.
            "physical_effect": resolved physical effect string (e.g. "fluid", "geometry").
            "mapping_type": resolved mapping type string (e.g. "step", "linear").
    """
    receptor_mesh_path = f"{reconstruction_dir}/reconstruction/receptor_mesh.glb"
    effector_mesh_path = f"{reconstruction_dir}/reconstruction/effector_mesh.glb"
    base_mesh_path = f"{reconstruction_dir}/reconstruction/base_mesh.glb"
    articulation_results_path = f"{articulation_dir}/articulation/articulation_results.json"
    function_results_path = f"{function_dir}/function/function_results.json"
    with open(articulation_results_path, "r") as f:
        articulation_results = json.load(f)
    with open(function_results_path, "r") as f:
        function_results = json.load(f)

    physical_effect = PHYSICAL_EFFECT_MAP[function_results["1"]]
    mapping_type = NUMERICAL_FUNCTION_MAP[function_results["2"]]

    receptor_state_min, receptor_state_max = _get_articulation_state_range(articulation_results, "receptor")
    receptor_state_min = max(receptor_state_min, 0.0) if isinstance(receptor_state_min, float) else receptor_state_min
    receptor_state_max = max(receptor_state_max, 0.0) if isinstance(receptor_state_max, float) else receptor_state_max
    effector_state_min, effector_state_max = _get_articulation_state_range(articulation_results, "effector")
    effector_state_min = max(effector_state_min, 0.0) if isinstance(effector_state_min, float) else effector_state_min
    effector_state_max = max(effector_state_max, 0.0) if isinstance(effector_state_max, float) else effector_state_max

    os.makedirs(output_dir, exist_ok=True)
    urdf_result = build_urdf_from_reconstruction(
        receptor_mesh_path=receptor_mesh_path,
        effector_mesh_path=effector_mesh_path,
        base_mesh_path=base_mesh_path,
        articulation_results=articulation_results,
        output_dir=output_dir,
        robot_name=robot_name,
        rigid_connected=kwargs.get("rigid_connected", False),
    )
    if physical_effect == "geometry":
        kwargs["usd_path"] = f"f\"{{abs_dir}}/object.usd\""

    mapping_config = load_mapping_template(mapping_type)

    effect_kwargs = dict(kwargs)
    if physical_effect == "fluid":
        effect_kwargs["urdf_path"] = f"f\"{{abs_dir}}/object.urdf\""
        if "emitter_position" not in effect_kwargs:
            effect_kwargs["emitter_position"] = compute_emitter_position(
                receptor_mesh_path=receptor_mesh_path,
                effector_mesh_path=effector_mesh_path,
            )
        effect_kwargs["emitter_position"] = translate_position(
            effect_kwargs["emitter_position"],
            urdf_result["virtual_root_to_base_xyz"],
        )
    elif physical_effect == "geometry":
        if "usd_path" not in effect_kwargs:
            raise ValueError("usd_path must be provided in kwargs for geometry physical effect.")
        receptor_result = articulation_results.get("receptor")
        effect_kwargs["rigid_connected"] = isinstance(receptor_result, str) or receptor_result is None
        

    build_function(
        physical_effect=physical_effect,
        mapping_config=mapping_config,
        receptor_state_min=receptor_state_min,
        receptor_state_max=receptor_state_max,
        effector_state_min=effector_state_min,
        effector_state_max=effector_state_max,
        delta=delta,
        output_dir=output_dir,
        **effect_kwargs,
    )

    function_script_path = os.path.join(output_dir, f"{physical_effect}_function_instanced.py")
    return {
        "urdf_result": urdf_result,
        "function_script_path": function_script_path,
        "physical_effect": physical_effect,
        "mapping_type": mapping_type,
    }


if __name__ == "__main__":
    args = parse_arguments()
    result = compile_function_instance(
        reconstruction_dir=args.reconstruction_dir,
        articulation_dir=args.articulation_dir,
        function_dir=args.function_dir,
        output_dir=args.output_dir,
        robot_name=args.robot_name,
        delta=args.delta,
        emitter_position=args.emitter_position,
        rigid_connected=args.rigid_connected,
    )
    print("Compilation complete. Results:")
    print(json.dumps(result, indent=2))
