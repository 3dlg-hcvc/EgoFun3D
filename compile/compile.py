import yaml
import os


dir_abs_path = os.path.dirname(os.path.abspath(__file__))

def load_parameters(physical_effect: str) -> dict:
    parameter_config_file = os.path.join(dir_abs_path, f"{physical_effect}_function.yaml")
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
            if isinstance(value, str):
                definition_list.append(f'{key} = "{value}"\n')
            else:
                definition_list.append(f"{key} = {value}\n")
    return ''.join(definition_list)


def generate_function_call(mapping: str, receptor_state_name: str, effector_state_name: str) -> str:
    if mapping == "binary":
        function_call = f"binary_mapping({receptor_state_name})"
    elif mapping == "linear":
        function_call = f"linear_mapping({receptor_state_name})"
    elif mapping == "step":
        function_call = f"step_mapping({receptor_state_name})"
    elif mapping == "cumulative":
        function_call = f"cumulative_mapping({receptor_state_name}, {effector_state_name})"
    else:
        raise ValueError(f"Unsupported mapping type: {mapping}")
    return function_call
        

def build_fluid_function(physics_parameters: dict, mapping_config: dict, urdf_path: str, emitter_position: tuple,
                         receptor_state_min: float|bool, receptor_state_max: float|bool, effector_state_min: float|bool, effector_state_max: float|bool,
                         delta: float = None):
    parameters_dict = {"MAX_DROPLET_SIZE": physics_parameters["MAX_DROPLET_SIZE"], 
                       "MIN_DROPLET_SIZE": physics_parameters["MIN_DROPLET_SIZE"],
                       "URDF_PATH": urdf_path,
                       "EMITTER_POSITION": emitter_position}
    
    mapping_instance = instantiate_mapping_function(mapping_config, receptor_state_min, receptor_state_max, effector_state_min, effector_state_max, delta)
    parameters_dict["MAPPING_FUNCTION"] = mapping_instance

    with open(os.path.join(dir_abs_path, "fluid_function.py"), 'r') as f:
        fluid_function_template = f.read()
    
    definitions = generate_definition(parameters_dict)
    function_call = generate_function_call(mapping_config["type"], physics_parameters["RECETPOR_STATE_NAME"], physics_parameters["EFFECTOR_STATE_NAME"])
    fluid_function_code = fluid_function_template.replace("INSERT_DEFINITION", definitions).replace("MAPPING_FUNCTION", function_call)

    with open(os.path.join(dir_abs_path, "fluid_function_instanced.py"), 'w') as f:
        f.write(fluid_function_code)


def generate_joint_init(rigid_connected: bool, receptor_state_min: float, effector_state_min: float) -> str:
    joint_init_list = [f"effector_joint: {effector_state_min},\n"]
    if not rigid_connected:
        joint_init_list.append(f"receptor_joint: {receptor_state_min}\n")
    return ''.join(joint_init_list)


def generate_actuator_config(rigid_connected: bool) -> str:
    actuator_config_list = ["\"effector_act\": ImplicitActuatorCfg(joint_names_expr=[\"effector_joint\"], damping=5, stiffness=20),\n"]
    if not rigid_connected:
        actuator_config_list.append("\"receptor_act\": ImplicitActuatorCfg(joint_names_expr=[\"receptor_joint\"], damping=5, stiffness=20)")
    actuator_config = ''.join(actuator_config_list)
    actuator_config = "{" + actuator_config + "}"
    return actuator_config


def build_geometry_function(physics_parameters: dict, mapping_config: dict, usd_path: str, rigid_connected: bool,
                            receptor_state_min: float|bool, receptor_state_max: float|bool, effector_state_min: float, effector_state_max: float,
                            delta: float = None):
    parameters_dict = {"USD_PATH": usd_path, "RIGID_CONNECTED": rigid_connected}
    
    if not rigid_connected:
        mapping_instance = instantiate_mapping_function(mapping_config, receptor_state_min, receptor_state_max, effector_state_min, effector_state_max, delta)
        parameters_dict["MAPPING_FUNCTION"] = mapping_instance

    with open(os.path.join(dir_abs_path, "geometry_function.py"), 'r') as f:
        geometry_function_template = f.read()
    
    definitions = generate_definition(parameters_dict)
    joint_initialize = generate_joint_init(rigid_connected, receptor_state_min, effector_state_min)
    actuators_config = generate_actuator_config(rigid_connected)
    if rigid_connected:
        function_call = str(effector_state_max)
    else:
        function_call = generate_function_call(mapping_config["type"], physics_parameters["RECETPOR_STATE_NAME"], physics_parameters["EFFECTOR_STATE_NAME"])
    geometry_function_code = geometry_function_template.replace("INSERT_DEFINITION", definitions)\
                                                       .replace("INITIALIZE_JOINT", joint_initialize)\
                                                       .replace("ACTUATORS_CONFIG", actuators_config)\
                                                       .replace("MAPPING_FUNCTION", function_call)

    with open(os.path.join(dir_abs_path, "geometry_function_instanced.py"), 'w') as f:
        f.write(geometry_function_code)