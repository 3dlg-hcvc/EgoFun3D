import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import point_cloud_utils as pcu
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy


def load_mesh_data(geometry_path: str, part_annotation_path: str, function_instance_id: str) -> dict:
    full_mesh = o3d.io.read_triangle_mesh(geometry_path)
    with open(part_annotation_path, "r") as f:
        annotations = json.load(f)
    if function_instance_id not in annotations.keys():
        raise ValueError(f"Segment ID {function_instance_id} not found in annotations.")
    geometry_annotations = {}

    part_vertices_indices = []
    full_vertices_len = np.asarray(full_mesh.vertices).shape[0]
    full_vertices = np.arange(full_vertices_len)
    for role in ["receptor", "effector"]:
        print(annotations[function_instance_id].keys())
        part_name = annotations[function_instance_id][role]["label"]
        part_indices = annotations[function_instance_id][role]["indices"]
        part_vertices_indices.extend(part_indices)
        pid = annotations[function_instance_id][role]["pid"]
        if not part_indices:
            raise ValueError(f"No indices found for role {role} in segment {function_instance_id}.")
        part_mesh = full_mesh.select_by_index(part_indices)
        geometry_annotations[role] = {
            "part_name": part_name,
            "part_mesh": part_mesh,
            "pid": pid
        }
    if "remove" in annotations.keys():
        remove_indices = annotations["remove"]["remove"]["indices"]
        part_vertices_indices.extend(remove_indices)
    part_vertices_indices = list(set(part_vertices_indices))
    base_vertices = np.delete(full_vertices, part_vertices_indices, axis=0)
    base_mesh = full_mesh.select_by_index(base_vertices.tolist())
    geometry_annotations["base"] = {
        "part_name": "base",
        "part_mesh": base_mesh,
        "pid": 0
    }
    # geometry_annotations["relation"] = annotations[function_instance_id]["description"]
    return geometry_annotations


def load_point_cloud_data(geometry_path: str, part_annotation_path: str, function_instance_id: int) -> dict:
    full_pcd = pcu.load_mesh_v(geometry_path, np.float32)  # (N, 3)
    with open(part_annotation_path, "r") as f:
        annotations = json.load(f)
    if function_instance_id not in annotations:
        raise ValueError(f"Segment ID {function_instance_id} not found in annotations.")
    geometry_annotations = {}
    part_points_indices = []
    for role in ["receptor", "effector"]:
        # print(annotations[function_instance_id].keys())
        part_name = annotations[function_instance_id][role]["label"]
        part_indices = annotations[function_instance_id][role]["indices"]
        pid = annotations[function_instance_id][role]["pid"]
        if not part_indices:
            raise ValueError(f"No indices found for role {role} in segment {function_instance_id}.")
        part_pcd = full_pcd[part_indices]
        geometry_annotations[role] = {
            "part_pcd": part_pcd,
            "pid": pid
        }
        part_points_indices.extend(part_indices)
    if "remove" in annotations.keys():
        remove_indices = annotations["remove"]["remove"]["indices"]
        part_points_indices.extend(remove_indices)
    part_points_indices = list(set(part_points_indices))
    base_pcd = np.delete(full_pcd, part_points_indices, axis=0)
    center_point = np.array([-1.055603, -0.593659, -0.567707])
    dist_thresh = 1
    dist_to_center = np.linalg.norm(base_pcd - center_point, axis=1)
    base_pcd = base_pcd[dist_to_center < dist_thresh]
    geometry_annotations["base"] = {
        "part_pcd": base_pcd,
        "pid": 0
    }
    geometry_annotations["relation"] = annotations[function_instance_id]["description"]
    return geometry_annotations


def vec_to_str(v):
    return " ".join(f"{float(x):.8g}" for x in v)


def sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def neg(a):
    return [-float(x) for x in a]


def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def add_origin(parent: ET.Element, xyz=None, rpy=None):
    if xyz is None:
        xyz = [0.0, 0.0, 0.0]
    if rpy is None:
        rpy = [0.0, 0.0, 0.0]
    ET.SubElement(parent, "origin", {
        "xyz": vec_to_str(xyz),
        "rpy": vec_to_str(rpy),
    })


def add_mesh_geometry(parent: ET.Element, mesh_filename: str, scale=None):
    geometry = ET.SubElement(parent, "geometry")
    attrib = {"filename": mesh_filename}
    if scale is not None:
        attrib["scale"] = vec_to_str(scale)
    ET.SubElement(geometry, "mesh", attrib)


def add_inertial(link_elem: ET.Element, mass: float = 1.0):
    inertial = ET.SubElement(link_elem, "inertial")
    add_origin(inertial, [0, 0, 0], [0, 0, 0])
    ET.SubElement(inertial, "mass", {"value": str(float(mass))})
    ET.SubElement(inertial, "inertia", {
        "ixx": "0.001",
        "ixy": "0.0",
        "ixz": "0.0",
        "iyy": "0.001",
        "iyz": "0.0",
        "izz": "0.001",
    })


def sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
    )


def infer_root_link(meshes: Dict[str, o3d.geometry.TriangleMesh],
                    articulations: List[dict]) -> str:
    all_links = set(meshes.keys())
    children = {a["child"] for a in articulations}
    roots = list(all_links - children)
    if len(roots) == 1:
        return roots[0]
    if len(roots) == 0:
        raise ValueError("Could not infer root link: every link appears as a child.")
    raise ValueError(f"Multiple possible root links found: {roots}. Please specify root_link explicitly.")


def validate_articulations(meshes: Dict[str, o3d.geometry.TriangleMesh],
                           articulations: List[dict]) -> None:
    mesh_names = set(meshes.keys())
    child_to_joint = {}

    for a in articulations:
        parent = a["parent"]
        child = a["child"]

        if parent not in mesh_names:
            raise ValueError(f"Parent link '{parent}' not found in meshes.")
        if child not in mesh_names:
            raise ValueError(f"Child link '{child}' not found in meshes.")

        if child in child_to_joint:
            raise ValueError(
                f"Link '{child}' has more than one parent joint. URDF requires a tree."
            )
        child_to_joint[child] = a


def compute_base_recenter_translation(
    base_mesh: o3d.geometry.TriangleMesh,
) -> List[float]:
    """
    Translation that moves the base mesh so that:
      - AABB center x/y goes to 0
      - AABB min z goes to 0
    This translation is encoded in virtual_root -> base fixed joint.
    """
    aabb = base_mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    min_bound = aabb.get_min_bound()

    tx = -float(center[0])
    ty = -float(center[1])
    tz = -float(min_bound[2])
    return [tx, ty, tz]


def compute_link_frame_positions(
    meshes: Dict[str, o3d.geometry.TriangleMesh],
    articulations: List[dict],
    root_link: str,
) -> Dict[str, List[float]]:
    """
    Link-frame positions are kept in the ORIGINAL shared object/global frame.

    Convention:
      - root link frame is at [0, 0, 0] in the original object/global frame
      - each child link frame is at its parent joint's global xyz
    """
    joint_by_child = {a["child"]: a for a in articulations}
    link_frame_pos_global = {root_link: [0.0, 0.0, 0.0]}

    remaining = set(meshes.keys())
    remaining.remove(root_link)

    progressed = True
    while remaining and progressed:
        progressed = False
        for link_name in list(remaining):
            if link_name not in joint_by_child:
                raise ValueError(
                    f"Link '{link_name}' is not the root and has no parent articulation."
                )
            joint = joint_by_child[link_name]
            parent = joint["parent"]
            if parent in link_frame_pos_global:
                link_frame_pos_global[link_name] = [
                    float(x) for x in joint["origin"]["xyz"]
                ]
                remaining.remove(link_name)
                progressed = True

    if remaining:
        raise ValueError(
            f"Could not resolve articulation tree. Remaining links: {sorted(remaining)}"
        )

    return link_frame_pos_global


def export_meshes_to_obj(
    meshes: Dict[str, o3d.geometry.TriangleMesh],
    mesh_dir: str,
    write_triangle_uvs: bool = True,
) -> Dict[str, str]:
    os.makedirs(mesh_dir, exist_ok=True)

    mesh_relpaths = {}
    for name, mesh in meshes.items():
        safe_name = sanitize_name(name)
        filename = f"{safe_name}.obj"
        abs_path = os.path.join(mesh_dir, filename)

        mesh_to_save = copy.deepcopy(mesh)
        ok = o3d.io.write_triangle_mesh(
            abs_path,
            mesh_to_save,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=write_triangle_uvs,
        )
        if not ok:
            raise IOError(f"Failed to write mesh '{name}' to '{abs_path}'.")

        mesh_relpaths[name] = os.path.join(os.path.basename(mesh_dir), filename)

    return mesh_relpaths


def add_link(
    robot: ET.Element,
    link_name: str,
    mesh_relpath: Optional[str],
    link_frame_global: List[float],
    scale=(1.0, 1.0, 1.0),
    add_collision: bool = True,
    add_inertial_flag: bool = True,
    mass: float = 1.0,
):
    """
    Mesh geometry stays unchanged in the original shared object/global frame.

    Therefore:
      visual/collision origin = -link_frame_global

    Special case:
      if link_frame_global == [0, 0, 0], origin becomes zero automatically.
    """
    link_elem = ET.SubElement(robot, "link", {"name": sanitize_name(link_name)})

    if mesh_relpath is not None:
        visual = ET.SubElement(link_elem, "visual")
        add_origin(visual, xyz=neg(link_frame_global), rpy=[0, 0, 0])
        add_mesh_geometry(visual, mesh_relpath, scale=scale)

        if add_collision:
            collision = ET.SubElement(link_elem, "collision")
            add_origin(collision, xyz=neg(link_frame_global), rpy=[0, 0, 0])
            add_mesh_geometry(collision, mesh_relpath, scale=scale)

    if add_inertial_flag:
        add_inertial(link_elem, mass=mass)


def add_joint(
    robot: ET.Element,
    articulation: dict,
    parent_link_frame_global: List[float],
):
    """
    articulation["origin"]["xyz"] is in the original shared object/global frame.
    URDF joint origin must be expressed in the parent link frame.
    """
    joint_name = sanitize_name(articulation["name"])
    joint_type = articulation["joint_type"]
    parent_name = sanitize_name(articulation["parent"])
    child_name = sanitize_name(articulation["child"])

    joint_origin_global = [float(x) for x in articulation["origin"]["xyz"]]
    joint_rpy = [float(x) for x in articulation["origin"].get("rpy", (0, 0, 0))]
    joint_origin_in_parent = sub(joint_origin_global, parent_link_frame_global)

    joint_elem = ET.SubElement(robot, "joint", {
        "name": joint_name,
        "type": joint_type,
    })

    ET.SubElement(joint_elem, "parent", {"link": parent_name})
    ET.SubElement(joint_elem, "child", {"link": child_name})
    add_origin(joint_elem, xyz=joint_origin_in_parent, rpy=joint_rpy)

    if joint_type in {"revolute", "continuous", "prismatic"}:
        axis = articulation.get("axis", (0.0, 0.0, 1.0))
        ET.SubElement(joint_elem, "axis", {"xyz": vec_to_str(axis)})

    if joint_type in {"revolute", "prismatic"}:
        limit = articulation.get("limit", {})
        ET.SubElement(joint_elem, "limit", {
            "lower": str(float(limit.get("lower", 0.0))),
            "upper": str(float(limit.get("upper", 0.0))),
            "effort": str(float(limit.get("effort", 1.0))),
            "velocity": str(float(limit.get("velocity", 1.0))),
        })


def add_fixed_joint(
    robot: ET.Element,
    joint_name: str,
    parent_link: str,
    child_link: str,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
):
    joint_elem = ET.SubElement(robot, "joint", {
        "name": sanitize_name(joint_name),
        "type": "fixed",
    })
    ET.SubElement(joint_elem, "parent", {"link": sanitize_name(parent_link)})
    ET.SubElement(joint_elem, "child", {"link": sanitize_name(child_link)})
    add_origin(joint_elem, xyz=xyz, rpy=rpy)


def generate_urdf_from_open3d_meshes(
    meshes: Dict[str, o3d.geometry.TriangleMesh],
    articulations: List[dict],
    output_dir: str,
    robot_name: str = "articulated_object",
    root_link: Optional[str] = None,
    mesh_subdir: str = "meshes",
    scale=(1.0, 1.0, 1.0),
    add_collision: bool = True,
    add_inertial_flag: bool = True,
    default_mass: float = 1.0,
    urdf_filename: str = "object.urdf",
    insert_virtual_root: bool = True,
    virtual_root_name: str = "virtual_root",
    recenter_by_base: bool = True,
):
    if not meshes:
        raise ValueError("meshes is empty.")

    validate_articulations(meshes, articulations)

    if root_link is None:
        root_link = infer_root_link(meshes, articulations)
    if root_link not in meshes:
        raise ValueError(f"root_link '{root_link}' not found in meshes.")

    if insert_virtual_root and virtual_root_name in meshes:
        raise ValueError(
            f"virtual_root_name '{virtual_root_name}' collides with an existing mesh/link name."
        )

    # Compute recenter transform from base mesh, but DO NOT apply it to meshes/joints.
    if recenter_by_base:
        recenter_translation = compute_base_recenter_translation(meshes[root_link])
    else:
        recenter_translation = [0.0, 0.0, 0.0]

    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, mesh_subdir)
    mesh_relpaths = export_meshes_to_obj(meshes, mesh_dir)

    # Keep original articulation/global coordinates untouched.
    link_frame_pos_global = compute_link_frame_positions(
        meshes=meshes,
        articulations=articulations,
        root_link=root_link,
    )

    robot = ET.Element("robot", {"name": sanitize_name(robot_name)})

    if insert_virtual_root:
        add_link(
            robot=robot,
            link_name=virtual_root_name,
            mesh_relpath=None,
            link_frame_global=[0.0, 0.0, 0.0],
            scale=scale,
            add_collision=False,
            add_inertial_flag=add_inertial_flag,
            mass=default_mass,
        )

    for link_name in meshes.keys():
        add_link(
            robot=robot,
            link_name=link_name,
            mesh_relpath=mesh_relpaths[link_name],
            link_frame_global=link_frame_pos_global[link_name],
            scale=scale,
            add_collision=add_collision,
            add_inertial_flag=add_inertial_flag,
            mass=default_mass,
        )

    for articulation in articulations:
        parent_name = articulation["parent"]
        parent_link_frame_global = link_frame_pos_global[parent_name]
        add_joint(
            robot=robot,
            articulation=articulation,
            parent_link_frame_global=parent_link_frame_global,
        )

    if insert_virtual_root:
        add_fixed_joint(
            robot=robot,
            joint_name=f"{root_link}_virtual_root_joint",
            parent_link=virtual_root_name,
            child_link=root_link,
            xyz=recenter_translation,
            rpy=(0.0, 0.0, 0.0),
        )

    urdf_str = prettify_xml(robot)
    urdf_path = os.path.abspath(os.path.join(output_dir, urdf_filename))
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf_str)

    return {
        "urdf_path": urdf_path,
        "root_link": root_link,
        "virtual_root_name": virtual_root_name if insert_virtual_root else None,
        "virtual_root_to_base_xyz": recenter_translation,
    }


if __name__ == "__main__":
    articulation_file = "full_dataset/articulation_annotation/object1.articulations.json"
    mesh_file = "full_dataset/geometry/object1.ply"
    part_annotation_file = "full_dataset/part_annotation/object1.json"
    function_instance_id = "00"
    geometry_data = load_mesh_data(mesh_file, part_annotation_file, function_instance_id)
    # geometry_data = load_point_cloud_data(mesh_file, part_annotation_file, function_instance_id)
    if os.path.exists(articulation_file):
        with open(articulation_file, "r") as f:
            articulation_data = json.load(f)
    else:
        articulation_data = []
    part_name_list = ["receptor", "effector", "base"]
    articulation_list = []
    mesh_dict = {}
    for part_name in part_name_list:
        geometry = geometry_data[part_name]
        Rot90 = geometry["part_mesh"].get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        rot90_scipy_matrix = R.from_euler('x', 90, degrees=True).as_matrix()
        geometry["part_mesh"].rotate(Rot90, center=(0, 0, 0))
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(geometry["part_pcd"])
        # os.makedirs(f"visualizations/object6_urdf/meshes", exist_ok=True)
        # o3d.io.write_point_cloud(f"visualizations/object6_urdf/meshes/{part_name}.ply", o3d_pcd)
        mesh_dict[part_name] = geometry_data[part_name]["part_mesh"]
        # mesh_dict[part_name] = o3d_pcd
        if part_name == "base":
            continue
        pid = geometry["pid"]
        part_articulation = None
        for articulation in articulation_data:
            if articulation["pid"] == pid:
                part_articulation = articulation
                break
        if part_articulation is None:
            articulation_dict = {"parent": "base", "child": part_name, "joint_type": "fixed", "origin": {"xyz": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0)}, "name": f"{part_name}_joint"}
        else:
            origin_axis = np.array(part_articulation.get("axis", (0.0, 0.0, 1.0)))
            rotate_axis = rot90_scipy_matrix @ origin_axis
            part_articulation["axis"] = rotate_axis.tolist()
            origin_xyz = np.array(part_articulation.get("origin", (0.0, 0.0, 0.0)))
            rotate_xyz = rot90_scipy_matrix @ origin_xyz
            part_articulation["origin"] = rotate_xyz.tolist()
            articulation_dict = {
                "parent": "base",
                "child": part_name,
                "joint_type": part_articulation["type"],
                "axis": part_articulation.get("axis", (0.0, 0.0, 1.0)),
                "origin": {"xyz": part_articulation.get("origin", (0.0, 0.0, 0.0)), "rpy": (0.0, 0.0, 0.0)},
                "limit": {"lower": part_articulation.get("rangeMin", 0.0), "upper": part_articulation.get("rangeMax", 0.0), "effort": 1.0, "velocity": 1.0},
                "name": f"{part_name}_joint"
            }
        articulation_list.append(articulation_dict)
    # build_urdf_from_parts(
    #     mesh_dict,
    #     articulation_list,
    #     urdf_path="visualizations/object6_urdf/object6.urdf",
    #     robot_name="object6",
    #     mesh_dir="visualizations/object6_urdf/meshes",
    #     place_base_at_origin=True,
    #     base_link_name="base",
    #     virtual_link_name="virtual_root",
    # )

    result = generate_urdf_from_open3d_meshes(
        meshes=mesh_dict,
        articulations=articulation_list,
        output_dir="visualizations/object1_urdf2",
        robot_name="object1",
        root_link="base",
        insert_virtual_root=True,
        virtual_root_name="world_root",
        recenter_by_base=True,
    )