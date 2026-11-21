"""Build and render a video-guided two-wrist Action 4 preview scene.

Run with the repository's embedded Blender Python::

    .venv-hccepose/bin/python scripts/blender_real_scene_preview.py

The script intentionally writes to ``outputs/blender_real_scene_preview`` and
does not modify the existing BOP datasets or rendering pipeline.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "dataset/action4-v6-domain-pilot-200/models/obj_000001.glb"
DEFAULT_OUTPUT = ROOT / "outputs/blender_real_scene_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--pose", choices=("open", "thumbs_up"), default="open")
    parser.add_argument("--scene", choices=("raised", "tabletop"), default="raised")
    parser.add_argument("--camera-pitch-deg", type=float,
                        help="Action 4 pitch relative to the wrist rig; defaults depend on scene")
    parser.add_argument("--screen-tilt-deg", type=float, default=12.0,
                        help="Small tabletop tilt that exposes the top face while keeping the screen camera-facing")
    parser.add_argument("--version-tag", default="v2")
    return parser.parse_args()


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def material(name: str, color: tuple[float, float, float, float], *,
             roughness: float = 0.65, metallic: float = 0.0,
             emission: tuple[float, float, float, float] | None = None,
             emission_strength: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission"].default_value = emission
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def fabric_material() -> bpy.types.Material:
    mat = material("Cuff gray woven fabric", (0.12, 0.15, 0.16, 1.0), roughness=0.92)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 175.0
    tex.inputs["Detail"].default_value = 2.0
    tex.inputs["Roughness"].default_value = 0.72
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.20
    bump.inputs["Distance"].default_value = 0.0012
    links.new(tex.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def smooth(obj: bpy.types.Object) -> None:
    if obj.type == "MESH":
        for polygon in obj.data.polygons:
            polygon.use_smooth = True


def parent_local(obj: bpy.types.Object, parent: bpy.types.Object,
                 location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)) -> bpy.types.Object:
    obj.parent = parent
    obj.matrix_parent_inverse.identity()
    obj.location = location
    obj.rotation_euler = rotation
    return obj


def rounded_box(name: str, dimensions: tuple[float, float, float],
                location: tuple[float, float, float], mat: bpy.types.Material,
                *, parent: bpy.types.Object | None = None,
                rotation=(0.0, 0.0, 0.0), bevel=0.003) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        modifier = obj.modifiers.new("Soft manufactured edges", "BEVEL")
        modifier.width = min(bevel, min(dimensions) * 0.45)
        modifier.segments = 3
        modifier.affect = "EDGES"
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    assign_material(obj, mat)
    if parent is not None:
        parent_local(obj, parent, location, rotation)
    else:
        obj.location = location
        obj.rotation_euler = rotation
    return obj


def ellipsoid(name: str, dimensions: tuple[float, float, float],
              location: tuple[float, float, float], mat: bpy.types.Material,
              *, parent: bpy.types.Object | None = None,
              rotation=(0.0, 0.0, 0.0), segments=48) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=24,
                                        location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, mat)
    smooth(obj)
    if parent is not None:
        parent_local(obj, parent, location, rotation)
    else:
        obj.location = location
        obj.rotation_euler = rotation
    return obj


def capsule_between(name: str, start: Vector, end: Vector, radius: float,
                    mat: bpy.types.Material, parent: bpy.types.Object) -> list[bpy.types.Object]:
    direction = end - start
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=direction.length,
                                       location=(0.0, 0.0, 0.0))
    body = bpy.context.object
    body.name = f"{name}_shaft"
    assign_material(body, mat)
    smooth(body)
    parent_local(body, parent, midpoint, direction.to_track_quat("Z", "Y").to_euler())
    ends = [body]
    for index, point in enumerate((start, end)):
        ends.append(ellipsoid(f"{name}_joint_{index}", (2 * radius,) * 3,
                              point, mat, parent=parent, segments=32))
    return ends


def curve_tube(name: str, points: list[tuple[float, float, float]], radius: float,
               mat: bpy.types.Material, parent: bpy.types.Object) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 16
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bezier, coordinate in zip(spline.bezier_points, points):
        bezier.co = coordinate
        bezier.handle_left_type = "AUTO"
        bezier.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)
    parent_local(obj, parent)
    return obj


def cuff_cylinder(name: str, parent: bpy.types.Object, center_y: float,
                  mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1.0, depth=1.0,
                                       location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    # Default cylinder axis Z; after X=90 degrees its axis follows the wrist Y axis.
    obj.scale = (0.046, 0.030, 0.034)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, mat)
    smooth(obj)
    parent_local(obj, parent, (0.0, center_y, 0.060), (math.pi / 2, 0.0, 0.0))
    return obj


def cuff_ring(name: str, parent: bpy.types.Object, y: float,
              mat: bpy.types.Material, major=0.038, minor=0.0027) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=12,
                                    location=(0.0, 0.0, 0.0),
                                    major_radius=major, minor_radius=minor)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (1.0, 0.69, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, mat)
    smooth(obj)
    parent_local(obj, parent, (0.0, y, 0.060), (math.pi / 2, 0.0, 0.0))
    return obj


def tapered_forearm(name: str, parent: bpy.types.Object,
                    mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=0.76, radius2=1.0,
                                   depth=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    # Local -Z becomes the wrist end at +Y; local +Z becomes the wider elbow end.
    # Cone depth is already 1.0, so the Z scale is the full desired length.
    obj.scale = (0.052, 0.036, 0.540)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, mat)
    smooth(obj)
    parent_local(obj, parent, (0.0, -0.235, 0.055), (math.pi / 2, 0.0, 0.0))
    bevel = obj.modifiers.new("Soft forearm silhouette", "BEVEL")
    bevel.width = 0.006
    bevel.segments = 4
    return obj


def add_palm_creases(parent: bpy.types.Object, mats: dict[str, bpy.types.Material],
                     side: str) -> None:
    sign = -1.0 if side == "left" else 1.0
    crease_sets = [
        [(-0.029, 0.118, 0.0858), (-0.008, 0.111, 0.0862),
         (0.014, 0.113, 0.0860), (0.030, 0.124, 0.0852)],
        [(-0.025, 0.139, 0.0828), (-0.006, 0.132, 0.0852),
         (0.014, 0.133, 0.0848), (0.026, 0.142, 0.0822)],
        [(sign * 0.031, 0.090, 0.0820), (sign * 0.022, 0.106, 0.0848),
         (sign * 0.020, 0.125, 0.0840)],
    ]
    for index, points in enumerate(crease_sets):
        curve_tube(f"{side}_palm_crease_{index}", points, 0.00048,
                   mats["crease"], parent)


def create_hand(parent: bpy.types.Object, mats: dict[str, bpy.types.Material],
                pose: str, side: str) -> None:
    skin = mats["skin"]
    tapered_forearm(f"{side}_forearm", parent, skin)
    ellipsoid(f"{side}_wrist", (0.072, 0.075, 0.052),
              (0.0, 0.015, 0.057), skin, parent=parent)
    ellipsoid(f"{side}_palm", (0.084, 0.118, 0.043),
              (0.0, 0.117, 0.064), skin, parent=parent,
              rotation=(0.025, 0.0, 0.0))

    if pose == "open":
        x_positions = (-0.030, -0.010, 0.011, 0.031)
        lengths = (0.080, 0.094, 0.090, 0.073)
        for index, (x, length) in enumerate(zip(x_positions, lengths)):
            base = Vector((x, 0.161, 0.066))
            mid = Vector((x * 1.04, 0.161 + length * 0.54, 0.065 + 0.004 * (1 - index % 2)))
            tip = Vector((x * 1.08, 0.161 + length, 0.059))
            capsule_between(f"{side}_finger_{index}_proximal", base, mid, 0.0090, skin, parent)
            capsule_between(f"{side}_finger_{index}_distal", mid, tip, 0.0076, skin, parent)
        thumb_sign = -1.0 if side == "left" else 1.0
        base = Vector((thumb_sign * 0.037, 0.107, 0.057))
        mid = Vector((thumb_sign * 0.061, 0.135, 0.057))
        tip = Vector((thumb_sign * 0.073, 0.165, 0.060))
        capsule_between(f"{side}_thumb_proximal", base, mid, 0.0105, skin, parent)
        capsule_between(f"{side}_thumb_distal", mid, tip, 0.0090, skin, parent)
        add_palm_creases(parent, mats, side)
    else:
        # Four curled fingers form a compact fist while the thumb points upward.
        x_positions = (-0.028, -0.009, 0.010, 0.029)
        for index, x in enumerate(x_positions):
            base = Vector((x, 0.153, 0.070))
            mid = Vector((x, 0.176, 0.078))
            tip = Vector((x * 0.88, 0.166, 0.046))
            capsule_between(f"{side}_fist_{index}_proximal", base, mid, 0.0096, skin, parent)
            capsule_between(f"{side}_fist_{index}_distal", mid, tip, 0.0084, skin, parent)
        thumb_sign = -1.0 if side == "left" else 1.0
        base = Vector((thumb_sign * 0.035, 0.119, 0.069))
        mid = Vector((thumb_sign * 0.044, 0.145, 0.103))
        tip = Vector((thumb_sign * 0.046, 0.164, 0.143))
        capsule_between(f"{side}_thumb_up_proximal", base, mid, 0.0110, skin, parent)
        capsule_between(f"{side}_thumb_up_distal", mid, tip, 0.0092, skin, parent)


def create_cuff_and_mount(parent: bpy.types.Object,
                          mats: dict[str, bpy.types.Material], side: str) -> None:
    cuff_cylinder(f"{side}_cuff_fabric", parent, 0.005, mats["fabric"])
    for y in (-0.029, 0.039):
        cuff_ring(f"{side}_cuff_black_edge_{y:+.3f}", parent, y, mats["rubber"],
                  major=0.0395, minor=0.0038)
        cuff_ring(f"{side}_cuff_green_edge_{y:+.3f}", parent, y, mats["green"],
                  major=0.0410, minor=0.0030)

    rounded_box(f"{side}_outer_pad", (0.052, 0.027, 0.013),
                (0.0, 0.027, 0.091), mats["pad"], parent=parent,
                rotation=(0.10, 0.0, 0.0), bevel=0.006)
    rounded_box(f"{side}_mount_base", (0.056, 0.045, 0.010),
                (0.0, -0.004, 0.091), mats["mount"], parent=parent, bevel=0.004)

    # Knurled central pivot and stacked hinge blocks.
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.016, depth=0.048,
                                       location=(0.0, 0.0, 0.0))
    pivot = bpy.context.object
    pivot.name = f"{side}_mount_pivot"
    assign_material(pivot, mats["mount"])
    smooth(pivot)
    parent_local(pivot, parent, (0.0, -0.018, 0.106), (0.0, math.pi / 2, 0.0))
    rounded_box(f"{side}_hinge_lower", (0.032, 0.027, 0.018),
                (0.0, -0.024, 0.107), mats["mount"], parent=parent, bevel=0.004)
    rounded_box(f"{side}_short_arm", (0.028, 0.052, 0.012),
                (0.0, -0.050, 0.112), mats["mount"], parent=parent,
                rotation=(0.08, 0.0, 0.0), bevel=0.004)


def camera_decals(camera_root: bpy.types.Object, mats: dict[str, bpy.types.Material],
                  side: str, identifier: str, marker_x: float) -> None:
    # Preserve the GLB screen, body, controls and original material.  Only add
    # the two thin paper decals observed in the reference video.
    # The camera root is rolled 180 degrees so local +Y is visually above the
    # screen.  Positive local X therefore projects to the screen's left side.
    rounded_box(f"{side}_marker_back", (0.0175, 0.0185, 0.00010),
                (marker_x, -0.0005, -0.01600), mats["label"],
                parent=camera_root, bevel=0.0005)
    patterns = {
        "left": {0, 1, 2, 3, 4, 5, 9, 10, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24},
        "right": {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 18, 19, 20, 21, 22, 23, 24},
    }
    cell = 0.0030
    for row in range(5):
        for column in range(5):
            index = row * 5 + column
            if index not in patterns[side]:
                continue
            x = marker_x + (column - 2) * cell
            y = -0.0005 + (2 - row) * cell
            rounded_box(f"{side}_marker_{row}_{column}",
                        (0.00265, 0.00265, 0.00006),
                        (x, y, -0.01608), mats["ink"], parent=camera_root,
                        bevel=0.00012)

    # Top label sits just above the positive local Y face.  The GLB's own
    # record button is intentionally left untouched.
    rounded_box(f"{side}_top_label", (0.027, 0.00012, 0.010),
                (0.005, 0.02166, 0.001), mats["label"],
                parent=camera_root, bevel=0.0010)
    bpy.ops.object.text_add(location=(0.0, 0.0, 0.0))
    text_obj = bpy.context.object
    text_obj.name = f"{side}_top_label_text"
    text_obj.data.body = identifier
    text_obj.data.align_x = "CENTER"
    text_obj.data.align_y = "CENTER"
    text_obj.data.size = 0.0062
    text_obj.data.extrude = 0.00010
    text_obj.data.materials.append(mats["ink"])
    parent_local(text_obj, camera_root, (0.005, 0.02175, 0.001),
                 (math.pi / 2, 0.0, 0.0))


def create_camera_instance(base_mesh: bpy.types.Object, parent: bpy.types.Object,
                           mats: dict[str, bpy.types.Material], side: str,
                           outer_sign: float, camera_pitch_degrees: float,
                           camera_roll_degrees: float,
                           camera_offset_y: float) -> bpy.types.Object:
    camera_root = bpy.data.objects.new(f"{side}_Action4_root", None)
    bpy.context.collection.objects.link(camera_root)
    parent_local(camera_root, parent, (0.0, camera_offset_y, 0.130))
    camera_root.rotation_mode = "QUATERNION"
    base_rotation = Euler((math.radians(camera_pitch_degrees), 0.0,
                           math.radians(outer_sign * 2.0)), "XYZ").to_quaternion()
    camera_root.rotation_quaternion = base_rotation @ Quaternion(
        (0.0, 0.0, 1.0), math.radians(camera_roll_degrees))

    body = base_mesh.copy()
    body.data = base_mesh.data
    body.name = f"{side}_Action4_body"
    bpy.context.collection.objects.link(body)
    parent_local(body, camera_root)
    body.scale = (0.001, 0.001, 0.001)
    body.hide_render = False
    body.hide_viewport = False
    if body.data is not base_mesh.data or body.modifiers:
        raise RuntimeError("Action 4 instance must preserve the imported GLB mesh without modifiers")
    marker_x = 0.019 if abs(camera_roll_degrees) > 90.0 else -0.019
    camera_decals(camera_root, mats, side,
                  "dji-003" if side == "left" else "dji-004", marker_x)

    connector_x = 0.039
    rounded_box(f"{side}_cable_connector", (0.014, 0.014, 0.010),
                (connector_x, -0.001, -0.001), mats["rubber"],
                parent=camera_root, bevel=0.0020)
    rounded_box(f"{side}_strain_relief", (0.013, 0.009, 0.008),
                (0.047, -0.001, -0.001), mats["rubber"],
                parent=camera_root, bevel=0.0020)

    curve_tube(f"{side}_cable", [
        (-0.050, -0.036, 0.128),
        (-0.061, -0.080, 0.105),
        (-0.067, -0.165, 0.071),
        (-0.060, -0.285, 0.052),
        (-0.083, -0.410, 0.050),
    ], 0.0028, mats["cable"], parent)
    return camera_root


def create_wrist_assembly(base_mesh: bpy.types.Object, mats: dict[str, bpy.types.Material],
                          side: str, location: tuple[float, float, float],
                          rotation: tuple[float, float, float], pose: str,
                          camera_pitch_degrees: float,
                          camera_roll_degrees: float,
                          camera_offset_y: float) -> bpy.types.Object:
    root = bpy.data.objects.new(f"{side}_wrist_rig", None)
    bpy.context.collection.objects.link(root)
    root.location = location
    root.rotation_euler = rotation
    create_hand(root, mats, pose, side)
    create_cuff_and_mount(root, mats, side)
    outer_sign = -1.0 if side == "left" else 1.0
    create_camera_instance(base_mesh, root, mats, side, outer_sign,
                           camera_pitch_degrees, camera_roll_degrees,
                           camera_offset_y)
    return root


def create_grid_floor(mats: dict[str, bpy.types.Material], surface_z: float = -0.006) -> None:
    rounded_box("White cutting mat", (2.20, 2.10, 0.012),
                (0.0, 0.05, surface_z - 0.006), mats["floor"], bevel=0.006)
    for coordinate in [index * 0.05 for index in range(-20, 21)]:
        rounded_box(f"Grid vertical {coordinate:+.2f}", (0.0009, 2.00, 0.0006),
                    (coordinate, 0.05, surface_z + 0.0004), mats["grid"], bevel=0.0)
        rounded_box(f"Grid horizontal {coordinate:+.2f}", (2.00, 0.0009, 0.0006),
                    (0.0, coordinate + 0.05, surface_z + 0.0004), mats["grid"], bevel=0.0)


def create_raised_background(mats: dict[str, bpy.types.Material]) -> None:
    # The opening reference view is dominated by a vertical pale wall.  The
    # gridded cutting mat appears only as a narrow strip at the bottom.
    create_grid_floor(mats, surface_z=-0.110)


def create_denim(mats: dict[str, bpy.types.Material]) -> None:
    # A low, folded garment patch provides the blue context seen through most of the video.
    mesh = bpy.data.meshes.new("Denim garment mesh")
    vertices = [
        (-0.22, 0.27, 0.002), (0.23, 0.25, 0.004), (0.30, 0.54, 0.005),
        (0.07, 0.62, 0.008), (-0.29, 0.54, 0.004), (-0.33, 0.36, 0.003),
    ]
    mesh.from_pydata(vertices, [], [[0, 1, 2, 3, 4, 5]])
    mesh.update()
    obj = bpy.data.objects.new("Blue denim garment", mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, mats["denim"])
    solidify = obj.modifiers.new("Denim thickness", "SOLIDIFY")
    solidify.thickness = 0.004
    bevel = obj.modifiers.new("Soft cloth edge", "BEVEL")
    bevel.width = 0.012
    bevel.segments = 4
    subdiv = obj.modifiers.new("Soft cloth surface", "SUBSURF")
    subdiv.levels = 2
    subdiv.render_levels = 2


def aim_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def orient_action4_screen_to_viewer(camera_root: bpy.types.Object,
                                    wrist_root: bpy.types.Object,
                                    viewer_location: Vector,
                                    top_tilt_degrees: float) -> None:
    """Aim the GLB rear screen at the render camera without changing its mesh.

    The GLB rear-screen outward normal is local -Z and its visual upper edge is
    local +Y.  We first construct a camera-facing orthonormal basis, then mix a
    small amount of the view direction into +Y so the labelled top face remains
    visible instead of becoming exactly edge-on.
    """
    bpy.context.view_layer.update()
    world_position = camera_root.matrix_world.translation
    view_direction = (viewer_location - world_position).normalized()
    hand_direction = wrist_root.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    screen_up = hand_direction - view_direction * hand_direction.dot(view_direction)
    if screen_up.length < 1e-6:
        screen_up = Vector((0.0, 0.0, 1.0))
    screen_up.normalize()

    tilt = math.radians(top_tilt_degrees)
    tilted_up = (screen_up * math.cos(tilt) + view_direction * math.sin(tilt)).normalized()
    screen_normal = (view_direction * math.cos(tilt) - screen_up * math.sin(tilt)).normalized()
    local_z_world = -screen_normal
    local_x_world = tilted_up.cross(local_z_world).normalized()
    local_y_world = local_z_world.cross(local_x_world).normalized()
    world_rotation = Matrix((local_x_world, local_y_world, local_z_world)).transposed().to_quaternion()

    parent_rotation = wrist_root.matrix_world.to_quaternion()
    camera_root.rotation_mode = "QUATERNION"
    camera_root.rotation_quaternion = parent_rotation.inverted() @ world_rotation
    bpy.context.view_layer.update()

    # Reattach the first cable control points to the connector after aiming.
    connector = bpy.data.objects.get(camera_root.name.replace("Action4_root", "strain_relief"))
    cable = bpy.data.objects.get(camera_root.name.replace("Action4_root", "cable"))
    if connector is not None and cable is not None and cable.type == "CURVE":
        connector_world = connector.matrix_world @ Vector((0.007, 0.0, 0.0))
        connector_local = wrist_root.matrix_world.inverted() @ connector_world
        points = cable.data.splines[0].bezier_points
        points[0].co = connector_local
        if len(points) > 2:
            points[1].co = connector_local.lerp(points[2].co, 0.42)


def setup_camera_and_lights(scene: bpy.types.Scene, scene_mode: str) -> None:
    camera_data = bpy.data.cameras.new("Reference-like overhead camera")
    camera = bpy.data.objects.new("Reference-like overhead camera", camera_data)
    bpy.context.collection.objects.link(camera)
    if scene_mode == "raised":
        camera.location = (0.0, -0.80, 0.47)
        camera_data.lens = 33.0
        camera_target = (0.0, 0.02, 0.53)
        camera_data.dof.focus_distance = 0.68
    else:
        camera.location = (0.0, -0.74, 0.71)
        camera_data.lens = 45.0
        camera_target = (0.0, 0.025, 0.065)
        camera_data.dof.focus_distance = 0.84
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = True
    camera_data.dof.aperture_fstop = 6.0
    aim_at(camera, camera_target)
    scene.camera = camera

    for name, location, energy, size, color in (
        ("Large soft key", (-0.38, -0.36, 1.15), 95.0, 0.72, (1.0, 0.94, 0.86)),
        ("Cool fill", (0.48, -0.02, 0.78), 42.0, 0.55, (0.78, 0.88, 1.0)),
        ("Top bounce", (0.0, 0.58, 0.92), 30.0, 0.48, (1.0, 1.0, 1.0)),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        aim_at(light, (0.0, 0.04, 0.04))


def build_materials() -> dict[str, bpy.types.Material]:
    return {
        "skin": material("Skin", (0.34, 0.17, 0.10, 1.0), roughness=0.58),
        "crease": material("Subtle palm creases", (0.17, 0.055, 0.025, 1.0), roughness=0.90),
        "fabric": fabric_material(),
        "green": material("Fluorescent green piping", (0.16, 0.72, 0.008, 1.0), roughness=0.72),
        "rubber": material("Black rubber", (0.012, 0.016, 0.019, 1.0), roughness=0.78),
        "mount": material("Mount black plastic", (0.025, 0.030, 0.034, 1.0), roughness=0.52),
        "pad": material("Translucent-looking wrist pad", (0.76, 0.79, 0.80, 1.0), roughness=0.64),
        "cable": material("Cable rubber", (0.008, 0.010, 0.012, 1.0), roughness=0.82),
        "label": material("Matte white labels", (0.88, 0.88, 0.83, 1.0), roughness=0.82),
        "ink": material("Label and marker ink", (0.010, 0.012, 0.013, 1.0), roughness=0.86),
        "floor": material("Warm white cutting mat", (0.50, 0.52, 0.50, 1.0), roughness=0.94),
        "grid": material("Cutting mat grid", (0.12, 0.15, 0.16, 1.0), roughness=0.88),
        "denim": material("Blue denim", (0.018, 0.065, 0.14, 1.0), roughness=0.96),
        "wall": material("Pale gray wall", (0.31, 0.33, 0.35, 1.0), roughness=0.96),
    }


def configure_render(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = args.samples
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.eevee.use_soft_shadows = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.65
    scene.view_settings.gamma = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Preview World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.40, 0.43, 0.48, 1.0)
    background.inputs["Strength"].default_value = 0.86


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    mats = build_materials()
    bpy.ops.import_scene.gltf(filepath=os.fspath(model_path))
    imported_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(imported_meshes) != 1:
        raise RuntimeError(f"Expected one Action 4 mesh in {model_path}, got {len(imported_meshes)}")
    base_mesh = imported_meshes[0]
    base_mesh.hide_render = True
    base_mesh.hide_viewport = True

    if args.scene == "raised":
        create_raised_background(mats)
        # Together with the local 180-degree roll, 210 degrees gives a
        # forward-facing rear screen and a forward/upward-facing top surface.
        camera_pitch = args.camera_pitch_deg if args.camera_pitch_deg is not None else 210.0
        create_wrist_assembly(
            base_mesh, mats, "left", (-0.190, -0.015, 0.455),
            (math.pi / 2, math.radians(-5.0), math.radians(-3.0)),
            args.pose, camera_pitch, 180.0, -0.018)
        create_wrist_assembly(
            base_mesh, mats, "right", (0.190, 0.005, 0.455),
            (math.pi / 2, math.radians(5.0), math.radians(3.0)),
            args.pose, camera_pitch, 180.0, -0.018)
    else:
        create_grid_floor(mats)
        create_denim(mats)
        camera_pitch = args.camera_pitch_deg if args.camera_pitch_deg is not None else 195.0
        create_wrist_assembly(
            base_mesh, mats, "left", (-0.185, 0.005, 0.0),
            (0.0, 0.0, math.radians(-7.0)), args.pose,
            camera_pitch, 180.0, -0.074)
        create_wrist_assembly(
            base_mesh, mats, "right", (0.185, 0.025, 0.0),
            (0.0, 0.0, math.radians(7.0)), args.pose,
            camera_pitch, 180.0, -0.074)

    scene = bpy.context.scene
    configure_render(scene, args)
    setup_camera_and_lights(scene, args.scene)
    if args.scene == "tabletop":
        viewer_location = scene.camera.matrix_world.translation
        for side in ("left", "right"):
            orient_action4_screen_to_viewer(
                bpy.data.objects[f"{side}_Action4_root"],
                bpy.data.objects[f"{side}_wrist_rig"],
                viewer_location,
                args.screen_tilt_deg,
            )

    image_path = output_dir / f"real_scene_{args.scene}_{args.pose}_{args.version_tag}.png"
    blend_path = output_dir / f"real_scene_{args.scene}_{args.pose}_{args.version_tag}.blend"
    scene.render.filepath = os.fspath(image_path)
    bpy.ops.wm.save_as_mainfile(filepath=os.fspath(blend_path))
    bpy.ops.render.render(write_still=True)
    print(f"render={image_path}")
    print(f"blend={blend_path}")


if __name__ == "__main__":
    main()
