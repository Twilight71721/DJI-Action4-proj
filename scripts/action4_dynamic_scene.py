"""Action-conditioned cloth, wrist, lighting and verified AprilTag assets.

Cloth states are kinematic deformations, not a cloth/contact simulation.
"""

import math
from pathlib import Path

import bpy
import cv2
import numpy as np
from mathutils import Euler, Vector

import blender_real_scene_preview as p

ACTIONS = ("button", "sleeve_grasp", "right_sleeve_fold", "left_sleeve_fold",
           "hem_fold", "press", "turnover", "spread")
LIGHTS = ("neutral_soft", "warm_side", "cool_dim", "backlit")


def standard_marker(root, side, output):
    """Replace the old invented glyph with the bit-exact observed tag family."""
    identifier = 18 if side == "left" else 19
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    code = cv2.aruco.generateImageMarker(dictionary, identifier, 512)
    code = np.pad(code, 64, constant_values=255)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(code)
    if ids is None or ids.ravel().tolist() != [identifier]:
        raise RuntimeError("Generated AprilTag failed independent decoder")
    path = Path(output) / f"apriltag_36h11_{identifier}.png"
    cv2.imwrite(str(path), code)
    # Suppress old procedural glyph objects, preserving the existing scene data.
    for obj in root.children:
        if obj.name.startswith(f"{side}_marker_"):
            obj.hide_render = True
    material = p.material(f"{side}_printed_AprilTag", (.9, .9, .86, 1), roughness=.72)
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Specular"].default_value = .17
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(path))
    texture.interpolation = "Linear"
    material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    mesh = bpy.data.meshes.new(f"{side}_paper_tag_mesh")
    # White quiet zone is part of the printed texture. Surface remains behind
    # the camera's rear normal (-Z), just above the separate LCD overlay.
    # Frame 000000: detected tag corners unprojected through the estimated
    # body pose. Paper includes a 25% white border beyond the black code.
    cx, cy, size = (.0171, .0041, .0296) if side == "left" else (.0203, .0046, .0277)
    mesh.from_pydata([(cx-size/2, cy-size/2, -.01624), (cx+size/2, cy-size/2, -.01624),
                      (cx+size/2, cy+size/2, -.01624), (cx-size/2, cy+size/2, -.01624)],
                     [], [(3, 2, 1, 0)])
    uv = mesh.uv_layers.new()
    coords = [(1, 0), (0, 0), (0, 1), (1, 1)]
    for loop in mesh.loops:
        uv.data[loop.index].uv = coords[loop.vertex_index]
    obj = bpy.data.objects.new(f"{side}_AprilTag_36h11_{identifier}", mesh)
    bpy.context.collection.objects.link(obj)
    p.parent_local(obj, root)
    p.assign_material(obj, material)
    solid = obj.modifiers.new("Paper thickness", "SOLIDIFY")
    solid.thickness = .00006
    return {"family": "AprilTag 36h11", "id": identifier,
            "texture": path.name, "decoded_texture_id": int(ids[0, 0])}


def deform_cloth(points, action, progress):
    q = np.array(points, dtype=float, copy=True)
    def sleeve(sign, amount):
        hinge = .19
        distance = np.maximum(sign * q[:, 0] - hinge, 0)
        # A finite bend region avoids a razor-sharp crease at the shoulder.
        angle = amount * math.pi * np.clip(distance / .045, 0, 1)
        q[:, 0] = np.where(distance > 0, sign * (hinge + distance * np.cos(angle)), q[:, 0])
        q[:, 2] += distance * np.sin(angle) + .008 * amount * (distance > 0)
    if action in ("right_sleeve_fold", "left_sleeve_fold", "press", "hem_fold"):
        if action != "left_sleeve_fold":
            sleeve(1, progress if action == "right_sleeve_fold" else .98)
        if action != "right_sleeve_fold":
            sleeve(-1, progress if action == "left_sleeve_fold" else .98)
    if action == "hem_fold":
        d = np.maximum(.29 - q[:, 1], 0)
        a = math.pi * progress * np.clip(d / .04, 0, 1)
        q[:, 1] = np.where(d > 0, .29 - d * np.cos(a), q[:, 1])
        q[:, 2] += d * np.sin(a) + .014 * progress * (d > 0)
    elif action == "button":
        influence = np.exp(-((q[:, 0] / .045)**2 + ((q[:, 1]-.40)/.10)**2))
        q[:, 2] += (.015 + .04 * progress) * influence
        q[:, 0] += np.sign(q[:, 0]) * .018 * progress * influence
    elif action == "sleeve_grasp":
        influence = np.exp(-(((q[:, 0]-.34)/.13)**2 + ((q[:, 1]-.16)/.16)**2))
        q[:, 2] += .12 * progress * influence
    elif action == "turnover":
        q[:, 2] += .07 * progress * np.exp(-((q[:, 1]-.25)/.23)**2)
    # Compress the garment under a pressing hand without making z negative.
    if action == "press":
        q[:, 2] *= .75 + .15 * (1-progress)
    return q


class ClothMotion:
    def __init__(self, objects):
        self.meshes, self.curves = [], []
        objects = list(objects)
        # Bake seam tubes once so cloth and stitching use exactly the same
        # deformation. Animated AUTO Bezier handles can otherwise stretch.
        for obj in list(objects):
            if obj.type != "CURVE":
                continue
            obj.data.resolution_u = 4
            obj.data.bevel_resolution = 1
            bpy.context.view_layer.update()
            mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            replacement = bpy.data.objects.new(obj.name + " deformable", mesh)
            bpy.context.collection.objects.link(replacement)
            replacement.matrix_world = obj.matrix_world.copy()
            obj.hide_render = True
            objects.remove(obj)
            objects.append(replacement)
        for obj in objects:
            if obj.type == "MESH":
                obj.shape_key_add(name="Basis")
                matrix = np.asarray(obj.matrix_world)
                local = np.asarray([v.co[:] for v in obj.data.vertices])
                world = (matrix @ np.column_stack([local, np.ones(len(local))]).T).T[:, :3]
                self.meshes.append((obj, world, np.linalg.inv(matrix)))
            elif obj.type == "CURVE":
                for spline in obj.data.splines:
                    for vertex in spline.bezier_points:
                        self.curves.append((obj, vertex, np.array(obj.matrix_world @ vertex.co)))

    def set_frame(self, action, progress, frame, frames):
        for obj, world, inverse in self.meshes:
            xyz = deform_cloth(world, action, progress)
            local = (inverse @ np.column_stack([xyz, np.ones(len(xyz))]).T).T[:, :3]
            key = obj.shape_key_add(name=f"{action}_{frame:03d}")
            for vertex, value in zip(key.data, local):
                vertex.co = value
            for tick in range(frames):
                key.value = float(tick == frame)
                key.keyframe_insert(data_path="value", frame=tick)
        for obj, vertex, world in self.curves:
            vertex.co = obj.matrix_world.inverted() @ Vector(deform_cloth([world], action, progress)[0])
            vertex.keyframe_insert(data_path="co", frame=frame)


def hand_pose(action, side):
    if action in ("press", "spread"):
        return "open"
    if action in ("right_sleeve_fold", "sleeve_grasp") and side == "left":
        return "open"
    return "pinch" if action == "button" else "grasp"


def pose_wrist(wrist, side, action, progress, frame, rng):
    sign = -1 if side == "left" else 1
    anchors = {
        "button": ((-.045, .37, .075), (.045, .37, .075)),
        "sleeve_grasp": ((-.08, .34, .065), (.34, .16, .055)),
        "right_sleeve_fold": ((-.08, .34, .065), (.35, .16, .055)),
        "left_sleeve_fold": ((-.35, .16, .055), (.07, .35, .065)),
        "hem_fold": ((-.12, .08, .055), (.12, .08, .055)),
        "press": ((-.09, .35, .06), (.07, .22, .06)),
        "turnover": ((-.18, .23, .09), (.18, .23, .09)),
        "spread": ((-.34, .16, .06), (.34, .16, .06)),
    }
    rolls = {"button": (-65, 55), "sleeve_grasp": (-35, 70),
             "right_sleeve_fold": (-15, 55), "left_sleeve_fold": (-60, 30),
             "hem_fold": (-35, 45), "press": (-5, 25),
             "turnover": (-115, 115), "spread": (-20, 20)}
    index = 0 if side == "left" else 1
    target = deform_cloth([anchors[action][index]], action, progress)[0]
    wrist.rotation_euler = Euler((rng.uniform(-.08, .08),
                                  math.radians(rolls[action][index] + rng.uniform(-12, 12)),
                                  math.radians(sign * 18 + rng.uniform(-8, 8))), "XYZ")
    hand = bpy.data.objects[f"{side}_anatomical_hand_forearm"]
    points = np.asarray([v.co[:] for v in hand.data.vertices])
    tip = points[points[:, 1] > points[:, 1].max()-.025].mean(axis=0)
    target += [rng.uniform(-.008, .008), rng.uniform(-.008, .008), .010]
    wrist.location = Vector(target) - wrist.rotation_euler.to_matrix() @ Vector(tip)
    wrist.keyframe_insert(data_path="location", frame=frame)
    wrist.keyframe_insert(data_path="rotation_euler", frame=frame)
    return {"side": side, "contact_anchor_m": target.tolist(),
            "wrist_euler_degrees": [math.degrees(v) for v in wrist.rotation_euler],
            "hand_pose": hand_pose(action, side)}


def lighting(scene, name, rng, frame):
    # Vary direction, source size, key/fill ratio, color and world intensity.
    profiles = {
        "neutral_soft": ((-.45, -.25, 1.25), 100, .9, (1, .94, .86), 38, .30),
        "warm_side": ((-.75, .15, .75), 160, .24, (1, .64, .34), 12, .12),
        "cool_dim": ((.65, -.1, 1.05), 42, .65, (.58, .76, 1), 10, .09),
        "backlit": ((.1, .95, .8), 190, .20, (1, .88, .70), 17, .12),
    }
    position, energy, size, color, fill, ambient = profiles[name]
    records = []
    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        if obj.name == "Large soft key":
            obj.location = Vector(position) + Vector(rng.uniform(-.1, .1, 3))
            obj.data.energy, obj.data.size, obj.data.color = energy*rng.uniform(.85, 1.15), size, color
        elif obj.name == "Cool fill":
            obj.data.energy = fill
        else:
            obj.data.energy = fill*.45
        p.aim_at(obj, (0, .28, .05))
        for prop in ("location", "rotation_euler"):
            obj.keyframe_insert(data_path=prop, frame=frame)
        for prop in ("energy", "size", "color"):
            obj.data.keyframe_insert(data_path=prop, frame=frame)
        records.append({"name": obj.name, "position_m": list(obj.location),
                        "energy": obj.data.energy, "size_m": obj.data.size, "color": list(obj.data.color)})
    world = scene.world.node_tree.nodes["Background"].inputs["Strength"]
    world.default_value = ambient
    world.keyframe_insert(data_path="default_value", frame=frame)
    return {"profile": name, "ambient_strength": ambient, "lights": records}
