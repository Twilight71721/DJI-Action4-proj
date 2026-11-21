"""Render a two-wrist scene with HCCEPose's vendored BlenderProc and BOP writer.

Uses the existing aligned GLB unchanged, MakeHuman CC0 hand geometry, and
reference-guided context assets. Real video pixels are never used as textures.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/Blender/vendor"
sys.path.insert(0, str(VENDOR))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("EGL_DEVICE_ID", "0")

import bpy
import cv2
import numpy as np
from mathutils import Euler, Vector, Quaternion

bpy.ops.preferences.addon_enable(module="cycles")
import blenderproc as bproc
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.writer.WriterUtility import _WriterUtility

import blender_real_scene_preview as p
import action4_dynamic_scene as dynamic
import action4_cable_geometry as cable_geometry
import action4_reference_profiles as reference
from action4_wrist_assets import (anatomical_hand, detail_materials, folded_denim,
                                 live_screen, reconnect_mount_and_cable, fitted_cuff_and_mount, orient_hand_for_tabletop)


def args_from_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=p.DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--scene", choices=("raised", "tabletop"), default="raised")
    parser.add_argument("--pose", choices=("open", "grasp", "pinch", "fist", "thumbs_up"), default="open")
    parser.add_argument("--width", type=int, default=1218)
    parser.add_argument("--height", type=int, default=924)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--reference-profile", type=int, choices=range(len(reference.SPECS)))
    parser.add_argument("--reference-spec", type=Path)
    parser.add_argument("--variation", choices=("reference", "nearby", "hard"), default="reference")
    parser.add_argument("--light-offset", type=int)
    parser.add_argument("--extended-checks", action="store_true")
    parser.add_argument("--action", choices=dynamic.ACTIONS)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    if args.action and args.scene != "tabletop":
        parser.error("action-conditioned scenes require --scene tabletop")
    return args


def make_screen_capture(output, scene, mode, seed, device_root=None):
    """Generate the LCD view in Blender before adding its surface and UI."""
    main_camera = scene.camera
    data = bpy.data.cameras.new("On-device synthetic live view")
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    # A separate virtual view of the synthetic hands/background supplies the
    # display content. It is an approximation, not a recording from the video.
    if device_root is not None:
        camera.location = device_root.matrix_world @ Vector((.015, .003, .019))
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = device_root.matrix_world.to_quaternion() @ Quaternion((0,1,0), math.pi)
        data.clip_start = .001
    elif mode == "raised":
        camera.location = (0.18, -0.35, 0.52)
        p.aim_at(camera, (0.11, 0.045, 0.61))
    else:
        camera.location = (0.14, -0.24, 0.29)
        p.aim_at(camera, (0.03, 0.23, 0.015))
    data.lens = 15
    scene.camera = camera
    width, height = scene.render.resolution_x, scene.render.resolution_y
    samples = scene.cycles.samples
    scene.render.resolution_x, scene.render.resolution_y = 384, 216
    scene.cycles.samples = 12
    screen_path = output / f"screen_{seed}.png"
    scene.render.filepath = str(screen_path)
    bpy.ops.render.render(write_still=True)
    frame = cv2.imread(str(screen_path))
    if frame is None:
        raise RuntimeError("Synthetic screen capture failed")
    cv2.putText(frame, "100%", (293, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (238, 240, 238), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (338, 10), (366, 22), (125, 230, 155), 1)
    cv2.rectangle(frame, (341, 13), (362, 19), (100, 205, 115), -1)
    cv2.circle(frame, (359, 108), 9, (240, 241, 235), 1, cv2.LINE_AA)
    cv2.line(frame, (353, 108), (365, 108), (235, 235, 235), 1)
    cv2.putText(frame, "4K  30  RS", (125, 203), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (55, 210, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(screen_path), frame)
    scene.camera = main_camera
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.cycles.samples = samples
    bpy.data.objects.remove(camera, do_unlink=True)
    return screen_path


def main():
    args = args_from_cli()
    spec = reference.profile(args.reference_profile) if args.reference_profile is not None else None
    if args.reference_spec:
        spec=json.loads(args.reference_spec.read_text())
    if spec and args.extended_checks:
        spec.update(variation=args.variation,variation_scale={"reference":1.0,"nearby":1.5,"hard":2.0}[args.variation])
    if spec:
        args.scene = spec["scene"]
        args.action = spec["cloth_action"] if spec["scene"] == "tabletop" else None
        args.pose = "thumbs_up" if spec["action"] == "thumbs_up" else "open"
    start = time.monotonic()
    output = args.output.resolve()
    work = output / "render_metadata" / f"batch_{args.batch:03d}"
    work.mkdir(parents=True, exist_ok=False)
    Utility.temp_dir = str(work / "temporary")
    Path(Utility.temp_dir).mkdir()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    bproc.init()
    bproc.renderer.set_render_devices(desired_gpu_device_type="OPTIX", desired_gpu_ids=[0])
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.035
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -1.10
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.62, 0.65, 0.68, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3
    mats = p.build_materials()
    detail_materials(mats)
    loaded = bproc.loader.load_obj(str(args.model.resolve()))
    if len(loaded) != 1:
        raise RuntimeError("The aligned GLB must contain exactly one mesh")
    base = loaded[0].blender_obj
    sys.path.insert(0, str(ROOT / "dataset/code"))
    from action4_domain_v6 import darken_target_materials
    body_factor = .12 if spec else .20
    darken_target_materials(loaded[0], factor=body_factor)
    for mat in base.data.materials:
        shader = mat.node_tree.nodes.get("Principled BSDF")
        if shader:
            shader.inputs["Specular"].default_value = .20
    base.hide_render = base.hide_viewport = True
    vertices = len(base.data.vertices)
    faces = len(base.data.polygons)
    if (vertices, faces) != (915032, 1499206):
        raise RuntimeError(f"Unexpected GLB topology: {vertices} vertices, {faces} faces")
    cloth_motion = None
    if args.scene == "raised":
        p.create_grid_floor(mats, surface_z=-0.11)
    else:
        p.create_grid_floor(mats)
        before = set(scene.objects)
        if spec:
            from action4_tailored_denim import create_garment
            create_garment(mats, args.seed)
        else:
            folded_denim(mats, args.seed)
        if args.action:
            bpy.context.view_layer.update()
            cloth_motion = dynamic.ClothMotion(set(scene.objects) - before)
    p.rounded_box("Pale wall", (5, 0.035, 3), (0, 1.25, 1.1), mats["wall"], bevel=0)
    p.setup_camera_and_lights(scene, args.scene)
    if args.scene == "raised":
        scene.camera.location = (0, -0.57, 0.51)
    scene.camera.data.dof.use_dof = False
    # Keep the real camera's broad field of view instead of the old close-up
    # orthographic-looking preview. Focal length is recorded per frame.
    scene.camera.data.lens = 25 if args.scene == "raised" else 28
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.data.energy *= 0.7
            obj.data.use_shadow = True
    old_hand_builder = p.create_hand
    def hand_builder(parent, materials, pose, side):
        hand = anatomical_hand(parent, materials, pose, side)
        if args.action and not spec:
            for vertex in hand.data.vertices:
                vertex.co.y -= 4 * max(-vertex.co.y - .12, 0)
        if spec:
            from action4_wrist_assets import shape_reference_forearm
            shape_reference_forearm(hand, side)
        if args.scene == "tabletop":
            orient_hand_for_tabletop(hand)
        return hand
    p.create_hand = hand_builder
    p.create_cuff_and_mount = fitted_cuff_and_mount
    objects, wrists, roots, connections = [], [], [], []
    separation = float(rng.uniform(0.12, 0.145))
    for index, side in enumerate(("left", "right")):
        sign = -1 if index == 0 else 1
        location = (sign * separation, rng.uniform(-0.005, 0.025), 0.45 if args.scene == "raised" else 0.065)
        rotation = (math.pi / 2 if args.scene == "raised" else rng.uniform(-0.15, 0.08),
                    math.radians(sign * 5), math.radians(sign * (7 if args.scene == "raised" else 23)))
        pose = dynamic.hand_pose(spec["hand_action"] if spec else args.action, side) if args.action else args.pose
        wrist = p.create_wrist_assembly(base, mats, side, location, rotation, pose,
                                       210, 180, 0.054 if args.scene == "raised" else -0.04)
        root = bpy.data.objects[f"{side}_Action4_root"]
        # The reference marker covers roughly one third of the display width.
        marker = bpy.data.objects[f"{side}_marker_back"]
        marker.dimensions = (.024, .024, .00010)
        marker.location.x = .0155
        marker.location.z = -.01615
        for decal in list(root.children):
            if decal.name.startswith(f"{side}_marker_") and decal != marker:
                decal.location.x = .0155 + (decal.location.x - .019) * 1.35
                decal.location.y = -.0005 + (decal.location.y + .0005) * 1.35
                decal.location.z = -.01623
                decal.scale.x = decal.scale.y = 1.55
        label_text = bpy.data.objects[f"{side}_top_label_text"]
        label_text.rotation_euler = (-math.pi / 2, math.pi, 0)
        red = p.material(f"{side}_record_button_red_ring", (.30, .009, .012, 1), roughness=.72)
        bpy.ops.mesh.primitive_torus_add(major_radius=.0055, minor_radius=.00045,
                                        major_segments=48, minor_segments=10)
        ring = bpy.context.object
        ring.name = f"{side}_top_button_red_ring"
        p.assign_material(ring, red)
        p.parent_local(ring, root, (-.019, .0217, -.007), (math.pi / 2, 0, 0))
        bpy.context.view_layer.update()
        if spec:
            root.location = (0,0,.13)
        elif args.action:
            # Fixed fixture relative to the wrist: no look-at constraint.
            # Mirror the fixture roll so both cable sockets exit away from the arm.
            root.rotation_quaternion = Euler((math.radians(200), math.radians(sign*12), math.pi if side == "left" else 0), "XYZ").to_quaternion()
            root.location.x = sign * .048
            root.location.z = .080
        else:
            p.orient_action4_screen_to_viewer(root, wrist, scene.camera.location,
                                             rng.uniform(14, 24) if args.scene == "raised" else rng.uniform(14, 35))
        if args.scene == "tabletop" and not args.action:
            root.rotation_quaternion = root.rotation_quaternion @ Quaternion((0, 1, 0),
                                       math.radians(sign * rng.uniform(20, 60)))
        dynamic.standard_marker(root, side, work)
        connection = None if spec else reconnect_mount_and_cable(side, mats)
        body = bpy.data.objects[f"{side}_Action4_body"]
        obj = bproc.types.MeshObject(body)
        obj.set_cp("category_id", 1)
        obj.set_cp("is_bop_object", True)
        obj.set_cp("bop_dataset_name", output.name)
        obj.set_cp("model_path", str(args.model.resolve()))
        objects.append(obj)
        wrists.append(wrist)
        roots.append(root)
        connections.append(connection)
    p.create_hand = old_hand_builder
    if spec:
        for root, wrist, side in zip(roots, wrists, ("left", "right")):
            reference.pose_wrist(wrist, side, spec, spec["progress"], 0, rng)
            mount_anchor = reference.fit_wrist_assembly(root, wrist, side, spec, mats)
            reconnect_mount_and_cable(side, mats, mount_anchor)
    obstacles = {side: cable_geometry.hand_obstacle(side) for side in ("left", "right")}
    cuff_report = reference.audit_cuffs(obstacles) if spec else None
    mount_report = reference.audit_mount_parts(obstacles) if spec else None
    bottom_mount_report = reference.audit_bottom_mount() if spec else None
    extended_geometry=None
    if args.extended_checks:
        import action4_extended_checks as extended
        extended_geometry=extended.prepare()
    routes = [cable_geometry.route_cable(side, mats, obstacles[side],extended_geometry) for side in ("left", "right")]
    bpy.context.view_layer.update()
    original_locations = [w.location.copy() for w in wrists]
    original_rotations = [w.rotation_euler.copy() for w in wrists]
    original_camera = scene.camera.location.copy()
    if args.scene == "tabletop":
        original_camera += Vector((rng.uniform(-.03, .03), rng.uniform(.01, .12), rng.uniform(-.08, .01)))
    corners = np.asarray([c["xyz"] for c in json.loads((args.model.parent / "corners_3d.json").read_text())["corners"]])
    metadata = []
    for frame in range(args.frames):
        action_metadata, light_metadata = None, None
        if args.action:
            progress = float(np.clip(spec["progress"] + rng.uniform(-.07,.07)*spec.get('variation_scale',1),0,1)) if spec else .25 + .74 * frame / max(args.frames-1, 1)
            cloth_motion.set_frame(args.action, progress, frame, args.frames)
            action_metadata = {"name": args.action, "progress": progress,
                               "cloth_method": "kinematic deformation, no contact solver", "hands": []}
        for index, wrist in enumerate(wrists):
            if spec:
                hand_metadata = reference.pose_wrist(wrist, ("left", "right")[index], spec, progress if args.action else 0, frame, rng)
                if action_metadata:
                    action_metadata["hands"].append(hand_metadata)
            elif args.action:
                action_metadata["hands"].append(dynamic.pose_wrist(wrist, ("left", "right")[index], args.action, progress, frame, rng))
            else:
                wrist.location = original_locations[index] + Vector(rng.uniform([-0.025, -0.03, -0.015], [0.025, 0.03, 0.025]))
                wrist.rotation_euler = Euler(np.asarray(original_rotations[index]) + rng.uniform(-0.14, 0.14, 3), "XYZ")
                wrist.keyframe_insert(data_path="location", frame=frame)
                wrist.keyframe_insert(data_path="rotation_euler", frame=frame)
        scene.camera.location = original_camera + Vector(rng.uniform([-0.025, -0.075, -0.025], [0.025, 0.075, 0.025]))
        target = (0, 0.005, 0.555) if args.scene == "raised" else (0, 0.16, 0.065)
        p.aim_at(scene.camera, target)
        if args.action:
            scene.camera.location = Vector((rng.uniform(-.08, .08), rng.uniform(-.55, -.42), rng.uniform(.80, .96)))
            p.aim_at(scene.camera, (0, .25, .04))
            scene.camera.rotation_euler.rotate_axis("Z", math.radians(rng.uniform(-12, 12)))
        scene.camera.data.lens = float(rng.uniform(25, 33) if args.scene == "raised" else rng.uniform(29, 38))
        if spec:
            reference.observer(scene, spec, rng)
        scene.camera.data.keyframe_insert(data_path="lens", frame=frame)
        bpy.context.view_layer.update()
        camera_pose = np.asarray(scene.camera.matrix_world).copy()
        bproc.camera.add_camera_pose(camera_pose, frame=frame)
        for light in (o for o in scene.objects if o.type == "LIGHT"):
            baseline = {"Large soft key": 66.5, "Cool fill": 29.4, "Top bounce": 21}[light.name]
            light.data.energy = baseline * float(rng.uniform(0.85, 1.15))
            light.data.keyframe_insert(data_path="energy", frame=frame)
        if args.action or spec:
            light_metadata = dynamic.lighting(scene, dynamic.LIGHTS[(frame + (args.batch if args.light_offset is None else args.light_offset)) % len(dynamic.LIGHTS)], rng, frame)
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        rejected_proposals = []
        for attempt in range(25):
            try:
                cable_report = cable_geometry.audit_frame(routes, obstacles)
                fixture_report = reference.audit_fixture(obstacles) if spec else None
                lens_report = reference.lens_direction_report(spec) if spec else None
                hand_report = reference.audit_handedness(spec) if spec else None
                extended_report=extended.audit(extended_geometry,routes) if args.extended_checks else None
                break
            except RuntimeError as error:
                if not spec or attempt == 24:
                    raise
                rejected_proposals.append(str(error))
                for index, wrist in enumerate(wrists):
                    hand_metadata = reference.pose_wrist(wrist, ("left", "right")[index], spec, progress if args.action else 0, frame, rng)
                    if action_metadata:
                        action_metadata["hands"][index] = hand_metadata
                scene.frame_set(frame)
                bpy.context.view_layer.update()
        K = bproc.camera.get_intrinsics_as_K_matrix()
        instance_metadata = []
        for obj, side in zip(objects, ("left", "right")):
            T = np.diag([1, -1, -1, 1]) @ np.linalg.inv(camera_pose) @ np.asarray(obj.blender_obj.matrix_world)
            camera_points = (T @ np.column_stack([corners, np.ones(8)]).T).T[:, :3]
            uvw = (K @ camera_points.T).T
            uv = uvw[:, :2] / uvw[:, 2:3]
            if not np.all(camera_points[:, 2] > 0):
                raise RuntimeError("Target behind camera")
            instance_metadata.append({"side": side, "corners_2d_independent": uv.tolist(),
                                      "camera_distance_m": float(np.linalg.norm(T[:3, 3])),
                                      "mesh_identity_preserved": obj.blender_obj.data is base.data})
            rotation = obj.blender_obj.matrix_world.to_quaternion()
            view = (scene.camera.location - obj.blender_obj.matrix_world.translation).normalized()
            facing = {name: float((rotation @ Vector(normal)).dot(view)) for name, normal in
                      (("rear_screen", (0,0,-1)), ("front_lens", (0,0,1)), ("top", (0,1,0)),
                       ("bottom_mount", (0,-1,0)), ("side_left", (-1,0,0)), ("side_right", (1,0,0)))}
            instance_metadata[-1]["dominant_view"] = max(facing, key=facing.get)
            instance_metadata[-1]["face_view_cosines"] = facing
        frame_pose = args.pose if not args.action else "/".join(dynamic.hand_pose(args.action, s) for s in ("left", "right"))
        metadata.append({"frame": frame, "scene": args.scene, "hand_pose": frame_pose,
                         "seed": args.seed, "cam_K": K.tolist(), "instances": instance_metadata,
                         "action": action_metadata, "lighting": light_metadata,
                         "cable_clearance": cable_report, "fixture_clearance": fixture_report, "lens_direction": lens_report, "handedness": hand_report, "cuff_clearance": cuff_report, "mount_clearance": mount_report, "bottom_mount_clearance": bottom_mount_report,
                         "reference_profile": spec, "rejected_geometry_proposals": rejected_proposals})
        if args.extended_checks:
            metadata[-1]['extended_geometry_checks']=extended_report
    for action in bpy.data.actions:
        for curve in action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = "CONSTANT"
    scene.frame_set(0)
    bpy.context.view_layer.update()
    # Both LCDs display the device's own virtual lens view. Only the rear
    # receives a paper tag; the front lens and its geometry remain untouched.
    for index, (root, side) in enumerate(zip(roots, ("left", "right"))):
        screen_path = make_screen_capture(work, scene, args.scene, args.seed+index, root)
        live_screen(root, side, screen_path, "rear")
        screen = cv2.imread(str(screen_path))
        height, width = screen.shape[:2]
        front_path = work / f"front_screen_{side}.png"
        cv2.imwrite(str(front_path), screen[:, (width-height)//2:(width+height)//2])
        live_screen(root, side, front_path, "front")
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(work / "scene.blend"))
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.render(output_dir=str(work / "temporary"), return_data=False)
    depth_entry = Utility.find_registered_output_by_key("depth")
    depths = []
    for frame in range(args.frames):
        raw = cv2.imread(depth_entry["path"] % frame, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32:
            raise RuntimeError("Expected floating-point EXR depth")
        raw = raw[:, :, 0] if raw.ndim == 3 else raw
        depths.append(np.where(np.isfinite(raw) & (raw > 0) & (raw < 6.5), raw, 0).astype(np.float32))
    colors = _WriterUtility.load_registered_outputs({"colors"})["colors"]
    bproc.writer.write_bop(str(output.parent), dataset=output.name, target_objects=objects,
                          depths=depths, colors=colors, depth_scale=0.1,
                          color_file_format="PNG", frames_per_chunk=args.frames,
                          calc_mask_info_coco=True, num_worker=0, delta=1)
    report = {"pipeline": "HCCEPose vendored BlenderProc / Cycles OPTIX / native BOP writer",
              "blenderproc_path": str(Path(bproc.__file__).resolve()),
              "glb_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "glb_vertices": vertices, "glb_faces": faces, "glb_textures_preserved": True,
              "body_base_color_multiplier": body_factor, "body_specular": 0.20,
              "resolution": [args.width, args.height], "samples": args.samples,
              "reference_pixels_used": False, "screen_source": "per-device synthetic lens view at frame 0 + UI; held fixed within batch",
              "display_faces": ["front_LCD_only_no_tag", "rear_LCD_with_partial_paper_tag"],
              "hand_source": "MakeHuman CC0 base mesh, cropped and posed",
              "action": args.action, "reference_profile": spec,
              "marker_family": "AprilTag 36h11; rear display only",
              "elapsed_seconds": time.monotonic() - start, "frames": metadata}
    (work / "metadata.json").write_text(json.dumps(report, indent=2))
    print("BATCH_COMPLETE", args.batch, len(colors), "seconds", report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
