"""Reference-guided context assets; the target GLB is never remeshed."""

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

import blender_real_scene_preview as preview

ROOT = Path(__file__).resolve().parents[1]


def read_hand_source():
    vertices, faces, groups, uv_coords, uv_groups = [], [], {}, [], {}
    group = ""
    for line in (ROOT / "assets/action4_realism/makehuman/base.obj").read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(list(map(float, fields[1:4])))
        elif fields[0] == "vt":
            uv_coords.append(list(map(float, fields[1:3])))
        elif fields[0] == "g":
            group = fields[1]
            groups[group] = []
            uv_groups[group] = []
        elif fields[0] == "f":
            face = [int(f.split("/")[0]) - 1 for f in fields[1:]]
            groups[group].append(face)
            uv_groups[group].append([int(f.split("/")[1]) - 1 for f in fields[1:]])
    vertices = np.asarray(vertices)
    joints = {k: vertices[np.unique(np.asarray(f).ravel())].mean(axis=0)
              for k, f in groups.items() if k.startswith("joint-l-")}
    wrist = joints["joint-l-hand"]
    elbow = joints["joint-l-elbow"]
    axis_y = wrist - elbow
    axis_y /= np.linalg.norm(axis_y)
    axis_x = joints["joint-l-finger-5-1"] - joints["joint-l-finger-2-1"]
    axis_x -= axis_y * axis_x.dot(axis_y)
    axis_x /= np.linalg.norm(axis_x)
    axis_z = np.cross(axis_x, axis_y)
    basis = np.stack([axis_x, axis_y, axis_z], axis=1)
    local = (vertices - wrist) @ basis * 0.1
    # Keep only the left hand and forearm beyond a cut near the elbow.
    selected = ((vertices[:, 0] > 2.5) & (local[:, 1] > -0.25)
                & (local[:, 1] < .23) & (np.abs(local[:, 0]) < .12)
                & (np.abs(local[:, 2]) < .12))
    face_indices = [i for i, f in enumerate(groups["body"]) if selected[f].all()]
    faces = [groups["body"][i] for i in face_indices]
    face_uvs = [[uv_coords[j] for j in uv_groups["body"][i]] for i in face_indices]
    ids = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(ids)}
    local = local[ids]
    local_joints = {k: (p - wrist) @ basis * 0.1 for k, p in joints.items()}
    return local, [[remap[i] for i in f] for f in faces], local_joints, face_uvs


def anatomical_hand(parent, mats, pose, side):
    vertices, faces, joints, face_uvs = read_hand_source()
    landmark_indices = {}
    for name, key in (("thumb", "joint-l-finger-1-1"), ("little", "joint-l-finger-5-1")):
        landmark_indices[name] = np.argsort(np.linalg.norm(vertices-joints[key],axis=1))[:6].tolist()
    # Finger rotation is blended only above the MCP joints. Keep the shared
    # palm/forearm mesh continuous rather than assembling capsule fingers.
    finger_bases = np.asarray([joints[f"joint-l-finger-{n}-1"] for n in range(1, 6)])
    finger_tips = np.asarray([joints[f"joint-l-finger-{n}-4"] for n in range(1, 6)])
    distances = []
    for base, tip in zip(finger_bases, finger_tips):
        d = tip - base
        t = np.clip((vertices - base) @ d / d.dot(d), 0, 1)
        distances.append(np.linalg.norm(vertices - base - t[:, None] * d, axis=1))
    assignments = np.argmin(np.stack(distances, axis=1), axis=1)
    for index, (base, tip) in enumerate(zip(finger_bases, finger_tips)):
        selected = assignments == index
        direction = (tip - base) / np.linalg.norm(tip - base)
        source = vertices[selected].copy()
        chain = np.asarray([joints[f"joint-l-finger-{index + 1}-{j}"] for j in range(1, 5)])
        axis = np.cross(direction, np.array([0., 0., 1.]))
        axis /= np.linalg.norm(axis)
        angles = {"open": [0.2, 0.25, 0.12], "grasp": [0.65, 1.0, 0.6],
                  "pinch": [0.5, 0.8, 0.4], "fist": [0.9, 1.2, 0.65],
                  "thumbs_up": [1.4, 1.6, 1.0]}[pose]
        if index == 0:
            angles = [0, 0, 0] if pose in ("open", "thumbs_up") else [0.25, 0.45, 0.3]
        elif pose == "pinch" and index > 1:
            angles = [0.2, 0.4, 0.2]
        posed_chain = chain.copy()
        points = source.copy()
        for j, angle in enumerate(angles):
            weight = np.clip((source - chain[j]) @ direction / 0.010 + 0.3, 0, 1)
            a = angle * weight
            relative = points - posed_chain[j]
            points = (relative * np.cos(a)[:, None]
                      + np.cross(axis, relative) * np.sin(a)[:, None]
                      + np.outer(relative @ axis, axis) * (1 - np.cos(a))[:, None]
                      + posed_chain[j])
            relative_chain = posed_chain[j + 1:] - posed_chain[j]
            posed_chain[j + 1:] = (relative_chain * math.cos(angle)
                                  + np.cross(axis, relative_chain) * math.sin(angle)
                                  + np.outer(relative_chain @ axis, axis) * (1 - math.cos(angle))
                                  + posed_chain[j])
        if index == 0 and pose == "thumbs_up":
            a = -1.0 * np.clip((source-base) @ direction / .025, 0, 1)
            relative = points-base
            x, y = relative[:,0].copy(), relative[:,1].copy()
            relative[:,0] = np.cos(a)*x-np.sin(a)*y
            relative[:,1] = np.sin(a)*x+np.cos(a)*y
            points = relative+base
        vertices[selected] = points
    # Continue the proximal forearm beyond the image; no cut cap is visible.
    vertices[:, 1] -= 1.5 * np.maximum(-vertices[:, 1] - 0.08, 0)
    vertices[:, 0] *= 1.20
    vertices[:, 2] *= 1.15
    # A relaxed extended wrist places the fingers behind the forward-mounted
    # camera, matching the raised reference rather than a flat palm billboard.
    weight = np.clip(vertices[:, 1] / 0.045, 0, 1)
    bend = -0.45 * weight if pose == "open" else -0.15 * weight
    y, z = vertices[:, 1].copy(), vertices[:, 2].copy()
    vertices[:, 1] = np.cos(bend) * y - np.sin(bend) * z
    vertices[:, 2] = np.sin(bend) * y + np.cos(bend) * z
    vertices[:, 1] += 0.032
    vertices[:, 2] += 0.060
    if side == "right":
        vertices[:, 0] *= -1
        faces = [list(reversed(f)) for f in faces]
        face_uvs = [list(reversed(f)) for f in face_uvs]
    mesh = bpy.data.meshes.new(f"{side}_MakeHuman_hand_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="MakeHuman UV")
    for polygon, coordinates in zip(mesh.polygons, face_uvs):
        for loop_index, coordinate in zip(polygon.loop_indices, coordinates):
            uv.data[loop_index].uv = coordinate
    obj = bpy.data.objects.new(f"{side}_anatomical_hand_forearm", mesh)
    bpy.context.collection.objects.link(obj)
    preview.parent_local(obj, parent)
    obj["anatomical_side"] = side
    obj["thumb_landmark_vertices"] = landmark_indices["thumb"]
    obj["little_landmark_vertices"] = landmark_indices["little"]
    preview.assign_material(obj, mats["skin"])
    preview.smooth(obj)
    sub = obj.modifiers.new("Anatomical surface subdivision", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    return obj


def detail_materials(mats):
    skin = mats["skin"]
    nodes, links = skin.node_tree.nodes, skin.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Subsurface"].default_value = 0.045
    shader.inputs["Subsurface Radius"].default_value = (0.006, 0.0025, 0.0012)
    shader.inputs["Roughness"].default_value = 0.52
    shader.inputs["Specular"].default_value = 0.3
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18
    noise.inputs["Detail"].default_value = 3
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.31, 0.17, 0.10, 1)
    ramp.color_ramp.elements[1].color = (0.48, 0.31, 0.20, 1)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs[0], shader.inputs["Base Color"])
    skin_path = ROOT / "assets/action4_realism/makehuman/skin_diffuse.png"
    if skin_path.is_file():
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(skin_path))
        tint = nodes.new("ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        tint.inputs[0].default_value = 1
        tint.inputs[2].default_value = (0.43, 0.34, 0.25, 1)
        links.new(tex.outputs["Color"], tint.inputs[1])
        links.new(tint.outputs[0], shader.inputs["Base Color"])
    pores = nodes.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = 1000
    pores.inputs["Detail"].default_value = 2
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.00015
    links.new(pores.outputs["Fac"], bump.inputs["Height"])
    normal_path = ROOT / "assets/action4_realism/makehuman/skin_normal.png"
    if normal_path.is_file():
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(normal_path))
        tex.image.colorspace_settings.name = "Non-Color"
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = 0.6
        links.new(tex.outputs["Color"], normal.inputs["Color"])
        links.new(normal.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    for name in ("fabric", "denim"):
        mat = mats[name]
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        shader = nodes.get("Principled BSDF")
        shader.inputs["Sheen"].default_value = 0.35
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 650
        noise.inputs["Detail"].default_value = 2
        wave = nodes.new("ShaderNodeTexWave")
        wave.bands_direction = "DIAGONAL"
        wave.inputs["Scale"].default_value = 90 if name == "fabric" else 200
        wave.inputs["Distortion"].default_value = 2
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.28
        bump.inputs["Distance"].default_value = 0.00035
        links.new(wave.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
        ramp = nodes.new("ShaderNodeValToRGB")
        colors = ((0.025, 0.038, 0.055, 1), (0.11, 0.145, 0.185, 1)) if name == "denim" else (
            (0.08, 0.09, 0.095, 1), (0.2, 0.22, 0.23, 1))
        for element, color in zip(ramp.color_ramp.elements, colors):
            element.color = color
        links.new(noise.outputs["Fac"], ramp.inputs[0])
        links.new(ramp.outputs[0], shader.inputs["Base Color"])


def folded_denim(mats, seed):
    rng = np.random.default_rng(seed)
    nx, ny = 51, 61
    vertices, faces = [], []
    phase = rng.uniform(0, math.tau, 3)
    for j in range(ny):
        y = 0.06 + j / (ny - 1) * 0.61
        for i in range(nx):
            x = (i / (nx - 1) - 0.5) * 0.46
            z = (0.012 + 0.018 * (1 + math.sin(25 * x + 10 * y + phase[0]))
                 * math.exp(-((y - 0.31) / 0.27) ** 2)
                 + 0.01 * math.sin(47 * x - 18 * y + phase[1]) ** 2
                 + 0.026 * math.exp(-((x + 0.1) / 0.06) ** 2))
            vertices.append((x + 0.015 * math.sin(y * 24), y, z))
            if i and j and not (y > 0.60 and abs(x) < 0.08):
                a = j * nx + i
                faces.append((a - nx - 1, a - nx, a, a - 1))
    mesh = bpy.data.meshes.new("Folded denim topology")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Folded blue garment", mesh)
    bpy.context.collection.objects.link(obj)
    preview.assign_material(obj, mats["denim"])
    preview.smooth(obj)
    sub = obj.modifiers.new("Cloth subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 1
    thick = obj.modifiers.new("Cloth edge thickness", "SOLIDIFY")
    thick.thickness = 0.0013
    # Sewn parallel seams follow the displaced surface rather than float above it.
    thread = preview.material("Denim stitching", (0.18, 0.14, 0.07, 1), roughness=0.9)
    anchor = bpy.data.objects.new("Cloth stitches root", None)
    bpy.context.collection.objects.link(anchor)
    for column in (3, 4, nx - 5, nx - 4):
        points = [(vertices[j * nx + column][0], vertices[j * nx + column][1],
                   vertices[j * nx + column][2] + 0.001) for j in range(0, ny, 2)]
        preview.curve_tube(f"Denim seam {column}", points, 0.00035, thread, anchor)
    def surface_z(x, y):
        return (0.012 + 0.018 * (1 + math.sin(25 * x + 10 * y + phase[0]))
                * math.exp(-((y - 0.31) / 0.27) ** 2)
                + 0.01 * math.sin(47 * x - 18 * y + phase[1]) ** 2
                + 0.026 * math.exp(-((x + 0.1) / 0.06) ** 2))

    def panel(name, coordinates, mat):
        panel_mesh = bpy.data.meshes.new(name)
        panel_mesh.from_pydata(coordinates, [], [tuple(range(len(coordinates)))])
        panel_mesh.update()
        part = bpy.data.objects.new(name, panel_mesh)
        bpy.context.collection.objects.link(part)
        preview.assign_material(part, mat)
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(panel_mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=9, use_grid_fill=True)
        if "sleeve" in name:
            for v in bm.verts:
                v.co.z = surface_z(v.co.x, v.co.y) + .004
        elif "pocket" in name:
            for v in bm.verts:
                v.co.z = surface_z(v.co.x, v.co.y) + .003
        else:
            for v in bm.verts:
                v.co.z += .002 * math.sin(v.co.x * 72 + v.co.y * 40)
        bm.to_mesh(panel_mesh)
        bm.free()
        preview.smooth(part)
        solid = part.modifiers.new("Fabric thickness", "SOLIDIFY")
        solid.thickness = 0.002
        return part

    # Shirt front details distinguish this from a featureless blue sheet.
    for sign in (-1, 1):
        sleeve = []
        for x, y in [(sign * .20, .53), (sign * .30, .52),
                     (sign * .40, .14), (sign * .30, .12), (sign * .18, .40)]:
            sleeve.append((x, y, surface_z(x, y) + .005))
        panel(f"Denim sleeve {sign}", sleeve, mats["denim"])
        preview.curve_tube(f"Sleeve cuff sewing {sign}",
                           [(sign * (.30 + .10 * t), .125 + .02 * t,
                             surface_z(sign * (.30 + .10 * t), .125 + .02 * t) + .007)
                            for t in np.linspace(0, 1, 24)], .00035, thread, anchor)
        coords = [(sign * .085, .56, .06), (sign * .012, .54, .075),
                  (sign * .01, .65, .085), (sign * .085, .67, .07), (sign * .12, .60, .055)]
        panel(f"Folded shirt collar {sign}", coords, mats["denim"])
        x0 = sign * .125
        pocket = [(x0 + x, y, surface_z(x0 + x, y) + .004)
                  for x, y in [(-.046, .47), (.046, .47), (.043, .36), (0, .35), (-.043, .36)]]
        panel(f"Chest pocket {sign}", pocket, mats["denim"])
        preview.curve_tube(f"Pocket stitching {sign}", pocket + [pocket[0]], .00035, thread, anchor)
    for xx in (-.013, .013):
        preview.curve_tube(f"Shirt button placket {xx}",
                           [(xx, y, surface_z(xx, y) + .004) for y in np.linspace(.07, .56, 45)],
                           .0005, thread, anchor)
    button = preview.material("Brushed shirt buttons", (.23, .24, .20, 1), roughness=.48, metallic=.55)
    for j, y in enumerate(np.linspace(.10, .54, 7)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=.004, depth=.0018,
                                           location=(0, y, surface_z(0, y) + .004))
        obj_button = bpy.context.object
        obj_button.name = f"Shirt button {j}"
        preview.assign_material(obj_button, button)
    return obj


def fitted_cuff_and_mount(parent, mats, side):
    from action4_cable_geometry import hand_obstacle
    tree, _ = hand_obstacle(side)
    center_z = .060
    n, rings = 128, 9
    ymin, ymax = -.061, .009
    def skin_point(y, angle, offset):
        direction = Vector((math.cos(angle), 0, math.sin(angle)))
        center = Vector((0, y, center_z))
        hit, _, _, distance = tree.ray_cast(center, direction)
        if hit is None or not .008 < distance < .10:
            raise RuntimeError(f"Cannot fit {side} cuff to actual wrist cross-section")
        return center + direction*(distance + offset)
    vertices, faces = [], []
    for y in np.linspace(ymin, ymax, rings):
        for i in range(n):
            vertices.append(skin_point(y, math.tau*i/n, .0025))
    for j in range(rings-1):
        for i in range(n):
            k=(i+1)%n
            # Outward normals: circumferential tangent cross axial tangent.
            faces.append((j*n+i,(j+1)*n+i,(j+1)*n+k,j*n+k))
    mesh = bpy.data.meshes.new(f"{side}_fitted_cuff")
    mesh.from_pydata(vertices, [], faces); mesh.update()
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    preview.parent_local(obj,parent);preview.assign_material(obj,mats["fabric"]);preview.smooth(obj)
    thick=obj.modifiers.new("Woven cuff thickness", "SOLIDIFY")
    thick.thickness=.003;thick.offset=1
    obj["fit_method"]="evaluated skin radial sections; 2.5 mm clearance; outward thickness"
    for y in (ymin,ymax):
        points=[skin_point(y,math.tau*i/n,.0053) for i in range(n+1)]
        preview.curve_tube(f"{side}_green_piping_{y}",points,.0011,mats["green"],parent)
        black=[skin_point(y,math.tau*i/n,.0038) for i in range(n+1)]
        preview.curve_tube(f"{side}_black_cuff_lip_{y}",black,.0012,mats["rubber"],parent)
    preview.rounded_box(f"{side}_mount_base", (0.049, 0.065, 0.006),
                        (0, -0.003, 0.094), mats["mount"], parent=parent, bevel=0.002)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.020, depth=0.009)
    pivot = bpy.context.object
    pivot.name = f"{side}_round_mount_pivot"
    preview.assign_material(pivot, mats["mount"])
    preview.smooth(pivot)
    preview.parent_local(pivot, parent, (0, -0.005, 0.101))
    bpy.ops.mesh.primitive_torus_add(major_radius=0.015, minor_radius=0.0025,
                                    major_segments=64, minor_segments=12)
    ring = bpy.context.object
    ring.name = f"{side}_mount_pivot_rim"
    preview.assign_material(ring, mats["rubber"])
    preview.parent_local(ring, parent, (0, -0.005, 0.108))
    preview.rounded_box(f"{side}_hinge_lower", (0.025, 0.018, 0.016),
                        (0, 0.018, 0.11), mats["mount"], parent=parent, bevel=0.003)
    preview.rounded_box(f"{side}_outer_pad", (0.014, 0.045, 0.011),
                        (0.033, 0.008, 0.081), mats["pad"], parent=parent,
                        rotation=(0, 0.5, 0), bevel=0.005)


def live_screen(camera_root, side, texture_path, face="rear"):
    if face not in ("front", "rear"):
        raise ValueError(face)
    mat = preview.material(f"{side}_{face}_LCD_glass", (0.06, 0.07, 0.08, 1), roughness=0.19)
    shader = mat.node_tree.nodes.get("Principled BSDF")
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(texture_path), check_existing=True)
    # LCD content emits light; it is not a diffuse printed texture. Lighting
    # the full image through Base Color washed the display out like white paper.
    shader.inputs["Base Color"].default_value = (.003, .003, .003, 1)
    shader.inputs["Specular"].default_value = .20
    mat.node_tree.links.new(tex.outputs["Color"], shader.inputs["Emission"])
    shader.inputs["Emission Strength"].default_value = 1.0
    # The rear display is local -Z; reverse the UV horizontal coordinate so
    # text reads correctly from the rear rather than through the display.
    mesh = bpy.data.meshes.new(f"{side}_{face}_screen_surface")
    if face == "rear":
        x0, x1, y0, y1, z = -.0300, .0300, -.0174, .0168, -.01598
        winding = (3, 2, 1, 0)
        coords = [(1, 0), (0, 0), (0, 1), (1, 1)]
    else:
        # Verified flat front LCD region of the original millimetre GLB;
        # the lens occupies the opposite (+X) half and is not covered.
        x0, x1, y0, y1, z = -.0273, -.0038, -.0101, .0137, .01194
        winding = (0, 1, 2, 3)
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    mesh.from_pydata([(x0,y0,z), (x1,y0,z), (x1,y1,z), (x0,y1,z)], [], [winding])
    uv = mesh.uv_layers.new()
    for loop in mesh.loops:
        uv.data[loop.index].uv = coords[loop.vertex_index]
    obj = bpy.data.objects.new(f"{side}_{face}_live_screen", mesh)
    bpy.context.collection.objects.link(obj)
    preview.parent_local(obj, camera_root)
    preview.assign_material(obj, mat)
    return obj


def reconnect_mount_and_cable(side, mats, mount_anchor=None):
    wrist = bpy.data.objects[f"{side}_wrist_rig"]
    root = bpy.data.objects[f"{side}_Action4_root"]
    bpy.context.view_layer.update()
    start = Vector(mount_anchor if mount_anchor is not None else (0, .020, .108))
    transform = root.matrix_world.inverted() @ wrist.matrix_world
    start_camera = transform @ start
    # The plate contacts only the -Y bottom. Route the arm below that plane
    # before approaching the centre; a diagonal from the wrist cuts the body.
    plate = preview.rounded_box(f"{side}_bottom_mount_plate", (.032, .006, .019),
                                (0, -.0247, .001), mats["mount"], parent=root, bevel=.001)
    preview.rounded_box(f"{side}_bottom_mount_neck", (.014, .012, .013),
                        (0, -.0337, .001), mats["mount"], parent=root, bevel=.001)
    end = Vector((0, -.042, .001))
    if start_camera.y <= -.033:
        route = [start_camera, Vector((start_camera.x, -.042, start_camera.z)), end]
    else:
        sign = 1 if start_camera.x >= 0 else -1
        outside_x = sign * max(abs(start_camera.x), .052)
        route = [start_camera, Vector((outside_x, start_camera.y, start_camera.z)),
                 Vector((outside_x, -.042, start_camera.z)), end]
    parts = []
    for i, (a, b) in enumerate(zip(route, route[1:])):
        direction = b-a
        if direction.length < .0001:
            continue
        name = f"{side}_mount_connection" if not parts else f"{side}_mount_connection_{i}"
        part = preview.rounded_box(name, (.010, .010, direction.length+.001),
                                   (a+b)*.5, mats["mount"], parent=root,
                                   rotation=direction.to_track_quat("Z", "Y").to_euler(), bevel=.001)
        part['mount_path_segment'] = True
        parts.append(part)
    root['bottom_mount_contact_local_m'] = [0, -.0217, .001]
    root['bottom_mount_route_local_m'] = [list(v) for v in route]
    obj = parts[0]
    connector = bpy.data.objects[f"{side}_strain_relief"]
    point = wrist.matrix_world.inverted() @ (connector.matrix_world @ Vector((0.007, 0, 0)))
    cable = bpy.data.objects[f"{side}_cable"]
    points = cable.data.splines[0].bezier_points
    sign = 1 if point.x >= 0 else -1
    for p, xyz in zip(points, [point, (point.x + sign * 0.018, point.y - 0.07, point.z - 0.03),
                              (sign * 0.064, -0.18, 0.074), (sign * 0.073, -0.32, 0.056),
                              (sign * 0.085, -0.54, 0.035)]):
        p.co = xyz
    return obj


def orient_hand_for_tabletop(hand):
    """Turn the palm down with a proper rotation; never reflect only its Z axis."""
    for vertex in hand.data.vertices:
        vertex.co.x = -vertex.co.x
        vertex.co.z = .12 - vertex.co.z
    # This is a 180-degree rotation about local Y around z=.06 (det +1).
    # Preserve face winding and UV association; a single-Z reflection swaps hands.


def shape_reference_forearm(hand, side):
    """Blend the wrist into a curved, varying forearm without changing the hand."""
    from scipy.interpolate import PchipInterpolator
    depths = [0, .08, .17, .26, .36, .55, .90]
    width = PchipInterpolator(depths, [.037, .042, .049, .047, .041, .047, .056])
    height = PchipInterpolator(depths, [.029, .034, .039, .037, .032, .038, .045])
    sign = -1 if side == 'left' else 1
    for vertex in hand.data.vertices:
        if vertex.co.y >= -.10:
            continue
        source = vertex.co.copy()
        distance = min(-source.y-.10, .80)
        stretch = min(distance/.06, 1)
        stretch = stretch*stretch*(3-2*stretch)
        depth = distance*(1+.7*stretch)
        theta = math.atan2(source.z-.060, source.x)
        blend = min(depth/.065, 1)
        blend = blend*blend*(3-2*blend)
        centre_x = sign*.012*math.sin(min(depth/.55,1)*math.pi*.65)
        centre_z = .060+.006*math.sin(min(depth/.45,1)*math.pi)
        anatomical = 1+.045*math.cos(2*theta+.6)*math.sin(min(depth/.3,1)*math.pi)
        target_x = centre_x+float(width(depth))*math.cos(theta)*anatomical
        target_z = centre_z+float(height(depth))*math.sin(theta)
        vertex.co.x = source.x*(1-blend)+target_x*blend
        vertex.co.z = source.z*(1-blend)+target_z*blend
        vertex.co.y = -.10-depth
    # Stretched forearm UVs also stretch the source normal map into long
    # ridges. Retain the detailed hand map and fade it along the forearm.
    attribute = hand.data.attributes.new(name='forearm_blend', type='FLOAT', domain='POINT')
    for vertex in hand.data.vertices:
        t = min(max((-vertex.co.y-.10)/.08, 0), 1)
        attribute.data[vertex.index].value = t*t*(3-2*t)
    material = hand.data.materials[0]
    nodes, links = material.node_tree.nodes, material.node_tree.links
    if nodes.get('Forearm normal attenuation') is None:
        attr = nodes.new('ShaderNodeAttribute');attr.attribute_name = 'forearm_blend'
        strength = nodes.new('ShaderNodeMapRange');strength.name = 'Forearm normal attenuation'
        strength.inputs['From Min'].default_value = 0
        strength.inputs['From Max'].default_value = 1
        strength.inputs['To Min'].default_value = .6
        strength.inputs['To Max'].default_value = .035
        links.new(attr.outputs['Fac'], strength.inputs['Value'])
        for node in nodes:
            if node.type == 'NORMAL_MAP':
                links.new(strength.outputs['Result'], node.inputs['Strength'])
    hand.data.update()
