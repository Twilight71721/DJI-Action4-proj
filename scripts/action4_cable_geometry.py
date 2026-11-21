"""Route real tube geometry beside the forearm and audit its surface clearance."""

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

RADIUS = .0018
SAFETY = .001


def hand_obstacle(side):
    hand = bpy.data.objects[f"{side}_anatomical_hand_forearm"]
    wrist = bpy.data.objects[f"{side}_wrist_rig"]
    bpy.context.view_layer.update()
    evaluated = hand.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    transform = wrist.matrix_world.inverted() @ evaluated.matrix_world
    # Close the cropped forearm only in the collision proxy. Open boundaries
    # make ray parity undefined; the rendered hand mesh remains unchanged.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    boundary = [edge for edge in bm.edges if edge.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    points = [transform @ v.co for v in bm.verts]
    faces = [tuple(v.index for v in face.verts) for face in bm.faces]
    tree = BVHTree.FromPolygons(points, faces)
    array = np.asarray(points)
    bm.free()
    evaluated.to_mesh_clear()
    return tree, array


def clearance(tree, point, bounds):
    nearest, normal, _, distance = tree.find_nearest(Vector(point))
    if nearest is None:
        raise RuntimeError("Hand BVH has no surface")
    if np.any(np.asarray(point) < bounds[0]) or np.any(np.asarray(point) > bounds[1]):
        return float(distance)
    # Nearest-face normals are not a valid sign test far from a concave/open
    # limb. Ray parity along the transverse axis avoids the proximal Y cut.
    direction = Vector((1, .00017, .00031)).normalized()
    origin = Vector(point)
    crossings = 0
    for _ in range(32):
        hit, _, _, _ = tree.ray_cast(origin, direction)
        if hit is None:
            break
        crossings += 1
        origin = hit + direction*.000001
    else:
        raise RuntimeError("Ambiguous hand parity query")
    return float(-distance if crossings % 2 else distance)


def route_cable(side, mats, obstacle, fixture_obstacles=None):
    tree, vertices = obstacle
    wrist = bpy.data.objects[f"{side}_wrist_rig"]
    connector = bpy.data.objects[f"{side}_strain_relief"]
    bpy.context.view_layer.update()
    relative = wrist.matrix_world.inverted() @ connector.matrix_world
    start = np.array(relative @ Vector((.007, 0, 0)))
    direction = np.array(relative.to_quaternion() @ Vector((1,0,0)))
    sign = -1 if start[0] < 0 else 1
    # Leave the actual plug along its axis, then turn around the outside of
    # the arm. Dense POLY samples avoid hidden Bezier overshoot between knots.
    bounds = (vertices.min(axis=0), vertices.max(axis=0))
    candidates=[(.020,.008,0,.060)] if fixture_obstacles is None else [
        (offset,exit_length,lead,height) for height in (.060,.100,.140,.180)
        for lead in (0,.020,.040,.060)
        for offset in (.020,.030,.040)
        for exit_length in (.008,.016,.024,.036,.050,.075,.100,.150,.200,.250)]
    rejected=[]
    for offset,exit_length,lead,height in candidates:
        outer_x = sign * (np.max(sign * vertices[:,0]) + offset)
        # A straight socket exit prevents the bend from cutting back through
        # the camera body when the fixture is strongly rotated.
        bend_start = start + direction*lead
        control = np.array([bend_start, bend_start + direction*exit_length,
                            [outer_x, -.095, height], [outer_x, -.18, height]])
        t = np.linspace(0, 1, 220)[:,None]
        bend = ((1-t)**3*control[0] + 3*(1-t)**2*t*control[1]
                + 3*(1-t)*t*t*control[2] + t**3*control[3])
        y = np.linspace(-.181, -1.85, 1700)
        continuation = np.column_stack([outer_x + sign*.006*(1-np.cos((y+.18)*3)), y,
                                        height + .004*(1-np.cos((y+.18)*2))])
        straight = start + np.linspace(0, lead, max(1, int(np.ceil(lead/.001))), endpoint=False)[:,None]*direction
        points = np.vstack([straight, bend, continuation]) if lead else np.vstack([bend, continuation])
        spacing=float(np.linalg.norm(np.diff(points,axis=0),axis=1).max())
        margin=RADIUS+SAFETY+spacing*.5
        minimum=min(clearance(tree, point, bounds) for point in points)
        reason=None
        if minimum < margin:
            reason=f'own arm: {minimum:.6f}m'
        for part in fixture_obstacles or []:
            if part['side']!=side or part['hand']:continue
            lo,hi=part['bounds']
            near=points[np.all((points>=lo-margin)&(points<=hi+margin),axis=1)]
            if len(near):
                distance=min(clearance(part['tree'],point,part['bounds']) for point in near)
                if distance < margin:
                    reason=f"{part['name']}: {distance:.6f}m"
                    break
        if reason is None:break
        rejected.append(dict(lateral_offset_m=offset,exit_tangent_m=exit_length,straight_exit_m=lead,height_m=height,reason=reason))
    else:
        raise RuntimeError(f'{side} cannot route cable safely around its fixed fixture: {rejected}')
    old = bpy.data.objects[f"{side}_cable"]
    old.hide_render = True
    curve = bpy.data.curves.new(f"{side}_collision_checked_cable", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = RADIUS
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points)-1)
    for vertex, point in zip(spline.points, points):
        vertex.co = (*point, 1)
    curve.materials.append(mats["cable"])
    obj = bpy.data.objects.new(curve.name, curve)
    bpy.context.collection.objects.link(obj)
    obj.parent = wrist
    return {"side": side, "points": points, "radius": RADIUS,
            "own_arm_min_center_clearance_m": minimum,
            "route_parameters":dict(lateral_offset_m=offset,exit_tangent_m=exit_length,straight_exit_m=lead,height_m=height,rejected_candidates=rejected),
            "connector_gap_m": float(np.linalg.norm(points[0]-start))}


def audit_frame(routes, obstacles):
    bpy.context.view_layer.update()
    report = []
    for route in routes:
        source = bpy.data.objects[f"{route['side']}_wrist_rig"]
        distances = {}
        spacing = float(np.linalg.norm(np.diff(route["points"],axis=0),axis=1).max())
        for side, (tree, vertices) in obstacles.items():
            target = bpy.data.objects[f"{side}_wrist_rig"]
            transform = np.asarray(target.matrix_world.inverted() @ source.matrix_world)
            points = (transform @ np.column_stack([route["points"], np.ones(len(route["points"]))]).T).T[:,:3]
            bounds = (vertices.min(axis=0), vertices.max(axis=0))
            minimum = min(clearance(tree, point, bounds) for point in points)
            distances[side] = minimum-route["radius"]
            if distances[side] - spacing*.5 < SAFETY:
                raise RuntimeError(f"{route['side']} cable fails clearance against {side} arm: {distances[side]:.6f} m")
        report.append({"side": route["side"], "tube_radius_m": route["radius"],
                       "route_parameters": route.get('route_parameters'),
                       "surface_clearance_m": distances, "connector_gap_m": route["connector_gap_m"],
                       "conservative_segment_clearance_m": {s: d-spacing*.5 for s,d in distances.items()},
                       "samples": len(route["points"]), "maximum_sample_spacing_m":
                       spacing})
    return report
