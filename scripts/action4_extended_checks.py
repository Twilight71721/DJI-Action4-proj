"""Cross-limb and cable/fixture checks for expanded reference sampling."""
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from action4_cable_geometry import clearance, SAFETY


def prepare():
    bpy.context.view_layer.update()
    result=[]
    dg=bpy.context.evaluated_depsgraph_get()
    for side in ('left','right'):
        wrist=bpy.data.objects[f'{side}_wrist_rig']
        for obj in list(bpy.context.scene.objects):
            if not obj.name.startswith(side+'_') or obj.hide_render or obj.type not in ('MESH','CURVE'):
                continue
            if any(token in obj.name for token in ('screen','LCD','label','marker','AprilTag','button','cable')):
                continue
            if 'strain_relief' in obj.name:
                continue
            transform=wrist.matrix_world.inverted()@obj.matrix_world
            body=obj.name==side+'_Action4_body'
            if body:
                points=[transform@Vector(v) for v in obj.bound_box]
                faces=[(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
            else:
                evaluated=obj.evaluated_get(dg)
                mesh=evaluated.to_mesh()
                points=[transform@v.co for v in mesh.vertices]
                faces=[tuple(face.vertices) for face in mesh.polygons]
                evaluated.to_mesh_clear()
            if not faces:
                continue
            array=np.asarray(points)
            result.append(dict(name=obj.name,side=side,points=array,faces=faces,
                               tree=BVHTree.FromPolygons(points,faces),bounds=(array.min(0),array.max(0)),
                               hand='anatomical_hand_forearm' in obj.name))
    return result


def audit(parts,routes):
    bpy.context.view_layer.update()
    matrices={s:bpy.data.objects[f'{s}_wrist_rig'].matrix_world.copy() for s in ('left','right')}
    world={}
    for part in parts:
        xyz=np.asarray([matrices[part['side']]@Vector(p) for p in part['points']])
        world[part['name']]=(xyz,xyz.min(0),xyz.max(0))
    cross_checks=0
    for a in parts:
        if a['side']!='left':continue
        aa,amin,amax=world[a['name']]
        for b in parts:
            if b['side']!='right':continue
            bb,bmin,bmax=world[b['name']]
            if np.any(amax<bmin) or np.any(bmax<amin):continue
            cross_checks+=1
            if BVHTree.FromPolygons(aa.tolist(),a['faces']).overlap(BVHTree.FromPolygons(bb.tolist(),b['faces'])):
                raise RuntimeError(f"Cross-limb intersection: {a['name']} / {b['name']}")
            # Triangle overlap alone misses containment and coplanar faces.
            for source,target in ((a,b),(b,a)):
                transform=np.asarray(matrices[target['side']].inverted()@matrices[source['side']])
                centres=np.asarray([source['points'][list(face)].mean(0) for face in source['faces']])
                samples=np.vstack((source['points'],centres))
                samples=(transform@np.column_stack((samples,np.ones(len(samples)))).T).T[:,:3]
                lo,hi=target['bounds']
                candidates=samples[np.all((samples>lo+1e-5)&(samples<hi-1e-5),axis=1)]
                if any(clearance(target['tree'],point,target['bounds']) < -1e-5 for point in candidates):
                    raise RuntimeError(f"Cross-limb intersection/containment: {source['name']} / {target['name']}")
    checked=0
    minimum=1e6
    for route in routes:
        spacing=float(np.linalg.norm(np.diff(route['points'],axis=0),axis=1).max())
        margin=route['radius']+SAFETY+spacing*.5
        for part in parts:
            if part['hand']:continue  # Denser signed checks already run against both arms.
            transform=np.asarray(matrices[part['side']].inverted()@matrices[route['side']])
            pts=(transform@np.column_stack([route['points'],np.ones(len(route['points']))]).T).T[:,:3]
            lo,hi=part['bounds']
            near=pts[np.all((pts>=lo-margin)&(pts<=hi+margin),axis=1)]
            checked+=1
            if not len(near):continue
            distance=min(clearance(part['tree'],point,part['bounds']) for point in near)
            minimum=min(minimum,distance-route['radius']-spacing*.5)
            if distance<margin:
                raise RuntimeError(f"{route['side']} cable intersects/approaches {part['name']}: {distance:.6f}m")
    return dict(cross_limb_intersections=0,cross_limb_narrow_checks=cross_checks,
                cable_fixture_pairs_checked=checked,cable_fixture_near_min_m=None if minimum==1e6 else minimum,
                scope='Both arm/fixture assemblies against each other; cable tube bounds against cuffs/mounts/body envelopes. Existing own-skin and bottom-contact audits also required.')
