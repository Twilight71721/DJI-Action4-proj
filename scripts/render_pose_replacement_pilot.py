"""Small local HCCEPose foreground pilot; existing training/render runs are untouched."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPModel
from bop_bb8.geometry import crop_and_scale_intrinsics, project_corners


def rotation(degrees):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('xyz', degrees, degrees=True).as_matrix()


def view_rotation(direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    right = np.cross([0., 1., 0.], d)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(-d, right), -d])


def prepare(out):
    if (out/'plan.json').exists():
        raise FileExistsError('Plan already exists; use a new output directory')
    anchors = json.loads((out/'initial_anchors.json').read_text())
    (out/'sources').mkdir(exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    for name in ['models_info.json', 'corners_3d.json']:
        shutil.copy2(ROOT/'dataset/action4-cleanblur5000-20260923/models'/name, out/'models'/name)
    model = BOPModel.from_models_info(out/'models/models_info.json')
    expected = np.array([x['xyz'] for x in json.loads((out/'models/corners_3d.json').read_text())['corners']])
    assert np.allclose(model.corners_mm(), expected, rtol=0, atol=1e-10)
    # Hand traced on the saved 512px inspection crops. These are removal regions,
    # not segmentation GT, and never provide the synthetic object's pose labels.
    polygons = [
        [[131,156],[150,145],[178,139],[198,141],[248,162],[349,198],[383,221],[411,250],
         [412,267],[359,365],[347,377],[327,376],[117,327],[111,313]],
        [[103,181],[114,170],[138,162],[164,157],[185,157],[202,164],[366,170],[389,176],
         [417,219],[418,238],[379,347],[369,360],[353,361],[112,338],[96,327],[96,227]],
        [[95,165],[202,161],[390,166],[434,187],[449,224],[448,342],[365,351],[197,365],
         [119,341],[102,328]],
        [[173,141],[190,126],[219,116],[251,112],[277,115],[294,129],[330,126],[345,136],
         [365,206],[386,313],[377,387],[357,412],[336,423],[173,419],[153,409],[136,315],
         [133,277],[140,250],[162,192],[174,169]],
    ]
    groups = []
    for i, a in enumerate(anchors):
        raw = ROOT/f'outputs/video_reference_audit/head_left_rgb_raw/frames_full/frame_{a["frame"]:06d}.jpg'
        image = cv2.imread(str(raw))
        x,y,w,h = a['crop_xywh']
        roi = cv2.resize(image[y:y+h,x:x+w], (512,512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out/'sources'/f'anchor_{i}.png'), roi)
        mask = np.zeros((512,512), np.uint8)
        polygon=a['removal_polygon_512'] if 'removal_polygon_512' in a else polygons[i]
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
        cv2.imwrite(str(out/'sources'/f'anchor_{i}_removal.png'), mask)
        if a.get('occluder_polygons_512'):
            occluder=np.zeros_like(mask)
            for poly in a['occluder_polygons_512']:
                cv2.fillPoly(occluder,[np.array(poly,np.int32)],255)
            a['occluder_mask']=f'sources/anchor_{i}_occluder.png'
            cv2.imwrite(str(out/a['occluder_mask']),occluder)
        # Anchor 2 contains unresolved foreground cable/mount occlusion.
        a.update(removal_polygon_512=polygon, source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                 reference_pose_is_estimated=True, known_occlusion_unresolved=(i == 2),
                 source_group=f'video_segment_{"raised_start" if a["frame"] == 0 else "tabletop_button"}')
        K = crop_and_scale_intrinsics(np.array(a['fit']['K']), a['crop_xywh'], (512,512))
        base_R, t = np.array(a['fit']['R']), np.array(a['fit']['t_m'])*1000
        jobs = []
        for j, delta in enumerate([[0,0,0],[1.5,0,0],[0,1.5,0],[0,0,-2]]):
            R = rotation(delta) @ base_R
            jobs.append(dict(id=f'replace_{i}_{j}', kind='replacement', anchor=i,
                family=f'reference_{a["frame"]}_{a["side"]}', source_group=a['source_group'],
                R=R.tolist(), t_mm=t.tolist(), delta_degrees=delta,
                light_energy=a.get('light_energy',10.), exposure=a.get('exposure',-.45)))
        groups.append(dict(id=f'anchor_{i}', side=a['side'], K=K.tolist(), records=jobs))
    # Broader view bins are augmentation proposals, not calibrated real poses.
    directions = [('rear',[0,.25,-1]),('rear_left',[.6,.2,-1]),('rear_right',[-.6,.2,-1]),
        ('front',[0,.15,1]),('front_left',[.6,.15,1]),('front_right',[-.6,.15,1]),
        ('left_side',[1,.2,.1]),('right_side',[-1,.2,-.1]),('top_rear',[0,1,-.5]),
        ('top_front',[0,1,.5]),('bottom_rear',[0,-1,-.6]),('bottom_front',[0,-1,.6]),
        ('uploaded_left',None),('uploaded_right',None)]
    rng = np.random.default_rng(20260924)
    for side in ['left','right']:
        jobs = []
        for name, d in directions:
            base_R = np.array(anchors[0 if name == 'uploaded_left' else 1]['fit']['R']) if d is None else view_rotation(d)
            for j in range(4):
                delta = [float(rng.uniform(-8,8)),float(rng.uniform(-8,8)),[-30,-10,10,30][j]]
                R = rotation(delta) @ base_R
                t = [float(rng.uniform(-.004,.004)*1000),float(rng.uniform(-.004,.004)*1000),float(rng.uniform(135,180))]
                jobs.append(dict(id=f'pose_{side}_{name}_{j}',kind='foreground',family=name,
                    source_group=f'pose_neighborhood_{name}', R=R.tolist(), t_mm=t,
                    delta_degrees=delta, light_energy=float(rng.uniform(6,16)),exposure=float(rng.uniform(-.8,.2))))
        groups.append(dict(id=f'poses_{side}',side=side,K=[[700,0,256],[0,700,256],[0,0,1]],records=jobs))
    payload = dict(seed=20260924, width=512,height=512,anchors=anchors,groups=groups,
        status='pilot_not_full_training_release',
        corner_order='current bop_bb8.geometry.CORNER_SIGNS and models/corners_3d.json',
        reference_calibrated=False,background_bulk_location='server',
        glb=str(ROOT/'dataset/action4-v6-domain-pilot-200/models/obj_000001.glb'))
    (out/'plan.json').write_text(json.dumps(payload,indent=2))
    print('PREPARED',len(groups),sum(len(g['records']) for g in groups),flush=True)


def render(out, group_id, samples):
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    os.environ.setdefault('EGL_DEVICE_ID','0')
    sys.path.insert(0,str(ROOT/'third_party/Blender/vendor'))
    import bpy
    from mathutils import Matrix, Vector
    from bpy_extras.object_utils import world_to_camera_view
    bpy.ops.preferences.addon_enable(module='cycles')
    import blenderproc as bproc
    from blenderproc.python.utility.Utility import Utility
    import blender_real_scene_preview as p
    from action4_dynamic_scene import standard_marker
    from action4_wrist_assets import live_screen
    plan = json.loads((out/'plan.json').read_text())
    plan['glb'] = str((ROOT / plan['glb']).resolve())
    group = next(g for g in plan['groups'] if g['id'] == group_id)
    dest = out/'renders'/group_id
    dest.mkdir(parents=True,exist_ok=False)
    temporary = dest/'temporary'; temporary.mkdir()
    Utility.temp_dir = str(temporary)
    started = time.monotonic()
    bproc.init()
    bproc.renderer.set_render_devices(desired_gpu_device_type='OPTIX',desired_gpu_ids=[0])
    scene = bpy.context.scene
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = .03
    scene.cycles.max_bounces = 4
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = 'Standard'
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.4,.4,.4,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = .5
    bproc.renderer.set_output_format('PNG',enable_transparency=True)
    loaded = bproc.loader.load_obj(plan['glb'])
    if len(loaded) != 1:
        raise RuntimeError('Expected exact one-mesh GLB')
    body = loaded[0].blender_obj
    vertex_count, face_count = len(body.data.vertices),len(body.data.polygons)
    assert (vertex_count,face_count) == (915032,1499206)
    bounds = np.array([v.co[:] for v in body.data.vertices])
    model = BOPModel.from_models_info(out/'models/models_info.json')
    assert np.allclose(bounds.min(0),model.min_xyz_mm,atol=1e-4)
    assert np.allclose(bounds.max(0),model.max_xyz_mm,atol=1e-4)
    sys.path.insert(0,str(ROOT/'dataset/code'))
    from action4_domain_v6 import darken_target_materials
    darken_target_materials(loaded[0],factor=.12)
    for mat in body.data.materials:
        node=mat.node_tree.nodes.get('Principled BSDF')
        if node: node.inputs['Specular'].default_value=.20
    root=bpy.data.objects.new('DeviceRoot',None);scene.collection.objects.link(root)
    p.parent_local(body,root);body.scale=(.001,)*3
    root.rotation_mode='QUATERNION'
    side=group['side']; mats=p.build_materials()
    p.camera_decals(root,mats,side,'dji-003' if side=='left' else 'dji-004',.019)
    bpy.data.objects[f'{side}_top_label_text'].rotation_euler=(-math.pi/2,math.pi,0)
    standard_marker(root,side,dest)
    red=p.material(f'{side}_button_red',(.30,.009,.012,1),roughness=.72)
    bpy.ops.mesh.primitive_torus_add(major_radius=.0055,minor_radius=.00045,major_segments=48,minor_segments=10)
    ring=bpy.context.object;p.assign_material(ring,red)
    p.parent_local(ring,root,(-.019,.0217,-.007),(math.pi/2,0,0))
    source=ROOT/'dataset/action4-screen-cable-corrected32-20260921/render_metadata/batch_000'
    live_screen(root,side,source/f'front_screen_{side}.png','front')
    live_screen(root,side,source/f'screen_{202609230+(side=="right")}.png','rear')
    bproc.camera.set_intrinsics_from_K_matrix(np.array(group['K']),512,512,clip_start=.001,clip_end=10.)
    lamp=bpy.data.objects.new('Key',bpy.data.lights.new('Key','AREA'));scene.collection.objects.link(lamp)
    lamp.data.size=.5;lamp.location=(-.2,-.3,.05)
    obj=bproc.types.MeshObject(body)
    obj.set_cp('category_id',1);obj.set_cp('model_path',plan['glb'])
    cv_to_blender=np.diag([1,-1,-1,1])
    max_projection_error=0.
    for frame,r in enumerate(group['records']):
        scene.frame_set(frame)
        root.location=np.array(r['t_mm'])/1000
        root.rotation_quaternion=Matrix(r['R']).to_quaternion()
        root.keyframe_insert('location',frame=frame);root.keyframe_insert('rotation_quaternion',frame=frame)
        lamp.location=r.get('light_location',[-.2,-.3,.05])
        lamp.keyframe_insert('location',frame=frame)
        lamp.data.size=r.get('light_size',.5);lamp.data.keyframe_insert('size',frame=frame)
        lamp.data.color=r.get('light_color',[1.,1.,1.]);lamp.data.keyframe_insert('color',frame=frame)
        for mat in body.data.materials:
            node=mat.node_tree.nodes.get('Principled BSDF')
            if node:
                for field,socket in [('body_roughness','Roughness'),('body_specular','Specular')]:
                    if field in r:
                        node.inputs[socket].default_value=r[field]
                        node.inputs[socket].keyframe_insert('default_value',frame=frame)
        lamp.rotation_euler=(root.location-lamp.location).to_track_quat('-Z','Y').to_euler()
        lamp.keyframe_insert('rotation_euler',frame=frame)
        lamp.data.energy=r['light_energy'];lamp.data.keyframe_insert('energy',frame=frame)
        scene.view_settings.exposure=r['exposure'];scene.view_settings.keyframe_insert('exposure',frame=frame)
        bproc.camera.add_camera_pose(cv_to_blender,frame=frame)
        bpy.context.view_layer.update()
        uv,valid,_=project_corners(model.corners_mm(),group['K'],r['R'],r['t_mm'],image_size=(512,512))
        if not valid.all():raise RuntimeError(f'Pilot corner outside image: {r["id"]}')
        projected=[]
        for xyz in model.corners_mm():
            p3=root.matrix_world@Vector(xyz/1000)
            xy=world_to_camera_view(scene,scene.camera,p3)
            projected.append([xy.x*512,(1-xy.y)*512])
        # Blender uses image centers at (W-1)/2, so the API projection has a
        # half-pixel raster convention difference from world_to_camera_view.
        error=float(np.max(np.linalg.norm(np.array(projected)-uv,axis=1)))
        max_projection_error=max(error,max_projection_error)
        if error>1.:raise RuntimeError(f'Blender/OpenCV mismatch {error}')
        r.update(keypoints_2d=uv.tolist(),valid=valid.tolist(),blender_projection_error_px=error)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(samples)
    # Use the same native writer as the existing production renderer, including
    # independent mesh rasterization for BOP body masks and visibility.
    bproc.renderer.render(output_dir=str(temporary),return_data=False)
    depths=[];colors=[]
    depth_entry=Utility.find_registered_output_by_key('depth')
    for i in range(len(group['records'])):
        rgba=cv2.imread(str(temporary/f'rgb_{i:04d}.png'),cv2.IMREAD_UNCHANGED)
        assert rgba is not None and rgba.shape==(512,512,4)
        rgba_dir=dest/'rgba';rgba_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(rgba_dir/f'{i:06d}.png'),rgba)
        colors.append(cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_BGR2RGB))
        raw=cv2.imread(depth_entry['path'] % i,cv2.IMREAD_UNCHANGED)
        raw=raw[:,:,0] if raw.ndim==3 else raw
        depths.append(np.where(np.isfinite(raw)&(raw>0)&(raw<10),raw,0).astype(np.float32))
    bproc.writer.write_bop(str(dest),dataset='bop',target_objects=[obj],depths=depths,colors=colors,
        depth_scale=.1,color_file_format='PNG',frames_per_chunk=len(colors),calc_mask_info_coco=True,
        num_worker=0,delta=1)
    report=dict(group=group,pipeline='HCCEPose vendored BlenderProc / Cycles OPTIX / native BOP writer',
        blenderproc_path=str(Path(bproc.__file__).resolve()),glb_sha256=hashlib.sha256(Path(plan['glb']).read_bytes()).hexdigest(),
        vertex_count=vertex_count,face_count=face_count,body_color_multiplier=.12,body_specular=.20,
        geometry_unchanged=True,reference_K_estimated=group_id.startswith('anchor'),
        max_blender_projection_error_px=max_projection_error,elapsed_seconds=time.monotonic()-started)
    (dest/'metadata.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',group_id,report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['prepare','render'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--group')
    ap.add_argument('--samples',type=int,default=32)
    args=ap.parse_args()
    if args.mode=='prepare':prepare(args.output.resolve())
    else:render(args.output.resolve(),args.group,args.samples)
