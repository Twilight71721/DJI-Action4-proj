"""Fit manually annotated real-frame landmarks; render a diagnostic, never training RGB.

K and metric depth are estimates. No video pixels enter Blender materials.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / 'outputs/video_reference_audit/head_left_rgb_raw/frames_full'


def annotations():
    # Frame 0 annotations were picked on the 1813 x 1376 displayed image.
    rear = [[35,21,-15.8],[-35,21,-15.8],[-35,-21,-15.8],[35,-21,-15.8],
            [35,21,12],[-35,21,12]]
    scale = np.array([3248/1813, 2464/1376])
    return [
        dict(frame=0, side='left', face='rear', xyz_mm=rear,
             uv=(np.array([[682,589],[936,659],[878,793],[635,740],[670,556],[901,615]])*scale).tolist()),
        dict(frame=0, side='right', face='rear', xyz_mm=rear,
             uv=(np.array([[1179,637],[1526,645],[1485,797],[1168,769],[1191,576],[1496,585]])*scale).tolist()),
        dict(frame=500, side='left', face='front',
             xyz_mm=[[-35,21,11.84],[35,21,11.84],[35,-21,11.84],[-35,-21,11.84],[15,3,13.24]],
             uv=(np.array([[180,75],[440,87],[455,226],[180,219],[330,135]])+[1280,1700]).tolist()),
        dict(frame=500, side='right', face='front',
             xyz_mm=[[-3.8,13.7,11.94],[-3.8,-10.1,11.94],[-27.3,-10.1,11.94],[-27.3,13.7,11.94],
                     [15,3,15.8]],
             uv=(np.array([[676,216],[781,210],[767,267],[656,274],[722,160]])+[1280,1700]).tolist()),
    ]


def fit(item, focal):
    xyz = np.asarray(item['xyz_mm'], dtype=np.float64)*.001
    uv = np.asarray(item['uv'], dtype=np.float64)
    K = np.array([[focal,0,1624],[0,focal,1232],[0,0,1.]])
    ok, r, t = cv2.solvePnP(xyz, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError('PnP failed')
    r, t = cv2.solvePnPRefineLM(xyz, uv, K, None, r, t)
    projected = cv2.projectPoints(xyz,r,t,K,None)[0][:,0]
    R = cv2.Rodrigues(r)[0]
    direction = -R.T@t[:,0]
    direction /= np.linalg.norm(direction)
    normal = {'front':[0,0,1], 'rear':[0,0,-1], 'top':[0,1,0]}[item['face']]
    if np.dot(direction, normal) <= 0 or np.any((R@xyz.T+t)[2] <= 0):
        raise RuntimeError(f'Wrong visible face: {item}')
    return dict(K=K.tolist(), R=R.tolist(), t_m=t[:,0].tolist(), projected_uv=projected.tolist(),
                landmark_rmse_px=float(np.sqrt(np.mean(np.sum((projected-uv)**2, axis=1)))),
                face_cosines=dict(front=float(direction[2]),rear=float(-direction[2]),top=float(direction[1])))


def render(records, output):
    import bpy
    from mathutils import Matrix, Vector
    import blender_real_scene_preview as p
    from action4_dynamic_scene import standard_marker
    from action4_wrist_assets import live_screen
    sys.path.insert(0, str(ROOT/'third_party/Blender/vendor'))
    bpy.ops.preferences.addon_enable(module='cycles')
    import blenderproc as bproc
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.engine='CYCLES'; scene.cycles.samples=32
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='OPTIX'
    scene.cycles.device='GPU'
    scene.render.film_transparent=True
    scene.render.image_settings.color_mode='RGBA'
    scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.4,.4,.4,1)
    scene.view_settings.view_transform='Standard'
    cam=bpy.data.objects.new('Observer',bpy.data.cameras.new('Observer'))
    scene.collection.objects.link(cam);scene.camera=cam
    cam.matrix_world=Matrix(np.diag([1,-1,-1,1]).tolist())
    bpy.ops.import_scene.gltf(filepath=str(ROOT/'dataset/action4-v6-domain-pilot-200/models/obj_000001.glb'))
    base=next(o for o in scene.objects if o.type=='MESH')
    base.hide_render=True
    for mat in base.data.materials:
        sh=mat.node_tree.nodes.get('Principled BSDF');sh.inputs['Specular'].default_value=.15
        socket=sh.inputs['Base Color']
        if socket.is_linked:
            src=socket.links[0].from_socket
            mul=mat.node_tree.nodes.new('ShaderNodeMixRGB');mul.blend_type='MULTIPLY'
            mul.inputs[0].default_value=1;mul.inputs[2].default_value=(.12,.12,.12,1)
            mat.node_tree.links.new(src,mul.inputs[1]);mat.node_tree.links.new(mul.outputs[0],socket)
    mats=p.build_materials()
    # Existing synthetic-only screen captures; no real image texture is loaded.
    source=ROOT/'dataset/action4-screen-cable-corrected32-20260921/render_metadata/batch_000'
    for record in records:
        created=set(scene.objects)
        side=record['side'];root=bpy.data.objects.new(f'{side}_fit',None);scene.collection.objects.link(root)
        body=bpy.data.objects.new(f'{side}_body',base.data);scene.collection.objects.link(body)
        p.parent_local(body,root);body.scale=(.001,)*3
        p.camera_decals(root,mats,side,'dji-003' if side=='left' else 'dji-004',.019)
        label=bpy.data.objects[f'{side}_top_label'];label.location.y=.02166
        text=bpy.data.objects[f'{side}_top_label_text'];text.location.y=.02175;text.rotation_euler=(-math.pi/2,math.pi,0)
        standard_marker(root,side,output)
        red=p.material(f'{side}_button_red',(.30,.009,.012,1),roughness=.72)
        bpy.ops.mesh.primitive_torus_add(major_radius=.0055,minor_radius=.00045,major_segments=48,minor_segments=10)
        ring=bpy.context.object;p.assign_material(ring,red)
        p.parent_local(ring,root,(-.019,.0217,-.007),(math.pi/2,0,0))
        live_screen(root,side,source/f'front_screen_{side}.png','front')
        live_screen(root,side,source/f'screen_{202609230+(side=="right")}.png','rear')
        transform=np.eye(4);transform[:3,:3]=record['fit']['R'];transform[:3,3]=record['fit']['t_m']
        root.matrix_world=Matrix(transform.tolist())
        K=np.asarray(record['fit']['K'])*.5;K[2,2]=1
        bproc.camera.set_intrinsics_from_K_matrix(K,1624,1232)
        lamp=bpy.data.objects.new('Key',bpy.data.lights.new('Key','AREA'));scene.collection.objects.link(lamp)
        lamp.data.energy=10;lamp.data.size=.5;lamp.location=(-.2,-.3,.05)
        lamp.rotation_euler=(root.location-lamp.location).to_track_quat('-Z','Y').to_euler()
        name=f"frame_{record['frame']:06d}_{side}"
        scene.render.filepath=str(output/f'{name}_render.png')
        bpy.ops.render.render(write_still=True)
        record['render']=f'{name}_render.png'
        for obj in set(scene.objects)-created:bpy.data.objects.remove(obj,do_unlink=True)


def review(records, output):
    from PIL import Image, ImageDraw
    panels=[]
    for r in records:
        real=Image.open(FRAMES/f"frame_{r['frame']:06d}.jpg").convert('RGB').resize((1624,1232))
        synth=Image.open(output/r['render']).convert('RGBA')
        box=synth.getbbox()
        x0,y0,x1,y1=box;box=(max(0,x0-45),max(0,y0-45),min(1624,x1+45),min(1232,y1+45))
        overlay=real.copy().convert('RGBA')
        edges=cv2.Canny(np.asarray(synth)[:,:,3],80,160)
        arr=np.asarray(overlay).copy();arr[edges>0,:3]=[0,255,220];overlay=Image.fromarray(arr)
        draw=ImageDraw.Draw(overlay)
        for n,(u,v) in enumerate(r['uv']):
            u*=.5;v*=.5;draw.ellipse((u-2,v-2,u+2,v+2),fill='yellow');draw.text((u+3,v),str(n),fill='yellow')
        neutral=Image.new('RGBA',synth.size,(90,95,105,255));neutral.alpha_composite(synth)
        panel=Image.new('RGB',(1200,410),(24,29,35));d=ImageDraw.Draw(panel)
        for idx,(im,title) in enumerate([(real,'REAL'),(neutral,'GLB FIT - estimated 3D'),(overlay,'OUTLINE OVERLAY - diagnostic only')]):
            crop=im.crop(box);crop.thumbnail((395,345));panel.paste(crop,(idx*400+(400-crop.width)//2,42+(345-crop.height)//2))
            d.text((idx*400+8,24),title,fill='white')
        d.text((8,5),f"Frame {r['frame']:06d} / {r['side']} / {r['face']} / manual landmark RMSE {r['fit']['landmark_rmse_px']:.1f} full-res px",fill='white')
        name=f"frame_{r['frame']:06d}_{r['side']}_comparison.jpg";panel.save(output/name,quality=95);panels.append(name)
    html='''<!doctype html><meta charset="utf-8"><title>真实帧二维对齐检查</title><style>body{background:#171d23;color:#eee;font:17px sans-serif;max-width:1400px;margin:30px auto}img{width:100%}p{line-height:1.7}</style><h1>真实帧二维对齐检查</h1><p>仅用于相机朝向检查，尚不是训练数据或最终场景。黄色点为人工参考点，青色线为精确 GLB 的投影轮廓；真实图仅用于并排/叠加诊断，没有进入渲染器纹理。K、深度和三维姿态均为估计；标点误差不能当作真实姿态精度。遮挡、手腕、支架和衣物尚未拟合。</p>'''
    for name in panels:html+=f'<img src="{name}">'
    (output/'index.html').write_text(html)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--review-only',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.review_only:records=json.loads((out/'alignment.json').read_text())['records']
    else:
        records=annotations()
        for r in records:
            r['fit']=fit(r,2000)
            r['focal_sensitivity']=[dict(focal_px=f,**fit(r,f)) for f in (1600,2400)]
        reference=cv2.imread(str(FRAMES/'frame_000000.jpg'))
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
        corners,ids,_=detector.detectMarkers(reference)
        (out/'detected_tags.json').write_text(json.dumps({'frame':0,'corners':{str(int(i[0])):c[0].tolist() for i,c in zip(ids,corners)}},indent=2))
        render(records,out)
        (out/'alignment.json').write_text(json.dumps(dict(status='estimated_2d_diagnostic_not_training',calibrated=False,records=records),indent=2))
    review(records,out)

if __name__=='__main__':main()
