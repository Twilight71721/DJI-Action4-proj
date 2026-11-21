"""Observed video representatives and physically attached wrist camera fixtures.

Profiles are qualitative reconstructions, not calibrated real-frame 6D labels.
"""
import math
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector, Euler
import blender_real_scene_preview as p
import action4_dynamic_scene as dynamic
import action4_cable_geometry as cable

# Reference frame, action, cloth progress, rolls, contact Y, camera bank.
SPECS = [
    (0,'raised',0,(-20,20),.0,0),
    (150,'spread',0,(-35,40),.48,3),
    (300,'button',.2,(-60,65),.52,-8),
    (500,'button',.3,(-65,55),.45,-10),
    (700,'flat_press',0,(-20,25),.30,2),
    (950,'button',.5,(-72,62),.47,8),
    (1200,'button',.8,(-75,75),.14,-3),
    (1450,'button',.6,(-60,72),.24,3),
    (1650,'button',.5,(-75,40),.30,0),
    (1850,'button',.4,(-55,65),.40,2),
    (2050,'button',.2,(-60,75),.46,-4),
    (2200,'button',.1,(-75,65),.54,-5),
    (2350,'sleeve_grasp',.15,(-25,70),.29,5),
    (2420,'right_sleeve_fold',.55,(-25,20),.29,-6),
    (2490,'left_sleeve_fold',.68,(-20,30),.29,10),
    (2550,'hem_fold',.35,(-55,75),.08,5),
    (2650,'hem_fold',.96,(-50,40),.08,4),
    (2750,'press',1,(-45,25),.28,-14),
    (2800,'raised_open',0,(-15,20),0,0),
    (2840,'thumbs_up',0,(-20,25),0,0),
]


def profile(index):
    ref,action,progress,rolls,y,bank=SPECS[index]
    raised=action in ('raised','raised_open','thumbs_up')
    cloth_action={'flat_press':'spread'}.get(action,action)
    hand_action={'flat_press':'press'}.get(action,action)
    return dict(index=index,reference_frame=ref,action=action,cloth_action=cloth_action,
                hand_action=hand_action,progress=progress,rolls=rolls,contact_y=y,bank=bank,
                scene='raised' if raised else 'tabletop',
                reconstruction='qualitative representative-frame guided; unknown real K and depth')


def configure_fixture(root, wrist, side, spec):
    """Attach the mount to the inner cuff, independent of the observer/cable."""
    inside=1 if side=='left' else -1
    center=Vector((0,0,.064))
    q=Quaternion((0,1,0),inside*math.pi/2)
    for suffix in ('mount_base','round_mount_pivot','mount_pivot_rim','hinge_lower'):
        obj=bpy.data.objects[f'{side}_{suffix}']
        obj.location=center+q@(obj.location-center)+Vector((inside*.012,0,0))
        obj.rotation_mode='QUATERNION';obj.rotation_quaternion=q@obj.rotation_euler.to_quaternion()
    pad=bpy.data.objects[f'{side}_outer_pad']
    pad.location=(-inside*.027,.008,.083);pad.dimensions=(.032,.045,.024)
    y=Vector((inside,0,0));z=Vector((0,.8,-.6));x=y.cross(z)
    root.rotation_quaternion=Matrix((x,y,z)).transposed().to_quaternion()
    root.location=(inside*.086,-.005,.064)
    if spec['scene']=='raised':
        index=0 if side=='left' else 1;sign=-inside
        rest=(Matrix.Rotation(math.radians(sign*5),3,'Z') @ Matrix.Rotation(math.radians(75),3,'X') @ Matrix.Rotation(math.radians(spec['rolls'][index]),3,'Y'))
        if spec['action']=='raised':
            up=Vector((0,-.35,.94)).normalized();forward=Vector((0,.94,.35)).normalized()
        else:
            up=Vector((0,.35,.94)).normalized();forward=Vector((0,-.94,.35)).normalized()
        desired=Matrix((up.cross(forward),up,forward)).transposed()
        root.rotation_quaternion=(rest.inverted()@desired).to_quaternion()
    return (inside*.056,.018,.064)


def pose_wrist(wrist, side, spec, progress, frame, rng):
    index=0 if side=='left' else 1;sign=-1 if side=='left' else 1
    action=spec['action']
    spread=spec.get('variation_scale',1.0)
    roll=spec['rolls'][index]+float(rng.uniform(-5,5))*spread
    if spec['scene']=='raised':
        wrist.rotation_euler=(Matrix.Rotation(math.radians(sign*5),3,'Z') @ Matrix.Rotation(math.radians(75+rng.uniform(-3,3)*spread),3,'X') @ Matrix.Rotation(math.radians(roll),3,'Y')).to_euler()
        wrist.location=(sign*(.18+rng.uniform(-.008,.008)*spread),0,.46)
        target=None
    else:
        anchors={
            'button':((-.060,spec['contact_y'],.08),(.060,spec['contact_y'],.08)),
            'spread':((-.19,.50,.065),(.19,.50,.065)),
            'flat_press':((-.12,.29,.063),(.12,.32,.063)),
            'sleeve_grasp':((-.10,.37,.065),(.49,.29,.065)),
            'right_sleeve_fold':((-.10,.37,.065),(.48,.29,.065)),
            'left_sleeve_fold':((-.48,.29,.065),(.09,.39,.065)),
            'hem_fold':((-.14,.09,.06),(.14,.09,.06)),
            'press':((-.09,.27,.09),(.10,.40,.09)),
        }
        target=dynamic.deform_cloth([anchors[action][index]],spec['cloth_action'],progress)[0]
        # Keep fingers above the multi-layer folded hem.
        if action=='press':target[2]=max(target[2],.12)
        yaw=sign*(24 if action=='button' else 17)+rng.uniform(-5,5)*spread
        wrist.rotation_euler=Euler((math.radians(rng.uniform(-5,5)*spread),math.radians(roll),math.radians(yaw)),'XYZ')
        hand=bpy.data.objects[f'{side}_anatomical_hand_forearm']
        points=np.asarray([v.co[:] for v in hand.data.vertices])
        tip=points[points[:,1]>points[:,1].max()-.025].mean(axis=0)
        target+=rng.uniform([-.007*spread,-.007*spread,.004],[.007*spread,.007*spread,.012])
        wrist.location=Vector(target)-wrist.rotation_euler.to_matrix()@Vector(tip)
    for prop in ('location','rotation_euler'):wrist.keyframe_insert(data_path=prop,frame=frame)
    return dict(side=side,contact_anchor_m=None if target is None else target.tolist(),
                wrist_euler_degrees=[math.degrees(v) for v in wrist.rotation_euler])


def observer(scene,spec,rng):
    if spec['scene']=='raised':
        scene.camera.location=(0,-.62,.54);p.aim_at(scene.camera,(0,0,.50));scene.camera.data.lens=25
    else:
        scene.camera.location=Vector((rng.uniform(-.025,.025),-.49,.88+rng.uniform(-.035,.035)))
        p.aim_at(scene.camera,(0,.30,.035))
        scene.camera.rotation_euler.rotate_axis('Z',math.radians(spec['bank']+rng.uniform(-2,2)))
        scene.camera.data.lens=float(rng.uniform(27,30))
    if spec.get('variation_scale',1)>1:
        hard=spec.get('variation')=='hard'
        target=Vector((0,0,.50) if spec['scene']=='raised' else (0,.30,.035))
        radius=.14 if hard else .07
        scene.camera.location=target+(scene.camera.location-target)*float(rng.uniform(1-radius,1+radius))
        scene.camera.location.x+=float(rng.uniform(-radius/2,radius/2))
        p.aim_at(scene.camera,target)
        scene.camera.rotation_euler.rotate_axis('Z',math.radians(spec['bank']+rng.uniform(-8,8) if hard else spec['bank']+rng.uniform(-4,4)))


def refine_denim(mats):
    """Denim wash, flatter panels, longer sleeves, pocket flaps and seams."""
    mat=mats['denim'];nodes=mat.node_tree.nodes;links=mat.node_tree.links
    sh=nodes.get('Principled BSDF');sh.inputs['Roughness'].default_value=.91;sh.inputs['Specular'].default_value=.18
    pos=nodes.new('ShaderNodeNewGeometry')
    coarse=nodes.new('ShaderNodeTexNoise');coarse.inputs['Scale'].default_value=36;coarse.inputs['Detail'].default_value=5;coarse.inputs['Roughness'].default_value=.8
    links.new(pos.outputs['Position'],coarse.inputs['Vector'])
    ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.25;ramp.color_ramp.elements[0].color=(.014,.022,.038,1)
    ramp.color_ramp.elements[1].position=.75;ramp.color_ramp.elements[1].color=(.067,.085,.12,1)
    links.new(coarse.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],sh.inputs['Base Color'])
    waves=[]
    for axis in ('X','Y'):
        w=nodes.new('ShaderNodeTexWave');w.bands_direction=axis;w.inputs['Scale'].default_value=950;w.inputs['Distortion'].default_value=1.5
        links.new(pos.outputs['Position'],w.inputs['Vector']);waves.append(w)
    mix=nodes.new('ShaderNodeMath');mix.operation='MULTIPLY'
    for i,w in enumerate(waves):links.new(w.outputs['Fac'],mix.inputs[i])
    bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.4;bump.inputs['Distance'].default_value=.0003
    links.new(mix.outputs[0],bump.inputs['Height']);links.new(bump.outputs[0],sh.inputs['Normal'])
    # Stretch the existing sleeve region outward before creating fold shape keys.
    for obj in list(bpy.context.scene.objects):
        if obj.type=='MESH' and ('Denim sleeve' in obj.name):
            for v in obj.data.vertices:
                amount=max(abs(v.co.x)-.20,0)/.20
                v.co.x+=math.copysign(.15*amount,v.co.x);v.co.y+=.14*amount
        if obj.type=='CURVE' and 'Sleeve cuff' in obj.name:
            for sp in obj.data.splines:
                for v in sp.bezier_points:
                    amount=max(abs(v.co.x)-.20,0)/.20
                    v.co.x+=math.copysign(.15*amount,v.co.x);v.co.y+=.14*amount
    from scipy.interpolate import LinearNDInterpolator
    body=bpy.data.objects['Folded blue garment']
    points=np.array([v.co[:] for v in body.data.vertices])
    surface=LinearNDInterpolator(points[:,:2],points[:,2],fill_value=.03)
    # Existing panels stay attached: add flaps/seams using their own surface.
    thread=p.material('Faded double stitching',(.17,.15,.105,1),roughness=.95)
    for sign in (-1,1):
        pocket=bpy.data.objects[f'Chest pocket {sign}']
        xyz=np.asarray([v.co[:] for v in pocket.data.vertices]);z=float(xyz[:,2].max())+.001
        flap=p.rounded_box(f'Pocket flap {sign}',(.089,.028,.0018),(sign*.125,.458,z),mat,bevel=.003)
        p.curve_tube(f'Chest yoke seam {sign}',[(sign*x,.51,float(surface(sign*x,.51))+.002) for x in np.linspace(.018,.213,30)],.00055,thread,None)
    anchor=bpy.data.objects['Cloth stitches root']
    for yy in (.082,.098):
        p.curve_tube(f'Waistband double seam {yy}',[(x,yy,float(surface(x,yy))+.002) for x in np.linspace(-.215,.215,50)],.00045,thread,anchor)

    for obj in bpy.context.scene.objects:
        if obj.name.startswith('Shirt button '):obj.scale.x=obj.scale.y=1.6
    verts=[];faces=[];n=64
    for layer in (0,1):
        for j in range(n):
            a=2*math.pi*j/n
            verts.append((.078*math.cos(a),.61+.05*math.sin(a),.043+layer*(.026+.012*math.sin(a))))
    for j in range(n):faces.append((j,(j+1)%n,(j+1)%n+n,j+n))
    mesh=bpy.data.meshes.new('Standing collar mesh');mesh.from_pydata(verts,[],faces)
    collar=bpy.data.objects.new('Standing denim collar',mesh);bpy.context.collection.objects.link(collar)
    p.assign_material(collar,mat);p.smooth(collar)
    solid=collar.modifiers.new('Collar fabric','SOLIDIFY');solid.thickness=.0015
    skin=mats['skin'].node_tree.nodes.get('Principled BSDF')
    skin.inputs['Roughness'].default_value=.72;skin.inputs['Specular'].default_value=.18


def audit_fixture(obstacles):
    """Conservative camera OBB surface and mount segment clearance against arms."""
    bpy.context.view_layer.update();reports=[]
    # Coordinates are metres in device-root space, including the exact GLB bbox.
    bounds=np.array([[-.03525,-.02158,-.0159],[.03525,.02158,.0159]])
    samples=[]
    for axis in range(3):
        other=[j for j in range(3) if j!=axis]
        for value in bounds[:,axis]:
            for a in np.linspace(*bounds[:,other[0]],5):
                for b in np.linspace(*bounds[:,other[1]],5):
                    v=np.zeros(3);v[axis]=value;v[other]=[a,b];samples.append(v)
    for side in ('left','right'):
        root=bpy.data.objects[f'{side}_Action4_root'];mins={}
        for target,(tree,verts) in obstacles.items():
            wrist=bpy.data.objects[f'{target}_wrist_rig'];T=np.asarray(wrist.matrix_world.inverted()@root.matrix_world)
            points=(T@np.column_stack([samples,np.ones(len(samples))]).T).T[:,:3]
            limits=(verts.min(axis=0),verts.max(axis=0))
            minimum=min(cable.clearance(tree,pt,limits) for pt in points)
            if minimum<.001:raise RuntimeError(f'{side} camera approaches/intersects {target} arm: {minimum:.6f}')
            mins[target]=minimum
        reports.append(dict(side=side,camera_obb_sample_min_clearance_m=mins,mount='inner cuff; camera -Y bottom attached to hinge'))
    return reports


def aim_fixture_at_cloth(root, wrist, side, spec):
    """Estimate the fixed mount tilt from a work target, never from the observer."""
    if spec['scene']!='tabletop':return
    bpy.context.view_layer.update()
    target=Vector((0,spec['contact_y'] if spec['action']=='button' else .34,.04))
    center=root.matrix_world.translation
    z=(wrist.matrix_world.to_quaternion().inverted()@(target-center)).normalized()
    inside=Vector((1 if side=='left' else -1,0,0))
    y=(inside-z*inside.dot(z)).normalized()
    x=y.cross(z).normalized()
    root.rotation_quaternion=Matrix((x,y,z)).transposed().to_quaternion()
    bpy.context.view_layer.update()


def lens_direction_report(spec):
    if spec['scene']!='tabletop':return []
    target=Vector((0,spec['contact_y'] if spec['action']=='button' else .34,.04))
    reports=[]
    for side in ('left','right'):
        root=bpy.data.objects[f'{side}_Action4_root']
        direction=root.matrix_world.to_quaternion()@Vector((0,0,1))
        toward=(target-root.matrix_world.translation).normalized()
        cosine=direction.dot(toward)
        if cosine<.85:raise RuntimeError(f'{side} lens does not face the shirt work area: cosine={cosine:.4f}')
        reports.append(dict(side=side,work_target_m=list(target),lens_world_direction=list(direction),cosine_to_work_target=cosine))
    return reports


def audit_handedness(spec):
    if spec['scene']!='tabletop':return []
    result=[]
    for side,other in (('left','right'),('right','left')):
        hand=bpy.data.objects[f'{side}_anatomical_hand_forearm']
        thumb=sum((hand.data.vertices[i].co for i in hand['thumb_landmark_vertices']),Vector())/len(hand['thumb_landmark_vertices'])
        little=sum((hand.data.vertices[i].co for i in hand['little_landmark_vertices']),Vector())/len(hand['little_landmark_vertices'])
        wrist=bpy.data.objects[f'{side}_wrist_rig'];second=bpy.data.objects[f'{other}_wrist_rig']
        inward=(second.matrix_world.translation-wrist.matrix_world.translation).normalized()
        radial=hand.matrix_world.to_quaternion()@(thumb-little)
        projection=radial.dot(inward)
        if projection<=0:raise RuntimeError(f'{side} thumb points away from the inter-hand space: {projection:.6f} m')
        result.append(dict(side=side,thumb_to_little_inward_projection_m=projection,
                           thumb_world=list(hand.matrix_world@thumb),little_world=list(hand.matrix_world@little)))
    return result


def fit_wrist_assembly(root, wrist, side, spec, mats):
    """Place the whole fixture on the medial/lower cuff in the reference pose."""
    bpy.context.view_layer.update()
    inside=1 if side=='left' else -1
    rotation=wrist.matrix_world.to_quaternion()
    desired=Vector((inside,0,-.22)) if spec['scene']=='tabletop' else Vector((0,-1,0))
    local=rotation.inverted()@desired
    local.y=0
    normal=local.normalized()
    center=Vector((0,-.027,.060))
    tree,_=cable.hand_obstacle(side)
    hit,_,_,radius=tree.ray_cast(center,normal)
    if hit is None:raise RuntimeError('Mount normal missed wrist skin')
    q=Vector((0,0,1)).rotation_difference(normal)
    old_center=Vector((0,0,.064))
    canonical={
        'mount_base':((0,-.003,.094),(0,0,0)),
        'round_mount_pivot':((0,-.005,.101),(0,0,0)),
        'mount_pivot_rim':((0,-.005,.108),(0,0,0)),
        'hinge_lower':((0,.018,.110),(0,0,0)),
    }
    for suffix,(location,euler) in canonical.items():
        obj=bpy.data.objects[f'{side}_{suffix}']
        obj.location=center+q@(Vector(location)-old_center)+normal*(radius+.006-.030)
        obj.rotation_mode='QUATERNION';obj.rotation_quaternion=q@Euler(euler).to_quaternion()
    # The base must follow the cuff: a flat plate can cut the wrist between
    # otherwise clear corners. Build its contact surface from skin sections.
    base=bpy.data.objects[f'{side}_mount_base']
    angle=math.atan2(normal.z,normal.x)
    verts=[];faces=[];columns=25;rows=9
    for j,ypos in enumerate(np.linspace(-.054,.0,rows)):
        for i,theta in enumerate(np.linspace(angle-.62,angle+.62,columns)):
            radial=Vector((math.cos(theta),0,math.sin(theta)))
            origin=Vector((0,ypos,.060));point,_,_,distance=tree.ray_cast(origin,radial)
            if point is None:raise RuntimeError('Mount saddle missed wrist')
            verts.append(origin+radial*(distance+.0055))
            if i and j:
                k=j*columns+i;faces.append((k-columns-1,k-1,k,k-columns))
    mesh=bpy.data.meshes.new(f'{side}_conforming_mount_saddle');mesh.from_pydata(verts,[],faces);mesh.update()
    base.data=mesh;base.modifiers.clear();base.location=(0,0,0);base.rotation_quaternion=(1,0,0,0);base.scale=(1,1,1)
    p.assign_material(base,mats['mount']);p.smooth(base)
    solid=base.modifiers.new('Outward saddle thickness','SOLIDIFY');solid.thickness=.0045;solid.offset=1
    opposite=-normal;_,_,_,outside_radius=tree.ray_cast(center,opposite)
    pad=bpy.data.objects[f'{side}_outer_pad']
    pad.location=center+opposite*(outside_radius+.014)
    pad.rotation_mode='QUATERNION';pad.rotation_quaternion=Vector((0,0,1)).rotation_difference(opposite)
    pad.dimensions=(.034,.045,.018)
    anchor=center+normal*(radius+.025)+Vector((0,.008,0))
    root.location=anchor+normal*.028
    if spec['scene']=='tabletop':
        target=Vector((0,spec['contact_y'] if spec['action']=='button' else .34,.04))
        for _ in range(3):
            bpy.context.view_layer.update()
            z=(rotation.inverted()@(target-root.matrix_world.translation)).normalized()
            y=(normal-z*normal.dot(z)).normalized()
            root.rotation_quaternion=Matrix((y.cross(z),y,z)).transposed().to_quaternion()
            root.location=anchor+y*.041+Vector((0,-.012,0))
    else:
        up=Vector((0,-.35,.94) if spec['action']=='raised' else (0,.35,.94)).normalized()
        forward=Vector((0,.94,.35) if spec['action']=='raised' else (0,-.94,.35)).normalized()
        desired=Matrix((up.cross(forward),up,forward)).transposed()
        root.rotation_quaternion=(rotation.to_matrix().inverted()@desired).to_quaternion()
        # The front-facing raised tilt puts the tall cuff saddle closer to
        # the lower body edge; lift the bottom adapter along the device up axis.
        height = .042 if spec['action']=='raised' else .057
        # The thumbs-up palm needs extra radial space at the camera's lower
        # corner; reconnect_mount_and_cable rebuilds the bottom adapter below.
        palm_clearance = 0 if spec['action']=='raised' else .012
        root.location=anchor+(root.rotation_quaternion@Vector((0,height,0)))+normal*palm_clearance
    root['attachment_reference_frame']=spec['reference_frame']
    root['mount_radial_local']=list(normal)
    return tuple(anchor)


def audit_cuffs(obstacles):
    """Check the final solidified cuff, not only its construction vertices."""
    from mathutils.bvhtree import BVHTree
    bpy.context.view_layer.update();results=[]
    dg=bpy.context.evaluated_depsgraph_get()
    for side,(skin,vertices) in obstacles.items():
        wrist=bpy.data.objects[f'{side}_wrist_rig']
        obj=bpy.data.objects[f'{side}_fitted_cuff'];ev=obj.evaluated_get(dg);mesh=ev.to_mesh()
        T=wrist.matrix_world.inverted()@ev.matrix_world
        points=[T@v.co for v in mesh.vertices]
        faces=[tuple(poly.vertices) for poly in mesh.polygons]
        cuff_tree=BVHTree.FromPolygons(points,faces)
        intersections=len(skin.overlap(cuff_tree))
        samples=points+[sum((points[i] for i in f),Vector())/len(f) for f in faces]
        minimum=min(cable.clearance(skin,point,(vertices.min(0),vertices.max(0))) for point in samples)
        ev.to_mesh_clear()
        if intersections or minimum<.0005:
            raise RuntimeError(f'{side} cuff intersects skin: pairs={intersections}, clearance={minimum:.6f} m')
        results.append(dict(side=side,skin_intersection_triangle_pairs=intersections,min_sample_clearance_m=minimum,
                            thickness_direction='outward',method='evaluated solidified cuff BVH intersection + signed surface distance'))
    return results


def audit_mount_parts(obstacles):
    from mathutils.bvhtree import BVHTree
    bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();reports=[]
    for side,(skin,vertices) in obstacles.items():
        wrist=bpy.data.objects[f'{side}_wrist_rig']
        suffixes = ['mount_base','round_mount_pivot','mount_pivot_rim','hinge_lower','outer_pad','bottom_mount_plate','bottom_mount_neck','cable_connector','strain_relief']
        suffixes += [obj.name[len(side)+1:] for obj in bpy.data.objects if obj.name.startswith(f'{side}_mount_connection')]
        for suffix in suffixes:
            obj=bpy.data.objects[f'{side}_{suffix}'];ev=obj.evaluated_get(dg);mesh=ev.to_mesh()
            T=wrist.matrix_world.inverted()@ev.matrix_world
            points=[T@v.co for v in mesh.vertices];faces=[tuple(poly.vertices) for poly in mesh.polygons]
            pairs=len(skin.overlap(BVHTree.FromPolygons(points,faces)))
            probes=points+[sum((points[i] for i in f),Vector())/len(f) for f in faces]
            minimum=min(cable.clearance(skin,point,(vertices.min(0),vertices.max(0))) for point in probes)
            ev.to_mesh_clear()
            if pairs or minimum<.0005:raise RuntimeError(f'{side} {suffix} intersects/approaches skin: {pairs} pairs, {minimum:.6f} m')
            reports.append(dict(side=side,part=suffix,skin_intersection_triangle_pairs=pairs,min_sample_clearance_m=minimum))
    return reports


def audit_bottom_mount():
    """Reject any bracket that crosses the conservative camera body envelope."""
    from mathutils.bvhtree import BVHTree
    bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();reports=[]
    for side in ('left','right'):
        root=bpy.data.objects[f'{side}_Action4_root']
        body=bpy.data.objects[f'{side}_Action4_body']
        T=root.matrix_world.inverted()@body.matrix_world
        corners=[T@Vector(corner) for corner in body.bound_box]
        bounds=np.asarray(corners);lo=bounds.min(0);hi=bounds.max(0)
        vertices=[(x,y,z) for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]
        faces=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
        envelope=BVHTree.FromPolygons(vertices,faces)
        parts=[bpy.data.objects[f'{side}_{name}'] for name in ('mount_base','round_mount_pivot','mount_pivot_rim','hinge_lower','bottom_mount_plate','bottom_mount_neck')]
        parts += [o for o in bpy.data.objects if o.name.startswith(f'{side}_mount_connection')]
        for obj in parts:
            ev=obj.evaluated_get(dg);mesh=ev.to_mesh()
            transform=root.matrix_world.inverted()@ev.matrix_world
            points=[transform@v.co for v in mesh.vertices]
            polygons=[tuple(poly.vertices) for poly in mesh.polygons]
            pairs=len(envelope.overlap(BVHTree.FromPolygons(points,polygons)))
            xyz=np.asarray(points)
            contained=int(np.all((xyz>lo+1e-6)&(xyz<hi-1e-6),axis=1).sum())
            if pairs or contained:
                raise RuntimeError(f'{obj.name} crosses camera body envelope: {pairs} triangle pairs, {contained} internal vertices')
            if obj.name.endswith('bottom_mount_plate'):
                gap=float(lo[1]-xyz[:,1].max())
                if not 0<=gap<=.0005:
                    raise RuntimeError(f'{side} bottom plate not seated below camera: gap={gap:.6f} m')
            else:gap=None
            ev.to_mesh_clear()
            reports.append(dict(side=side,part=obj.name,body_envelope_intersection_pairs=pairs,
                                body_internal_vertices=contained,bottom_contact_gap_m=gap))
    return reports
