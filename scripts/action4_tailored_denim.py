"""Sewn denim panels with flattened sleeve tubes and a continuous turned collar.

Reference-scale geometry, not a cloth/contact simulation. UVs are rest-space
metres so the fine twill moves with the fabric during existing fold shape keys.
"""
import math
import bpy
import numpy as np
from mathutils import Vector
import blender_real_scene_preview as p


def denim_material(mat):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    sh = nodes.get('Principled BSDF')
    sh.inputs['Roughness'].default_value = .88
    sh.inputs['Specular'].default_value = .2
    sh.inputs['Sheen'].default_value = .22
    uv = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 18
    noise.inputs['Detail'].default_value = 3
    links.new(uv.outputs['UV'], noise.inputs['Vector'])
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (.027, .038, .060, 1)
    ramp.color_ramp.elements[1].color = (.050, .065, .092, 1)
    links.new(noise.outputs['Fac'], ramp.inputs[0])
    fine = nodes.new('ShaderNodeTexNoise');fine.inputs['Scale'].default_value = 900
    links.new(uv.outputs['UV'], fine.inputs['Vector'])
    wave = nodes.new('ShaderNodeTexWave');wave.bands_direction = 'DIAGONAL'
    wave.inputs['Scale'].default_value = 140;wave.inputs['Distortion'].default_value = .3
    links.new(uv.outputs['UV'], wave.inputs['Vector'])
    mix = nodes.new('ShaderNodeMixRGB');mix.blend_type = 'MULTIPLY';mix.inputs[0].default_value = .15
    links.new(ramp.outputs[0], mix.inputs[1]);links.new(fine.outputs['Fac'], mix.inputs[2])
    links.new(mix.outputs[0], sh.inputs['Base Color'])
    bump = nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value = .25;bump.inputs['Distance'].default_value = .00018
    links.new(wave.outputs['Fac'], bump.inputs['Height']);links.new(bump.outputs[0], sh.inputs['Normal'])


def surface(x, y):
    return (.010 + .006*(1+math.sin(29*x+9*y))*math.exp(-((y-.34)/.27)**2)
            + .004*math.sin(63*x-17*y)**2*math.exp(-((abs(x)-.16)/.08)**2)
            + .003*math.cos(47*y+12*x)**2
            + .013*math.exp(-((abs(x)-(.22-.40*(y-.44)))/.014)**2-((y-.45)/.12)**2)
            + .006*math.sin(94*x+15*y)**2*math.exp(-((y-.15)/.10)**2-((abs(x)-.16)/.07)**2))


def mesh_grid(name, nu, nv, coordinate, mat, tube=False, thickness=.0011):
    vertices = [coordinate(i/(nu-1), j/(nv if tube else nv-1)) for i in range(nu) for j in range(nv)]
    faces = []
    for i in range(nu-1):
        for j in range(nv if tube else nv-1):
            k = (j+1)%nv
            faces.append((i*nv+j, (i+1)*nv+j, (i+1)*nv+k, i*nv+k))
    mesh = bpy.data.meshes.new(name);mesh.from_pydata(vertices, [], faces);mesh.update()
    uv = mesh.uv_layers.new(name='Rest fabric metres')
    for loop in mesh.loops:
        pt = vertices[loop.vertex_index];uv.data[loop.index].uv = (pt[0], pt[1])
    obj = bpy.data.objects.new(name, mesh);bpy.context.collection.objects.link(obj)
    p.assign_material(obj, mat);p.smooth(obj)
    if thickness:
        solid = obj.modifiers.new('Denim thickness', 'SOLIDIFY');solid.thickness = thickness;solid.offset = 0
    return obj


def stitching(name, points, mat):
    xyz = np.asarray(points);length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    curve = bpy.data.curves.new(name, 'CURVE');curve.dimensions = '3D';curve.bevel_depth = .00019;curve.bevel_resolution = 1
    for start in np.arange(0, length[-1]-.0016, .0028):
        spline = curve.splines.new('POLY');spline.points.add(1)
        for vertex, s in zip(spline.points, (start, start+.0016)):
            vertex.co = (*[np.interp(s, length, xyz[:, axis]) for axis in range(3)], 1)
    curve.materials.append(mat);obj = bpy.data.objects.new(name, curve);bpy.context.collection.objects.link(obj)
    return obj


def button(name, x, y, z, material, radius=.0056):
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=.0018, location=(x, y, z))
    obj = bpy.context.object;obj.name = name;p.assign_material(obj, material)
    dx = (surface(x+.001, y)-surface(x-.001, y))/.002
    dy = (surface(x, y+.001)-surface(x, y-.001))/.002
    obj.rotation_euler = Vector((-dx, -dy, 1)).to_track_quat('Z', 'Y').to_euler()
    bevel = obj.modifiers.new('Rounded metal rim', 'BEVEL');bevel.width = .0005;bevel.segments = 2
    return obj


def create_garment(mats, seed):
    mat = mats['denim'];denim_material(mat)
    thread = p.material('Subtle offwhite denim stitches', (.15, .135, .105, 1), roughness=.95)
    metal = p.material('Dull nickel snaps', (.21, .22, .20, 1), metallic=.7, roughness=.42)
    # Two front panels share a sewn centre opening and a shaped neckline.
    for sign in (-1, 1):
        def front(u, v):
            xx = .004+.228*u
            top = .666-.087*math.exp(-(xx/.082)**4)-.024*u*u
            y = .057+v*(top-.057)
            width = 1-.055*math.sin(math.pi*v)+.025*math.sin(2*math.pi*v)
            x = sign*xx*width
            return (x, y, surface(x, y))
        mesh_grid(f'Tailored front panel {sign}', 65, 89, front, mat)
        for offset in (.005, .010):
            pts = [front(1-offset/.23, v) for v in np.linspace(0, 1, 130)]
            stitching(f'Side double stitching {sign} {offset}', [(x,y,z+.0008) for x,y,z in pts], thread)
        # Slightly puffed pockets; the flap is a fabric grid that follows the body.
        center = sign*.119
        def pocket(u, v):
            x = center+(u-.5)*.086
            bottom = .351+.014*abs(2*u-1)
            y = bottom+v*(.465-bottom)
            z = surface(x,y)+.002+.0025*math.sin(math.pi*u)*math.sin(math.pi*v)
            return x,y,z
        mesh_grid(f'Sewn chest pocket {sign}', 25, 32, pocket, mat)
        outline = [pocket(u,.03) for u in np.linspace(.05,.95,32)]
        outline += [pocket(.95,v) for v in np.linspace(.03,.98,40)]
        stitching(f'Pocket edge stitches {sign}', [(x,y,z+.0008) for x,y,z in outline],thread)
        stitching(f'Pocket other edge {sign}', [(x,y,z+.0008) for x,y,z in [pocket(.05,v) for v in np.linspace(.03,.98,40)]],thread)
        def flap(u,v):
            x = center+(u-.5)*.091
            bottom = .442-.011*(1-abs(2*u-1))
            y = .474*(1-v)+bottom*v
            return x,y,surface(x,y)+.004+.0018*math.sin(math.pi*v)
        mesh_grid(f'Flexible pocket flap {sign}',25,14,flap,mat)
        stitching(f'Flap seam {sign}', [(x,y,z+.0008) for x,y,z in [flap(u,.88) for u in np.linspace(.045,.955,45)]],thread)
        button(f'Pocket snap {sign}',center,.441,surface(center,.441)+.006,metal,.0047)
        for yy in (.501,.506):
            stitching(f'Chest yoke {sign} {yy}',[(sign*x,yy,surface(sign*x,yy)+.0008) for x in np.linspace(.018,.220,85)],thread)
    # Placket, narrow hem and real stitch dashes replace floating gold curves.
    mesh_grid('Overlapping button placket',9,96,lambda u,v: ((u-.5)*.025+.004,.063+v*.510,surface((u-.5)*.025+.004,.063+v*.510)+.002),mat)
    mesh_grid('Denim waistband',65,9,lambda u,v: ((u-.5)*.46,.060+v*.029,surface((u-.5)*.46,.060+v*.029)+.0014),mat)
    for xx in (-.006,.014):
        stitching(f'Placket stitches {xx}',[(xx,y,surface(xx,y)+.003) for y in np.linspace(.073,.566,180)],thread)
    for yy in (.065,.084):
        stitching(f'Waist hem stitches {yy}',[(x,yy,surface(x,yy)+.002) for x in np.linspace(-.220,.220,160)],thread)
    for j,y in enumerate(np.linspace(.105,.551,6)):
        button(f'Shirt snap {j}',.003,y,surface(.003,y)+.004,metal)
    # Sleeves are tapered flattened tubes with openings and separate cuffs.
    for sign in (-1,1):
        def sleeve(t,v):
            angle=math.tau*v
            x=.224+.300*t;y=.545-.265*t+.015*math.sin(math.pi*t)
            cross_x=(.010*(1-t)+.033*t)*math.cos(angle)
            cross_y=(.095*(1-t)+.040*t)*math.cos(angle)
            height=.009+.008*math.sin(math.pi*t)
            wrinkle=.0035*math.sin(42*t+2*angle)*math.sin(math.pi*t)**2*(.6+.4*math.sin(angle))
            return (sign*(x+cross_x),y+cross_y,height+.005+height*math.sin(angle)+wrinkle)
        mesh_grid(f'Tapered sleeve tube {sign}',81,48,sleeve,mat,tube=True,thickness=.0009)
        def cuff(u,v):
            x,y,z=sleeve(.89+.11*u,v)
            return x,y,z+.001*math.sin(math.tau*v)
        mesh_grid(f'Sewn sleeve cuff {sign}',10,48,cuff,mat,tube=True)
        for t in (.905,.985):
            stitching(f'Cuff stitches {sign} {t}',[(x,y,z+.0007) for x,y,z in [sleeve(t,v) for v in np.linspace(.025,.475,85)]],thread)
        stitching(f'Sleeve outer seam {sign}',[(x,y,z+.0008) for x,y,z in [sleeve(t,.03) for t in np.linspace(.05,.98,150)]],thread)
    # Continuous turned collar: raised neck stand, soft roll, two front tips.
    def collar(u,v):
        a=math.radians(-65+310*u)
        width=.019+.029*max(-math.sin(a),0)
        x=(.077+width*v)*math.cos(a)
        y=.619+(.047+width*v)*math.sin(a)
        z=.034+.006*max(math.sin(a),0)-.018*v+.004*math.sin(math.pi*v)
        z+=.0015*math.sin(5*a)*math.sin(math.pi*v)
        return x,y,z
    mesh_grid('Turned denim collar',101,18,collar,mat,thickness=.0015)
    def stand(u,v):
        x,y,z=collar(u,0)
        return x,y,.012+(z-.012)*v
    mesh_grid('Curved collar stand',101,10,stand,mat)
    stitching('Collar edge stitching',[(x,y,z+.0009) for x,y,z in [collar(u,.91) for u in np.linspace(.015,.985,220)]],thread)
    mesh_grid('Inside neckline backing',20,20,lambda u,v: ((u-.5)*.15,.585+.075*v,.008),mat)
    sh=mats['skin'].node_tree.nodes.get('Principled BSDF');sh.inputs['Roughness'].default_value=.7;sh.inputs['Specular'].default_value=.18
