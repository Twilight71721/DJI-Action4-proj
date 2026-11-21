"""Expanded sampling must reject cross-arm fixtures and cable/cuff collisions."""
import sys
import unittest
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if bpy is not None:
    from action4_extended_checks import prepare,audit


@unittest.skipIf(bpy is None,'requires bpy')
class TestExtendedChecks(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        for side in ('left','right'):
            root=bpy.data.objects.new(side+'_wrist_rig',None)
            bpy.context.collection.objects.link(root)
            bpy.ops.mesh.primitive_cube_add(size=.04)
            obj=bpy.context.object;obj.name=side+'_test_cuff';obj.parent=root
        bpy.data.objects['right_wrist_rig'].location.x=.20

    def test_separated_fixtures_pass(self):
        self.assertEqual(audit(prepare(),[])['cross_limb_intersections'],0)

    def test_overlapping_fixtures_rejected(self):
        bpy.data.objects['right_wrist_rig'].location.x=.01
        with self.assertRaisesRegex(RuntimeError,'Cross-limb intersection'):
            audit(prepare(),[])

    def test_cable_through_cuff_rejected(self):
        route=dict(side='left',points=np.array([[x,0,0] for x in np.linspace(-.04,.04,161)]),radius=.0018)
        with self.assertRaisesRegex(RuntimeError,'cable intersects/approaches'):
            audit(prepare(),[route])


if __name__=='__main__':unittest.main()
