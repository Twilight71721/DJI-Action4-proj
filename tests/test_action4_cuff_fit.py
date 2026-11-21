"""Regression: cuffs must clear the evaluated hand after the tabletop rotation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import bpy
except ImportError:
    bpy=None

@unittest.skipIf(bpy is None,'requires bpy')
class TestCuffFit(unittest.TestCase):
    def test_solidified_cuff_clears_both_actual_wrists(self):
        import blender_real_scene_preview as p
        from action4_wrist_assets import anatomical_hand,orient_hand_for_tabletop,fitted_cuff_and_mount
        from action4_cable_geometry import hand_obstacle
        from action4_reference_profiles import audit_cuffs
        bpy.ops.wm.read_factory_settings(use_empty=True);m=p.build_materials()
        for side,pose in [('left','grasp'),('right','pinch')]:
            wrist=bpy.data.objects.new(f'{side}_wrist_rig',None);bpy.context.collection.objects.link(wrist)
            hand=anatomical_hand(wrist,m,pose,side);orient_hand_for_tabletop(hand)
            fitted_cuff_and_mount(wrist,m,side)
        reports=audit_cuffs({side:hand_obstacle(side) for side in ['left','right']})
        self.assertEqual(len(reports),2)
        for report in reports:
            self.assertEqual(report['skin_intersection_triangle_pairs'],0)
            self.assertGreater(report['min_sample_clearance_m'],.0005)
            self.assertEqual(report['thickness_direction'],'outward')

if __name__=='__main__':unittest.main()
