"""Semantic left/right checks on the actual hand asset, not proxy primitives."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import bpy
except ImportError:
    bpy=None

@unittest.skipIf(bpy is None,'requires the project bpy environment')
class TestHandAnatomy(unittest.TestCase):
    def test_tabletop_thumbs_point_medially_without_changing_surface_orientation(self):
        import blender_real_scene_preview as p
        from action4_wrist_assets import anatomical_hand,orient_hand_for_tabletop
        bpy.ops.wm.read_factory_settings(use_empty=True)
        materials=p.build_materials()
        for pose in ('open','pinch','grasp'):
            for side,sign in (('left',1),('right',-1)):
                root=bpy.data.objects.new(f'{side}_{pose}',None);bpy.context.collection.objects.link(root)
                hand=anatomical_hand(root,materials,pose,side)
                before=np.array([v.co[:] for v in hand.data.vertices])
                winding=[tuple(poly.vertices) for poly in hand.data.polygons]
                orient_hand_for_tabletop(hand)
                after=np.array([v.co[:] for v in hand.data.vertices])
                thumb=after[list(hand['thumb_landmark_vertices'])].mean(0)
                little=after[list(hand['little_landmark_vertices'])].mean(0)
                self.assertGreater(sign*(thumb[0]-little[0]),.04,(pose,side))
                self.assertEqual(winding,[tuple(poly.vertices) for poly in hand.data.polygons])
                # Recover the linear transform independently and check handedness.
                T=np.linalg.lstsq(np.column_stack([before,np.ones(len(before))]),after,rcond=None)[0]
                self.assertAlmostEqual(np.linalg.det(T[:3]),1.,places=5)
                self.assertLess(np.max(np.abs(np.column_stack([before,np.ones(len(before))])@T-after)),1e-6)

    def test_forearm_refinement_preserves_hand_and_cuff_attachment_region(self):
        import blender_real_scene_preview as p
        from action4_wrist_assets import anatomical_hand, shape_reference_forearm
        bpy.ops.wm.read_factory_settings(use_empty=True)
        materials=p.build_materials()
        for side in ('left','right'):
            root=bpy.data.objects.new(side,None);bpy.context.collection.objects.link(root)
            hand=anatomical_hand(root,materials,'pinch',side)
            before=np.array([v.co[:] for v in hand.data.vertices])
            protected=before[:,1]>=-.10
            shape_reference_forearm(hand,side)
            after=np.array([v.co[:] for v in hand.data.vertices])
            np.testing.assert_array_equal(before[protected],after[protected])
            self.assertTrue(np.isfinite(after).all())
            self.assertLess(np.abs(after[~protected,0]).max(),.075)
            self.assertGreater(np.max(np.abs(after[~protected]-before[~protected])),.01)

    def test_source_contains_only_hand_and_forearm_near_wrist(self):
        from action4_wrist_assets import read_hand_source
        vertices,faces,_,_=read_hand_source()
        self.assertGreater(len(vertices),1000)
        self.assertLess(np.abs(vertices[:,0]).max(),.12)
        self.assertLess(np.abs(vertices[:,2]).max(),.12)
        self.assertLess(vertices[:,1].max(),.23)
        self.assertEqual(set(np.asarray([i for f in faces for i in f])),set(range(len(vertices))))

if __name__=='__main__':unittest.main()
