"""Regression: a bottom endpoint cannot excuse a bracket crossing a side wall."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipIf(bpy is None, 'requires bpy')
class TestBottomMount(unittest.TestCase):
    def setUp(self):
        import blender_real_scene_preview as p
        from mathutils import Vector
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.p = p
        self.mat = p.material('test_mount', (.1, .1, .1, 1))
        for side in ('left', 'right'):
            root = bpy.data.objects.new(f'{side}_Action4_root', None)
            bpy.context.collection.objects.link(root)
            # A conservative envelope is intentional: the production audit
            # obtains these dimensions from the unchanged imported GLB.
            p.rounded_box(f'{side}_Action4_body', (.0705, .04315082, .03178656),
                          (0, 0, 0), self.mat, parent=root, bevel=0)
            for name in ('mount_base', 'round_mount_pivot', 'mount_pivot_rim', 'hinge_lower'):
                p.rounded_box(f'{side}_{name}', (.008, .008, .008),
                              (-.08, 0, 0), self.mat, parent=root, bevel=0)
            p.rounded_box(f'{side}_bottom_mount_plate', (.032, .006, .019),
                          (0, -.0247, .001), self.mat, parent=root, bevel=0)
            p.rounded_box(f'{side}_bottom_mount_neck', (.014, .012, .013),
                          (0, -.0337, .001), self.mat, parent=root, bevel=0)
            for i, (a, b) in enumerate([((-.06, 0, .001), (-.06, -.042, .001)),
                                       ((-.06, -.042, .001), (0, -.042, .001))]):
                a, b = Vector(a), Vector(b)
                p.rounded_box(f'{side}_mount_connection_{i}', (.010, .010, (b-a).length),
                              (a+b)*.5, self.mat, parent=root,
                              rotation=(b-a).to_track_quat('Z', 'Y').to_euler(), bevel=0)

    def test_bracket_below_bottom_clears_body(self):
        from action4_reference_profiles import audit_bottom_mount
        reports = audit_bottom_mount()
        self.assertTrue(reports)
        self.assertTrue(all(r['body_envelope_intersection_pairs'] == 0 for r in reports))
        gaps = [r['bottom_contact_gap_m'] for r in reports if r['bottom_contact_gap_m'] is not None]
        self.assertEqual(len(gaps), 2)
        self.assertTrue(all(0 <= gap <= .0005 for gap in gaps))

    def test_old_diagonal_is_rejected_even_with_bottom_endpoint(self):
        from mathutils import Vector
        from action4_reference_profiles import audit_bottom_mount
        a, b = Vector((-.06, 0, .001)), Vector((0, -.020, .001))
        self.p.rounded_box('left_mount_connection_bad', (.010, .010, (b-a).length),
                           (a+b)*.5, self.mat, parent=bpy.data.objects['left_Action4_root'],
                           rotation=(b-a).to_track_quat('Z', 'Y').to_euler(), bevel=0)
        with self.assertRaisesRegex(RuntimeError, 'crosses camera body envelope'):
            audit_bottom_mount()


if __name__ == '__main__':
    unittest.main()
