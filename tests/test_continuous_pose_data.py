import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_continuous_pose_bank import sample_records
from prepare_online_pose_backgrounds import split_backgrounds


class TestContinuousPoseData(unittest.TestCase):
    def test_appearance_and_light_changes_preserve_pose_sequence(self):
        anchors=[dict(R=np.eye(3).tolist())]
        base=sample_records(anchors,np.zeros(3),(128,32))
        variant=sample_records(anchors,np.zeros(3),(128,32),appearance_seed=41,light_seed=42)
        self.assertTrue(any(a['appearance']!=b['appearance'] for a,b in zip(base,variant)))
        self.assertTrue(any(a['light_location']!=b['light_location'] for a,b in zip(base,variant)))
        for a,b in zip(base,variant):
            np.testing.assert_array_equal(a['R'],b['R'])
            np.testing.assert_array_equal(a['t_mm'],b['t_mm'])
            R=np.array(a['R']);np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R),1.)
        rotation_keys=[tuple(np.array(r['R']).round(8).flat) for r in base]
        self.assertEqual(len(set(rotation_keys)),len(base))

    def test_no_fixed_roll_to_appearance_mapping(self):
        from scipy.spatial.transform import Rotation
        records=sample_records([dict(R=np.eye(3).tolist())],np.zeros(3),(2048,128))
        for side in ['left','right']:
            rs=[r['R'] for r in records if r['appearance']==side and r['family']=='uniform_so3']
            angles=Rotation.from_matrix(rs).as_euler('xyz',degrees=True)[:,2]
            bins=np.histogram(angles,[-180,-90,0,90,180])[0]
            self.assertTrue((bins>50).all())

    def test_background_frame_groups_and_hashes_do_not_leak(self):
        pool=[dict(sha256=str(i),background_group=str(i//8)) for i in range(200)]
        pool.append(dict(sha256='0',background_group='duplicate_in_other_source'))
        train,hold=split_backgrounds(pool,20260927)
        self.assertFalse({pool[i]['sha256'] for i in train}&{pool[i]['sha256'] for i in hold})
        self.assertFalse({pool[i]['background_group'] for i in train}&{pool[i]['background_group'] for i in hold})


if __name__=='__main__':unittest.main()
