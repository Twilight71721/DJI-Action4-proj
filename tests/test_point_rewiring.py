import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_point_rewiring import cube_topologies, reconnect, edge_key
from bop_bb8.geometry import CORNER_SIGNS


class TestRewiring(unittest.TestCase):
    def test_recovers_connectivity_after_shuffling_without_moving_points(self):
        matrix=np.array([[103,17],[31,74],[-28,39]],float)
        canonical=CORNER_SIGNS@matrix+np.array([23.,18.])
        permutation=np.array([4,2,7,0,6,1,3,5])
        points=canonical[permutation]
        result=reconnect(points)
        expected=edge_key(np.argsort(permutation))
        self.assertEqual(tuple(map(tuple,result['edges'])),expected)
        np.testing.assert_array_equal(result['points_original'],points)
        self.assertLess(result['rewired_affine_rmse_px'],1e-10)

    def test_original_valid_topology_stays_and_all_graphs_are_unique(self):
        permutations,graphs=cube_topologies()
        self.assertEqual(len(graphs),840)
        self.assertEqual(len(set(graphs)),840)
        points=CORNER_SIGNS@np.array([[103,17],[31,74],[-28,39]],float)
        result=reconnect(points)
        self.assertEqual(result['changed_edges'],0)

    def test_collapse_is_not_hidden_by_inventing_a_box(self):
        points=np.zeros((8,2));result=reconnect(points)
        np.testing.assert_array_equal(result['points_original'],points)
        self.assertEqual(len(result['near_coincident_pairs']),28)
        self.assertFalse(result['points_moved'])


if __name__=='__main__':
    unittest.main()
