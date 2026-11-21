import unittest

from scripts.prepare_yolo_bop import clip_box, normalize_box, scene_assignments


class TestYoloBop(unittest.TestCase):
    def test_clipped_box_round_trip(self):
        box = clip_box([-10, 20, 50, 100], 100, 80)
        self.assertEqual(box, [0, 20, 40, 80])
        cx, cy, w, h = normalize_box(box, 100, 80)
        self.assertEqual([(cx - w / 2) * 100, (cy - h / 2) * 80,
                          (cx + w / 2) * 100, (cy + h / 2) * 80], box)

    def test_offscreen_and_invalid_boxes(self):
        self.assertIsNone(clip_box([110, 20, 20, 10], 100, 80))
        for box in ([1, 2, -1, 3], [1, 2, float("nan"), 3]):
            with self.assertRaises(ValueError):
                clip_box(box, 100, 80)

    def test_reject_scene_leakage_and_unassigned_scene(self):
        config = {"groups": [dict(name="a", split="train", scenes=["0"]),
                             dict(name="b", split="val", scenes=["0"])]}
        with self.assertRaises(ValueError):
            scene_assignments(config, ["0"])
        config["groups"][1]["scenes"] = ["1"]
        with self.assertRaises(ValueError):
            scene_assignments(config, ["0", "1", "2"])
        self.assertEqual(scene_assignments(config, ["0", "1"])["1"], ("val", "b"))


if __name__ == "__main__":
    unittest.main()
