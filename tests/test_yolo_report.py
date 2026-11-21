import unittest

from scripts.report_yolo_action4 import score_frame


class TestYoloMatching(unittest.TestCase):
    def test_duplicates_are_false_positives(self):
        predictions = [{"xyxy": [0, 0, 10, 10]}, {"xyxy": [0, 0, 10, 10]}]
        targets = [{"xyxy": [0, 0, 10, 10]}, {"xyxy": [20, 20, 30, 30]}]
        result = score_frame(predictions, targets)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))

    def test_empty_predictions_count_all_misses(self):
        result = score_frame([], [{"xyxy": [0, 0, 10, 10]}])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))

    def test_low_iou_is_not_a_match(self):
        result = score_frame([{"xyxy": [8, 8, 18, 18]}], [{"xyxy": [0, 0, 10, 10]}])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
