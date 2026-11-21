import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from monitor_action4_render import completed_counts, diagnose, process_matches, progress_due, update_incident


class RenderMonitorTests(unittest.TestCase):
    def test_progress_interval_survives_restart(self):
        memory = {'last_progress_notification': 1000}
        self.assertFalse(progress_due(memory, 1299, 300))
        self.assertTrue(progress_due(memory, 1300, 300))

    def test_counts_only_committed_batches_with_variable_sizes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'completed').mkdir()
            (root / 'completed/0000.json').write_text('{"frames": 5}')
            (root / 'completed/0001.json').write_text('{"frames": 20}')
            (root / 'completed/0002.tmp').write_text('{"frames": 20}')
            self.assertEqual(completed_counts(root), (2, 25))

    def test_failure_and_dead_process(self):
        self.assertIn('失败', diagnose({'state': 'failed'}, True, 0, 1200))
        self.assertIn('退出', diagnose({'state': 'rendering'}, False, 0, 1200))

    def test_progress_and_stall(self):
        self.assertIsNone(diagnose({'state': 'rendering'}, True, 1199, 1200))
        self.assertIn('疑似', diagnose({'state': 'rendering'}, True, 1201, 1200))
        self.assertIsNone(diagnose({'state': 'final_validation'}, True, 1800, 1200))

    def test_completion_does_not_alert_when_process_exits(self):
        self.assertIsNone(diagnose({'state': 'complete'}, False, 9000, 1200))

    def test_dedup_and_recovery(self):
        memory = {}
        self.assertTrue(update_incident(memory, 'failed', [1, 20]))
        memory['notified'] = True
        self.assertFalse(update_incident(memory, 'different error', [1, 20]))
        self.assertFalse(update_incident(memory, None, [1, 20]))
        self.assertTrue(update_incident(memory, 'failed again', [1, 20]))
        memory['notified'] = True
        self.assertTrue(update_incident(memory, 'new run failed', [2, 20]))

    def test_unrelated_or_absent_process_rejected(self):
        import os
        self.assertFalse(process_matches(os.getpid(), Path('/tmp')))
        self.assertFalse(process_matches(None, Path('/tmp')))


if __name__ == '__main__':
    unittest.main()
