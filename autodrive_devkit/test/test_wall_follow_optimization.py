import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from wall_follow_optimization import TrialState


class TrialStateTest(unittest.TestCase):
    def test_records_exactly_three_laps(self):
        state = TrialState(4, 2)
        for count, lap_time in ((5, 9.2), (6, 9.1), (7, 9.3)):
            state.observe_collision(2)
            state.observe_lap(count, lap_time)
        self.assertEqual("complete", state.status)
        self.assertEqual([9.2, 9.1, 9.3], state.lap_times)

    def test_any_collision_rejects(self):
        state = TrialState(0, 0)
        state.observe_lap(1, 9.2)
        state.observe_collision(1)
        self.assertEqual("rejected", state.status)
        self.assertIn("collision", state.reason)

    def test_skipped_lap_rejects(self):
        state = TrialState(0, 0)
        state.observe_lap(2, 9.2)
        self.assertEqual("rejected", state.status)

    def test_invalid_time_rejects(self):
        state = TrialState(0, 0)
        state.observe_lap(1, 0.0)
        self.assertEqual("rejected", state.status)


if __name__ == "__main__":
    unittest.main()

