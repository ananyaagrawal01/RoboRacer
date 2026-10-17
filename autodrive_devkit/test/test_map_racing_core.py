import math
import unittest

from autodrive_roboracer.map_racing_core import ClosedPath, PathFollower, PathRecorder, wrap_angle


class MapRacingCoreTest(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(-math.pi / 2, wrap_angle(3 * math.pi / 2))

    def test_recorder_spacing_and_build(self):
        recorder = PathRecorder(spacing=0.15)
        for index in range(80):
            angle = 2 * math.pi * index / 80
            recorder.add(math.cos(angle), math.sin(angle))
        path = recorder.build(smoothing_radius=1)
        self.assertGreater(len(path.points), 20)

    def test_target_wraps_closed_path(self):
        path = ClosedPath([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertEqual(0, path.target_index(3, 0.5))

    def test_left_offset(self):
        path = ClosedPath([(0, 0), (1, 0), (1, 1), (0, 1)])
        shifted = path.offset_left(0.1)
        self.assertGreater(shifted.points[0][0], 0.0)
        self.assertGreater(shifted.points[0][1], 0.0)

    def test_follower_steers_toward_left_target(self):
        path = ClosedPath([(0, 0), (1, 1), (0, 2), (-1, 1)])
        follower = PathFollower(path, lookahead=0.5)
        throttle, steering, _ = follower.compute((0, 0), 0.0)
        self.assertGreater(steering, 0.0)
        self.assertGreater(throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
