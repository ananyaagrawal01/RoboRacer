"""Discover a clean lap with wall following, then replay it predictively."""

import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32

from autodrive_roboracer.map_racing_core import ClosedPath, PathFollower, PathRecorder, clamp
from autodrive_roboracer.wall_follow_core import Config, WallFollower, robust_sector


class MapRacingNode(Node):
    def __init__(self):
        super().__init__("map_racing_node")
        parameters = {
            "map_file": "/home/autodrive_devkit/racing_line.json",
            "force_discovery": False,
            "acquisition_lap": True,
            "stop_after_racing_laps": 0,
            "record_spacing": 0.08,
            "smoothing_radius": 2,
            "discovery_lookahead": 0.70,
            "discovery_front_slow_distance": 2.10,
            "discovery_throttle": 0.25,
            "path_lookahead": 0.65,
            "path_left_offset": 0.10,
            "steering_gain": 0.55,
            "max_steer": 0.78,
            "race_throttle": 0.34,
            "min_throttle": 0.12,
            "curvature_slowdown": 1.8,
            "emergency_distance": 0.42,
            "side_safety_distance": 0.38,
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in parameters}

        config = Config(lookahead=self.p["discovery_lookahead"],
                        front_slow_distance=self.p["discovery_front_slow_distance"],
                        cruise_throttle=self.p["discovery_throttle"])
        self.wall_follower = WallFollower(config)
        self.recorder = PathRecorder(self.p["record_spacing"])
        self.path_follower = None
        self.pending_path = None
        self.mode = "discovery"
        self.position = None
        self.yaw = None
        self.lap_count = None
        self.start_lap = None
        self.recording_full_lap = False
        self.racing_start_lap = None
        self.finished = False
        self.last_reported_lap = None
        self.collision_start = None
        self.collision_count = None
        self.last_scan_at = None
        self.last_control_at = None
        self.faulted = False

        map_file = self.p["map_file"]
        if os.path.exists(map_file) and not self.p["force_discovery"]:
            loaded_path = ClosedPath.load(map_file)
            if self.p["acquisition_lap"]:
                self.pending_path = loaded_path
                self.mode = "acquisition"
            else:
                self._activate_path(loaded_path)

        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10)
        self.steering_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.create_subscription(Odometry, "/autodrive/roboracer_1/odom",
                                 self.on_odom, 10)
        self.create_subscription(Int32, "/autodrive/roboracer_1/lap_count",
                                 self.on_lap, 10)
        self.create_subscription(Int32, "/autodrive/roboracer_1/collision_count",
                                 self.on_collision, 10)
        self.create_subscription(LaserScan, "/autodrive/roboracer_1/lidar",
                                 self.on_scan, 10)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info(f"map racer ready in {self.mode} mode")

    def _activate_path(self, path):
        path = path.offset_left(self.p["path_left_offset"])
        self.path_follower = PathFollower(
            path, lookahead=self.p["path_lookahead"],
            steering_gain=self.p["steering_gain"], max_steer=self.p["max_steer"],
            cruise_throttle=self.p["race_throttle"],
            min_throttle=self.p["min_throttle"],
            curvature_slowdown=self.p["curvature_slowdown"])
        self.mode = "racing"
        self.racing_start_lap = self.lap_count

    def publish(self, throttle, steering):
        self.throttle_pub.publish(Float32(data=float(throttle)))
        self.steering_pub.publish(Float32(data=float(steering)))

    def stop(self):
        self.publish(0.0, 0.0)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = (float(p.x), float(p.y))
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.mode == "discovery":
            self.recorder.add(p.x, p.y)

    def on_lap(self, msg):
        self.lap_count = int(msg.data)
        if self.start_lap is None:
            self.start_lap = self.lap_count
            return
        if self.mode == "acquisition" and self.lap_count > self.start_lap:
            if (self.lap_count != self.start_lap + 1
                    or self.collision_count != self.collision_start):
                self.get_logger().error("acquisition lap was not clean; stopping")
                self.faulted = True
                self.stop()
                return
            self._activate_path(self.pending_path)
            self.get_logger().info("clean acquisition lap complete; switching to racing mode")
            return
        if (self.mode == "racing" and self.racing_start_lap is not None
                and self.lap_count > self.racing_start_lap
                and self.lap_count != self.last_reported_lap):
            self.last_reported_lap = self.lap_count
            completed = self.lap_count - self.racing_start_lap
            self.get_logger().info(f"completed mapped racing lap {completed}")
            limit = int(self.p["stop_after_racing_laps"])
            if limit > 0 and completed >= limit:
                self.finished = True
                self.stop()
                self.get_logger().info("mapped racing validation complete; vehicle stopped")
            return
        if self.mode != "discovery" or self.lap_count <= self.start_lap:
            return
        if self.lap_count != self.start_lap + 1:
            self.get_logger().error("lap counter skipped; refusing discovered path")
            self.faulted = True
            self.stop()
            return
        if self.collision_count != self.collision_start:
            self.get_logger().error("discovery lap had a collision; refusing path")
            self.faulted = True
            self.stop()
            return
        if not self.recording_full_lap:
            # The node may have started after a checkpoint respawn, so the
            # first finish crossing can represent only a partial lap. Begin a
            # fresh recording at that crossing and accept only the next lap.
            self.recorder = PathRecorder(self.p["record_spacing"])
            self.start_lap = self.lap_count
            self.collision_start = self.collision_count
            self.recording_full_lap = True
            self.get_logger().info("finish line acquired; recording one complete clean lap")
            return
        try:
            path = self.recorder.build(self.p["smoothing_radius"])
            path.save(self.p["map_file"], {"source": "clean discovery lap"})
            self._activate_path(path)
            self.get_logger().info(
                f"saved {len(path.points)} path points; switching to racing mode")
        except (OSError, ValueError) as error:
            self.get_logger().error(f"could not create racing path: {error}")
            self.faulted = True
            self.stop()

    def on_collision(self, msg):
        self.collision_count = int(msg.data)
        if self.collision_start is None:
            self.collision_start = self.collision_count
        elif self.collision_count > self.collision_start and not self.faulted:
            self.faulted = True
            self.get_logger().error("collision detected; stopping map racer")
            self.stop()

    def on_scan(self, msg):
        self.last_scan_at = time.monotonic()
        if self.faulted or self.finished:
            self.stop()
            return
        if self.position is None or self.yaw is None or self.collision_count is None:
            self.stop()
            return
        if self.collision_count > self.collision_start:
            self.stop()
            return
        scan = (msg.ranges, msg.angle_min, msg.angle_increment)
        front = robust_sector(scan, 0, 9)
        if front is None or front <= self.p["emergency_distance"]:
            self.stop()
            return
        if self.mode in ("discovery", "acquisition"):
            now = time.monotonic()
            dt = 0.025 if self.last_control_at is None else now - self.last_control_at
            self.last_control_at = now
            throttle, steering, _ = self.wall_follower.compute(scan, dt)
        else:
            throttle, steering, _ = self.path_follower.compute(self.position, self.yaw)
            left = robust_sector(scan, 90, 6)
            right = robust_sector(scan, -90, 6)
            safety = self.p["side_safety_distance"]
            if left is not None and left < safety:
                steering = min(steering, -0.35)
                throttle = min(throttle, self.p["min_throttle"])
            if right is not None and right < safety:
                steering = max(steering, 0.35)
                throttle = min(throttle, self.p["min_throttle"])
            if front < self.p["discovery_front_slow_distance"]:
                throttle *= clamp((front - self.p["emergency_distance"]) / 1.2, 0.0, 1.0)
        self.publish(throttle, steering)

    def watchdog(self):
        if self.last_scan_at is None or time.monotonic() - self.last_scan_at > 0.35:
            self.stop()


def main(args=None):
    rclpy.init(args=args)
    node = MapRacingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
