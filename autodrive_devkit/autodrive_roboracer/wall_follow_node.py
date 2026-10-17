"""ROS 2 entrypoint for the Penn RoboRacer LiDAR wall-following challenge."""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from autodrive_roboracer.wall_follow_core import Config, WallFollower


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        defaults = Config()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        config = Config(**{name: self.get_parameter(name).value
                           for name in defaults.__dataclass_fields__})
        self.controller = WallFollower(config)
        self.steer_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.on_scan,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                       reliability=ReliabilityPolicy.BEST_EFFORT))
        self.last_scan_at = None
        self.last_stamp = None
        self.last_log_at = 0.0
        self.watchdog = self.create_timer(0.10, self.check_watchdog)
        self.get_logger().info(f'Wall follower ready: {config.wall_side} wall, '
                               f'target {config.desired_distance:.2f} m')

    def publish(self, throttle, steering):
        self.throttle_pub.publish(Float32(data=float(throttle)))
        self.steer_pub.publish(Float32(data=float(steering)))

    def on_scan(self, msg):
        now = time.monotonic()
        dt = 0.05 if self.last_stamp is None else max(0.01, min(now - self.last_stamp, 0.20))
        self.last_stamp = now
        self.last_scan_at = now
        scan = (msg.ranges, msg.angle_min, msg.angle_increment)
        throttle, steering, diag = self.controller.compute(scan, dt)
        self.publish(throttle, steering)
        if now - self.last_log_at > 2.0:
            self.get_logger().info(
                f"mode={diag['mode']} front={diag['front']} "
                f"side={diag['side']} throttle={throttle:.2f} steer={steering:.2f}")
            self.last_log_at = now

    def check_watchdog(self):
        if self.last_scan_at is None or time.monotonic() - self.last_scan_at > 0.35:
            self.controller.reset()
            self.publish(0, 0)

    def stop(self):
        self.publish(0, 0)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
