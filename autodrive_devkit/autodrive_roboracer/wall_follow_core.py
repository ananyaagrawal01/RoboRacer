"""ROS-independent LiDAR wall follower and safety controller.

Angles use the ROS convention: positive is left, 0 is forward.
Commands are normalized to [-1, 1]. Tune on the practice course.
"""
from dataclasses import dataclass
import math
import statistics


def clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class Config:
    wall_side: str = 'right'
    desired_distance: float = 0.85
    lookahead: float = 0.75
    kp: float = 1.5
    kd: float = 0.20
    max_steer: float = 0.78
    cruise_throttle: float = 0.28
    min_throttle: float = 0.10
    corner_throttle: float = 0.13
    front_slow_distance: float = 2.2
    front_stop_distance: float = 0.65
    emergency_distance: float = 0.42
    max_throttle_slew: float = 0.8  # normalized command per second
    min_valid_range: float = 0.06
    max_valid_range: float = 10.0


def robust_sector(scan, center_degrees, half_width_degrees=4.0):
    """Return median finite measured range in a scan sector, or None."""
    ranges, angle_min, increment = scan
    if not ranges or not math.isfinite(increment) or increment <= 0:
        return None
    indices = [round((math.radians(a) - angle_min) / increment)
               for a in (center_degrees - half_width_degrees,
                         center_degrees - half_width_degrees / 2,
                         center_degrees,
                         center_degrees + half_width_degrees / 2,
                         center_degrees + half_width_degrees)]
    vals = []
    for i in indices:
        if 0 <= i < len(ranges):
            x = ranges[i]
            if math.isfinite(x) and x >= 0.06 and x <= 10.0:
                vals.append(float(x))
    return statistics.median(vals) if vals else None


class WallFollower:
    def __init__(self, config=None):
        self.cfg = config or Config()
        if self.cfg.wall_side not in ('left', 'right'):
            raise ValueError('wall_side must be left or right')
        self.previous_error = None
        self.previous_throttle = 0.0

    def reset(self):
        self.previous_error = None
        self.previous_throttle = 0.0

    def compute(self, scan, dt=0.05):
        """Compute (throttle, steering, diagnostic dict). Fail closed on bad data."""
        c = self.cfg
        dt = clamp(float(dt), 0.01, 0.20)
        direction = 1 if c.wall_side == 'left' else -1
        side = robust_sector(scan, direction * 90)
        forward_side = robust_sector(scan, direction * 45)
        front = robust_sector(scan, 0, 9)
        front_left = robust_sector(scan, 25, 7)
        front_right = robust_sector(scan, -25, 7)
        diag = {'wall_side': c.wall_side, 'side': side, 'side_forward': forward_side,
                'front': front, 'front_left': front_left, 'front_right': front_right,
                'mode': 'tracking'}

        # Do not drive when crucial measurements are absent.
        if front is None:
            self.reset()
            diag['mode'] = 'no_front_scan'
            return 0.0, 0.0, diag
        if front < c.emergency_distance:
            self.reset()
            diag['mode'] = 'emergency_stop'
            return 0.0, 0.0, diag

        if side is not None and forward_side is not None:
            # Alpha is the angle between wall and car heading.
            theta = math.radians(45)
            alpha = math.atan2(forward_side * math.cos(theta) - side,
                               forward_side * math.sin(theta))
            projected = side * math.cos(alpha) + c.lookahead * math.sin(alpha)
            error = c.desired_distance - projected
            derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
            self.previous_error = error
            # For right wall, being too far from wall means steer right (negative).
            steering = -direction * (c.kp * error + c.kd * derivative)
            diag.update(error=error, projected_distance=projected, alpha=alpha)
        else:
            self.previous_error = None
            diag['mode'] = 'wall_missing'
            # Search gently for wall; if ahead is constrained, steer toward open space.
            steering = direction * 0.24

        # Forward obstacle avoidance: steer toward side with more clearance.
        # Front sectors suppress risk when wall geometry is misleading at corners.
        if front < c.front_slow_distance:
            left_clear = front_left if front_left is not None else 0.0
            right_clear = front_right if front_right is not None else 0.0
            avoid_sign = 1.0 if left_clear > right_clear else -1.0
            if abs(left_clear - right_clear) < 0.12:
                avoid_sign = -direction  # Prefer to turn away from followed wall.
            strength = clamp((c.front_slow_distance - front) /
                             max(0.01, c.front_slow_distance - c.front_stop_distance), 0, 1)
            steering = steering * (1 - strength) + avoid_sign * c.max_steer * strength
            diag['mode'] = 'obstacle_avoidance'

        steering = clamp(steering, -c.max_steer, c.max_steer)
        # Corner and obstacle speed scheduling; conservative baseline.
        steer_factor = 1 - 0.65 * abs(steering) / max(c.max_steer, 0.01)
        target_throttle = max(c.min_throttle, c.cruise_throttle * steer_factor)
        if front < c.front_slow_distance:
            clearance = clamp((front - c.front_stop_distance) /
                              max(0.01, c.front_slow_distance - c.front_stop_distance), 0, 1)
            target_throttle = min(target_throttle, c.corner_throttle * clearance)
        if front <= c.front_stop_distance:
            target_throttle = 0.0
            diag['mode'] = 'front_stop'
        # Allow immediate braking; only limit acceleration.
        throttle = min(target_throttle, self.previous_throttle + c.max_throttle_slew * dt)
        self.previous_throttle = throttle
        return throttle, steering, diag
