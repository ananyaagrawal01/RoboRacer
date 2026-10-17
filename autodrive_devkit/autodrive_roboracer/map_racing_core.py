"""ROS-independent trajectory recording and pursuit control."""

import json
import math


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PathRecorder:
    def __init__(self, spacing=0.08):
        self.spacing = spacing
        self.points = []

    def add(self, x, y):
        point = (float(x), float(y))
        if not self.points or math.dist(self.points[-1], point) >= self.spacing:
            self.points.append(point)
            return True
        return False

    def build(self, smoothing_radius=2):
        if len(self.points) < 20:
            raise ValueError("too few trajectory points for a closed racing path")
        points = self.points[:]
        radius = max(0, int(smoothing_radius))
        if radius:
            smoothed = []
            count = len(points)
            for index in range(count):
                window = [points[(index + offset) % count]
                          for offset in range(-radius, radius + 1)]
                smoothed.append((sum(p[0] for p in window) / len(window),
                                 sum(p[1] for p in window) / len(window)))
            points = smoothed
        return ClosedPath(points)


class ClosedPath:
    def __init__(self, points):
        self.points = [(float(x), float(y)) for x, y in points]
        if len(self.points) < 3:
            raise ValueError("closed path needs at least three points")

    def nearest_index(self, position, hint=None, backward=12, forward=35):
        x, y = position
        if hint is None:
            candidates = range(len(self.points))
        else:
            candidates = [(hint + offset) % len(self.points)
                          for offset in range(-backward, forward + 1)]
        return min(candidates,
                   key=lambda i: (self.points[i][0] - x) ** 2
                   + (self.points[i][1] - y) ** 2)

    def offset_left(self, distance):
        shifted = []
        count = len(self.points)
        for index, point in enumerate(self.points):
            previous = self.points[(index - 1) % count]
            following = self.points[(index + 1) % count]
            tx, ty = following[0] - previous[0], following[1] - previous[1]
            length = math.hypot(tx, ty)
            if length <= 1e-9:
                shifted.append(point)
            else:
                shifted.append((point[0] - distance * ty / length,
                                point[1] + distance * tx / length))
        return ClosedPath(shifted)

    def target_index(self, start, distance):
        travelled = 0.0
        index = start
        for _ in range(len(self.points)):
            nxt = (index + 1) % len(self.points)
            travelled += math.dist(self.points[index], self.points[nxt])
            index = nxt
            if travelled >= distance:
                break
        return index

    def curvature_ahead(self, start, distance=1.2):
        middle = self.target_index(start, distance * 0.5)
        end = self.target_index(start, distance)
        p0, p1, p2 = self.points[start], self.points[middle], self.points[end]
        heading_a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        heading_b = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return abs(wrap_angle(heading_b - heading_a)) / max(distance, 0.01)

    def save(self, path, metadata=None):
        with open(path, "w") as handle:
            json.dump({"version": 1, "points": self.points,
                       "metadata": metadata or {}}, handle, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            data = json.load(handle)
        if data.get("version") != 1:
            raise ValueError("unsupported racing path version")
        return cls(data["points"])


class PathFollower:
    def __init__(self, path, lookahead=0.65, steering_gain=0.55,
                 max_steer=0.78, cruise_throttle=0.34,
                 min_throttle=0.12, curvature_slowdown=1.8):
        self.path = path
        self.lookahead = lookahead
        self.steering_gain = steering_gain
        self.max_steer = max_steer
        self.cruise_throttle = cruise_throttle
        self.min_throttle = min_throttle
        self.curvature_slowdown = curvature_slowdown
        self.last_index = None

    def compute(self, position, yaw):
        nearest = self.path.nearest_index(position, self.last_index)
        self.last_index = nearest
        target_index = self.path.target_index(nearest, self.lookahead)
        target = self.path.points[target_index]
        desired = math.atan2(target[1] - position[1], target[0] - position[0])
        heading_error = wrap_angle(desired - yaw)
        steering = clamp(self.steering_gain * (2.0 * math.sin(heading_error)
                                                / max(self.lookahead, 0.05)),
                         -self.max_steer, self.max_steer)
        curvature = self.path.curvature_ahead(nearest)
        throttle = self.cruise_throttle / (1.0 + self.curvature_slowdown * curvature)
        throttle *= 1.0 - 0.45 * abs(steering) / max(self.max_steer, 0.01)
        throttle = clamp(throttle, self.min_throttle, self.cruise_throttle)
        return throttle, steering, {"nearest": nearest, "target": target_index,
                                    "curvature": curvature,
                                    "heading_error": heading_error}
