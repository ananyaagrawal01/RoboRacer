#!/usr/bin/env python3
"""Safely optimize wall_follow_node using AutoDRIVE's ROS 2 race telemetry.

Run this inside the ROS 2 container after sourcing the Humble and workspace
setup files. The optimizer only terminates controller subprocesses it starts.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
from typing import Dict, Iterable, List

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Int32

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wall_follow_optimization import TrialState  # noqa: E402


TOPICS = {
    "lap_count": "/autodrive/roboracer_1/lap_count",
    "lap_time": "/autodrive/roboracer_1/lap_time",
    "last_lap_time": "/autodrive/roboracer_1/last_lap_time",
    "collision_count": "/autodrive/roboracer_1/collision_count",
    "position": "/autodrive/roboracer_1/ips",
    "lidar": "/autodrive/roboracer_1/lidar",
}

CSV_FIELDS = [
    "timestamp", "trial_id", "phase", "parameter", "repeat",
    "desired_distance", "lookahead", "kp", "kd", "max_steer",
    "cruise_throttle", "min_throttle", "corner_throttle",
    "front_slow_distance", "front_stop_distance", "emergency_distance",
    "max_throttle_slew",
    "lap_1", "lap_2", "lap_3", "mean_lap", "median_lap", "stdev_lap",
    "collision_start", "collision_end", "collision_delta",
    "lap_count_start", "lap_count_end", "reset_verified", "start_position",
    "status", "reason", "elapsed_seconds", "controller_returncode",
]


class ResetRequired(RuntimeError):
    """Raised when AutoDRIVE checkpoint respawn cannot restore the grid."""


class RaceMonitor(Node):
    def __init__(self):
        super().__init__("wall_follow_optimizer")
        self.values = {}
        self.updated = {}
        self.reset_reference = None
        self.reset_pub = self.create_publisher(Bool, "/autodrive/reset_command", 1)
        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 1)
        self.steering_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 1)
        self.create_subscription(Int32, TOPICS["lap_count"],
                                 lambda m: self._set("lap_count", m.data), 10)
        self.create_subscription(Float32, TOPICS["lap_time"],
                                 lambda m: self._set("lap_time", m.data), 10)
        self.create_subscription(Float32, TOPICS["last_lap_time"],
                                 lambda m: self._set("last_lap_time", m.data), 10)
        self.create_subscription(Int32, TOPICS["collision_count"],
                                 lambda m: self._set("collision_count", m.data), 10)
        self.create_subscription(Point, TOPICS["position"], self._position, 10)
        self.create_subscription(LaserScan, TOPICS["lidar"],
                                 lambda m: self._set("lidar", len(m.ranges)), 10)

    def _set(self, name, value):
        self.values[name] = value
        self.updated[name] = time.monotonic()

    def _position(self, msg):
        self._set("position", (float(msg.x), float(msg.y), float(msg.z)))

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(0.05, end - time.monotonic()))

    def wait_fresh(self, names: Iterable[str], timeout: float, max_age=1.0):
        end = time.monotonic() + timeout
        names = tuple(names)
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.monotonic()
            if all(name in self.values and now - self.updated[name] <= max_age
                   for name in names):
                return True
        return False

    def stop_vehicle(self):
        for _ in range(3):
            self.throttle_pub.publish(Float32(data=0.0))
            self.steering_pub.publish(Float32(data=0.0))
            self.spin_for(0.05)

    def pulse_reset(self):
        """Stop and immediately separate a vehicle from ongoing wall contact."""
        self.stop_vehicle()
        self.reset_pub.publish(Bool(data=True))
        self.spin_for(0.15)
        self.reset_pub.publish(Bool(data=False))
        self.spin_for(0.50)

    def reset_and_verify(self, timeout: float, position_tolerance: float):
        required = ("lap_count", "last_lap_time", "collision_count", "position", "lidar")
        if not self.wait_fresh(required, min(timeout, 5.0)):
            return False, "required telemetry is absent or stale"
        self.pulse_reset()

        end = time.monotonic() + timeout
        samples = []
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.wait_fresh(required, 0.2):
                continue
            p = self.values["position"]
            samples.append((time.monotonic(), p, self.values["lap_count"],
                            self.values["collision_count"]))
            samples = [s for s in samples if samples[-1][0] - s[0] <= 0.75]
            if len(samples) >= 4:
                positions = [s[1] for s in samples]
                spread = max(math.dist(positions[0], p) for p in positions)
                counters_stable = len({(s[2], s[3]) for s in samples}) == 1
                if spread <= position_tolerance and counters_stable:
                    if self.reset_reference is None:
                        self.reset_reference = p
                    elif math.dist(self.reset_reference, p) > position_tolerance:
                        return False, "reset pose differs from the reference start pose"
                    return True, ""
        return False, "reset did not produce a stable pose and counters"


def terminate_owned(process):
    if process is None or process.poll() is not None:
        return
    # ros2 run replaces/spawns through the CLI; signal the dedicated process
    # group so the actual ROS node cannot survive as an orphaned publisher.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def controller_command(params: Dict[str, float]):
    cmd = ["ros2", "run", "autodrive_roboracer", "wall_follow_node", "--ros-args"]
    for name, value in params.items():
        cmd.extend(["-p", f"{name}:={value}"])
    return cmd


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_trial(node, params, args, trial_id, phase, parameter, repeat):
    started = time.monotonic()
    row = {field: "" for field in CSV_FIELDS}
    row.update(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), trial_id=trial_id,
               phase=phase, parameter=parameter, repeat=repeat, **params)
    ok, reason = node.reset_and_verify(args.reset_timeout, args.position_tolerance)
    row["reset_verified"] = ok
    if not ok:
        row.update(status="reset_failed", reason=reason,
                   elapsed_seconds=round(time.monotonic() - started, 3))
        append_csv(args.csv, row)
        return row

    start_laps = int(node.values["lap_count"])
    start_collisions = int(node.values["collision_count"])
    start_position = node.values["position"]
    state = TrialState(start_laps, start_collisions)
    collision_peak = start_collisions
    row.update(lap_count_start=start_laps, collision_start=start_collisions,
               start_position=" ".join(f"{x:.6f}" for x in start_position))
    process = None
    try:
        process = subprocess.Popen(controller_command(params), start_new_session=True)
        deadline = time.monotonic() + args.trial_timeout
        motion_position = node.values.get("position")
        last_motion = time.monotonic()
        motion_grace = last_motion + 4.0
        last_seen_lap = start_laps
        while not state.done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if process.poll() is not None:
                state.reject(f"controller exited early with code {process.returncode}")
                break
            if not node.wait_fresh(("lap_count", "last_lap_time", "collision_count", "lidar"),
                                   0.25, max_age=args.telemetry_max_age):
                state.reject("race telemetry became stale")
                break
            collision_peak = max(collision_peak, int(node.values["collision_count"]))
            state.observe_collision(collision_peak)
            position = node.values.get("position")
            if position is not None and motion_position is not None:
                if math.dist(position, motion_position) >= 0.04:
                    motion_position = position
                    last_motion = time.monotonic()
                elif (time.monotonic() > motion_grace
                      and time.monotonic() - last_motion > args.stuck_timeout):
                    state.reject("vehicle remained stationary")
                    break
            lap_count = int(node.values["lap_count"])
            if lap_count != last_seen_lap:
                lap_update = node.updated.get("lap_count", 0.0)
                sync_deadline = time.monotonic() + 0.5
                while (node.updated.get("last_lap_time", 0.0) < lap_update
                       and time.monotonic() < sync_deadline):
                    rclpy.spin_once(node, timeout_sec=0.02)
                state.observe_lap(lap_count, float(node.values["last_lap_time"]))
                last_seen_lap = lap_count
        if state.status == "complete":
            # Drain the telemetry burst containing the third-lap event so a
            # collision reported in the same simulator frame cannot slip by.
            node.spin_for(0.3)
            collision_peak = max(collision_peak, int(node.values["collision_count"]))
            state.observe_collision(collision_peak)
        if not state.done:
            state.reject("three-lap trial timeout")
    finally:
        node.stop_vehicle()
        terminate_owned(process)
        if state.reason == "wall collision detected":
            node.pulse_reset()

    collision_end = collision_peak
    lap_end = int(node.values.get("lap_count", start_laps))
    laps = state.lap_times
    row.update(status=state.status, reason=state.reason,
               collision_end=collision_end,
               collision_delta=collision_end - start_collisions,
               lap_count_end=lap_end,
               elapsed_seconds=round(time.monotonic() - started, 3),
               controller_returncode="" if process is None else process.returncode)
    for index, value in enumerate(laps[:3], 1):
        row[f"lap_{index}"] = round(value, 6)
    if len(laps) == 3 and state.status == "complete":
        row["mean_lap"] = round(statistics.mean(laps), 6)
        row["median_lap"] = round(statistics.median(laps), 6)
        row["stdev_lap"] = round(statistics.stdev(laps), 6)
    append_csv(args.csv, row)
    return row


def score(rows: List[Dict]):
    valid = [float(r["mean_lap"]) for r in rows if r.get("status") == "complete"]
    return statistics.median(valid) if valid and len(valid) == len(rows) else math.inf


PARAMETERS = {
    "desired_distance": {"low": 0.65, "high": 1.05, "step": 0.08,
                         "min_step": 0.005},
    "lookahead": {"low": 0.45, "high": 1.00, "step": 0.10, "min_step": 0.005},
    "kp": {"low": 0.70, "high": 2.60, "step": 0.30, "min_step": 0.025},
    "kd": {"low": 0.03, "high": 0.50, "step": 0.07, "min_step": 0.005},
    "max_steer": {"low": 0.55, "high": 0.95, "step": 0.07,
                  "min_step": 0.005},
    "cruise_throttle": {"low": 0.20, "high": 0.42, "step": 0.025,
                        "min_step": 0.0025},
    "min_throttle": {"low": 0.05, "high": 0.18, "step": 0.025,
                     "min_step": 0.0025},
    "corner_throttle": {"low": 0.07, "high": 0.24, "step": 0.03,
                        "min_step": 0.0025},
    "front_slow_distance": {"low": 1.40, "high": 2.90, "step": 0.25,
                            "min_step": 0.015},
    "front_stop_distance": {"low": 0.45, "high": 0.90, "step": 0.08,
                            "min_step": 0.005},
    "emergency_distance": {"low": 0.28, "high": 0.55, "step": 0.05,
                           "min_step": 0.005},
    "max_throttle_slew": {"low": 0.35, "high": 1.60, "step": 0.20,
                          "min_step": 0.02},
}

BASELINE = {
    "desired_distance": 0.85,
    "lookahead": 0.70,
    "kp": 1.50,
    "kd": 0.20,
    "max_steer": 0.78,
    "cruise_throttle": 0.28,
    "min_throttle": 0.10,
    "corner_throttle": 0.13,
    "front_slow_distance": 2.10,
    "front_stop_distance": 0.65,
    "emergency_distance": 0.42,
    "max_throttle_slew": 0.80,
}


def save_state(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_or_create_state(args):
    if args.state.exists():
        with args.state.open() as handle:
            state = json.load(handle)
        print(f"Resuming {state['trials_completed']} / {args.max_trials} trials from {args.state}")
        for name, value in BASELINE.items():
            state["incumbent"].setdefault(name, value)
            state["steps"].setdefault(name, PARAMETERS[name]["step"])
            state["failures"].setdefault(name, 0)
        return state
    return {
        "version": 1,
        "incumbent": dict(BASELINE),
        "incumbent_score": None,
        "steps": {name: spec["step"] for name, spec in PARAMETERS.items()},
        "failures": {name: 0 for name in PARAMETERS},
        "iteration": 0,
        "trials_completed": 0,
        "trial_id": int(time.time()),
        "seed": args.seed,
    }


def propose(state, rng):
    names = list(PARAMETERS)
    name = names[state["iteration"] % len(names)]
    spec = PARAMETERS[name]
    low, high = spec["low"], spec["high"]
    incumbent = state["incumbent"]
    if name == "front_slow_distance":
        low = max(low, incumbent["front_stop_distance"] + 0.15)
    elif name == "front_stop_distance":
        low = max(low, incumbent["emergency_distance"] + 0.08)
        high = min(high, incumbent["front_slow_distance"] - 0.15)
    elif name == "emergency_distance":
        high = min(high, incumbent["front_stop_distance"] - 0.08)
    elif name == "min_throttle":
        high = min(high, incumbent["corner_throttle"])
    elif name == "corner_throttle":
        low = max(low, incumbent["min_throttle"])
        high = min(high, incumbent["cruise_throttle"])
    elif name == "cruise_throttle":
        low = max(low, incumbent["corner_throttle"])
    current = float(state["incumbent"][name])
    # Regular global exploration prevents a local coordinate search from
    # permanently overlooking a distant safe/faster region.
    if state["iteration"] % 11 == 10:
        value = rng.uniform(low, high)
        phase = "global_exploration"
    else:
        distance = state["steps"][name] * rng.choice((0.5, 1.0, 1.5, 2.0))
        value = current + rng.choice((-1.0, 1.0)) * distance
        phase = "adaptive_coordinate"
    value = round(max(low, min(high, value)), 4)
    if value == current:
        value = round(low if current != low else high, 4)
    params = dict(state["incumbent"])
    params[name] = value
    return name, params, phase


def optimize(node, args):
    state = load_or_create_state(args)
    rng = random.Random(state.get("seed", args.seed) + state["iteration"])

    def execute(params, phase, parameter, repeat):
        if state["trials_completed"] >= args.max_trials:
            return None
        state["trial_id"] += 1
        row = run_trial(node, params, args, state["trial_id"], phase, parameter, repeat)
        state["trials_completed"] += 1
        save_state(args.state, state)
        print(f"trial {state['trials_completed']}/{args.max_trials}: {phase} "
              f"{parameter}={params.get(parameter, '-')} status={row['status']} "
              f"mean={row.get('mean_lap', '')}", flush=True)
        if row["status"] == "reset_failed":
            raise ResetRequired(row["reason"])
        return row

    if state["incumbent_score"] is None:
        rows = []
        for repeat in range(1, args.promising_repeats + 1):
            row = execute(state["incumbent"], "baseline_validation", "baseline", repeat)
            if row is None:
                break
            rows.append(row)
        baseline_score = score(rows)
        if math.isfinite(baseline_score):
            state["incumbent_score"] = baseline_score
            save_state(args.state, state)
            print(f"Validated initial incumbent: {state['incumbent']} score={baseline_score:.4f}")
        else:
            print("Initial baseline was rejected; searching one parameter at a time for a safe incumbent")

    while state["trials_completed"] < args.max_trials:
        name, candidate, phase = propose(state, rng)
        rows = []
        for repeat in range(1, args.candidate_repeats + 1):
            row = execute(candidate, phase, name, repeat)
            if row is None:
                break
            rows.append(row)
            if row["status"] != "complete":
                break
        candidate_score = score(rows)
        incumbent_score = state["incumbent_score"]

        # Spend repeat trials only on collision-free candidates close enough to
        # plausibly beat the incumbent. Every repeat must remain collision-free.
        promising = math.isfinite(candidate_score) and (
            incumbent_score is None
            or candidate_score <= float(incumbent_score) * (1.0 + args.validation_margin))
        if promising:
            for repeat in range(len(rows) + 1, args.promising_repeats + 1):
                row = execute(candidate, "promising_validation", name, repeat)
                if row is None:
                    break
                rows.append(row)
                if row["status"] != "complete":
                    break
            candidate_score = score(rows)

        if math.isfinite(candidate_score) and (
                incumbent_score is None or candidate_score < float(incumbent_score)):
            state["incumbent"] = candidate
            state["incumbent_score"] = candidate_score
            state["failures"][name] = 0
            print(f"NEW INCUMBENT score={candidate_score:.4f}: {candidate}", flush=True)
        else:
            state["failures"][name] += 1
            if state["failures"][name] % 4 == 0:
                spec = PARAMETERS[name]
                state["steps"][name] = max(
                    spec["min_step"], round(state["steps"][name] * 0.70, 6))
        state["iteration"] += 1
        save_state(args.state, state)

    print(f"Optimization budget complete. Best validated parameters: "
          f"{state['incumbent']} score={state['incumbent_score']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "reset-check", "optimize"))
    parser.add_argument("--csv", type=Path, default=Path("optimization_results.csv"))
    parser.add_argument("--trial-timeout", type=float, default=90.0)
    parser.add_argument("--stuck-timeout", type=float, default=6.0)
    parser.add_argument("--reset-timeout", type=float, default=8.0)
    parser.add_argument("--telemetry-max-age", type=float, default=1.0)
    parser.add_argument("--position-tolerance", type=float, default=0.03)
    parser.add_argument("--promising-repeats", type=int, default=3)
    parser.add_argument("--candidate-repeats", type=int, default=1)
    parser.add_argument("--validation-margin", type=float, default=0.04)
    parser.add_argument("--max-trials", type=int, default=300)
    parser.add_argument("--state", type=Path, default=Path("optimization_state.json"))
    parser.add_argument("--seed", type=int, default=20261017)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = RaceMonitor()
    try:
        required = ("lap_count", "last_lap_time", "collision_count", "position", "lidar")
        if not node.wait_fresh(required, 8.0):
            print("FAIL: simulator telemetry is absent or stale; enable the AutoDRIVE bridge",
                  file=sys.stderr)
            return 2
        print("PASS: required simulator telemetry is live")
        if args.mode == "preflight":
            return 0
        if args.mode == "optimize":
            existing = [name for name in node.get_node_names()
                        if name.rsplit("/", 1)[-1] == "wall_follow_node"]
            if existing:
                print("FAIL: an existing wall_follow_node is already running; stop it "
                      "before optimization", file=sys.stderr)
                return 4
        ok, reason = node.reset_and_verify(args.reset_timeout, args.position_tolerance)
        if not ok:
            print(f"FAIL: {reason}", file=sys.stderr)
            return 3
        print("PASS: reset produced stable pose and counters")
        if args.mode == "optimize":
            try:
                optimize(node, args)
            except ResetRequired as error:
                print(f"PAUSED: {error}. Fully restart the race at the grid, then "
                      "rerun the identical command to resume.", file=sys.stderr)
                return 5
        return 0
    finally:
        node.stop_vehicle()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
