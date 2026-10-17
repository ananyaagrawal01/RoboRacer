"""ROS-independent trial bookkeeping for wall-follow optimization."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrialState:
    """Track exactly the next three completed laps and reject any collision."""

    start_lap_count: int
    start_collision_count: int
    target_laps: int = 3
    lap_times: List[float] = field(default_factory=list)
    status: str = "running"
    reason: str = ""
    _last_lap_count: int = field(init=False)

    def __post_init__(self):
        self._last_lap_count = self.start_lap_count

    @property
    def done(self):
        return self.status != "running"

    def observe_collision(self, count: int):
        if self.done:
            return
        if count < self.start_collision_count:
            self.reject("collision counter decreased during trial")
        elif count > self.start_collision_count:
            self.reject("wall collision detected")

    def observe_lap(self, count: int, last_lap_time: Optional[float]):
        if self.done or count == self._last_lap_count:
            return
        if count != self._last_lap_count + 1:
            self.reject("lap counter changed by more than one")
            return
        if last_lap_time is None or last_lap_time <= 0:
            self.reject("lap completed without a valid last_lap_time")
            return
        self._last_lap_count = count
        self.lap_times.append(float(last_lap_time))
        if len(self.lap_times) == self.target_laps:
            self.status = "complete"

    def reject(self, reason: str):
        self.status = "rejected"
        self.reason = reason

