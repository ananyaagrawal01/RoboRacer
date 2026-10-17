# Wall-follow optimization harness

The harness evaluates exactly three completed laps after each verified reset.
Any increase in the simulator collision counter rejects the trial. It never
stops the simulator, bridge, container, or controller processes it did not
start itself.

Run it inside `penn-challenge` after enabling the simulator communication
bridge:

```bash
docker exec -it penn-challenge bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
cd /home/autodrive_devkit

# No reset or controller launch; checks that required telemetry is fresh.
python3 src/autodrive_devkit/tools/optimize_wall_follow.py preflight

# Sends one reset pulse and verifies stable, repeatable initial conditions.
python3 src/autodrive_devkit/tools/optimize_wall_follow.py reset-check

# Long-running adaptive search, one parameter at a time, with resume state.
python3 src/autodrive_devkit/tools/optimize_wall_follow.py optimize \
  --max-trials 1000 \
  --csv optimization_results.csv \
  --state optimization_state.json
```

The initial baseline is `lookahead=0.70`, `front_slow_distance=2.10`, and
`cruise_throttle=0.28`. The adaptive coordinate search changes only one
parameter from the validated incumbent at a time, periodically explores the
full range, and reduces its step sizes as the search converges. A candidate
replaces the incumbent only when all of its validation repeats are
collision-free and its median three-lap mean improves. The JSON state is saved
after every trial, so the same command resumes an interrupted run.

Useful limits:

```bash
--trial-timeout 90
--reset-timeout 8
--position-tolerance 0.03
--promising-repeats 3
```

If telemetry is missing/stale or reset is not repeatable, the command stops
without launching the controller. Each completed or failed trial is flushed
and synced to CSV immediately.

To remove only the optimizer files added for this workflow:

```bash
rm -f autodrive_devkit/tools/optimize_wall_follow.py \
      autodrive_devkit/tools/wall_follow_optimization.py \
      autodrive_devkit/tools/README.md \
      autodrive_devkit/test/test_wall_follow_optimization.py
rmdir autodrive_devkit/tools 2>/dev/null || true
```
