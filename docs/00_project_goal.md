# Project Goal

`embody` is a small embodied intelligence learning project.

The current project goal is to understand the basic structure of embodied intelligence by building a minimal computer-side 2D grid robot simulation.

The project should make this loop concrete:

```text
environment -> observation -> action -> feedback/reward -> log -> visualization -> policy iteration
```

## v0.1 Goal

Build a runnable 2D grid robot simulation with:

- Grid environment.
- Robot state: `x`, `y`, `direction`.
- Obstacles, walls, and goal.
- Actions: `forward`, `turn_left`, `turn_right`.
- Observations: `front_blocked`, `left_blocked`, `right_blocked`, `distance_to_goal`.
- Per-step reward or feedback.
- CSV run log.
- `trajectory.png` output.
- Simple policy iteration.

## Current Boundary

The current stage is not about:

- VLA.
- LLM control.
- Reinforcement learning.
- Isaac Sim, MuJoCo, or Habitat.
- ROS2 integration.
- STM32 hardware.
- Real robot deployment.

Those topics are future-stage candidates only after v0.1 is stable.
