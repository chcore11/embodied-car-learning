# v0.1 System Architecture

The current system is a computer-side 2D grid robot simulation.

## Core Modules

The v0.1 implementation should be split into simple responsibilities:

- Environment: owns the grid, walls, obstacles, and goal.
- Robot: owns `x`, `y`, and `direction`.
- Observation: converts environment and robot state into simple fields.
- Policy: maps observation to one action.
- Step logic: applies action and produces feedback or reward.
- Logger: writes one CSV row per step.
- Visualizer: creates `trajectory.png`.
- Runner: connects the loop for one reproducible run.

## Loop

```text
initialize environment and robot
observe
choose action
step environment
calculate feedback/reward
write log row
save trajectory
compare policy behavior
```

## Design Rules

- Keep the modules readable and small.
- Prefer explicit state over hidden framework behavior.
- Keep observations inspectable in CSV logs.
- Keep visualization simple.
- Avoid new dependencies unless they directly support logging or plotting.
- Do not add ROS2, STM32, Isaac Sim, MuJoCo, Habitat, VLA, LLMs, or RL in v0.1.
