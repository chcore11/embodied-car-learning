# embody

`embody` is an early-stage embodied intelligence learning project.

The current goal is not to build a VLA system, an LLM-controlled robot, a reinforcement learning platform, or a complex robotics stack. The current goal is to build a small, runnable, reproducible computer-side Python simulation that makes the basic embodied loop visible.

## Current Stage

Stage 1: 2D grid robot v0.1.

The first closed loop is:

```text
environment -> observation -> action -> feedback/reward -> log -> visualization -> policy iteration
```

v0.1 should include:

- A 2D grid environment.
- A robot with `x`, `y`, and `direction`.
- Boundary walls, obstacles, and one goal point.
- Actions: `forward`, `turn_left`, `turn_right`.
- Observations: `front_blocked`, `left_blocked`, `right_blocked`, `distance_to_goal`.
- Per-step reward or feedback.
- One CSV log per run.
- One `trajectory.png` per run.
- A simple, readable policy.

## Not Current Work

Do not introduce these into v0.1:

- VLA.
- Large language models.
- Reinforcement learning.
- Isaac Sim.
- MuJoCo.
- Habitat.
- ROS2 integration.
- STM32 hardware integration.
- Complex simulation frameworks.

These are future directions only after the minimal loop is stable.

## Repository Structure

```text
embodied-car-learning/
|-- AGENTS.md
|-- README.md
|-- data/
|   `-- README.md
|-- docs/
|   |-- 00_project_goal.md
|   |-- 01_embodied_ai_basic.md
|   |-- 02_robot_system_architecture.md
|   |-- 03_data_collection_plan.md
|   |-- 04_learning_roadmap.md
|   `-- 05_pytorch_learning_plan.md
|-- logs/
|   `-- day1.md
`-- skills/
    `-- embody-project-guardian/
        `-- SKILL.md
```

## Working Rule

Before planning, coding, refactoring, writing docs, or designing features, read:

```text
skills/embody-project-guardian/SKILL.md
```

Keep every next step small, testable, reversible, and aligned with the v0.1 loop.
