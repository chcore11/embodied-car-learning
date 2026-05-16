# Learning Roadmap

This roadmap keeps `embody` focused on the current minimal simulation before moving to larger embodied AI topics.

## Stage 0: Direction Setup

Goal:

- Define the project boundary.
- Keep the project separate from STM32, ROS2, LabPlot, real robot, and complex simulator work.
- Establish the project guardrail skill.

Deliverables:

- `AGENTS.md`.
- `skills/embody-project-guardian/SKILL.md`.
- Clear v0.1 scope.

## Stage 1: 2D Grid Robot v0.1

Goal:

- Build the first runnable embodied loop.

Required loop:

```text
environment -> observation -> action -> feedback/reward -> log -> visualization -> policy iteration
```

Deliverables:

- 2D grid environment.
- Robot state with `x`, `y`, and `direction`.
- `forward`, `turn_left`, `turn_right` actions.
- `front_blocked`, `left_blocked`, `right_blocked`, `distance_to_goal` observations.
- CSV log for each run.
- `trajectory.png` for each run.
- Simple hand-written policy.

## Stage 2: Better Observations and Maps

Goal:

- Improve observations without introducing complex perception.

Possible work:

- Local grid window.
- Multiple simple maps.
- Map files.
- Success rate, steps to goal, and collision count.

## Stage 3: Policy Improvement

Goal:

- Improve behavior after logging and visualization are stable.

Possible work:

- Rule-based policy variants.
- Breadth-first search or A*.
- Policy comparison from logged metrics.

## Stage 4: Simple Behavior Cloning

Goal:

- Add a small learning component only after the environment and logs are stable.

Possible work:

- Collect demonstrations from a rule-based policy.
- Train a small model from observations to actions.
- Compare cloned behavior with the source policy.

## Stage 5: Future Migration

Goal:

- Consider larger robotics and AI systems only after the minimal loop is clear.

Future candidates:

- ROS2.
- STM32.
- Real robot experiments.
- VLA-style models.
- Larger imitation learning.

These are not v0.1 work.

## Current Priority

The current priority is:

```text
small simulation -> clear logs -> simple visualization -> policy iteration
```
