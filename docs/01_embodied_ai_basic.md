# Embodied AI Basics

For this project, embodied intelligence means an agent repeatedly senses its situation, chooses an action, acts in an environment, receives feedback, records what happened, and improves its policy.

The minimal loop is:

```text
environment -> observation -> action -> feedback/reward -> log -> visualization -> policy iteration
```

## Mapping to v0.1

In the v0.1 2D grid simulation:

- Environment: a grid with walls, obstacles, and a goal.
- Observation: whether nearby cells are blocked and how far the robot is from the goal.
- Action: `forward`, `turn_left`, or `turn_right`.
- Feedback/reward: collision, progress toward goal, reaching the goal, or wasting steps.
- Log: one CSV row per step.
- Visualization: a trajectory image for the run.
- Policy iteration: compare simple policies and improve them using logs.

## Why Start Small

The first learning target is not advanced AI. It is understanding the closed loop.

A small grid robot is enough to expose the core questions:

- What does the agent observe?
- What actions are available?
- What happens after an action?
- How is success or failure measured?
- What evidence is saved after a run?
- How can the policy improve next time?

## Current Avoid List

Do not start with VLA, LLMs, reinforcement learning, real robots, ROS2, STM32, or complex simulators. These add too many moving parts before the basic loop is clear.
