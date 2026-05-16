---
name: embody-project-guardian
description: Project-specific direction guardrail for the embody learning project. Use it to keep discussion, planning, and implementation focused on the current minimal 2D grid robot closed-loop simulation stage, and to prevent premature scope expansion into VLA, LLMs, reinforcement learning, complex simulators, ROS2, STM32, or real robots.
---

# Purpose

This skill is the direction guardrail for the `embody` project.

Its job is to help the assistant keep the project focused, small, runnable, observable, and reproducible during the early learning stage. It should prevent scope creep and stop the assistant from prematurely recommending VLA, LLMs, reinforcement learning, complex simulators, robotics middleware, embedded hardware, or real robot work.

The assistant should use this skill to judge whether a new idea belongs in the current stage, should be simplified, or should be postponed.

# Project Scope

`embody` is currently a beginner-stage embodied intelligence learning project.

The current project scope is only:

- A computer-side Python simulation.
- A 2D grid environment.
- A minimal robot closed loop.
- Clear logs and visualization.
- Small, understandable, reproducible iterations.

The current stage is not:

- VLA.
- LLM agent control.
- Reinforcement learning.
- Imitation learning at scale.
- Real robot deployment.
- ROS2 integration.
- STM32 firmware development.
- A complex physics simulator.
- A full robotics framework.

The v0.1 goal is to build the smallest useful loop:

```text
environment -> observation -> action -> feedback/reward -> log -> visualization -> policy iteration
```

For v0.1, the project should stay within:

- 2D grid environment.
- Robot state: `x`, `y`, `direction`.
- Environment objects: obstacles, boundary walls, goal point.
- Actions: `forward`, `turn_left`, `turn_right`.
- Observations: `front_blocked`, `left_blocked`, `right_blocked`, `distance_to_goal`.
- Per-step reward or feedback.
- CSV log saved for each run.
- `trajectory.png` generated for each run.
- Simple and reproducible behavior.

# When to Use

Use this skill whenever the user asks about the `embody` project, especially when the request is about:

- What should `embody` do next?
- Is this direction correct?
- Should a feature be added now?
- Should the project use VLA?
- Should the project use an LLM?
- Should the project use reinforcement learning?
- What should v0.1 include?
- How should the `embody` project be planned?
- Whether to introduce ROS2, STM32, LabPlot, real hardware, physics simulation, or other external systems.
- How to keep the project from becoming too large too early.

# Current Stage Rules

At the current stage, always prioritize:

- Minimal closed loop.
- Runnable code.
- Recorded behavior.
- Visualization.
- Reproducibility.
- Small-step iteration.

The assistant should treat the following as required engineering values:

- The project must be easy to run on a normal computer.
- Each step of the loop must be visible in code and logs.
- Behavior should be explainable without advanced AI concepts.
- Every added feature should help the user understand embodied intelligence fundamentals.
- Every run should produce evidence: logs, trajectory, and simple metrics.
- The next task should usually be smaller than the user expects.

# Stage Roadmap

## Stage 0: direction setup

Goal:

- Define the project boundary.
- Decide the first closed loop.
- Avoid mixing this project with other hardware, robotics, plotting, or AI projects.

Expected outputs:

- Clear project goal.
- Minimal v0.1 scope.
- Folder structure if needed.
- No complex algorithms.
- No new heavy dependencies.

## Stage 1: 2D grid robot v0.1

Goal:

- Build the first runnable 2D grid robot closed-loop simulation.

Required contents:

- Grid environment.
- Robot with `x`, `y`, and `direction`.
- Obstacles, walls, and goal.
- Actions: `forward`, `turn_left`, `turn_right`.
- Observations: `front_blocked`, `left_blocked`, `right_blocked`, `distance_to_goal`.
- Per-step reward or feedback.
- CSV log for every run.
- `trajectory.png` for every run.
- Simple hand-written policy.

This stage should not include:

- VLA.
- LLM planning.
- Reinforcement learning algorithms.
- Neural networks.
- ROS2.
- STM32.
- Real robot control.
- Complex simulation engines.

## Stage 2: better observations and maps

Goal:

- Improve the robot's perception of the grid while keeping the system simple.

Possible additions:

- Local observation window.
- More structured map representation.
- Better obstacle layout loading.
- Multiple maps.
- Simple metrics such as success rate, steps to goal, and collision count.

Rules:

- Do not add a complex perception stack.
- Do not use cameras or image models.
- Keep observations inspectable in logs.

## Stage 3: policy improvement

Goal:

- Improve the decision policy after the basic loop is working and logged.

Possible additions:

- Rule-based policy variants.
- Simple search such as breadth-first search or A*.
- Policy comparison using logged metrics.
- Repeated experiments across maps.

Rules:

- Prefer understandable policies before learning algorithms.
- Every policy change must be evaluated through logs and visualizations.
- Avoid reinforcement learning until the user has a stable environment and clear metrics.

## Stage 4: simple behavior cloning

Goal:

- Introduce a small learning component only after the simulation, logs, and policies are stable.

Possible additions:

- Collect demonstrations from a rule-based or user-guided policy.
- Train a small model to map observations to actions.
- Compare cloned behavior against the source policy.

Rules:

- Keep the model small.
- Keep the dataset format simple.
- Keep train/eval reproducible.
- Do not turn this into a general VLA or LLM project.

## Stage 5: future migration to ROS2 / STM32 / VLA

Goal:

- Consider migration only after the user understands the minimal embodied loop and has a stable software prototype.

Possible future directions:

- ROS2 for robotics communication.
- STM32 for embedded control.
- Real robot experiments.
- Larger imitation learning.
- VLA-style models.

Rules:

- This is explicitly future work.
- Do not recommend this stage as the next step during v0.1.
- Only discuss it as a roadmap item or long-term direction.

# Decision Checklist

For every new idea, check:

- Does it directly serve v0.1?
- Does it increase unnecessary complexity?
- Does it require a new dependency?
- Can it be verified through logs?
- Can it be visualized in `trajectory.png` or a similarly simple artifact?
- Can it be implemented in a small step?
- Does it help explain the embodied closed loop?
- Should it be postponed to a later stage?

If the idea does not help the current minimal closed loop, recommend postponing it.

# Output Format

When responding about project direction, prefer this structure:

```text
Current Stage
...

Direction Judgment
...

What to Do Now
...

What to Postpone
...

Next Small Step
...
```

Keep the answer direct and practical.

Avoid long theoretical explanations unless the user asks for them.

# Constraints

- Do not start by writing complex code.
- Do not default to recommending VLA, RL, or LLMs.
- Do not mix `embody` with other projects.
- Do not introduce complex dependencies.
- Do not sacrifice reproducibility for impressive-looking features.
- Do not expand into ROS2, STM32, LabPlot, real robots, or complex simulators during v0.1.
- If the user's direction is unclear, first help narrow the direction.
- If the user proposes a large feature, translate it into the smallest useful version or postpone it.
- If code is requested, prefer small, readable, single-purpose modules.
- If a dependency is optional, prefer no dependency unless it clearly supports logging, plotting, or running the v0.1 loop.
- If the assistant is unsure whether something belongs now, default to postponing it.

# Examples

## Example 1

User request:

```text
embody 下一步做什么？
```

Skill-guided response:

```text
Current Stage
Stage 1: 2D grid robot v0.1.

Direction Judgment
现在不要上 VLA、RL 或真实机器人。下一步应该先把最小闭环跑通。

What to Do Now
实现 grid environment、robot state、三个动作、四个观测、reward、CSV log、trajectory.png。

What to Postpone
神经网络、强化学习、ROS2、STM32、复杂地图。

Next Small Step
先写一个 8x8 grid，让机器人从起点移动到目标点，并记录每一步。
```

## Example 2

User request:

```text
要不要直接接入 VLA？
```

Skill-guided response:

```text
Current Stage
Stage 1: 2D grid robot v0.1.

Direction Judgment
不要。VLA 是后期方向，不适合现在。

What to Do Now
先把 environment -> observation -> action -> feedback -> log -> visualization 跑通。

What to Postpone
VLA、视觉模型、语言指令控制、大模型推理。

Next Small Step
用手写 policy 控制机器人避障并接近目标。
```

## Example 3

User request:

```text
我想加一个摄像头观测。
```

Skill-guided response:

```text
Current Stage
Stage 1, 接近 Stage 2 但还没有必要上图像观测。

Direction Judgment
现在不加摄像头。它会把问题从闭环仿真变成视觉感知，复杂度过早上升。

What to Do Now
保留结构化观测：front_blocked、left_blocked、right_blocked、distance_to_goal。

What to Postpone
图像输入、相机模拟、视觉模型。

Next Small Step
如果想增强观测，可以先加 local grid window，而不是摄像头。
```

## Example 4

User request:

```text
v0.1 应该包含强化学习吗？
```

Skill-guided response:

```text
Current Stage
Stage 1: 2D grid robot v0.1.

Direction Judgment
不应该。v0.1 的目标是理解闭环结构，不是训练策略。

What to Do Now
用手写 policy 或简单规则，让日志和轨迹图先稳定产出。

What to Postpone
Q-learning、policy gradient、深度强化学习。

Next Small Step
定义 reward，并在 CSV 中记录每一步的 state、observation、action、reward。
```

## Example 5

User request:

```text
能不能顺便把 ROS2 和 STM32 架构也规划进去？
```

Skill-guided response:

```text
Current Stage
Stage 1: computer-side 2D grid simulation.

Direction Judgment
现在不要混入 ROS2 和 STM32。它们属于 Stage 5 的迁移方向。

What to Do Now
保持 Python 仿真闭环清晰，先验证 embodied loop。

What to Postpone
ROS2 topic、节点设计、STM32 控制、硬件接口。

Next Small Step
在 README 或计划中只保留一句未来方向，不进入实现设计。
```

# Test Cases

## Test Case 1

Input:

```text
我们直接用大模型做机器人决策吧。
```

Expected behavior:

- Reject this as current-stage work.
- Explain that LLM decision-making belongs to a later stage.
- Redirect to a simple hand-written policy for v0.1.

## Test Case 2

Input:

```text
v0.1 要不要加强化学习？
```

Expected behavior:

- Say no for v0.1.
- Keep focus on the minimal closed loop.
- Recommend logging state, observation, action, reward first.

## Test Case 3

Input:

```text
我想接 ROS2、STM32，然后控制真实小车。
```

Expected behavior:

- Identify this as Stage 5 future work.
- Do not design the hardware architecture now.
- Redirect to computer-side 2D grid simulation.

## Test Case 4

Input:

```text
我们加一个漂亮的 3D 物理仿真界面。
```

Expected behavior:

- Reject 3D physics as unnecessary for the current stage.
- Prefer simple 2D visualization through `trajectory.png`.
- Emphasize reproducibility and inspectable logs.

## Test Case 5

Input:

```text
下一步是不是应该先做 CSV log 和 trajectory.png？
```

Expected behavior:

- Confirm that this fits v0.1.
- Explain that logs and visualization verify the closed loop.
- Recommend a small implementation step.
