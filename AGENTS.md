# Codex Instructions for embody

This repository is the `embody` project.

Before any planning, coding, refactoring, documentation, or feature design, read:

```text
skills/embody-project-guardian/SKILL.md
```

Follow that skill as the project direction guardrail for all work in this repository.

## Current Stage

The current stage is a computer-side Python 2D grid robot simulation.

The goal is to run the minimal embodied intelligence closed loop:

```text
environment -> observation -> action -> feedback/reward -> log -> visualization -> policy iteration
```

Current work should serve this v0.1 direction:

- 2D grid environment.
- Robot state with `x`, `y`, and `direction`.
- Simple actions: `forward`, `turn_left`, `turn_right`.
- Simple observations.
- Per-step feedback or reward.
- CSV logs.
- Basic trajectory visualization.
- Small, reproducible iterations.

## Do Not Introduce Early

Do not prematurely introduce:

- VLA.
- Large language models.
- Reinforcement learning.
- Isaac Sim.
- MuJoCo.
- Habitat.
- ROS2 integration.
- STM32 hardware integration.
- Complex simulation frameworks.

These belong only to later-stage discussion unless the user explicitly asks for roadmap context.

## Default Workflow

For every request in this repository:

1. Read `skills/embody-project-guardian/SKILL.md`.
2. Identify the current project stage.
3. Decide whether the request directly serves v0.1.
4. Prefer small, testable, reversible changes.
5. If the user has not explicitly asked for code, do not proactively write code.
6. Do not mix `embody` with STM32, ROS2, LabPlot, or other separate projects.
7. If the request risks scope creep, narrow it to the smallest useful next step or recommend postponing it.

Keep responses and changes simple, direct, and aligned with the project guardrail.
