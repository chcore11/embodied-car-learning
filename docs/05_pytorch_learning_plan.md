# PyTorch Learning Plan

PyTorch is not part of the current v0.1 implementation.

For `embody`, PyTorch should be treated as future support for Stage 4 simple behavior cloning. It should not replace the current priority: building a small, logged, visualized 2D grid robot loop.

## Current Position

Do not start v0.1 with neural networks.

First complete:

- Environment.
- Observation.
- Action.
- Feedback or reward.
- CSV logging.
- Trajectory visualization.
- Simple policy iteration.

## Future Use

After the v0.1 simulation is stable, PyTorch may be useful for:

- Loading CSV logs as datasets.
- Training a small observation-to-action classifier.
- Comparing a learned policy with a rule-based policy.
- Understanding basic behavior cloning.

## Not Current Work

Do not add these now:

- Deep reinforcement learning.
- VLA models.
- LLM policy control.
- Large training pipelines.
- GPU-first infrastructure.
- Complex experiment frameworks.

## Stage 4 Candidate Workflow

Future behavior cloning could look like:

```text
rule-based policy run -> CSV demonstrations -> small dataset -> small model -> offline comparison
```

This should happen only after the logs and trajectory visualization are already reliable.
