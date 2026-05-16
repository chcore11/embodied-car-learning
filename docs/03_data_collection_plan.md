# v0.1 Data Collection Plan

The purpose of data collection in v0.1 is to verify the closed loop, not to train a large model.

Every simulation run should produce a CSV log. Each row should describe one step.

## Suggested CSV Fields

```csv
run_id,step,x,y,direction,front_blocked,left_blocked,right_blocked,distance_to_goal,action,reward,done,event
```

## Field Meaning

- `run_id`: stable identifier for one run.
- `step`: step number within the run.
- `x`, `y`: robot position.
- `direction`: robot heading.
- `front_blocked`, `left_blocked`, `right_blocked`: local obstacle or wall observations.
- `distance_to_goal`: simple distance metric to the goal.
- `action`: selected action.
- `reward`: feedback after the action.
- `done`: whether the run ended.
- `event`: optional short label such as `move`, `turn`, `collision`, `goal`, or `timeout`.

## Run Outputs

Each run should save:

- One CSV log.
- One `trajectory.png`.

## Acceptance Criteria

- A run can be replayed or understood from the CSV.
- The trajectory image matches the logged states.
- Policy changes can be compared using logs.
- The data format stays simple enough to inspect manually.
