# PyTorch Learning Plan

This document defines a simplified PyTorch learning plan for the `embodied-car-learning` project.

PyTorch is not the main direction of this project. The main direction is still the mobile robot system: motion control, encoder feedback, odometry, data logging, ROS2 integration, and future sensor-based perception.

PyTorch is a supporting tool. Its purpose is to help convert robot data into datasets, train simple models, and prepare for future embodied AI experiments.

## Core Position

The role of PyTorch in this project is:

```text
Robot Data → Dataset → Model → Action Prediction → Robot Behavior
```

The current goal is not to train large models, build VLA systems, or study advanced reinforcement learning.

The current goal is to understand the basic workflow of machine learning and apply it to simple robot-related data.

## Why Learn PyTorch

PyTorch will be useful for this project in three ways:

1. Processing robot motion and sensor data
2. Training simple action prediction models
3. Preparing for future imitation learning experiments

Example robot-style data:

```csv
front_distance,left_distance,right_distance,current_speed,action
30,60,20,0.2,turn_left
50,40,45,0.3,forward
10,80,25,0.1,stop
```

A simple model can learn:

```text
Input: robot state
Output: robot action
```

This is the first step toward learning-based robot behavior.

---

# Stage 1 - PyTorch Basics

## Goal

Understand the basic training workflow.

## Key Topics

- Tensor
- Tensor shape
- Autograd
- Loss function
- Optimizer
- Backpropagation
- Training loop
- Save and load model

## Practice

- Create and operate tensors
- Use autograd to calculate gradients
- Train a simple model to fit `y = 2x + 1`
- Save and load a trained model

## Deliverables

```text
experiments/pytorch_basics/01_tensor_basics.py
experiments/pytorch_basics/02_autograd.py
experiments/pytorch_basics/03_linear_regression.py
```

---

# Stage 2 - Neural Network Workflow

## Goal

Learn how to build and train a simple neural network.

## Key Topics

- `nn.Module`
- `forward()`
- Linear layer
- Activation function
- Train / eval mode
- Validation
- Model saving and loading

## Practice

- Build a small neural network
- Train it with simple numerical data
- Evaluate the model
- Save and reload the model for inference

## Deliverables

```text
experiments/pytorch_basics/04_simple_nn.py
experiments/pytorch_basics/05_train_eval_save.py
```

---

# Stage 3 - Dataset and DataLoader

## Goal

Learn how to convert CSV data into a PyTorch dataset.

This stage is important because future robot data will come from logs, CSV files, sensor records, and ROS2 data.

## Key Topics

- CSV data
- Pandas
- Custom Dataset
- DataLoader
- Batch
- Feature and label
- Classification task

## Practice

- Create a fake robot CSV dataset
- Write a custom PyTorch Dataset
- Load data with DataLoader
- Train a simple classification model

## Deliverables

```text
experiments/ml_action_prediction/sample_robot_data.csv
experiments/ml_action_prediction/dataset.py
experiments/ml_action_prediction/train.py
experiments/ml_action_prediction/inference.py
```

---

# Stage 4 - Simple Robot Action Prediction

## Goal

Build the first robot-related learning demo.

The model predicts a simple action from basic sensor-like input.

## Input

```text
front_distance
left_distance
right_distance
current_speed
```

## Output

```text
forward
stop
turn_left
turn_right
```

## Practice

- Generate a small rule-based dataset
- Train an action classification model
- Test the model with new inputs
- Compare model prediction with rule-based logic
- Write result notes

## Deliverables

```text
experiments/ml_action_prediction/README.md
experiments/ml_action_prediction/generate_data.py
experiments/ml_action_prediction/train_action_model.py
experiments/ml_action_prediction/test_action_model.py
experiments/ml_action_prediction/result_notes.md
```

---

# Stage 5 - From Fake Data to Real Robot Data

## Goal

Replace fake data with real robot logs after the mobile robot can record useful data.

Possible future real data format:

```csv
time_ms,left_count,right_count,left_speed,right_speed,target_v,target_w,pwm_l,pwm_r,x,y,theta,front_distance,left_distance,right_distance,command,result
```

## Practice

- Load real robot CSV data
- Clean the data
- Select input features
- Select prediction target
- Train a simple model
- Analyze limitations

## Deliverables

```text
real_robot_dataset.csv
data_cleaning.py
train_with_real_data.py
result_analysis.md
```

---

# Stage 6 - Imitation Learning Preparation

## Goal

Prepare for basic imitation learning.

The simple version is:

```text
Human controls the robot.
The system records state and action.
A model learns to predict human action from state.
```

## Data Structure

```text
State: sensor data, odometry, robot speed
Action: human command or target velocity
Result: success, failure, collision, deviation
```

## Deliverables

```text
teleoperation_data_format.md
state_action_dataset.csv
basic_behavior_cloning.py
offline_evaluation.md
```

---

# 30-Day Side Plan

PyTorch is a side line. It should not replace the robot engineering work.

Recommended time allocation:

```text
70% robot system
30% PyTorch learning
```

## Week 1 - Tensor and Training Basics

- Learn tensor operations
- Learn autograd
- Train linear regression
- Understand loss and optimizer

## Week 2 - Neural Network Workflow

- Learn `nn.Module`
- Build a simple neural network
- Write a complete training loop
- Save and load model parameters

## Week 3 - CSV Dataset and DataLoader

- Read CSV data
- Build custom Dataset
- Use DataLoader
- Train a classification model

## Week 4 - Robot Action Prediction Demo

- Generate fake robot data
- Train an action prediction model
- Run inference
- Write result notes

---

# Current Priority

The current priority is not to become an AI researcher immediately.

The current priority is to learn enough PyTorch to process robot data and build a simple action prediction experiment.

The most important connection is:

```text
Robot state → Action label → Dataset → Model → Prediction
```

If PyTorch learning cannot connect back to robot data, it should not be the priority of this project.
