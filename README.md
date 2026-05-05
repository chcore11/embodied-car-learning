# Embodied Car Learning

This repository is a long-term learning project for embodied AI on a mobile robot platform.

The goal is not to directly build a complex "AI robot" at the beginning, but to gradually develop a complete embodied system from low-level motion control, odometry, data collection, ROS2 integration, environmental perception, and eventually simple learning-based behavior.

## Project Goal

This project focuses on building the basic loop of embodied intelligence:

```text
Perception → Decision → Action → Feedback → Data → Improvement

The current focus is on the mobile robot chassis, including:

Motor control
Encoder-based speed measurement
Closed-loop speed control
Odometry estimation
Motion data logging
ROS2 integration
Sensor-based perception
Future imitation learning experiments
Roadmap
Stage 1: Motion Control and Odometry

Build a stable mobile robot base with reliable motor control, encoder feedback, speed estimation, and odometry calculation.

Stage 2: Data Collection

Record motion data such as encoder counts, wheel speeds, target velocity, actual velocity, PWM output, estimated pose, and command type.

Stage 3: ROS2 Integration

Connect the physical robot to ROS2 by supporting /cmd_vel input and publishing /odom output.

Stage 4: Environmental Perception

Add sensors such as ultrasonic sensors, ToF sensors, 2D LiDAR, or cameras for basic environment understanding.

Stage 5: Human Demonstration Data

Collect teleoperation data from human control, including robot state, sensor input, action commands, and task results.

Stage 6: Simple Embodied Learning

Explore basic imitation learning or rule-learning experiments based on the collected robot data.

Current Status

The project has just started. The current priority is to define the project structure and build a clear learning roadmap.

The first engineering focus is:

Stable chassis motion → Reliable odometry → Motion data logging → ROS2 connection
Repository Structure
embodied-car-learning/
├── README.md
├── docs/
├── logs/
├── firmware/
├── ros2_ws/
├── data/
├── scripts/
├── experiments/
└── assets/
Long-Term Direction

This project will gradually connect mobile robot engineering with embodied AI.

The near-term focus is engineering reliability. The long-term direction is to collect useful robot data and explore simple learning-based robot behaviors.