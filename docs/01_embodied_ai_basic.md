# Embodied AI Basics for Mobile Robots

## 1. My Understanding of Embodied AI

Embodied AI is not only about training a large model. For a mobile robot, embodied intelligence means that the robot can sense its own state and the environment, make decisions, take actions through motors, receive feedback from the real world, and improve its behavior through data.

## 2. The Minimal Loop

For this project, the minimal embodied loop is:

Perception → Decision → Action → Feedback → Data → Improvement

For a mobile robot, this means:

- Perception: encoder data, IMU data, distance sensors, LiDAR, camera
- Decision: speed command, steering command, path planning, behavior selection
- Action: motor control through PWM and driver circuits
- Feedback: wheel motion, robot pose change, sensor readings, task result
- Data: logs of state, action, and result
- Improvement: calibration, PID tuning, odometry correction, learning-based policy

## 3. Why a Mobile Robot Is a Good Entry Point

A mobile robot is a good entry point because it has a clear body, clear actions, clear feedback, and relatively low hardware cost.

It is simple enough to build as a student project, but complete enough to include the key components of embodied intelligence:

- motion control
- sensor feedback
- real-world uncertainty
- data collection
- decision making
- system integration

## 4. What This Project Should Avoid

This project should avoid jumping too early into advanced concepts such as VLA, reinforcement learning, imitation learning, or large-model-based robot control.

These topics are meaningful only after the robot can move reliably, measure its own state, collect useful data, and connect to a higher-level software system.

## 5. Current Learning Focus

The current focus is:

Stable chassis motion → Reliable encoder feedback → Speed closed-loop control → Odometry → Data logging → ROS2 integration

The short-term goal is not to make the robot look intelligent.

The short-term goal is to build a reliable platform that can generate real feedback and real data.