# Day 1 - Project Initialization

Today I created the `embodied-car-learning` repository as a long-term learning project for embodied AI on mobile robots.

The main purpose of this project is to build a real engineering foundation for embodied intelligence instead of treating embodied AI as an empty concept or buzzword.

At this stage, I should not directly jump into advanced topics such as VLA, reinforcement learning, imitation learning, or large-model-based robot control. Those topics may become meaningful later, but only after the robot platform can move reliably, measure its own state, collect useful data, and connect with a higher-level software system.

The core idea of this project is to gradually build the loop of embodied intelligence:

Perception → Decision → Action → Feedback → Data → Improvement

For a mobile robot, this loop should start from the most basic but important engineering layers:

motion control, encoder feedback, speed measurement, closed-loop control, odometry estimation, data logging, ROS2 integration, environmental perception, human teleoperation data, and eventually simple learning-based behavior.

The key understanding today is:

No stable motion control, no reliable data.  
No reliable data, no real embodied intelligence.

Therefore, the first priority of this project is not to make the robot look intelligent, but to make the robot platform stable, measurable, reproducible, and extensible.

The initial roadmap is:

1. Build a stable mobile robot chassis.
2. Implement encoder-based speed measurement.
3. Add closed-loop speed control.
4. Estimate odometry from wheel motion.
5. Record motion data during robot movement.
6. Connect the robot to ROS2.
7. Support `/cmd_vel` input and `/odom` output.
8. Add environmental sensors such as ultrasonic sensors, ToF sensors, 2D LiDAR, or cameras.
9. Collect human teleoperation data.
10. Explore simple imitation learning or rule-learning experiments based on collected data.

This project should become a bridge between traditional mobile robot engineering and future embodied AI experiments.

For now, the most important principle is to stay grounded:

Build the foundation first.

The current task is simple but important: define the repository structure, write the initial documentation, and keep the project moving forward with clear logs and reproducible progress.