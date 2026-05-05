# Learning Roadmap

This document defines the learning roadmap of the `embodied-car-learning` project.

The goal of this project is to build a mobile robot platform for embodied AI learning. The project will not start from advanced AI concepts directly. Instead, it will begin with motion control, odometry, data collection, ROS2 integration, and then gradually move toward perception, human demonstration data, and simple learning-based behavior.

The core principle is:

```text
No stable motion control, no reliable data.
No reliable data, no real embodied intelligence.
```

## Overall Roadmap

The long-term roadmap is:

```text
Motion Control
→ Encoder Feedback
→ Speed Closed-loop Control
→ Odometry
→ Data Logging
→ ROS2 Integration
→ Environmental Perception
→ Human Demonstration Data
→ Simple Embodied Learning
```

This roadmap follows the basic loop of embodied intelligence:

```text
Perception → Decision → Action → Feedback → Data → Improvement
```

For a mobile robot, this means the robot should be able to sense its own state and the environment, make decisions, execute actions through motors, receive feedback from the real world, record data, and improve its behavior through analysis or learning.

---

# Stage 1 - Reliable Mobile Robot Base

## Goal

Build a stable and measurable mobile robot chassis.

At this stage, the robot does not need to be intelligent. The most important task is to make the robot move reliably and generate trustworthy feedback data.

## Key Topics

- Motor driver control
- PWM output
- Direction control
- Encoder feedback
- Wheel speed calculation
- Basic motion commands
- Forward movement
- Turning movement
- Motion repeatability

## Questions to Answer

- How does PWM affect wheel speed?
- How are encoder pulses converted into wheel speed?
- Why do the left and right wheels move at different speeds?
- How much error appears when the robot moves 50 cm or 100 cm?
- Can the robot repeat the same motion command with similar results?

## Deliverables

- Basic motor control code
- Encoder reading code
- Wheel speed calculation document
- Basic movement test log
- Motion error test table

## Acceptance Criteria

- The robot can move forward stably.
- The robot can turn left and right.
- Encoder data from both wheels is readable.
- Wheel speed can be calculated.
- Motion error can be measured and recorded.

---

# Stage 2 - Speed Closed-loop Control

## Goal

Implement closed-loop speed control for the mobile robot chassis.

The robot should not only receive PWM commands. It should receive target speed commands and adjust PWM according to encoder feedback.

## Key Topics

- Target speed
- Actual speed
- Speed error
- PID control
- Left and right wheel synchronization
- Straight-line correction
- Parameter tuning

## Questions to Answer

- What is the difference between PWM control and speed control?
- How does PID reduce speed error?
- Why does the robot drift when both wheels receive the same PWM?
- How can encoder feedback be used to reduce drift?
- How should PID parameters be recorded and tested?

## Deliverables

- Speed closed-loop control code
- PID parameter table
- Speed response test data
- Straight-line motion test log
- Error analysis document

## Acceptance Criteria

- The robot can follow a target speed.
- Left and right wheel speed error is reduced.
- The robot can move more straight than open-loop control.
- PID parameters are recorded clearly.
- Test results can be reproduced.

---

# Stage 3 - Odometry Estimation

## Goal

Estimate the robot's position and orientation from wheel encoder data.

The robot should be able to estimate its own motion state, including position and heading angle.

## Key Topics

- Differential drive model
- Wheel displacement
- Linear velocity
- Angular velocity
- Position estimation
- Heading angle estimation
- Odometry drift
- Calibration

## Questions to Answer

- How are encoder pulses converted into wheel displacement?
- How are left and right wheel displacements used to estimate robot movement?
- How are `x`, `y`, and `theta` calculated?
- Why does odometry drift over time?
- What parameters need to be calibrated?

## Deliverables

- Odometry calculation document
- Encoder-to-distance conversion formula
- Test data for forward movement
- Test data for turning movement
- Odometry error analysis

## Acceptance Criteria

- The robot can estimate forward displacement.
- The robot can estimate rotation angle.
- Odometry data can be logged.
- Main error sources are identified.
- Calibration parameters are documented.

---

# Stage 4 - Motion Data Logging

## Goal

Build a basic data collection system for robot motion.

The robot should be able to record state, action, and result during movement.

## Key Topics

- Timestamp
- Encoder data
- Wheel speed
- Target speed
- PWM output
- Estimated pose
- Command type
- Test result
- CSV logging
- Serial data output

## Basic Data Format

```csv
time,left_encoder,right_encoder,left_speed,right_speed,target_v,target_w,pwm_l,pwm_r,x,y,theta,command,result
```

## Questions to Answer

- What data should be recorded during robot movement?
- What is the difference between state, action, and result?
- How can serial output be saved as CSV?
- How can data be used to analyze motion error?
- What data is useful for future imitation learning?

## Deliverables

- Data field definition document
- Serial output protocol
- CSV logging script
- Sample motion dataset
- Data analysis notes

## Acceptance Criteria

- Motion data can be recorded.
- Data can be saved in CSV format.
- Each row contains state, action, and result information.
- Test data can be used for error analysis.
- Data format is documented clearly.

---

# Stage 5 - ROS2 Integration

## Goal

Connect the physical mobile robot to ROS2.

The robot should support ROS2 command input and publish robot state feedback.

## System Flow

```text
ROS2 /cmd_vel
↓
Serial Protocol
↓
STM32 Motion Controller
↓
Motor + Encoder
↓
Odometry Feedback
↓
ROS2 /odom
```

## Key Topics

- ROS2 node
- `/cmd_vel`
- `/odom`
- TF
- `base_link`
- `odom`
- Serial communication
- Robot state publishing
- Velocity command parsing

## Questions to Answer

- What is `/cmd_vel`?
- What is `/odom`?
- What is TF?
- What is the relationship between `odom` and `base_link`?
- How does ROS2 send motion commands to STM32?
- How does STM32 send odometry data back to ROS2?

## Deliverables

- Serial protocol document
- ROS2 node for command sending
- ROS2 node for odometry publishing
- `/cmd_vel` test
- `/odom` test
- TF structure document

## Acceptance Criteria

- ROS2 can send velocity commands to the robot.
- STM32 can receive and execute commands.
- The robot can publish odometry data.
- `/cmd_vel` and `/odom` are verified.
- Basic TF relationship is established.

---

# Stage 6 - Environmental Perception

## Goal

Add sensors so the robot can perceive the environment.

The robot should gradually move from self-state feedback to environment-state feedback.

## Sensor Roadmap

```text
Encoder + IMU
→ Ultrasonic / ToF Sensors
→ 2D LiDAR
→ Camera
→ Multi-sensor Data Synchronization
```

## Key Topics

- Obstacle distance
- Sensor noise
- Sensor calibration
- Obstacle detection
- Basic obstacle avoidance
- LiDAR scan
- Camera image data
- Time synchronization

## Questions to Answer

- What can each sensor measure?
- How reliable is the sensor data?
- How can the robot detect obstacles?
- How can sensor data be combined with odometry?
- Why is time synchronization important?

## Deliverables

- Sensor selection notes
- Sensor wiring document
- Obstacle detection test
- Basic obstacle avoidance demo
- Sensor data log

## Acceptance Criteria

- At least one environment sensor works.
- The robot can detect nearby obstacles.
- Sensor data can be recorded.
- Sensor data can be connected with robot motion data.
- Simple obstacle avoidance is possible.

---

# Stage 7 - Navigation and Task Execution

## Goal

Enable the robot to complete simple navigation or task-oriented movement.

The robot should not only move by manual command, but also execute simple goals.

## Key Topics

- Goal point
- Path following
- Obstacle avoidance
- Map
- Localization
- Navigation behavior
- Task success rate

## Questions to Answer

- How does the robot move from one point to another?
- How does the robot avoid obstacles?
- What causes navigation failure?
- How can task success be measured?
- What data should be recorded during a task?

## Deliverables

- Simple navigation demo
- Goal execution test
- Task result log
- Failure case analysis
- Navigation data record

## Acceptance Criteria

- The robot can execute a simple goal.
- The robot can avoid obvious obstacles.
- Task success and failure can be recorded.
- Navigation behavior can be analyzed from data.

---

# Stage 8 - Human Demonstration Data

## Goal

Collect human teleoperation data for future imitation learning.

The robot should record how a human controls it in different situations.

## Key Topics

- Teleoperation
- State-action pair
- Human command
- Sensor state
- Task result
- Demonstration dataset

## Data Structure

```text
State: robot speed, odometry, sensor readings
Action: human command, target velocity, steering command
Result: success, failure, collision, distance error
```

## Example

```text
State: front distance = 25 cm, left distance = 60 cm, right distance = 20 cm
Action: turn left
Result: obstacle avoided successfully
```

## Questions to Answer

- What does the human see when controlling the robot?
- What action does the human choose?
- What is the result of that action?
- How can this process be recorded as data?
- Is the collected data clean enough for learning?

## Deliverables

- Teleoperation interface
- Demonstration data format
- Sample demonstration dataset
- State-action-result records
- Data quality notes

## Acceptance Criteria

- Human control commands can be recorded.
- Sensor state can be recorded at the same time.
- Robot result can be labeled.
- A small demonstration dataset is created.

---

# Stage 9 - Simple Embodied Learning

## Goal

Explore simple learning-based behavior using collected robot data.

At this stage, the project can begin to connect engineering data with AI methods.

## Key Topics

- Rule-based behavior
- Imitation learning
- Classification model
- Action prediction
- Policy evaluation
- Dataset quality

## Possible First Experiment

Input:

```text
front_distance, left_distance, right_distance, current_speed
```

Output:

```text
forward, stop, turn_left, turn_right
```

## Questions to Answer

- Can the model predict a reasonable action from sensor data?
- Is the dataset large enough?
- Is the data balanced?
- Does the learned behavior work on the real robot?
- What failure cases appear?

## Deliverables

- Simple dataset
- Action prediction model
- Offline evaluation result
- Real robot test
- Failure analysis

## Acceptance Criteria

- A simple model can be trained.
- The model can predict actions from robot state.
- The prediction can be tested on the robot or in simulation.
- Failure cases are documented.

---

# 30-Day Focus Plan

For the first 30 days, the project should not focus on advanced AI.

The focus should be:

```text
Reliable chassis motion
→ Encoder feedback
→ Speed closed-loop control
→ Odometry
→ Data logging
→ ROS2 integration design
```

## Week 1 - Concept and Architecture

Goals:

- Understand embodied AI from a mobile robot perspective.
- Define the minimal embodied loop.
- Draw the robot system architecture.
- Define data collection fields.

Deliverables:

- `docs/01_embodied_ai_basic.md`
- `docs/02_robot_system_architecture.md`
- `docs/03_data_collection_plan.md`
- `docs/04_learning_roadmap.md`

## Week 2 - Motion and Encoder

Goals:

- Verify motor control.
- Verify encoder feedback.
- Calculate wheel speed.
- Test forward movement error.

Deliverables:

- Encoder test log
- Wheel speed calculation notes
- Motion test table
- Error analysis

## Week 3 - Closed-loop Control and Odometry

Goals:

- Implement basic speed closed-loop control.
- Tune PID parameters.
- Calculate odometry.
- Test forward and turning motion.

Deliverables:

- PID parameter table
- Odometry calculation document
- Motion data CSV
- Error analysis log

## Week 4 - Data Logging and ROS2 Plan

Goals:

- Build a basic data logging pipeline.
- Save motion data to CSV.
- Design serial protocol.
- Plan ROS2 `/cmd_vel` and `/odom` integration.

Deliverables:

- Data logging script
- Sample dataset
- Serial protocol document
- ROS2 integration plan

---

# Current Priority

The current priority is not VLA, reinforcement learning, or large model control.

The current priority is:

```text
Build a reliable robot body.
Make the data trustworthy.
Connect the robot to a real robotics software system.
```

Only after these foundations are built, advanced embodied AI methods will become meaningful.
