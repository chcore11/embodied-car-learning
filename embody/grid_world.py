from __future__ import annotations

import csv
import math
import random
import struct
import zlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Direction(str, Enum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"


class Action(str, Enum):
    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


TURN_LEFT = {
    Direction.NORTH: Direction.WEST,
    Direction.WEST: Direction.SOUTH,
    Direction.SOUTH: Direction.EAST,
    Direction.EAST: Direction.NORTH,
}

TURN_RIGHT = {
    Direction.NORTH: Direction.EAST,
    Direction.EAST: Direction.SOUTH,
    Direction.SOUTH: Direction.WEST,
    Direction.WEST: Direction.NORTH,
}

MOVE_DELTA = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}


@dataclass(frozen=True)
class RobotState:
    x: int
    y: int
    direction: Direction


@dataclass(frozen=True)
class Observation:
    front_blocked: bool
    left_blocked: bool
    right_blocked: bool
    distance_to_goal: int
    true_front_blocked: bool
    true_left_blocked: bool
    true_right_blocked: bool


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    step: int
    x: int
    y: int
    direction: Direction
    front_blocked: bool
    left_blocked: bool
    right_blocked: bool
    distance_to_goal: int
    action: Action
    action_success: bool
    collision: bool
    true_front_blocked: bool
    true_left_blocked: bool
    true_right_blocked: bool
    reward: float
    done: bool
    event: str


@dataclass(frozen=True)
class RunResult:
    run_id: str
    records: list[StepRecord]
    trajectory: list[tuple[int, int]]
    total_reward: float
    reached_goal: bool
    csv_path: Path
    trajectory_path: Path
    collision_count: int = 0
    action_failure_count: int = 0
    action_fail_prob: float = 0.0
    sensor_noise_prob: float = 0.0
    random_seed: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class StepResult:
    state: RobotState
    reward: float
    done: bool
    event: str
    action_success: bool = True
    collision: bool = False


class GridWorld:
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: set[tuple[int, int]],
        goal: tuple[int, int],
        action_fail_prob: float = 0.0,
        sensor_noise_prob: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.goal = goal
        self.action_fail_prob = action_fail_prob
        self.sensor_noise_prob = sensor_noise_prob
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        if not self.in_bounds(goal):
            raise ValueError("goal must be inside the grid")
        if goal in self.obstacles:
            raise ValueError("goal cannot be an obstacle")
        if not 0.0 <= action_fail_prob <= 1.0:
            raise ValueError("action_fail_prob must be between 0.0 and 1.0")
        if not 0.0 <= sensor_noise_prob <= 1.0:
            raise ValueError("sensor_noise_prob must be between 0.0 and 1.0")

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, cell: tuple[int, int]) -> bool:
        return not self.in_bounds(cell) or cell in self.obstacles

    def distance_to_goal(self, state: RobotState) -> int:
        return abs(state.x - self.goal[0]) + abs(state.y - self.goal[1])

    def observe(self, state: RobotState) -> Observation:
        true_observation = self.true_observe(state)
        return Observation(
            front_blocked=self._maybe_flip(true_observation.front_blocked),
            left_blocked=self._maybe_flip(true_observation.left_blocked),
            right_blocked=self._maybe_flip(true_observation.right_blocked),
            distance_to_goal=true_observation.distance_to_goal,
            true_front_blocked=true_observation.true_front_blocked,
            true_left_blocked=true_observation.true_left_blocked,
            true_right_blocked=true_observation.true_right_blocked,
        )

    def true_observe(self, state: RobotState) -> Observation:
        left_direction = TURN_LEFT[state.direction]
        right_direction = TURN_RIGHT[state.direction]
        front_blocked = self.is_blocked(next_cell(state, state.direction))
        left_blocked = self.is_blocked(next_cell(state, left_direction))
        right_blocked = self.is_blocked(next_cell(state, right_direction))
        return Observation(
            front_blocked=front_blocked,
            left_blocked=left_blocked,
            right_blocked=right_blocked,
            distance_to_goal=self.distance_to_goal(state),
            true_front_blocked=front_blocked,
            true_left_blocked=left_blocked,
            true_right_blocked=right_blocked,
        )

    def step(self, state: RobotState, action: Action) -> StepResult:
        if self.action_fail_prob > 0.0 and self._rng.random() < self.action_fail_prob:
            return StepResult(
                state=state,
                reward=-0.5,
                done=False,
                event="action_failed",
                action_success=False,
                collision=False,
            )

        previous_distance = self.distance_to_goal(state)

        if action == Action.TURN_LEFT:
            return StepResult(
                state=RobotState(state.x, state.y, TURN_LEFT[state.direction]),
                reward=-0.2,
                done=False,
                event="turn",
                action_success=True,
                collision=False,
            )

        if action == Action.TURN_RIGHT:
            return StepResult(
                state=RobotState(state.x, state.y, TURN_RIGHT[state.direction]),
                reward=-0.2,
                done=False,
                event="turn",
                action_success=True,
                collision=False,
            )

        if action != Action.FORWARD:
            raise ValueError(f"unknown action: {action}")

        target = next_cell(state, state.direction)
        if self.is_blocked(target):
            return StepResult(
                state=state,
                reward=-2.0,
                done=False,
                event="collision",
                action_success=True,
                collision=True,
            )

        new_state = RobotState(target[0], target[1], state.direction)
        if target == self.goal:
            return StepResult(
                state=new_state,
                reward=10.0,
                done=True,
                event="goal",
                action_success=True,
                collision=False,
            )

        new_distance = self.distance_to_goal(new_state)
        progress_reward = float(previous_distance - new_distance)
        return StepResult(
            state=new_state,
            reward=progress_reward - 0.1,
            done=False,
            event="move",
            action_success=True,
            collision=False,
        )

    def _maybe_flip(self, value: bool) -> bool:
        if self.sensor_noise_prob <= 0.0:
            return value
        return not value if self._rng.random() < self.sensor_noise_prob else value


def next_cell(state: RobotState, direction: Direction) -> tuple[int, int]:
    dx, dy = MOVE_DELTA[direction]
    return state.x + dx, state.y + dy


def create_default_world() -> tuple[GridWorld, RobotState]:
    world = GridWorld(
        width=8,
        height=8,
        obstacles={(2, 1), (2, 2), (2, 3), (4, 4), (5, 4), (5, 5)},
        goal=(7, 7),
    )
    start = RobotState(x=0, y=0, direction=Direction.EAST)
    return world, start


def run_episode(
    world: GridWorld,
    start: RobotState,
    policy,
    output_dir: Path,
    run_id: str = "v0_1_demo",
    max_steps: int = 64,
    csv_name: str | None = None,
    trajectory_name: str | None = None,
) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (csv_name or f"{run_id}.csv")
    trajectory_path = output_dir / (trajectory_name or f"{run_id}_trajectory.png")

    state = start
    records: list[StepRecord] = []
    trajectory = [(state.x, state.y)]
    total_reward = 0.0
    reached_goal = False
    collision_count = 0
    action_failure_count = 0

    for step_index in range(max_steps):
        observation = world.observe(state)
        action = policy.choose_action(world, state, observation)
        result = world.step(state, action)
        state = result.state
        total_reward += result.reward
        reached_goal = result.done and result.event == "goal"
        if result.collision:
            collision_count += 1
        if not result.action_success:
            action_failure_count += 1
        trajectory.append((state.x, state.y))

        next_observation = world.observe(state)
        records.append(
            StepRecord(
                run_id=run_id,
                step=step_index,
                x=state.x,
                y=state.y,
                direction=state.direction,
                front_blocked=next_observation.front_blocked,
                left_blocked=next_observation.left_blocked,
                right_blocked=next_observation.right_blocked,
                distance_to_goal=next_observation.distance_to_goal,
                action=action,
                action_success=result.action_success,
                collision=result.collision,
                true_front_blocked=next_observation.true_front_blocked,
                true_left_blocked=next_observation.true_left_blocked,
                true_right_blocked=next_observation.true_right_blocked,
                reward=result.reward,
                done=result.done,
                event=result.event,
            )
        )

        if result.done:
            break

    write_csv(csv_path, records)
    write_trajectory_png(trajectory_path, world, trajectory)
    failure_reason = None
    if not reached_goal and len(records) >= max_steps:
        failure_reason = "max_steps_exceeded"
    return RunResult(
        run_id=run_id,
        records=records,
        trajectory=trajectory,
        total_reward=total_reward,
        reached_goal=reached_goal,
        csv_path=csv_path,
        trajectory_path=trajectory_path,
        collision_count=collision_count,
        action_failure_count=action_failure_count,
        action_fail_prob=world.action_fail_prob,
        sensor_noise_prob=world.sensor_noise_prob,
        random_seed=world.random_seed,
        failure_reason=failure_reason,
    )


def write_csv(path: Path, records: list[StepRecord]) -> None:
    fieldnames = [
        "run_id",
        "step",
        "x",
        "y",
        "direction",
        "front_blocked",
        "left_blocked",
        "right_blocked",
        "distance_to_goal",
        "action",
        "action_success",
        "collision",
        "true_front_blocked",
        "true_left_blocked",
        "true_right_blocked",
        "reward",
        "done",
        "event",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_id": record.run_id,
                    "step": record.step,
                    "x": record.x,
                    "y": record.y,
                    "direction": record.direction.value,
                    "front_blocked": int(record.front_blocked),
                    "left_blocked": int(record.left_blocked),
                    "right_blocked": int(record.right_blocked),
                    "distance_to_goal": record.distance_to_goal,
                    "action": record.action.value,
                    "action_success": int(record.action_success),
                    "collision": int(record.collision),
                    "true_front_blocked": int(record.true_front_blocked),
                    "true_left_blocked": int(record.true_left_blocked),
                    "true_right_blocked": int(record.true_right_blocked),
                    "reward": f"{record.reward:.2f}",
                    "done": int(record.done),
                    "event": record.event,
                }
            )


def write_trajectory_png(
    path: Path,
    world: GridWorld,
    trajectory: list[tuple[int, int]],
    cell_size: int = 40,
) -> None:
    width = world.width * cell_size
    height = world.height * cell_size
    pixels = [
        [(245, 245, 245) for _ in range(width)]
        for _ in range(height)
    ]

    for x in range(world.width + 1):
        draw_line(pixels, x * cell_size, 0, x * cell_size, height - 1, (210, 210, 210))
    for y in range(world.height + 1):
        draw_line(pixels, 0, y * cell_size, width - 1, y * cell_size, (210, 210, 210))

    for obstacle in world.obstacles:
        fill_cell(pixels, obstacle, cell_size, (60, 60, 60))

    fill_cell(pixels, world.goal, cell_size, (80, 170, 90))

    for first, second in zip(trajectory, trajectory[1:]):
        x1, y1 = cell_center(first, cell_size)
        x2, y2 = cell_center(second, cell_size)
        draw_line(pixels, x1, y1, x2, y2, (40, 120, 220), thickness=5)

    if trajectory:
        fill_circle(pixels, cell_center(trajectory[0], cell_size), 9, (40, 90, 200))
        fill_circle(pixels, cell_center(trajectory[-1], cell_size), 9, (220, 80, 70))

    write_png(path, pixels)


def fill_cell(
    pixels: list[list[tuple[int, int, int]]],
    cell: tuple[int, int],
    cell_size: int,
    color: tuple[int, int, int],
) -> None:
    x, y = cell
    for py in range(y * cell_size + 2, (y + 1) * cell_size - 2):
        for px in range(x * cell_size + 2, (x + 1) * cell_size - 2):
            set_pixel(pixels, px, py, color)


def cell_center(cell: tuple[int, int], cell_size: int) -> tuple[int, int]:
    x, y = cell
    return x * cell_size + cell_size // 2, y * cell_size + cell_size // 2


def draw_line(
    pixels: list[list[tuple[int, int, int]]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for index in range(steps + 1):
        t = index / steps
        x = round(x1 + (x2 - x1) * t)
        y = round(y1 + (y2 - y1) * t)
        fill_circle(pixels, (x, y), max(0, thickness // 2), color)


def fill_circle(
    pixels: list[list[tuple[int, int, int]]],
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center
    for py in range(cy - radius, cy + radius + 1):
        for px in range(cx - radius, cx + radius + 1):
            if math.hypot(px - cx, py - cy) <= radius:
                set_pixel(pixels, px, py, color)


def set_pixel(
    pixels: list[list[tuple[int, int, int]]],
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
        pixels[y][x] = color


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    raw_rows = []
    for row in pixels:
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row))
    raw_data = b"".join(raw_rows)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        body = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw_data)),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)
