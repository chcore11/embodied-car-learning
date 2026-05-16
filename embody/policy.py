from __future__ import annotations

from .grid_world import Action, Direction, GridWorld, Observation, RobotState, TURN_LEFT, TURN_RIGHT, next_cell


class GreedyGridPolicy:
    """Simple v0.1 policy: face a direction that reduces goal distance, then move."""

    name = "baseline_greedy"

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> Action:
        if not observation.front_blocked and _direction_improves(
            observation.direction,
            observation.dx_to_goal,
            observation.dy_to_goal,
        ):
            return Action.FORWARD

        preferred = _preferred_directions(observation.dx_to_goal, observation.dy_to_goal)
        for direction in preferred:
            if direction == observation.direction and not observation.front_blocked:
                return Action.FORWARD
            if direction == TURN_LEFT[observation.direction]:
                return Action.TURN_LEFT
            if direction == TURN_RIGHT[observation.direction]:
                return Action.TURN_RIGHT

        if not observation.left_blocked:
            return Action.TURN_LEFT
        if not observation.right_blocked:
            return Action.TURN_RIGHT
        return Action.TURN_RIGHT

    def choose_action(
        self,
        world: GridWorld,
        state: RobotState,
        observation: Observation,
    ) -> Action:
        if not observation.front_blocked and self._forward_improves(world, state):
            return Action.FORWARD

        preferred = self._preferred_directions(world, state)
        for direction in preferred:
            if direction == state.direction and not observation.front_blocked:
                return Action.FORWARD
            if direction == TURN_LEFT[state.direction]:
                return Action.TURN_LEFT
            if direction == TURN_RIGHT[state.direction]:
                return Action.TURN_RIGHT

        if not observation.left_blocked:
            return Action.TURN_LEFT
        if not observation.right_blocked:
            return Action.TURN_RIGHT
        return Action.TURN_RIGHT

    def _forward_improves(self, world: GridWorld, state: RobotState) -> bool:
        target = next_cell(state, state.direction)
        if world.is_blocked(target):
            return False
        return _distance(target, world.goal) < world.distance_to_goal(state)

    def _preferred_directions(
        self,
        world: GridWorld,
        state: RobotState,
    ) -> list[Direction]:
        directions: list[Direction] = []
        goal_x, goal_y = world.goal

        if goal_x > state.x:
            directions.append(Direction.EAST)
        elif goal_x < state.x:
            directions.append(Direction.WEST)

        if goal_y > state.y:
            directions.append(Direction.SOUTH)
        elif goal_y < state.y:
            directions.append(Direction.NORTH)

        return sorted(
            directions,
            key=lambda direction: _distance(next_cell(state, direction), world.goal),
        )


class ObstacleAwareGoalPolicy:
    """Small rule policy that avoids observed blockers and reduces simple loops."""

    name = "obstacle_aware_goal"

    def __init__(self) -> None:
        self.visited: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self.visited.clear()

    def select_action(self, observation: Observation) -> Action:
        self.visited.add((observation.x, observation.y))

        if not observation.front_blocked and _direction_improves(
            observation.direction,
            observation.dx_to_goal,
            observation.dy_to_goal,
        ):
            return Action.FORWARD

        forward_cell = _future_cell(observation.x, observation.y, observation.direction)
        if not observation.front_blocked and forward_cell not in self.visited:
            return Action.FORWARD

        candidates = [
            (Action.TURN_LEFT, TURN_LEFT[observation.direction], observation.left_blocked),
            (Action.TURN_RIGHT, TURN_RIGHT[observation.direction], observation.right_blocked),
        ]
        unblocked = [item for item in candidates if not item[2]]
        if unblocked:
            return min(
                unblocked,
                key=lambda item: self._turn_score(observation, item[1]),
            )[0]

        if not observation.front_blocked:
            return Action.FORWARD
        return Action.TURN_RIGHT

    def choose_action(
        self,
        world: GridWorld,
        state: RobotState,
        observation: Observation,
    ) -> Action:
        return self.select_action(observation)

    def _turn_score(self, observation: Observation, direction: Direction) -> tuple[int, int]:
        future = _future_cell(observation.x, observation.y, direction)
        repeat_penalty = 1 if future in self.visited else 0
        goal_score = -_direction_progress(
            direction,
            observation.dx_to_goal,
            observation.dy_to_goal,
        )
        return repeat_penalty, goal_score


GreedyPolicy = GreedyGridPolicy
RuleBasedPolicy = GreedyGridPolicy


def _distance(cell: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def _preferred_directions(dx_to_goal: int, dy_to_goal: int) -> list[Direction]:
    directions: list[Direction] = []
    if dx_to_goal > 0:
        directions.append(Direction.EAST)
    elif dx_to_goal < 0:
        directions.append(Direction.WEST)

    if dy_to_goal > 0:
        directions.append(Direction.SOUTH)
    elif dy_to_goal < 0:
        directions.append(Direction.NORTH)

    return sorted(
        directions,
        key=lambda direction: -_direction_progress(direction, dx_to_goal, dy_to_goal),
    )


def _direction_improves(
    direction: Direction,
    dx_to_goal: int,
    dy_to_goal: int,
) -> bool:
    return _direction_progress(direction, dx_to_goal, dy_to_goal) > 0


def _direction_progress(
    direction: Direction,
    dx_to_goal: int,
    dy_to_goal: int,
) -> int:
    if direction == Direction.EAST:
        return dx_to_goal
    if direction == Direction.WEST:
        return -dx_to_goal
    if direction == Direction.SOUTH:
        return dy_to_goal
    if direction == Direction.NORTH:
        return -dy_to_goal
    return 0


def _future_cell(x: int, y: int, direction: Direction) -> tuple[int, int]:
    if direction == Direction.EAST:
        return x + 1, y
    if direction == Direction.WEST:
        return x - 1, y
    if direction == Direction.SOUTH:
        return x, y + 1
    return x, y - 1
