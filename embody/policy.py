from __future__ import annotations

from .grid_world import Action, Direction, GridWorld, Observation, RobotState, TURN_LEFT, TURN_RIGHT, next_cell


class GreedyGridPolicy:
    """Simple v0.1 policy: face a direction that reduces goal distance, then move."""

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


def _distance(cell: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
