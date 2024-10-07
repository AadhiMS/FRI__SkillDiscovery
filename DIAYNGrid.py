import random
from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        num_goals=5,  # Number of diverse goals
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_goals = num_goals  # Store number of goals for DIAYN

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        # Create diverse missions
        goals = [
            "reach the green door",
            "find the key in room A",
            "open the red door",
            "get to the bottom-right corner",
            "collect all keys",
        ]
        return random.choice(goals)  # Randomly select a goal

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Vertical separation wall
        for i in range(height):
            self.grid.set(5, i, Wall())
        
        # Place doors and keys for multiple goals
        for i in range(self.num_goals):
            self.grid.set(5, 6 + i, Door(COLOR_NAMES[i % len(COLOR_NAMES)], is_locked=True))
            self.grid.set(3, 6 + i, Key(COLOR_NAMES[i % len(COLOR_NAMES)]))

        # Place the goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()  # Set a mission

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Calculate intrinsic reward based on diversity
        # This is a placeholder; customize based on your criteria
        intrinsic_reward = self.calculate_intrinsic_reward()
        reward += intrinsic_reward

        return obs, reward, done, info

    def calculate_intrinsic_reward(self) -> float:
        # Implement your intrinsic reward calculation here
        return random.uniform(0, 1)  # Example: random reward

def main():
    env = SimpleEnv(render_mode="human")

    # Enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()