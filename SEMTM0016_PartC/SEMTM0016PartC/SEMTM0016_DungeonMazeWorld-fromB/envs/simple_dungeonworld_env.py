"""
Simple HeroBot and the MazeDungeon Environment.
"""

from enum import IntEnum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from core.dungeonworld_grid import MazeGrid, Wall, Orc, Wingedbat, Lizard


class Actions(IntEnum):
    # Enumeration of possible actions
    # Turn right, turn left, move forwards
    turn_right = 0
    turn_left = 1
    move_forwards = 2


class Directions(IntEnum):
    # Enumeration of cardinal directions the robot can face
    # taking north as top of maze
    north = 0
    east = 1
    south = 2
    west = 3


class DungeonMazeEnv(gym.Env):
    """
    2D maze grid world environment for robot.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=16):
        """
        Initialises the simulation environment with the given grid size.
        """
        self.grid_size = grid_size
        self.window_size = 512

        # We have 3 actions, corresponding to "turn right", "turn left", "move forwards"
        self.action_space = spaces.Discrete(len(Actions))

        # Observations are dictionaries with:
        # The robot postion encoded as an element of {0, ..., size-1}^2,
        # The robot direction encoded as an integer {0, ..., 4},
        # The robot camera view encoded as a dummy 20x20 pixel greyscale image, {0, ..., 255}^(20x20)
        # The target postion encoded as an element of {0, ..., size-1}^2.
        self.observation_space = spaces.Dict(
            {
                "robot_position": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
                "robot_direction": spaces.Discrete(len(Directions)),
                "robot_camera_view": spaces.Box(
                    low=0, high=255, shape=(20, 20), dtype=np.int32
                ),
                "target_position": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.

        self.window = None
        self.clock = None

        # MODIFY: Create entities
        self.entities = []  # List to track objects like orcs
        self._initialize_entities()  # Call function to place orcs

    # MODIFY: Function to initialize entities
    def _initialize_entities(self):
        """Manually place orcs in the environment"""
        orc1 = Orc(pos=(3, 4), image_id=0)  # Example position
        orc2 = Orc(pos=(5, 2), image_id=1)  # Another orc
        self.entities.append(orc1)
        self.entities.append(orc2)

    def get_observations(self):
        """
        Returns a dictionary containing the robot's position, direction and camera view
        and the target position.
        """
        return {
            "robot_position": self.robot_position,
            "robot_direction": self.robot_direction,
            "robot_camera_view": self.robot_camera_view,
            "target_position": self.target_position,
        }

    def get_robot_direction_vector(self):
        """
        Get the direction vector for the robot, pointing in the direction
        of forward movement.
        """
        direction_vectors = [
            # Up (negative Y)
            np.array((0, -1)),
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
        ]
        assert self.robot_direction >= 0 and self.robot_direction < 4
        return direction_vectors[self.robot_direction]

    def get_robot_front_pos(self):
        """
        Get the position of the cell that is right in front of the robot
        """
        return self.robot_position + self.get_robot_direction_vector()

    def get_robot_camera_view(self):
        """
        Returns the 'camera view' for the robot i.e. the image the object in the cell
        in front of the robot. If the cell is empty, then returns a white image.
        """
        # Get the position in front of the robot
        position_in_front = self.get_robot_front_pos()
        # print("position_in_front: ", position_in_front)

        # Get the contents of the cell in front of the agent
        cell_in_front = self.maze.get_cell_item(*position_in_front)
        # print("cell_in_front: ", cell_in_front)

        if cell_in_front is None:
            # if nothing in front return a white image
            return np.ones((20, 20)) * 255
        else:
            return cell_in_front.get_camera_view()

    def reset(self, seed=None, options=None):
        """
        Initialises the environment for a new episode with a randomly generated maze.
        Robot is always initialised at position [1,1] facing south.
        Target is always initialised at [-2,-2].
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create the grid capturing the maze as walls
        self.maze = MazeGrid(size=self.grid_size, empty=False, np_rng=self.np_random)
        maze_init = self.maze

        # Set the target location
        self.target_position = np.array([self.grid_size - 2, self.grid_size - 2])

        # Set the robot's location, direction, inital camera view
        self.robot_position = np.array([1, 1])
        self.robot_direction = Directions.south
        self.robot_camera_view = self.get_robot_camera_view()

        # MODIFY: Add obstacles, enemies, or power-ups
        orc_entity_1 = Orc(pos=(3, 4), image_id=5)  # Use any image_id between 0-99
        self.maze.add_cell_item(3, 4, orc_entity_1)  # Orc at (1,2)

        # orc_entity = Orc(pos=(5, 4), image_id=9)  # Use any image_id between 0-99
        # self.maze.add_cell_item(5, 4, orc_entity)  # Orc at (1,2)

        # Add Lizard entity
        lizard_entity = Lizard(pos=(2, 6), image_id=6)  # Use any image_id between 0-99
        self.maze.add_cell_item(2, 6, lizard_entity)  # Lizard at (2,6)

        # Add Wingedbat entity
        wingedbat_entity = Wingedbat(pos=(6, 3), image_id=7)  # Use any image_id between 0-99
        self.maze.add_cell_item(6, 3, wingedbat_entity)  # Wingedbat at (4,3)

        # # Add Wingedbat entity
        # wingedbat_entity = Wingedbat(pos=(5, 3), image_id=7)  # Use any image_id between 0-99
        # self.maze.add_cell_item(5, 3, wingedbat_entity)  # Wingedbat at (4,3)

        # Update the observations
        observation = self.get_observations()

        if self.render_mode == "human":
            self._render_frame()

        # return observation, {}
        return observation, maze_init, {}

    # MODIFY: create a new maze without resetting the environment
    def new_reset(self, maze, seed=None, options=None):
        """
        Resets the environment for a new episode WITHOUT creating a new maze.
        The robot is always initialized at position [1,1] facing south.
        The target is always initialized at [-2,-2].
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.maze = maze

        # Ensure the maze remains unchanged
        if not hasattr(self, 'maze') or self.maze is None:
            raise ValueError("Maze must be initialized before calling reset. Ensure it is created in __init__().")            

        # Set the target location
        self.target_position = np.array([self.grid_size - 2, self.grid_size - 2])

        # Reset the robot's position, direction, and camera view
        self.robot_position = np.array([1, 1])
        self.robot_direction = Directions.south
        self.robot_camera_view = self.get_robot_camera_view()

        # Update the observations
        observation = self.get_observations()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def step(self, action):
        """
        Performs one step of the simulation with the given action.
        Returning the new state, reward and whether the episode has terminated.
        """
        # MODIFY: Check if the action is valid
        cell_at_current_pos = self.maze.get_cell_item(*self.robot_position)
        if cell_at_current_pos is not None and not cell_at_current_pos.can_overlap():
            return self.get_observations(), -100, False, False, {}

        reward = -1
        terminated = False

        # if np.array_equal(self.robot_position, self.target_position):
        #     reward = 0

        # Get the position in front of the robot
        position_in_front = self.get_robot_front_pos()

        # Get the contents of the cell in front of the agent
        cell_in_front = self.maze.get_cell_item(*position_in_front)

        # Track the current position before moving
        prev_position = self.robot_position.copy()

        # Attempt actions
        if action == Actions.turn_left:
            self.robot_direction -= 1
            if self.robot_direction < 0:
                self.robot_direction += 4

            # MODIFY: Check if the robot is facing an obstacle after turning
            new_front_pos = self.get_robot_front_pos()  # Recalculate front position after turning
            new_cell_in_front = self.maze.get_cell_item(*new_front_pos)

            # MODIFY: Check if an obstacle is in front after turning
            if new_cell_in_front is not None and not new_cell_in_front.can_overlap():
                reward = -100  # Heavy penalty for facing an obstacle
        elif action == Actions.turn_right:
            self.robot_direction += 1
            if self.robot_direction > 3:
                self.robot_direction -= 4

            # MODIFY: Check if the robot is facing an obstacle after turning
            new_front_pos = self.get_robot_front_pos()  # Recalculate front position after turning
            new_cell_in_front = self.maze.get_cell_item(*new_front_pos)

            # MODIFY: Check if an obstacle is in front after turning
            if new_cell_in_front is not None and not new_cell_in_front.can_overlap():
                reward = -100  # Heavy penalty for facing an obstacle
        elif action == Actions.move_forwards:
            # MODIFY: Check if the robot is trying to move out of bounds
            # if cell_in_front is not None and not cell_in_front.can_overlap():
            #     reward = -100  # Heavy penalty for facing an obstacle

            if cell_in_front is not None:
                if cell_in_front.is_target():
                    reward = -1
                else:
                    if not cell_in_front.can_overlap():
                        reward = -100  # Heavy penalty for facing an obstacle
                    elif cell_in_front.can_overlap():
                        # if cell_in_front.can_be_killed_by_bow() and cell_in_front.can_be_killed_by_sword():
                        if cell_in_front.can_be_killed_by_bow():
                            reward = -1
                        else:
                            reward = -40

            if cell_in_front is None or cell_in_front.can_overlap():
                self.robot_position = position_in_front
            else:
                # Terminate with penalty as robot tried to crash into an object in the cell in front.
                # MODIFY: 
                # terminated = True
                reward = -100

            # if np.array_equal(position_in_front, previous_position):
            #     reward = -100  # Heavy penalty for backtracking
            # elif cell_in_front is None or cell_in_front.can_overlap():
            #     self.robot_position = position_in_front
            #     # previous_position = previous_position  # Update only after a valid move
            # else:
            #     # Terminate with penalty as robot tried to crash into an object in the cell in front.
            #     # terminated = True
            #     reward = -100

        else:
            assert False, "unknown action"

        # # **Prevent turning back to the previous position**
        # if np.array_equal(self.robot_position, prev_position):
        #     reward = -50  # Discourage backtracking

        # Update the robot's camera view
        self.robot_camera_view = self.get_robot_camera_view()

        # Update the observations
        observation = self.get_observations()

        # An episode is terminated if the agent has reached the target
        if np.array_equal(self.robot_position, self.target_position):
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_position,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the walls
        for cell in self.maze.grid:
            if cell is not None and cell.type == "wall":
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(
                        pix_square_size * cell.pos,
                        (pix_square_size, pix_square_size),
                    ),
                )

        # MODIFY: Draw the enemies
        for obj in self.maze.grid:
            # Draw Orc entities
            if isinstance(obj, Orc):
                orc_surface = pygame.surfarray.make_surface(obj.image)
                orc_surface = pygame.transform.scale(orc_surface, (int(pix_square_size * 1.0), int(pix_square_size * 1.0)))  # Increase size
                orc_pos = (obj.pos[0] * pix_square_size, obj.pos[1] * pix_square_size)
                canvas.blit(orc_surface, orc_pos)

            # # Draw Halfling entities
            # if isinstance(obj, Halfling):
            #     halfling_surface = pygame.surfarray.make_surface(obj.image)
            #     halfling_surface = pygame.transform.scale(halfling_surface, (int(pix_square_size * 1.0), int(pix_square_size * 1.0)))  # Increase size
            #     halfling_pos = (obj.pos[0] * pix_square_size, obj.pos[1] * pix_square_size)
            #     canvas.blit(halfling_surface, halfling_pos)

            # # Draw Human entities
            # if isinstance(obj, Human):
            #     human_surface = pygame.surfarray.make_surface(obj.image)
            #     human_surface = pygame.transform.scale(human_surface, (int(pix_square_size * 1.0), int(pix_square_size * 1.0)))  # Increase size
            #     human_pos = (obj.pos[0] * pix_square_size, obj.pos[1] * pix_square_size)
            #     canvas.blit(human_surface, human_pos)

            # Draw Lizard entities
            if isinstance(obj, Lizard):
                lizard_surface = pygame.surfarray.make_surface(obj.image)
                lizard_surface = pygame.transform.scale(lizard_surface, (int(pix_square_size * 1.0), int(pix_square_size * 1.0)))  # Increase size
                lizard_pos = (obj.pos[0] * pix_square_size, obj.pos[1] * pix_square_size)
                canvas.blit(lizard_surface, lizard_pos)

            # Draw Wingedbat entities
            if isinstance(obj, Wingedbat):
                wingedbat_surface = pygame.surfarray.make_surface(obj.image)
                wingedbat_surface = pygame.transform.scale(wingedbat_surface, (int(pix_square_size * 1.0), int(pix_square_size * 1.0)))  # Increase size
                wingedbat_pos = (obj.pos[0] * pix_square_size, obj.pos[1] * pix_square_size)
                canvas.blit(wingedbat_surface, wingedbat_pos)

        # MODIFY: Draw orcs uing the entities list
        # Uncomment the following lines if you want to draw orcs using the entities list
        # for entity in self.entities:
        #     if isinstance(entity, Orc):
        #         orc_surface = pygame.surfarray.make_surface(entity.image)
        #         orc_surface = pygame.transform.scale(orc_surface, (int(pix_square_size * 1.2), int(pix_square_size * 1.2)))  # Increase size
        #         orc_pos = (entity.pos[0] * pix_square_size, entity.pos[1] * pix_square_size)
        #         canvas.blit(orc_surface, orc_pos)


        # Now we draw the robot with direction it's facing
        if self.robot_direction == Directions.north:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                (
                    (self.robot_position + np.array([0.1, 0.9])) * pix_square_size,
                    (self.robot_position + np.array([0.9, 0.9])) * pix_square_size,
                    (self.robot_position + np.array([0.5, 0.1])) * pix_square_size,
                ),
            )
        elif self.robot_direction == Directions.east:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                (
                    (self.robot_position + np.array([0.1, 0.9])) * pix_square_size,
                    (self.robot_position + np.array([0.1, 0.1])) * pix_square_size,
                    (self.robot_position + np.array([0.9, 0.5])) * pix_square_size,
                ),
            )
        elif self.robot_direction == Directions.south:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                (
                    (self.robot_position + np.array([0.9, 0.1])) * pix_square_size,
                    (self.robot_position + np.array([0.1, 0.1])) * pix_square_size,
                    (self.robot_position + np.array([0.5, 0.9])) * pix_square_size,
                ),
            )
        elif self.robot_direction == Directions.west:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                (
                    (self.robot_position + np.array([0.9, 0.1])) * pix_square_size,
                    (self.robot_position + np.array([0.9, 0.9])) * pix_square_size,
                    (self.robot_position + np.array([0.1, 0.5])) * pix_square_size,
                ),
            )

        # Finally, draw some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
