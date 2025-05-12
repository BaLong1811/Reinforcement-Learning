from envs.simple_dungeonworld_env import DungeonMazeEnv
import numpy as np
from enum import IntEnum

class Actions(IntEnum):
    turn_right = 0
    turn_left = 1
    move_forwards = 2

SIZE = 8

def rollout(env, policy, max_steps=100):
    """
    Samples a complete trajectory in the environment using the given policy.
    
    Args:
        env (DungeonMazeEnv): The environment instance.
        policy (callable): A function that takes an observation and returns an action.
        max_steps (int): The maximum number of steps before termination.

    Returns:
        list: A trajectory containing (state, action, reward) tuples.
    """
    trajectory = []
    observation = []
    obs, _ = env.reset()
    # env.reset(seed=124)
    
    for i in range(max_steps):
        state = obs.copy()
        action = policy(state)
        print("action: ", action)
        obs, reward, terminated, stuck, info = env.step(action)
        direction = obs['robot_direction']  # Extract direction from observation dictionary
        position = obs['robot_position']
        print("position: ", position)
        print("reward: ", reward)        
        
        trajectory.append((state, action, reward))
        observation.append(direction)
        
        if terminated:
            break
    
    return trajectory, observation

# random policy
def random_policy(observation):
    return Actions.turn_right
    # return np.random.choice(len(Actions))  # Random action selection

# Usage 
if __name__ == "__main__":
    # env = DungeonMazeEnv(grid_size=6)
    env = DungeonMazeEnv(render_mode="human", grid_size=SIZE)
    traj, observation = rollout(env, random_policy)
    
    # for step in traj:
    for step in observation:
        print(step)

