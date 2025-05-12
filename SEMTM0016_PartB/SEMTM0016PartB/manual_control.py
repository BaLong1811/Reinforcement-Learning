from envs.simple_dungeonworld_env import DungeonMazeEnv

SIZE = 8

# Load the simple dungeon maze env
env = DungeonMazeEnv(render_mode="human", grid_size=SIZE)

# Initialise with set seed
env.reset(seed=124)

# Manual control for checking
for i in range(100):
    action = input("input action, 0=turn right, 1=turn left, 2=move forwards")
    next_state, reward, terminated, stuck, info = env.step(int(action))
    direction = next_state['robot_direction']  # Extract direction from observation dictionary
    position = next_state['robot_position']
    print("position: ", position)
    print("direction: ", direction)
    # print("next_state", next_state)
    print("reward", reward)
    # print("terminated", terminated)
