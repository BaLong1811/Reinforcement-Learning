import numpy as np
import copy
from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions, Directions

gamma = 1  # Discount factor
eps = 1e-4  # Convergence threshold
# eps = 1  # Convergence threshold

def evaluate_policy(env, policy, eps):
    num = env.grid_size
    # value_matrix = np.full((num, num, len(Directions)), -100)  # 3D matrix: (x, y, direction)
    value_matrix = np.zeros((num, num, len(Directions)))  # 4D matrix: (x, y, direction, action)

    # test
    count = 0

    while True:
    # while count == 0:
        value_prev = np.copy(value_matrix)

        for i in range(num):
            for j in range(num):
                for d in range(len(Directions)):  # Iterate over all directions
                    if i == 0 or j == 0 or i == num-1 or j == num-1:
                        # value_matrix[i, j, d] = -100  # Borders remain at -100
                        # value_matrix[i, j, d] -= 100  # Borders remain at -100

                        next_value = -100  # Penalty for moving into boundary
                        reward = -100  # Penalty for moving into boundary
                        # value_matrix[i, j, d] += policy[i, j, d] * (reward + gamma * next_value)
                        # Sum over all possible actions using probabilities
                        value_matrix[i, j, d] = np.sum(policy[i, j, d, :] * (reward + gamma * next_value))
                        continue

                    value = 0
                    for action in range(len(Actions)):
                        env.robot_position = np.array([i, j])
                        env.robot_direction = d

                        obs, reward, _, _, _ = env.step(Actions(action))
                        next_pos = obs['robot_position']
                        next_dir = obs['robot_direction']

                        # if (next_pos[1] == 1 and next_pos[0] == 1) or (next_pos[1] == num-2 and next_pos[0] == num-2):
                        if (next_pos[1] == num-2 and next_pos[0] == num-2):
                            next_value = 0  # Target cell
                            reward = 0  # Target cell
                        elif 0 < next_pos[0] < num-1 and 0 < next_pos[1] < num-1:
                            # Check if the next position is an obstacle
                            if env.maze.get_cell_item(*next_pos) is not None:
                                next_value = -100  # Obstacle penalty
                            else:
                                next_value = value_prev[next_pos[0], next_pos[1], next_dir]
                            # print("next_value: ", next_value)
                        else:
                            next_value = -100  # Penalty for moving into boundary
                        
                        # print(f"Position ({i}, {j}, {d}) - next_value: {next_value}")
                        # print(f"Position ({i}, {j}, {d}) - reward: {reward}")
                        value += policy[i, j, d, action] * (reward + gamma * next_value)

                    value_matrix[i, j, d] = value

        # print("value_matrix at 1 1: ", value_matrix[1, 1, :])
        # print("value_matrix at 1 2: ", value_matrix[1, 2, :])
        diff = np.mean((value_prev-value_matrix)**2) ## calculate the difference
        print("diff: ", diff)
        if diff < eps: ## if less then the thresold, break
            break

        # test
        count += 1

    return value_matrix

import numpy as np

def update_value_function(env, gamma, eps, value_matrix):
    num = env.grid_size

    while True:
        value_prev = np.copy(value_matrix)
        delta = 0

        for i in range(num):
            for j in range(num):
                for d in range(len(Directions)):  # Iterate over all directions
                    if i == 0 or j == 0 or i == num-1 or j == num-1:
                        value_matrix[i, j, d] = -100  # Penalty for moving into boundary
                        continue

                    max_value = float('-inf')

                    for action in range(len(Actions)):
                        env.robot_position = np.array([i, j])
                        env.robot_direction = d

                        obs, reward, _, _, _ = env.step(Actions(action))
                        next_pos = obs['robot_position']
                        next_dir = obs['robot_direction']

                        if (next_pos[1] == num-2 and next_pos[0] == num-2):
                            next_value = 0  # Target cell
                            reward = 0  # Target cell
                        elif 0 < next_pos[0] < num-1 and 0 < next_pos[1] < num-1:
                            if env.maze.get_cell_item(*next_pos) is not None:
                                next_value = -100  # Obstacle penalty
                            else:
                                next_value = value_prev[next_pos[0], next_pos[1], next_dir]
                        else:
                            next_value = -100  # Penalty for moving into boundary

                        action_value = reward + gamma * next_value
                        max_value = max(max_value, action_value)  # Take the maximum over all actions

                    delta = max(delta, abs(value_matrix[i, j, d] - max_value))
                    value_matrix[i, j, d] = max_value  # Update value function

        diff = np.mean((value_prev - value_matrix) ** 2)  # Check convergence
        print("diff:", diff)
        if diff < eps:
            break

    return value_matrix

def get_q_matrix(env, value_matrix):
    num = env.grid_size
    q_matrix = np.full((num, num, len(Directions), len(Actions)), -100)

    for i in range(num):
        for j in range(num):
            for d in range(len(Directions)):
                if i == 0 or j == 0 or i == num-1 or j == num-1:
                    next_value = -100  # Penalty for moving into boundary
                    reward = -100  # Penalty for moving into boundary
                    q_matrix[i, j, d, :] += reward + gamma * next_value
                    continue  # Skip border cells

                for action in range(len(Actions)):
                    env.robot_position = np.array([i, j])
                    env.robot_direction = d

                    obs, reward, _, _, _ = env.step(Actions(action))
                    next_pos = obs['robot_position']
                    next_dir = obs['robot_direction']

                    # if (next_pos[1] == 1 and next_pos[0] == 1) or (next_pos[1] == num-2 and next_pos[0] == num-2):
                    if (next_pos[1] == num-2 and next_pos[0] == num-2):
                        next_value = 0  # Target cell
                        reward = 0  # Target cell
                    elif 0 < next_pos[0] < num-1 and 0 < next_pos[1] < num-1:
                        # Check if the next position is an obstacle
                        if env.maze.get_cell_item(*next_pos) is not None:
                            next_value = -100  # Obstacle penalty

                        next_value = value_matrix[next_pos[0], next_pos[1], next_dir]
                    else:
                        next_value = -100  # Penalty for moving into a border

                    print(f"Position ({i}, {j}, {d}, {action}) - next_value: {next_value}")
                    print(f"Position ({i}, {j}, {d}, {action}) - reward: {reward}")
                    q_matrix[i, j, d, action] = reward + gamma * next_value

    print("q_matrix at 1 1 south: ", q_matrix[1, 1, Directions.south, :])
    return q_matrix

def get_optimal(q_matrix):
    num_x, num_y, num_directions, num_actions = q_matrix.shape
    policy = np.zeros((num_x, num_y, num_directions, num_actions))

    for i in range(num_x):
        for j in range(num_y):
            for d in range(num_directions):
                if i == 0 or j == 0 or i == num_x-1 or j == num_y-1:
                    continue  # Skip border cells

                best_action = np.argmax(q_matrix[i, j, d])
                policy[i, j, d, best_action] = 1
                policy[i, j, d, best_action] = policy[i, j, d, best_action]/np.sum(policy[i, j, d, best_action])

    print("policy at 1 1 south: ", policy[1, 1, Directions.south, :])

    return policy

def policy_iteration(env, eps):
    num = env.grid_size
    policy = np.ones((num, num, len(Directions), len(Actions))) / len(Actions)  # Uniform policy
    
    count = 0
    while True:
        value_matrix = evaluate_policy(env, policy, eps)
        q_matrix = get_q_matrix(env, value_matrix)
        new_policy = get_optimal(q_matrix)

        # if np.array_equal(new_policy, policy):
        #     print("Converged after {} iterations".format(count))
        #     break

        diff = np.mean((new_policy-policy)**2) ## calculate the difference
        print("diff: ", diff)
        if diff < eps: ## if less then the thresold, break
            break

        policy = new_policy
        count += 1

    return policy, value_matrix

def value_iteration(env, eps):
    num = env.grid_size
    value_matrix = np.zeros((num, num, len(Directions)))  # 3D matrix: (x, y, direction)
    while True:
        # value_prev = np.copy(value_matrix)
        new_value_matrix = update_value_function(env, gamma, eps, value_matrix)
        q_matrix = get_q_matrix(env, value_matrix)
        policy = get_optimal(q_matrix)

        diff = np.mean((value_matrix - new_value_matrix) ** 2)  # Check convergence
        print("diff:", diff)    
        if diff < eps:
            break

        value_matrix = new_value_matrix
        # policy = new_policy

    return policy, value_matrix

def rollout(env, policy, initial_state, max_steps=100):
    trajectory = []
    obs = initial_state

    for _ in range(max_steps):
        state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])
        # state = (1, 1, Directions.south)  # Assuming south is the initial direction
        print("state: ", state)
        
        action_probabilities = policy[state]
        print("action_probabilities: ", action_probabilities)

        action = np.random.choice(len(Actions), p=action_probabilities)

        obs, reward, terminated, _, _ = env.step(Actions(action))
        trajectory.append((state, action, reward))

        if terminated:
            break

    return trajectory

if __name__ == "__main__":
    grid_size = 8
    
    env = DungeonMazeEnv(grid_size=grid_size)
    # env = DungeonMazeEnv(grid_size=grid_size, render_mode="human")
    
    # RESET THE ENVIRONMENT FOR THE FIRST TIME
    # fixed seed for maze generation
    # obs, maze, _ = env.reset(seed=124)

    # random seed for maze generation
    obs, maze, _ = env.reset()
    # print("maze: ", maze)
    # END OF RESET
    
    # policy iteration
    # optimal_policy, value_matrix = policy_iteration(env, eps)

    # value iteration
    optimal_policy, value_matrix = value_iteration(env, eps)

    env = DungeonMazeEnv(grid_size=grid_size, render_mode="human")

    # RESET THE ENVIRONMENT FOR THE SECOND TIME
    # adapt the policy to the new environment
    obs, _ = env.new_reset(maze)

    # reset the environment with the same seed to get the same maze
    # obs, maze, _ = env.reset(seed=124)
    # END OF RESET
    
    traj = rollout(env, optimal_policy, obs)

    for step in traj:
        print(step)
