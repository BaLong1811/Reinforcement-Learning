import numpy as np
import random
import matplotlib.pyplot as plt
from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions, Directions
import time

gamma = 0.99  # Discount factor
num_episodes = 1500  # Number of episodes for training
max_steps = 100  # Maximum steps per episode
grid_size = 10

def monte_carlo_on_policy(env, initial_state, maze, epsilon=0.1):
    Q = np.zeros((grid_size, grid_size, len(Directions), len(Actions)))  # Q-values
    Returns = {((x, y, d), a): [] for x in range(grid_size) for y in range(grid_size) 
                for d in range(len(Directions)) for a in range(len(Actions))}  # Returns list
    # policy = np.zeros((grid_size, grid_size, len(Directions)), dtype=int)  # Policy table 
    policy = np.full((grid_size, grid_size, len(Directions), len(Actions)), 1 / len(Actions))
    mc_rewards = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs, _, _ = env.reset(seed=144)
        obs = initial_state
        state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])
        action = random.choice(range(len(Actions)))  # Ensure all pairs have probability > 0
        episode_data = [(state, action, 0)]  # Initialize with chosen action
        cumulated_reward = 0
        done = False
        
        for _ in range(max_steps):
            obs, reward, terminated, _, _ = env.step(Actions(action))
            next_state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])
            next_action = random.choice(range(len(Actions)))
            episode_data.append((next_state, next_action, reward))
            state, action = next_state, next_action
            cumulated_reward += reward
            
            if terminated:
                break
        
        mc_rewards.append(cumulated_reward)
        G = 0  # Initialize return
        visited_state_action_pairs = set()
        
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            if (state, action) not in visited_state_action_pairs:
                Returns[(state, action)].append(G)
                Q[state][action] = np.mean(Returns[(state, action)])
                # policy[state] = np.argmax(Q[state])  # Update policy greedily

                best_action = np.argmax(Q[state])  # Greedy action
                
                # Update soft policy
                for a in range(len(Actions)):
                    if a == best_action:
                        policy[state][a] = 1 - epsilon + (epsilon / len(Actions))
                    else:
                        policy[state][a] = epsilon / len(Actions)

                # Update the state-action pair to avoid duplicates    
                visited_state_action_pairs.add((state, action))
    
    return mc_rewards, Q

def rollout(env, q_table, initial_state, max_steps=100):
    obs = initial_state
    trajectory = []
    
    for _ in range(max_steps):
        state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])
        action = np.argmax(q_table[state])  # Select best action
        
        obs, reward, terminated, _, _ = env.step(Actions(action))
        trajectory.append((state, action, reward))
        
        if terminated:
            break
    
    return trajectory

if __name__ == "__main__":
    # Initialize the environment
    env = DungeonMazeEnv(grid_size=grid_size)
    obs, maze, _ = env.reset(seed=144)
    
    start_time = time.perf_counter()
    MC_reward, Q_values = monte_carlo_on_policy(env, obs, maze)
    elapsed_time = time.perf_counter() - start_time

    env = DungeonMazeEnv(grid_size=grid_size, render_mode="human")
    obs, _, _ = env.reset(seed=144)
    traj = rollout(env, Q_values, obs)
    
    for step in traj:
        print(step)
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
