import numpy as np
import random
import matplotlib.pyplot as plt
from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions, Directions
import time

gamma = 0.99  # Discount factor
alpha = 1.0  # Learning rate
epsilon = 0.1  # Exploration rate
num_episodes = 300  # Number of episodes for training. everything larger than 260 is working
max_steps = 100  # Maximum steps per episode
grid_size = 10

# Use Q-learning as temporal difference learning method
def temporal_difference_learning(env, initial_state, maze):
    Q = np.zeros((grid_size, grid_size, len(Directions), len(Actions)))  # Q-values
    q_learning_rewards = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs, _, _ = env.reset(seed=148)  # Reset the environment
        # obs, _ = env.new_reset(maze)  # Reset the environment
        obs = initial_state
        state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])
        cumulated_reward = 0
        done = False
        
        while not done:
            for _ in range(max_steps):
                # Choose action using epsilon-greedy
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(range(len(Actions)))
                else:
                    action = np.argmax(Q[state])

                # # Choose action iteratively through all actions
                # action = (_ % len(Actions))

                obs, reward, terminated, _, _ = env.step(Actions(action))
                done = terminated
                next_state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])
                
                # TD update rule
                best_next_action = np.argmax(Q[next_state])
                Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

                state = next_state
                cumulated_reward += reward
                
                if terminated:
                    break

        q_learning_rewards.append(cumulated_reward)
    
    return q_learning_rewards, Q

def rollout(env, q_table, initial_state, max_steps=100):
    obs = initial_state
    trajectory = []
    
    for _ in range(max_steps):
        state = (obs['robot_position'][0], obs['robot_position'][1], obs['robot_direction'])

        # action_probabilities = q_table[state]
        # print("action_probabilities: ", action_probabilities)

        # action = np.random.choice(len(Actions), p=action_probabilities)

        action = np.argmax(q_table[state])  # Select best action
        
        obs, reward, terminated, _, _ = env.step(Actions(action))
        trajectory.append((state, action, reward))
        
        if terminated:
            break
    
    return trajectory

def hyperparameter_search(env, alpha_values, gamma_values, epsilon_values, num_episodes=5000):
    results = []
    
    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                _, rewards = temporal_difference_learning(env, num_episodes, alpha, gamma, epsilon)
                avg_reward = np.mean(rewards[-100:])  # Average over last 100 episodes
                results.append((alpha, gamma, epsilon, avg_reward))
                print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon} -> Avg Reward: {avg_reward}")
    
    return results

def plot_hyperparameter_tuning(results):
    alphas, gammas, epsilons, avg_rewards = zip(*results)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(alphas, avg_rewards)
    plt.xlabel('Alpha')
    plt.ylabel('Avg Reward')
    plt.title('Alpha vs Avg Reward')
    
    plt.subplot(1, 3, 2)
    plt.scatter(gammas, avg_rewards)
    plt.xlabel('Gamma')
    plt.ylabel('Avg Reward')
    plt.title('Gamma vs Avg Reward')
    
    plt.subplot(1, 3, 3)
    plt.scatter(epsilons, avg_rewards)
    plt.xlabel('Epsilon')
    plt.ylabel('Avg Reward')
    plt.title('Epsilon vs Avg Reward')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the environment
    env = DungeonMazeEnv(grid_size=grid_size)
    # random maze generation
    # obs, maze, _ = env.reset()

    # fixed map generation
    obs, maze, _ = env.reset(seed=148)  # Reset the environment
    start_time = time.perf_counter()
    Q_reward, Q_values = temporal_difference_learning(env, obs, maze)
    elapsed_time = time.perf_counter() - start_time

    # perform a rollout using the learned Q-values with the same maze above
    env = DungeonMazeEnv(grid_size=grid_size, render_mode="human")
    # obs, _ = env.new_reset(maze)
    # fixed map generation
    obs, _, _ = env.reset(seed=148)  # Reset the environment
    traj = rollout(env, Q_values, obs)
    
    for step in traj:
        print(step)

    print(f"Elapsed time: {elapsed_time:.2f} seconds")