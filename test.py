# test.py

from stable_baselines3 import PPO
from ran_environment import RANEnvironment
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 1. Instantiate the environment
    env = RANEnvironment(max_steps=100)

    # 2. Load the trained agent
    model = PPO.load("ran_agent_ppo")

    print("Testing script started. Running simulation with trained agent...")

    # Run the simulation for a few episodes
    num_episodes = 5
    episode_durations = []
    total_power_history = []
    users_connected_history = []
    users_dropped_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_power = []
        episode_users_connected = []
        episode_users_dropped = []
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_power.append(info["total_power_consumption"])
            episode_users_connected.append(info["users_connected"])
            episode_users_dropped.append(info["users_dropped"])
            step_count += 1
        
        episode_durations.append(step_count)
        total_power_history.append(episode_power)
        users_connected_history.append(episode_users_connected)
        users_dropped_history.append(episode_users_dropped)

        print(f"Episode {episode + 1} finished after {step_count} steps.")

    print("Simulation complete. Plotting results...")

    # Plotting results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for i, power_data in enumerate(total_power_history):
        plt.plot(power_data, label=f'Episode {i+1}')
    plt.title('Total Power Consumption Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Power (Watts)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    for i, connected_data in enumerate(users_connected_history):
        plt.plot(connected_data, label=f'Episode {i+1}')
    plt.title('Users Connected Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for i, dropped_data in enumerate(users_dropped_history):
        plt.plot(dropped_data, label=f'Episode {i+1}')
    plt.title('Users Dropped Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("simulation_results.png")
    print("Results saved to simulation_results.png")

    # Optional: Display the plot (uncomment if running in an environment with GUI)
    # plt.show()