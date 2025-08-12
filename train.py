# train.py
from stable_baselines3 import PPO
from ran_environment import RANEnvironment

# 1. Instantiate the environment
env = RANEnvironment(max_steps=100)
obs, info = env.reset()

# 2. Instantiate the agent/model
# The 'MlpPolicy' is a standard neural network policy
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the model!
# This will run thousands of simulated episodes.
model.learn(total_timesteps=50000)

# 4. Save the trained agent
model.save("ran_agent_ppo")

print("Training complete and model saved.")