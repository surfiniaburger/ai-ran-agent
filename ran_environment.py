import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy

class BaseStation:
    def __init__(self, x, y, power_consumption_watts=500, max_transmit_power=40, is_on=True):
        self.x = x
        self.y = y
        self.power_consumption_watts = power_consumption_watts
        self.max_transmit_power = max_transmit_power
        self.is_on = is_on
        self.users_connected = 0

class UserEquipment:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class RANEnvironment(gym.Env):
    """
    A simulated Radio Access Network (RAN) environment for reinforcement learning.
    """
    def __init__(self, width=10000, height=10000, num_base_stations=3, num_users=10, max_steps=100):
        super(RANEnvironment, self).__init__()

        self.width = width
        self.height = height
        self.num_base_stations = num_base_stations
        self.num_users = num_users
        self.max_steps = max_steps
        self.current_step = 0

        # Define action and observation spaces
        # Action space: for each base station, choose to be ON (1) or OFF (0)
        self.action_space = spaces.MultiDiscrete([2] * self.num_base_stations)

        # Observation space: for each base station, we observe:
        # 1. is_on (0 or 1)
        # 2. number of users connected
        self.observation_space = spaces.Box(low=0, high=self.num_users, shape=(self.num_base_stations, 2), dtype=np.float32)

        # Create base stations
        self.base_stations = self._create_base_stations()

        # Create users
        self.users = self._create_users()

        print("RANEnvironment initialized")

    def _create_base_stations(self):
        base_stations = []
        # Place base stations in a somewhat regular pattern
        for i in range(self.num_base_stations):
            x = (i + 1) * self.width / (self.num_base_stations + 1)
            y = self.height / 2
            base_stations.append(BaseStation(x, y))
        return base_stations

    def _create_users(self):
        users = []
        for _ in range(self.num_users):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            users.append(UserEquipment(x, y))
        return users

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset user positions
        self.users = self._create_users()

        # Reset base station states (optional, they could retain their on/off state)
        for bs in self.base_stations:
            bs.is_on = True
            bs.users_connected = 0
        
        self.current_step = 0 # Reset step counter
        self._update_connectivity()
        
        print("RANEnvironment reset")
        return self._get_observation(), self._get_info()

    def step(self, action):
        # 1. Apply the action: Turn base stations on or off
        for i, a in enumerate(action):
            self.base_stations[i].is_on = bool(a)

        # 2. Update the environment state
        # In a more complex model, users would move here. For now, we just recalculate connectivity.
        self._update_connectivity()

        # 3. Calculate the reward
        reward = self._calculate_reward()

        # 4. Check if the episode is terminated
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # 5. Get the new observation
        observation = self._get_observation()
        
        # 6. Get info
        info = self._get_info()

        print(f"RANEnvironment step: action={action}")
        return observation, reward, terminated, False, info

    def _update_connectivity(self):
        # Reset user counts
        for bs in self.base_stations:
            bs.users_connected = 0
        
        self.users_dropped = 0

        for user in self.users:
            strongest_signal = -np.inf
            best_bs = None
            for bs in self.base_stations:
                if bs.is_on:
                    distance = np.sqrt((user.x - bs.x)**2 + (user.y - bs.y)**2)
                    if distance == 0:
                        signal_strength = np.inf
                    else:
                        # Simple path loss model: Signal_Strength = Transmit_Power / (distance^2)
                        # We'll use a simplified version without actual dBm for now
                        signal_strength = bs.max_transmit_power / (distance**2)
                    
                    if signal_strength > strongest_signal:
                        strongest_signal = signal_strength
                        best_bs = bs
            
            # A simple threshold for connectivity
            # This needs to be tuned based on the world size and transmit power
            if strongest_signal > 1e-5: 
                best_bs.users_connected += 1
            else:
                self.users_dropped += 1

    def _calculate_reward(self):
        # Reward function from the plan:
        # - Small negative reward for power consumption
        # - Large negative reward for dropped users
        # - Small positive reward for connected users
        
        power_consumption = sum(bs.power_consumption_watts for bs in self.base_stations if bs.is_on)
        users_connected = sum(bs.users_connected for bs in self.base_stations)

        reward = 0
        reward -= power_consumption * 0.01
        reward -= self.users_dropped * 100
        reward += users_connected * 1
        for bs in self.base_stations:
            if not bs.is_on and bs.users_connected == 0:
                reward += 25 # A small bonus for being smart!
        
        return reward

    def _get_observation(self):
        obs = np.zeros((self.num_base_stations, 2), dtype=np.float32)
        for i, bs in enumerate(self.base_stations):
            obs[i, 0] = 1 if bs.is_on else 0
            obs[i, 1] = bs.users_connected
        return obs

    def _get_info(self):
        return {
            "total_power_consumption": sum(bs.power_consumption_watts for bs in self.base_stations if bs.is_on),
            "users_connected": sum(bs.users_connected for bs in self.base_stations),
            "users_dropped": self.users_dropped
        }

    def render(self, mode='human'):
        print("RANEnvironment render")
        pass

    def close(self):
        print("RANEnvironment closed")
        pass