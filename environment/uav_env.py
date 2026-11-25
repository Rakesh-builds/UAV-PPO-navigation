
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamics.uav_dynamics import UAV


class UAVEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.uav = UAV()

        self.target_x = 10.0
        self.target_y = 10.0

        self.max_acceleration = 2.0
        self.max_steps = 500

        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(4,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    def get_observation(self):

        dx = self.target_x - self.uav.x
        dy = self.target_y - self.uav.y

        return np.array([
            dx,
            dy,
            self.uav.vx,
            self.uav.vy
        ], dtype=np.float32)

    def calculate_distance(self):

        dx = self.target_x - self.uav.x
        dy = self.target_y - self.uav.y

        return np.sqrt(dx**2 + dy**2)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.uav = UAV()

        self.uav.x = 0.0
        self.uav.y = 0.0
        self.uav.vx = 0.0
        self.uav.vy = 0.0

        self.target_x = 10.0
        self.target_y = 10.0

        self.current_step = 0

        observation = self.get_observation()

        info = {
            "distance":float(self.calculate_distance())
        }

        return observation, info

    
    def step(self, action):

        action = np.clip(action, -1.0, 1.0)

        ax = action[0] * self.max_acceleration
        ay = action[1] * self.max_acceleration

        previous_distance = self.calculate_distance()

        self.uav.step(ax, ay)

        current_distance = self.calculate_distance()

        reward = previous_distance - current_distance - 0.01

        velocity = np.sqrt(
            self.uav.vx**2 + self.uav.vy**2
        )

        success = (
            current_distance < 0.5
            and velocity < 0.5
        )

        terminated = False
        truncated = False

        if success:
            reward += 100
            terminated = True

        if (
            abs(self.uav.x) > 50
            or abs(self.uav.y) > 50
        ):
            reward -= 50
            terminated = True

        self.current_step += 1

        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        observation = self.get_observation()

        info = {
            "distance": float(current_distance),
            "success": bool(success),
            "velocity": float(velocity)
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )
