
from gymnasium.utils.env_checker import check_env

from environment.uav_env import UAVEnv

env = UAVEnv()

check_env(env)

print("Gymnasium environment checker passed!")

env.close()
