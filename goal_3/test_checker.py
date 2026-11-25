
from gymnasium.utils.env_checker import check_env

from goal_2.uav_env import UAVEnv

env = UAVEnv()

check_env(env)

print("Gymnasium environment checker passed!")

env.close()
