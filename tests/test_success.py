
from environment.uav_env import UAVEnv

env = UAVEnv()

observation, info = env.reset()

# Place UAV close to the target
env.uav.x = 9.8
env.uav.y = 9.8

# Give it a very small velocity
env.uav.vx = 0.1
env.uav.vy = 0.1

# Take one action with zero acceleration
action = [0.0, 0.0]

observation, reward, terminated, truncated, info = env.step(action)

print("Observation:", observation)
print("Reward:", reward)
print("Distance:", info["distance"])
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()
