
from environment.uav_env import UAVEnv

env = UAVEnv()

observation, info = env.reset()

print("Initial observation:", observation)
print("Initial info:", info)

for i in range(10):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print("Step:", i + 1)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)

    if terminated or truncated:
        break

env.close()
