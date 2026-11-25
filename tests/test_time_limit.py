
from environment.uav_env import UAVEnv

env = UAVEnv()

observation, info = env.reset()

# Apply zero acceleration continuously
action = [0.0, 0.0]

for step in range(500):

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

print("Steps:", step + 1)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Final distance:", info["distance"])

env.close()
