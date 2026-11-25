
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from goal_2.uav_env import UAVEnv


# 1. Load the trained PPO model
model = PPO.load("goal_4/uav_ppo_model")

# 2. Create and reset the environment
env = UAVEnv()
observation, info = env.reset()

# Store the initial position and distance
positions = [[env.uav.x, env.uav.y]]
distances = [env.calculate_distance()]

terminated = False
truncated = False
total_reward = 0.0
steps = 0

# 3. Run the trained policy
while not (terminated or truncated):

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    observation, reward, terminated, truncated, info = env.step(action)

    # Record UAV position after each step
    positions.append([env.uav.x, env.uav.y])

    # Record distance from target
    distances.append(info["distance"])

    total_reward += reward
    steps += 1


# 4. Convert recorded data into NumPy arrays
positions = np.array(positions)
distances = np.array(distances)

dt = env.uav.dt
time = np.arange(len(distances)) * dt

start = positions[0]
final = positions[-1]

target_x = env.target_x
target_y = env.target_y


# 5. Create two plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


# Plot 1: UAV trajectory
axes[0].plot(
    positions[:, 0],
    positions[:, 1],
    label="UAV trajectory"
)

axes[0].scatter(
    start[0], start[1],
    color="green", marker="o",
    s=100, label="Start"
)

axes[0].scatter(
    target_x, target_y,
    color="red", marker="*",
    s=200, label="Target"
)

axes[0].scatter(
    final[0], final[1],
    color="blue", marker="x",
    s=100, label="Final position"
)

axes[0].set_title("PPO UAV Trajectory")
axes[0].set_xlabel("X Position (m)")
axes[0].set_ylabel("Y Position (m)")
axes[0].legend()
axes[0].grid(True)
axes[0].axis("equal")


# Plot 2: Distance from target over time
axes[1].plot(
    time,
    distances,
    label="Distance to target"
)

axes[1].set_title("Distance vs Time")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Distance (m)")
axes[1].legend()
axes[1].grid(True)


# 6. Display results
plt.tight_layout()

plt.savefig(
    "goal_6/ppo_trajectory_analysis.png",
    dpi=300
)

plt.show()

print("\n========== TRAJECTORY RESULTS ==========")
print("Total steps:", steps)
print("Total simulation time:", round(steps * dt, 2), "seconds")
print("Total reward:", round(total_reward, 2))
print("Final X position:", round(final[0], 3))
print("Final Y position:", round(final[1], 3))
print("Final distance:", round(distances[-1], 3), "m")
print("Final velocity:", round(info["velocity"], 3), "m/s")
print("Success:", info["success"])

env.close()
