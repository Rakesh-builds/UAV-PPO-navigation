
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO

from environment.uav_env import UAVEnv


# --------------------------------------------------
# 1. Load the trained PPO model
# --------------------------------------------------

model = PPO.load("training/uav_ppo_model")

# --------------------------------------------------
# 2. Create and reset the environment
# --------------------------------------------------

env = UAVEnv()
obs, info = env.reset()

# --------------------------------------------------
# 3. Store UAV flight data
# --------------------------------------------------

positions = [(env.uav.x, env.uav.y)]
distances = [env.calculate_distance()]
# Store reward at every timestep
rewards = []

velocities = [
    np.sqrt(env.uav.vx**2 + env.uav.vy**2)
]

dt = 0.1

# --------------------------------------------------
# 4. Run the trained PPO policy
# --------------------------------------------------

for step in range(env.max_steps):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    # Save the reward received at this timestep
    rewards.append(reward)

    positions.append((env.uav.x, env.uav.y))

    distances.append(env.calculate_distance())

    speed = np.sqrt(env.uav.vx**2 + env.uav.vy**2)
    velocities.append(speed)

    if terminated or truncated:
        break

env.close()

positions = np.array(positions)
distances = np.array(distances)
velocities = np.array(velocities)
rewards = np.array(rewards)

# Running total of rewards
cumulative_rewards = np.cumsum(rewards)

# --------------------------------------------------
# 5. Target position
# --------------------------------------------------

target_x = 10.0
target_y = 10.0

# --------------------------------------------------
# 6. Create animation figure
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))

ax.set_title("PPO UAV Target Navigation")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

ax.grid(True)
ax.set_aspect("equal", adjustable="box")

padding = 1.5

ax.set_xlim(
    min(positions[:, 0].min(), target_x) - padding,
    max(positions[:, 0].max(), target_x) + padding
)

ax.set_ylim(
    min(positions[:, 1].min(), target_y) - padding,
    max(positions[:, 1].max(), target_y) + padding
)

# --------------------------------------------------
# 7. Draw target
# --------------------------------------------------

ax.scatter(
    target_x,
    target_y,
    color="red",
    marker="*",
    s=250,
    label="Target"
)

# --------------------------------------------------
# 8. Create trajectory line
# --------------------------------------------------

trajectory_line, = ax.plot(
    [],
    [],
    color="blue",
    linewidth=2,
    label="UAV trajectory"
)

# --------------------------------------------------
# 9. Create moving UAV marker
# --------------------------------------------------

uav_marker, = ax.plot(
    [],
    [],
    marker="o",
    color="blue",
    markersize=10,
    label="UAV"
)

status_text = ax.text(
    0.02,
    0.96,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

ax.legend(loc="lower right")

# --------------------------------------------------
# 10. Initialize animation
# --------------------------------------------------

def init():

    trajectory_line.set_data([], [])
    uav_marker.set_data([], [])
    status_text.set_text("")

    return trajectory_line, uav_marker, status_text

# --------------------------------------------------
# 11. Update each animation frame
# --------------------------------------------------

def update(frame):

    trajectory_line.set_data(
        positions[:frame + 1, 0],
        positions[:frame + 1, 1]
    )

    uav_marker.set_data(
        [positions[frame, 0]],
        [positions[frame, 1]]
    )

    elapsed_time = frame * dt

    status_text.set_text(
        f"Time: {elapsed_time:.1f} s\n"
        f"Distance: {distances[frame]:.3f} m\n"
        f"Speed: {velocities[frame]:.3f} m/s\n"
        f"Step: {frame}"
    )

    return trajectory_line, uav_marker, status_text

# --------------------------------------------------
# 12. Generate animation
# --------------------------------------------------

animation = FuncAnimation(
    fig,
    update,
    frames=len(positions),
    init_func=init,
    interval=100,
    blit=True,
    repeat=True
)

# --------------------------------------------------
# 13. Save GIF
# --------------------------------------------------

animation.save(
    "visualization/ppo_flight.gif",
    writer=PillowWriter(fps=10)
)

print("\nGIF successfully saved!")
print(f"Total frames: {len(positions)}")
print(f"Final distance: {distances[-1]:.3f} m")
print(f"Final speed: {velocities[-1]:.3f} m/s")


# --------------------------------------------------
# 14. Plot reward performance
# --------------------------------------------------

time_steps = np.arange(1, len(rewards) + 1) * dt

fig_reward, axes = plt.subplots(2, 1, figsize=(10, 8))

# Graph 1: Reward per timestep
axes[0].plot(
    time_steps,
    rewards,
    color="blue",
    linewidth=1.5,
    label="Step reward"
)

axes[0].axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=0.8
)

axes[0].set_title("Reward per Timestep")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Reward")
axes[0].grid(True)
axes[0].legend()

# Graph 2: Cumulative reward
axes[1].plot(
    time_steps,
    cumulative_rewards,
    color="green",
    linewidth=2,
    label="Cumulative reward"
)

axes[1].set_title("Cumulative Reward During UAV Flight")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Total reward")
axes[1].grid(True)
axes[1].legend()

fig_reward.tight_layout()

# Save reward graphs
fig_reward.savefig(
    "visualization/ppo_reward_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

print("Reward graph saved: visualization/ppo_reward_analysis.png")
print(f"Total episode reward: {np.sum(rewards):.3f}")

plt.show()

