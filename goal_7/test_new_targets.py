
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from goal_2.uav_env import UAVEnv


# Load trained PPO model
model = PPO.load("goal_7/uav_ppo_100k")

# Unseen target positions
targets = [
    (10.0, -10.0),
    (-10.0, 10.0),
    (-10.0, -10.0),
    (15.0, 5.0),
    (5.0, 15.0)
]

env = UAVEnv()

success_count = 0

# Store trajectories and distances for plotting
all_trajectories = []
all_distances = []

print("\n========== EXPERIMENT B: NEW TARGETS ==========")

for index, (target_x, target_y) in enumerate(targets):

    observation, info = env.reset()

    # Set new target
    env.target_x = target_x
    env.target_y = target_y

    observation = env.get_observation()

    terminated = False
    truncated = False

    episode_reward = 0.0
    steps = 0

    # Record UAV positions
    trajectory_x = [env.uav.x]
    trajectory_y = [env.uav.y]

    # Record distance from target
    distances = [env.calculate_distance()]

    while not (terminated or truncated):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

        # Save current position
        trajectory_x.append(env.uav.x)
        trajectory_y.append(env.uav.y)

        # Save current distance
        distances.append(info["distance"])

    success = info["success"]

    if success:
        success_count += 1

    # Save this episode's data
    all_trajectories.append({
        "x": trajectory_x,
        "y": trajectory_y,
        "target": (target_x, target_y),
        "success": success
    })

    all_distances.append(distances)

    print(f"\nTest {index + 1}")
    print("Target:", (target_x, target_y))
    print("Steps:", steps)
    print("Success:", success)
    print("Final distance:", round(info["distance"], 3), "m")
    print("Final velocity:", round(info["velocity"], 3), "m/s")
    print("Episode reward:", round(episode_reward, 2))


# Print final results
print("\n========== FINAL RESULTS ==========")
print("Successful targets:", success_count, "/", len(targets))
print(
    "Generalization success rate:",
    success_count / len(targets) * 100,
    "%"
)


# =====================================
# GRAPH 1: UAV TRAJECTORIES
# =====================================

plt.figure(figsize=(10, 8))

for i, trajectory in enumerate(all_trajectories):

    plt.plot(
        trajectory["x"],
        trajectory["y"],
        label=f"Target {i + 1}",
        linewidth=2
    )

    target_x, target_y = trajectory["target"]

    # Mark target position
    plt.scatter(
        target_x,
        target_y,
        marker="*",
        s=180
    )

# Mark starting position
plt.scatter(
    0,
    0,
    color="black",
    marker="o",
    s=100,
    label="Starting Position"
)

plt.title("PPO UAV Trajectories for Unseen Targets")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

plt.axhline(0, linewidth=0.7)
plt.axvline(0, linewidth=0.7)

plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.savefig("goal_7/experiment_b_trajectories.png", dpi=300)


# =====================================
# GRAPH 2: DISTANCE VS TIME
# =====================================

plt.figure(figsize=(10, 6))

for i, distances in enumerate(all_distances):

    time = [step * env.uav.dt for step in range(len(distances))]

    plt.plot(
        time,
        distances,
        label=f"Target {i + 1}",
        linewidth=2
    )

plt.title("Distance to Target vs Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Distance (meters)")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("goal_7/experiment_b_distance.png", dpi=300)

plt.show()

env.close()

print("\nGraphs saved successfully!")
print("1. goal_7/experiment_b_trajectories.png")
print("2. goal_7/experiment_b_distance.png")
