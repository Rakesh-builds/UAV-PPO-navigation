
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from goal_2.uav_env import UAVEnv


# Load the existing 100K PPO model
model = PPO.load("goal_7/uav_ppo_100k")

# Simulation settings
WIND_ACCELERATION = 0.5
WIND_START_STEP = 20
WIND_END_STEP = 40

TARGET_X = 10.0
TARGET_Y = 10.0


def run_episode(disturbance=False):

    env = UAVEnv()

    observation, info = env.reset()

    env.target_x = TARGET_X
    env.target_y = TARGET_Y

    observation = env.get_observation()

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    # Store trajectory
    x_positions = [env.uav.x]
    y_positions = [env.uav.y]

    # Store distance from target
    distances = [env.calculate_distance()]

    while not (terminated or truncated):

        # PPO selects an action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += reward

        # Apply external wind acceleration
        if (
            disturbance
            and WIND_START_STEP <= steps < WIND_END_STEP
            and not (terminated or truncated)
        ):

            dt = env.uav.dt

            # Wind changes horizontal velocity
            env.uav.vx = np.clip(
                env.uav.vx + WIND_ACCELERATION * dt,
                -5.0,
                5.0
            )

            # Update position using the new velocity
            env.uav.x += env.uav.vx * dt

            # Update observation after disturbance
            observation = env.get_observation()

            # Update information for logging
            info["distance"] = float(env.calculate_distance())

            info["velocity"] = float(
                np.sqrt(env.uav.vx**2 + env.uav.vy**2)
            )

        # Record state after the disturbance
        x_positions.append(env.uav.x)
        y_positions.append(env.uav.y)

        distances.append(env.calculate_distance())

    result = {
        "success": info["success"],
        "steps": steps,
        "reward": total_reward,
        "distance": info["distance"],
        "velocity": info["velocity"],
        "x": x_positions,
        "y": y_positions,
        "distances": distances
    }

    env.close()

    return result


# =====================================
# RUN BOTH CONDITIONS
# =====================================

print("\n========== NOMINAL TEST ==========")

nominal = run_episode(disturbance=False)

print("Success:", nominal["success"])
print("Steps:", nominal["steps"])
print("Final distance:", round(nominal["distance"], 3), "m")
print("Final velocity:", round(nominal["velocity"], 3), "m/s")
print("Reward:", round(nominal["reward"], 2))


print("\n========== DISTURBED TEST ==========")

disturbed = run_episode(disturbance=True)

print("Success:", disturbed["success"])
print("Steps:", disturbed["steps"])
print("Final distance:", round(disturbed["distance"], 3), "m")
print("Final velocity:", round(disturbed["velocity"], 3), "m/s")
print("Reward:", round(disturbed["reward"], 2))


# =====================================
# GRAPH 1: TRAJECTORY COMPARISON
# =====================================

plt.figure(figsize=(9, 7))

plt.plot(
    nominal["x"],
    nominal["y"],
    label="Nominal",
    linewidth=2
)

plt.plot(
    disturbed["x"],
    disturbed["y"],
    label="With Wind Disturbance",
    linewidth=2
)

plt.scatter(0, 0, color="black", marker="o", s=100, label="Start")

plt.scatter(
    TARGET_X,
    TARGET_Y,
    color="red",
    marker="*",
    s=200,
    label="Target"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("PPO UAV: Nominal vs Disturbed Trajectory")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("goal_7/experiment_c_trajectory.png", dpi=300)


# =====================================
# GRAPH 2: DISTANCE COMPARISON
# =====================================

plt.figure(figsize=(9, 6))

nominal_time = [
    i * 0.1 for i in range(len(nominal["distances"]))
]

disturbed_time = [
    i * 0.1 for i in range(len(disturbed["distances"]))
]

plt.plot(
    nominal_time,
    nominal["distances"],
    label="Nominal",
    linewidth=2
)

plt.plot(
    disturbed_time,
    disturbed["distances"],
    label="With Wind Disturbance",
    linewidth=2
)

plt.axvspan(
    WIND_START_STEP * 0.1,
    WIND_END_STEP * 0.1,
    alpha=0.2,
    label="Wind Disturbance Period"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Distance to Target (m)")
plt.title("Distance Comparison: Nominal vs Disturbed")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("goal_7/experiment_c_distance.png", dpi=300)

plt.show()

print("\nExperiment C completed!")
print("Graphs saved in goal_7/")
