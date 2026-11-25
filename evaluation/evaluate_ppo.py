
from stable_baselines3 import PPO
from environment.uav_env import UAVEnv


# Load trained PPO model
model = PPO.load("training/uav_ppo_model")

# Create UAV environment
env = UAVEnv()

n_episodes = 10

successes = 0
total_rewards = 0

for episode in range(n_episodes):

    observation, info = env.reset()

    terminated = False
    truncated = False

    episode_reward = 0
    steps = 0

    while not (terminated or truncated):

        # Predict action using trained policy
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

    if info["success"]:
        successes += 1

    total_rewards += episode_reward

    print(f"\nEpisode: {episode + 1}")
    print("Steps:", steps)
    print("Reward:", round(episode_reward, 2))
    print("Success:", info["success"])
    print("Final distance:", round(info["distance"], 3))
    print("Final velocity:", round(info["velocity"], 3))


print("\n========== FINAL RESULTS ==========")

print("Success rate:", successes / n_episodes * 100, "%")
print("Average reward:", total_rewards / n_episodes)

env.close()
