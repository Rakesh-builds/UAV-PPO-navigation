
from stable_baselines3 import PPO
from goal_2.uav_env import UAVEnv


# Create environment
env = UAVEnv()

# Create a fresh PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    seed=42
)

# Train for the first 20,000 requested timesteps
print("\n========== TRAINING: 20K CHECKPOINT ==========")

model.learn(total_timesteps=20_000)

# Save the 20K checkpoint
model.save("goal_7/uav_ppo_20k_comparison")

print("\n20K checkpoint saved!")


# Continue training the SAME agent for another 80,000 timesteps
print("\n========== CONTINUING TRAINING TO 100K ==========")

model.learn(
    total_timesteps=80_000,
    reset_num_timesteps=False
)

# Save the 100K checkpoint
model.save("goal_7/uav_ppo_100k")

print("\n100K checkpoint saved!")
print("Experiment A training completed!")

env.close()
