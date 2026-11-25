
from stable_baselines3 import PPO

from goal_2.uav_env import UAVEnv


# 1. Create the UAV environment
env = UAVEnv()


# 2. Create the PPO agent
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
    verbose=1
)


# 3. Train the PPO agent
model.learn(total_timesteps=20000)


# 4. Save the trained model
model.save("goal_4/uav_ppo_model")


# 5. Close the environment
env.close()

print("PPO training completed!")
print("Model saved successfully!")
