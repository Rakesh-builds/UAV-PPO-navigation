
from environment.uav_env import UAVEnv

env = UAVEnv()

number_of_episodes = 5

for episode in range(number_of_episodes):

    observation, info = env.reset()

    total_reward = 0
    steps = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Generate a random action
        action = env.action_space.sample()

        # Apply action to the environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    print(f"\nEpisode: {episode + 1}")
    print("Steps:", steps)
    print("Total reward:", round(total_reward, 3))
    print("Final observation:", observation)
    print("Final distance:", info["distance"])
    print("Terminated:", terminated)
    print("Truncated:", truncated)

env.close()
