
from environment.uav_env import UAVEnv


def run_test(test_name, x, y):

    env = UAVEnv()
    env.reset()

    # Deliberately place UAV outside the permitted region
    env.uav.x = x
    env.uav.y = y

    # Zero acceleration
    action = [0.0, 0.0]

    observation, reward, terminated, truncated, info = env.step(action)

    print("\n", test_name)
    print("Position:", (env.uav.x, env.uav.y))
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()


# Test 1: Positive X boundary violation
run_test("Test 1: Positive X", 50.1, 10)

# Test 2: Negative Y boundary violation
run_test("Test 2: Negative Y", 10, -50.1)
