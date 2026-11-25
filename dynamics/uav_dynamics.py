
import numpy as np
import matplotlib.pyplot as plt


class UAV:

    def __init__(self):

        # Initial position
        self.x = 0.0
        self.y = 0.0

        # Initial velocity
        self.vx = 0.0
        self.vy = 0.0

        # Simulation parameters
        self.dt = 0.1

        # Physical/control limits
        self.max_acceleration = 2.0
        self.max_velocity = 5.0

        # Store trajectory
        self.trajectory = []

    def get_state(self):

        return np.array([
            self.x,
            self.y,
            self.vx,
            self.vy
        ])

    def step(self, ax, ay):

        # 1. Limit acceleration commands
        ax = np.clip(
            ax,
            -self.max_acceleration,
            self.max_acceleration
        )

        ay = np.clip(
            ay,
            -self.max_acceleration,
            self.max_acceleration
        )

        # 2. Update velocity
        self.vx += ax * self.dt
        self.vy += ay * self.dt

        # 3. Limit velocity
        self.vx = np.clip(
            self.vx,
            -self.max_velocity,
            self.max_velocity
        )

        self.vy = np.clip(
            self.vy,
            -self.max_velocity,
            self.max_velocity
        )

        # 4. Update position using new velocity
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # 5. Store current position
        self.trajectory.append([self.x, self.y])


# Run the Goal 1 demonstration only when this file
# is executed directly, not when another file imports it.

if __name__ == "__main__":

    # Create UAV
    uav = UAV()

    # Initial position
    uav.trajectory.append([uav.x, uav.y])

    # Manually supplied acceleration commands
    commands = []

    # Accelerate right
    for i in range(20):
        commands.append((2.0, 0.0))

    # Accelerate upward
    for i in range(20):
        commands.append((0.0, 2.0))

    # Accelerate left
    for i in range(20):
        commands.append((-2.0, 0.0))

    # Accelerate downward
    for i in range(20):
        commands.append((0.0, -2.0))

    # Execute simulation
    for ax, ay in commands:

        uav.step(ax, ay)

        print("State:", uav.get_state())

    # Convert trajectory into NumPy array
    trajectory = np.array(uav.trajectory)

    # Plot trajectory
    plt.figure(figsize=(9, 7))

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label="UAV trajectory"
    )

    plt.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        color="green",
        label="Start"
    )

    plt.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        color="red",
        label="Final position"
    )

    plt.xlabel("X Position (metres)")
    plt.ylabel("Y Position (metres)")

    plt.title("2D UAV Movement Simulation")

    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    plt.show()
