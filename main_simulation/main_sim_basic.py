import numpy as np
from sim_env import UAVSimEnv
from cloud_controller import compute_controls

NUM_UAVS = 5

def main():
    env = UAVSimEnv(num_uavs=NUM_UAVS, gui=True)

    try:
        for t in range(10000):

            # 1) Get drone states
            states = env.get_states()

            # 2) Get obstacle positions
            obstacles = env.get_obstacles()

            # 3) Cloud controller returns forces for each drone
            controls = compute_controls(states, obstacles)

            # 4) Apply the controls in the physics engine
            env.step(controls)

    finally:
        env.close()


if __name__ == "__main__":
    main()
