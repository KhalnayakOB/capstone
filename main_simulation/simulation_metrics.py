import numpy as np
import pybullet as p
from envs.sim_env_city_v2 import UAVCityEnvV2

def run_test(steps=1000):

    env = UAVCityEnvV2(gui=True)

    success = 0
    collisions = 0
    runs = 10

    for r in range(runs):

        env._build_scene()
        drone = env.drones[0]
        goal = env.get_drone_goal(0)

        for step in range(steps):

            pos, _ = p.getBasePositionAndOrientation(drone)
            pos = np.array(pos)

            dist = np.linalg.norm(pos[:2] - goal[:2])

            if dist < 1.0:
                success += 1
                break

            contacts = p.getContactPoints()
            if len(contacts) > 0:
                collisions += 1

            env.step()

    print("\n📊 SIMULATION RESULTS")
    print("----------------------")
    print(f"Success Rate: {success/runs*100:.2f}%")
    print(f"Total Collisions: {collisions}")

    env.close()


if __name__ == "__main__":
    run_test()