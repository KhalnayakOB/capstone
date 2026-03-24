import numpy as np
import pandas as pd
from envs.sim_env_city_v2 import UAVCityEnvV2

DATASET_SIZE = 1500


def get_nearest_obstacle(px, py, obstacles):
    min_dist = float("inf")
    closest = (0, 0)

    for ox, oy, r in obstacles:
        d = np.sqrt((px - ox)**2 + (py - oy)**2) - r
        if d < min_dist:
            min_dist = d
            closest = (ox, oy)

    return min_dist, closest


def main():
    env = UAVCityEnvV2(gui=False)
    data = []

    for run in range(DATASET_SIZE):
        print(f"Run {run+1}/{DATASET_SIZE}")

        # MULTI-GOAL RANDOMIZATION
        start = env.get_drone_start(0)
        goal = np.array([
            np.random.uniform(20, 25),
            np.random.uniform(-20, 20),
            2
        ])

        obstacles = env.get_obstacles_2d()

        current = start[:2]
        goal_2d = goal[:2]

        for _ in range(60):

            px, py = current

            # NOISE (ROBUSTNESS)
            px += np.random.normal(0, 0.1)
            py += np.random.normal(0, 0.1)

            direction = goal_2d - np.array([px, py])
            dist = np.linalg.norm(direction)

            if dist < 0.5:
                break

            direction = direction / (dist + 1e-6)

            next_point = np.array([px, py]) + direction * 0.5

            obs_dist, (ox, oy) = get_nearest_obstacle(px, py, obstacles)

            obs_dx = px - ox
            obs_dy = py - oy

            data.append([
                px, py,
                goal_2d[0], goal_2d[1],
                dist,
                obs_dist,
                obs_dx, obs_dy,
                direction[0], direction[1]
            ])

            current = next_point

    df = pd.DataFrame(data, columns=[
        "px","py","gx","gy",
        "dist","obs_dist",
        "obs_dx","obs_dy",
        "dx","dy"
    ])

    df.to_csv("dataset_rrt.csv", index=False)
    print("✅ Dataset saved")


if __name__ == "__main__":
    main()