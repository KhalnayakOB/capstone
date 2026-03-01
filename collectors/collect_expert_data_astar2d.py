# collect_expert_data.py

import numpy as np
import pybullet as p

from sim_env_city_v2 import UAVCityEnvV2
from cloud_controller import compute_path_astar, compute_straight_path


def build_features(pos, vel, goal):
    """
    Build a 6D feature vector:
    [goal_x - x, goal_y - y, goal_z - z, vx, vy, vz]
    """
    rel = goal - pos
    return np.concatenate([rel, vel])


def run_episode(env, speed=8.0, grid_res=1.0, safety_margin=0.6,
                max_steps=3000):
    """
    Run ONE episode using the cloud controller (A* path) and log data.

    Returns:
        X: (N, 6) feature array
        Y: (N, 3) expert actions (vx, vy, vz)
    """
    data_X = []
    data_Y = []

    state = env.get_drone_state()
    pos = state["pos"]
    vel = state["vel"]
    goal = env.end_point.copy()

    # ---- PLAN WITH A* (2D) ----
    bounds = ((-env.pad_half_x, env.pad_half_x),
              (-env.pad_half_y, env.pad_half_y))
    obstacles = env.get_obstacles_2d()

    path_2d = compute_path_astar(
        start=pos,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        grid_res=grid_res,
        safety_margin=safety_margin,
    )

    # Fallback: straight-line (2D) if A* fails
    if not path_2d:
        path_2d = compute_straight_path(
            start=pos,
            goal=goal,
            step=1.0,
        )

    # Convert 2D path (x, y) to 3D waypoints (x, y, goal_z)
    path = []
    for (x, y) in path_2d:
        path.append(np.array([x, y, goal[2]], dtype=float))

    if len(path) < 2:
        return None, None

    wp_idx = 0
    t = 0
    drone_id = env.drone_id

    while t < max_steps:
        if not p.isConnected():
            break

        state = env.get_drone_state()
        pos = state["pos"]
        vel = state["vel"]

        # Stop if near goal
        if np.linalg.norm(pos - goal) < 1.0:
            break

        if wp_idx >= len(path):
            break

        target = path[wp_idx]
        delta = target - pos
        dist_to_wp = float(np.linalg.norm(delta))

        if dist_to_wp < 0.7:
            wp_idx += 1
            continue

        direction = delta / (dist_to_wp + 1e-6)
        vx_cmd = direction[0] * speed
        vy_cmd = direction[1] * speed
        vz_cmd = direction[2] * speed

        # --- log one sample ---
        x_feat = build_features(pos, vel, goal)
        y_cmd = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)
        data_X.append(x_feat)
        data_Y.append(y_cmd)

        # apply velocity and step sim
        p.resetBaseVelocity(
            drone_id,
            linearVelocity=[vx_cmd, vy_cmd, vz_cmd],
            angularVelocity=[0.0, 0.0, 0.0],
        )
        env.step()

        t += 1

    if len(data_X) == 0:
        return None, None

    return np.stack(data_X, axis=0), np.stack(data_Y, axis=0)


def main():
    # DIRECT mode = no GUI → faster data collection
    env = UAVCityEnvV2(gui=False, dt=1.0 / 240.0)

    all_X = []
    all_Y = []

    n_episodes = 30   # you can increase later for more data

    try:
        for ep in range(n_episodes):
            print(f"Episode {ep+1}/{n_episodes}")
            X, Y = run_episode(env)

            if X is not None:
                all_X.append(X)
                all_Y.append(Y)

            # new random map
            env.reset_scene()

        if not all_X:
            print("No data collected!")
            return

        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)

        print("Collected dataset:")
        print("  X shape:", X.shape)
        print("  Y shape:", Y.shape)

        np.savez("expert_data.npz", X=X, Y=Y)
        print("Saved to expert_data.npz")

    finally:
        env.close()


if __name__ == "__main__":
    main()
