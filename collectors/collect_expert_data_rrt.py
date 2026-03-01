import numpy as np

from sim_env_city_v2 import UAVCityEnvV2   # <-- change name if your class is different
from cloud_controller import compute_path_rrt_star


def generate_expert_trajectory_from_path(path_xyz: np.ndarray,
                                         speed: float = 5.0,
                                         spatial_step: float = 1.0):
    """
    Given a 3D path (N, 3) from RRT*, generate expert state-action samples.

    X: [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z]
    Y: [vx, vy, vz]  (desired velocity along the path)

    speed: how fast the drone is assumed to move (m/s)
    spatial_step: approx distance in meters between samples.
    """
    if path_xyz.shape[0] < 2:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    goal = path_xyz[-1].copy()
    X_samples = []
    Y_samples = []

    # Start at first waypoint
    current_pos = path_xyz[0].copy()

    for i in range(path_xyz.shape[0] - 1):
        p0 = path_xyz[i]
        p1 = path_xyz[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue

        # Number of samples along this segment
        n_steps = max(int(seg_len / spatial_step), 1)
        step_vec = seg_vec / n_steps

        # Direction for desired velocity
        dir_vec = step_vec / np.linalg.norm(step_vec)
        v_des = dir_vec * speed  # constant speed along the path

        for _ in range(n_steps):
            # State features: current position + final goal position
            state_feat = np.concatenate([current_pos, goal])
            X_samples.append(state_feat)
            Y_samples.append(v_des.copy())
            current_pos = current_pos + step_vec

    X = np.array(X_samples, dtype=np.float32)
    Y = np.array(Y_samples, dtype=np.float32)
    return X, Y


def run_episode(env: UAVCityEnvV2):
    """
    One episode:
      - reset env (new city layout)
      - read obstacles and map bounds
      - compute RRT* path from start to end
      - turn path into (X, Y) expert samples
    """
    # New random layout
    env.reset_scene()

    # Get 2D circular obstacles
    obstacles = env.get_obstacles_2d()  # [(x, y, radius), ...]

    # Map bounds from the pad size (V2 env should have pad_half_x, pad_half_y)
    bounds = (
        (-env.pad_half_x, env.pad_half_x),
        (-env.pad_half_y, env.pad_half_y),
    )

    # Start / goal in XY (assumes env.start_point / env.end_point exist)
    start_xy = (float(env.start_point[0]), float(env.start_point[1]))
    goal_xy = (float(env.end_point[0]), float(env.end_point[1]))

    # Compute RRT* path in 3D (z is constant inside this function)
    path_xyz = compute_path_rrt_star(start_xy, goal_xy, obstacles, bounds)

    if path_xyz.shape[0] < 2:
        print("  [WARN] RRT* could not find a good path, skipping episode.")
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Generate state-action expert pairs along that path
    X, Y = generate_expert_trajectory_from_path(path_xyz,
                                                speed=5.0,
                                                spatial_step=1.0)
    return X, Y


def main():
    # You can change these
    num_episodes = 50
    save_path = "expert_data_rrt.npz"

    # Create environment in DIRECT mode (no GUI, faster)
    env = UAVCityEnvV2(gui=False, dt=1.0 / 240.0)

    all_X = []
    all_Y = []

    for ep in range(1, num_episodes + 1):
        print(f"Episode {ep}/{num_episodes}")
        X, Y = run_episode(env)
        if X.shape[0] == 0:
            continue
        all_X.append(X)
        all_Y.append(Y)

    if not all_X:
        print("No successful episodes, not saving anything.")
        return

    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    print(f"Collected expert dataset shapes: X={X_all.shape}, Y={Y_all.shape}")
    np.savez_compressed(save_path, X=X_all, Y=Y_all)
    print(f"Saved RRT* expert data to: {save_path}")


if __name__ == "__main__":
    main()
