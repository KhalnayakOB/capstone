import numpy as np

from sim_env_city_v2 import UAVCityEnvV2
from rrt_star_3d import compute_path_rrt_star_3d


def build_3d_obstacles_from_2d(obstacles_2d, default_height=10.0, base_z=0.0):
    """
    Convert 2D circular obstacles (x, y, radius) into simple 3D columns:
        (ox, oy, oz, radius_xy, height_z)
    """
    obstacles_3d = []
    for (x, y, r) in obstacles_2d:
        ox = float(x)
        oy = float(y)
        oz = float(base_z)
        radius_xy = float(r)
        height_z = float(default_height)
        obstacles_3d.append((ox, oy, oz, radius_xy, height_z))
    return obstacles_3d


def generate_expert_trajectory_from_3d_path(path_xyz: np.ndarray,
                                            speed: float = 5.0,
                                            spatial_step: float = 1.0):
    """
    Given a 3D path (N, 3) from RRT*3D, generate expert state-action samples.

    X: [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z]
    Y: [vx, vy, vz]
    """
    if path_xyz.shape[0] < 2:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    goal = path_xyz[-1].copy()
    X_samples = []
    Y_samples = []

    current_pos = path_xyz[0].copy()

    for i in range(path_xyz.shape[0] - 1):
        p0 = path_xyz[i]
        p1 = path_xyz[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue

        n_steps = max(int(seg_len / spatial_step), 1)
        step_vec = seg_vec / n_steps

        dir_norm = np.linalg.norm(step_vec)
        if dir_norm < 1e-6:
            continue

        dir_vec = step_vec / dir_norm
        v_des = dir_vec * speed

        for _ in range(n_steps):
            state_feat = np.concatenate([current_pos, goal])
            X_samples.append(state_feat)
            Y_samples.append(v_des.copy())
            current_pos = current_pos + step_vec

    X = np.array(X_samples, dtype=np.float32)
    Y = np.array(Y_samples, dtype=np.float32)
    return X, Y


def run_episode(env: UAVCityEnvV2,
                z_bounds=(1.0, 12.0),
                obstacle_height=10.0):
    """
    One episode:
      - Reset env
      - Build 3D obstacles
      - Run 3D RRT*
      - Convert path to (X, Y)
    """
    env.reset_scene()

    obstacles_2d = env.get_obstacles_2d()
    obstacles_3d = build_3d_obstacles_from_2d(
        obstacles_2d,
        default_height=obstacle_height,
        base_z=0.0,
    )

    bounds_3d = (
        (-env.pad_half_x, env.pad_half_x),
        (-env.pad_half_y, env.pad_half_y),
        z_bounds,
    )

    start_xyz = (
        float(env.start_point[0]),
        float(env.start_point[1]),
        float(env.start_point[2]),
    )
    goal_xyz = (
        float(env.end_point[0]),
        float(env.end_point[1]),
        float(env.end_point[2]),
    )

    path = compute_path_rrt_star_3d(
        start_xyz=start_xyz,
        goal_xyz=goal_xyz,
        obstacles=obstacles_3d,
        bounds=bounds_3d,
    )

    path_xyz = np.array(path, dtype=np.float32)
    if path_xyz.shape[0] < 2:
        print("  [WARN] 3D RRT* could not find a good path, skipping episode.")
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    X, Y = generate_expert_trajectory_from_3d_path(
        path_xyz,
        speed=5.0,
        spatial_step=1.0,
    )
    return X, Y


def main():
    num_episodes = 200
    save_path = "expert_data_rrt_3d.npz"

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
        print("No successful episodes; nothing to save.")
        return

    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    print(f"Collected 3D RRT* expert dataset: X={X_all.shape}, Y={Y_all.shape}")
    np.savez_compressed(save_path, X=X_all, Y=Y_all)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
