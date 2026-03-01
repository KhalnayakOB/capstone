import numpy as np
import pybullet as p

# IMPORTANT: correct import based on your folder structure
from envs.sim_env_city import UAVCityEnv


# ---------------- SAFE STRAIGHT PATH ----------------
def compute_straight_path(start_pos, goal_pos, step_size=0.15):
    start_pos = np.array(start_pos, dtype=float)
    goal_pos = np.array(goal_pos, dtype=float)

    direction = goal_pos - start_pos
    dist = np.linalg.norm(direction)

    if dist < 1e-6:
        return [start_pos.copy()]

    direction /= dist
    steps = int(dist / step_size)

    return [
        start_pos + i * step_size * direction
        for i in range(steps + 1)
    ]


# ---------------- MAIN ----------------
def main():
    env = UAVCityEnv()
    drone_id = env.uav_id

    speed = 6.0
    autopilot = False
    autopilot_path = []
    autopilot_index = 0

    RESET = p.addUserDebugParameter("Reset (New Map)", 0, 1, 0)

    print("Controls:")
    print("W/S = forward/back | A/D = left/right")
    print("P = toggle AUTOPILOT | Q = quit")

    while True:
        keys = p.getKeyboardEvents()

        # -------- TOGGLE AUTOPILOT --------
        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            autopilot = not autopilot
            print("AUTOPILOT:", "ON" if autopilot else "OFF")

            if autopilot:
                pos, _ = env.get_state()
                autopilot_path = compute_straight_path(
                    start_pos=pos,
                    goal_pos=env.goal_pos
                )
                autopilot_index = 0

                if len(autopilot_path) == 0:
                    print("Autopilot failed → disabling")
                    autopilot = False

        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

        # -------- RESET --------
        if p.readUserDebugParameter(RESET) > 0.5:
            env.reset(new_map=True)
            autopilot = False

        vx = vy = vz = 0.0
        pos, vel = env.get_state()

        # -------- MANUAL --------
        if not autopilot:
            if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
                vy += speed
            if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
                vy -= speed
            if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
                vx -= speed
            if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
                vx += speed

        # -------- AUTOPILOT --------
        else:
            if autopilot_index < len(autopilot_path):
                target = autopilot_path[autopilot_index]
                diff = target - pos
                dist = np.linalg.norm(diff[:2])

                if dist < 0.2:
                    autopilot_index += 1
                else:
                    direction = diff / (np.linalg.norm(diff) + 1e-6)
                    vx = speed * direction[0]
                    vy = speed * direction[1]
            else:
                print("Autopilot reached goal")
                autopilot = False

        p.resetBaseVelocity(
            drone_id,
            linearVelocity=[vx, vy, vz],
            angularVelocity=[0, 0, 0],
        )

        env.step()

    env.close()


if __name__ == "__main__":
    main()
