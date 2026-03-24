import pybullet as p
import time

from envs.sim_env_city_v2 import UAVCityEnvV2
from envs.multi_uav_controller import MultiUAVController


SPEED = 15
VERT_SPEED = 12


def main():

    env = UAVCityEnvV2(gui=True)
    if not p.isConnected():
        p.connect(p.GUI)
    controller = MultiUAVController(env, env.drones)

    drone = controller.drone_ids[0]

    env.set_camera_view_third_person()

    print("""
Controls:

U/J : Forward / Back
H/K : Left / Right
↑/↓ : Up / Down

1 : Third person camera
2 : Top view
3 : Drone follow camera

P : Toggle autopilot
Z : Show / Hide RRT path

Q : Quit
""")

    while True:

        keys = p.getKeyboardEvents()

        vx = vy = vz = 0

        # -------------------------
        # CAMERA
        # -------------------------

        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            env.set_camera_view_third_person()

        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            env.set_camera_view_top()

        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:

            pos, orn = p.getBasePositionAndOrientation(drone)
            yaw = p.getEulerFromQuaternion(orn)[2]
            yaw_deg = yaw * 57.3

            env.set_camera_view_game(pos, yaw_deg)

        # -------------------------
        # RRT VISUALIZATION
        # -------------------------

        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            controller.toggle_rrt_visualization()

        # -------------------------
        # MANUAL CONTROL
        # -------------------------

        if not controller.autopilot:

            if ord('u') in keys:
                vy = SPEED

            if ord('j') in keys:
                vy = -SPEED

            if ord('h') in keys:
                vx = -SPEED

            if ord('k') in keys:
                vx = SPEED

            if p.B3G_UP_ARROW in keys:
                vz = VERT_SPEED

            if p.B3G_DOWN_ARROW in keys:
                vz = -VERT_SPEED

            p.resetBaseVelocity(drone, linearVelocity=[vx, vy, vz])

        # -------------------------
        # AUTOPILOT
        # -------------------------

        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            controller.toggle_autopilot()

        controller.update_autopilot()

        # -------------------------
        # EXIT
        # -------------------------

        if ord('q') in keys:
            break

        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()