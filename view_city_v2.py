import time
import pybullet as p
from envs.sim_env_city_v2 import UAVCityEnvV2


def main():
    print("Launching sim_env_city_v2 (visual preview)...")

    env = UAVCityEnvV2(gui=True)

    print("""
V2 CITY PREVIEW MODE
-------------------
This is VISUAL ONLY.
No controls are enabled.

- Buildings, trees, tanks, bridges
- Pedestrians will move
- Start / End markers visible

Close the PyBullet window or press Ctrl+C to exit.
""")

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("Closing preview...")

    env.close()


if __name__ == "__main__":
    main()
