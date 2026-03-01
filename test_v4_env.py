from envs.sim_env_city_v4 import UAVCityEnvV4

def main():
    env = UAVCityEnvV4(gui=True)
    print("Start:", env.start_point)
    print("Goal:", env.end_point)

    # Run simulation for a few seconds so you can look around
    for _ in range(1000):
        env.step_dummy()

    env.close()

if __name__ == "__main__":
    main()
