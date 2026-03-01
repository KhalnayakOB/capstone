from sim_env_city_v3 import UAVCityEnvV3

env = UAVCityEnvV3(gui=True, dt=1.0/240.0)
env.set_camera_view_wide_city()

for i in range(2000):
    env.update_hud(distance=i*0.05, collisions=i//300)  # just to see text
    env.step()

env.close()
  