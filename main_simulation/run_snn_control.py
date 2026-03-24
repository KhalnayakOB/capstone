import time
import numpy as np
import torch
import pybullet as p
from sklearn.preprocessing import StandardScaler

from envs.sim_env_city_v2 import UAVCityEnvV2


# --------------------------------------------------
# FIX FOR PYTORCH 2.6 SAFE LOADING
# --------------------------------------------------
torch.serialization.add_safe_globals([StandardScaler])


# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# LOAD MODEL (FIXED)
# --------------------------------------------------
checkpoint = torch.load(
    "snn_model.pth",
    map_location=device,
    weights_only=False
)


# --------------------------------------------------
# MODEL (same as training)
# --------------------------------------------------
class SNN(torch.nn.Module):

    def __init__(self):
        super(SNN, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 2),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


model = SNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler_X = checkpoint["scaler_X"]


# --------------------------------------------------
# PHYSICS CONTROL
# --------------------------------------------------
FORCE_SCALE = 20
VERT_FORCE = 12


def move_drone(drone_id, vx, vy, vz):

    force = [
        float(vx * FORCE_SCALE),
        float(vy * FORCE_SCALE),
        float(vz * VERT_FORCE)
    ]

    p.applyExternalForce(
        objectUniqueId=drone_id,
        linkIndex=-1,
        forceObj=force,
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME
    )


# --------------------------------------------------
# SNN POLICY
# --------------------------------------------------
def snn_policy(pos, goal):

    pos_2d = pos[:2]
    goal_2d = goal[:2]

    dist = np.linalg.norm(goal_2d - pos_2d)

    state = np.array([
        pos_2d[0],
        pos_2d[1],
        goal_2d[0],
        goal_2d[1],
        dist
    ]).reshape(1, -1)

    state = scaler_X.transform(state)
    state = torch.tensor(state, dtype=torch.float32).to(device)

    with torch.no_grad():
        action = model(state).cpu().numpy()[0]

    vx = action[0] * 2.5
    vy = action[1] * 2.5
    vz = 0.0

    return vx, vy, vz


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
def main():

    env = UAVCityEnvV2(gui=True)
    drones = env.drones

    print("\n🚁 SNN CONTROL RUNNING\n")
    print("Press Q to quit\n")

    while True:

        keys = p.getKeyboardEvents()

        for i, drone in enumerate(drones):

            pos, _ = p.getBasePositionAndOrientation(drone)
            pos = np.array(pos)

            goal = env.get_drone_goal(i)

            vx, vy, vz = snn_policy(pos, goal)

            move_drone(drone, vx, vy, vz)

        if ord('q') in keys:
            break

        env.step()

    env.close()


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    main()