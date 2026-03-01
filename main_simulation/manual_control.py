import time
from typing import List
import numpy as np
import pybullet as p
import pybullet_data


class UAVCityEnvV2:
    """
    Stable cinematic city environment (FIXED VERSION)

    ✔ Safe single connection
    ✔ Large visible drone
    ✔ Default working camera
    ✔ Stable stepping
    ✔ Compatible with manual_control
    """

    def __init__(self, gui: bool = True, dt: float = 1.0 / 240.0):

        # ---------- SAFE CONNECTION ----------
        if p.isConnected():
            p.disconnect()

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        self.dt = dt

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        self._build_scene()
        self.set_camera_view_third_person()

    # --------------------------------------------------
    # SCENE BUILD
    # --------------------------------------------------
    def _build_scene(self):

        p.resetSimulation()

        # Ground
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.25])

        # Big rooftop pad
        self.pad_half_x = 25
        self.pad_half_y = 25

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, self.pad_half_y, 0.1]
        )

        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, self.pad_half_y, 0.1],
            rgbaColor=[0.85, 0.85, 0.9, 1]
        )

        self.pad_id = p.createMultiBody(0, col, vis, [0, 0, 0])

        # Buildings
        self.building_ids = []
        rng = np.random.default_rng(2)

        for _ in range(40):
            x = rng.uniform(-22, 22)
            y = rng.uniform(-22, 22)
            h = rng.uniform(2, 6)

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, h])
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[1, 1, h],
                rgbaColor=[0.6, 0.65, 0.75, 1]
            )

            bid = p.createMultiBody(0, col, vis, [x, y, h])
            self.building_ids.append(bid)

        # ---------- DRONE (BIG & VISIBLE) ----------
        self.start_pos = np.array([-20, 0, 3])

        drone_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.9, 0.9, 0.25])
        drone_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.9, 0.9, 0.25],
            rgbaColor=[0.05, 0.3, 1.0, 1]  # bright blue
        )

        self.drone_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=drone_col,
            baseVisualShapeIndex=drone_vis,
            basePosition=self.start_pos.tolist()
        )

    # --------------------------------------------------
    # CAMERA
    # --------------------------------------------------
    def set_camera_view_third_person(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=35,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 2]
        )

    def set_camera_follow(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=40,
            cameraPitch=-25,
            cameraTargetPosition=pos
        )

    def set_camera_top(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=60,
            cameraYaw=0,
            cameraPitch=-89,
            cameraTargetPosition=pos
        )

    # --------------------------------------------------
    # STATE
    # --------------------------------------------------
    def get_drone_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang = p.getBaseVelocity(self.drone_id)
        return {
            "pos": np.array(pos),
            "orn": np.array(orn),
            "vel": np.array(vel),
            "ang_vel": np.array(ang),
        }

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------
    def step(self):
        p.stepSimulation()
        self.set_camera_follow()
        time.sleep(self.dt)

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------
    def reset_scene(self):
        p.resetBasePositionAndOrientation(
            self.drone_id,
            self.start_pos.tolist(),
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------
    def close(self):
        if p.isConnected():
            p.disconnect()