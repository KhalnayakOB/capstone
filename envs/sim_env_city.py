import time
import numpy as np
import pybullet as p
import pybullet_data


class UAVCityEnv:
    def __init__(self, gui=True, dt=1 / 120):
        self.dt = dt

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # GUI settings
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.loadURDF("plane.urdf")

        # Arena dimensions
        self.box_half_x = 4.0
        self.box_half_y = 4.0
        self.box_height = 2.5

        self._create_walls()

        # Start & goal
        self.start_pos = np.array([-3.2, 0.0, 1.2])
        self.goal_pos = np.array([3.2, 0.0, 1.2])

        # Obstacles
        self.num_obstacles = 10
        self.obstacle_ids = []
        self._spawn_obstacles()

        # UAV
        self.uav_id = self._spawn_uav()

        self._set_camera()

    # ---------------- CAMERA ----------------
    def _set_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=90,
            cameraPitch=-89,
            cameraTargetPosition=[0, 0, 0],
        )

    # ---------------- WALLS ----------------
    def _create_walls(self):
        thickness = 0.1
        h = self.box_height / 2

        def wall(half, pos):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half,
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
            p.createMultiBody(0, col, vis, pos)

        wall([self.box_half_x, thickness, h], [0, self.box_half_y, h])
        wall([self.box_half_x, thickness, h], [0, -self.box_half_y, h])
        wall([thickness, self.box_half_y, h], [self.box_half_x, 0, h])
        wall([thickness, self.box_half_y, h], [-self.box_half_x, 0, h])

    # ---------------- OBSTACLES ----------------
    def _spawn_obstacles(self):
        for oid in self.obstacle_ids:
            p.removeBody(oid)
        self.obstacle_ids.clear()

        half = [0.35, 0.35, 0.35]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.9, 0.4, 0.2, 1]
        )

        for _ in range(self.num_obstacles):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            oid = p.createMultiBody(0, col, vis, [x, y, half[2]])
            self.obstacle_ids.append(oid)

    # ---------------- UAV ----------------
    def _spawn_uav(self):
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05]
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.05],
            rgbaColor=[0.2, 0.4, 0.9, 1],
        )
        return p.createMultiBody(1.0, col, vis, self.start_pos.tolist())

    # ---------------- API ----------------
    def get_state(self):
        pos, _ = p.getBasePositionAndOrientation(self.uav_id)
        vel, _ = p.getBaseVelocity(self.uav_id)
        return np.array(pos), np.array(vel)

    def reset(self, new_map=True):
        if new_map:
            self._spawn_obstacles()
        p.resetBasePositionAndOrientation(
            self.uav_id, self.start_pos.tolist(), [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.uav_id, [0, 0, 0], [0, 0, 0])

    def step(self):
        p.stepSimulation()
        time.sleep(self.dt)

    def close(self):
        if p.isConnected():
            p.disconnect()
