import time
from typing import List
import numpy as np
import pybullet as p
import pybullet_data


class UAVCityEnvV2:

    def __init__(self, gui: bool = True, dt: float = 1 / 240):
        self.dt = dt
        self.gui = gui

        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        # Collision tracking
        self.collision_count = 0
        self._collision_text_id = None

        self._build_scene()

    # --------------------------------------------------
    # BUILD SCENE
    # --------------------------------------------------
    def _build_scene(self):

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.25])

        self._create_rooftop_pad()

        self.building_ids: List[int] = []
        self.building_info = []

        self.tree_ids: List[int] = []
        self.tree_positions: List[tuple] = []

        self.water_tank_ids: List[int] = []
        self.tower_ids: List[int] = []
        self.tower_positions: List[tuple] = []

        self.bridge_ids: List[int] = []
        self.pedestrian_ids: List[int] = []

        self._ped_paths = []
        self._ped_t = []
        self._ped_dir = []

        self._create_buildings_on_pad()
        self._create_trees()
        self._create_rooftop_tanks()
        self._create_cell_towers()
        self._create_bridges()
        self._create_pedestrians()

        # Start/End logic
        self._define_start_end_points()

        # Drones
        self.drones = self._create_multi_drones(5)
        self.drone_id = self.drones[0]

        self._create_start_end_markers_and_path()
        self._ensure_hd_buffer()

    # --------------------------------------------------
    # START / END (FIXED)
    # --------------------------------------------------
    def _define_start_end_points(self):

        rng = np.random.default_rng()

        self.drone_start_positions = []
        self.drone_end_positions = []

        for i in range(5):
            y_start = rng.uniform(-20, 20)
            y_end = rng.uniform(-20, 20)

            start = np.array([-24, y_start, 2])
            end = np.array([24, y_end, 2])

            self.drone_start_positions.append(start)
            self.drone_end_positions.append(end)

        self.start_point = self.drone_start_positions[0]
        self.end_point = self.drone_end_positions[0]
        self.start_pos = self.start_point.copy()

    def get_drone_goal(self, idx):
        return self.drone_end_positions[idx]

    def get_drone_start(self, idx):
        return self.drone_start_positions[idx]

    # --------------------------------------------------
    # DRONES
    # --------------------------------------------------
    def _create_multi_drones(self, n):

        drones = []

        for i in range(n):

            pos = self.drone_start_positions[i]

            color = [0.1, 0.3, 1, 1] if i == 0 else [1, 0.85, 0.1, 1]

            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.6, 0.45, 0.09],
            )

            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.6, 0.45, 0.09],
                rgbaColor=color,
            )

            drone = p.createMultiBody(
                1.2,
                col,
                vis,
                pos.tolist(),
            )

            p.changeDynamics(
                drone,
                -1,
                linearDamping=0.9,
                angularDamping=0.9,
                lateralFriction=0,
            )

            drones.append(drone)

        return drones

    # --------------------------------------------------
    # STEP (FIXED)
    # --------------------------------------------------
    def step(self):

        # ---- Pedestrian movement ----
        for i, pid in enumerate(self.pedestrian_ids):

            start, end = self._ped_paths[i]

            self._ped_t[i] += 0.005 * self._ped_dir[i]

            if self._ped_t[i] > 1:
                self._ped_t[i] = 1
                self._ped_dir[i] = -1
            elif self._ped_t[i] < 0:
                self._ped_t[i] = 0
                self._ped_dir[i] = 1

            pos = (1 - self._ped_t[i]) * start + self._ped_t[i] * end

            p.resetBasePositionAndOrientation(pid, pos.tolist(), [0, 0, 0, 1])

        # ---- Collision detection (FIXED) ----
        contacts = p.getContactPoints()

        unique_pairs = set()
        for c in contacts:
            pair = (c[1], c[2])
            unique_pairs.add(pair)

        self.collision_count += len(unique_pairs)

        # ---- HUD collision counter (FIXED) ----
        if self._collision_text_id is None:
            self._collision_text_id = p.addUserDebugText(
                f"Collisions: {self.collision_count}",
                [0, 0, 0],
                textColorRGB=[1, 0, 0],
                textSize=2,
                parentObjectUniqueId=self.drones[0],
            )
        else:
            p.addUserDebugText(
                f"Collisions: {self.collision_count}",
                [0, 0, 0],
                textColorRGB=[1, 0, 0],
                textSize=2,
                parentObjectUniqueId=self.drones[0],
                replaceItemUniqueId=self._collision_text_id,
            )

        p.stepSimulation()
        time.sleep(self.dt)

    # --------------------------------------------------
    # CAMERA
    # --------------------------------------------------
    def set_camera_view_third_person(self):
        p.resetDebugVisualizerCamera(35, 35, -30, [0, 0, 1])

    def set_camera_view_top(self):
        p.resetDebugVisualizerCamera(45, 0, -89, [0, 0, 0])

    def set_camera_view_game(self, drone_pos, yaw):
        p.resetDebugVisualizerCamera(
            18,
            float(yaw),
            -20,
            [float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])],
        )

    def _ensure_hd_buffer(self, width=1920, height=1080):
        try:
            p.getCameraImage(width, height)
        except Exception:
            pass

    # --------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------
    def _create_rooftop_pad(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[25, 25, 0.1])
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[25, 25, 0.1],
            rgbaColor=[0.87, 0.87, 0.9, 1],
        )
        self.pad_id = p.createMultiBody(0, col, vis, [0, 0, -0.1])

    def _create_buildings_on_pad(self):
        rng = np.random.default_rng()
        centers = []

        for _ in range(30):
            for _ in range(200):
                x = rng.uniform(-20, 20)
                y = rng.uniform(-20, 20)
                pt = np.array([x, y])
                if all(np.linalg.norm(pt - c) >= 6 for c in centers):
                    centers.append(pt)
                    break

        for (x, y) in centers:
            half_h = rng.uniform(2, 5)

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, half_h])
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[1, 1, half_h],
                rgbaColor=[0.6, 0.7, 0.8, 1],
            )

            bid = p.createMultiBody(0, col, vis, [x, y, half_h])

            self.building_ids.append(bid)
            self.building_info.append({"x": x, "y": y, "half_h": half_h})

    def _create_trees(self, n=20):
        rng = np.random.default_rng()

        trunk_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=1.6)
        trunk_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.12,
            length=1.6,
            rgbaColor=[0.4, 0.25, 0.1, 1],
        )

        crown_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.6)
        crown_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.6,
            rgbaColor=[0.1, 0.5, 0.1, 1],
        )

        for _ in range(n):
            x = rng.uniform(-22, 22)
            y = rng.uniform(-22, 22)

            trunk = p.createMultiBody(0, trunk_col, trunk_vis, [x, y, 0.8])
            crown = p.createMultiBody(0, crown_col, crown_vis, [x, y, 2])

            self.tree_ids.extend([trunk, crown])
            self.tree_positions.append((x, y))

    def _create_rooftop_tanks(self): pass
    def _create_cell_towers(self): pass
    def _create_bridges(self): pass
    def _create_pedestrians(self, n=5): pass
    def _create_start_end_markers_and_path(self): pass

    def get_obstacles_2d(self):
        return [(b["x"], b["y"], 1.6) for b in self.building_info]

    def close(self):
        if p.isConnected():
            p.disconnect()