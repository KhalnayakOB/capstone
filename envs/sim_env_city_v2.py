import time
from typing import List

import numpy as np
import pybullet as p
import pybullet_data


class UAVCityEnvV2:
    """
    Cinematic city environment:

    - BIG map: pad ~50m x 50m
    - Rooftop pad with grid
    - Buildings, trees, water tanks, cell towers, bridges, pedestrians
    - START/END markers + green "optimal" line
    - Bright blue drone for visibility
    - get_obstacles_2d() for A* planner
    """

    def __init__(self, gui: bool = True, dt: float = 1.0 / 240.0):
        self.dt = dt
        self.gui = gui

        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        self._build_scene()

    # --------------------------------------------------
    # SCENE BUILD / RESET
    # --------------------------------------------------
    def _build_scene(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Debug visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        for flag in [
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ]:
            try:
                p.configureDebugVisualizer(flag, 0)
            except Exception:
                pass

        try:
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        except Exception:
            pass
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        except Exception:
            pass

        # Floor
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.25])

        # Pad & environment
        self._create_rooftop_pad()

        self.building_ids: List[int] = []
        self.building_info = []
        self._create_buildings_on_pad()

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

        self._create_trees()
        self._create_rooftop_tanks()
        self._create_cell_towers()
        self._create_bridges()
        self._create_pedestrians()

        self._define_start_end_points()
        self.drone_id = self._create_drone()
        self._create_start_end_markers_and_path()

        self._ensure_hd_buffer(1920, 1080)

    def reset_scene(self):
        self._build_scene()
        return self.drone_id

    # --------------------------------------------------
    # CAMERA MODES
    # --------------------------------------------------
    def set_camera_view_third_person(self):
        target = [0, 0, 1.0]
        p.resetDebugVisualizerCamera(
            cameraDistance=35.0,
            cameraYaw=35,
            cameraPitch=-30,
            cameraTargetPosition=target,
        )

    def set_camera_view_front(self):
        target = [0, 0, 1.5]
        p.resetDebugVisualizerCamera(
            cameraDistance=28.0,
            cameraYaw=180,
            cameraPitch=-5,
            cameraTargetPosition=target,
        )

    def set_camera_view_top(self):
        target = [0, 0, 0.0]
        p.resetDebugVisualizerCamera(
            cameraDistance=45.0,
            cameraYaw=0,
            cameraPitch=-89,
            cameraTargetPosition=target,
        )

    def set_camera_view_game(self, drone_pos, yaw_deg):
        target = [
            float(drone_pos[0]),
            float(drone_pos[1]),
            float(drone_pos[2]),
        ]
        p.resetDebugVisualizerCamera(
            cameraDistance=18.0,
            cameraYaw=float(yaw_deg),
            cameraPitch=-22,
            cameraTargetPosition=target,
        )

    def _ensure_hd_buffer(self, width: int = 1920, height: int = 1080):
        try:
            p.getCameraImage(width, height)
        except Exception:
            pass

    # --------------------------------------------------
    # ROOFTOP PAD + GRID
    # --------------------------------------------------
    def _create_rooftop_pad(self):
        # BIG map
        self.pad_half_x = 25.0
        self.pad_half_y = 25.0
        pad_thickness = 0.2

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, self.pad_half_y, pad_thickness / 2],
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, self.pad_half_y, pad_thickness / 2],
            rgbaColor=[0.87, 0.87, 0.9, 1.0],
        )
        self.pad_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, -pad_thickness / 2],
        )

        # Grid
        self.grid_line_ids: List[int] = []
        spacing = 1.0
        z = 0.001
        color = [0.8, 0.8, 0.85]
        thickness = 1

        xs = np.arange(-self.pad_half_x, self.pad_half_x + 1e-6, spacing)
        ys = np.arange(-self.pad_half_y, self.pad_half_y + 1e-6, spacing)

        for x in xs:
            start = [float(x), -self.pad_half_y, z]
            end = [float(x), self.pad_half_y, z]
            self.grid_line_ids.append(
                p.addUserDebugLine(start, end, color, thickness)
            )

        for y in ys:
            start = [-self.pad_half_x, float(y), z]
            end = [self.pad_half_x, float(y), z]
            self.grid_line_ids.append(
                p.addUserDebugLine(start, end, color, thickness)
            )

    # --------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------
    def _create_buildings_on_pad(self):
        base_half_xy = 1.0
        min_margin = 4.0
        min_dist = 6.0
        num_buildings = 30

        rng = np.random.default_rng()
        centers = []
        self.building_ids = []
        self.building_info = []

        for _ in range(num_buildings):
            for _try in range(400):
                x = rng.uniform(-self.pad_half_x + min_margin,
                                self.pad_half_x - min_margin)
                y = rng.uniform(-self.pad_half_y + min_margin,
                                self.pad_half_y - min_margin)
                pt = np.array([x, y])
                if all(np.linalg.norm(pt - c) >= min_dist for c in centers):
                    centers.append(pt)
                    break

        for (x, y) in centers:
            half_h = float(rng.uniform(2.0, 5.0))
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[base_half_xy, base_half_xy, half_h],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[base_half_xy, base_half_xy, half_h],
                rgbaColor=[0.62, 0.68, 0.78, 1.0],
            )
            bid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(x), float(y), half_h],
            )
            self.building_ids.append(bid)
            self.building_info.append({"x": float(x), "y": float(y), "half_h": half_h})

    # --------------------------------------------------
    # TREES
    # --------------------------------------------------
    def _create_trees(self, num_trees: int = 18):
        rng = np.random.default_rng()
        self.tree_ids = []
        self.tree_positions = []

        trunk_radius = 0.12
        trunk_half_h = 0.8
        trunk_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=trunk_radius, height=trunk_half_h * 2
        )
        trunk_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=trunk_radius,
            length=trunk_half_h * 2,
            rgbaColor=[0.4, 0.25, 0.1, 1.0],
        )

        crown_radius = 0.6
        crown_col = p.createCollisionShape(p.GEOM_SPHERE, radius=crown_radius)
        crown_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=crown_radius,
            rgbaColor=[0.1, 0.5, 0.1, 1.0],
        )

        for _ in range(num_trees):
            x = rng.uniform(-self.pad_half_x + 2.0, self.pad_half_x - 2.0)
            y = rng.uniform(-self.pad_half_y + 2.0, self.pad_half_y - 2.0)
            if abs(x) < 4 and abs(y) < 4:
                continue

            trunk_z = trunk_half_h
            trunk_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=trunk_col,
                baseVisualShapeIndex=trunk_vis,
                basePosition=[float(x), float(y), trunk_z],
            )
            crown_z = trunk_half_h * 2 + crown_radius * 0.7
            crown_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=crown_col,
                baseVisualShapeIndex=crown_vis,
                basePosition=[float(x), float(y), crown_z],
            )
            self.tree_ids.extend([trunk_id, crown_id])
            self.tree_positions.append((float(x), float(y)))

    # --------------------------------------------------
    # WATER TANKS
    # --------------------------------------------------
    def _create_rooftop_tanks(self, max_tanks: int = 10):
        self.water_tank_ids = []
        if not self.building_info:
            return

        rng = np.random.default_rng()
        chosen = rng.choice(len(self.building_info),
                            size=min(max_tanks, len(self.building_info)),
                            replace=False)

        tank_radius = 0.5
        tank_half_h = 0.6
        tank_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=tank_radius, height=tank_half_h * 2
        )
        tank_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=tank_radius,
            length=tank_half_h * 2,
            rgbaColor=[0.8, 0.85, 0.9, 1.0],
        )

        for idx in chosen:
            info = self.building_info[int(idx)]
            x, y, half_h = info["x"], info["y"], info["half_h"]
            z = 2 * half_h + tank_half_h + 0.05
            tid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=tank_col,
                baseVisualShapeIndex=tank_vis,
                basePosition=[x, y, z],
            )
            self.water_tank_ids.append(tid)

    # --------------------------------------------------
    # CELL TOWERS
    # --------------------------------------------------
    def _create_cell_towers(self):
        self.tower_ids = []
        self.tower_positions = []

        base_half = [0.4, 0.4, 0.3]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        base_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=base_half,
            rgbaColor=[0.3, 0.3, 0.35, 1.0],
        )

        mast_radius = 0.15
        mast_half_h = 4.0
        mast_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=mast_radius, height=mast_half_h * 2
        )
        mast_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=mast_radius,
            length=mast_half_h * 2,
            rgbaColor=[0.75, 0.75, 0.8, 1.0],
        )

        positions = [
            (self.pad_half_x - 3.0, self.pad_half_y - 3.0),
            (-self.pad_half_x + 3.0, self.pad_half_y - 3.0),
            (self.pad_half_x - 3.0, -self.pad_half_y + 3.0),
        ]

        for (x, y) in positions:
            base_z = 0.3
            base_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=base_col,
                baseVisualShapeIndex=base_vis,
                basePosition=[x, y, base_z],
            )
            mast_z = base_z + mast_half_h * 2
            mast_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=mast_col,
                baseVisualShapeIndex=mast_vis,
                basePosition=[x, y, mast_z],
            )
            self.tower_ids.extend([base_id, mast_id])
            self.tower_positions.append((float(x), float(y)))

    # --------------------------------------------------
    # BRIDGES
    # --------------------------------------------------
    def _create_bridges(self):
        self.bridge_ids = []

        bridge_half_x = self.pad_half_x
        bridge_half_y = 0.4
        bridge_half_h = 0.15
        bridge_z = 2.2

        col_x = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[bridge_half_x, bridge_half_y, bridge_half_h],
        )
        vis_x = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[bridge_half_x, bridge_half_y, bridge_half_h],
            rgbaColor=[0.7, 0.7, 0.75, 1.0],
        )
        bid = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_x,
            baseVisualShapeIndex=vis_x,
            basePosition=[0.0, 0.0, bridge_z],
        )
        self.bridge_ids.append(bid)

        bridge_half_x2 = 0.4
        bridge_half_y2 = self.pad_half_y
        col_y = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[bridge_half_x2, bridge_half_y2, bridge_half_h],
        )
        vis_y = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[bridge_half_x2, bridge_half_y2, bridge_half_h],
            rgbaColor=[0.7, 0.7, 0.75, 1.0],
        )
        bid2 = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_y,
            baseVisualShapeIndex=vis_y,
            basePosition=[0.0, 0.0, bridge_z + 0.05],
        )
        self.bridge_ids.append(bid2)

    # --------------------------------------------------
    # PEDESTRIANS
    # --------------------------------------------------
    def _create_pedestrians(self, num_peds: int = 5):
        self.pedestrian_ids = []
        self._ped_paths = []
        self._ped_t = []
        self._ped_dir = []

        ped_radius = 0.15
        ped_half_h = 0.6

        ped_col = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=ped_radius,
            height=ped_half_h * 2
        )

        ped_vis = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=ped_radius,
            length=ped_half_h * 2,
            rgbaColor=[0.9, 0.8, 0.5, 1.0],
        )

        rng = np.random.default_rng()
        for _ in range(num_peds):
            x0 = rng.uniform(-self.pad_half_x + 4.0, self.pad_half_x - 4.0)
            y0 = rng.uniform(-self.pad_half_y + 4.0, self.pad_half_y - 4.0)

            dx = rng.uniform(-6.0, 6.0)
            dy = rng.uniform(-6.0, 6.0)

            start = np.array([x0, y0, ped_half_h])
            end = np.array([x0 + dx, y0 + dy, ped_half_h])

            pid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=ped_col,
                baseVisualShapeIndex=ped_vis,
                basePosition=start.tolist(),
            )

            self.pedestrian_ids.append(pid)
            self._ped_paths.append((start, end))
            self._ped_t.append(0.0)
            self._ped_dir.append(1.0)

    def _update_pedestrians(self):
        speed = 1.5
        for i, pid in enumerate(self.pedestrian_ids):
            start, end = self._ped_paths[i]
            direction = self._ped_dir[i]
            t = self._ped_t[i]

            segment = end - start
            length = float(np.linalg.norm(segment))
            if length < 1e-6:
                continue

            dt_param = (speed * self.dt) / length
            t += direction * dt_param

            if t > 1.0:
                t = 1.0
                direction = -1.0
            elif t < 0.0:
                t = 0.0
                direction = 1.0

            self._ped_t[i] = t
            self._ped_dir[i] = direction
            pos = start + t * segment
            p.resetBasePositionAndOrientation(pid, pos.tolist(), [0, 0, 0, 1])

    # --------------------------------------------------
    # START / END POINTS + PATH
    # --------------------------------------------------
    def _define_start_end_points(self):
        self.start_point = np.array([-self.pad_half_x + 1.0, 0.0, 2.0])
        self.end_point = np.array([self.pad_half_x - 1.0, 0.0, 2.0])
        self.start_pos = self.start_point.copy()

    def _create_start_end_markers_and_path(self):
        radius = 0.25
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        vis_start = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[0.1, 0.85, 0.1, 1.0]
        )
        vis_end = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[0.9, 0.1, 0.1, 1.0]
        )

        self.start_marker_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis_start,
            basePosition=self.start_point.tolist(),
        )
        self.end_marker_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis_end,
            basePosition=self.end_point.tolist(),
        )

        self.start_text_id = p.addUserDebugText(
            "START",
            [self.start_point[0] - 1.5, self.start_point[1], self.start_point[2] + 1.1],
            textColorRGB=[0.1, 1.0, 0.1],
            textSize=2.5,
        )
        self.end_text_id = p.addUserDebugText(
            "END",
            [self.end_point[0] + 1.0, self.end_point[1], self.end_point[2] + 1.1],
            textColorRGB=[1.0, 0.3, 0.3],
            textSize=2.5,
        )

        self.path_line_id = p.addUserDebugLine(
            self.start_point.tolist(),
            self.end_point.tolist(),
            [0.0, 1.0, 0.0],
            3,
        )

    # --------------------------------------------------
    # DRONE
    # --------------------------------------------------
    def _create_drone(self) -> int:
        body_half = [0.6, 0.45, 0.09]
        body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half)

        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=body_half,
            rgbaColor=[0.12, 0.25, 0.9, 1.0],  # bright blue
        )

        drone_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=self.start_pos.tolist(),
        )
        return drone_id

    # --------------------------------------------------
    # OBSTACLES for PLANNER
    # --------------------------------------------------
    def get_obstacles_2d(self):
        """
        Returns list of (x, y, radius) obstacles for 2D path planning.
        Buildings + trees + cell towers.
        """
        obs = []

        # Buildings – slightly inflated radius
        for info in self.building_info:
            x, y = info["x"], info["y"]
            r = 1.6
            obs.append((x, y, r))

        # Trees
        for (x, y) in getattr(self, "tree_positions", []):
            obs.append((x, y, 1.0))

        # Cell towers
        for (x, y) in getattr(self, "tower_positions", []):
            obs.append((x, y, 1.4))

        return obs

    # --------------------------------------------------
    # STATE / STEP / CLOSE
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

    def step(self):
        p.stepSimulation()
        self._update_pedestrians()
        time.sleep(self.dt)

    def close(self):
        if p.isConnected():
            p.disconnect()
