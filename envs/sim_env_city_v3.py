import time
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data


class UAVCityEnvV3:
    """
    150m x 150m city environment for UAV navigation + SNN learning.

    Objects:
      - Large static floor (150 x 150 m) with grid.
      - Dozens of buildings with varying heights.
      - A few 'bridges' (elevated blocks across the corridor).
      - Trees, water-tanks, and simple cell towers.
      - Start point on the left edge, End point on right edge.
      - One UAV (box drone) spawned at START.

    Compatibility:
      - self.uav_ids = [drone_id]
      - get_states() -> (1, 6) array [x, y, z, vx, vy, vz]
      - get_obstacles_2d() returns (x, y, radius) for A* planner.
      - init_hud() / update_hud(distance, collisions) for debug text.
    """

    # --------------------------------------------------
    # INIT / CONSTRUCTION
    # --------------------------------------------------
    def __init__(self, gui: bool = True, dt: float = 1.0 / 240.0):
        self.dt = dt
        self.gui = gui

        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        self._build_scene()
        self.init_hud()

    def _build_scene(self):
        # Reset and basic settings
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Clean PyBullet UI
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        except Exception:
            pass
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
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        except Exception:
            pass

        # Plane underneath everything (for shadows)
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.3])

        # Floor & grid
        self._create_main_floor()

        # Containers for objects
        self.building_ids: List[int] = []
        self.building_info: List[dict] = []
        self.bridge_ids: List[int] = []
        self.bridge_info: List[dict] = []
        self.tree_ids: List[int] = []
        self.tree_positions: List[Tuple[float, float]] = []
        self.water_tank_ids: List[int] = []
        self.tower_ids: List[int] = []
        self.tower_positions: List[Tuple[float, float]] = []

        # Populate city
        self._create_buildings(num_buildings=90)
        self._create_bridges(num_bridges=3)
        self._create_trees(num_trees=45)
        self._create_rooftop_tanks(max_tanks=18)
        self._create_cell_towers()

        # Start / end markers & drone
        self._define_start_end_points()
        self._create_start_end_markers()
        self.drone_id = self._create_drone()
        self.uav_ids = [self.drone_id]  # backwards compatibility

        # Pre-warm camera buffer
        self._ensure_hd_buffer(1280, 720)

    # --------------------------------------------------
    # FLOOR + GRID
    # --------------------------------------------------
    def _create_main_floor(self):
        """Create a 150m x 150m static floor with a grid."""
        self.pad_half_x = 75.0  # total width = 150m
        self.pad_half_y = 75.0  # total depth = 150m
        pad_thickness = 0.3

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, self.pad_half_y, pad_thickness / 2],
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, self.pad_half_y, pad_thickness / 2],
            rgbaColor=[0.9, 0.9, 0.95, 1.0],
        )
        self.pad_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, -pad_thickness / 2],
        )

        # Draw simple grid lines
        self.grid_line_ids: List[int] = []
        z = 0.01
        spacing = 3.0
        color = [0.82, 0.82, 0.87]
        thickness = 1

        xs = np.arange(-self.pad_half_x, self.pad_half_x + 1e-6, spacing)
        ys = np.arange(-self.pad_half_y, self.pad_half_y + 1e-6, spacing)

        for x in xs:
            s = [float(x), -self.pad_half_y, z]
            e = [float(x), self.pad_half_y, z]
            self.grid_line_ids.append(p.addUserDebugLine(s, e, color, thickness))

        for y in ys:
            s = [-self.pad_half_x, float(y), z]
            e = [self.pad_half_x, float(y), z]
            self.grid_line_ids.append(p.addUserDebugLine(s, e, color, thickness))

    # --------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------
    def _create_buildings(self, num_buildings: int = 90):
        """
        Dense but navigable arrangement of buildings:
          - keep a corridor along y~0 for start→end flight.
        """
        self.building_ids = []
        self.building_info = []

        rng = np.random.default_rng()
        centers = []
        margin = 8.0
        min_dist = 5.0
        corridor_half_width = 4.5  # keep |y| < this free-ish

        for _ in range(num_buildings):
            for _try in range(500):
                x = rng.uniform(-self.pad_half_x + margin, self.pad_half_x - margin)
                y = rng.uniform(-self.pad_half_y + margin, self.pad_half_y - margin)

                # keep corridor around y=0 where drone flies
                if abs(y) < corridor_half_width and abs(x) < self.pad_half_x * 0.6:
                    continue

                pt = np.array([x, y])
                if all(np.linalg.norm(pt - c) >= min_dist for c in centers):
                    centers.append(pt)
                    break

        base_half_xy = 1.6
        for (x, y) in centers:
            half_h = float(rng.uniform(3.0, 10.0))
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[base_half_xy, base_half_xy, half_h],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[base_half_xy, base_half_xy, half_h],
                rgbaColor=[0.60, 0.67, 0.78, 1.0],
            )
            bid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(x), float(y), half_h],
            )
            self.building_ids.append(bid)
            radius = base_half_xy * 1.7
            self.building_info.append({"x": float(x), "y": float(y), "radius": float(radius)})

    # --------------------------------------------------
    # BRIDGES (elevated obstacles across the corridor)
    # --------------------------------------------------
    def _create_bridges(self, num_bridges: int = 3):
        """
        Simple 'bridges' that span across the central corridor at mid-height.
        """
        self.bridge_ids = []
        self.bridge_info = []

        rng = np.random.default_rng()
        z_center = 6.0
        half_thickness = 0.6

        for _ in range(num_bridges):
            # span roughly across x ∈ [-40, 40]
            bridge_half_x = rng.uniform(25.0, 40.0)
            bridge_half_y = rng.uniform(1.0, 2.0)
            y = rng.uniform(-3.0, 3.0)

            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[bridge_half_x, bridge_half_y, half_thickness],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[bridge_half_x, bridge_half_y, half_thickness],
                rgbaColor=[0.55, 0.58, 0.65, 1.0],
            )
            bid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[0.0, float(y), z_center],
            )
            self.bridge_ids.append(bid)
            self.bridge_info.append(
                {
                    "x": 0.0,
                    "y": float(y),
                    "radius": float(bridge_half_x),
                }
            )

    # --------------------------------------------------
    # TREES
    # --------------------------------------------------
    def _create_trees(self, num_trees: int = 45):
        self.tree_ids = []
        self.tree_positions = []

        rng = np.random.default_rng()
        trunk_radius = 0.14
        trunk_half_h = 0.9
        trunk_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=trunk_radius, height=trunk_half_h * 2
        )
        trunk_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=trunk_radius,
            length=trunk_half_h * 2,
            rgbaColor=[0.4, 0.25, 0.1, 1.0],
        )

        crown_radius = 0.7
        crown_col = p.createCollisionShape(p.GEOM_SPHERE, radius=crown_radius)
        crown_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=crown_radius,
            rgbaColor=[0.1, 0.55, 0.1, 1.0],
        )

        for _ in range(num_trees):
            x = rng.uniform(-self.pad_half_x + 4.0, self.pad_half_x - 4.0)
            y = rng.uniform(-self.pad_half_y + 4.0, self.pad_half_y - 4.0)

            # avoid strict central corridor
            if abs(x) < 10 and abs(y) < 6:
                continue

            trunk_z = trunk_half_h
            tid1 = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=trunk_col,
                baseVisualShapeIndex=trunk_vis,
                basePosition=[float(x), float(y), trunk_z],
            )
            crown_z = trunk_half_h * 2 + crown_radius * 0.7
            tid2 = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=crown_col,
                baseVisualShapeIndex=crown_vis,
                basePosition=[float(x), float(y), crown_z],
            )

            self.tree_ids.extend([tid1, tid2])
            self.tree_positions.append((float(x), float(y)))

    # --------------------------------------------------
    # WATER TANKS ON BUILDINGS
    # --------------------------------------------------
    def _create_rooftop_tanks(self, max_tanks: int = 18):
        self.water_tank_ids = []
        if not self.building_info:
            return

        rng = np.random.default_rng()
        chosen = rng.choice(
            len(self.building_info),
            size=min(max_tanks, len(self.building_info)),
            replace=False,
        )

        tank_radius = 0.6
        tank_half_h = 0.7
        tank_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=tank_radius, height=tank_half_h * 2
        )
        tank_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=tank_radius,
            length=tank_half_h * 2,
            rgbaColor=[0.85, 0.88, 0.93, 1.0],
        )

        for idx in chosen:
            info = self.building_info[int(idx)]
            x, y = info["x"], info["y"]
            z = 6.0 + tank_half_h  # approx top
            tid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=tank_col,
                baseVisualShapeIndex=tank_vis,
                basePosition=[x, y, z],
            )
            self.water_tank_ids.append(tid)

    # --------------------------------------------------
    # SIMPLE CELL TOWERS
    # --------------------------------------------------
    def _create_cell_towers(self):
        self.tower_ids = []
        self.tower_positions = []

        base_half = [0.5, 0.5, 0.35]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        base_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=base_half,
            rgbaColor=[0.3, 0.3, 0.35, 1.0],
        )

        mast_radius = 0.16
        mast_half_h = 5.0
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
            (self.pad_half_x - 7.0, self.pad_half_y - 7.0),
            (-self.pad_half_x + 7.0, self.pad_half_y - 7.0),
            (self.pad_half_x - 7.0, -self.pad_half_y + 7.0),
        ]

        for (x, y) in positions:
            base_z = 0.35
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
    # START / END + DRONE
    # --------------------------------------------------
    def _define_start_end_points(self):
        """Start at left border, End at right border, same altitude."""
        z_flight = 3.0
        self.start_point = np.array([-self.pad_half_x + 6.0, 0.0, z_flight])
        self.end_point = np.array([self.pad_half_x - 6.0, 0.0, z_flight])
        self.start_pos = self.start_point.copy()

    def _create_start_end_markers(self):
        radius = 0.4
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        vis_start = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[0.1, 0.9, 0.1, 1.0]
        )
        vis_end = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[0.9, 0.2, 0.2, 1.0]
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
            [self.start_point[0] - 2.5, self.start_point[1], self.start_point[2] + 1.5],
            textColorRGB=[0.1, 1.0, 0.1],
            textSize=2.0,
        )
        self.end_text_id = p.addUserDebugText(
            "END",
            [self.end_point[0] + 2.5, self.end_point[1], self.end_point[2] + 1.5],
            textColorRGB=[1.0, 0.3, 0.3],
            textSize=2.0,
        )

    def _create_drone(self) -> int:
        """Simple bright-blue box drone, clearly visible."""
        body_half = [0.6, 0.5, 0.1]
        body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half)
        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=body_half,
            rgbaColor=[0.1, 0.25, 0.95, 1.0],  # bright blue
        )
        drone_id = p.createMultiBody(
            baseMass=1.3,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=self.start_pos.tolist(),
        )
        return drone_id

    # --------------------------------------------------
    # CAMERA HELPERS (for GTA-like control scripts)
    # --------------------------------------------------
    def set_camera_view_top(self):
        """Orthographic top view over entire map."""
        target = [0.0, 0.0, 0.0]
        p.resetDebugVisualizerCamera(
            cameraDistance=140.0,
            cameraYaw=0.0,
            cameraPitch=-89.0,
            cameraTargetPosition=target,
        )

    def set_camera_view_wide_city(self):
        """Nice 3/4 view of the city."""
        target = [0.0, 0.0, 10.0]
        p.resetDebugVisualizerCamera(
            cameraDistance=130.0,
            cameraYaw=35.0,
            cameraPitch=-35.0,
            cameraTargetPosition=target,
        )

    def set_camera_chase(self, drone_pos, yaw_deg: float):
        """
        GTA-like chase camera behind the drone.
        Your control code should call this with the drone's yaw.
        """
        target = [
            float(drone_pos[0]),
            float(drone_pos[1]),
            float(drone_pos[2]),
        ]
        p.resetDebugVisualizerCamera(
            cameraDistance=20.0,
            cameraYaw=float(yaw_deg),
            cameraPitch=-23.0,
            cameraTargetPosition=target,
        )

    def _ensure_hd_buffer(self, width: int = 1280, height: int = 720):
        try:
            p.getCameraImage(width, height)
        except Exception:
            pass

    # --------------------------------------------------
    # HUD (distance + collision counters)
    # --------------------------------------------------
    def init_hud(self):
        """Create (or reset) onscreen Distance + Collisions text."""
        self.distance_text_id = p.addUserDebugText(
            "Distance: 0.00 m",
            [0, self.pad_half_y - 10.0, 15.0],
            textColorRGB=[0.1, 0.2, 1.0],
            textSize=2.0,
        )
        self.collision_text_id = p.addUserDebugText(
            "Collisions: 0",
            [0, self.pad_half_y - 20.0, 15.0],
            textColorRGB=[1.0, 0.1, 0.1],
            textSize=2.0,
        )

    def update_hud(self, distance: float, collisions: int):
        """Call this from your control loop to update the HUD."""
        try:
            p.removeUserDebugItem(self.distance_text_id)
        except Exception:
            pass
        try:
            p.removeUserDebugItem(self.collision_text_id)
        except Exception:
            pass

        self.distance_text_id = p.addUserDebugText(
            f"Distance: {distance:.2f} m",
            [0, self.pad_half_y - 10.0, 15.0],
            textColorRGB=[0.1, 0.2, 1.0],
            textSize=2.0,
        )
        self.collision_text_id = p.addUserDebugText(
            f"Collisions: {collisions}",
            [0, self.pad_half_y - 20.0, 15.0],
            textColorRGB=[1.0, 0.1, 0.1],
            textSize=2.0,
        )

    # --------------------------------------------------
    # OBSTACLES FOR PATH PLANNING
    # --------------------------------------------------
    def get_obstacles_2d(self):
        """
        Return list of (x, y, radius) obstacles for 2D A* planner.

        Includes:
          - buildings
          - bridges
          - trees
          - cell towers
        """
        obs: List[Tuple[float, float, float]] = []

        for info in self.building_info:
            obs.append((info["x"], info["y"], info["radius"]))

        for info in self.bridge_info:
            obs.append((info["x"], info["y"], info["radius"]))

        for (x, y) in self.tree_positions:
            obs.append((x, y, 1.2))

        for (x, y) in self.tower_positions:
            obs.append((x, y, 1.6))

        return obs

    # --------------------------------------------------
    # STATE, STEP, CLOSE
    # --------------------------------------------------
    def get_drone_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang = p.getBaseVelocity(self.drone_id)
        return {
            "pos": np.array(pos, dtype=np.float32),
            "orn": np.array(orn, dtype=np.float32),
            "vel": np.array(vel, dtype=np.float32),
            "ang_vel": np.array(ang, dtype=np.float32),
        }

    def get_states(self):
        """
        Backwards-compatible: return [x, y, z, vx, vy, vz] as (1, 6).
        """
        s = self.get_drone_state()
        x, y, z = s["pos"]
        vx, vy, vz = s["vel"]
        return np.array([[x, y, z, vx, vy, vz]], dtype=np.float32)

    def step(self):
        p.stepSimulation()
        time.sleep(self.dt)

    def reset_scene(self):
        """Rebuild city with new random layout and reset HUD."""
        self._build_scene()
        self.init_hud()
        return self.drone_id

    def close(self):
        if p.isConnected():
            p.disconnect()
