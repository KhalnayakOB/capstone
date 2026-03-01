import pybullet as p
import pybullet_data
import numpy as np
import random
import time


class UAVCityEnv:

    """
    Medium-density 150x150m city environment for dataset generation.

    Features:
    - 150 x 150 m floor
    - Medium density of:
        * Buildings (boxes, varying footprint & height)
        * Trees (cylinders)
        * Cell towers (tall cylinders)
        * Bridges (elevated boxes)
    - Start/goal points on opposite X sides
    - Provides 2D obstacle list for path planners
    - Designed for DIRECT (no-GUI) use, but GUI works too
    """

    def __init__(self, gui: bool = False, dt: float = 1.0 / 240.0, seed: int | None = None):
        """
        gui: if True, shows PyBullet GUI, otherwise DIRECT mode.
        dt: simulation timestep (not critical for dataset generation).
        seed: optional random seed for reproducibility.
        """
        self.gui = gui
        self.dt = dt

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Connect to PyBullet
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)

        # Map half sizes (150 x 150 total)
        self.pad_half_x = 75.0
        self.pad_half_y = 75.0

        # Storage
        self.floor_id = None
        self.wall_ids: list[int] = []
        self.obstacle_ids: list[int] = []
        self.obstacles_2d: list[tuple[float, float, float]] = []

        # Start / goal points (3D)
        self.start_point = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.end_point = np.array([0.0, 0.0, 2.0], dtype=np.float32)

        # Create static world
        self._create_floor_and_walls()

        # Create first random city layout
        self.reset_scene()

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    def _create_floor_and_walls(self):
        """Create a large floor and boundary walls."""
        # Floor as a big box
        floor_half_extents = [self.pad_half_x, self.pad_half_y, 0.5]
        floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half_extents)
        floor_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=floor_half_extents,
            rgbaColor=[0.4, 0.4, 0.4, 1.0]
        )
        self.floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0.0, 0.0, -0.5],
        )

        # Create boundary walls (thin tall boxes)
        wall_thickness = 0.5
        wall_height = 10.0

        # +X wall
        wall_xp = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness, self.pad_half_y, wall_height / 2],
        )
        wall_xp_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness, self.pad_half_y, wall_height / 2],
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
        )
        self.wall_ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_xp,
                baseVisualShapeIndex=wall_xp_vis,
                basePosition=[self.pad_half_x + wall_thickness, 0.0, wall_height / 2],
            )
        )

        # -X wall
        wall_xn = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness, self.pad_half_y, wall_height / 2],
        )
        wall_xn_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness, self.pad_half_y, wall_height / 2],
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
        )
        self.wall_ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_xn,
                baseVisualShapeIndex=wall_xn_vis,
                basePosition=[-self.pad_half_x - wall_thickness, 0.0, wall_height / 2],
            )
        )

        # +Y wall
        wall_yp = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, wall_thickness, wall_height / 2],
        )
        wall_yp_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, wall_thickness, wall_height / 2],
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
        )
        self.wall_ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_yp,
                baseVisualShapeIndex=wall_yp_vis,
                basePosition=[0.0, self.pad_half_y + wall_thickness, wall_height / 2],
            )
        )

        # -Y wall
        wall_yn = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, wall_thickness, wall_height / 2],
        )
        wall_yn_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.pad_half_x, wall_thickness, wall_height / 2],
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
        )
        self.wall_ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_yn,
                baseVisualShapeIndex=wall_yn_vis,
                basePosition=[0.0, -self.pad_half_y - wall_thickness, wall_height / 2],
            )
        )

    # ------------------------------------------------------------------
    # Scene (city) generation
    # ------------------------------------------------------------------
    def reset_scene(self):
        """
        Clear existing obstacles, spawn a new medium-density city layout,
        and sample new start/goal points.
        """
        self._clear_dynamic_objects()
        self.obstacles_2d = []

        # 1) Sample start & goal points on opposite X sides
        self._sample_start_and_goal()

        # 2) Spawn obstacles
        self._spawn_buildings_medium()
        self._spawn_trees_medium()
        self._spawn_towers_medium()
        self._spawn_bridges_medium()

    def _clear_dynamic_objects(self):
        """Remove all existing obstacles (but keep floor & walls)."""
        for oid in self.obstacle_ids:
            try:
                p.removeBody(oid)
            except Exception:
                pass
        self.obstacle_ids.clear()

    def _sample_start_and_goal(self):
        """
        Place start on left side, goal on right side, with some random Y.
        Altitude ~2m.
        """
        margin_x = 5.0
        min_y = -self.pad_half_y + 10.0
        max_y = self.pad_half_y - 10.0

        y_start = random.uniform(min_y, max_y)
        y_goal = random.uniform(min_y, max_y)

        self.start_point = np.array(
            [-self.pad_half_x + margin_x, y_start, 2.0], dtype=np.float32
        )
        self.end_point = np.array(
            [self.pad_half_x - margin_x, y_goal, 2.0], dtype=np.float32
        )

    # ---------- Buildings ----------
    def _spawn_buildings_medium(self):
        """
        Medium density buildings:
          ~35–45 random buildings,
          varied footprints & heights.
        """
        num_buildings = random.randint(35, 45)

        for _ in range(num_buildings):
            # Random center position inside some margin from walls
            margin = 10.0
            x = random.uniform(-self.pad_half_x + margin, self.pad_half_x - margin)
            y = random.uniform(-self.pad_half_y + margin, self.pad_half_y - margin)

            # Avoid too close to start/goal
            if np.linalg.norm([x - self.start_point[0], y - self.start_point[1]]) < 10:
                continue
            if np.linalg.norm([x - self.end_point[0], y - self.end_point[1]]) < 10:
                continue

            # Random footprint and height
            half_w = random.uniform(2.0, 6.0)   # width/2
            half_d = random.uniform(2.0, 6.0)   # depth/2
            half_h = random.uniform(4.0, 15.0)  # height/2

            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[half_w, half_d, half_h],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[half_w, half_d, half_h],
                rgbaColor=[0.7, 0.7, 0.7, 1.0],
            )
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, half_h],
            )
            self.obstacle_ids.append(bid)

            # Approx 2D radius from center
            radius = float(np.sqrt(half_w**2 + half_d**2) + 1.0)
            self.obstacles_2d.append((x, y, radius))

    # ---------- Trees ----------
    def _spawn_trees_medium(self):
        """
        Medium amount of trees: ~20 scattered cylinders.
        """
        num_trees = random.randint(18, 24)
        trunk_radius = 0.3
        trunk_height = 2.0

        col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=trunk_radius,
            height=trunk_height,
        )

        for _ in range(num_trees):
            margin = 10.0
            x = random.uniform(-self.pad_half_x + margin, self.pad_half_x - margin)
            y = random.uniform(-self.pad_half_y + margin, self.pad_half_y - margin)

            if np.linalg.norm([x - self.start_point[0], y - self.start_point[1]]) < 6:
                continue
            if np.linalg.norm([x - self.end_point[0], y - self.end_point[1]]) < 6:
                continue

            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=trunk_radius,
                length=trunk_height,
                rgbaColor=[0.2, 0.8, 0.2, 1.0],
            )
            tid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, trunk_height / 2],
            )
            self.obstacle_ids.append(tid)

            # Slightly larger radius for 2D collision
            self.obstacles_2d.append((x, y, trunk_radius + 0.7))

    # ---------- Cell Towers ----------
    def _spawn_towers_medium(self):
        """
        Cell towers: tall thin cylinders.
        """
        num_towers = random.randint(4, 6)
        radius = 0.6
        height = 18.0

        col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )

        for _ in range(num_towers):
            margin = 15.0
            x = random.uniform(-self.pad_half_x + margin, self.pad_half_x - margin)
            y = random.uniform(-self.pad_half_y + margin, self.pad_half_y - margin)

            if np.linalg.norm([x - self.start_point[0], y - self.start_point[1]]) < 10:
                continue
            if np.linalg.norm([x - self.end_point[0], y - self.end_point[1]]) < 10:
                continue

            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=[0.8, 0.2, 0.2, 1.0],
            )
            tid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, height / 2],
            )
            self.obstacle_ids.append(tid)

            self.obstacles_2d.append((x, y, radius + 1.0))

    # ---------- Bridges ----------
    def _spawn_bridges_medium(self):
        """
        Bridges: elevated rectangular obstacles the UAV must fly around/under.
        """
        num_bridges = random.randint(2, 4)

        for _ in range(num_bridges):
            # Bridges span mostly across X or Y
            span_along_x = random.choice([True, False])

            if span_along_x:
                # Bridge across X, centered at some Y
                bridge_len = random.uniform(40.0, 70.0)
                bridge_width = random.uniform(3.0, 7.0)
                bridge_thickness = 1.0
                height = random.uniform(4.0, 8.0)

                y_center = random.uniform(
                    -self.pad_half_y + 20.0, self.pad_half_y - 20.0
                )
                x_center = 0.0  # spanning approx across middle

                half_extents = [bridge_len / 2, bridge_width / 2, bridge_thickness / 2]
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=[0.5, 0.5, 0.2, 1.0],
                )
                bid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x_center, y_center, height],
                )
                self.obstacle_ids.append(bid)

                # Approx 2D radius
                radius = float(np.sqrt((bridge_len / 2) ** 2 + (bridge_width / 2) ** 2))
                self.obstacles_2d.append((x_center, y_center, radius + 1.5))

            else:
                # Bridge across Y, centered at some X
                bridge_len = random.uniform(40.0, 70.0)
                bridge_width = random.uniform(3.0, 7.0)
                bridge_thickness = 1.0
                height = random.uniform(4.0, 8.0)

                x_center = random.uniform(
                    -self.pad_half_x + 20.0, self.pad_half_x - 20.0
                )
                y_center = 0.0

                half_extents = [bridge_width / 2, bridge_len / 2, bridge_thickness / 2]
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=[0.5, 0.5, 0.2, 1.0],
                )
                bid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x_center, y_center, height],
                )
                self.obstacle_ids.append(bid)

                radius = float(np.sqrt((bridge_len / 2) ** 2 + (bridge_width / 2) ** 2))
                self.obstacles_2d.append((x_center, y_center, radius + 1.5))

    # ------------------------------------------------------------------
    # Interface for planners
    # ------------------------------------------------------------------
    def get_obstacles_2d(self) -> list[tuple[float, float, float]]:
        """
        Return list of (x, y, radius) for use in 2D/3D path planners.
        Buildings, trees, towers, bridges are all approximated as circles in top view.
        """
        return list(self.obstacles_2d)

    def step_dummy(self, num_steps: int = 1):
        """
        Optionally step physics if needed (not required for dataset generation).
        """
        for _ in range(num_steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.dt)

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass
