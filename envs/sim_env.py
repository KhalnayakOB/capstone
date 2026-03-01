import time
from typing import List

import numpy as np
import pybullet as p
import pybullet_data


class UAVSimEnv:
    def __init__(self, num_uavs: int = 5, gui: bool = True, dt: float = 1 / 240):
        self.num_uavs = num_uavs
        self.dt = dt

        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Ground
        self.plane_id = p.loadURDF("plane.urdf")

        # Room size (square room)
        self.room_size = 5.0
        self._create_room()

        # Drones
        self.uav_ids = self._create_uavs()

        # Obstacles (nicer, mixed shapes)
        self.obstacle_ids = self._create_obstacles()

    def _create_room(self):
        """Create 4 walls forming a simple indoor room."""
        thickness = 0.1
        height = 3.0
        z_center = height / 2.0

        # [half_x, half_y, half_z] for each wall
        walls = [
            # +Y wall
            ([self.room_size, thickness, height / 2.0], [0, self.room_size, z_center]),
            # -Y wall
            ([self.room_size, thickness, height / 2.0], [0, -self.room_size, z_center]),
            # +X wall
            ([thickness, self.room_size, height / 2.0], [self.room_size, 0, z_center]),
            # -X wall
            ([thickness, self.room_size, height / 2.0], [-self.room_size, 0, z_center]),
        ]

        for half_extents, pos in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.8, 0.8, 0.8, 1.0],
            )
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
            )

    def _create_uavs(self) -> List[int]:
        """Create small box-shaped UAVs at random positions."""
        uav_ids: List[int] = []

        half_extents = [0.15, 0.15, 0.05]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.4, 0.8, 1.0],  # blue-ish drones
        )

        for _ in range(self.num_uavs):
            x = float(np.random.uniform(-2.0, 2.0))
            y = float(np.random.uniform(-2.0, 2.0))
            z = float(np.random.uniform(1.0, 2.0))

            uid = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, z],
            )
            uav_ids.append(uid)

        return uav_ids

    def _create_obstacles(self) -> List[int]:
        """
        Create nicer-looking mixed obstacles:
        - two tall cylinder pillars
        - a central 'gate' made of two posts + top beam
        - a couple of chunky blocks
        """
        obstacle_ids: List[int] = []

        # 1) Tall cylinder pillars
        cyl_radius = 0.25
        cyl_height = 2.2
        cyl_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=cyl_radius, height=cyl_height
        )
        cyl_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=cyl_radius,
            length=cyl_height,
            rgbaColor=[0.9, 0.7, 0.2, 1.0],  # golden-ish pillars
        )

        cyl_positions = [
            [1.5, -1.5, cyl_height / 2.0],
            [-1.5, 1.5, cyl_height / 2.0],
        ]

        for pos in cyl_positions:
            oid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=cyl_col,
                baseVisualShapeIndex=cyl_vis,
                basePosition=pos,
            )
            obstacle_ids.append(oid)

        # 2) Central "gate" (two posts + one top beam)
        post_half = [0.15, 0.15, 0.8]
        beam_half = [0.8, 0.15, 0.15]

        post_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=post_half)
        beam_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=beam_half)

        post_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=post_half,
            rgbaColor=[0.3, 0.9, 0.3, 1.0],  # green posts
        )
        beam_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=beam_half,
            rgbaColor=[0.3, 0.7, 0.9, 1.0],  # bluish beam
        )

        # Posts (like a doorway)
        gate_y = 0.0
        gate_x_offset = 1.0
        z_post_center = post_half[2]

        left_post_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=post_col,
            baseVisualShapeIndex=post_vis,
            basePosition=[-gate_x_offset, gate_y, z_post_center],
        )
        right_post_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=post_col,
            baseVisualShapeIndex=post_vis,
            basePosition=[gate_x_offset, gate_y, z_post_center],
        )

        obstacle_ids.append(left_post_id)
        obstacle_ids.append(right_post_id)

        # Top beam (connecting posts)
        z_beam_center = 2 * post_half[2] + beam_half[2] * 0.5
        beam_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=beam_col,
            baseVisualShapeIndex=beam_vis,
            basePosition=[0.0, gate_y, z_beam_center],
        )
        obstacle_ids.append(beam_id)

        # 3) A couple of chunky blocks on the floor
        block_half = [0.4, 0.4, 0.3]
        block_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half)
        block_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=block_half,
            rgbaColor=[0.9, 0.4, 0.2, 1.0],  # orange blocks
        )

        block_positions = [
            [0.0, 2.0, block_half[2]],
            [-2.0, -1.0, block_half[2]],
        ]

        for pos in block_positions:
            oid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=block_col,
                baseVisualShapeIndex=block_vis,
                basePosition=pos,
            )
            obstacle_ids.append(oid)

        return obstacle_ids

    def get_states(self):
        """Return a list of dicts with state for each UAV."""
        states = []
        for uid in self.uav_ids:
            pos, orn = p.getBasePositionAndOrientation(uid)
            lin_vel, ang_vel = p.getBaseVelocity(uid)

            states.append(
                {
                    "pos": np.array(pos, dtype=float),
                    "vel": np.array(lin_vel, dtype=float),
                    "orn": np.array(orn, dtype=float),
                    "ang_vel": np.array(ang_vel, dtype=float),
                }
            )

        return states

    def get_obstacles(self):
        """Return list of obstacle positions as numpy arrays."""
        positions = []
        for oid in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(oid)
            positions.append(np.array(pos, dtype=float))
        return positions

    def step(self, controls):
        """
        Apply 3D force vectors to each UAV and advance the simulation.

        controls: list of [Fx, Fy, Fz], one per UAV.
        """
        assert len(controls) == self.num_uavs

        for uid, force in zip(self.uav_ids, controls):
            force = np.array(force, dtype=float).tolist()
            p.applyExternalForce(
                objectUniqueId=uid,
                linkIndex=-1,
                forceObj=force,
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
            )

        p.stepSimulation()
        time.sleep(self.dt)

    def close(self):
        p.disconnect()
