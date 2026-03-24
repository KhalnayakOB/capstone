import pybullet as p
import numpy as np

from planners.rrt_star_3d import RRTStar3D
from planners.path_smoother import shortcut_smooth
from physics.drone_dynamics import DroneDynamics


class MultiUAVController:

    def __init__(self, env, drone_ids):

        self.env = env
        self.client_id = env.client_id
        self.drone_ids = drone_ids

        self.autopilot = False

        # ------------------------------
        # FLIGHT PARAMETERS
        # ------------------------------

        self.max_speed = 12
        self.arrival_radius = 0.6
        self.avoid_distance = 2.5
        self.repulsion_gain = 2.5

        # ------------------------------
        # MAP LIMITS
        # ------------------------------

        self.x_limits = (-25, 25)
        self.y_limits = (-25, 25)
        self.z_limits = (2, 12)

        # obstacles from environment
        self.obstacles = env.get_obstacles_2d()

        # ------------------------------
        # PATH DATA
        # ------------------------------

        self.paths = []
        self.path_index = []

        # ------------------------------
        # RRT VISUALIZATION
        # ------------------------------

        self.debug_lines = []
        self.show_rrt = False

        # ------------------------------
        # DRONE DYNAMICS
        # ------------------------------

        self.dynamics = []

        for i, drone in enumerate(self.drone_ids):

            if i == 0:
                self.dynamics.append(None)
            else:
                self.dynamics.append(DroneDynamics(drone))

        self._initialize_paths()

    # --------------------------------------------------

    def _initialize_paths(self):

        for drone in self.drone_ids:

            pos, _ = p.getBasePositionAndOrientation(drone)
            start = np.array(pos)

            goal = start + np.array([20, 0, 3])

            planner = RRTStar3D(
                start,
                goal,
                self.obstacles,
                self.x_limits,
                self.y_limits,
                self.z_limits,
            )

            path = planner.plan()

            if path is None:
                path = [start, goal]
            else:
                path = shortcut_smooth(path, self.obstacles)

            self.paths.append(path)
            self.path_index.append(0)

    # --------------------------------------------------

    def toggle_autopilot(self):

        self.autopilot = not self.autopilot
        print("Autopilot:", self.autopilot)

    # --------------------------------------------------

    def toggle_rrt_visualization(self):

        self.show_rrt = not self.show_rrt

        for line in self.debug_lines:
            p.removeUserDebugItem(line)

        self.debug_lines.clear()

        if not self.show_rrt:
            return

        for path in self.paths:

            for i in range(len(path) - 1):

                a = path[i]
                b = path[i + 1]

                line = p.addUserDebugLine(
                    a.tolist(),
                    b.tolist(),
                    [0, 1, 0],
                    2
                )

                self.debug_lines.append(line)

    # --------------------------------------------------

    def _avoid_drones(self, drone_index, pos):

        repulsion = np.zeros(3)

        for i, other in enumerate(self.drone_ids):

            if i == drone_index:
                continue

            other_pos, _ = p.getBasePositionAndOrientation(other)
            other_pos = np.array(other_pos)

            diff = pos - other_pos
            dist = np.linalg.norm(diff)

            if dist < self.avoid_distance and dist > 0.001:
                repulsion += (diff / dist) * self.repulsion_gain

        return repulsion

    # --------------------------------------------------

    def _follow_path(self, drone_index):

        drone = self.drone_ids[drone_index]

        pos, _ = p.getBasePositionAndOrientation(drone)
        pos = np.array(pos)

        path = self.paths[drone_index]
        idx = self.path_index[drone_index]

        if idx >= len(path):
            idx = len(path) - 1

        target = path[idx]

        direction = target - pos
        dist = np.linalg.norm(direction)

        if dist < self.arrival_radius:
            self.path_index[drone_index] += 1
            return

        direction = direction / dist

        avoid = self._avoid_drones(drone_index, pos)

        velocity = direction * self.max_speed + avoid

        speed = np.linalg.norm(velocity)

        if speed > self.max_speed:
            velocity = velocity / speed * self.max_speed

        target_pos = pos + velocity * 0.5

        self.dynamics[drone_index].move_to(target_pos)

    # --------------------------------------------------

    def update_autopilot(self):

        if not self.autopilot:
            return

        for i in range(len(self.drone_ids)):

            if i == 0:
                continue

            self._follow_path(i)