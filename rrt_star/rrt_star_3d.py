import math
import random
from typing import List, Tuple, Optional


class Node3D:
    def __init__(self, x, y, z, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent     # index of parent in node list
        self.cost = cost         # cost from start to this node


class RRTStar3D:
    """
    Full 3D RRT* with:
      - goal bias
      - rewiring
      - 3D collision checks
      - 3D obstacles (treated as cylinders or boxes)
    """

    def __init__(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        obstacles: List[Tuple[float, float, float, float, float]],
        # (ox, oy, oz, radius_xy, height_z)
        step_size=3.0,
        goal_sample_rate=0.10,
        max_iter=3000,
        search_radius=10.0,
        goal_threshold=4.0,
    ):
        self.start = start
        self.goal = goal

        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.zmin, self.zmax = bounds[2]

        # obstacles = (ox, oy, oz, radius, height)
        self.obstacles = obstacles

        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.goal_threshold = goal_threshold

        self.nodes = [Node3D(start[0], start[1], start[2], parent=None, cost=0.0)]

    # -------------- UTILS -----------------

    def _dist(self, n1: Node3D, n2: Node3D):
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2)

    def _sample_point(self):
        # Goal bias
        if random.random() < self.goal_sample_rate:
            return self.goal
        
        return (
            random.uniform(self.xmin, self.xmax),
            random.uniform(self.ymin, self.ymax),
            random.uniform(self.zmin, self.zmax),
        )

    def _nearest(self, x, y, z):
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(self.nodes):
            d = math.dist((node.x, node.y, node.z), (x, y, z))
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _steer(self, from_node: Node3D, tx, ty, tz):
        dx = tx - from_node.x
        dy = ty - from_node.y
        dz = tz - from_node.z
        d = math.sqrt(dx*dx + dy*dy + dz*dz)

        if d <= self.step_size:
            return (tx, ty, tz)
        
        scale = self.step_size / d
        return (
            from_node.x + dx * scale,
            from_node.y + dy * scale,
            from_node.z + dz * scale
        )

    # -------------- COLLISION CHECKING (3D) -----------------

    def _point_collision(self, x, y, z):
        for ox, oy, oz, r_xy, h in self.obstacles:
            # Within vertical height?
            if oz <= z <= oz + h:
                # Horizontal collision
                if (x - ox)**2 + (y - oy)**2 <= (r_xy)**2:
                    return True
        return False

    def _segment_collision(self, x1, y1, z1, x2, y2, z2, resolution=1.0):
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        steps = max(int(dist / resolution), 1)

        for i in range(steps + 1):
            t = i / steps
            x = x1 + dx * t
            y = y1 + dy * t
            z = z1 + dz * t
            if self._point_collision(x, y, z):
                return True
        
        return False

    # -------------- NEIGHBORS -----------------

    def _neighbors(self, new_node: Node3D):
        idxs = []
        for i, node in enumerate(self.nodes):
            if self._dist(node, new_node) <= self.search_radius:
                idxs.append(i)
        return idxs

    # -------------- MAIN RRT* ALGORITHM -----------------

    def plan(self):
        goal_idx = None

        for _ in range(self.max_iter):

            # 1) sample
            sx, sy, sz = self._sample_point()

            # 2) nearest node
            nearest_idx = self._nearest(sx, sy, sz)
            nearest = self.nodes[nearest_idx]

            # 3) steer
            nx, ny, nz = self._steer(nearest, sx, sy, sz)

            # 4) bounds + collision
            if not (self.xmin <= nx <= self.xmax): continue
            if not (self.ymin <= ny <= self.ymax): continue
            if not (self.zmin <= nz <= self.zmax): continue
            if self._segment_collision(nearest.x, nearest.y, nearest.z,
                                       nx, ny, nz):
                continue

            # create node
            new_node = Node3D(nx, ny, nz)

            # 5) choose best parent
            near_ids = self._neighbors(new_node)
            best_parent = nearest_idx
            best_cost = nearest.cost + self._dist(nearest, new_node)

            for nid in near_ids:
                neighbor = self.nodes[nid]
                new_cost = neighbor.cost + self._dist(neighbor, new_node)
                if new_cost < best_cost:
                    if not self._segment_collision(neighbor.x, neighbor.y, neighbor.z,
                                                   nx, ny, nz):
                        best_cost = new_cost
                        best_parent = nid

            new_node.cost = best_cost
            new_node.parent = best_parent
            self.nodes.append(new_node)
            new_idx = len(self.nodes) - 1

            # 6) rewire
            for nid in near_ids:
                neighbor = self.nodes[nid]
                new_cost = new_node.cost + self._dist(neighbor, new_node)
                if new_cost < neighbor.cost:
                    if not self._segment_collision(neighbor.x, neighbor.y, neighbor.z,
                                                   neighbor.x, neighbor.y, neighbor.z):
                        neighbor.parent = new_idx
                        neighbor.cost = new_cost

            # 7) goal reached?
            if math.dist((nx, ny, nz), self.goal) <= self.goal_threshold:
                goal_idx = new_idx
                break

        if goal_idx is None:
            # choose closest to goal
            best = 0
            best_d = float("inf")
            gx, gy, gz = self.goal
            for i, node in enumerate(self.nodes):
                d = math.dist((node.x, node.y, node.z), (gx, gy, gz))
                if d < best_d:
                    best_d = d
                    best = i
            goal_idx = best

        return self._backtrack(goal_idx)

    def _backtrack(self, idx):
        path = []
        while idx is not None:
            n = self.nodes[idx]
            path.append((n.x, n.y, n.z))
            idx = n.parent
        return path[::-1]


def compute_path_rrt_star_3d(start_xyz, goal_xyz, obstacles, bounds):
    """
    obstacles = list of (ox, oy, oz, radius, height)
    bounds = ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    """
    planner = RRTStar3D(
        start=start_xyz,
        goal=goal_xyz,
        bounds=bounds,
        obstacles=obstacles,
        step_size=3.0,
        goal_sample_rate=0.10,
        max_iter=3500,
        search_radius=12.0,
        goal_threshold=5.0,
    )

    return planner.plan()
