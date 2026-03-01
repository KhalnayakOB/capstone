import math
import random
from typing import List, Tuple, Optional


class RRTStarNode:
    def __init__(self, x: float, y: float, parent: Optional[int] = None, cost: float = 0.0):
        self.x = x
        self.y = y
        self.parent = parent  # index of parent node in self.nodes list
        self.cost = cost      # cost from start to this node


class RRTStarPlanner:
    """
    2D RRT* planner with:
      - goal bias (goal_sample_rate)
      - rewiring in a search radius
      - obstacle-aware collision checking

    Obstacles are treated as circles: (x, y, radius).
    Bounds are ((xmin, xmax), (ymin, ymax)).
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
        step_size: float = 3.0,
        goal_sample_rate: float = 0.1,
        max_iter: int = 2000,
        search_radius: float = 10.0,
        goal_threshold: float = 3.0,
    ):
        self.start = start
        self.goal = goal
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.obstacles = obstacles

        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.goal_threshold = goal_threshold

        self.nodes: List[RRTStarNode] = []
        self.nodes.append(RRTStarNode(start[0], start[1], parent=None, cost=0.0))

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def _distance(self, n1: RRTStarNode, n2: RRTStarNode) -> float:
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def _sample_random_point(self) -> Tuple[float, float]:
        # Goal bias: with probability goal_sample_rate, sample goal directly
        if random.random() < self.goal_sample_rate:
            return self.goal
        x = random.uniform(self.xmin, self.xmax)
        y = random.uniform(self.ymin, self.ymax)
        return (x, y)

    def _nearest_node_index(self, x: float, y: float) -> int:
        # Euclidean nearest neighbor
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(self.nodes):
            d = math.hypot(node.x - x, node.y - y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _steer(
        self,
        from_node: RRTStarNode,
        to_x: float,
        to_y: float,
        step_size: Optional[float] = None,
    ) -> Tuple[float, float]:
        if step_size is None:
            step_size = self.step_size
        dx = to_x - from_node.x
        dy = to_y - from_node.y
        d = math.hypot(dx, dy)
        if d <= step_size:
            return (to_x, to_y)
        else:
            scale = step_size / d
            return (from_node.x + dx * scale, from_node.y + dy * scale)

    # ----------------------------------------------------------
    # Collision checking
    # ----------------------------------------------------------
    def _is_point_in_collision(self, x: float, y: float) -> bool:
        for ox, oy, r in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= (r + 0.5) ** 2:
                return True
        return False

    def _is_segment_in_collision(
        self, x1: float, y1: float, x2: float, y2: float, resolution: float = 1.0
    ) -> bool:
        """Check line segment collision with obstacles by sampling points."""
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            return self._is_point_in_collision(x1, y1)

        n_samples = max(int(dist / resolution), 1)
        for i in range(n_samples + 1):
            t = i / n_samples
            x = x1 + dx * t
            y = y1 + dy * t
            if self._is_point_in_collision(x, y):
                return True
        return False

    # ----------------------------------------------------------
    # Neighborhood
    # ----------------------------------------------------------
    def _near_nodes_indices(self, new_node: RRTStarNode) -> List[int]:
        indices = []
        for i, node in enumerate(self.nodes):
            d = self._distance(node, new_node)
            if d <= self.search_radius:
                indices.append(i)
        return indices

    # ----------------------------------------------------------
    # Main RRT* expansion
    # ----------------------------------------------------------
    def plan(self) -> List[Tuple[float, float]]:
        """
        Run RRT* and return a path as list of (x, y) from start to goal.
        Returns an empty list if no path found.
        """
        goal_reached_node_idx = None

        for it in range(self.max_iter):
            # 1) Sample
            rx, ry = self._sample_random_point()

            # 2) Nearest
            nearest_idx = self._nearest_node_index(rx, ry)
            nearest_node = self.nodes[nearest_idx]

            # 3) Steer
            new_x, new_y = self._steer(nearest_node, rx, ry, step_size=self.step_size)

            # 4) Bounds + collision
            if not (self.xmin <= new_x <= self.xmax and self.ymin <= new_y <= self.ymax):
                continue
            if self._is_segment_in_collision(nearest_node.x, nearest_node.y, new_x, new_y):
                continue

            new_node = RRTStarNode(new_x, new_y)

            # 5) Choose best parent among neighbors
            near_ids = self._near_nodes_indices(new_node)
            min_cost = nearest_node.cost + self._distance(nearest_node, new_node)
            best_parent = nearest_idx

            for nid in near_ids:
                neighbor = self.nodes[nid]
                if self._is_segment_in_collision(neighbor.x, neighbor.y, new_x, new_y):
                    continue
                c = neighbor.cost + self._distance(neighbor, new_node)
                if c < min_cost:
                    min_cost = c
                    best_parent = nid

            new_node.cost = min_cost
            new_node.parent = best_parent
            self.nodes.append(new_node)
            new_idx = len(self.nodes) - 1

            # 6) Rewire neighbors
            for nid in near_ids:
                neighbor = self.nodes[nid]
                new_cost = new_node.cost + self._distance(new_node, neighbor)
                if new_cost < neighbor.cost:
                    # Check if rewiring is collision free
                    if not self._is_segment_in_collision(
                        new_node.x, new_node.y, neighbor.x, neighbor.y
                    ):
                        neighbor.parent = new_idx
                        neighbor.cost = new_cost

            # 7) Check for goal proximity
            if math.hypot(new_x - self.goal[0], new_y - self.goal[1]) <= self.goal_threshold:
                # Found a node close to the goal
                goal_reached_node_idx = new_idx
                break

        if goal_reached_node_idx is None:
            # No node within goal_threshold: try best node anyway
            best_idx = None
            best_dist = float("inf")
            for i, node in enumerate(self.nodes):
                d = math.hypot(node.x - self.goal[0], node.y - self.goal[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is None:
                return []
            goal_reached_node_idx = best_idx

        # Backtrack path
        return self._backtrack_path(goal_reached_node_idx)

    def _backtrack_path(self, node_idx: int) -> List[Tuple[float, float]]:
        path: List[Tuple[float, float]] = []
        cur_idx = node_idx
        while cur_idx is not None:
            node = self.nodes[cur_idx]
            path.append((node.x, node.y))
            cur_idx = node.parent
        path.reverse()
        return path


def compute_rrt_star_path(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    obstacles: List[Tuple[float, float, float]],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    step_size: float = 3.0,
    goal_sample_rate: float = 0.1,
    max_iter: int = 2500,
    search_radius: float = 12.0,
    goal_threshold: float = 3.0,
) -> List[Tuple[float, float]]:
    """
    Convenience wrapper to create an RRTStarPlanner and run it.
    Returns a list of (x, y) waypoints from start to “near” goal.
    """
    planner = RRTStarPlanner(
        start=start_xy,
        goal=goal_xy,
        bounds=bounds,
        obstacles=obstacles,
        step_size=step_size,
        goal_sample_rate=goal_sample_rate,
        max_iter=max_iter,
        search_radius=search_radius,
        goal_threshold=goal_threshold,
    )
    path = planner.plan()
    return path
