import numpy as np
import random


class RRTPlanner:

    def __init__(self, start, goal, obstacles, x_lim, y_lim):

        self.start = np.array(start[:2])
        self.goal = np.array(goal[:2])
        self.obstacles = obstacles

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.step_size = 2.0
        self.goal_sample_rate = 0.1
        self.max_iter = 500

        self.nodes = [self.start]
        self.parents = {tuple(self.start): None}

    # --------------------------------------------------
    # DISTANCE
    # --------------------------------------------------
    def _dist(self, a, b):
        return np.linalg.norm(a - b)

    # --------------------------------------------------
    # NEAREST NODE
    # --------------------------------------------------
    def _nearest(self, point):
        return min(self.nodes, key=lambda n: self._dist(n, point))

    # --------------------------------------------------
    # COLLISION CHECK
    # --------------------------------------------------
    def _collision(self, point):

        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(point - np.array([ox, oy])) < r:
                return True

        return False

    # --------------------------------------------------
    # STEER
    # --------------------------------------------------
    def _steer(self, from_node, to_point):

        direction = to_point - from_node
        length = np.linalg.norm(direction)

        if length == 0:
            return from_node

        direction = direction / length

        new_point = from_node + direction * self.step_size
        return new_point

    # --------------------------------------------------
    # PLAN
    # --------------------------------------------------
    def plan(self):

        for _ in range(self.max_iter):

            if random.random() < self.goal_sample_rate:
                sample = self.goal
            else:
                sample = np.array([
                    random.uniform(*self.x_lim),
                    random.uniform(*self.y_lim)
                ])

            nearest = self._nearest(sample)
            new_node = self._steer(nearest, sample)

            if self._collision(new_node):
                continue

            self.nodes.append(new_node)
            self.parents[tuple(new_node)] = nearest

            # reached goal
            if self._dist(new_node, self.goal) < self.step_size:
                return self._extract_path(new_node)

        return None

    # --------------------------------------------------
    # EXTRACT PATH
    # --------------------------------------------------
    def _extract_path(self, node):

        path = [node]

        while self.parents[tuple(node)] is not None:
            node = self.parents[tuple(node)]
            path.append(node)

        path.reverse()
        return path