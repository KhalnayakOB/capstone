import numpy as np
import random


class Node:

    def __init__(self, position):

        self.position = np.array(position)
        self.parent = None
        self.cost = 0


class RRTStar3D:

    def __init__(
        self,
        start,
        goal,
        obstacles,
        x_limits,
        y_limits,
        z_limits,
        step_size=2.5,
        goal_sample_rate=0.1,
        max_iter=2000,
        search_radius=5.0,
    ):

        self.start = Node(start)
        self.goal = Node(goal)

        self.obstacles = obstacles

        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits

        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.search_radius = search_radius

        self.nodes = [self.start]

    # --------------------------------------------------
    # MAIN PLANNER
    # --------------------------------------------------

    def plan(self):

        for _ in range(self.max_iter):

            rnd = self.sample()

            nearest = self.get_nearest_node(rnd)

            new = self.steer(nearest, rnd)

            if not self.collision(new.position):

                neighbors = self.get_neighbors(new)

                new = self.choose_parent(new, neighbors)

                self.nodes.append(new)

                self.rewire(new, neighbors)

                if self.goal_reached(new):

                    return self.generate_path(new)

        return None

    # --------------------------------------------------
    # RANDOM SAMPLE
    # --------------------------------------------------

    def sample(self):

        if random.random() < self.goal_sample_rate:
            return self.goal.position

        x = random.uniform(*self.x_limits)
        y = random.uniform(*self.y_limits)
        z = random.uniform(*self.z_limits)

        return np.array([x, y, z])

    # --------------------------------------------------
    # NEAREST NODE
    # --------------------------------------------------

    def get_nearest_node(self, point):

        distances = [
            np.linalg.norm(node.position - point)
            for node in self.nodes
        ]

        return self.nodes[np.argmin(distances)]

    # --------------------------------------------------
    # STEER
    # --------------------------------------------------

    def steer(self, from_node, to_point):

        direction = to_point - from_node.position
        dist = np.linalg.norm(direction)

        if dist > self.step_size:
            direction = direction / dist * self.step_size

        new_pos = from_node.position + direction

        new_node = Node(new_pos)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(direction)

        return new_node

    # --------------------------------------------------
    # COLLISION CHECK
    # --------------------------------------------------

    def collision(self, point):

        for (ox, oy, r) in self.obstacles:

            d = np.sqrt((point[0] - ox) ** 2 + (point[1] - oy) ** 2)

            if d < r + 1.0:
                return True

        return False

    # --------------------------------------------------
    # NEIGHBORS
    # --------------------------------------------------

    def get_neighbors(self, node):

        neighbors = []

        for n in self.nodes:

            if np.linalg.norm(n.position - node.position) < self.search_radius:
                neighbors.append(n)

        return neighbors

    # --------------------------------------------------
    # CHOOSE BEST PARENT
    # --------------------------------------------------

    def choose_parent(self, node, neighbors):

        best = node.parent
        best_cost = node.cost

        for n in neighbors:

            d = np.linalg.norm(n.position - node.position)
            cost = n.cost + d

            if cost < best_cost:

                node.parent = n
                node.cost = cost

                best_cost = cost

        return node

    # --------------------------------------------------
    # REWIRE
    # --------------------------------------------------

    def rewire(self, node, neighbors):

        for n in neighbors:

            d = np.linalg.norm(n.position - node.position)

            new_cost = node.cost + d

            if new_cost < n.cost:

                n.parent = node
                n.cost = new_cost

    # --------------------------------------------------
    # GOAL CHECK
    # --------------------------------------------------

    def goal_reached(self, node):

        return np.linalg.norm(node.position - self.goal.position) < self.step_size

    # --------------------------------------------------
    # PATH EXTRACTION
    # --------------------------------------------------

    def generate_path(self, node):

        path = []

        while node is not None:

            path.append(node.position)
            node = node.parent

        path.reverse()

        return path