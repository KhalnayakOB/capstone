import numpy as np
import random


def shortcut_smooth(path, obstacles, iterations=100):

    if path is None or len(path) < 3:
        return path

    path = list(path)

    for _ in range(iterations):

        if len(path) <= 2:
            break

        i = random.randint(0, len(path) - 2)
        j = random.randint(i + 1, len(path) - 1)

        if j == i + 1:
            continue

        if not collision_between(path[i], path[j], obstacles):

            new_path = []
            new_path.extend(path[:i + 1])
            new_path.append(path[j])
            new_path.extend(path[j + 1:])

            path = new_path

    return path


def collision_between(p1, p2, obstacles):

    steps = 10

    for i in range(steps):

        t = i / steps

        p = p1 * (1 - t) + p2 * t

        for (ox, oy, r) in obstacles:

            d = np.sqrt((p[0] - ox) ** 2 + (p[1] - oy) ** 2)

            if d < r + 1.0:
                return True

    return False