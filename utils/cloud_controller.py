# utils/cloud_controller.py
# Lightweight cloud-controller helper used by manual_control_v2.py and collectors.
# This file provides:
# - compute_straight_path(start, goal, num_points=100)
# - compute_rrt_star_path(...) which will call rrt_star functions if available,
#   otherwise falls back to a straight-line path.
#
# Written to be robust to module reorganization.

import numpy as np

# Try to import RRT* planners if available under rrt_star package
_compute_rrt3d = None
try:
    # prefer the packaged rrt_star.rrt_star_3d if present
    from rrt_star.rrt_star_3d import compute_path_rrt_star_3d as _compute_rrt3d
except Exception:
    try:
        # fallback to top-level module name if that exists
        from rrt_star_3d import compute_path_rrt_star_3d as _compute_rrt3d
    except Exception:
        _compute_rrt3d = None

def compute_straight_path(start_xyz, goal_xyz, num_points=100):
    """
    Return a straight-line 3D path (list of [x,y,z]) from start to goal.
    start_xyz, goal_xyz: tuples or arrays of length 3
    num_points: number of intermediate waypoints (including start+goal)
    """
    s = np.asarray(start_xyz, dtype=float)
    g = np.asarray(goal_xyz, dtype=float)
    if num_points < 2:
        return [s.tolist(), g.tolist()]
    pts = np.linspace(s, g, num_points)
    return [tuple(p.tolist()) for p in pts]

def compute_rrt_star_path(start_xyz, goal_xyz, obstacles=None, bounds=None, **kwargs):
    """
    Wrapper to call RRT* 3D planner if available. Parameters:
      - start_xyz: (x,y,z)
      - goal_xyz: (x,y,z)
      - obstacles: list of obstacles in format expected by planner (or None)
      - bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None
      - kwargs: passed to planner if used

    If planner not available, returns a straight-line path.
    """
    if _compute_rrt3d is not None:
        try:
            path = _compute_rrt3d(
                start_xyz=start_xyz,
                goal_xyz=goal_xyz,
                obstacles=obstacles,
                bounds=bounds,
                **kwargs
            )
            # ensure path is a list of (x,y,z)
            return [tuple(map(float, p)) for p in path]
        except Exception as e:
            # if planner fails for any reason, fall back gracefully
            print("[cloud_controller] rrt_star call failed, falling back to straight path:", e)
            return compute_straight_path(start_xyz, goal_xyz, num_points=100)
    else:
        # planner not present — fallback to straight-line
        return compute_straight_path(start_xyz, goal_xyz, num_points=100)

# Backwards-compatible aliases (some code calls compute_straight_path directly)
def compute_path_astar(*args, **kwargs):
    """
    Placeholder for A* style path. For now, use straight path.
    """
    # Keep signature flexible; simply forward to straight path
    if len(args) >= 2:
        return compute_straight_path(args[0], args[1], num_points=100)
    return []

