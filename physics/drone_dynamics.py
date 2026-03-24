import pybullet as p
import numpy as np


class DroneDynamics:

    def __init__(self, drone_id):

        self.drone_id = drone_id

        # physical parameters
        self.mass = 1.2
        self.max_thrust = 20

        # PID gains
        self.kp = 2.5
        self.kd = 1.2

        self.prev_error = np.zeros(3)

    # --------------------------------------------------
    # POSITION CONTROL
    # --------------------------------------------------

    def move_to(self, target):

        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, _ = p.getBaseVelocity(self.drone_id)

        pos = np.array(pos)
        vel = np.array(vel)

        error = target - pos
        d_error = error - self.prev_error

        self.prev_error = error

        # PD controller
        accel = self.kp * error + self.kd * d_error

        thrust = self.mass * accel

        thrust_mag = np.linalg.norm(thrust)

        if thrust_mag > self.max_thrust:
            thrust = thrust / thrust_mag * self.max_thrust

        p.applyExternalForce(
            self.drone_id,
            -1,
            thrust.tolist(),
            pos.tolist(),
            p.WORLD_FRAME
        )