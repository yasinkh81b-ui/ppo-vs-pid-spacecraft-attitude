"""
Gymnasium environment for spacecraft attitude control (for PPO)
Needs satellite_sim.py and pid_control.py in the same folder.

Observation (6): [ e_x, e_y, e_z,  5*w_x, 5*w_y, 5*w_z ]
    e = sign(q_e0) * vector part of the error quaternion (0 when on target)
Action (3): torque command in [-1, 1], scaled by TAU_MAX (N*m)
Optional (use_integral=True): 3 extra observations = scaled integral of the error vector
    (accumulated only while |e| < 0.1, clamped) - gives PPO the "memory" that a PID's I-term has.
Reward: dense, negative cost = -( |e| + 0.2*|w| + 0.05*|u|^2 ), plus a small bonus when within ~1 degree
Episode: 400 control steps x 0.1 s = 40 s. Random start attitude, random spin,
         random constant disturbance torque (so PPO sees the same kind of problem as the PID).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from satellite_sim import rk4_step
from pid_control import quat_conj, quat_mult, TAU_MAX

SIM_DT = 0.02      # integrator step (s)
SUBSTEPS = 5       # -> control period = 0.1 s
MAX_STEPS = 400    # 40 s per episode
DT_CTRL = SIM_DT * SUBSTEPS
I_MAX = 0.5        # clamp on the error integral (same as the PID)
I_SCALE = 10.0     # scale of the integral in the observation
E_ON = 0.1         # integrate only when |e| < E_ON (same anti-windup rule as the PID)
Q_DESIRED = np.array([1.0, 0.0, 0.0, 0.0])


def error_vector(q, q_d=Q_DESIRED):
    q_e = quat_mult(quat_conj(q_d), q)
    s = 1.0 if q_e[0] >= 0 else -1.0
    return s * q_e[1:]


class AttitudeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, fixed_start=None, fixed_disturbance=None, use_integral=False):
        """
        fixed_start: optional state x0 (7,) for evaluation, otherwise random each episode
        fixed_disturbance: optional (3,) torque for evaluation, otherwise random each episode
        use_integral: add the (scaled) integral of the error to the observation (9 numbers instead of 6)
        """
        super().__init__()
        self.use_integral = use_integral
        self.integral = np.zeros(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9 if use_integral else 6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.fixed_start = None if fixed_start is None else np.asarray(fixed_start, float)
        self.fixed_d = None if fixed_disturbance is None else np.asarray(fixed_disturbance, float)
        self.x = None
        self.d = np.zeros(3)
        self.steps = 0

    def _obs(self):
        e = error_vector(self.x[:4])
        parts = [e, 5.0 * self.x[4:]]
        if self.use_integral:
            parts.append(I_SCALE * self.integral)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if self.fixed_start is not None:
            self.x = self.fixed_start.copy()
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            ang = np.radians(rng.uniform(20.0, 150.0))
            q0 = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])
            w0 = rng.uniform(-0.05, 0.05, size=3)
            self.x = np.concatenate([q0, w0])
        self.d = self.fixed_d.copy() if self.fixed_d is not None else rng.uniform(-0.003, 0.003, size=3)
        self.steps = 0
        self.integral = np.zeros(3)
        return self._obs(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, float), -1.0, 1.0)
        tau = u * TAU_MAX + self.d
        for _ in range(SUBSTEPS):
            self.x = rk4_step(self.x, tau, SIM_DT)
        self.steps += 1

        e = error_vector(self.x[:4])
        w = self.x[4:]
        e_norm = np.linalg.norm(e)                    # = sin(angle/2)
        if e_norm < E_ON:
            self.integral = np.clip(self.integral + e * DT_CTRL, -I_MAX, I_MAX)
        reward = -(e_norm + 0.2 * np.linalg.norm(w) + 0.05 * float(u @ u))
        if e_norm < 0.0087:                           # within ~1 degree
            reward += 0.1
        truncated = self.steps >= MAX_STEPS
        return self._obs(), float(reward), False, truncated, {}


def attitude_error_deg_from_state(x):
    return float(np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(error_vector(x[:4])), 0, 1))))
