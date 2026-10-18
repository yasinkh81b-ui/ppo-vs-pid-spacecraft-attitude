"""
PID attitude controller (quaternion error)
Uses satellite_sim.py (must be in the same folder).

Control law:
  q_e   = q_d^-1 (x) q          (error quaternion, body vs. desired)
  tau   = -Kp * sign(q_e0) * q_e_vec  - Kd * w  - Ki * integral(sign(q_e0) * q_e_vec)
The sign(q_e0) term picks the shorter rotation (q and -q are the same attitude).
"""
import numpy as np
from satellite_sim import simulate, quat_mult, J

TAU_MAX = 0.05  # actuator limit (N*m) per axis - reaction wheels can't give infinite torque


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


class PID:
    def __init__(self, kp, kd, ki, q_desired, dt=0.01, e_on=None, i_max=0.5):
        self.kp, self.kd, self.ki = kp, kd, ki
        self.q_d = np.asarray(q_desired, dtype=float)
        self.dt = dt
        self.integral = np.zeros(3)
        self.e_on = e_on    # None = integrate always; else only when |e| < e_on (anti-windup)
        self.i_max = i_max  # clamp on the integral state

    def __call__(self, t, x):
        q, w = x[:4], x[4:]
        q_e = quat_mult(quat_conj(self.q_d), q)
        e = np.sign(q_e[0] if q_e[0] != 0 else 1.0) * q_e[1:]
        tau = -self.kp * e - self.kd * w - self.ki * self.integral
        not_saturated = np.all(np.abs(tau) < TAU_MAX)
        near_target = self.e_on is None or np.linalg.norm(e) < self.e_on
        if not_saturated and near_target:
            self.integral = np.clip(self.integral + e * self.dt, -self.i_max, self.i_max)
        return np.clip(tau, -TAU_MAX, TAU_MAX)


def attitude_error_deg(traj, q_d):
    """Total rotation angle (deg) between the satellite and the target."""
    errs = []
    for q in traj[:, :4]:
        q_e = quat_mult(quat_conj(q_d), q)
        errs.append(np.degrees(2 * np.arccos(np.clip(abs(q_e[0]), 0, 1))))
    return np.array(errs)


if __name__ == "__main__":
    # Start: rotated 90 deg about (1,1,0)/sqrt2, small spin; target: identity attitude
    ang = np.radians(90)
    ax_ = np.array([1, 1, 0]) / np.sqrt(2)
    q0 = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax_)])
    x0 = np.concatenate([q0, [0.02, -0.01, 0.01]])
    q_d = np.array([1.0, 0, 0, 0])

    # Constant disturbance torque (e.g. solar pressure / magnetic) - this is where the I term matters
    d = np.array([0.002, -0.001, 0.0015])

    results = {}
    configs = [
        ("PD",                         dict(ki=0.0)),
        ("PID naive (windup)",         dict(ki=0.01)),
        ("PID + anti-windup",          dict(ki=0.02, e_on=0.1)),
    ]
    for name, cfg in configs:
        ctrl = PID(kp=0.1, kd=0.15, q_desired=q_d, **cfg)
        t, traj = simulate(x0, lambda t, x: ctrl(t, x) + d, t_end=90.0, dt=0.01)
        err = attitude_error_deg(traj, q_d)
        idx = np.where(err > 1.0)[0]
        print(f"{name}: final error = {err[-1]:.3f} deg, "
              f"settling to 1 deg = {t[idx[-1]] if len(idx) else 0.0:.1f} s, "
              f"max error after t=15s = {err[int(15/0.01):].max():.2f} deg")
        results[name] = (t, err)

    try:
        import matplotlib.pyplot as plt
        for name, (t, err) in results.items():
            plt.plot(t, err, label=name)
        plt.xlabel("time (s)"); plt.ylabel("attitude error (deg)")
        plt.legend(); plt.grid(True); plt.show()
    except ImportError:
        pass
