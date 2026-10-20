"""
Week 1 - Spacecraft attitude simulator
State: x = [q0, q1, q2, q3, wx, wy, wz]
  q = [w, x, y, z] (scalar-first quaternion, body relative to inertial)
  w = angular velocity in the body frame (rad/s)
Dynamics:
  q_dot = 0.5 * q (x) [0, w]
  J w_dot = tau - w x (J w)        (Euler's rotation equations)
Integrator: RK4
"""
import numpy as np

# Inertia matrix of the satellite (kg*m^2) - diagonal, not symmetric on purpose
J = np.diag([0.05, 0.06, 0.04])
J_inv = np.linalg.inv(J)


def quat_mult(a, b):
    """Hamilton product a (x) b, both scalar-first."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def derivatives(x, tau):
    q, w = x[:4], x[4:]
    q_dot = 0.5 * quat_mult(q, np.array([0.0, *w]))
    w_dot = J_inv @ (tau - np.cross(w, J @ w))
    return np.concatenate([q_dot, w_dot])


def rk4_step(x, tau, dt):
    k1 = derivatives(x, tau)
    k2 = derivatives(x + 0.5 * dt * k1, tau)
    k3 = derivatives(x + 0.5 * dt * k2, tau)
    k4 = derivatives(x + dt * k3, tau)
    x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_new[:4] /= np.linalg.norm(x_new[:4])  # keep the quaternion unit-length
    return x_new


def simulate(x0, torque_fn, t_end=20.0, dt=0.01):
    n = int(t_end / dt)
    t = np.arange(n + 1) * dt
    traj = np.zeros((n + 1, 7))
    traj[0] = x0
    for i in range(n):
        traj[i + 1] = rk4_step(traj[i], torque_fn(t[i], traj[i]), dt)
    return t, traj


if __name__ == "__main__":
    # Test 1: no torque, initial spin -> angular momentum |J w| must stay constant
    x0 = np.array([1, 0, 0, 0, 0.3, 0.1, 0.05], dtype=float)
    t, traj = simulate(x0, lambda t, x: np.zeros(3))
    h = np.linalg.norm((J @ traj[:, 4:].T).T, axis=1)
    print("Angular momentum drift:", h.max() - h.min())
    print("Quaternion norm error :", abs(np.linalg.norm(traj[-1, :4]) - 1))

    # Test 2: constant torque about z from rest -> w_z should be tau/J_zz * t
    x0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
    t, traj = simulate(x0, lambda t, x: np.array([0, 0, 0.01]), t_end=5.0)
    print("w_z at t=5 :", traj[-1, 6], "(expected", 0.01 / 0.04 * 5.0, ")")

    try:
        import matplotlib.pyplot as plt
        x0 = np.array([1, 0, 0, 0, 0.3, 0.1, 0.05], dtype=float)
        t, traj = simulate(x0, lambda t, x: np.zeros(3))
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t, traj[:, :4]); ax[0].set_ylabel("quaternion")
        ax[1].plot(t, traj[:, 4:]); ax[1].set_ylabel("w (rad/s)")
        ax[1].set_xlabel("time (s)")
        plt.show()
    except ImportError:
        pass
