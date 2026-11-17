"""2D football trajectory simulator (Part 1): Euler and RK4, quadratic drag, bouncing ground."""
import numpy as np
from scipy.optimize import brentq

# ---- Parameters from the assignment ----
M = 0.43          # kg
R = 0.11          # m
CD = 0.25
RHO = 1.2         # kg/m^3
G = 9.81          # m/s^2
E = 0.7           # coefficient of restitution (vertical velocity only)
AREA = np.pi * R**2
K = 0.5 * RHO * CD * AREA   # drag constant: F_drag = K * |v| * v


def deriv(s, drag=True):
    """Time derivative of state s = [x, y, vx, vy]."""
    x, y, vx, vy = s
    if drag:
        speed = np.hypot(vx, vy)
        ax = -(K / M) * speed * vx #Acceleration in x direction due to drag
        ay = -G - (K / M) * speed * vy #Acceleration in y direction due to drag and gravity
    else:
        ax, ay = 0.0, -G #Acceleration in x direction is 0 and in y direction is -G
    return np.array([vx, vy, ax, ay])


def euler_step(s, dt, drag=True):
    return s + dt * deriv(s, drag)


def rk4_step(s, dt, drag=True):
    k1 = deriv(s, drag)
    k2 = deriv(s + 0.5 * dt * k1, drag)
    k3 = deriv(s + 0.5 * dt * k2, drag)
    k4 = deriv(s + dt * k3, drag)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4) #RK4 is the weighted average of the four slopes, with k1 and k4 having weight 1 and k2 and k3 having weight 2


STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def _time_to_ground(s, dt, step, drag):
    """Bisection on partial step h in [0, dt] so that y(h) = 0, using the SAME integrator
    (keeps the method's convergence order intact)."""
    lo, hi = 0.0, dt
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if step(s, mid, drag)[1] > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) #Bisection method to find the time when the ball hits the ground (y=0) within the time step dt. It narrows down the interval [lo, hi] until it finds a time mid where y(mid) is close to 0.


def simulate(v0, theta, dt, method="rk4", drag=True, e=E,
             t_max=30.0, stop_at_first_landing=True, max_bounces=10):
    """Simulate a kicked ball launched from the origin.

    v0 : launch speed (m/s), theta : launch angle (radians), dt : time step (s)
    Returns dict with arrays t, x, y, vx, vy and, if landed, 'range' and 'flight_time'.
    """
    step = STEPPERS[method]
    s = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
    t = 0.0
    ts, states = [t], [s.copy()]
    landing = None
    bounces = 0

    while t < t_max:
        s_new = step(s, dt, drag)

        if s_new[1] < 0.0:
            # crossed the ground during this step: take partial step to y = 0
            h = _time_to_ground(s, dt, step, drag)
            s_hit = step(s, h, drag)
            s_hit[1] = 0.0
            t += h
            ts.append(t); states.append(s_hit.copy())

            if landing is None:
                landing = (s_hit[0], t)            # (range, flight time) of first landing
            bounces += 1
            if stop_at_first_landing or bounces >= max_bounces:
                break

            s_hit[3] = -e * s_hit[3]               # bounce: reverse vertical velocity * e
            s = s_hit
            continue                                # next iteration steps from the bounce point

        s = s_new
        t += dt
        ts.append(t); states.append(s.copy())

    arr = np.array(states)
    out = {"t": np.array(ts), "x": arr[:, 0], "y": arr[:, 1],
           "vx": arr[:, 2], "vy": arr[:, 3]}
    if landing is not None:
        out["range"], out["flight_time"] = landing
    return out


def launch_speed_for_range(D, theta, dt=0.01, method="rk4", drag=True, v_lo=0.5, v_hi=100.0):
    """Launch speed (m/s) that makes the first landing exactly D metres away at angle theta.
    Root-finds range(v0) - D = 0 with Brent's method (range increases with v0 at fixed theta)."""
    def miss(v0):
        return simulate(v0, theta, dt, method=method, drag=drag)["range"] - D
    return brentq(miss, v_lo, v_hi, xtol=1e-10)


def flight_time_for_range(D, theta, dt=0.01, method="rk4", drag=True):
    """(launch speed, flight time) of the ball that lands D metres away at angle theta."""
    v0 = launch_speed_for_range(D, theta, dt=dt, method=method, drag=drag)
    return v0, simulate(v0, theta, dt, method=method, drag=drag)["flight_time"]


def energy(sim):
    """Total mechanical energy at each saved point (for the Part 1 verification later)."""
    v2 = sim["vx"]**2 + sim["vy"]**2
    return 0.5 * M * v2 + M * G * sim["y"]


if __name__ == "__main__":
    v0, th = 25.0, np.radians(35)
    for m in ("euler", "rk4"):
        r = simulate(v0, th, dt=0.01, method=m, drag=False)
        print(f"{m:5s} no drag: range={r['range']:.4f}  T={r['flight_time']:.4f}")
    print("analytic:", v0**2 * np.sin(2 * th) / G, 2 * v0 * np.sin(th) / G)
    r = simulate(v0, th, dt=0.01, method="rk4", drag=True)
    print(f"rk4 with drag: range={r['range']:.3f}  T={r['flight_time']:.3f}")
