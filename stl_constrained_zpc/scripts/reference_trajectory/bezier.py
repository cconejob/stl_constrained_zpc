import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d

def generate_cubic_bezier_control_points(current_state, goal_state, alpha=5.0):
    X0, Y0, cos0, sin0, v0 = current_state
    Xg, Yg, cosg, sing, vg = goal_state
    
    t0 = np.array([cos0, sin0])
    tg = np.array([cosg, sing])

    P0 = np.array([X0, Y0])
    P1 = P0 + (v0 / alpha) * t0
    P3 = np.array([Xg, Yg])
    P2 = P3 - (vg / alpha) * tg

    return [P0, P1, P2, P3]

def generate_fifth_bezier_control_points(current_state, goal_state, alpha=5.0, delta=0.5):
    X0, Y0, cos0, sin0, v0 = current_state
    Xg, Yg, cosg, sing, vg = goal_state

    t0 = np.array([cos0, sin0])
    tg = np.array([cosg, sing])

    P0 = np.array([X0, Y0])
    P1 = P0 + (v0 / alpha) * t0
    P2 = P1 + delta * t0
    P5 = np.array([Xg, Yg])
    P4 = P5 - (vg / alpha) * tg
    P3 = P4 - delta * tg

    return [P0, P1, P2, P3, P4, P5]

def bezier_curve(control_points, num_points=1000):
    n = len(control_points) - 1
    t_vals = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    
    for i in range(n + 1):
        curve += comb(n, i) * ((1 - t_vals) ** (n - i))[:, None] * (t_vals ** i)[:, None] * control_points[i]
    
    return curve, t_vals

def arc_length_resample(curve, velocities, freq):
    dt = 1.0 / freq
    N = len(velocities)

    # Compute cumulative arc length
    deltas = np.diff(curve, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Compute desired distances at each time step
    s_vals = np.cumsum(velocities * dt)
    s_vals = np.insert(s_vals, 0, 0.0)[:N]

    # Interpolate x, y over arc length
    interp_x = interp1d(arc_lengths, curve[:, 0], kind='linear')
    interp_y = interp1d(arc_lengths, curve[:, 1], kind='linear')
    
    x_new = interp_x(s_vals)
    y_new = interp_y(s_vals)
    return np.stack([x_new, y_new], axis=1)


def generate_bezier_with_velocities(current_state, goal_state, alpha=8.0, delta=0.5, N=100, freq=10, order=3):
    # Unpack current and goal states
    v0 = current_state[-1]; vg = goal_state[-1]

    # Control points
    if order == 3:
        control_points = generate_cubic_bezier_control_points(current_state, goal_state, alpha=alpha)
    else:
        control_points = generate_fifth_bezier_control_points(current_state, goal_state, alpha=alpha, delta=delta)

    # Generate dense curve
    dense_curve = bezier_curve(control_points, num_points=int((N+1) * freq))

    # Interpolate velocity from v0 to vg
    t_vals = np.linspace(0, (N+1)/freq, N+1)
    v = (2 * t_vals**3 - 3 * t_vals**2 + 1) * v0 + (-2 * t_vals**3 + 3 * t_vals**2) * vg

    # Generate dense Bézier
    dense_curve, _ = bezier_curve(control_points, num_points=100)

    # Resample based on velocity and freq
    sampled_curve = arc_length_resample(dense_curve, v, freq)

    # Compute yaw and speed based on real distances
    delta_pos = np.diff(sampled_curve, axis=0, prepend=sampled_curve[0:1])
    distances = np.linalg.norm(delta_pos, axis=1)
    v = distances * freq  # v = dx/dt with dt = 1/freq

    yaw = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    trajectory = np.column_stack([sampled_curve[:, 0], sampled_curve[:, 1], cos_yaw, sin_yaw, v])[1:]
    
    return trajectory
