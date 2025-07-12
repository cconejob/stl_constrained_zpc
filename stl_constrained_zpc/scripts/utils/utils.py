import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import pymap3d
import os

from stl_constrained_zpc.scripts.reachability.Zonotope import Zonotope
from stl_constrained_zpc.scripts.reachability.Interval import Interval


def plot_states_inputs(x_meas_vec_0, x_meas_vec_1, u):
    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot all trajectory points
    ax.plot(x_meas_vec_0[0, :], x_meas_vec_0[1, :], 'r', label='Initial', marker='o', linestyle='None')
    ax.plot(x_meas_vec_1[0, :], x_meas_vec_1[1, :], 'b', label='Final', marker='x', linestyle='None')
    ax.legend()
    ax.set_title('States')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Equal aspect ratio
    ax.axis('equal')

    # Plot the initial and final points of the yaw, speed and yaw rate
    _, axs = plt.subplots(3, 2, figsize=(10, 10))

    axs[0, 0].plot(x_meas_vec_0[2, :], 'r')
    axs[0, 0].set_title('Cos Yaw Initial')
    axs[0, 1].plot(x_meas_vec_1[2, :], 'b')
    axs[0, 1].set_title('Cos Yaw Final')

    axs[1, 0].plot(x_meas_vec_0[4, :], 'r')
    axs[1, 0].set_title('Speed Initial')
    axs[1, 1].plot(x_meas_vec_1[4, :], 'b')
    axs[1, 1].set_title('Speed Final')

    axs[2, 0].plot(u[0, :], 'r')
    axs[2, 0].set_title('Steering')
    axs[2, 1].plot(u[1, :], 'b')
    axs[2, 1].set_title('Throttle')

    plt.show()

def plot_vehicle(current_state, next_states=None, ax=None, control_command=None, alpha=0.5, failure_mode=False):
    """
    Plot the vehicle rectangle representing the vehicle and the reachable sets.

    Args:
        current_state (np.array): Current state of the vehicle. [x, y, cos_yaw, sin_yaw, v]
        next_states (np.array, optional): Next states of the vehicle. Defaults to None.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes object. Defaults to None.
        control_command (np.array, optional): Control command applied to the vehicle. [steer, acceleration/brake]. Defaults to None.
        alpha (float, optional): Transparency of the rectangle. Defaults to 0.5.

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot): Axes object
    """
    # Vehicle dimensions
    length = 4.5
    width = 1.75
    x, y, cos_yaw, sin_yaw, vx = current_state
    yaw = np.arctan2(sin_yaw, cos_yaw)

    if control_command is not None:
        steer, throttle = control_command

    # Rectangle vertices
    x1 = x + length/2 * np.cos(yaw) + width/2 * np.sin(yaw)
    y1 = y + length/2 * np.sin(yaw) - width/2 * np.cos(yaw)
    x2 = x + length/2 * np.cos(yaw) - width/2 * np.sin(yaw)
    y2 = y + length/2 * np.sin(yaw) + width/2 * np.cos(yaw)
    x3 = x - length/2 * np.cos(yaw) - width/2 * np.sin(yaw)
    y3 = y - length/2 * np.sin(yaw) + width/2 * np.cos(yaw)
    x4 = x - length/2 * np.cos(yaw) + width/2 * np.sin(yaw)
    y4 = y - length/2 * np.sin(yaw) - width/2 * np.cos(yaw)

    if ax is None:
        # Plot the rectangle representing the vehicle
        plt.plot([x1, x2], [y1, y2], 'k')
        plt.plot([x2, x3], [y2, y3], 'k')
        plt.plot([x3, x4], [y3, y4], 'k')
        plt.plot([x4, x1], [y4, y1], 'k')

        # Fill the rectangle with a color
        plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], 'gray', alpha=alpha)

        # Plot x and y positions of the vehicle for all timesteps
        if next_states is not None:
            plt.plot(next_states[0], next_states[1], marker='o', linestyle='-', color='b', label='Next states', alpha=alpha)

        # Plot the speed as a text (m/s) in the upper left corner of the axis
        plt.text(0.05, 0.95, f"Speed: {vx:.2f} m/s", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)

        # Plot the control command as a text in the upper left corner of the axis, below the speed
        if control_command is not None:
            plt.text(0.05, 0.9, f"Steer: {steer:.2f} rad, Throttle: {throttle:.2f} m/s2", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)

        # Plot the direction of the vehicle as an arrow
        plt.arrow(x, y, length/1.5 * np.cos(yaw), length/1.5 * np.sin(yaw), head_width=0.5, head_length=0.5, fc='k', ec='k')

        # Equal aspect ratio
        plt.axis('equal')

        # Labels
        plt.xlabel('X')
        plt.ylabel('Y')

        # Legend in the upper right corner of the axis
        plt.legend(loc='upper right')

        # Title
        plt.title('Vehicle initial position and reachable sets')

    else:
        # Plot the rectangle representing the vehicle
        ax.plot([x1, x2], [y1, y2], 'k')
        ax.plot([x2, x3], [y2, y3], 'k')
        ax.plot([x3, x4], [y3, y4], 'k')
        ax.plot([x4, x1], [y4, y1], 'k')

        # Fill the rectangle with a color
        ax.fill([x1, x2, x3, x4], [y1, y2, y3, y4], 'gray', alpha=alpha)

        # Plot x and y positions of the vehicle for all timesteps
        if next_states is not None:
            ax.plot(next_states[0], next_states[1], marker='o', linestyle='-', color='b', label='Next states', alpha=alpha)

        # Plot the speed as a text (m/s) in the upper left corner of the axis
        ax.text(0.05, 0.925, f"Speed: {vx:.2f} m/s", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=12)

        # Plot the control command as a text in the upper left corner of the axis, below the speed
        if control_command is not None:
            ax.text(0.05, 0.875, f"Steer: {steer:.2f} rad", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=12)
            ax.text(0.05, 0.825, f"Throttle: {throttle:.2f} m/s2", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=12)
        
        # Plot text information on the speed plot to indicate the failure mode if applicable
        # Write it in the top left corner of the speed plot in bold red
        if failure_mode:
            ax.text(0.05, 0.975, "Safety Mode", transform=ax.transAxes,
                               fontsize=12, fontweight='bold', color='red', ha='left', va='top')
        else:
            ax.text(0.05, 0.975, "Nominal", transform=ax.transAxes,
                               fontsize=12, fontweight='bold', color='green', ha='left', va='top')
            
        # Plot the direction of the vehicle as an arrow
        ax.arrow(x, y, length/1.5 * np.cos(yaw), length/1.5 * np.sin(yaw), head_width=0.5, head_length=0.5, fc='k', ec='k')

        # Equal aspect ratio
        ax.axis('equal')

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Legend in the upper right corner of the axis
        ax.legend(loc='upper right')

        # Title
        ax.set_title('Vehicle initial position and reachable sets')

    return ax

def plot_reachable_sets(R_data, ax=None, color='y', ax_yaw=None, alpha=0.2):

    # Separate reachable sets for each timestep and plot them in the same figure (xy plane)
    for i in range(len(R_data)):
        if R_data[i] is None:
            continue
        
        if i == 1:
            R_data[i].project([0, 1]).plot(ax=ax, alpha=alpha, color=color, label='Reachable set')
            if ax_yaw is not None:
                R_data[i].project([2, 3]).plot(ax=ax_yaw, alpha=alpha, color=color, label='Reachable set')
        else:
            R_data[i].project([0, 1]).plot(ax=ax, alpha=alpha, color=color)
            if ax_yaw is not None:
                R_data[i].project([2, 3]).plot(ax=ax_yaw, alpha=alpha, color=color)

        # Plot the center of the reachable set with a x and add label for the first reachable set
        if ax is not None:
            ax.plot(R_data[i].center()[0], R_data[i].center()[1], 'rx')
        else:
            plt.plot(R_data[i].center()[0], R_data[i].center()[1], 'rx')

        if ax_yaw is not None:
            ax_yaw.plot(R_data[i].project([2, 3]).center()[0], R_data[i].project([2, 3]).center()[1], 'rx')
                
    if ax is not None:
        if R_data[0] is not None:
            ax.plot(R_data[0].center()[0], R_data[0].center()[1], 'rx', label='Center of the reachable set')
            ax.legend()
            if ax_yaw is not None:
                ax_yaw.plot(R_data[0].project([2, 3]).center()[0], R_data[0].project([2, 3]).center()[1], 'rx', label='Center of the reachable set')
                ax_yaw.legend()
        
    else:
        plt.plot(R_data[1].center()[0], R_data[1].center()[1], 'rx', label='Center of the reachable set')
        plt.legend()   

    return ax

def get_idxs(u, x_meas_vec_0, initpoints, steps, get_w):
    """
    Get the relevant indexes for the training and testing data. The indexes are selected based on the inputs and states of the vehicle.
    Training indexes are selected based on the maximum and minimum values of the inputs and states, and some random initial points.
    Testing indexes is selected as the remaining indexes that are not in the training data.

    Args:
        u (np.array): Inputs
        x_meas_vec_0 (np.array): Initial states
        x_meas_vec_1 (np.array): Final states
        initpoints (int): Number of initial points
        steps (int): Number of time steps

    Returns:
        idxs (np.array): Indexes for the training data
        idxs_test (np.array): Indexes for the testing data
    """
    # Select some relevant initial points from the input and state vectors to train the model
    idx_list = list()
    if initpoints >=8:
        idx_inputs = get_idx_inputs(u, steps)
        idx_states = get_idx_states(x_meas_vec_0, steps, get_w)
        idx_list = idx_inputs + idx_states

    # Randomly select the remaining initial points if the number of selected points is less than the desired number of initial points
    if x_meas_vec_0.shape[1] - steps < initpoints- len(idx_list):
        idx_init_points = [0]
    else:
        idx_init_points = np.random.choice(x_meas_vec_0.shape[1]-steps, initpoints- len(idx_list), replace=False)
    
    # Combine the random and selected initial points and generate the indexes (integers) for the input and state vectors
    idx_init_points = np.concatenate((idx_list, idx_init_points))
    idxs = np.array([np.arange(i, i + steps, dtype=int) for i in idx_init_points]).flatten()

    # Get the indexes for the test data. All indexes available in data are included in the test data
    idxs_test = np.arange(x_meas_vec_0.shape[1]-steps)

    return idxs, idxs_test

def get_input_state_vectors(u, x_meas_vec_0, x_meas_vec_1, idxs):
    """
    Get the input and state vectors for the required data. 

    Args:
        u (np.array): Inputs
        x_meas_vec_0 (np.array): Initial states
        x_meas_vec_1 (np.array): Final states
        idxs (np.array): Indexes for the required data

    Returns:
        u (np.array): Inputs for the required data
        x_meas_vec_0 (np.array): Initial states for the required data
        x_meas_vec_1 (np.array): Final states for the required data
    """
    if len(idxs) < 1:
        idxs = np.arange(x_meas_vec_0.shape[1])
    
    return u[:, idxs], x_meas_vec_0[:, idxs], x_meas_vec_1[:, idxs]

def get_idx_inputs(u, steps):
    """
    Get the index of the maximum and minimum steering and throttle values. 

    Args:
        u (np.array): Inputs
        steps (int): Number of time steps

    Returns:
        idx_list (list): Indexes of the maximum and minimum steering and throttle values
    """
    # Get the index of the maximum and minimum steering values
    idx_steering = [np.argmax(u[0, :])-int(steps/2), np.argmin(u[0, :])-int(steps/2)]

    # Get the index of the maximum and minimum throttle values
    idx_throttle = [np.argmax(u[1, :])-int(steps/2), np.argmin(u[1, :])-int(steps/2)]

    # Print minimum and maximum input values
    print(f"Steering min: {np.min(u[0, :])} rad, max: {np.max(u[0, :])} rad")
    print(f"Throttle min: {np.min(u[1, :])} m/s2, max: {np.max(u[1, :])} m/s2")

    # Correct the indexes
    idx_list = correct_idx_list(idx_steering + idx_throttle, len(u[0, :]), steps)

    return idx_list

def get_idx_states(x_meas_vec_0, steps, get_w):
    """
    Get the index of the maximum and minimum speed values, the value where the yaw angle points 
    closest to the north, east, south and west directions, and the maximum and minimum yaw rate values.

    Args:
        x_meas_vec_0 (np.array): Initial states
        steps (int): Number of time steps

    Returns:
        idx_list (list): Indexes of the maximum and minimum speed values, the value where the yaw angle points 
        closest to the north, east, south and west directions, and the maximum and minimum yaw rate values
    """
    # Get the index of the maximum and minimum speed values
    idx_speed = [np.argmax(x_meas_vec_0[4, :])-int(steps/2), np.argmin(x_meas_vec_0[4, :])-int(steps/2)]

    # Get the index of the maximum and minimum yaw rate values
    idx_yaw_rate = [np.argmax(x_meas_vec_0[5, :])-int(steps/2), np.argmin(x_meas_vec_0[5, :])-int(steps/2)] if get_w else []

    idx_north_south = [np.argmin(x_meas_vec_0[3, :])-int(steps/2), np.argmax(x_meas_vec_0[3, :])-int(steps/2)]
    idx_east_west = [np.argmin(x_meas_vec_0[2, :])-int(steps/2), np.argmax(x_meas_vec_0[2, :])-int(steps/2)]

    # Print minimum and maximum state values
    print(f"Speed min: {np.min(x_meas_vec_0[4, :])} m/s, max: {np.max(x_meas_vec_0[4, :])} m/s") 

    if get_w:
        print(f"Yaw rate min: {np.min(x_meas_vec_0[5, :])} rad/s, max: {np.max(x_meas_vec_0[5, :])} rad/s") 

    # Correct the indexes
    idx_list = correct_idx_list(idx_speed + idx_north_south + idx_east_west + idx_yaw_rate, len(x_meas_vec_0[0, :]), steps)

    return idx_list

def correct_idx_list(idx_list, length_u, steps):
    """
    Correct the indexes to be within the range of the input and state vectors.

    Args:
        idx_list (list): Indexes to correct
        length_u (int): Length of the input vector
        steps (int): Number of time steps

    Returns:
        idx_list (list): Corrected indexes
    """
    # Check if the indexes are lower than len(u) - steps
    for i in range(len(idx_list)):
        if idx_list[i] < 0:
            idx_list[i] = 0
        elif idx_list[i] > length_u - steps:
            idx_list[i] = length_u - steps

    return idx_list

def find_closest_idx_point(point, array):
        """
        Find the closest index of the test point to the given point.

        Args:
            point (list): Point to find the closest index.
            array (np.array): Test points.

        Returns:
            idx (int): Index of the test point.
        """
        dist = np.linalg.norm(array[:2] - np.array(point).reshape(2,1), axis=0)
        idx = np.argmin(dist)

        return idx

def polygon_to_zonotope(polygon, reducing_order=2):
    """
    Convert a Shapely polygon to a better-approximated zonotope 
    using edge vectors instead of full vertex differences.
    
    Args:
        polygon (shapely.geometry.Polygon): Input polygon
        reducing_order (int): Order of the Girard reduction
    
    Returns:
        Zonotope: Zonotope object
    """
    # Extract unique vertices (remove last duplicate)
    vertices = np.array(polygon.exterior.coords[:-1])
    
    # Compute the centroid for XY dimensions
    centroid = np.mean(vertices, axis=0).reshape(-1, 1)

    # Compute edge vectors (difference between consecutive vertices)
    edge_vectors = np.diff(vertices, axis=0, append=[vertices[0]])  # Cyclic edges
    
    # Use half of each edge vector as a generator for the zonotope of XY dimensions
    generators = (edge_vectors.T / 4)  # Scale to 1/4 to avoid over-approximation

    zonotope_2d = Zonotope(centroid, generators)
    
    return zonotope_2d.reduce('girard', reducing_order)

def polygon_to_interval(polygon):
    """
    Convert a Shapely polygon to a better-approximated interval 
    using edge vectors instead of full vertex differences.
    
    Args:
        polygon (shapely.geometry.Polygon): Input polygon
        reducing_order (int): Order of the Girard reduction
    
    Returns:
        Interval: Interval object
    """
    # Extract unique vertices (remove last duplicate)
    vertices = np.array(polygon.exterior.coords[:-1])

    # Compute upper and lower bounds for each dimension
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)

    # Create intervals for each dimension
    x_interval = Interval(x_min, x_max)
    y_interval = Interval(y_min, y_max)

    return x_interval, y_interval

def translate_polygon(road_polygon, initial_latitude, initial_longitude):
    # Create a list to store the translated coordinates
    translated_coords = []
    
    # Translate each point in the exterior of the polygon
    for point in road_polygon.exterior.coords:
        x, y, _ = gps2xyz(point[1], point[0], 0.0, initial_latitude, initial_longitude, 0.0)
        translated_coords.append((x, y))
    
    # Create a new polygon with the translated coordinates
    translated_polygon = Polygon(translated_coords)
    return translated_polygon

def enclose_points(polygon, num_vertices=10):
    """
    Encloses the exterior vertices of a Shapely polygon with a convex polytope.
    
    Args:
        polygon (Polygon): A Shapely Polygon object.
        num_vertices (int): The number of vertices to use in the convex hull.
    
    Returns:
        Polygon: A convex hull approximation of the original polygon with at most num_vertices.
    """
    # Extract exterior coordinates
    points = np.array(polygon.exterior.coords)
    
    # Compute convex hull
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    
    # If the number of vertices is greater than num_vertices, reduce the number of vertices
    if len(hull_vertices) > num_vertices:
        step = len(hull_vertices) // num_vertices
        hull_vertices = hull_vertices[::step][:num_vertices]
    
    # Return the convex hull as a Shapely Polygon
    return Polygon(hull_vertices)

def get_random_idxs_for_sliding_window(sliding_window_size, training_size):
    """
    Get random indexes for the sliding window.

    Args:
        sliding_window_size (int): Size of the sliding window
        training_size (int): Size of the training data

    Returns:
        idxs (np.array): Random indexes for the sliding window
    """
    if sliding_window_size >= training_size:
        return np.arange(training_size)
    else:
        return np.sort(np.random.choice(training_size, sliding_window_size, replace=False))

def completion_bar(current, total):
    """
    Print the completion bar for the reachability analysis.

    Args:
        current (int): Current iteration.
        total (int): Total number of iterations.
    """
    bar_length = 50
    progress = current/total
    block = int(round(bar_length * progress))
    text = f"\rProgress: [{'#' * block}{'.' * (bar_length - block)}] {progress*100:.1f}%"
    print(text, end="")

def admissible_yaw_intervals(cos_interval, sin_interval, prev_yaw=0.0, resolution=1000, add_half_pi=False):
    """
    Computes admissible yaw intervals given bounds on cos(theta) and sin(theta),
    expressed in the range [prev_yaw - pi, prev_yaw + pi].

    Parameters:
        cos_interval: tuple (min_cos, max_cos)
        sin_interval: tuple (min_sin, max_sin)
        prev_yaw: reference yaw angle (float, radians)
        resolution: number of samples in angle space
        substract_half_pi: boolean, whether to subtract pi/2 from the intervals

    Returns:
        List of Interval(inf, sup) objects, each representing a valid yaw interval.
    """
    # Sample theta around prev_yaw ± pi
    theta = np.linspace(prev_yaw - np.pi, prev_yaw + np.pi, resolution)
    cos_vals = np.cos(theta)
    sin_vals = np.sin(theta)

    # Apply the constraints
    mask = (
        (cos_vals >= cos_interval[0]) & (cos_vals <= cos_interval[1]) &
        (sin_vals >= sin_interval[0]) & (sin_vals <= sin_interval[1])
    )
    valid_theta = theta[mask]

    def group_intervals(thetas):
        if len(thetas) == 0:
            return []
        intervals = []
        start = thetas[0]
        prev = thetas[0]
        step = 2 * np.pi / resolution
        for t in thetas[1:]:
            if t - prev > 1.5 * step:
                intervals.append([start, prev])
                start = t
            prev = t
        intervals.append([start, prev])
        return intervals

    group = group_intervals(valid_theta)[0]

    if add_half_pi:
        group[0] += np.pi / 2
        group[1] += np.pi / 2

    return group

def continuous_angles(angles, initial_angle=0.):
    """
    Adjusts angles to be continuous around the initial angle.

    Args:
        angles (np.array): Array of angles in radians.
        initial_angle (float): Initial angle in radians.

    Returns:
        np.array: Adjusted angles in radians.
    """
    scalar_input = np.isscalar(angles)
    angles = np.atleast_1d(angles).astype(float)
    result = angles + 2 * np.pi * np.round((initial_angle - angles) / (2 * np.pi))

    return result[0] if scalar_input else result

def yaw_zonotope_from_cos_sin(zonotope_cos, zonotope_sin, initial_angle=0.0):
    """
    Approximates a zonotope for yaw (theta) given the zonotopes of cos(theta) and sin(theta).
    
    Args:
        zonotope_cos (Zonotope): Zonotope for cos(theta).
        zonotope_sin (Zonotope): Zonotope for sin(theta).
        initial_angle (float): Previous yaw angle (radians).

    Returns:
        tuple: (center, generators) of the approximated zonotope for theta.
    """
    c_cos = zonotope_cos.center()[0,0]
    c_sin = zonotope_sin.center()[0,0]

    G_cos = zonotope_cos.generators()
    G_sin = zonotope_sin.generators()

    # Compute yaw center from center values
    c_theta = continuous_angles(np.arctan2(c_sin, c_cos), initial_angle=initial_angle)

    # Compute the Jacobian (first-order partial derivatives)
    r2 = c_cos**2 + c_sin**2
    if r2 == 0:
        raise ValueError("Invalid center: c_cos and c_sin cannot both be zero.")

    dtheta_dcos = -c_sin / r2
    dtheta_dsin =  c_cos / r2

    # Ensure G_cos and G_sin are (n,) arrays
    G_cos = np.atleast_1d(G_cos)
    G_sin = np.atleast_1d(G_sin)

    # Compute projected yaw generators
    G_theta = dtheta_dcos * G_cos + dtheta_dsin * G_sin

    return c_theta, G_theta[0,0]

def smooth_min(a, b, alpha=10.0):
    """
    Smooth approximation of min(a, b).
    The higher the alpha, the closer the result is to the true min.
    """
    return np.log(np.exp(alpha * a) + np.exp(alpha * b)) / alpha

def xyz2gps(x, y, z, origin_latitude, origin_longitude, origin_altitude=float()):
    """
    Translates from local coordinates XYZ to global coordinates GPS latitude, longitude and altitude.

    Args:
            x (float): X coordinate in local coordinates.
            y (float): Y coordinate in local coordinates.
            z (float): Z coordinate in local coordinates.
            origin_x (float): X coordinate of the origin in local coordinates.
            origin_y (float): Y coordinate of the origin in local coordinates.
            origin_z (float): Z coordinate of the origin in local coordinates.
        
    Returns:
            float, float, float: GPS coordinates latitude, longitude and altitude.
    """
    lat, lon, _ = pymap3d.enu2geodetic(x, y, z, origin_latitude, origin_longitude, origin_altitude, ell=pymap3d.utils.Ellipsoid("wgs84"))
    
    return lat, lon

def gps2xyz(lat, lon, h, origin_latitude, origin_longitude, origin_altitude, format=str(), yaw_N=float()):
    """
    Translates from GPS coordinates latitude, longitude and altitude to local coordinates XYZ.

    Args:
            lat (float): Latitude in GPS coordinates.
            lon (float): Longitude in GPS coordinates.
            h (float): Altitude in GPS coordinates.
            origin_latitude (float): Latitude of the origin in GPS coordinates.
            origin_longitude (float): Longitude of the origin in GPS coordinates.
            origin_altitude (float): Altitude of the origin in GPS coordinates.
            format (str, optional): Format of the output coordinates. Defaults to str().
            yaw_N (float, optional): Yaw of the vehicle in GPS coordinates. Defaults to float().

    Returns:
            float, float, float: X, Y and Z coordinates in local coordinates.
    """
    x, y, z = pymap3d.geodetic2enu(lat, lon, h, origin_latitude, origin_longitude, origin_altitude, ell=pymap3d.utils.Ellipsoid("wgs84"))

    if "global" in format:
            return x, y, z

    if "local" in format:
            [x, y] = global_to_local([0., 0., yaw_N], [[x, y]], is_object=True)[0]

    return x, y, z

def global_to_local(state, array_global, is_object=False, topic=""):
    """
    Transformation from global to local coordinates:
    
    Args:
            state (list): actual position in global coordinates.
            array_global (list): vector in global coordinates.
            is_object (bool, optional): if True, the vector is an object (without orientation). Defaults to False.
            topic (str, optional): topic of the vector. Defaults to None.
    """
    # Define rotation matrix and invert it
    R = np.array([[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]])
    R_inv = np.linalg.inv(R)

    # Initialize final path
    local_array_path = list()

    for e in array_global:
            # X_local = inv(R) * (X_global - T)
            vec = np.array([e[0], e[1]])
            vec -= np.array([state[0], state[1]])
            vec = np.matmul(R_inv, vec)
            vec = [vec[0], vec[1]]
            
            # If angles need to be rotated
            if not is_object:
                    vec = [vec[0], vec[1], e[2] - state[2]]
                    # Angles always between [-pi, pi]
                    vec[2] = wrap_angle(vec[2])
                    
            if topic.lower()=="state":
                    vec += [e[3], e[4], e[5]]

            local_array_path.append(vec)

    if len(local_array_path) == 0:
            local_array_path = [[]]

    return local_array_path


def local_to_global(state, array_local, is_object=False, topic=""):
    """
    Transformation from local to global coordinates:

    Args:
        state (list): actual position in global coordinates.
        array_local (list): vector in local coordinates.
        is_object (bool, optional): if True, the vector is an object (without orientation). Defaults to False.
        topic (str, optional): topic of the vector. Defaults to None.
    """
    # Define rotation matrix
    R = np.array([[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]])

    # Initialize final path
    global_array_path = list()

    for e in array_local:
        # X_global = R * X_local + T
        vec = np.array([e[0], e[1]])
        vec = np.matmul(R, vec)
        vec += np.array([state[0], state[1]])
        vec = [vec[0], vec[1]]

        # If angles need to be rotated
        if not is_object:
            vec = [vec[0], vec[1], wrap_angle(e[2] + state[2])]
        
        if topic.lower() == "state":
            vec += [e[3], e[4], e[5]]

        global_array_path.append(vec)

    if len(global_array_path) == 0:
        global_array_path = [[]]

    return global_array_path

def wrap_angle(phases, lower_bound=-np.pi, upper_bound=np.pi):
    """
    Wrap angles to [lower_bound, upper_bound).

    Args:
        phases (np.array): array of angles
        lower_bound (float, optional): lower bound of the desired range. Defaults to -np.pi.
        upper_bound (float, optional): upper bound of the desired range. Defaults to np.pi.

    Returns:
        np.array: wrapped angles
    """
    range_width = upper_bound - lower_bound
    
    return (phases - lower_bound) % range_width + lower_bound

def get_states_yaw(state_cos_sin, add_vy=False):
    """
    Get the states including the yaw angle from the cos and sin values.

            Args:
                    state_cos_sin (np.array): State yaw cos and sin. [x, y, cos(yaw), sin(yaw), v]
                    add_vy (bool, optional): If True, add the y velocity. Defaults to False.

            Returns:
                    np.array: State including the yaw angle. [x, y, yaw, v]
    """
    yaw = np.arctan2(state_cos_sin[3], state_cos_sin[2])
    state = np.array([state_cos_sin[0], state_cos_sin[1], yaw, state_cos_sin[4]])

    if add_vy:
            state = np.array([state_cos_sin[0], state_cos_sin[1], yaw, state_cos_sin[4], 2.588*state_cos_sin[5]])

    return state

def get_states_yaw_cos_sin(state):
    """
    Get the states including the yaw angle from the cos and sin values.
    
            Args:
                    state (np.array): State including the yaw angle. [x, y, yaw, v]
    
            Returns:
                    np.array: State yaw cos and sin. [x, y, cos(yaw), sin(yaw), v]
    
    """
    state_cos_sin = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])
    
    return state_cos_sin

def represent_percentage_time_bar(start_time, current_time, duration):
    """
    Represent the percentage of the time passed.

    Args:
        current_time (float): Current time.
    """
    percentage = (current_time - start_time) / duration
    bar = int(percentage * 20)
    print(f"\r[{'='*bar}{' '*(20-bar)}] {percentage*100:.0f}%", end="")

def package_path():
    """
    Get the package path.

    Returns:
        str: Package path.
    """
    return os.path.dirname(os.path.abspath(__file__)).replace("/scripts/utils", "")