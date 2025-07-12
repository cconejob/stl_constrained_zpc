#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import gridspec  # Import gridspec for custom subplot layout
from collections import deque
import os
import numpy as np
import matplotlib.patches as patches

from stl_constrained_zpc.scripts.utils.utils import plot_vehicle, admissible_yaw_intervals, continuous_angles, yaw_zonotope_from_cos_sin


class ZPCVisualizer:
    def __init__(self, show_plots=True, save_plots=False, plots_folder="/plots", debug=False, plot_states_time=False, plot_yaw=False):
        """
        Class for visualizing the Zonotopic Predictive Controller. 

        Args:
            show_plots (bool): Flag to show the plots.
            save_plots (bool): Flag to save the plots.
            plots_folder (str): Folder to save the plots.
            debug (bool): Flag to print debug information.
            plot_states_time (bool): Flag to plot the states over time.

        Attributes:
            show_plots (bool): Flag to show the plots.
            save_plots (bool): Flag to save the plots.
            fig (object): Figure object for the plots.
            ax_xy (object): Axis object for the XY plot.
            ax_yaw (object): Axis object for the yaw plot.
            ax_speed (object): Axis object for the speed plot.
            iteration_count (int): Counter for the number of iterations.
            saved_plot_files (list): List of saved plot files.
            plots_folder (str): Folder to save the plots.
        """
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.debug = debug
        self.plot_states_time = plot_states_time
        self.plot_yaw = plot_yaw
        self.x_trace = deque(maxlen=300)
        self.y_trace = deque(maxlen=300)

        # Enable interactive mode for Matplotlib
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))  # Adjust figure size

        # Set title to figure
        self.fig.suptitle("NZPC Current Step", fontsize=18)

        # Define grid layout: 1 column on the left, 2 rows stacked on the right
        gs = gridspec.GridSpec(2, 2)  # Left is twice as wide as right
        gs.set_width_ratios([1.5, 1])  # Set width ratios for the grid
        gs.update(hspace=0.4) # Adjust vertical space between subplots
        gs.update(wspace=0.2) # Adjust horizontal space between subplots

        # Create subplots with specific positions
        self.ax_xy = self.fig.add_subplot(gs[:, 0])  # Large plot for XY (occupies 2 rows)
        self.ax_yaw = self.fig.add_subplot(gs[0, 1])  # Upper right for yaw
        self.ax_speed = self.fig.add_subplot(gs[1, 1])  # Lower right for speed

        if plot_yaw:
            # Initialize yaw variable to avoid discontinuity in the yaw plot
            self.prev_yaw = None

        # Draw initial figure
        self.iteration_count = int()
        self.saved_plot_files = list()
        self.saved_plot_files_time = list()
        self.plots_folder = plots_folder

        # Create the plots folder if it does not exist
        if self.save_plots:
            if not os.path.exists(self.plots_folder):
                os.makedirs(self.plots_folder)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_6D_zonotopes(self, zonotopes_6D, color='red', alpha=0.05, label=None, plot_center_reach=False, add_vehicle_dimensions=False, four_d=False):
        """
        Plot the side information zonotope.

        Args:
            zonotopes_6D (list): List of Zonotope objects representing a 6D zonotope.
            color (str): Color for the plot.
            alpha (float): Transparency for the plot.
            label (str): Label for the plot.
            plot_center_reach (bool): Flag to plot the center of the reach set.
            add_vehicle_dimensions (bool): Flag to add vehicle dimensions to the xy plot.
        """
        c_history_6D = deque(maxlen=len(zonotopes_6D))
        G_history_6D = deque(maxlen=len(zonotopes_6D))
        c_yaw_history_6D = deque(maxlen=len(zonotopes_6D))
        G_yaw_history_6D = deque(maxlen=len(zonotopes_6D))

        for i, zonotope_6D in enumerate(zonotopes_6D):
            if zonotope_6D is None:
                continue
            zonotope_position_6D = zonotope_6D.project([0, 1])
            zonotope_speed_6D = zonotope_6D.project([4]).reduce('girard', 1)

            self.ax_xy = zonotope_position_6D.plot(ax=self.ax_xy, color=color, alpha=alpha)
            
            if not self.plot_yaw:
                zonotope_yaw_6D = zonotope_6D.project([2, 3])
                self.ax_yaw = zonotope_yaw_6D.plot(ax=self.ax_yaw, color=color, alpha=alpha)

            else:
                zonotope_cos_yaw = zonotope_6D.project([2]).reduce('girard', 1)
                zonotope_sin_yaw = zonotope_6D.project([3]).reduce('girard', 1)
                c_yaw, G_yaw = yaw_zonotope_from_cos_sin(zonotope_cos_yaw, zonotope_sin_yaw, initial_angle=self.prev_yaw)
                c_yaw_history_6D.append(c_yaw)
                G_yaw_history_6D.append(G_yaw)

            # If the flag is set, plot the center with an 'x' in the reach set for the xy plot
            if plot_center_reach:
                self.ax_xy.plot(zonotope_position_6D.center()[0], zonotope_position_6D.center()[1], 'x', color='red',
                                label="Center of Reachable Set")
            
            c_history_6D.append(zonotope_speed_6D.center().reshape(-1, 1))
            G_history_6D.append(zonotope_speed_6D.generators())
            if i == len(zonotopes_6D)-1:
                self.ax_speed = zonotope_speed_6D.plot_1D(ax=self.ax_speed, alpha=min(alpha*(i+1)/2, 1), color=color, c_history=c_history_6D, G_history=G_history_6D, label=label)
                c_history_6D.clear()
                G_history_6D.clear()
                self.ax_xy = zonotope_position_6D.plot(ax=self.ax_xy, color=color, alpha=alpha, label=label)
                if self.plot_yaw:
                    self.ax_yaw = zonotope_cos_yaw.plot_1D(ax=self.ax_yaw, alpha=min(alpha*(i+1)/2, 1), color=color, c_history=c_yaw_history_6D, G_history=G_yaw_history_6D, label=label)
                    c_yaw_history_6D.clear()
                    G_yaw_history_6D.clear()
                else:
                    self.ax_yaw = zonotope_yaw_6D.plot(ax=self.ax_yaw, color=color, alpha=alpha, label=label)

    def plot_6D_interval(self, interval_6D, color='red', alpha=0.05, label=None):
        """
        Plot 6D interval across prediction steps.
        
        - interval_6D: shape = (5, N)
        Row 0: X
        Row 1: Y
        Row 2: Cos(Yaw)
        Row 3: Sin(Yaw)
        Row 4: Speed
        """
        # Plot XY and yaw (cos/sin) → loop over columns (timesteps)
        for column in range(interval_6D.shape[1]):
            interval_x = interval_6D[0, column]
            interval_y = interval_6D[1, column]
            interval_cos_yaw = interval_6D[2, column]
            interval_sin_yaw = interval_6D[3, column]

            # === XY subplot ===
            if interval_x is not None and interval_y is not None:
                x0 = float(interval_x.inf[0])
                x1 = float(interval_x.sup[0])
                y0 = float(interval_y.inf[0])
                y1 = float(interval_y.sup[0])
                width = x1 - x0
                height = y1 - y0
                rect = patches.Rectangle((x0, y0), width, height, color=color, alpha=alpha, label=label if column == 0 else None)
                self.ax_xy.add_patch(rect)

            # === Yaw subplot ===
            if not self.plot_yaw:
                if interval_cos_yaw is not None and interval_sin_yaw is not None:
                    x0 = float(interval_cos_yaw.inf[0])
                    x1 = float(interval_cos_yaw.sup[0])
                    y0 = float(interval_sin_yaw.inf[0])
                    y1 = float(interval_sin_yaw.sup[0])
                    width = x1 - x0
                    height = y1 - y0
                    rect = patches.Rectangle((x0, y0), width, height, color=color, alpha=alpha, label=label if column == 0 else None)
                    self.ax_yaw.add_patch(rect)
        
        timesteps = np.arange(interval_6D.shape[1])
        
        # Plot Yaw evolution → loop over columns to build time series
        if self.plot_yaw:
            yaw_inf = []
            yaw_sup = []
            for column in range(interval_6D.shape[1]):
                interval_cos_yaw = interval_6D[2, column]
                interval_sin_yaw = interval_6D[3, column]
                if interval_cos_yaw is not None and interval_sin_yaw is not None:
                    interval_yaw = admissible_yaw_intervals([interval_cos_yaw.inf, interval_cos_yaw.sup], [interval_sin_yaw.inf, interval_sin_yaw.sup], prev_yaw=self.prev_yaw)
                    yaw_inf.append(interval_yaw[0])
                    yaw_sup.append(interval_yaw[1])
                else:
                    yaw_inf.append(np.nan)
                    yaw_sup.append(np.nan)

            self.ax_yaw.fill_between(timesteps, yaw_inf, yaw_sup, color=color, alpha=min(alpha*3, 1), label=label, edgecolor='black')

        # Plot Speed evolution → loop over columns to build time series
        speed_inf = []
        speed_sup = []

        for column in range(interval_6D.shape[1]):
            interval_speed = interval_6D[4, column]
            if interval_speed is not None:
                speed_inf.append(interval_speed.inf[0])
                speed_sup.append(interval_speed.sup[0])
            else:
                speed_inf.append(np.nan)
                speed_sup.append(np.nan)

        self.ax_speed.fill_between(timesteps, speed_inf, speed_sup, color=color, alpha=min(alpha*3, 1), label=label, edgecolor='black')

    def plot_hd_info(self, hd_zonotope, color='blue', alpha=0.1, label=None):
        """
        Plot the HD information zonotope.

        - hd_zonotope: shape = (4, N)
        Row 0: XY (Zonotopes)
        Row 2: Cos(Yaw)
        Row 3: Sin(Yaw)
        Row 4: Speed

        Args:
            hd_zonotope (object): HD zonotope object.
            color (str): Color for the plot.
            alpha (float): Transparency for the plot.
            label (str): Label for the plot.
        """
        # Plot XY and yaw (cos/sin) → loop over columns (timesteps)
        for column in range(hd_zonotope.shape[1]):
            zonotope_xy = hd_zonotope[0, column]
            interval_cos_yaw = hd_zonotope[1, column]
            interval_sin_yaw = hd_zonotope[2, column]

            # === XY subplot ===
            if zonotope_xy is not None:
                self.ax_xy = zonotope_xy.plot(ax=self.ax_xy, color=color, alpha=alpha, label=label if column == 0 else None)

            # === Yaw subplot ===
            if not self.plot_yaw:
                if interval_cos_yaw is not None and interval_sin_yaw is not None:
                    x0 = float(interval_cos_yaw.inf)
                    x1 = float(interval_cos_yaw.sup)
                    y0 = float(interval_sin_yaw.inf)
                    y1 = float(interval_sin_yaw.sup)
                    width = x1 - x0
                    height = y1 - y0
                    rect = patches.Rectangle((x0, y0), width, height, color=color, alpha=alpha, label=label if column == 0 else None)
                    self.ax_yaw.add_patch(rect)

        timesteps = np.arange(hd_zonotope.shape[1])

        # Plot Yaw evolution → loop over columns to build time series
        yaw_inf = []
        yaw_sup = []
        if self.plot_yaw:
            for column in range(hd_zonotope.shape[1]):
                interval_cos_yaw = hd_zonotope[2, column]
                interval_sin_yaw = hd_zonotope[3, column]
                if interval_cos_yaw is not None and interval_sin_yaw is not None:
                    interval_yaw = admissible_yaw_intervals([interval_cos_yaw.inf, interval_cos_yaw.sup], [interval_sin_yaw.inf, interval_sin_yaw.sup], prev_yaw=self.prev_yaw, add_half_pi=True)
                    print("Interval Yaw HD Map:", interval_yaw)
                    yaw_inf.append(interval_yaw[0])
                    yaw_sup.append(interval_yaw[1])
                else:
                    yaw_inf.append(np.nan)
                    yaw_sup.append(np.nan)

            self.ax_yaw.fill_between(timesteps, yaw_inf, yaw_sup, color=color, alpha=min(alpha*3, 1), label=label, edgecolor='black')
        
        # Plot Speed evolution → loop over columns to build time series
        speed_inf = []
        speed_sup = []
        for column in range(hd_zonotope.shape[1]):
            interval_speed = hd_zonotope[3, column]
            if interval_speed is not None:
                speed_inf.append(interval_speed.inf[0])
                speed_sup.append(interval_speed.sup[0])
            else:
                speed_inf.append(np.nan)
                speed_sup.append(np.nan)

        self.ax_speed.fill_between(timesteps, speed_inf, speed_sup, color=color, alpha=min(alpha*3, 1), label=label, edgecolor='black')

    def plot_optimal_states(self, optimal_states, is_trajectory=False):
        """
        Plot the optimal states on the XY, yaw, and speed plots.

        Args:
            optimal_states (np.array): Array of optimal states.
        """
        label = "Optimal States" if not is_trajectory else "Vehicle Trajectory"
        if self.plot_yaw:
            angles = continuous_angles(np.arctan2(optimal_states[:, 3], optimal_states[:, 2]), initial_angle=self.prev_yaw)
        for i, state in enumerate(optimal_states):
            if i == 0:
                self.ax_xy.plot(state[0], state[1], 'ks', label="Current vehicle state")
                if not self.plot_yaw:
                    self.ax_yaw.plot(state[2], state[3], 'ks', label="Current vehicle state")
                else:
                    self.ax_yaw.plot(0, angles[i], 'ks', label="Current vehicle state")
                
                self.ax_speed.plot(0, state[4], 'ks', label="Current vehicle state")
            
            elif i == len(optimal_states) - 1:
                self.ax_xy.plot(state[0], state[1], 'bo', label=label)
                if not self.plot_yaw:
                    self.ax_yaw.plot(state[2], state[3], 'bo', label=label)
                else:
                    self.ax_yaw.plot(i, angles[i], 'bo', label=label)
                
                self.ax_speed.plot(i, state[4], 'bo', label=label)
            
            else:
                self.ax_xy.plot(state[0], state[1], 'bo')
                if not self.plot_yaw:
                    self.ax_yaw.plot(state[2], state[3], 'bo')
                else:
                    self.ax_yaw.plot(i, angles[i], 'bo')
                
                self.ax_speed.plot(i, state[4], 'bo')

    def plot_obstacle_interval(self, obstacle_interval, color='black', alpha=0.25, label=None):
        """
        Plot the obstacle interval.

        Args:
            obstacle_interval (list): List of obstacle intervals.
            color (str): Color for the plot.
            alpha (float): Transparency for the plot.
            label (str): Label for the plot.
        """
        obstacle_interval_x, obstacle_interval_y = obstacle_interval
        for i in range(len(obstacle_interval_x)):
            if i == 0:
                if obstacle_interval_x[i] is None or obstacle_interval_y[i] is None:
                    continue
                x0 = float(obstacle_interval_x[i].inf)
                x1 = float(obstacle_interval_x[i].sup)
                y0 = float(obstacle_interval_y[i].inf)
                y1 = float(obstacle_interval_y[i].sup)

                print("Obstacle Interval:", x0, x1, y0, y1)
                width = x1 - x0
                height = y1 - y0
                rect = patches.Rectangle((x0, y0), width, height, color=color, alpha=alpha, label=label if i == 0 else None)
                self.ax_xy.add_patch(rect)

    def set_parameters(self, ax, title=None, xlim=None, ylim=None, xticks=int(), xlabel=None, ylabel=None, grid=False, legend=False):
        """
        Set the parameters for the plots. 

        Args:
            ax (object): Axis object for the plot.
            title (str): Title for the plots.
            xlim (tuple): Tuple with the limits for the x axis.
            ylim (tuple): Tuple with the limits for the y axis.
            xticks (int): Number of ticks for the x axis.
            xlabel (str): Label for the x axis.
            ylabel (str): Label for the y axis.
            grid (bool): Flag to set the grid for the plot.
            legend (bool): Flag to set the legend for the plot.

        Returns:
            ax (object): Axis object with the parameters set.
        """
        # Set labels
        ax.set_title(title, fontsize=16)

        # Set limits for the plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Set ticks for the plot if needed with fontsize 10
        if xticks > 0:
            ax.set_xticks(range(0, xticks, 1))

        # Set labels for the plot
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        # Avoid decimal numbers in the ticks
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        # Set legend for the plot with size 14
        if legend:
            ax.legend(loc='upper right', fontsize=12)

        # Set grid for plot if needed
        if grid:
            ax.grid()

        return ax
    
    def add_figure_legend(self, loc='upper right', time=False):
        """
        Add a single legend for all plots in the figure.

        Args:
            loc (str): Location of the legend in the figure.
        """
        handles, labels = [], []

        # Collect legend entries from all subplots
        for ax in [self.ax_yaw, self.ax_speed, self.ax_xy]:
            h, l = ax.get_legend_handles_labels()
            if h:  # Only add if there are handles
                handles.extend(h)
                labels.extend(l)

        if handles:
            # Remove duplicates while preserving order
            unique_labels = []
            unique_handles = []
            for h, l in zip(handles, labels):
                if l not in unique_labels and l:  # Ignore empty labels
                    unique_labels.append(l)
                    unique_handles.append(h)

            # Add a single legend to the figure
            self.ax_xy.legend(unique_handles, unique_labels, loc=loc, fontsize=12)
        else:
            print("Warning: No labeled artists found to add to legend.")

    def update_visualization(self, current_state=None, side_zonotopes=None, reach_zonotopes=None,
                             optimal_states=None, optimal_control=None, navigation_array=None, obstacle_interval=None,
                             plot_center_reach=False, is_trajectory=False, stl_interval=None, hd_zonotope=None,
                             failure_mode=False, steps=5, plot_trace=False):
        """
        Run the visualizer for the Zonotopic Predictive Controller. 
        
        Args:
            current_state (np.array): Array of current state.
            side_zonotopes (list): List of side information zonotopes.
            reach_zonotopes (list): List of reach zonotopes.
            optimal_states (np.array): Array of optimal states.
            optimal_control (np.array): Array of optimal control.
            navigation_array (np.array): Array of navigation information.
            obstacle_interval (list): List of obstacle zonotopes.
            object_position (np.array): Array of object position. [x, y]
            object_too_close (bool): Flag to indicate if the object is too close.
            plot_center_reach (bool): Flag to plot the center of the reach set.
            is_trajectory (bool): Flag to indicate if the optimal states are a trajectory.
            stl_interval (np.array): Array of STL interval.
            hd_zonotope (np.array): Array of HD zonotope.
            failure_mode (bool): Flag to indicate if the vehicle is in failure mode.
            steps (int): Number of steps for the prediction horizon.
            plot_trace (bool): Flag to plot the trace of the vehicle.
        """
        if optimal_control is None:
            return
        
        if self.plot_yaw and self.prev_yaw is None:
            self.prev_yaw = np.arctan2(current_state[3], current_state[2])

        # Clear previous plots
        self.ax_yaw.cla()
        self.ax_speed.cla()

        # Clear the previous vehicle plot but keep the map
        self.ax_xy.cla()

        # Plot the navigation information
        if navigation_array is not None:
            self.ax_xy.plot(navigation_array[:, 0], navigation_array[:, 1], 'g--', marker='x', label="Reference Trajectory")
            if not self.plot_yaw:
                self.ax_yaw.plot(navigation_array[:, 2], navigation_array[:, 3], 'g--', marker='x', label="Reference Trajectory")
            else:
                # Unwrap angles to remove discontinuities
                angles = continuous_angles(np.arctan2(navigation_array[:, 3], navigation_array[:, 2]), initial_angle=self.prev_yaw)

                # Plot the angles
                self.ax_yaw.plot(angles, 'g--', marker='x', label="Reference Trajectory")

            self.ax_speed.plot(navigation_array[:, 4], 'g--', marker='x', label="Reference Trajectory")

        # Plot the side information zonotope on yaw and speed plots
        if side_zonotopes is not None:
            self.plot_6D_zonotopes(side_zonotopes, color='red', alpha=0.1, label="Side Information")

        if hd_zonotope is not None:
            self.plot_hd_info(hd_zonotope, color='red', alpha=0.1, label="HD Information")

        # Plot the predicted trajectory for the model-based reachability analysis
        if reach_zonotopes is not None:
            self.plot_6D_zonotopes(reach_zonotopes, color='yellow', alpha=0.1, label="Reachable Sets", plot_center_reach=plot_center_reach)

        # Plot the STL interval if provided
        if stl_interval is not None:
            self.plot_6D_interval(stl_interval, color='green', alpha=0.1, label="STL Information")

        if obstacle_interval is not None:
            print("Plotting obstacle interval...")
            print("Obstacle Interval:", obstacle_interval)
            self.plot_obstacle_interval(obstacle_interval, color='black', alpha=0.8, label="Obstacle Information")

        # Plot the vehicle in the XY plane (Left side)
        if current_state is not None:
            self.ax_xy = plot_vehicle(current_state=current_state[:5], ax=self.ax_xy, control_command=optimal_control,
                                      failure_mode=failure_mode)

        # Plot the optimal states
        if optimal_states is not None:
            self.plot_optimal_states(optimal_states, is_trajectory=is_trajectory)

        if plot_trace:
            # Plot the trace of the vehicle
            if current_state is not None:
                self.x_trace.append(current_state[0])
                self.y_trace.append(current_state[1])
                self.ax_xy.plot(list(self.x_trace), list(self.y_trace), 'k-', alpha=0.5, label="Vehicle Trace")

        # Set parameters for the plots
        self.ax_xy = self.set_parameters(self.ax_xy, title="Position Plane", xlim=(current_state[0]-8, current_state[0]+15), ylim=(current_state[1]-10, current_state[1]+10), 
                            xlabel="X [m]", ylabel="Y [m]")
        
        if not self.plot_yaw:
            self.ax_yaw = self.set_parameters(self.ax_yaw, title="Yaw Plane", xlim=(-1, 1), ylim=(-1, 1), 
                                xlabel="cos(yaw)", ylabel="sin(yaw)", grid=True)
        else:
            self.ax_yaw = self.set_parameters(self.ax_yaw, title="Yaw vs Time Step", xlim=(0, steps-1), xticks=(steps), 
                                              ylim=(self.prev_yaw-3, self.prev_yaw+3), xlabel=r"Time step ($k$)", ylabel=r"$\theta$ [rad]", grid=True)
            
        self.ax_speed = self.set_parameters(self.ax_speed, title="Speed vs Time Step", xlim=(0, steps-1), ylim=(0, 15), xticks=(steps), 
                            xlabel=r"Time step ($k$)", ylabel=r"$v_x$ [m/s]", grid=True)
        
        self.add_figure_legend()

        # Save the plot if the flag is set
        if self.save_plots:
            filename = f"{self.plots_folder}/frame_{len(self.saved_plot_files):03d}.png"
            self.fig.savefig(filename, dpi=150)
            self.saved_plot_files.append(filename)

        self.iteration_count += 1

        if self.show_plots:
            # Draw the updated figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.debug:
            print("\n==============================================")
            print("Current State:", current_state)
            print("Optimal Next Input:", optimal_control)
            print("Side Information:", hd_zonotope)
            print("STL Information:", stl_interval)
            print("Reachable Sets:", reach_zonotopes)
            print("==============================================\n")

        # Update the previous yaw value for the next iteration
        if self.plot_yaw:
            self.prev_yaw = continuous_angles(np.arctan2(current_state[3], current_state[2]), initial_angle=self.prev_yaw)

        
