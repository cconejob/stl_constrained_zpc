import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import deque
import os
import numpy as np

from stl_constrained_zpc.scripts.utils.utils import admissible_yaw_intervals, continuous_angles


class ZPCTimePlotter:
    def __init__(self, show_plots=True, save_plots=False, plots_folder="./plots", maxlen=300, frequency=10, plot_yaw=False):
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.plots_folder = plots_folder
        self.frequency = frequency
        self.iteration_count = 0
        self.prev_failure_mode = False
        self.plot_yaw = plot_yaw
        self.maxlen = maxlen

        # History
        self.time = deque(maxlen=maxlen)
        self.state_data = {
            'x': deque(maxlen=maxlen),
            'y': deque(maxlen=maxlen),
            'cos_yaw': deque(maxlen=maxlen),
            'sin_yaw': deque(maxlen=maxlen),
            'yaw': deque(maxlen=maxlen),
            'speed': deque(maxlen=maxlen),
        }
        self.stl_data = {
            'x': deque(maxlen=maxlen),
            'y': deque(maxlen=maxlen),
            'cos_yaw': deque(maxlen=maxlen),
            'sin_yaw': deque(maxlen=maxlen),
            'yaw': deque(maxlen=maxlen),
            'speed': deque(maxlen=maxlen),
            't': deque(maxlen=maxlen),
        }
        self.obstacle_data = {
            'x': deque(maxlen=maxlen),
            'y': deque(maxlen=maxlen),
            't': deque(maxlen=maxlen),
        }

        # Create figure and axes
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle("Temporal Evolution of the AV", fontsize=18)
        gs = gridspec.GridSpec(2, 2)
        gs.update(hspace=0.4) # Adjust vertical space between subplots
        gs.update(wspace=0.2) # Adjust horizontal space between subplots
        self.ax_x = self.fig.add_subplot(gs[0,0])
        self.ax_y = self.fig.add_subplot(gs[1,0])
        self.ax_yaw = self.fig.add_subplot(gs[0,1])
        self.ax_speed = self.fig.add_subplot(gs[1,1])
        self.legend_set = False

        # Pre-initialize line objects
        if not self.plot_yaw:
            self.lines = {
                'x': self.ax_x.plot([], [], '-', label="X", color='orange')[0],
                'y': self.ax_y.plot([], [], 'g-', label="Y")[0],
                'cos_yaw': self.ax_yaw.plot([], [], 'c-', label="cos(yaw)")[0],
                'sin_yaw': self.ax_yaw.plot([], [], 'm-', label="sin(yaw)")[0],
                'speed': self.ax_speed.plot([], [], 'b-', label="Speed")[0],
            }
        else:
            self.lines = {
                'x': self.ax_x.plot([], [], '-', label="X", color='orange')[0],
                'y': self.ax_y.plot([], [], 'g-', label="Y")[0],
                'yaw': self.ax_yaw.plot([], [], 'm-', label="Yaw")[0],
                'speed': self.ax_speed.plot([], [], 'b-', label="Speed")[0],
            }

        # STL fill_between placeholders (as PolyCollections)
        if not self.plot_yaw:
            self.stl_fills = {
                'x': self.ax_x.fill_between([], [], [], alpha=0.2, label="STL x", color='orange'),
                'y': self.ax_y.fill_between([], [], [], alpha=0.2, label="STL y", color='g'),
                'cos_yaw': self.ax_yaw.fill_between([], [], [], alpha=0.2, label="STL cos(yaw)", color='c'),
                'sin_yaw': self.ax_yaw.fill_between([], [], [], alpha=0.2, label="STL sin(yaw)", color='m'),
                'speed': self.ax_speed.fill_between([], [], [], alpha=0.2, label="STL speed", color='b'),
            }
        else:
            self.stl_fills = {
                'x': self.ax_x.fill_between([], [], [], alpha=0.2, label="STL x", color='orange'),
                'y': self.ax_y.fill_between([], [], [], alpha=0.2, label="STL y", color='g'),
                'yaw': self.ax_yaw.fill_between([], [], [], alpha=0.2, label="STL yaw", color='m'),
                'speed': self.ax_speed.fill_between([], [], [], alpha=0.2, label="STL speed", color='b'),
            }

        # Obstacle fill_between placeholders (as PolyCollections)
        self.obstacle_fills = {
            'x': self.ax_x.fill_between([], [], [], alpha=0.2, label="Obstacle x", color='orange'),
            'y': self.ax_y.fill_between([], [], [], alpha=0.2, label="Obstacle y", color='g'),
        }

        self.saved_plot_files = []
        if self.save_plots and not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)

        self.prev_yaw = None

        # Set plot limits
        self.set_limits()

    def set_limits(self):
        self.ax_x.set_title("X vs Time", fontsize=16)
        self.ax_y.set_title("Y vs Time", fontsize=16)

        if not self.plot_yaw:
            self.ax_yaw.set_title("Cos(yaw)/Sin(yaw) vs Time", fontsize=16)
            self.ax_yaw.set_ylabel("cos(yaw) and sin(yaw)", fontsize=14)
            self.ax_yaw.set_ylim(-1, 1)
        else:
            self.ax_yaw.set_title("Yaw vs Time", fontsize=16)
            self.ax_yaw.set_ylabel("Yaw [rad]", fontsize=14)
            self.ax_yaw.set_ylim(-3.5, 3.5)

        self.ax_speed.set_title("Speed vs Time", fontsize=16)

        for ax in [self.ax_x, self.ax_y, self.ax_yaw, self.ax_speed]:
            ax.set_xlabel("Time [s]", fontsize=14)
            ax.grid(True)

            # Avoid decimal numbers in the ticks
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=10)

        self.ax_x.set_ylabel("X [m]", fontsize=14)
        self.ax_y.set_ylabel("Y [m]", fontsize=14)
        self.ax_speed.set_ylabel("Speed [m/s]", fontsize=14)
        
        self.ax_speed.set_ylim(0, 15)

    def forward(self, current_state=None, stl_interval=None, failure_mode=False, obstacle_interval=None):
        self.ax_x.set_ylim(current_state[0] - 50, current_state[0] + 50)
        self.ax_y.set_ylim(current_state[1] - 20, current_state[1] + 20)
        
        t = self.iteration_count / self.frequency
        if self.prev_yaw is None:
            self.prev_yaw = np.arctan2(current_state[3], current_state[2])

        if not self.plot_yaw:
            states = ['x', 'y', 'cos_yaw', 'sin_yaw', 'speed']
            plots = [self.ax_x, self.ax_y, self.ax_yaw, self.ax_yaw, self.ax_speed]
        else:
            states = ['x', 'y', 'yaw', 'speed']
            plots = [self.ax_x, self.ax_y, self.ax_yaw, self.ax_speed]
            states_stl = ['x', 'y', 'cos_yaw', 'sin_yaw', 'speed']

        # === Add state ===
        self.time.append(t)
        if current_state is not None:
            if len(self.state_data['x']) < 1:
                self.initial_x = current_state[0]
                self.initial_y = current_state[1]

            self.state_data['x'].append(current_state[0])
            self.state_data['y'].append(current_state[1])
            self.state_data['cos_yaw'].append(current_state[2])
            self.state_data['sin_yaw'].append(current_state[3])
            self.state_data['yaw'].append(np.arctan2(current_state[3], current_state[2]))
            self.state_data['yaw'] = deque(continuous_angles(self.state_data['yaw'], initial_angle=self.prev_yaw), maxlen=self.maxlen)
            self.state_data['speed'].append(current_state[4])
            self.prev_yaw = self.state_data['yaw'][-1]

        # === Add STL ===
        if stl_interval is not None:
            for key, idx in zip(states_stl, range(len(states_stl))):
                if stl_interval[idx,0] is None:
                    print(f"STL interval for {key} is None")
                    continue
                else:
                    if idx == 0:
                        self.stl_data['t'].append(t)
                    if self.plot_yaw and 'cos_yaw' in key:
                        if stl_interval[2,0] is None or stl_interval[3,0] is None:
                            print(f"STL interval for yaw is None")
                            continue
                        interval_yaw = admissible_yaw_intervals([stl_interval[2,0].inf[0], stl_interval[2,0].sup[0]], [stl_interval[3,0].inf[0], stl_interval[3,0].sup[0]], prev_yaw=self.prev_yaw)
                        self.stl_data['yaw'].append([interval_yaw[0], interval_yaw[1]])
                    else:
                        self.stl_data[key].append([stl_interval[idx,0].inf[0], stl_interval[idx,0].sup[0]])

        # === Add Obstacles ===
        if obstacle_interval is not None:
            for key, idx in zip(states[:2], range(len(states[:2]))):
                if obstacle_interval[idx][0] is None:
                    print(f"obstacle interval for {key} is None")
                    continue
                else:
                    if idx == 0:
                        self.obstacle_data['t'].append(t)
                    self.obstacle_data[key].append([obstacle_interval[idx][0].inf, obstacle_interval[idx][0].sup])

        # === Update state lines ===
        for key in states:
            self.lines[key].set_data(self.time, self.state_data[key])

        self.ax_x.set_xlim(left=max(0, t - 15), right=t)
        self.ax_y.set_xlim(left=max(0, t - 15), right=t)
        self.ax_yaw.set_xlim(self.ax_x.get_xlim())
        if self.plot_yaw:
            self.ax_yaw.set_ylim(self.state_data['yaw'][-1] - 3.5, self.state_data['yaw'][-1] + 3.5)
        self.ax_speed.set_xlim(self.ax_x.get_xlim())

        # === Update STL fill_between ===
        def update_fill_stl(ax, key):
            t_vals = list(self.stl_data['t'])
            bounds = np.array(self.stl_data[key])
            if bounds.shape[0] == 0:
                return
            if hasattr(self.stl_fills[key], "remove"):
                self.stl_fills[key].remove()

            self.stl_fills[key] = ax.fill_between(t_vals, bounds[-1, 0], bounds[-1, 1], alpha=0.2, label=f"STL {key}", color=self.lines[key].get_color())

        for ax, key in zip(plots, states):
            update_fill_stl(ax, key)

        # === Update Obstacles fill_between ===
        def update_fill_obstacle(ax, key):
            t_vals = list(self.obstacle_data['t'])
            bounds = np.array(self.obstacle_data[key])
            if bounds.shape[0] == 0:
                return
            if hasattr(self.obstacle_fills[key], "remove"):
                self.obstacle_fills[key].remove()

            self.obstacle_fills[key] = ax.fill_between(t_vals, bounds[-1, 0], bounds[-1, 1], alpha=0.5, label=f"Obstacle {key}", color='black')

        for ax, key in zip(plots[:2], states[:2]):
            update_fill_obstacle(ax, key)

        # === Failure mode indicator ===
        if failure_mode and not self.prev_failure_mode:
            for ax in [self.ax_x, self.ax_y, self.ax_yaw, self.ax_speed]:
                ax.axvline(x=t, color='red', linestyle='--', linewidth=1, label="Failure Mode")
                self.legend_set = False  # Reset legend to show failure mode line
        self.prev_failure_mode = failure_mode

        # === Finalize plot ===
        self.add_legend_once()

        if self.save_plots:
            fname = f"{self.plots_folder}/frame_time_{len(self.saved_plot_files):03d}.png"
            self.fig.savefig(fname, dpi=150)
            self.saved_plot_files.append(fname)

        if self.show_plots:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        self.iteration_count += 1

    def add_legend_once(self):
        # Add legend only once to avoid duplication
        if not self.legend_set:
            for ax in [self.ax_x, self.ax_y, self.ax_yaw, self.ax_speed]:
                ax.legend(loc='upper left', fontsize=12)
            self.legend_set = True