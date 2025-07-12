import matplotlib.pyplot as plt
import numpy as np
import time

    
class EgoPlotter():
    def __init__(self, title=str(), rows=2, columns=3, plot_inputs=True):
        self.title = title.lower()
        self.define_plot(rows, columns)
        self.plot_inputs_enabled = plot_inputs
        self.columns = columns
        self.rows = rows
        
    def define_plot(self, num_rows, num_columns):
        self.fig, self.ax = plt.subplots(num_rows, num_columns)
        self.fig.suptitle(self.title, fontsize=20)
        
    def plot_states(self, t, states, raw=False, noise=False):    
        style = "-"    
        linewidth = 1
        label = self.title
        if "kin" in self.title:
            color = "r"
        else:
            color = "b"

        if raw:
            color = "g"
            linewidth += 1
            label = "real"
        elif noise:
            color = "y"
            style = "--"
            linewidth += 1
            label = "noisy"

        if self.columns > 1:
            self.ax[0,0].plot(states[:,0], states[:,1], style, color=color, label=label, linewidth=linewidth)
            self.ax[0,0].axis('equal')
            self.ax[1,0].plot(t, states[:,3], style, color=color, label=label, linewidth=1)
        else:
            self.ax[0].plot(states[:,0], states[:,1], style, color=color, label=label, linewidth=linewidth)
            self.ax[0].axis('equal')
            self.ax[1].plot(t, states[:,3], style, color=color, label=label, linewidth=1)
        
    def plot_states_ego(self, t, x, y, yaw, v):
        st = np.array([x, y, yaw, v])
        style = "-"
        color = "kin"
        
        if self.columns > 1: 
            self.ax[0,0].plot(st[0], st[1], style, color=color, linewidth=2)
            self.ax[0,0].axis('equal')
            self.ax[1,0].plot(t, st[3], style, color=color, linewidth=2)
        else:
            self.ax[0].plot(st[0], st[1], style, color=color, linewidth=2)
            self.ax[0].axis('equal')
            self.ax[1].plot(t, st[3], style, color=color, linewidth=2)
         
    def states_settings(self):
        if self.columns > 1: 
            self.set_subplot(self.ax[0,0], xlabel="x [m]", ylabel="y [m]", title="trajectory")
            self.set_subplot(self.ax[1,0], ylabel="v [m/s]", title="total speed", ylim=[0., 10.])
        else:
            self.set_subplot(self.ax[0], xlabel="x [m]", ylabel="y [m]", title="trajectory")
            self.set_subplot(self.ax[1], ylabel="v [m/s]", title="total speed", ylim=[-0., 10.])  
        
    def plot_inputs(self, t, inputs):
        if self.plot_inputs_enabled:
            inp = np.array(inputs.inputs)
            color = "g" if "norm" in inputs.mode else "r"
            self.ax[0,-1].plot(t, inp[:,0], color=color, label=inputs.mode, linewidth=1)
            self.ax[1,-1].plot(t, inp[:,1], color=color, label=inputs.mode, linewidth=1)
        
    def plot_raw_inputs(self, t, inputs, color="g", label="real"):
        if self.plot_inputs_enabled:
            self.ax[0,-1].plot(t, inputs[:,0], color=color, label=label, linewidth=1)
            self.ax[1,-1].plot(t, inputs[:,1], color=color, label=label, linewidth=1)
        
    def plot_raw_steer(self, t, steer):
        if self.plot_inputs_enabled:
            color = "r"
            self.ax[0,-1].plot(t, steer, color=color, label="sensor", linewidth=1)
        
    def plot_raw_ax(self, t, ax):
        if self.plot_inputs_enabled:
            color = "r"
            self.ax[1,-1].plot(t, ax, color=color, label="sensor", linewidth=1)
        
    def inputs_settings(self):
        if self.plot_inputs_enabled:
            self.set_subplot(self.ax[1,-1], ylabel="a [m/s^2]", ylim=[-4., 2.], title="long. acceleration")
            self.set_subplot(self.ax[0,-1], ylabel="steer [rad]", ylim=[-0.5, 0.5], title="steering angle")
            
    def plot_xy(self, x, y, style="b-", label=""):
        self.ax[0].plot(x, y, style, label=label)
        
    def plot_speed(self, t, v, style="b-", label=""):
        self.ax[1].plot(t, v, style, label=label)
            
    def set_subplot(self, ax, xlabel="t [s]", ylabel=str(), xlim=None, ylim=None, title=str()):
        ax.title.set_text(title)  
        ax.grid(True)          
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
    def restart_all_subplots(self):
        for ax in plt.gcf().get_axes():
            ax.cla() 
    
    def finish_simulation(self):
        date_string = time.strftime("%Y-%m-%d-%H:%M")
        self.fig.set_size_inches(22, 12)
        self.fig.savefig(f'/home/tda/Downloads/PhD_documents/Images/{self.title} {date_string}.png', dpi=100)
        print("\nSimulation finished correctly")