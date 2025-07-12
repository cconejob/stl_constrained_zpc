import numpy as np
from collections import deque
import math

from stl_constrained_zpc.scripts.reachability.vehicle_model.physics_based.ego_plotter import EgoPlotter
from stl_constrained_zpc.scripts.utils.utils import wrap_angle


class Ego(EgoPlotter):
    def __init__(self, vehicle_parameters, plot=bool(), title_plot=str(), plot_inputs=False):
        self.add_info(vehicle_parameters)
        self.plot = plot
        
        if plot:
            if plot_inputs:
                super().__init__(title=title_plot, rows=2, columns=2, plot_inputs=True)
                self.inputs_settings()
            else:
                super().__init__(title=title_plot, rows=2, columns=1, plot_inputs=False)
            self.states_settings()
            
        self.speed = 1e-3
            
        self.t = deque(maxlen=1000)
        self.trajectory = deque(maxlen=1000)
        self.speed_vector = deque(maxlen=1000)
        self.vx_min = 1e-2
        self.vy_min = 1e-5
        self.speed_angle_min = 1e-3
        
    def add_info(self, d):
        """
        It only creates attributes if all parameters are in the correct type and value range.
        """
        for k, v in d.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        
    def update_inputs(self, u):
        """
        Store the current inputs of the vehicle.

        Args:
            u (numpy.Array): Current inputs of the vehicle (steering angle [rad], longitudinal acceleration [m/s^2]).
        """
        self.steering_angle = wrap_angle(u[0])
        self.longitudinal_acceleration = u[1]
        
        if self.plot:
            self.plot_raw_steer(self.t, self.steering_angle)
            self.plot_raw_ax(self.t, self.longitudinal_acceleration)
        
    def update_states(self, x, dt, n_states=6):
        """
        Store the current states of the vehicle.
        
        Args:
            x (numpy.Array): Current states of the vehicle (x [m], y [m], yaw [rad], speed [m/s], yaw rate [rad/s], lateral speed [m/s]).
            n_states (int, optional): Number of states of the vehicle. Defaults to 6 (x, y, yaw, vx, vy, w). Options: 4 (x, y, yaw, v).
        """
        if len(self.t) == 0:
            self.t.append(0)
        else:
            self.t.append(self.t[-1] + dt)
            
        self.x = x[0]
        self.y = x[1]
        self.yaw = x[2]
        
        if n_states > 4 and len(x) > 4:
            self.longitudinal_speed = x[3] if x[3] > 0. else 0.
            self.lateral_speed = x[4] if self.longitudinal_speed > self.vx_min else 0.
            self.yaw_rate = x[5] if self.longitudinal_speed > self.vx_min else 0.
            self.speed = math.sqrt(self.longitudinal_speed**2 + self.lateral_speed**2)*np.sign(self.longitudinal_speed)
            self.w_wheel = self.wheel_angular_velocity()
        else:
            self.speed = x[3] if x[3] > self.vx_min else self.vx_min
            self.longitudinal_speed = self.speed
            self.lateral_speed = 0.
            self.yaw_rate = 0.
            self.w_wheel = self.wheel_angular_velocity()
        
        self.trajectory.append([self.x, self.y])
        self.speed_vector.append([self.speed])
        
        if n_states > 4:
            self.state = np.array([self.x, self.y, self.yaw, self.longitudinal_speed, self.lateral_speed, self.yaw_rate])
        else:
            self.state = np.array([self.x, self.y, self.yaw, self.speed])
        
    def slip_angle_lat_rear(self):
        """
        Calculate the lateral slip angle of the rear tires.
            
        Returns:
            float: Slip angle of the rear tires. [rad]
        """
        vx = self.longitudinal_speed; vy = self.lateral_speed; w = self.yaw_rate; speed_angle = self.speed_angle_nl()
        lr = self.lr
        
        if vx > self.vx_min and abs(speed_angle) > self.speed_angle_min:
            slip_angle_lat_rear = math.atan2((vy - lr*w),vx)
        else:
            slip_angle_lat_rear = 0.
        
        return slip_angle_lat_rear
    
    def slip_angle_lat_front(self):
        """
        Calculate the lateral slip angle of the front tires.

        Returns:
            float: Slip angle of the front tires. [rad]
        """
        vx = self.longitudinal_speed; vy = self.lateral_speed; w = self.yaw_rate
        steer = self.steering_angle; speed_angle = self.speed_angle_nl()
        lf = self.lf
        
        if vx > self.vx_min and abs(speed_angle) > self.speed_angle_min:
            slip_angle_lat_front = math.atan2((vy + lf*w),vx) - steer
        else:
            slip_angle_lat_front = 0.
        
        return slip_angle_lat_front
    
    def slip_angle_long_rear(self):
        """
        Calculate the longitudinal slip angle of the rear tires.
        
        Returns:
            float: Longitudinal slip angle of the rear tires. [rad]
        """
        vx = self.longitudinal_speed; w_wheel = self.w_wheel
        R_eff = self.R_eff
        
        return np.sign(w_wheel*R_eff - vx)*abs((w_wheel*R_eff - vx)/max(vx, w_wheel*R_eff)) if vx > self.vx_min else 1e-10
        
    def tire_forces(self):
        """
        Calculate the tire force of the rear tires.

        Returns:
            (float, float, float): Tire forces (Fyf, Fyr, Fxr). 
        """
        Fyf = - self.Cf * self.slip_angle_lat_front()
        Fyr = - self.Cr * self.slip_angle_lat_rear()
        Fxr = self.Cr * self.slip_angle_long_rear()

        return Fyf, Fyr, Fxr
    
    def longitudinal_forces(self):
        """
        Calculate the longitudinal forces of the vehicle.

        Returns:
            (float, float): Longitudinal forces (F_rolling, F_aero). 
        """
        F_rolling = self.rolling_force()
        F_aero = self.drag_force()
        
        return F_rolling, F_aero
    
    def rolling_force(self):
        """
        Calculate the rolling resistance force.
        
        Returns:
            float: Rolling resistance force. [N]
        """
        vx = self.longitudinal_speed
        
        return self.m * self.g * self.mu
    
    def drag_force(self):
        """
        Calculate the aerodynamic drag force.
        
        Returns:
            float: Aerodynamic force. [N]
        """
        vx = self.longitudinal_speed
        
        return 0.5 * self.Cd * self.rho * self.Ad * vx**2
    
    def wheel_angular_velocity(self):
        """
        Calculate the angular velocity of the wheels.
        
        Returns:
            float: Angular velocity of the wheels [rad/s]
        """
        vx = self.longitudinal_speed
        R_eff = self.R_eff
        
        return vx/R_eff
    
    def speed_angle_nl(self):
        """
        Get the speed angle of the vehicle from the steering angle with the non-linear equation.

        Returns:
            float: Speed angle [rad]
        """
        lr = self.lr; lf = self.lf
        steer = self.steering_angle
        
        return np.sign(steer) * abs(math.atan2((lr*math.tan(steer)),(lf+lr)))
    
    def speed_angle_linearlized(self, steer):
        """
        Get the speed angle of the vehicle from the steering angle with the linearlized equation.

        Returns:
            float: Speed angle [rad]
        """
        lr = self.lr; lf = self.lf
        
        return np.sign(steer) * abs(lr/(lf+lr)*math.tan(steer))
    
    def plot_trajectory(self):
        """
        Plot the trajectory and the speed of the vehicle.        
        """
        traj = np.array(self.trajectory)  
        self.plot_xy(x=traj[:,0], y=traj[:,1])
        self.plot_speed(t=np.array(self.t), v=np.array(self.speed_vector))
        self.finish_simulation()
    