import numpy as np
from math import sin, cos

from stl_constrained_zpc.scripts.reachability.vehicle_model.physics_based.ego import Ego
from stl_constrained_zpc.scripts.utils.state_integration import *


class NLKinematicBicycleModel(Ego):
    """
    Nonlinear kinematic bicycle model for the ego vehicle. All forces are calculated in the vehicle frame.

    Args:
        Ego (class): Ego vehicle object.
        vehicle_parameters (dict): Vehicle parameters.
        n_states (int): Number of states to use in the model. 4 for [x, y, yaw, v], 6 for [x, y, yaw, v, vx, vy].
        
    Attributes:
        dt (float): Time step [s].
    """
    def __init__(self, vehicle_parameters, dt=0.01, n_states=4):
        self.dt = dt
        self.n_states = n_states
        self.integrator = Integrator(dt=dt, method='rk4')
        super().__init__(vehicle_parameters, plot=False, title_plot="NL kin bicycle model")

    def get_next_state(self):
        """
        Update the state of the vehicle.

        Returns:
            np.array: Updated state of the vehicle. [x, y, yaw, x_dot, y_dot, yaw_dot]
        """
        # Get current velocities and states
        states_0 = self.state[:3]
        velocity_0 = self.state[3]
        ax = self.longitudinal_acceleration
        
        # Integrate the accelerations to get the next velocities
        velocities, speed = self.get_velocities(velocity_0, ax)

        # Integrate the velocities to get the states
        states = self.integrator.integrate_state_with_orientation(velocities, x0=states_0)
        states[2] = (states[2])
        
        # Concatenate the states and velocities
        predicted_state = np.hstack((states, np.array([speed])))

        return predicted_state
        
    def get_velocities(self, v0, ax):
        """
        Get the next velocities of the vehicle given the current state and input with the Kinematic Non-Linear Bicycle model approach.

        Args:
            v0 (np.array): Current speed of the vehicle [m/s]
            ax (float): Longitudinal acceleration of the vehicle [m/s^2]

        Returns:
            np.array: Next velocities of the vehicle. [vx, vy, yaw_rate]
            float: Next speed of the vehicle [m/s]
        """
        beta = self.speed_angle_nl()
        v = self.integrator.euler(v0*cos(beta), ax, self.dt)/cos(beta)
        v = 1e-6 if v < 1e-6 else v
        
        return np.array([v*cos(beta), v*sin(beta), v*sin(beta)/self.lr]), v
    
    def predict(self, x0, u):
        """
        Get the next state of the vehicle given the current state and input with the Kinematic Non-Linear Bicycle model approach.

        Args:
            x0 (np.array): Current state of the vehicle. [x, y, yaw, vx, vy, w]
            u (np.array): Input vector of the vehicle. [steering, acceleration]

        Returns:
            np.array: Next state of the vehicle. [x, y, yaw, vx, vy, w]
        """
        # Store the current state and input
        self.update_inputs(u)
        self.update_states(x0, self.dt, self.n_states)
        
        # Update the state of the vehicle using the accelerations and the current state integrating the state second derivatives
        predicted_state = self.get_next_state()
        
        return predicted_state
    
    
def main():
    return

    
if __name__ == "__main__":    
    main()