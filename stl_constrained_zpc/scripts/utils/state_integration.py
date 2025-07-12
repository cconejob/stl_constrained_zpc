import numpy as np
import matplotlib.pyplot as plt


class Integrator():
    def __init__(self, x0=None, x_dot0=None, x_dotdot0=None, dt=0.1, method='rk4'):
        self.x_dotdot0 = np.zeros(3) if x_dotdot0 is None else x_dotdot0
        self.x_dot0 = np.zeros(3) if x_dot0 is None else x_dot0
        self.x0 = np.zeros(3) if x0 is None else x0
        self.dt = dt
        self.method = method.lower()
        
    def euler(self, x0, dx, dt, dx0=None, method='trapezoidal'):
        """
        Integrate one state of the vehicle. Euler integration.

        Args:
            x0 (float): Current state of the vehicle. [longitudinal, lateral, yaw]
            dx (float): State derivative of the vehicle. [longitudinal, lateral, yaw]
            dt (float): Time step. [s]
            dx0 (float, optional): Previous state derivative of the vehicle. [longitudinal, lateral, yaw]. Defaults to None.
            method (str, optional): Integration method. Defaults to 'trapezoidal'. Options: ['trapezoidal', 'end', 'start']

        Returns:
            float: Integrated state of the vehicle. [longitudinal, lateral, yaw]
        """
        method = method.lower()
        
        if method == 'trapezoidal':
            dx = (dx + dx0)/2 if dx0 is not None else dx
        
        elif method == 'end':
            dx = dx
            
        elif method == 'start':
            dx = dx0 if dx0 is not None else dx
            
        else:
            raise NotImplementedError(f"Method {method.lower()} not implemented yet.")
        
        return x0 + dx * dt

    def rk4(self, x0, dx, dt, dx0):
        """
        Integrate one state of the vehicle. Runge-Kutta 4th order integration.

        Args:
            x0 (float): Current state of the vehicle. [longitudinal, lateral, yaw]
            dx (float): State derivative of the vehicle. [longitudinal, lateral, yaw]
            dt (float): Time step. [s]
            dx0 (float): Previous state derivative of the vehicle. [longitudinal, lateral, yaw]

        Returns:
            float: Integrated state of the vehicle. [longitudinal, lateral, yaw]
        """
        k1 = dt * dx0
        k2 = dt * (dx0+dx)/2
        k3 = dt * (dx0+dx)/2
        k4 = dt * dx
        k = (k1+2*k2+2*k3+k4)/6
        x1 = x0 + k
        
        return x1

    def integrate_accelerations(self, x_dotdot1, x_dot0=None, x_dotdot0=None, dt=None):
        """
        Integrate the current accelerations of the vehicle with the specified method.

        Args:
            x_dotdot1 (np.array): Current accelerations of the vehicle. [longitudinal, lateral, yaw]
            x_dot0 (np.array, optional): Current velocities of the vehicle. [longitudinal, lateral, yaw]. Defaults to None.
            x_dotdot0 (np.array, optional): Current accelerations of the vehicle. [longitudinal, lateral, yaw]. Defaults to None.
            dt (float, optional): Time step. [s]. Defaults to None.

        Returns:
            np.array: Current vehicle velocities. [longitudinal, lateral, yaw]
        """
        x_dot0 = self.x_dot0 if x_dot0 is None else x_dot0
        x_dotdot0 = self.x_dotdot0 if x_dotdot0 is None else x_dotdot0
        dt = self.dt if dt is None else dt
        
        if self.method.lower() == 'euler':
            x_dot1 = np.vectorize(self.euler)(x_dot0, x_dotdot1, self.dt, x_dotdot0, method="start")
        elif self.method.lower() == 'rk4':
            x_dot1 = np.vectorize(self.rk4)(x_dot0, x_dotdot1, self.dt, x_dotdot0)
        else:
            raise NotImplementedError(f"Method {self.method.lower()} not implemented yet.")
        
        self.x_dotdot0 = x_dotdot1
            
        return x_dot1

    def integrate_state_with_orientation(self, x_dot1, x0=None, x_dot0=None, dt=None):
        """
        Integrate the state of the vehicle.

        Args:
            x_dot1 (np.array): Current velocities of the vehicle. [longitudinal, lateral, yaw]
            x0 (np.array, optional): Current state of the vehicle. [longitudinal, lateral, yaw]. Defaults to None.
            x_dot0 (np.array, optional): Current velocities of the vehicle. [longitudinal, lateral, yaw]. Defaults to None.
            dt (float, optional): Time step. [s]. Defaults to None.

        Returns:
            np.array: Integrated state of the vehicle. [longitudinal, lateral, yaw]
        """
        x0 = self.x0 if x0 is None else x0
        x_dot0 = self.x_dot0 if x_dot0 is None else x_dot0
        dt = self.dt if dt is None else dt
        
        x0, y0, yaw0 = x0
        vx, vy, yaw_rate = x_dot1
        
        if self.method.lower() == 'euler':
            yaw = self.euler(yaw0, yaw_rate, dt, self.x_dot0[2])
        elif self.method.lower() == 'rk4':
            yaw = self.rk4(yaw0, yaw_rate, dt, self.x_dot0[2])
        else:
            raise NotImplementedError(f"Method {self.method.lower()} not implemented yet.")

        yaw_mean = (yaw0 + yaw)/2
        x = x0 + vx * self.dt * np.cos(yaw_mean) - vy * self.dt * np.sin(yaw_mean)
        y = y0 + vx * self.dt * np.sin(yaw_mean) + vy * self.dt * np.cos(yaw_mean)
        
        self.x_dot0 = x_dot1
        
        return np.array([x, y, yaw])
    

def main():
    #define function
    def f(x,y):
        return x + y

    #initial conditions
    x0 = 0
    y0 = 1

    #step size
    h = 0.1

    #list of x values
    x = np.arange(x0, 10, h)

    #list of y values
    y = [y0]

    #calculate y for each x
    for i in range(1, len(x)):
        y.append(y[i-1] + h*f(x[i-1], y[i-1]))

    #plot
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
if __name__ == '__main__':
    main()