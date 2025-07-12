import numpy as np
import dill
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import argparse
import time
import os
import datetime

from stl_constrained_zpc.scripts.reachability.data_driven_reachability_analysis import DataDrivenReachabilityAnalysis
from stl_constrained_zpc.scripts.utils.utils import find_closest_idx_point, completion_bar, package_path
from stl_constrained_zpc.scripts.stl.stl_to_interval import add_stl_formula, run_stl_formulation
from stl_constrained_zpc.scripts.reference_trajectory.bezier import generate_bezier_with_velocities
from stl_constrained_zpc.scripts.visualization.zpc_visualizer import ZPCVisualizer
from stl_constrained_zpc.scripts.visualization.zpc_time_plotter import ZPCTimePlotter
from stl_constrained_zpc.scripts.reachability.vehicle_model.physics_based.nl_kinematic_bicycle_model import NLKinematicBicycleModel


# Global variables
ego_properties = {'lr': 1.449, 
                'lf': 1.139, 
                'm': 1664, 
                'Iz': 2198, 
                'Cr': 120000, 
                'Cf': 85000, 
                'g': 9.81, 
                'mu': 0.0256, 
                'Cd': 1.64, 
                'rho': 1.204, 
                'Ad': 0.475, 
                'R_eff': 0.3072, 
                'v_max': 13.89, 
                'steer_max': 0.5236, 
                'ax_max': 3.0, 
                'ax_min': -3.5, 
                'ay_max': 3.0}


class ZonotopicPredictiveControl:
    def __init__(self, nl_model: DataDrivenReachabilityAnalysis, steps_reach=2, num_states=6, num_inputs=2, steer_bounds=np.array([-0.52, 0.52]), accel_bounds=np.array([-3, 0.5]), 
                 plots_folder=None, show_plots=False, debug=False, plot_yaw=False):
        """
        Initializes the Data-driven Zonotopic Predictive Control (ZPC) with a precomputed data-driven reachability model.
        States are assumed to be in the form: [x, y, cos_yaw, sin_yaw, v, w].
        Control inputs are assumed to be in the form: [steer, acceleration].

        Args:
            nl_model (DataDrivenReachabilityAnalysis): Data-driven reachability model.
            steps_reach (int): Prediction horizon.
            num_states (int): Number of state variables.
            num_inputs (int): Number of control inputs.
            steer_bounds (np.ndarray): Steering angle bounds.
            accel_bounds (np.ndarray): Acceleration bounds.
            plots_folder (str): Folder to save the plots.
            show_plots (bool): Flag to show the plots.
            debug (bool): Flag to enable debugging.
        """
        self.nl_model = nl_model
        self.N = steps_reach
        self.num_states = num_states
        self.num_inputs = num_inputs
        save_plots = False if plots_folder is None else True
        self.debug = debug
        self.last_R_data = None

        # Give more weight to yaw as the error in yaw is lower than the error in x and y
        self.Qx = np.ones((num_states, num_states)); self.Qx[2, 2] = 100
        self.Qu = np.eye(num_inputs); self.Qu[0, 0] = 7.5
        self.u_bounds = np.vstack([steer_bounds, accel_bounds])

        # Initialize the ZPC Visualizer for plotting the results if required
        if save_plots or show_plots:
            self.zpc_visualizer = ZPCVisualizer(show_plots=show_plots, save_plots=save_plots, plots_folder=plots_folder, debug=debug, plot_states_time=True, plot_yaw=plot_yaw)
            self.zpc_time_plotter = ZPCTimePlotter(plots_folder=plots_folder, show_plots=show_plots, save_plots=save_plots, plot_yaw=plot_yaw)
            
        else:
            self.zpc_visualizer = None

        # Emergency command to stop the vehicle in the shape (N, num_inputs)
        self.emergency_cmd = np.tile([0, -4.5], (self.N, 1))

    def compute_optimal_reachability(self, current_state, optimal_control, data_driven=True, update_sliding_window=False):
        """
        Computes the optimal reachability for the given state and control input using the data-driven reachability model.

        Args:
            current_state (np.ndarray): Current state vector.
            optimal_control (np.ndarray): Optimal control input vector.
            data_driven (bool): Flag to indicate if the reachability is data-driven or not.

        Returns:
            list: List of zonotopes representing the reachability at each time step.
        """
        if data_driven and self.nl_model.sliding_window > 0:
            R_data, _ = self.nl_model.forward(current_state, optimal_control, update_sliding_window=update_sliding_window)
        else:
            R_data, _ = self.nl_model.forward(current_state, optimal_control)

        return R_data
    
    def extract_states_inputs(self, XU_flat):
        """
        Extracts the state and control input variables from the flattened array.
        Initial format: [X_0, X1, ..., X_N, U_0, U_1, ..., U_N-1, S_u_0, S_u_1, ..., S_u_N-1, S_l_0, S_l_1, ..., S_l_N-1]
        Final format: X = [X_0, X1, ..., X_N], U = [U_0, U_1, ..., U_N-1], S_u = [S_u_0, S_u_1, ..., S_u_N-1], S_l = [S_l_0, S_l_1, ..., S_l_N-1]
        Where X is the state vector and U is the control input vector.
        X = [x, y, cos_yaw, sin_yaw, v, w]
        U = [steer, acceleration]

        Args:
            XU_flat (np.ndarray): Flattened state and control input variables.
        
        Returns:
            np.ndarray: State variables. X shape: (N, num_states)
            np.ndarray: Control inputs. U shape: (N, num_inputs)
        """
        N = self.N; num_states = self.num_states; num_inputs = self.num_inputs

        X = XU_flat[:N * num_states].reshape((N, num_states))
        U = XU_flat[N * num_states:N * (num_states + num_inputs)].reshape((N, num_inputs))

        return X, U
    
    def initial_state_constraints(self, X0, XU_flat):
        """
        Initializes the linear constraints for the optimization problem.
        Enforces the initial state constraint: X0 = XU_flat[:num_states].

        Args:
            X0 (np.ndarray): Initial state vector.
            XU_flat (np.ndarray): Initial state and control input vector.

        Returns:
            np.ndarray: Constraint matrix A.
            np.ndarray: Lower bound vector lb.
            np.ndarray: Upper bound vector ub.
        """
        A_init = np.zeros((self.num_states, len(XU_flat)))
        A_init[:, :self.num_states] = np.eye(self.num_states) 
        lb_init = X0; ub_init = X0

        return A_init, lb_init, ub_init
    
    def control_input_bounds(self, XU_flat):
        """
        Initializes the linear constraints for the control input bounds.
        Enforces the control input bounds: lb_u <= U <= ub_u.

        Args:
            XU_flat (np.ndarray): Initial state and control input vector.

        Returns:
            np.ndarray: Constraint matrix A.
            np.ndarray: Lower bound vector lb.
            np.ndarray: Upper bound vector ub.
        """
        # Initialize the constraint matrix
        A_u_bounds = np.zeros((self.num_inputs * self.N, len(XU_flat)))
        
        # Identity matrix to apply constraints to control inputs
        for k in range(self.N):
            A_u_bounds[self.num_inputs * k:self.num_inputs * (k + 1), 
                    self.num_states * self.N + k * self.num_inputs:self.num_states * self.N + (k + 1) * self.num_inputs] = np.eye(self.num_inputs)

        # Fix bounds using repeat for proper broadcasting
        lb_u = np.tile(self.u_bounds[:, 0], self.N)
        ub_u = np.tile(self.u_bounds[:, 1], self.N)

        return A_u_bounds, lb_u, ub_u

    def state_bounds(self, XU_flat):
        """
        Initializes the linear constraints for the state bounds.
        Enforces the state bounds for yaw and speed: lb_x <= X <= ub_x.
        X = [x, y, cos_yaw, sin_yaw, v, w]
        It enforces the constraints for timesteps k=1 to N. (k=0 is the initial state)

        Args:
            XU_flat (np.ndarray): Initial state and control input vector. Format: [X_0, X1, ..., X_N, U_0, U_1, ..., U_N-1, S_u_0, S_u_1, ..., S_u_N-1, S_l_0, S_l_1, ..., S_l_N-1]

        Returns:
            np.ndarray: Constraint matrix A.
            np.ndarray: Lower bound vector lb.
            np.ndarray: Upper bound vector ub.
        """
        n_states_constrained = 3  # Number of states constrained (yaw and speed)
        # Initialize the constraint matrix
        A_x_bounds = np.zeros((n_states_constrained * (self.N-1), len(XU_flat)))

        # Yaw and Speed constraints
        for k in range(0, (self.N-1)):
            A_x_bounds[n_states_constrained * k, (k+1) * self.num_states + 2] = 1  # cos_yaw (X[:, 2])
            A_x_bounds[n_states_constrained * k + 1, (k+1) * self.num_states + 3] = 1  # sin_yaw (X[:, 3])
            A_x_bounds[n_states_constrained * k + 2, (k+1) * self.num_states + 4] = 1  # Speed (X[:, 4])

        # Lower and upper bounds for yaw and speed
        # cos_yaw and sin_yaw are bounded between -1 and 1
        # Speed is bounded between 0 and 30
        lb_x = np.tile([-1, -1, 0], (self.N-1)) 
        ub_x = np.tile([1, 1, 30], (self.N-1))

        return A_x_bounds, lb_x, ub_x
    
    
    def compute_control(self, X0, prev_u, X_ref, interval_stl=None):
        """
        Computes the optimal control input using ZPC with the data-driven reachability model.
        
        Args:
            X0 (np.ndarray): Initial state vector.
            prev_u (np.ndarray): Previous control input.
            interval_stl (list): List of intervals for the STL constraints.
        
        Returns:
            np.ndarray: Optimal control input [u0, ..., uN-1].
            np.ndarray: Optimal state trajectory [x0, ..., xN].
        """
        # Initial state and control input vector flattened for optimization
        XU_init = np.hstack([
            np.tile(X0, self.N),
            np.tile(prev_u, self.N)
        ])

        # **Linear Constraints**

        # Initial State Constraint: X0 = X[0]
        A_init, lb_init, ub_init = self.initial_state_constraints(X0, XU_init)

        # Control Input Bounds
        A_u_bounds, lb_u, ub_u = self.control_input_bounds(XU_init)

        # # Linear Constraints List
        linear_constraints = []
        linear_constraints = [
            LinearConstraint(A_init, lb_init, ub_init),  # X0 = X[0]
            LinearConstraint(A_u_bounds, lb_u, ub_u),    # Control input bounds
        ]
        
        # **Nonlinear Constraints**

        # Yaw Normalization Constraint: cos²(yaw) + sin²(yaw) = 1
        def yaw_normalization(XU_flat):
            """
            Enforces cos²(yaw) + sin²(yaw) = 1 for each timestep k in the trajectory.
            XU_flat is the decision variable vector containing all states and controls.
            X = [X0, X1, ..., XN], U = [U0, U1, ..., UN-1]

            Args:
                XU_flat (np.ndarray): Flattened state and control input variables.

            Returns:
                np.ndarray: Constraint violations for the yaw normalization.
            """
            X, _ = self.extract_states_inputs(XU_flat)
            
            constraints = []
            for k in range(1, self.N):
                constraints.append(X[k, 2] ** 2 + X[k, 3] ** 2 - 1)

            return np.array(constraints, dtype=np.float64)

        # Constraints for Reachability
        def reachability_constraint(XU_flat):
            """
            Computes the constraint violations for the reachability of the nonlinear model.

            Args:
                XU_flat (np.ndarray): Flattened state and control input variables.
            
            Returns:
                np.ndarray: Constraint violations for the reachability.
            """
            X, U = self.extract_states_inputs(XU_flat)
            constraints = []
            u_reach = U.T

            # Compute the reachability using the data-driven model
            R_data, _ = self.nl_model.forward(X[0], u_reach)
            
            for k in range(1, self.N):
                if R_data is not None:
                    reach_interval = R_data[k].to_interval()                    
                    
                    # Compute the slack constraints
                    slack_upper = reach_interval.sup.T[0] - X[k]
                    slack_lower = X[k] - reach_interval.inf.T[0]

                    constraints.extend(slack_upper.tolist())
                    constraints.extend(slack_lower.tolist())
                else:
                    constraints.extend([1] * self.num_states)
                    constraints.extend([1] * self.num_states)

            self.last_R_data = R_data
            
            return np.array(constraints, dtype=np.float64)

        # Define the nonlinear constraints
        nonlinear_constraints = []
        nonlinear_constraints = [NonlinearConstraint(reachability_constraint, 0, np.inf), NonlinearConstraint(yaw_normalization, 0, 0)]
        
        # Interval Constraint for the side information and STL requirements
        def interval_constraint(XU_flat, interval):
            """
            Computes the constraint violations for the interval constraints.

            Args:
                XU_flat (np.ndarray): Flattened state and control input variables.
                interval (list): List of intervals.

            Returns:
                np.ndarray: Constraint violations.
            """
            X, _ = self.extract_states_inputs(XU_flat)
            constraints = []
            if interval is None:
                return np.array(constraints, dtype=np.float64)

            for k in range(1, self.N):
                interval_k = interval[:,k]
                if interval_k is None:
                    continue
                
                for i in range(self.num_states):
                    if interval_k[i] is None:
                        continue

                    lower_constraint = np.array([X[k, i] - interval_k[i].inf.T[0]])
                    upper_constraint = np.array([interval_k[i].sup.T[0] - X[k, i]])

                    # Append constraints
                    constraints.extend(lower_constraint.tolist())
                    constraints.extend(upper_constraint.tolist())

            return np.array(constraints, dtype=np.float64)

        # Add the constrained zonotopes from the STL requirements to the nonlinear constraints
        if interval_stl is not None:
            nonlinear_constraints.append(NonlinearConstraint(lambda x: interval_constraint(x, interval_stl), 0, np.inf))
            if self.debug:
                print("Adding STL constraints")
        
        # Cost Function
        def cost_function(XU_flat):
            X, U = self.extract_states_inputs(XU_flat)
            
            # Control rate cost
            U = np.vstack([prev_u, U])
            steer_rate_cost = np.sum(np.diff(U[:, 0]) ** 2) * self.Qu[0, 0]
            acceleration_rate_cost = np.sum(np.diff(U[:, 1]) ** 2) * self.Qu[1, 1]
            control_rate_cost = steer_rate_cost + acceleration_rate_cost
            
            # Trajectory cost (penalizing the distance between the optimal reachability and the predicted state)
            trajectory_cost = 0

            trajectory_cost_xy = (np.linalg.norm(X[1:, :2] - X_ref[:(X.shape[0]-1), :2]) ** 2) * self.Qx[0, 0]
            trajectory_cost_yaw = (np.linalg.norm(X[1:, 2:4] - X_ref[:(X.shape[0]-1), 2:4]) ** 2) * self.Qx[2, 2]
            trajectory_cost_speed = (np.linalg.norm(X[1:, 4] - X_ref[:(X.shape[0]-1), 4]) ** 2) * self.Qx[4, 4]
            trajectory_cost += trajectory_cost_xy + trajectory_cost_yaw + trajectory_cost_speed

            # If reference trajectory is provided, compute the cost based on the reference trajectory
            if X_ref is not None:
                R_data = self.last_R_data
                if R_data is not None:
                    for k in range(1, self.N):
                        if R_data[k] is None:
                            trajectory_cost = 1e6
                        else:
                            center_point_yaw = R_data[k].center()[2:4].flatten()
                            trajectory_cost_yaw = (np.linalg.norm(center_point_yaw - X_ref[k, 2:4])) * self.Qx[2, 2]

                            trajectory_cost += trajectory_cost_speed

            return control_rate_cost + trajectory_cost

        # Solve the optimization problem using the Sequential Least Squares Programming (SLSQP) method
        result = minimize(
            cost_function,
            XU_init,
            method="SLSQP",
            constraints=[*linear_constraints, *nonlinear_constraints],
            options={"disp": self.debug, 'maxiter': 20, 'ftol': 1e-1},
        )

        optimal_x, optimal_u = self.extract_states_inputs(result.x)
        
        if self.debug:
            print(f"Optimal control input: {optimal_u[0, :]}")
            print(f"Optimal state trajectory: {optimal_x[0, :]}")
        
        # Check if the optimization was successful
        if not result.success:
            if self.debug:
                print("\n=====================================")
                print("Optimization failed")
                print("Activating AEB")
                print("=====================================\n")
            return self.emergency_cmd, optimal_x

        return optimal_u, optimal_x


def run_zpc(data_list: list,
            zpc: ZonotopicPredictiveControl,
            idx: int,
            steps: int,
            new_stl_formula=None,
            index_plot=None,
            frequency=10,
            next_state=None,
            next_state_kin=None, 
            optimal_u=None,
            failure_mode=False,
            goal_state=None,
            simulation_mode=False,):
    """
    Run the reachability analysis for a random state from the test data from a loaded model.

    Args:
        data_list (list): List of data. [x0, x1, u]
        zpc (ZonotopicPredictiveControl): Zonotopic Predictive Control object.
        idx (int): Index of the test data.
        steps (int): Number of steps for the reachability analysis.
        cz_stl (list): Constrained zonotopes for the STL safety formulas.
        cz_side (list): Constrained zonotopes for the side information.
        index_plot (list): List of indexes to plot.
        frequency (int): Frequency of the control loop.
        next_state (np.ndarray): Next state of the vehicle.
        next_state_kin (np.ndarray): Next state of the vehicle in kinematic form.
        optimal_u (np.ndarray): Optimal control input.
        failure_mode (bool): Flag to indicate if the vehicle is in failure mode.
    """
    global stl_formulas, cz_intersection_stl_reach, debug, init_time, init_idx, prev_waypoints
    x0, x1, u = data_list[:3]  # Unpack the data list

    # ==== INITIALIZE ====
    current_state = x0[:, idx]
    next_states = x1[:, idx:idx+steps].T
    current_input = u[:, idx:idx+steps]
    current_timestep = init_time+((idx-init_idx)/frequency)

    # ==== STL FORMULAS ====
    stl_formulas = add_stl_formula(current_state=current_state, new_stl_formula=new_stl_formula,
                                   stl_formulas=stl_formulas, timestep=current_timestep)
    
    # Run the STL formulation
    interval_intersection_stl = run_stl_formulation(current_state=current_state[:zpc.num_states], stl_formulas=stl_formulas, 
                                                    steps_reach=steps, frequency=frequency, timestep=current_timestep)

    position_failure_mode = True
    # Check if trajectory xy is bounded by STL or it is None
    if np.all(interval_intersection_stl[0, :]== None)  and np.all(interval_intersection_stl[1, :] == None):
        position_failure_mode = False
    
    if np.all(interval_intersection_stl == None):
        interval_intersection_stl = None
    
    if not failure_mode and not simulation_mode:
        # Extract the current state, next states, and current input from the test data
        current_state_kin = current_state[:2].tolist() + [np.arctan2(current_state[3], current_state[2]), current_state[4]]
        
    else:
        if debug:
            print("\n=====================================")
            print("Failure Mode")
            print("=====================================\n")
        current_state = next_state; 
        current_state_kin = next_state_kin
        current_input = np.tile(optimal_u, (steps, 1)).T
        next_states = generate_bezier_with_velocities(current_state[:zpc.num_states], goal_state, N=steps, freq=frequency, order=5)
        prev_waypoints = None

    # Get the road properties for the navigation and side information

    # ==== NZPC CONTROL ====
    optimal_u = zpc.emergency_cmd
    reach_zonotopes = None
    optimal_x = next_states
    
    if failure_mode or simulation_mode:
        # Compute the optimal control input and state trajectory using ZPC
        t_init_control = time.time()
        optimal_u, optimal_x = zpc.compute_control(X0=current_state[:zpc.num_states], prev_u=current_input[:, 0], X_ref=next_states,
                                                    interval_stl=interval_intersection_stl)

        if zpc.debug:
            print(f"Time to compute ZPC: {time.time() - t_init_control}")

            # Save the computation time in a csv file
            with open(f"{zpc.zpc_visualizer.plots_folder}/computation_time.csv", "a") as f:
                f.write(f"{idx},{time.time() - t_init_control}\n")
        
    # Compute the reachability zonotopes
    reach_zonotopes = zpc.compute_optimal_reachability(current_state[:zpc.num_states], optimal_u.T, data_driven=True, update_sliding_window=True)
    
    if index_plot is not None:
        if idx in index_plot:
            # Update the visualization for the Zonotopic Predictive Control
            zpc.zpc_visualizer.update_visualization(current_state=current_state,
                                                reach_zonotopes=reach_zonotopes,
                                                optimal_states=optimal_x,
                                                optimal_control=optimal_u[0, :],
                                                stl_interval=interval_intersection_stl,
                                                navigation_array=next_states,
                                                failure_mode=failure_mode,
                                                steps=steps,
                                                plot_trace=True,)
            
            zpc.zpc_time_plotter.forward(current_state=current_state,
                                        stl_interval=interval_intersection_stl,
                                        failure_mode=failure_mode,)

    
    next_state_kin = simulation_model.predict(current_state_kin, optimal_u[0, :])
    next_state = np.array([next_state_kin[0], next_state_kin[1], np.cos(next_state_kin[2]), np.sin(next_state_kin[2]), next_state_kin[3], 0.])
        
    return next_state, next_state_kin, optimal_u[0, :]


def print_relevant_info(args):
    """
    Print all relevant information for the reachability analysis. 

    Args:
        args (object): Arguments object for the reachability analysis.
    """
    relevant_info = f"""
    =====================================
    ZONOTOPIC PREDICTIVE CONTROLLER
    =====================================
    Data Driven Model name: {args.model_name}
    Number of steps: {args.steps}
    Reducing order: {args.reducing_order}
    Sliding window: {args.sliding_window}
    Plot: {args.plot}
    Index: {args.index}
    Point X: {args.point_x}
    Point Y: {args.point_y}
    Frequency: {args.frequency}
    Current state: {args.current_state if args.current_state is not None else 'None'}
    =====================================
    """
    print(relevant_info)
    time.sleep(2)
    

# Example Usage
if __name__ == "__main__":
    # Path to the models folder
    data_driven_model_path = f"{package_path()}/models"

    # Path to the plots folder
    folder_path = f"{package_path()}/plots"

    # Set the batch name and frequency as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help="Name of the trained data-driven model", default="roundabout", required=True)
    parser.add_argument('--steps', '-s', type=int, help="Number of time steps for reachability analysis", default=20, required=False)
    parser.add_argument('--reducing_order', '-r', type=int, help="Reducing order for the reachability analysis", default=20, required=False)
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the states and inputs", required=False)
    parser.add_argument('--index', '-i', type=int, help="Index of the test", default=0, required=False)
    parser.add_argument('--point_x', '-x', type=float, help="X coordinate of the point", default=0.0, required=False)
    parser.add_argument('--point_y', '-y', type=float, help="Y coordinate of the point", default=0.0, required=False)
    parser.add_argument('--frequency', '-f', type=float, help="Frequency of the data", default=10.0, required=False)
    parser.add_argument('--current_state', '-cs', nargs='+', type=float, help="Current state of the vehicle", required=False)
    parser.add_argument('--debug', '-d', action='store_true', help="Debug mode", required=False)
    parser.add_argument('--sliding_window', '-sw', type=int, help="Sliding window for the reachability analysis", default=0, required=False)
    parser.add_argument('--lipschitz', '-l', action='store_true', help="Recompute the Lipschitz constant", required=False)
    parser.add_argument('--stl_formula', '-stl', type=str, help="STL formula for the reachability analysis. Example: G [1,inf] (v<=5)", default=None, required=False)
    parser.add_argument('--failure_iteration', '-fi', type=int, help="Iteration for the failure mode", default=1000, required=False)
    parser.add_argument('--simulation_iteration', '-si', type=int, help="Iteration for the simulation mode", default=1000, required=False)
    parser.add_argument('--yaw', action='store_true', help="Yaw angle for the reachability analysis", required=False)
    args = parser.parse_args()

    # Print all relevant information for the reachability analysis
    print_relevant_info(args)

    # Load the trained data driven model for the reachability analysis
    model_file_path = f"{data_driven_model_path}/reachability_object_{args.model_name}.pkl"
    with open(model_file_path, "rb") as file:
        data_driven_model = dill.load(file)

    # Setup paths for saving plots
    plots_folder = f"{folder_path}/../plots/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(plots_folder, exist_ok=True)


    x0, x1, u, _, _ = data_driven_model.x_meas_vec_0_test, data_driven_model.x_meas_vec_1_test, data_driven_model.u_test, None, None

    # Initialize the reachability analysis object
    nl_reach = DataDrivenReachabilityAnalysis(data_driven_model, steps=args.steps, reducing_order=args.reducing_order, 
                                              debug=False, freq=args.frequency, plot=False, sliding_window=args.sliding_window,
                                              lipschitz=args.lipschitz, zpc=False)
    
    # Initialize the Zonotopic Predictive Control object
    zpc = ZonotopicPredictiveControl(nl_reach, steps_reach=args.steps, show_plots=args.plot, plots_folder=plots_folder, debug=args.debug, num_states=nl_reach.dim_x, plot_yaw=args.yaw)

    # Initialize the vehicle model
    simulation_model = NLKinematicBicycleModel(vehicle_parameters=ego_properties, dt=0.1, n_states=4)

    # Initialize global variables for the reachability analysis
    global stl_formulas, cz_intersection_stl_reach, debug, init_time, init_idx, prev_waypoints
    init_time = time.time()
    init_idx = 0
    stl_formulas = []
    cz_intersection_stl_reach = None
    next_state, next_state_kin = None, None
    optimal_u = None
    debug = args.debug
    data_list = [x0, x1, u]  # No obstacles in the data list for now and link_id is None

    # Define the new STL formulas and goal state
    new_stl_formulas = ["G [3,100] (vx<=4)", "G [3,100] (sin_yaw>=-0.5)", "G [3,100] (sin_yaw<=0.5)", "G [3,100] (cos_yaw<=-0.5)", "G [3,100] (x>=870)", "G [3,100] (x<=890)", "G [3,100] (y>=523)", "G [3,100] (y<=527)"]
    goal = np.array([872.4, 528., -1., 0., 0.])

    selected_stl_formula = args.stl_formula if args.stl_formula is not None else new_stl_formulas
    prev_waypoints = None

    # Get a random initial point to test the reachability analysis if the index is not provided by the user
    if args.index > 0:
        idx_test = args.index
        run_zpc(data_list, zpc, idx_test, args.steps)

    # Set the point to test the reachability analysis if the coordinates are provided by the user
    point = None
    if args.point_x != 0.0 or args.point_y != 0.0:
        point = [args.point_x, args.point_y]
        idx_test = find_closest_idx_point(point, data_driven_model.x_meas_vec_0_test)
        run_zpc(data_list, zpc, idx_test, args.steps)

    # If the current state is provided by the user, run the reachability analysis for the given state
    if args.current_state is not None:
        current_state = np.array(args.current_state)
        idx_test = find_closest_idx_point(current_state[:2], data_driven_model.x_meas_vec_0_test)
        run_zpc(data_list, zpc, idx_test, args.steps)

    if nl_reach.sliding_window > 0:
        n_iterations = 5*nl_reach.sliding_window
        
        # Get index list for saving plots for the sliding window
        index_plot = np.arange(0, n_iterations, 1)
        idx_test = int(data_driven_model.get_random_test_idx()) if args.index == 0 else int(args.index)
        init_idx = idx_test
        index_plot += idx_test

        # Iterate over the test data and plot the reachable sets for each timestep
        for iteration in range(n_iterations):
            stl_formula = selected_stl_formula if iteration == args.failure_iteration else None
            failure_mode = True if iteration >= args.failure_iteration else False
            simulation_mode = failure_mode or iteration >= args.simulation_iteration

            next_state, next_state_kin, optimal_u = run_zpc(data_list, zpc, idx_test + iteration, args.steps, index_plot=index_plot, 
                                                            frequency=args.frequency, new_stl_formula=stl_formula, next_state=next_state, 
                                                            next_state_kin=next_state_kin,optimal_u=optimal_u, failure_mode=failure_mode,
                                                            goal_state=goal, simulation_mode=simulation_mode)
            
            completion_bar(iteration+1, n_iterations)

        nl_reach.generate_reach_gif()
    
    else:
        # Iterate over the test data and plot the reachable sets for each timestep
        for iteration in range(len(data_driven_model.x_meas_vec_0_test[0])):
            idx_test = int(data_driven_model.get_random_test_idx())
            run_zpc(data_driven_model, zpc, idx_test, args.steps)