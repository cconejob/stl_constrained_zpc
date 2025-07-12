import time
import numpy as np 
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import dill
import argparse
import rospkg
from collections import deque
import os
import datetime

from stl_constrained_zpc.scripts.reachability.Interval import Interval
from stl_constrained_zpc.scripts.reachability.Zonotope import Zonotope
from stl_constrained_zpc.scripts.reachability.MatZonotope import MatZonotope
from stl_constrained_zpc.scripts.utils.utils import find_closest_idx_point, get_random_idxs_for_sliding_window, completion_bar
from stl_constrained_zpc.scripts.reachability.vehicle_model.data_driven.train_data_driven_model import DataDrivenModel
from stl_constrained_zpc.scripts.visualization.zpc_visualizer import ZPCVisualizer


class DataDrivenReachabilityAnalysis():
    def __init__(self, data_driven_model: DataDrivenModel, steps=20, normalize=False, reducing_order=2, debug=False,
                  freq=10.0, local=False, plot=False, sliding_window=0, lipschitz=False, zpc=False,
                  plots_folder=None, show_plots=False, plot_yaw=False, add_vehicle_dimension=True):
        """
        Data Driven Non linear reachability analysis class. 

        Args:
            data_driven_model (object): Data driven model object.
            steps (int): Number of steps for the reachability analysis.
            debug (bool): Print the properties of the reachability analysis.
            freq (float): Frequency of the data.
            local (bool): Use local in the reachability analysis.
            plot (bool): Plot reachable sets.
            sliding_window (int): Sliding window for the reachability analysis.
            lipschitz (bool): Recompute the Lipschitz constant.
            zpc (bool): Use the zonotopic predictive control.
            plots_folder (str): Folder to save the plots.
            show_plots (bool): Show the plots.

        Attributes:
            options (object): Options object for the reachability analysis.
            params (object): Parameters object for the reachability analysis.
            dim_x (int): Dimension of the state space.
            dim_u (int): Dimension of the input space.
            steps (int): Number of steps for the reachability analysis.
            reducing_order (int): Reducing order for the reachability analysis.
            debug (bool): Print the properties of the reachability analysis.
            local (bool): Use local in the reachability analysis.
            plot (bool): Plot reachable sets.
        """
        sliding_window = int(sliding_window/2)
        self.params = data_driven_model.params
        self.options = self.params2options(self.params, data_driven_model.options)
        self.dim_x = data_driven_model.dim_x
        self.dim_u = data_driven_model.dim_u
        self.steps = steps
        self.normalize = normalize
        self.reducing_order = reducing_order
        self.debug = debug
        self.local = local
        self.freq = freq
        self.plot = plot
        self.G = 1e-10* np.ones((self.dim_x, self.dim_x)); 
        generator_v = 1e-2*np.ones((self.dim_x, 1)); generator_v[2] = 1e-4; generator_v[3] = 1e-4
        generator_w = 1e-4*np.ones((self.dim_x, 1))
        self.sliding_window = sliding_window
        self.start_reach = True
        self.Zw = self.options.params["Wmatzono"]
        self.Zeps = self.options.params["Zeps"]
        self.lipschitz = lipschitz
        self.Zeps_flag = True
        self.zpc = zpc
        save_plots = False if plots_folder is None else True
        self.add_vehicle_dimension = add_vehicle_dimension

        if add_vehicle_dimension:
            length = 4.5; width = 1.75
            center = np.zeros(self.dim_x).reshape((self.dim_x, 1))
            generator = np.array([[length/2, 0], [0, width/2], [0, 0], [0, 0], [0, 0]])
            self.z_vehicle_dimension = Zonotope(center, generator)
        
        # Compute the noise zonotope
        self.Zw = Zonotope(self.Zw.center, self.Zw.generators[0])

        if sliding_window > 0:
            idxs = get_random_idxs_for_sliding_window(sliding_window, len(self.options.params["X_0T"].T))
            self.y_0 = deque(self.options.params["X_0T"][:, idxs].T, maxlen=sliding_window)
            self.u = deque(self.options.params["U_full"][:, idxs].T, maxlen=sliding_window)
            self.y_1 = deque(self.options.params["X_1T"][:, idxs].T, maxlen=sliding_window)
            idxs_fixed = get_random_idxs_for_sliding_window(sliding_window, len(self.options.params["X_0T"].T))
            self.y_0_fixed = self.options.params["X_0T"][:, idxs_fixed].T
            self.u_fixed = self.options.params["U_full"][:, idxs_fixed].T
            self.y_1_fixed = self.options.params["X_1T"][:, idxs_fixed].T
            self.first_iteration = True
            self.start_reach = False
            self.Zw, _, self.MZw = self.generate_noise_zonotope(generator=generator_w)
            self.Zv, _, self.MZv = self.generate_noise_zonotope(generator=generator_v)

        if lipschitz:
            self.iteration = int(1)
            self.data_driven_model = data_driven_model

        # Initialize the ZPC Visualizer for plotting the results if required
        if save_plots or show_plots:
            print(f"Initializing ZPC Visualizer")
            self.zpc_visualizer = ZPCVisualizer(show_plots=show_plots, save_plots=save_plots, plots_folder=plots_folder, debug=debug, plot_map=True, plot_yaw=plot_yaw)
        else:
            self.zpc_visualizer = None

    def forward(self, r = [0, 0, 1, 0, 0], u = [0, 0], update_sliding_window=False):
        """
        This function runs the reachability analysis for a non linear system.

        Args:
            r (list): Initial state of the system. [x, y, cos_yaw, sin_yaw, v, w].
            u (list): Input to the system. [steering, acceleration].

        Returns:
            R_data (list): List of reachable sets for each timestep.
            derivatives (list): List of derivatives for each timestep.
        """
        t0 = time.time()

        # Add data into current buffer for the sliding window
        if self.sliding_window > 0 and update_sliding_window:
            self.update_sliding_window(r, u[:,0])
            if self.debug:
                print(f"Time to update sliding window: {time.time() - t0:.3f} seconds")

        t0 = time.time()

        # Compute the Lipschitz constant for the system
        if self.lipschitz:
            if self.iteration%int(self.sliding_window/10) == 0:
                u_0 = np.concatenate([np.array(self.u), self.u_fixed])
                y_0 = np.concatenate([np.array(self.y_0), self.y_0_fixed])
                y_1 = np.concatenate([np.array(self.y_1), self.y_1_fixed])
                _, eps = self.data_driven_model.compute_lipschitz_constant(u=np.array(u_0).T, x_meas_vec_0=np.array(y_0).T, x_meas_vec_1=np.array(y_1).T)
                self.Zeps = Zonotope(np.array(np.zeros((self.dim_x, 1))), eps * np.diag(np.ones((self.dim_x, 1)).T[0]))
                if self.debug:
                    print(f"Time to compute Lipschitz constant: {time.time() - t0:.3f} seconds")
                self.iteration += 1

        t0 = time.time()

        # Initial reachable set for the system
        if self.start_reach:
            R0 = Zonotope(np.array(r).reshape((self.dim_x, 1)), self.G)
            R_data, derivatives = self.reach_DT(u, R0)
            if self.debug:
                print(f"Time to compute reachable dt: {time.time() - t0:.3f} seconds")
            return R_data, derivatives
        
        return None, None

    def reach_DT(self, u, R0):
        """
        Reachability analysis for a non linear system.

        Args:
            u (np.array): Input to the system. Dimensions: [steering angle, acceleration].
            R0 (object): Initial reachable set for the system. Dimensions: [X, Y, yaw, v].
            varargin (list): Additional arguments.
        """
        # Initial reachable set for the system
        R0 = R0.reduce('girard', self.reducing_order)

        # Normalize the reachable set for the system if needed and add a small offset
        if self.normalize:
            R0.Z[:2, 0] = np.array([0.1, 0.1])

        # Project the input to the control input space
        R_data = [R0]
        derivatives = []
        u = np.tile(u, (1, self.steps)) if u.shape[1] == 1 else u

        # Run the reachability analysis for the system for each timestep
        for i in range(self.steps):
            U = Zonotope(np.array(u[:, i]).reshape((self.dim_u, 1)), np.diag([1e-4, 0.01]))
            new_set, dc_dr, dc_du = self.linReach_DT(R_data[-1], U)
            if self.add_vehicle_dimension and i==0:
                new_set += self.z_vehicle_dimension
                new_set = new_set.rotate(np.arctan2(new_set.center()[3], new_set.center()[2])[0])
            else:
                delta_yaw = np.arctan2(new_set.center()[3], new_set.center()[2])[0] - prev_yaw
                new_set = new_set.rotate(delta_yaw)
            R_data.append(new_set)
            derivatives.append([dc_dr, dc_du])

            if self.debug:
                print("Control input steering angle [deg]", u[0,i]*180/np.pi)
                print("Control input acceleration [m/s2]", u[1,i])
                print("Reachable set of step", i, "center", R_data[i+1].center().T)
            
            prev_yaw = np.arctan2(R_data[i].center()[3], R_data[i].center()[2])[0]

        return R_data, derivatives

    def linReach_DT(self, Ri, U):
        """
        Computes the next reachable set for a nonlinear system based on a data-driven model.

        Args:
            Ri (Zonotope): Reachable set for the current timestep.
            U (Zonotope): Input set for the system at the current timestep.

        Returns:
            result (Zonotope): Reachable set for the next timestep.
            M_x (np.array): Derivative of the reachable set w.r.t. state.
            M_u (np.array): Derivative of the reachable set w.r.t. input.
        """
        t0 = time.time()
        
        # Get centers of the reachable set and input set
        yStar = Ri.center()
        uStar = U.center()

        # Extract stored system data
        u = np.concatenate([np.array(self.u), self.u_fixed]).T.reshape((self.dim_u, 2*self.sliding_window)) if self.sliding_window > 0 else self.options.params["U_full"]
        y_0 = np.concatenate([np.array(self.y_0), self.y_0_fixed]).T.reshape((self.dim_x, 2*self.sliding_window)) if self.sliding_window > 0 else self.options.params["X_0T"]
        y_1 = np.concatenate([np.array(self.y_1), self.y_1_fixed]).T.reshape((self.dim_x, 2*self.sliding_window)) if self.sliding_window > 0 else self.options.params["X_1T"]

        # Prepare matrices for computation
        yStarMat = np.tile(yStar, (1, y_0.shape[1]))
        uStarMat = np.tile(uStar, (1, u.shape[1]))
        oneMat = np.ones((1, u.shape[1]))

        # Substract the centers of the noise zonotopes to y_1
        y_1_prime = y_1 - self.Zw.center() - self.Zv.center()

        # Compute system derivatives (IAB)
        point_matrix = np.vstack([oneMat, y_0 - yStarMat, u - uStarMat])
        M = np.dot(y_1_prime, pinv(point_matrix))
        
        # Compute residual disturbance
        L = -1 * (self.MZw + np.dot(M, point_matrix)) + y_1

        # Convert residual disturbance to zonotope form
        VInt = L.interval_matrix()
        leftLimit = VInt.Inf
        rightLimit = VInt.Sup
        Zl = -1 * (self.Zw + self.Zv) + Zonotope(Interval(leftLimit.min(axis=1).T, rightLimit.max(axis=1).T)) 

        # Compute the next reachable set
        y = (Ri) + (-1 * yStar) + (-1 * self.Zv)
        u = U + (-1 * uStar)
        result = (y.cart_prod(u).cart_prod(np.array([1])) * M) + self.Zw + Zl + self.Zv + self.Zeps

        M_x = M[:self.dim_x, 1:1 + self.dim_x]
        M_u = M[:self.dim_x, 1 + self.dim_x:1 + self.dim_x + self.dim_u]

        if self.debug:
            print(f"Time to compute linReach_DT: {time.time() - t0:.4f} seconds")

        return result, M_x, M_u
    
    def params2options(self, params, options):
        """
        Convert the parameters to options for the reachability analysis.

        Args:
            params (object): Parameters object for the reachability analysis.
            options (object): Options object for the reachability analysis.

        Returns:
            options (object): Options object for the reachability analysis.
        """
        for key, value in params.params.items():
            options.params[key] = value
        return options 
    
    def update_sliding_window(self, r, u):
        """
        Update the sliding window for the reachability analysis. 

        Args:
            r (list): Initial state of the system. [x, y, cos_yaw, sin_yaw, v, w].
            u (list): Input to the system. [steering, acceleration].
        """
        r, u = np.array(r), np.array(u)
        
        if self.first_iteration:
            self.prev_x, self.prev_u = r, u
            self.first_iteration = False
        else:
            self.y_0.append(self.prev_x)
            self.y_1.append(r)
            self.u.append(self.prev_u)
            self.prev_x, self.prev_u = r, u
            self.start_reach = len(self.y_0) == self.sliding_window

    def generate_noise_zonotope(self, generator):
        """
        Generate the noise zonotope for the reachability analysis. The noise zonotope is used to compute the Lipschitz constant.

        Args:
            wfac (float): Factor to generate the noise z

        Returns:
            W (Zonotope): Noise zonotope
            GW (np.array): Generators of the noise zonotope
            Wmatzono (MatZonotope): Noise zonotope in matrix form
        """
        # Generate the noise zonotope
        W = Zonotope(np.array(np.zeros((self.dim_x, 1))), generator.reshape(-1, 1) * np.ones((self.dim_x, 1)))
        GW = []
        for i in range(W.generators().shape[1]):
            vec = np.reshape(W.Z[:, i + 1], (self.dim_x, 1))
            dummy = []
            dummy.append(np.hstack((vec, np.zeros((self.dim_x, 2*self.sliding_window - 1)))))

            for j in range(1, 2*self.sliding_window, 1):
                right = np.reshape(dummy[i][:, 0:j], (self.dim_x, -1))
                left = dummy[i][:, j:]
                dummy.append(np.hstack((left, right)))
            GW.append(np.array(dummy))

        # Convert the noise zonotope to a matrix zonotope
        GW = GW[0]
        Wmatzono = MatZonotope(np.zeros((self.dim_x, 2*self.sliding_window)), GW)

        return W, GW, Wmatzono
    
    def create_reach_plot_folder(self):
        """
        Create a folder for the plots of the reachable sets.
        """
        safety_path = rospkg.RosPack().get_path('safety')
        self.plots_folder = f"{safety_path}/scripts/data_driven_model/plots/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)

        print(f"Saving plots in {self.plots_folder}")
        
        # Create a README file with all parameters
        with open(f"{self.plots_folder}/README.txt", "w") as file:
            file.write(f"Steps: {self.steps}\n")
            file.write(f"Reducing order: {self.reducing_order}\n")
            file.write(f"Sliding window: {self.sliding_window}\n")
            file.write(f"Plot: {self.plot}\n")
            file.write(f"Frequency: {self.freq}\n")
            file.write(f"Debug: {self.debug}\n")
            file.write(f"Lipschitz: {self.lipschitz}\n")
            file.close()

    def save_reach_plot(self, fig=None, plot_name=None):
        """
        Save the plot of the reachable sets.

        Args:
            fig (object): Figure object for the plot.
            plot_name (str): Name of the plot.
        """
        if fig is None:
            plt.savefig(f"{self.plots_folder}/{plot_name}.png")

        else:
            fig.savefig(f"{self.plots_folder}/{plot_name}.png")

    def generate_reach_gif(self):
        # Generate gif from the saved plots
        os.system(f"convert -delay 20 -loop 0 {self.plots_folder}/*.png {self.plots_folder}/reachability.gif")
        print(f"\nSaved gif in {self.plots_folder}/reachability.gif")

def compute_error_reachable_set(R_data, next_states):
    """
    Compute the error between the first reachable set center and the first next state.
    The position error is computed as the sum of the distances between the reachable set centers and the next states.
    The yaw error is computed as the difference between the yaw angles of the reachable set and the next states.
    The velocity error is computed as the difference between the velocities of the reachable set and the next states.

    Args:
        R_data (list): List of reachable sets for each timestep.
        next_states (np.array): Next states for the system.

    Returns:
        error (float): Error between the reachable set and the next states.
    """
    reachable_set_center = R_data[1].center().squeeze()
    print("Computing error between reachable set and next states")
    print("Next state", next_states[1])
    print("Reachable set", reachable_set_center)

    # Compute the position error
    pos_error = np.linalg.norm(reachable_set_center[:2] - next_states[1][:2])

    # Compute the yaw error
    yaw_error = np.arctan2(reachable_set_center[3], reachable_set_center[2]) - np.arctan2(next_states[1][3], next_states[1][2])

    # Compute the velocity error
    vel_error = np.linalg.norm(reachable_set_center[4] - next_states[1][4])

    return pos_error, yaw_error, vel_error

    
def run_reachability_analysis(loaded_reach, nl_reach: DataDrivenReachabilityAnalysis, idx, steps, index_plot=None):
    """
    Run the reachability analysis for a random state from the test data from a loaded model.

    Args:
        loaded_reach (object): Loaded model object.
        nl_reach (object): Reachability analysis object.
        idx (int): Index of the test data.
        steps (int): Number of steps for the reachability analysis.
        index_plot (list): List of indexes to plot
    """
    t0 = time.time()
    current_state = loaded_reach.x_meas_vec_0_test[:, idx]
    next_states = loaded_reach.x_meas_vec_1_test[:, idx-1:idx+steps]
    current_input = loaded_reach.u_test[:, idx:idx+steps]

    R_data, _ = nl_reach.forward(current_state, current_input, update_sliding_window=True)
    print(f"Time to compute reachable set: {time.time() - t0:.3f} seconds")

    if nl_reach.start_reach:
        if nl_reach.zpc_visualizer is not None:
            if index_plot is not None:
                if idx in index_plot:
                    print("current state", current_state)
                    nl_reach.zpc_visualizer.update_visualization(current_state=current_state,
                                                                reach_zonotopes=R_data,
                                                                optimal_states=next_states.T,
                                                                optimal_control=current_input[:, 0],
                                                                plot_center_reach=True,
                                                                is_trajectory=True,
                                                                steps=steps,)


def print_relevant_info(args):
    """
    Print all relevant information for the reachability analysis. 

    Args:
        args (object): Arguments object for the reachability analysis.
    """
    print("\n=====================================")
    print(f"DATA-DRIVEN REACHABILITY ANALYSIS")
    print("=====================================")

    time.sleep(1)

    print("\n=====================================")
    print(f"Data Driven Model name: {args.model_name}")
    print(f"Number of steps: {args.steps}")
    print(f"Reducing order: {args.reducing_order}")
    print(f"Sliding window: {args.sliding_window}")
    print(f"Plot: {args.plot}")
    if args.index > 0:
        print(f"Index: {args.index}")
    if args.point_x != 0.0 or args.point_y != 0.0:
        print(f"Point X: {args.point_x}")
        print(f"Point Y: {args.point_y}")
    print(f"Frequency: {args.frequency}")
    if args.current_state is not None:
        print(f"Current state: {args.current_state}")
    print("=====================================\n")

    time.sleep(2)


if __name__ == "__main__":
    # Path to the data driven model
    safety_path = rospkg.RosPack().get_path('safety')
    data_driven_model_path = f"{safety_path}/scripts/data_driven_model/models"

    # Set the batch name and frequency as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help="Name of the trained data-driven model", required=True)
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
    parser.add_argument('--plot_yaw', '-py', action='store_true', help="Plot yaw angle", required=False)
    args = parser.parse_args()

    # Print all relevant information for the reachability analysis
    print_relevant_info(args)

    # Load the trained data driven model for the reachability analysis
    model_file_path = f"{data_driven_model_path}/reachability_object_{args.model_name}.pkl"
    with open(model_file_path, "rb") as file:
        data_driven_model = dill.load(file)

    # Check if the model is normalized with respect to the initial state
    normalize = False
    if "norm" in args.model_name:
        normalize = True

    # Check if the model is computed with local coordinates
    local = False
    if "loc" in args.model_name:
        local = True

    safety_path = rospkg.RosPack().get_path('safety')
    plots_folder = f"{safety_path}/scripts/data_driven_model/plots/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(plots_folder, exist_ok=True)

    # Initialize the reachability analysis object
    nl_reach = DataDrivenReachabilityAnalysis(data_driven_model, steps=args.steps, normalize=normalize, reducing_order=args.reducing_order, 
                                              debug=args.debug, freq=args.frequency, local=local, plot=args.plot, sliding_window=args.sliding_window,
                                              lipschitz=args.lipschitz, show_plots=args.plot, plots_folder=plots_folder, plot_yaw=args.plot_yaw, add_vehicle_dimension=True)   
    
    # Get a random initial point to test the reachability analysis if the index is not provided by the user
    if args.index > 0:
        idx_test = args.index
        run_reachability_analysis(data_driven_model, nl_reach, idx_test, args.steps)

    # Set the point to test the reachability analysis if the coordinates are provided by the user
    point = None
    if args.point_x != 0.0 or args.point_y != 0.0:
        point = [args.point_x, args.point_y]
        idx_test = find_closest_idx_point(point, data_driven_model.x_meas_vec_0_test)
        run_reachability_analysis(data_driven_model, nl_reach, idx_test, args.steps)

    # If the current state is provided by the user, run the reachability analysis for the given state
    if args.current_state is not None:
        current_state = np.array(args.current_state)
        idx_test = find_closest_idx_point(current_state[:2], data_driven_model.x_meas_vec_0_test)
        run_reachability_analysis(data_driven_model, nl_reach, idx_test, args.steps)

    if nl_reach.sliding_window > 0:
        n_iterations = 10*nl_reach.sliding_window
        # Get index list for saving plots for the sliding window
        index_plot = np.arange(0, n_iterations, int(nl_reach.sliding_window/20))
        idx_test = int(data_driven_model.get_random_test_idx()) if args.index == 0 else args.index
        index_plot += idx_test
        print(f"\nRunning reachability analysis for {n_iterations} iterations\n")

        # Iterate over the test data and plot the reachable sets for each timestep
        for iteration in range(n_iterations):
            run_reachability_analysis(data_driven_model, nl_reach, idx_test + iteration, args.steps, index_plot=index_plot)
            completion_bar(iteration+1, n_iterations)

        nl_reach.generate_reach_gif()
    
    else:
        # Iterate over the test data and plot the reachable sets for each timestep
        for iteration in range(len(data_driven_model.x_meas_vec_0_test[0])):
            idx_test = int(data_driven_model.get_random_test_idx())
            run_reachability_analysis(data_driven_model, nl_reach, idx_test, args.steps)