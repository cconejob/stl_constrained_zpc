import time
import os
import numpy as np 
import dill
import argparse

from stl_constrained_zpc.scripts.reachability.Zonotope import Zonotope
from stl_constrained_zpc.scripts.reachability.utils.Options import Options
from stl_constrained_zpc.scripts.reachability.MatZonotope import MatZonotope
from stl_constrained_zpc.scripts.reachability.utils.Params import Params
from stl_constrained_zpc.scripts.utils.utils import get_idxs, get_input_state_vectors, plot_states_inputs, global_to_local, get_states_yaw, get_states_yaw_cos_sin, package_path


class DataDrivenModel:
    def __init__(self, path="", batch_name="", frequency=10, initpoints=15, steps=30, local=False, 
                 plot=False, normalize=False, straight=False, all_steps=False, get_w=True):
        """
        Data-driven model for reachability analysis. The data-driven model is trained with the inputs and states of the vehicle.
        The inputs are the steering angle and throttle, while the states are the x, y, yaw, speed, lateral speed and yaw rate.
        
        * Input format: [steering angle, throttle]
        * State format: [x, y, cos(yaw), sin(yaw), speed, yaw rate]

        Args:
            path (str): Path to the data
            batch_name (str): Name of the batch
            frequency (int): Frequency of the data
            initpoints (int): Number of initial points
            steps (int): Number of time steps
            local (bool): Compute local coordinates
            plot (bool): Plot the states and inputs
            normalize (bool): Normalize the states
            straight (bool): Straight model
            all_steps (bool): All steps
            get_w (bool): Get the noise zonotope

        Attributes:
            local (bool): Compute local coordinates
            plot (bool): Plot the states and inputs
            normalize (bool): Normalize the states
            straight (bool): Straight model
            u (np.array): Inputs
            x_meas_vec_0 (np.array): Initial states. Format: [x, y, cos(yaw), sin(yaw), speed, yaw rate]
            x_meas_vec_1 (np.array): Final states
            u_test (np.array): Inputs for testing
            x_meas_vec_0_test (np.array): Initial states for testing
            x_meas_vec_1_test (np.array): Final states for testing
            dim_x (int): Dimension of the state vector
            dim_u (int): Dimension of the input vector
            dt (float): Time step
            totalsamples (int): Total number of samples
            params (Params): Parameters for the reachability analysis
            options (Options): Options for the reachability analysis
        """
        self.local = local
        self.normalize = normalize
        self.straight = straight
        path = os.path.join(path, "data")
        self.steps = steps

        # Load all data from the real vehicle
        u, x_meas_vec_0, x_meas_vec_1 = self.load_data(path, batch_name, get_w=get_w)

        # If all steps are required, train with all the indexes from the data
        if all_steps:
            self.initpoints = 1
            idxs_train = np.arange(0, x_meas_vec_0.shape[1], dtype=int)
            idxs_test = np.arange(0, x_meas_vec_0.shape[1], dtype=int)

        # Get indexes to train, starting from the middle of the data and get steps samples
        elif initpoints <= 1:
            self.initpoints = 1
            idxs_train = np.arange(0, steps, dtype=int) + int((x_meas_vec_0.shape[1] - steps) / 2)
            idxs_test = np.setdiff1d(np.arange(x_meas_vec_0.shape[1]), idxs_train)

        # Get the relevant indexes for the training and testing data from maximum and minimum values of the inputs and states
        else:
            self.initpoints = initpoints
            idxs_train, idxs_test = get_idxs(u, x_meas_vec_0, initpoints, steps, get_w)

        # Input and state vectors generated from the real vehicle data and the random initial points for the training and testing data
        self.u, self.x_meas_vec_0, self.x_meas_vec_1 = get_input_state_vectors(u, x_meas_vec_0, x_meas_vec_1, idxs_train)
        self.u_test, self.x_meas_vec_0_test, self.x_meas_vec_1_test = get_input_state_vectors(u, x_meas_vec_0, x_meas_vec_1, idxs_test)

        # Plot the states and inputs of the training data if required
        if plot:
            plot_states_inputs(self.x_meas_vec_0, self.x_meas_vec_1, u)

        # Dimensions of the input and state vectors
        self.dim_x = np.shape(self.x_meas_vec_1)[0]
        self.dim_u = np.shape(u)[0]
        self.dt = 1/frequency
        self.totalsamples = np.shape(self.x_meas_vec_1)[1]
        self.params = Params(tFinal = self.dt * self.totalsamples, dt = self.dt)
        self.options = Options()

        # Compute the noise zonotope for the reachability analysis with the data-driven model
        W, GW, Wmatzono = self.generate_noise_zonotope(wfac=1e-4)

        # Compute the Lipschitz constant for the reachability analysis with the data-driven model
        t0 = time.time()
        print(f"\nTraining data-driven model from {self.totalsamples} samples")

        L, eps = self.compute_lipschitz_constant(u=self.u, x_meas_vec_0=self.x_meas_vec_0, x_meas_vec_1=self.x_meas_vec_1)

        print("====================================")
        print("L =", L)
        print("eps =", eps)
        print("Time to generate data driven model took {} s".format(round(time.time() - t0, 2)))
        print("====================================\n")
        
        Zeps = Zonotope(np.array(np.zeros((self.dim_x, 1))), eps * np.diag(np.ones((self.dim_x, 1)).T[0]))

        # Set the parameters for the reachability analysis with the data-driven model
        self.set_parameters(W=W, GW=GW, Zeps=Zeps, Wmatzono=Wmatzono, u=self.u, x_meas_vec_0=self.x_meas_vec_0, x_meas_vec_1=self.x_meas_vec_1)

    def load_data(self, path, batch_name, get_w=True):
        """
        Load the data from the real vehicle. The data consists of the inputs and states of the vehicle. 
        The inputs are the steering angle and throttle, while the states are the x, y, yaw, speed, yaw rate and lateral acceleration.
        The data is post-processed to compute the local coordinates, normalize the states and compute the straight model.
        
        Args:
            path (str): Path to the data
            batch_name (str): Name of the batch

        Returns:
            u (np.array): Inputs. [steering angle, throttle]
            x_meas_vec_0 (np.array): Initial states. [x, y, cos(yaw), sin(yaw), speed, yaw rate]
            x_meas_vec_1 (np.array): Final states. [x, y, cos(yaw), sin(yaw), speed, yaw rate]
        """
        # Load all data from the real vehicle 
        u = np.load(os.path.join(path, f'U_{batch_name}.npy'), allow_pickle=True)
        x_meas_vec_0 = np.load(os.path.join(path, f'X0_{batch_name}.npy'), allow_pickle=True)
        x_meas_vec_1 = np.load(os.path.join(path, f'X1_{batch_name}.npy'), allow_pickle=True)

        if not get_w:
            u, x_meas_vec_0, x_meas_vec_1 = u[:5, :], x_meas_vec_0[:5, :], x_meas_vec_1[:5, :]

        # Normalize the states with respect to the initial state and add a small value to avoid division by zero
        if self.normalize:
            offset = np.array([0.1, 0.1]).reshape(2, 1)
            x_meas_vec_0[:2,:] = x_meas_vec_0[:2,:] - x_meas_vec_0[:2,0].reshape(2, 1) + offset
            x_meas_vec_1[:2,:] = x_meas_vec_1[:2,:] - x_meas_vec_1[:2,0].reshape(2, 1) + offset
            
        # Compute the local model
        if self.local:
            for i in range(x_meas_vec_0.shape[1]):
                # Process global coordinates
                current_state_global = get_states_yaw(x_meas_vec_0[:, i])
                next_state_global = get_states_yaw(x_meas_vec_1[:, i])
                
                # Local coordinates
                current_state_local = [0., 0., 1., 0., current_state_global[3], current_state_global[4]]
                next_state_local = global_to_local(current_state_global, [next_state_global])[0] + [next_state_global[3], next_state_global[4]]
                next_state_local = get_states_yaw_cos_sin(next_state_local)
                
                x_meas_vec_0[:, i] = np.array(current_state_local)
                x_meas_vec_1[:, i] = np.array(next_state_local)

        # Get the relevant indexes for the training and testing data if the model is a straight model
        init_idx_straight = 25
        final_idx_straight = 350
        if self.straight:
            x_meas_vec_0 = x_meas_vec_0[:, init_idx_straight:final_idx_straight]
            x_meas_vec_1 = x_meas_vec_1[:, init_idx_straight:final_idx_straight]
            u = u[:, init_idx_straight:final_idx_straight]

        return u, x_meas_vec_0, x_meas_vec_1

    def generate_noise_zonotope(self, wfac=1e-4):
        """
        Generate the noise zonotope for reachability analysis.

        Args:
            wfac (float): Factor to generate the noise zonotope.

        Returns:
            W (Zonotope): Noise zonotope
            GW (list of np.array): Generators of the noise zonotope
            Wmatzono (MatZonotope): Noise zonotope in matrix form
        """
        # Create the noise zonotope
        W = Zonotope(np.zeros((self.dim_x, 1)), wfac * np.ones((self.dim_x, 1)))  # disturbance

        # Generate noise matrix zonotope
        GW = []
        for i in range(W.generators().shape[1]):
            vec = W.Z[:, i + 1].reshape(self.dim_x, 1)
            for j in range(self.totalsamples):
                G_j = np.hstack((np.zeros((self.dim_x, j)), vec, np.zeros((self.dim_x, self.totalsamples - j - 1))))
                GW.append(G_j)

        # Convert to matrix zonotope
        Wmatzono = MatZonotope(np.zeros((self.dim_x, self.totalsamples)), np.array(GW))

        return W, GW, Wmatzono

    def compute_lipschitz_constant(self, u, x_meas_vec_0, x_meas_vec_1):
        """
        Compute the Lipschitz constant for the reachability analysis.

        Args:
            u (np.array): Inputs
            x_meas_vec_0 (np.array): Initial states
            x_meas_vec_1 (np.array): Final states

        Returns:
            L (np.array): Lipschitz constant for each state dimension
            eps (np.array): Epsilon for each state dimension
        """
        L = np.zeros(self.dim_x)
        eps = np.zeros(self.dim_x)

        # Compute the Lipschitz constant for each dimension of the state vector
        for i_dim in range(self.dim_x):

            # Compute the Lipschitz constant for each initial point and time step
            for ip in range(self.initpoints):
                start_idx = ip * self.steps
                end_idx = min(start_idx + self.steps, x_meas_vec_0.shape[1])

                # Skip the trajectory if there is insufficient data
                if end_idx <= start_idx:  
                    continue
                
                # Compute the x-, u- and x+ vectors for each step in the trajectory
                for i in range(start_idx, end_idx):
                    z1 = np.hstack((x_meas_vec_0[i_dim, i], u[:, i]))
                    f1 = x_meas_vec_1[i_dim, i]

                    # Compute the x-, u- and x+ vectors for each step in the trajectory
                    for j in range(start_idx, end_idx):

                        # Skip the comparison if the same point is being compared
                        if i == j:  
                            continue

                        z2 = np.hstack((x_meas_vec_0[i_dim, j], u[:, j]))
                        f2 = x_meas_vec_1[i_dim, j]

                        # Compute the norm for the x- and u- vectors
                        norm_z = np.linalg.norm(z1 - z2)

                        # Avoid division by zero if the norm is close to zero
                        if np.isclose(norm_z, 0.0):  
                            continue
                        
                        # Compute the norm for the x+ vectors
                        new_norm = np.linalg.norm(f1 - f2) / norm_z

                        # Update Lipschitz constant and epsilon if a higher value is found
                        if new_norm > L[i_dim]:
                            L[i_dim] = new_norm
                            eps[i_dim] = L[i_dim] * norm_z/2

        return L, eps
    
    def get_random_test_idx(self, steps_reach=20):
        """
        Get a random test index for the reachability analysis.

        Args:
            steps_reach (int): Number of time steps

        Returns:
            int: Random test index
        """
        if self.x_meas_vec_0_test.shape[1] - steps_reach < 1:
            return 0
        
        return np.random.choice(self.x_meas_vec_0_test.shape[1]-steps_reach, 1, replace=False)
    
    def set_parameters(self, W, GW, Zeps, Wmatzono, u, x_meas_vec_0, x_meas_vec_1):
        """
        Set the parameters for the reachability analysis with the data-driven model. 

        Args:
            W (Zonotope): Noise zonotope for the reachability analysis
            GW (np.array): Generators of the noise zonotope
            Wmatzono (MatZonotope): Noise zonotope in matrix form
            Zeps (Zonotope): Noise zonotope with epsilon
            u (np.array): Inputs
            x_meas_vec_0 (np.array): Initial states
            x_meas_vec_1 (np.array): Final states     
        """
        self.options.params["W"] = W
        self.options.params["GW"] = GW
        self.options.params["Wmatzono"] = Wmatzono
        self.options.params["Zeps"] = Zeps
        self.options.params["ZepsFlag"] = True
        self.options.params["Zeps_w"] = W + Zeps
        self.options.params["zonotopeOrder"] = 100
        self.options.params["tensorOrder"] = 2
        self.options.params["errorOrder"] = 5
        self.options.params["U_full"] = u
        self.options.params["X_0T"] = x_meas_vec_0
        self.options.params["X_1T"] = x_meas_vec_1
        self.options.params["dim_x"] = self.dim_x
        self.options.params["dim_u"] = self.dim_u


if __name__ == "__main__":
    # Path to the models folder
    data_driven_model_path = f"{package_path()}/models"

    # Set the batch name and frequency as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_name', '-b', type=str, help="Name of the batch", default="complete_tour", required=False)
    parser.add_argument('--frequency', '-f', type=int, help="Frequency of the data", default=10, required=False)
    parser.add_argument('--initpoints', '-i', type=int, help="Number of initial points", default=15, required=False)
    parser.add_argument('--steps', '-s', type=int, help="Number of time steps", default=2, required=False)
    parser.add_argument('--loc', '--local', '-l', action='store_true', help="Compute local coordinates", required=False)
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the states and inputs", required=False)
    parser.add_argument('--normalize', '-n', action='store_true', help="Normalize the states", required=False)
    parser.add_argument('--straight', '-st', action='store_true', help="Straight model", required=False)
    parser.add_argument('--all', '-a', action='store_true', help="Train the data-driven model with all the points of the batch", required=False)
    parser.add_argument('--get_w', '-w', action='store_true', help="Get the noise zonotope", required=False)
    args = parser.parse_args()

    # Check if the reachability object is already saved in a file
    extra = f"_{args.steps}"
    if args.loc:
        extra += "_loc"

    if args.normalize:
        extra += "_norm"

    if args.straight:
        extra += "_straight"

    if args.all:
        extra += "_all"
        class_file_path = f"{data_driven_model_path}/reachability_object_{args.batch_name}{extra}.pkl"

    else:
        class_file_path = f"{data_driven_model_path}/reachability_object_{args.batch_name}_{args.initpoints}{extra}.pkl"

    # Get model name (between reachability_object_ and .pkl)
    model_name = class_file_path.split("reachability_object_")[1].split(".pkl")[0]

    # Print all relevant information regarding the data-driven model and the parameters chosen by the user
    print("\n====================================")
    print("Batch name:", args.batch_name)
    print("Frequency:", args.frequency)
    print("Number of initial points:", args.initpoints)
    print("Number of time steps:", args.steps)
    print("Compute local coordinates:", args.loc)
    print("Normalize the states:", args.normalize)
    print("Straight model:", args.straight)
    print("All steps:", args.all)
    print("====================================\n")

    answer = 'y'
    if os.path.exists(class_file_path):
        print("The data-driven model for reachability under this parameters is already saved in a file")
        print("Do you want to train a new data-driven model? (y/n)")
        answer = input().lower()
    
    # Train a new data-driven model if the user wants
    if answer == 'y':
        print("\nTraining a new data-driven model for reachability analysis")
        reach = DataDrivenModel(data_driven_model_path, batch_name=args.batch_name, frequency=args.frequency, initpoints=args.initpoints, 
                                steps=args.steps, local=args.loc, plot=args.plot, normalize=args.normalize, 
                                straight=args.straight, all_steps=args.all, get_w=args.get_w)

        # Save the data-driven model to a file
        with open(class_file_path, "wb") as file:
            dill.dump(reach, file)
            print("\n====================================")
            print(f"Data-driven model for reachability analysis saved in {class_file_path}")
            # Get the name of the model object
            print(f"Model name: {model_name}")
            print("====================================\n")

    else:
        print("Training a new data-driven model is not required")