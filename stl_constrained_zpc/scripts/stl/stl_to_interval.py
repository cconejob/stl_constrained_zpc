import re
import numpy as np
import copy
import time
import argparse
import rospkg
import dill
import os
import datetime

from stl_constrained_zpc.scripts.reachability.Interval import Interval
from stl_constrained_zpc.scripts.utils.interval_operations import IntervalOperations
from stl_constrained_zpc.scripts.utils.utils import completion_bar
from stl_constrained_zpc.scripts.stl.stl_parser import STLParser

from stl_constrained_zpc.scripts.visualization.zpc_visualizer import ZPCVisualizer
from stl_constrained_zpc.scripts.visualization.zpc_time_plotter import ZPCTimePlotter


operations = IntervalOperations()


class STLAtomicConditionToInterval:
    def __init__(self, atomic_condition, initial_state, sample_time=0.1, state_list=["x", "y", "cos_yaw", "sin_yaw", "v"], initial_tick=None):
        """
        Convert STL atomic conditions into 5D interval for safety verification. The possible states are:
            [x, y, cos_yaw, sin_yaw, vx, w]

        It generates a slope-bounded interval based on the atomic conditions.

        Args:
            atomic_condition (list): List of atomic conditions. Each atomic condition is of the form:
                Operator [Time1, Time2] (Variable Operator Value) Example: "G [0,5] (vx<=0.1)"
            
            initial_state (float): Initial state of the system in the studied dimension.
        """
        self.initial_tick = time.time() if initial_tick is None else initial_tick
        self.atomic_condition = atomic_condition
        self.freq = sample_time
        self.state_list = state_list
        self.initial_state = initial_state

        # Default constraints for the states
        self.default_constraints = self._default_constraints()

        # Convert the atomic condition into a interval
        self.interval = self._convert_to_interval()

    def _default_constraints(self):
        constraints = {
            "x": (-1e5, 1e5),
            "y": (-1e5, 1e5),
            "cos_yaw": (-1, 1),
            "sin_yaw": (-1, 1),
            "yaw": (float(-np.pi), float(np.pi)),
            "vx": (0, 40.),
            "v": (0, 40.),
            "w": (-10., 10.)
        }
        return constraints

    def _set_constraint(self, var, op, value, time_interval):
        """
        Sets constraints for the zonotope based on the STL condition.
        
        Default constraints are set as follows:
            x: -1e5 < x < 1e5
            y: -1e5 < y < 1e5
            cos_yaw: -1 < cos_yaw < 1
            sin_yaw: -1 < sin_yaw < 1
            yaw: -pi < yaw < pi
            vx: 0 < vx < 40
            w: -10 < w < 10
        Args:
            var (str): Variable name.
            op (str): Operator.
            value (float): Value.
            time_interval (list): Temporal interval.
        """
        if '-inf' in value:
            value = -np.inf
        elif 'inf' in value:
            value = np.inf
        else:
            value = float(value)
        
        if '<' in op:
            lower = self.default_constraints[var][0]
            upper = value
        elif '>' in op:
            lower = value
            upper = self.default_constraints[var][1]
        else:
            lower = value
            upper = value

        self.var_name = var
        self.var_interval = [lower, upper]

        # Add temporal constraints
        if '-inf' in time_interval:
            time_interval[0] = -np.inf
        else:
            time_interval[0] = float(time_interval[0])
        
        if 'inf' in time_interval:
            time_interval[1] = np.inf
        else:
            time_interval[1] = float(time_interval[1])

        self.time_interval = list(time_interval)

    def _get_idx_from_var_name(self, var_name):
        """
        Get the index of the state from the state name.

        Args:
            var_name (str): State name.

        Returns:
            int: Index of the state.
        """
        try:
            return self.state_list.index(var_name)
        
        except ValueError:
            print("Invalid state name")
            if "vx" in var_name:
                print("State name is v")
                return self.state_list.index("v")
            elif "v" in var_name:
                print("State name is vx")
                return self.state_list.index("vx")
            elif var_name == "yaw":
                return [self.state_list.index("yaw")]
            elif "yaw" in var_name:
                return [self.state_list.index("cos_yaw"), self.state_list.index("sin_yaw")]
            else:
                print("Invalid state name")
                return None
    
    def _convert_to_interval(self):
        """
        Converts the atomic conditions into 5D intervals. 

        Args:
            initial_state (float): Initial state of the system in the studied dimension.
            margin (float): Margin for the interval.

        Returns:
            list: List of 5D intervals.
        """
        var, op, value, time_interval = self._parse_condition(self.atomic_condition)
        self._set_constraint(var, op, value, time_interval)
        self.idx_var = self._get_idx_from_var_name(self.var_name)
        
        interval = Interval(np.array([self.var_interval[0]]), np.array([self.var_interval[1]]))
        
        return interval
    
    def _parse_condition(self, condition):
        """
        Extracts variable, operator, value and time interval from the atomic condition.

        Args:
            condition (str): Atomic condition. Format is assumed to be as follows:
                Operator [Time1, Time2] (Variable Operator Value)
            
        Returns:
            tuple: Variable, operator, value and time interval.
        """
        pattern = r'([A-Z]+)_?\s*\[(\d+|inf),\s*(\d+|inf)\]\s*\(\s*([\w]+)\s*([<>=]+)\s*(-?[\d.]+)\s*\)'
        # If the condition is a list, convert it to a string with the first element
        if isinstance(condition, list):
            condition = condition[0]
        match = re.search(pattern, condition)

        if match:
            operator, time1, time2, var, op, value = match.groups()
            self.operator = operator
            time_interval = [float(time1) + self.initial_tick, float(time2) + self.initial_tick]
            return var, op, value, time_interval
        else:
            print("Condition", condition)
            print("Invalid atomic condition format")
            print("Format should be as follows: G [Time1, Time2] (Variable Operator Value)")
        return None, None, None, None
    
    def _within_time_interval(self, timestamp):
        """
        Checks if the timestamp is within the time interval.

        Args:
            timestamp (int): timestamp to be checked.

        Returns:
            bool: True if timestamp is within the time interval, False otherwise.
        """
        if timestamp >= self.time_interval[0] and timestamp <= self.time_interval[1]:
            return True
        return False
    
    def _before_time_interval(self, timestamp):
        """
        Checks if the timestamp is before the time interval.

        Args:
            timestamp (int): timestamp.

        Returns:
            bool: True if timestamp is before the time interval, False otherwise.
        """
        if timestamp < self.time_interval[0]:
            return True
        return False
    
    def _after_time_interval(self, timestamp):
        """
        Checks if the timestamp is after the time interval.

        Args:
            timestamp (int): timestamp.

        Returns:
            bool: True if timestamp is after the time interval, False otherwise.
        """
        if timestamp > self.time_interval[1]:
            return True
        return False
    
    def get_interval(self, timestamp=None):
        """
        Returns the generated zonotope at the given timestamp.

        Args:
            timestamp (float): timestamp.
        
        Returns:
            Zonotope: Zonotope at the given timestamp.
        """
        if timestamp is None:
            timestamp = time.time()

        if self._within_time_interval(timestamp):
            return self.interval
    
        else:
            return None
        

def run_stl_formulation(current_state, stl_formulas, steps_reach, frequency, plot=False, idx=None, index_plot=None,
                        current_input=None, timestep=None, state_list=["x", "y", "cos_yaw", "sin_yaw", "v"],
                        failure_mode=False, next_states=None):
    """
    Run the STL formulation in constrained zonotope space.

    Args:
        current_state (list): Current state of the system.
        stl_formulas (list): List of STL formulas.
        steps_reach (int): Number of time steps for reachability analysis.
        frequency (float): Frequency of the data.
        plot (bool): Plot the states and inputs.
        idx (int): Index of the test.
        index_plot (list): List of indices for saving plots.
        next_states (list): Next states of the system.
        current_input (list): Current input of the system.
        timestep (int): Timestep.
        state_list (list): List of states.
        failure_mode (bool): Failure mode.

    Returns:
        list: List of STL formula constrained
    """
    if timestep is None:
        current_time = time.time()
    else:
        current_time = timestep
    
    prediction_reach_time = current_time + steps_reach/frequency
    times = np.arange(current_time, prediction_reach_time, 1/frequency)

    # Initialize STL formula interval matrix with None values. 
    # Number of rows: number of states, number of columns: time steps
    interval_matrix = np.full((len(state_list), len(times)), None)

    for stl_formula in stl_formulas:
        for j in range(len(times)):
            # Get the interval for the current time step
            interval = stl_formula.get_interval(times[j])
            if interval is not None:
                idx_var = stl_formula.idx_var
                if interval_matrix[idx_var,j] is None:
                    interval_matrix[idx_var,j] = interval
                else:
                    interval_matrix[idx_var,j] = operations.intersection_intervals_v2(interval_matrix[idx_var,j], interval)
            
            # Remove the STL formula from the list if it is not active anymore
            elif j == len(times)-1 and stl_formula._after_time_interval(times[j]):
                print(f"STL formula {stl_formula.atomic_condition} is not active anymore")
                stl_formulas.remove(stl_formula)

    if plot:
        if index_plot is not None:
            if idx in index_plot:
                zpc_visualizer.update_visualization(current_state=current_state,
                                                    optimal_control=current_input[:, 0],
                                                    stl_interval=interval_matrix,
                                                    failure_mode=failure_mode,
                                                    steps=steps_reach,
                                                    optimal_states=next_states.T,
                                                    is_trajectory=True,)
                
                zpc_time_plotter.forward(current_state=current_state,
                                         stl_interval=interval_matrix,
                                        failure_mode=failure_mode,)

    return interval_matrix


def add_stl_formula(current_state, state_list=["x", "y", "cos_yaw", "sin_yaw", "v"],
                        new_stl_formula=None, stl_formulas=[], timestep=None):
    """
    Forward the STL formula to the next time step.

    Args:
        current_state (list): Current state of the system.
        stl_formulas (list): List of STL formulas.
        steps (int): Number of time steps for reachability analysis.
        frequency (float): Frequency of the data.
        idx (int): Index of the test.
        state_list (list): List of states.
        new_stl_formula (str): New STL formula to be added.

    Returns:
        list: List of STL formula constrained
    """
    # Get the number of states
    num_states = len(state_list)

    # ==== STL FORMULAS ====
    if new_stl_formula is not None:
        # If the new STL formula is a list, we need to parse each formula
        if isinstance(new_stl_formula, list):
            for stl_formula in new_stl_formula:  
                print(f"Adding STL formula: {stl_formula}")              
                if stl_formula is not None:
                    parser = STLParser(formula=stl_formula)
                    condition = parser.get_atomic_conditions()
                    stl_formulas.append(copy.deepcopy(STLAtomicConditionToInterval(condition, current_state[:num_states], state_list=state_list[:num_states], initial_tick=timestep)))
        
        # If the new STL formula is a string, we need to parse it
        else:
            parser = STLParser(formula=new_stl_formula)
            condition = parser.get_atomic_conditions()
            stl_formulas.append(copy.deepcopy(STLAtomicConditionToInterval(condition, current_state[:num_states], state_list=state_list[:num_states], initial_tick=timestep)))
    
    return stl_formulas


def print_relevant_info(args):
    """
    Print all relevant information for the reachability analysis. 

    Args:
        args (object): Arguments object for the reachability analysis.
    """
    print("\n=====================================")
    print(f"STL to Interval for ZPC")
    print("=====================================")

    time.sleep(1)

    print("\n=====================================")
    print(f"Model name: {args.model_name}")
    print(f"Number of steps: {args.steps}")
    print(f"Plot: {args.plot}")
    if args.index > 0:
        print(f"Index: {args.index}")
    print(f"Frequency: {args.frequency}")
    print("=====================================\n")

    time.sleep(2)

# Example usage
if __name__ == "__main__":
    # Path to the data driven model
    safety_path = rospkg.RosPack().get_path('safety')
    data_driven_model_path = f"{safety_path}/scripts/data_driven_model/models"

    # Set the batch name and frequency as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help="Name of the batch to get the current state of the vehicle", required=True)
    parser.add_argument('--steps', '-s', type=int, help="Number of time steps for reachability analysis", default=20, required=False)
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the states and inputs", required=False)
    parser.add_argument('--index', '-i', type=int, help="Index of the test", default=0, required=False)
    parser.add_argument('--frequency', '-f', type=float, help="Frequency of the data", default=10.0, required=False)
    parser.add_argument('--debug', '-d', action='store_true', help="Debug mode", required=False)
    parser.add_argument('--fault_iteration', '-fi', type=int, help="Fault iteration", required=False)
    parser.add_argument('--yaw', '-y', action='store_true', help="Plot yaw", required=False)
    parser.add_argument('--stl_formula', '-stl', type=str, help="STL formula for the reachability analysis. Example: G [1,inf] (v<=5)", default=None, required=False)
    args = parser.parse_args()

    # Print all relevant information for the reachability analysis
    print_relevant_info(args)

    # Load the trained data driven model for the reachability analysis
    model_file_path = f"{data_driven_model_path}/reachability_object_{args.model_name}.pkl"
    with open(model_file_path, "rb") as file:
        data_driven_model = dill.load(file)

    if args.plot:
        # Setup paths for saving plots
        safety_path = rospkg.RosPack().get_path('safety')
        plots_folder = f"{safety_path}/scripts/data_driven_model/plots/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(plots_folder, exist_ok=True)
        zpc_visualizer = ZPCVisualizer(show_plots=False, save_plots=True, plots_folder=plots_folder, debug=args.debug, plot_map=True, plot_yaw=args.yaw)
        zpc_time_plotter = ZPCTimePlotter(show_plots=False, save_plots=True, plots_folder=plots_folder, plot_yaw=args.yaw)

    # [x, y, cos_yaw, sin_yaw, vx, w]
    frequency = args.frequency
    steps_reach = args.steps
    init_time = time.time()

    stl_formulas = list()
    new_stl_formulas = ["G [3,10] (vx<=8.)", "G [3,10] (x>=845)", "G [3,10] (x<=890)", "G [3,10] (y>=520)", "G [3,10] (y<=525)", "G [2,10] (sin_yaw>=-0.5)", "G [2,10] (sin_yaw<=0.5)", "G [2,10] (cos_yaw<=-0.5)"]
    selected_stl_formula = args.stl_formula if args.stl_formula is not None else new_stl_formulas
    n_iterations = 500
    fault_iteration = args.fault_iteration if args.fault_iteration is not None else int(n_iterations/50)

    # Get index list for saving plots for the sliding window
    index_plot = np.arange(0, n_iterations, 5)
    idx_test = int(data_driven_model.get_random_test_idx()) if args.index == 0 else args.index
    init_idx = idx_test
    index_plot += idx_test
    print(f"\nRunning reachability analysis for {n_iterations} iterations\n")

    # Iterate over the test data and plot the reachable sets for each timestep
    for iteration in range(n_iterations):
        idx = iteration + idx_test
        current_state = data_driven_model.x_meas_vec_0_test[:, idx]
        next_states = data_driven_model.x_meas_vec_1_test[:, idx-1:idx+steps_reach]
        current_input = data_driven_model.u_test[:, idx:idx+steps_reach]
        failure_mode = (iteration >= fault_iteration)
        
        if iteration == fault_iteration:
            stl_formulas = add_stl_formula(current_state=current_state, state_list=["x", "y", "cos_yaw", "sin_yaw", "v"],
                                            new_stl_formula=selected_stl_formula, stl_formulas=stl_formulas, timestep=init_time+(iteration/frequency))
                
        # Run the STL formulation
        cz_intersection_stl_reach = run_stl_formulation(current_state=current_state, stl_formulas=stl_formulas, 
                                                        steps_reach=steps_reach,frequency=frequency, plot=args.plot,
                                                        idx=idx_test, index_plot=index_plot, next_states=next_states,
                                                        current_input=current_input, state_list=["x", "y", "cos_yaw", "sin_yaw", "v"],
                                                        failure_mode=failure_mode, timestep=init_time+(iteration/frequency))
        completion_bar(iteration+1, n_iterations)

    print("\nSTL Completed\n")