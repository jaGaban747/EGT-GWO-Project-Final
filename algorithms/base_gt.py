import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, BANDWIDTH

class LogitGameTheory:
    def __init__(self, tasks, edge_nodes, alpha, beta, gamma):
        """
        Initialize the Logit Dynamics game theory model.
        
        Parameters:
        - tasks: List of tasks with their requirements
        - edge_nodes: List of edge nodes with their capabilities
        - alpha: Weight parameter for latency utility
        - beta: Rationality parameter (inverse temperature)
        - gamma: Weight parameter for energy utility
        """
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize strategy probabilities uniformly
        self.strategy_probs = np.full((NUM_TASKS, NUM_EDGE_NODES), 1 / NUM_EDGE_NODES)
        
        # History tracking
        self.utility_history = []
        self.equilibrium_history = []
        self.response_times = []

    def calculate_utilities(self, task_idx, node_idx):
        """
        Calculate the utility for a task-node pairing.
        
        Returns:
        - utility: Combined utility value
        - response_time: Total response time (processing + transmission)
        """
        task = self.tasks[task_idx]
        node = self.edge_nodes[node_idx]
        
        # Processing time based on CPU requirements and capacity
        proc_time = task['cpu'] / node['cpu_cap']
        
        # Transmission time based on data size, bandwidth, and distance
        distance = np.linalg.norm(np.array(node['loc']) - np.array(task['loc']))
        tx_time = (task['data'] / BANDWIDTH) * (distance / 100)  # Normalized distance
        
        # Calculate response time (processing + transmission)
        response_time = proc_time + tx_time
        
        # Utility components
        latency_util = -self.alpha * response_time
        energy_util = -self.gamma * node['energy_cost'] * task['cpu']
        
        return latency_util + energy_util, response_time

    def update_strategies(self):
        """
        Update strategy probabilities based on current utilities using logit dynamics.
        
        Returns:
        - Updated strategy probabilities matrix
        """
        new_probs = np.zeros((NUM_TASKS, NUM_EDGE_NODES))
        all_utilities = []
        response_times = []
        
        for i in range(NUM_TASKS):
            utilities_and_times = [self.calculate_utilities(i, j) for j in range(NUM_EDGE_NODES)]
            utilities = [x[0] for x in utilities_and_times]
            times = [x[1] for x in utilities_and_times]
            
            all_utilities.append(utilities)
            response_times.append(times)
            
            # Apply logit choice function
            exp_utilities = np.exp(self.beta * np.array(utilities))
            new_probs[i] = exp_utilities / np.sum(exp_utilities)
        
        # Update internal state
        self.strategy_probs = new_probs
        self.utility_history.append(all_utilities)
        self.response_times.append(response_times)
        self.equilibrium_history.append(np.copy(new_probs))
        
        return new_probs

    def get_nash_equilibrium(self, max_iter=50, tol=1e-4):
        """
        Iteratively compute the Nash equilibrium using logit dynamics.
        
        Parameters:
        - max_iter: Maximum number of iterations
        - tol: Tolerance for convergence
        
        Returns:
        - Equilibrium strategy probabilities
        """
        for _ in range(max_iter):
            old_probs = self.strategy_probs.copy()
            self.update_strategies()
            
            # Check for convergence
            if np.max(np.abs(self.strategy_probs - old_probs)) < tol:
                break
                
        return self.strategy_probs

    def get_best_responses(self):
        """
        Get the best response strategies (pure strategies with highest probability).
        
        Returns:
        - Array of best edge node indices for each task
        """
        return np.argmax(self.strategy_probs, axis=1)

    def get_expected_utilities(self):
        """
        Calculate the expected utilities for each task under current strategies.
        
        Returns:
        - Array of expected utilities for each task
        """
        expected_utils = np.zeros(NUM_TASKS)
        
        for i in range(NUM_TASKS):
            utilities = [self.calculate_utilities(i, j)[0] for j in range(NUM_EDGE_NODES)]
            expected_utils[i] = np.sum(self.strategy_probs[i] * utilities)
            
        return expected_utils

    def reset_history(self):
        """Clear all history tracking variables."""
        self.utility_history = []
        self.equilibrium_history = []
        self.response_times = []