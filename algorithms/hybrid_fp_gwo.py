import numpy as np
from .base_gwo import BaseGWO
from config import ALPHA, BANDWIDTH, GAMMA, NUM_EDGE_NODES, NUM_TASKS, POP_SIZE

class FictitiousPlayGame:
    def __init__(self, tasks, edge_nodes, alpha, gamma):
        """
        Fictitious Play game theory model.
        
        Parameters:
        - tasks: List of tasks with their requirements
        - edge_nodes: List of edge nodes with their capabilities
        - alpha: Weight parameter for latency utility
        - gamma: Weight parameter for energy utility
        """
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.beliefs = np.full((NUM_TASKS, NUM_EDGE_NODES), 1 / NUM_EDGE_NODES)
        self.convergence_history = []
        self.response_times = []
        self.utility_history = []

    def calculate_utility(self, task_idx, node_idx):
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
        tx_time = (task['data'] / BANDWIDTH) * (distance / 100)
        
        # Calculate response time (processing + transmission)
        response_time = proc_time + tx_time
        
        # Utility components
        latency_util = -self.alpha * response_time
        energy_util = -self.gamma * node['energy_cost'] * task['cpu']
        
        return latency_util + energy_util, response_time

    def update_strategies(self):
        """
        Update strategies using Fictitious Play dynamics.
        
        Returns:
        - Updated belief probabilities matrix
        """
        solution = np.zeros(NUM_TASKS, dtype=int)
        current_utilities = []
        current_response_times = []
        
        for i in range(NUM_TASKS):
            best_utility = -np.inf
            best_response_time = 0
            
            for j in range(NUM_EDGE_NODES):
                utility, response_time = self.calculate_utility(i, j)
                current_utilities.append(utility)
                
                if utility > best_utility:
                    best_utility = utility
                    solution[i] = j
                    best_response_time = response_time
            
            current_response_times.append(best_response_time)
        
        # Update beliefs (empirical frequencies)
        for i in range(NUM_TASKS):
            self.beliefs[i, solution[i]] += 1
        
        # Normalize and smooth beliefs
        self.beliefs = 0.9 * (self.beliefs / np.sum(self.beliefs, axis=1, keepdims=True)) + 0.1/NUM_EDGE_NODES
        
        # Track history
        self.convergence_history.append(solution.copy())
        self.response_times.append(np.mean(current_response_times))
        self.utility_history.append(np.mean(current_utilities))
        
        return self.beliefs

    def get_nash_equilibrium(self, max_iter=50, tol=1e-4):
        """
        Iteratively compute the Nash equilibrium using Fictitious Play.
        
        Parameters:
        - max_iter: Maximum number of iterations
        - tol: Tolerance for convergence
        
        Returns:
        - Equilibrium belief probabilities
        """
        for _ in range(max_iter):
            old_beliefs = self.beliefs.copy()
            self.update_strategies()
            
            # Check for convergence
            if np.max(np.abs(self.beliefs - old_beliefs)) < tol:
                break
                
        return self.beliefs

class HybridFPGWO(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.fp_game = FictitiousPlayGame(tasks, edge_nodes, ALPHA, GAMMA)
        self.convergence_history = []
        self.response_time_history = []

    def _initialize_population(self):
        """Initialize population using Fictitious Play strategies."""
        # Warm-up iterations for FP
        for _ in range(5):
            self.fp_game.update_strategies()
        
        # Initialize population with mixed strategies
        for i in range(POP_SIZE):
            if i < POP_SIZE // 2:
                # Sample from FP beliefs
                self.population[i] = [np.random.choice(
                    NUM_EDGE_NODES, 
                    p=self.fp_game.beliefs[task_idx]
                ) for task_idx in range(NUM_TASKS)]
            else:
                # Random initialization for diversity
                self.population[i] = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)

    def _compute_fitness(self, solution):
        """Compute stable, normalized fitness with FP strategy penalty."""
        
        # Step 1: Compute and normalize base fitness
        base_fitness = super()._compute_base_fitness(solution)
        normalized_fitness = np.log1p(base_fitness)  # log(1 + base_fitness) for stability
        
        # Step 2: Compute penalty based on FP beliefs
        strategy_penalty = 0
        for task_idx, node_idx in enumerate(solution):
            chosen_prob = self.fp_game.beliefs[task_idx, node_idx]
            strategy_penalty += min(20, -np.log(chosen_prob + 1e-6))  # Cap extreme penalties

        # Step 3: Optionally track response time (used elsewhere)
        response_time = 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            response_time += proc_time + tx_time
        self.response_time_history.append(response_time / NUM_TASKS)

        # Step 4: Combine fitness and penalty
        penalty_weight = 0.1  # You can adjust this
        fitness = normalized_fitness + penalty_weight * strategy_penalty

        return fitness

    def optimize(self):
        """Run the hybrid FP-GWO optimization."""
        self._initialize_population()
        
        # Track initial convergence
        self.convergence_history.append(np.copy(self.fp_game.convergence_history))
        
        # Run standard GWO optimization
        result = super().optimize()
        
        # Add final convergence info
        self.convergence_history.append(self.fp_game.convergence_history[-1])
        return result