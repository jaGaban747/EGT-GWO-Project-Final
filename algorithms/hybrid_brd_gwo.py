import numpy as np
from .base_gwo import BaseGWO
from config import ALPHA, BANDWIDTH, GAMMA, NUM_EDGE_NODES, NUM_TASKS, POP_SIZE

class BestResponseGame:
    def __init__(self, tasks, edge_nodes, alpha, gamma):
        """
        Best Response Dynamics game theory model.
        
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
        self.strategy_probs = np.full((NUM_TASKS, NUM_EDGE_NODES), 1 / NUM_EDGE_NODES)
        self.convergence_history = []
        self.response_times = []

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

    def update_strategies(self, max_iter=10):
        """
        Update strategies using Best Response Dynamics.
        
        Parameters:
        - max_iter: Maximum number of BRD iterations
        
        Returns:
        - Updated strategy probabilities matrix
        """
        solution = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)
        converged = False
        iteration = 0
        
        for iteration in range(max_iter):
            new_solution = solution.copy()
            current_response_times = []
            
            for i in range(NUM_TASKS):
                best_utility = -np.inf
                best_node = solution[i]
                
                for j in range(NUM_EDGE_NODES):
                    utility, response_time = self.calculate_utility(i, j)
                    if utility > best_utility:
                        best_utility = utility
                        best_node = j
                        current_response_times.append(response_time)
                
                new_solution[i] = best_node
            
            self.response_times.append(np.mean(current_response_times))
            self.convergence_history.append(np.copy(new_solution))
            
            if np.all(new_solution == solution):
                converged = True
                break
                
            solution = new_solution
        
        # Update strategy probabilities (smoother version)
        for i in range(NUM_TASKS):
            self.strategy_probs[i] = 0.9 * self.strategy_probs[i] + 0.1
            self.strategy_probs[i, solution[i]] += 1
        
        self.strategy_probs /= np.sum(self.strategy_probs, axis=1, keepdims=True)
        return self.strategy_probs

class HybridBRDGWO(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.br_game = BestResponseGame(tasks, edge_nodes, ALPHA, GAMMA)
        self.convergence_history = []
        self.response_time_history = []

    def _initialize_population(self):
        """Initialize population using BRD strategies."""
        # Warm-up iterations for BRD
        for _ in range(5):
            self.br_game.update_strategies()
        
        # Initialize population with mixed strategies
        for i in range(POP_SIZE):
            if i < POP_SIZE // 2:
                # Sample from BRD probabilities
                self.population[i] = [np.random.choice(
                    NUM_EDGE_NODES, 
                    p=self.br_game.strategy_probs[task_idx]
                ) for task_idx in range(NUM_TASKS)]
            else:
                # Random initialization for diversity
                self.population[i] = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)

    def _compute_fitness(self, solution):
        """Compute combined fitness with BRD strategy penalty."""
        base_fitness = super()._compute_base_fitness(solution)
        
        # NORMALIZATION: Scale the fitness to be in the same range as other algorithms
        # Option 1: Simple scaling factor
        normalized_fitness = base_fitness / (NUM_TASKS * 0.01)
        
        # Option 2: Logarithmic scaling (alternative approach)
        # normalized_fitness = np.log(1 + base_fitness)
        
        # BRD strategy alignment penalty
        strategy_penalty = 0
        for task_idx, node_idx in enumerate(solution):
            chosen_prob = self.br_game.strategy_probs[task_idx, node_idx]
            strategy_penalty += -np.log(chosen_prob + 1e-10)
        
        # Track response time for comparison
        response_time = 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            response_time += proc_time + tx_time
        
        self.response_time_history.append(response_time / NUM_TASKS)
        
        # Return normalized fitness with penalty
        return normalized_fitness / (1 + 0.2 * strategy_penalty / NUM_TASKS)

    def optimize(self):
        """Run the hybrid BRD-GWO optimization."""
        self._initialize_population()
        
        # Track initial convergence
        self.convergence_history.append(np.copy(self.br_game.convergence_history))
        
        # Run standard GWO optimization
        result = super().optimize()
        
        # Add final convergence info
        self.convergence_history.append(self.br_game.convergence_history[-1])
        return result