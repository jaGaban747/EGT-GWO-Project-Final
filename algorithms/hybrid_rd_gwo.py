import numpy as np
from .base_gwo import BaseGWO
from config import ALPHA, BANDWIDTH, GAMMA, NUM_EDGE_NODES, NUM_TASKS, POP_SIZE

class ReplicatorDynamicsGame:
    def __init__(self, tasks, edge_nodes, alpha, gamma):
        """
        Replicator Dynamics game theory model.
        
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
        Update strategies using Replicator Dynamics.
        
        Returns:
        - Updated strategy probabilities matrix
        """
        fitness = np.zeros((NUM_TASKS, NUM_EDGE_NODES))
        current_response_times = np.zeros((NUM_TASKS, NUM_EDGE_NODES))
        
        # Calculate fitness for all task-node pairs
        for i in range(NUM_TASKS):
            for j in range(NUM_EDGE_NODES):
                fitness[i, j], current_response_times[i, j] = self.calculate_utility(i, j)
        
        # Calculate average fitness for each task
        avg_fitness = np.sum(self.strategy_probs * fitness, axis=1, keepdims=True)
        
        # Replicator dynamics update with numerical stability
        update_ratio = np.clip(fitness / (avg_fitness + 1e-10), 0.1, 10)  # Prevent extreme updates
        new_probs = self.strategy_probs * update_ratio
        
        # Normalize and add small noise for exploration
        new_probs = new_probs / np.sum(new_probs, axis=1, keepdims=True)
        new_probs = 0.95 * new_probs + 0.05 * np.random.uniform(0, 1, size=new_probs.shape)
        new_probs = new_probs / np.sum(new_probs, axis=1, keepdims=True)
        
        # Track history
        self.strategy_probs = new_probs
        self.convergence_history.append(np.argmax(new_probs, axis=1))
        self.response_times.append(np.mean(current_response_times))
        self.utility_history.append(np.mean(fitness))
        
        return new_probs

    def get_evolutionarily_stable_strategy(self, max_iter=50, tol=1e-4):
        """
        Iteratively compute evolutionarily stable strategy.
        
        Parameters:
        - max_iter: Maximum number of iterations
        - tol: Tolerance for convergence
        
        Returns:
        - Stable strategy probabilities
        """
        for _ in range(max_iter):
            old_probs = self.strategy_probs.copy()
            self.update_strategies()
            
            # Check for convergence
            if np.max(np.abs(self.strategy_probs - old_probs)) < tol:
                break
                
        return self.strategy_probs

class HybridRDGWO(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.rd_game = ReplicatorDynamicsGame(tasks, edge_nodes, ALPHA, GAMMA)
        self.convergence_history = []
        self.response_time_history = []

    def _initialize_population(self):
        """Initialize population using Replicator Dynamics strategies."""
        # Warm-up iterations for RD
        for _ in range(5):
            self.rd_game.update_strategies()
        
        # Initialize population with mixed strategies
        for i in range(POP_SIZE):
            if i < POP_SIZE // 2:
                # Sample from RD probabilities
                self.population[i] = [np.random.choice(
                    NUM_EDGE_NODES, 
                    p=self.rd_game.strategy_probs[task_idx]
                ) for task_idx in range(NUM_TASKS)]
            else:
                # Random initialization for diversity
                self.population[i] = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)

    def _compute_fitness(self, solution):
            """Compute stable, normalized fitness with RD strategy penalty."""
            
            # Step 1: Compute and normalize base fitness
            base_fitness = super()._compute_base_fitness(solution)
            normalized_fitness = np.log1p(base_fitness)  # log(1 + base_fitness)

            # Step 2: Compute strategy penalty from RD game probabilities
            strategy_penalty = 0
            for task_idx, node_idx in enumerate(solution):
                chosen_prob = self.rd_game.strategy_probs[task_idx, node_idx]
                strategy_penalty += min(20, -np.log(chosen_prob + 1e-6))  # Cap to prevent explosion

            # Step 3: Track response time (optional, used elsewhere)
            response_time = 0
            for task_idx, node_idx in enumerate(solution):
                task = self.tasks[task_idx]
                node = self.edge_nodes[node_idx]
                proc_time = task['cpu'] / node['cpu_cap']
                tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                response_time += proc_time + tx_time
            self.response_time_history.append(response_time / NUM_TASKS)

            # Step 4: Combine fitness and penalty using weighted sum
            penalty_weight = 0.1
            fitness = normalized_fitness + penalty_weight * strategy_penalty

            return fitness

        

    def optimize(self):
        """Run the hybrid RD-GWO optimization."""
        self._initialize_population()
        
        # Track initial convergence
        self.convergence_history.append(np.copy(self.rd_game.convergence_history))
        
        # Run standard GWO optimization
        result = super().optimize()
        
        # Add final convergence info
        self.convergence_history.append(self.rd_game.convergence_history[-1])
        return result