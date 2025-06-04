import numpy as np
from .base_gwo import BaseGWO
from config import ALPHA, BANDWIDTH, BETA, GAMMA, NUM_EDGE_NODES, NUM_TASKS, POP_SIZE
from .base_gt import LogitGameTheory

class HybridLDGWO(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.logit_game = LogitGameTheory(tasks, edge_nodes, ALPHA, BETA, GAMMA)
        self.equilibrium_history = []
        self.utility_history = []
        self.response_times = []
        
    def _initialize_population(self):
        # Get Nash equilibrium probabilities
        equilibrium_probs = self.logit_game.get_nash_equilibrium()
        
        # Initialize half of the population from equilibrium, half randomly
        for i in range(POP_SIZE):
            if i < POP_SIZE // 2:
                # Sample from equilibrium probabilities for each task
                self.population[i] = [np.random.choice(NUM_EDGE_NODES, p=equilibrium_probs[task_idx])
                                    for task_idx in range(NUM_TASKS)]
            else:
                # Random initialization for diversity
                self.population[i] = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)

    def _compute_utility(self, task_idx, node_idx):
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
        latency_util = -ALPHA * response_time
        energy_util = -GAMMA * node['energy_cost'] * task['cpu']
        
        return latency_util + energy_util, response_time

    def _update_strategies(self):
        new_probs = np.zeros((NUM_TASKS, NUM_EDGE_NODES))
        all_utilities = []
        response_times = []
        
        for i in range(NUM_TASKS):
            utilities_and_times = [self._compute_utility(i, j) for j in range(NUM_EDGE_NODES)]
            utilities = [x[0] for x in utilities_and_times]
            times = [x[1] for x in utilities_and_times]
            
            all_utilities.append(utilities)
            response_times.append(times)
            
            # Apply logit function to utilities
            exp_utilities = np.exp(BETA * np.array(utilities))
            new_probs[i] = exp_utilities / np.sum(exp_utilities)
        
        self.strategy_probs = new_probs
        self.utility_history.append(all_utilities)
        self.response_times.append(response_times)
        self.equilibrium_history.append(np.copy(new_probs))
        
        return new_probs

    def _compute_fitness(self, solution):
        """Compute stable, normalized fitness with game-theoretic strategy penalty."""
    
        # Step 1: Compute and normalize base fitness
        base_fitness = super()._compute_base_fitness(solution)
        normalized_fitness = np.log1p(base_fitness)  # log(1 + base_fitness) for safer scaling

        # Step 2: Compute strategy penalty (deviation from equilibrium)
        strategy_penalty = 0
        for task_idx, node_idx in enumerate(solution):
            chosen_prob = self.strategy_probs[task_idx, node_idx]
            strategy_penalty += min(20, -np.log(chosen_prob + 1e-6))  # Cap to avoid extreme penalty

        # Step 3: Combine with weight (no division for stability)
        penalty_weight = 0.1  # Tunable value
        fitness = normalized_fitness + penalty_weight * strategy_penalty

        return fitness


    def optimize(self):
        # Initialize population using game theory
        self._initialize_population()
        
        # Warm-up iterations for logit dynamics
        for _ in range(10):
            self._update_strategies()
        
        # Run standard GWO optimization
        return super().optimize()