import numpy as np
from .base_gwo import BaseGWO
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, POP_SIZE

class HybridFPGWO(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.beliefs = np.full((NUM_TASKS, NUM_EDGE_NODES), 1 / NUM_EDGE_NODES)
        
    def _compute_utility(self, task_idx, node_idx):
        task = self.tasks[task_idx]
        node = self.edge_nodes[node_idx]
        proc_time = task['cpu'] / node['cpu_cap']
        tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
        return -ALPHA * (proc_time + tx_time) - GAMMA * node['energy_cost'] * task['cpu']
    
    def _update_strategies(self):
        # Best respond to current beliefs
        solution = np.zeros(NUM_TASKS, dtype=int)
        for i in range(NUM_TASKS):
            best_utility = -np.inf
            for j in range(NUM_EDGE_NODES):
                utility = self._compute_utility(i, j)
                if utility > best_utility:
                    best_utility = utility
                    solution[i] = j
        
        # Update beliefs based on current solution
        for i in range(NUM_TASKS):
            self.beliefs[i, solution[i]] += 1
        self.beliefs /= np.sum(self.beliefs, axis=1, keepdims=True)
        
        return solution
    
    def _compute_fitness(self, solution):
        base_fitness = super()._compute_base_fitness(solution)
        strategy_penalty = 0
        for task_idx, node_idx in enumerate(solution):
            chosen_prob = self.beliefs[task_idx, node_idx]
            strategy_penalty += -np.log(chosen_prob + 1e-10)
        return base_fitness / (1 + strategy_penalty / NUM_TASKS)
    
    def optimize(self):
        # Initialize population using Fictitious Play
        for _ in range(10):  # Warm-up iterations
            self._update_strategies()
        for i in range(POP_SIZE):
            self.population[i] = [np.random.choice(NUM_EDGE_NODES, p=self.beliefs[task_idx]) 
                                  for task_idx in range(NUM_TASKS)]
        return super().optimize()