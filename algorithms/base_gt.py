import numpy as np
from config import *

class BaseGameTheory:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.zeros((POP_SIZE, NUM_TASKS), dtype=int)
        self.strategy_probs = np.full((NUM_TASKS, NUM_EDGE_NODES), 1/NUM_EDGE_NODES)
    
    def _update_strategies(self):
        """Logit Dynamics probability update."""
        for i in range(NUM_TASKS):
            utilities = np.array([self._compute_utility(i, j) for j in range(NUM_EDGE_NODES)])
            self.strategy_probs[i] = np.exp(BETA * utilities) / np.sum(np.exp(BETA * utilities))
    
    def _compute_utility(self, task_idx, node_idx):
        """Utility function incorporating mission-critical status."""
        task = self.tasks[task_idx]
        node = self.edge_nodes[node_idx]
        
        # Calculate processing and transmission time
        proc_time = task['cpu'] / node['cpu_cap']
        tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
        total_time = proc_time + tx_time
        
        # Penalize if deadline is violated (especially for mission-critical tasks)
        deadline_violation = max(0, total_time - task['deadline'])
        if task['mission_critical']:
            deadline_violation *= 2  # Double penalty for mission-critical tasks
            
        return - (ALPHA * total_time + GAMMA * node['energy_cost'] * task['cpu'] + deadline_violation)
    
    def _initialize_population(self):
        """Initialize solutions using Logit Dynamics."""
        for _ in range(10):  # Warm-up iterations
            self._update_strategies()
        self.population = np.array([
            [np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[task_idx]) 
            for task_idx in range(NUM_TASKS)]
            for _ in range(POP_SIZE)
        ])