import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, MAX_ITER
from .base_gwo import BaseGWO

class HybridMaxMin(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)

    def optimize(self):
        self.convergence = []
        solution = self.max_min_scheduling()
        fitness = self._compute_fitness(solution)
        
        # Create artificial convergence
        self.convergence = [fitness] * MAX_ITER
        return solution, self.convergence

    def max_min_scheduling(self):
        # Initialize
        schedule = np.zeros(NUM_TASKS, dtype=int)
        node_times = np.zeros(NUM_EDGE_NODES)
        
        # Sort tasks by decreasing computation requirement
        task_order = sorted(range(NUM_TASKS), 
                          key=lambda i: -self.tasks[i]['cpu'])
        
        for task_idx in task_order:
            task = self.tasks[task_idx]
            best_node = 0
            min_finish = float('inf')
            
            for node_idx in range(NUM_EDGE_NODES):
                node = self.edge_nodes[node_idx]
                proc_time = task['cpu'] / node['cpu_cap']
                tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                finish = node_times[node_idx] + proc_time + tx_time
                
                if finish < min_finish:
                    min_finish = finish
                    best_node = node_idx
            
            schedule[task_idx] = best_node
            node_times[best_node] = min_finish
            
        return schedule

    def _compute_fitness(self, solution):
        latency, energy = 0, 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            latency += ALPHA * (proc_time + tx_time)
            energy += GAMMA * node['energy_cost'] * task['cpu']
        return latency + energy