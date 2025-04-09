import numpy as np
from config import MAX_ITER, NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH
from .base_gwo import BaseGWO

class HybridHEFT(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)

    def optimize(self):
        self.convergence = []
        solution = self.heft_scheduling()
        fitness = self._compute_fitness(solution)
        
        # Create artificial convergence for visualization
        self.convergence = [fitness] * MAX_ITER  
        return solution, self.convergence

    def heft_scheduling(self):
        # Rank tasks by upward rank (simplified)
        task_ranks = []
        for task in self.tasks:
            min_time = min(
                task['cpu'] / node['cpu_cap'] + 
                (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                for node in self.edge_nodes
            )
            task_ranks.append(min_time)
        
        # Schedule in rank order
        schedule = np.zeros(NUM_TASKS, dtype=int)
        node_available = [0] * NUM_EDGE_NODES
        
        for task_idx in np.argsort(task_ranks)[::-1]:  # Highest rank first
            task = self.tasks[task_idx]
            best_node = 0
            best_finish = float('inf')
            
            for node_idx in range(NUM_EDGE_NODES):
                proc_time = task['cpu'] / self.edge_nodes[node_idx]['cpu_cap']
                tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(
                    self.edge_nodes[node_idx]['loc'] - task['loc'])
                finish = max(node_available[node_idx], 0) + proc_time + tx_time
                
                if finish < best_finish:
                    best_finish = finish
                    best_node = node_idx
            
            schedule[task_idx] = best_node
            node_available[best_node] = best_finish
            
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