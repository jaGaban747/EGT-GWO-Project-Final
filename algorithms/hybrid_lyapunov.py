import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, MAX_ITER
from .base_gwo import BaseGWO

class HybridLyapunov(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.queue_backlog = np.zeros(NUM_TASKS)
        self.V = 0.1  # Lyapunov trade-off parameter

    def optimize(self):
        self.convergence = []
        best_solution = None
        best_fitness = float('inf')
        
        for iter in range(MAX_ITER):
            solution = np.zeros(NUM_TASKS, dtype=int)
            
            # Task assignment
            for task_idx in range(NUM_TASKS):
                min_drift_penalty = float('inf')
                best_node = 0
                
                for node_idx in range(NUM_EDGE_NODES):
                    task = self.tasks[task_idx]
                    node = self.edge_nodes[node_idx]
                    
                    proc_time = task['cpu'] / node['cpu_cap']
                    tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                    
                    drift = self.queue_backlog[task_idx] * (proc_time + tx_time)
                    penalty = self.V * (ALPHA * (proc_time + tx_time) + GAMMA * node['energy_cost'] * task['cpu'])
                    total = drift + penalty
                    
                    if total < min_drift_penalty:
                        min_drift_penalty = total
                        best_node = node_idx
                
                solution[task_idx] = best_node
            
            # Update queues
            for task_idx in range(NUM_TASKS):
                node_idx = solution[task_idx]
                task = self.tasks[task_idx]
                node = self.edge_nodes[node_idx]
                
                proc_time = task['cpu'] / node['cpu_cap']
                tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                
                arrival_rate = task.get('arrival_rate', 1.0)
                service_rate = 1.0 / (proc_time + tx_time)
                self.queue_backlog[task_idx] = max(0, self.queue_backlog[task_idx] + arrival_rate - service_rate)
            
            # Track convergence
            current_fitness = self._compute_fitness(solution)
            self.convergence.append(current_fitness)
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = solution.copy()
                
        return best_solution, self.convergence

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