# hybrid_lyapunov.py

import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, POP_SIZE, MAX_ITER
from .base_gwo import BaseGWO

class HybridLyapunov(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.queue_backlog = np.zeros(NUM_TASKS)
        self.fitness = float('inf')

    def optimize(self):
        task_schedule = self.lyapunov_optimization()
        return task_schedule, self.fitness

    def lyapunov_optimization(self):
        """
        Implement Lyapunov optimization for task offloading
        Returns a task schedule (array where each index represents a task and the value represents the assigned edge node)
        """
        best_solution = np.zeros(NUM_TASKS, dtype=int)
        best_fitness = float('inf')
        
        # Initialize drift-plus-penalty parameter
        V = 0.1
        
        # Number of iterations for the Lyapunov optimization
        for iter_count in range(MAX_ITER):
            # Generate a candidate solution
            current_solution = np.zeros(NUM_TASKS, dtype=int)
            
            # For each task, find the optimal edge node assignment
            for task_idx in range(NUM_TASKS):
                min_drift_penalty = float('inf')
                best_node = 0
                
                # For each edge node, compute the drift-plus-penalty
                for node_idx in range(NUM_EDGE_NODES):
                    task = self.tasks[task_idx]
                    node = self.edge_nodes[node_idx]
                    
                    # Compute processing time and transmission time
                    proc_time = task['cpu'] / node['cpu_cap']
                    tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                    latency = proc_time + tx_time
                    energy = node['energy_cost'] * task['cpu']
                    
                    # Compute drift
                    drift = self.queue_backlog[task_idx] * latency
                    
                    # Compute penalty (weighted sum of latency and energy)
                    penalty = V * (ALPHA * latency + GAMMA * energy)
                    
                    # Compute drift-plus-penalty
                    drift_penalty = drift + penalty
                    
                    # Update best node if current is better
                    if drift_penalty < min_drift_penalty:
                        min_drift_penalty = drift_penalty
                        best_node = node_idx
                
                # Assign task to best node
                current_solution[task_idx] = best_node
            
            # Update queue backlogs
            for task_idx in range(NUM_TASKS):
                node_idx = current_solution[task_idx]
                task = self.tasks[task_idx]
                node = self.edge_nodes[node_idx]
                proc_time = task['cpu'] / node['cpu_cap']
                tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
                
                # Queue dynamics (arrival rate - service rate)
                arrival_rate = task['arrival_rate'] if 'arrival_rate' in task else 1.0
                service_rate = 1.0 / (proc_time + tx_time)
                
                # Update queue backlog (ensure it doesn't go negative)
                self.queue_backlog[task_idx] = max(0, self.queue_backlog[task_idx] + arrival_rate - service_rate)
            
            # Evaluate the current solution
            latency, energy = self._compute_fitness(current_solution)
            current_fitness = ALPHA * latency + GAMMA * energy
            
            # Update best solution if current is better
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution.copy()
        
        # Set the fitness for the returned solution
        self.fitness = best_fitness
        return best_solution

    def _compute_fitness(self, solution):
        # Compute fitness using Lyapunov optimization
        latency, energy = 0, 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            latency += proc_time + tx_time
            energy += node['energy_cost'] * task['cpu']
        return latency, energy