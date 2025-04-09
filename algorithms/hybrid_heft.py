# hybrid_heft.py

import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, POP_SIZE
from .base_gwo import BaseGWO

class HybridHEFT(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)

    def optimize(self):
        # Perform HEFT task scheduling for offloading
        schedule = self.heft_scheduling()
        return schedule, self.fitness

    def heft_scheduling(self):
        # Schedule tasks based on HEFT
        task_schedule = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)
        return task_schedule

    def _compute_fitness(self, solution):
        # Compute fitness based on HEFT task allocation
        latency, energy = 0, 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            latency += ALPHA * (proc_time + tx_time)
            energy += GAMMA * node['energy_cost'] * task['cpu']
        return latency, energy
