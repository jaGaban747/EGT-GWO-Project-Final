import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class BaseGWO:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, (POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.alpha_pos = self.beta_pos = self.delta_pos = None
        self.convergence = []
        
    def _compute_base_fitness(self, solution):
        latency = energy = 0
        node_loads = np.zeros(NUM_EDGE_NODES)
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            latency += ALPHA * (proc_time + tx_time)
            energy += GAMMA * node['energy_cost'] * task['cpu']
            node_loads[node_idx] += task['cpu']
        overload = np.sum(np.maximum(node_loads - [n['cpu_cap'] for n in self.edge_nodes], 0))
        penalty = 1e6 * overload
        return 1 / (latency + energy + penalty + 1e-10)
    
    def optimize(self):
        for iter in range(MAX_ITER):
            self.fitness = np.array([self._compute_fitness(sol) for sol in self.population])
            sorted_indices = np.argsort(self.fitness)[::-1]
            self.alpha_pos = self.population[sorted_indices[0]]
            self.beta_pos = self.population[sorted_indices[1]]
            self.delta_pos = self.population[sorted_indices[2]]

            a = 2 - (2 * iter) / MAX_ITER
            for i in range(POP_SIZE):
                A1, A2, A3 = 2*a*np.random.rand(NUM_TASKS) - a, 2*a*np.random.rand(NUM_TASKS) - a, 2*a*np.random.rand(NUM_TASKS) - a
                C1, C2, C3 = 2*np.random.rand(NUM_TASKS), 2*np.random.rand(NUM_TASKS), 2*np.random.rand(NUM_TASKS)
                new_pos = (self.alpha_pos*(1 - A1*C1) + self.beta_pos*(1 - A2*C2) + self.delta_pos*(1 - A3*C3)) / 3
                new_pos = np.clip(np.round(new_pos), 0, NUM_EDGE_NODES-1).astype(int)
                self.population[i] = new_pos

            self.convergence.append(1 / self.fitness[sorted_indices[0]])
        return self.alpha_pos, self.convergence

    def _compute_fitness(self, solution):
        # To be overridden by hybrid classes
        return self._compute_base_fitness(solution)