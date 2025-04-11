import numpy as np 
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, POP_SIZE, MAX_ITER
from .base_gwo import BaseGWO

class HybridPSO(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.velocity = np.zeros((POP_SIZE, NUM_TASKS))
        self.pbest = self.population.copy()
        self.pbest_fitness = np.array([self._compute_fitness(sol) for sol in self.population])
        self.gbest = self.pbest[np.argmax(self.pbest_fitness)]

    def optimize(self):
        w, c1, c2 = 0.5, 1.5, 1.5  # Inertia, cognitive, social
        self.convergence = []

        for _ in range(MAX_ITER):
            for i in range(POP_SIZE):
                r1 = np.random.rand(NUM_TASKS)
                r2 = np.random.rand(NUM_TASKS)

                cognitive = c1 * r1 * (self.pbest[i] - self.population[i])
                social = c2 * r2 * (self.gbest - self.population[i])
                self.velocity[i] = w * self.velocity[i] + cognitive + social

                self.population[i] = np.clip(np.round(self.population[i] + self.velocity[i]), 0, NUM_EDGE_NODES - 1).astype(int)

                fitness = self._compute_fitness(self.population[i])
                if fitness > self.pbest_fitness[i]:
                    self.pbest[i] = self.population[i].copy()
                    self.pbest_fitness[i] = fitness

            best_idx = np.argmax(self.pbest_fitness)
            self.gbest = self.pbest[best_idx]
            self.convergence.append(1 / self.pbest_fitness[best_idx])  # Assuming lower fitness is better

        return self.gbest, self.convergence

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
