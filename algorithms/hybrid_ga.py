import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, POP_SIZE, MAX_ITER
from .base_gwo import BaseGWO

class HybridGA(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.fitness_history = []

    def optimize(self):
        self.convergence = []
        for generation in range(MAX_ITER):
            # Evaluate fitness
            fitness = [self._compute_fitness(ind) for ind in self.population]
            self.convergence.append(min(fitness))
            
            # Selection (Tournament)
            new_population = []
            for _ in range(POP_SIZE):
                # Select 2 random individuals
                a, b = np.random.randint(0, POP_SIZE, 2)
                winner = a if fitness[a] < fitness[b] else b
                new_population.append(self.population[winner].copy())
            
            # Crossover (Single Point)
            for i in range(0, POP_SIZE, 2):
                if i+1 >= POP_SIZE: break
                crossover_point = np.random.randint(1, NUM_TASKS-1)
                temp = new_population[i][crossover_point:].copy()
                new_population[i][crossover_point:] = new_population[i+1][crossover_point:]
                new_population[i+1][crossover_point:] = temp
            
            # Mutation
            for i in range(POP_SIZE):
                if np.random.rand() < 0.1:  # 10% mutation rate
                    gene = np.random.randint(0, NUM_TASKS)
                    new_population[i][gene] = np.random.randint(0, NUM_EDGE_NODES)
            
            self.population = np.array(new_population)
            
        best_idx = np.argmin([self._compute_fitness(ind) for ind in self.population])
        return self.population[best_idx], self.convergence

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