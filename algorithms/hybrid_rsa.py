import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridRSA:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self.alpha = 0.1
        self.beta = 0.1
        self._initialize_population()

    def _initialize_population(self):
        for i in range(POP_SIZE):
            self.fitness[i] = self._compute_fitness(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()

    def _compute_fitness(self, solution):
        # Same implementation as TSA
        total_latency = 0
        total_energy = 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            total_latency += proc_time + tx_time
            total_energy += node['energy_cost'] * task['cpu']
        return ALPHA * total_latency + GAMMA * total_energy

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            ES = 2 * np.random.random() * (1 - iter/MAX_ITER)
            
            for i in range(POP_SIZE):
                R = self.best_position - self.population[i] / (self.best_position + 1e-10)
                P = self.alpha + (self.population[i] - np.mean(self.population, axis=0)) / (self.best_position + 1e-10)
                stepsize = ES * (P * np.random.random(size=NUM_TASKS)) * R
                
                new_pos = np.clip(
                    self.best_position - stepsize * np.random.random(size=NUM_TASKS),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
                
                new_fitness = self._compute_fitness(new_pos)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_pos.copy()
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve