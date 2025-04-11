import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridSSA:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self._initialize_population()

    def _initialize_population(self):
        for i in range(POP_SIZE):
            self.fitness[i] = self._compute_fitness(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()

    def _compute_fitness(self, solution):
        # Same implementation as WOA
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
            c1 = 2 * np.exp(-(4 * iter / MAX_ITER)**2)
            
            # Update leader
            for j in range(NUM_TASKS):
                c2 = np.random.random()
                c3 = np.random.random()
                if c3 < 0.5:
                    self.population[0,j] = np.clip(
                        self.best_position[j] + c1 * ((NUM_EDGE_NODES-1)*c2),
                        0, NUM_EDGE_NODES-1
                    ).astype(int)
                else:
                    self.population[0,j] = np.clip(
                        self.best_position[j] - c1 * ((NUM_EDGE_NODES-1)*c2),
                        0, NUM_EDGE_NODES-1
                    ).astype(int)
            
            # Update followers
            for i in range(1, POP_SIZE):
                self.population[i] = np.clip(
                    0.5 * (self.population[i-1] + self.population[i]),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
            
            # Evaluate
            for i in range(POP_SIZE):
                new_fitness = self._compute_fitness(self.population[i])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = self.population[i].copy()
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve