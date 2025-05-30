import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridAO:
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
            x = iter / MAX_ITER
            for i in range(POP_SIZE):
                if x < 0.5:  # Exploration
                    if np.random.random() < 0.5:
                        new_pos = np.clip(
                            self.best_position * (1 - x) + 
                            np.mean(self.population, axis=0) * x +
                            np.random.random(size=NUM_TASKS) * (NUM_EDGE_NODES-1),
                            0, NUM_EDGE_NODES-1
                        ).astype(int)
                    else:
                        levy = np.random.standard_cauchy(size=NUM_TASKS)
                        new_pos = np.clip(
                            self.best_position * levy + 
                            self.population[np.random.randint(POP_SIZE)] +
                            np.random.random(size=NUM_TASKS) * (NUM_EDGE_NODES-1),
                            0, NUM_EDGE_NODES-1
                        ).astype(int)
                else:  # Exploitation
                    if np.random.random() < 0.5:
                        new_pos = np.clip(
                            (self.best_position - np.mean(self.population, axis=0)) * (1 - x) + 
                            np.random.random(size=NUM_TASKS) * (NUM_EDGE_NODES-1),
                            0, NUM_EDGE_NODES-1
                        ).astype(int)
                    else:
                        QF = iter ** ((2*np.random.random()-1)/(1-MAX_ITER)**2)
                        new_pos = np.clip(
                            QF * self.best_position - 
                            (2*np.random.random(size=NUM_TASKS)-1) * 
                            np.abs(np.mean(self.population, axis=0) - self.population[i]),
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