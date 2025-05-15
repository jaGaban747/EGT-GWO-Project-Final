import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, POP_SIZE, MAX_ITER

class HybridCOA:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self._initialize_population()
        self.alpha = 0.1
        self.PP = 0.1  # Predator presence probability

    def _initialize_population(self):
        for i in range(POP_SIZE):
            self.fitness[i] = self._compute_fitness(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()

    def _compute_fitness(self, solution):
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

    def _attack_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Attack on iguanas
            r = np.random.random(size=NUM_TASKS)
            new_pos = np.clip(
                self.population[i] + r * (self.best_position - self.population[i]) +
                self.alpha * (1 - current_iter/MAX_ITER) * 
                (self.population[np.random.randint(POP_SIZE)] - 
                 self.population[np.random.randint(POP_SIZE)]),
                0, NUM_EDGE_NODES-1
            ).astype(int)
            
            new_fitness = self._compute_fitness(new_pos)
            if new_fitness < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_position = new_pos.copy()

    def _escape_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Escape from predators
            if np.random.random() < self.PP:
                r = np.random.random(size=NUM_TASKS)
                new_pos = np.clip(
                    self.population[i] + r * (self.population[i] - self.best_position) +
                    (1 - current_iter/MAX_ITER) * np.random.random(size=NUM_TASKS) *
                    (self.population[np.random.randint(POP_SIZE)] - 
                     self.population[np.random.randint(POP_SIZE)]),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
                
                new_fitness = self._compute_fitness(new_pos)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_pos.copy()

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            # Alternate between attack and escape
            if iter % 2 == 0:
                self._attack_phase(iter)
            else:
                self._escape_phase(iter)
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve

