import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridSHO:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self._initialize_population()
        self.beta = 0.05  # Step size coefficient
        self.theta = 0.1  # Spiral coefficient

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

    def _movement_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Random movement or following best
            if np.random.random() < 0.5:
                # Random movement
                r1 = np.random.random(size=NUM_TASKS)
                new_pos = np.clip(
                    self.population[i] + self.beta * (1 - current_iter/MAX_ITER) * 
                    (NUM_EDGE_NODES-1) * (2 * r1 - 1),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
            else:
                # Movement toward best
                r2 = np.random.random(size=NUM_TASKS)
                new_pos = np.clip(
                    self.best_position + self.theta * r2 * 
                    (self.population[i] - self.best_position),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
            
            new_fitness = self._compute_fitness(new_pos)
            if new_fitness < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_position = new_pos.copy()

    def _predation_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Spiral movement for predation
            l = np.random.uniform(-1, 1, size=NUM_TASKS)
            new_pos = np.clip(
                self.best_position + (self.population[i] - self.best_position) * 
                np.exp(self.theta * l) * np.cos(2 * np.pi * l),
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
            self._movement_phase(iter)
            self._predation_phase(iter)
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve