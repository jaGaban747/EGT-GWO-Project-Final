import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridDO:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self._initialize_population()
        self.g = 1e-2  # Gravity coefficient
        self.beta = 0.2  # Randomness coefficient

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

    def _rising_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Rising with wind
            v = np.random.normal(0, 1, size=NUM_TASKS)
            y = np.random.random()
            t = (current_iter + 1) / MAX_ITER
            
            new_pos = np.clip(
                self.population[i] + y * v * (self.best_position - y * self.population[i]) *
                (1 - t) + self.g * (1 - t)**2,
                0, NUM_EDGE_NODES-1
            ).astype(int)
            
            new_fitness = self._compute_fitness(new_pos)
            if new_fitness < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_position = new_pos.copy()

    def _descending_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Descending and landing
            t = (current_iter + 1) / MAX_ITER
            r = np.random.random(size=NUM_TASKS)
            
            new_pos = np.clip(
                self.best_position * (1 - t) + self.beta * t * (2 * r - 1) * self.best_position,
                0, NUM_EDGE_NODES-1
            ).astype(int)
            
            new_fitness = self._compute_fitness(new_pos)
            if new_fitness < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_position = new_pos.copy()

    def _seeding_phase(self, current_iter):
        for i in range(POP_SIZE):
            # Seed spreading
            r = np.random.random(size=NUM_TASKS)
            new_pos = np.clip(
                self.best_position + 0.1 * (2 * r - 1) * (NUM_EDGE_NODES-1) * 
                (1 - current_iter/MAX_ITER),
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
            if iter < MAX_ITER//3:
                self._rising_phase(iter)
            elif iter < 2*MAX_ITER//3:
                self._descending_phase(iter)
            else:
                self._seeding_phase(iter)
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve