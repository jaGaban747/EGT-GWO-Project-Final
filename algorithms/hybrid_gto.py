import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridGTO:
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

    def _exploration_phase(self, current_iter):
        a = (np.cos(2 * np.random.random()) + 1) * (1 - current_iter/MAX_ITER)
        C = (2 - 2 * current_iter/MAX_ITER) * np.random.random()
        
        for i in range(POP_SIZE):
            if np.random.random() < 0.5:
                # Migration to unknown place
                new_pos = np.clip(
                    (NUM_EDGE_NODES-1) * np.random.random(size=NUM_TASKS) * a -
                    C * self.population[i] + np.random.random() * self.best_position,
                    0, NUM_EDGE_NODES-1
                ).astype(int)
            else:
                # Migration to known place
                new_pos = np.clip(
                    self.best_position - C * (self.best_position - self.population[i]) +
                    np.random.random() * (self.population[np.random.randint(POP_SIZE)] - 
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

    def _exploitation_phase(self, current_iter):
        a = (np.cos(2 * np.random.random()) + 1) * (1 - current_iter/MAX_ITER)
        C = (2 - 2 * current_iter/MAX_ITER) * np.random.random()
        
        for i in range(POP_SIZE):
            if np.random.random() < 0.5:
                # Follow the silverback
                new_pos = np.clip(
                    C * self.best_position - self.population[i] +
                    np.random.random() * (self.population[i] - self.best_position),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
            else:
                # Competition for females
                new_pos = np.clip(
                    self.best_position - (self.best_position * a - self.population[i] * a) *
                    np.random.random(size=NUM_TASKS),
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
            if iter < MAX_ITER//2:
                self._exploration_phase(iter)
            else:
                self._exploitation_phase(iter)
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve