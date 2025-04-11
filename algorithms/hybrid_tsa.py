import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridTSA:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        # Add upper and lower bounds
        self.ub = NUM_EDGE_NODES - 1
        self.lb = 0
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

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            A1 = 2 - (2 * iter / MAX_ITER)
            A2 = 1 + (iter / MAX_ITER)
            
            for i in range(POP_SIZE):
                if np.random.random() < 0.5:  # Jet propulsion
                    new_pos = np.clip(
                        self.best_position + A1 * (self.ub - self.lb) * np.random.random(size=NUM_TASKS),
                        self.lb, self.ub
                    ).astype(int)
                else:  # Swarm behavior
                    dist = np.abs(self.best_position - self.population[i])
                    new_pos = np.clip(
                        self.best_position + A2 * dist * np.random.random(size=NUM_TASKS),
                        self.lb, self.ub
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