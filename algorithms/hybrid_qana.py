import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridQANA:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self.quantum_rotations = np.random.uniform(0, 2*np.pi, size=(POP_SIZE, NUM_TASKS))
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

    def _quantum_measure(self, angle):
        prob = (np.cos(angle/2))**2
        return 1 if np.random.random() < prob else 0

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            for i in range(POP_SIZE):
                # Quantum rotation
                delta_theta = np.pi * (1 - iter/MAX_ITER) * np.random.random(size=NUM_TASKS)
                self.quantum_rotations[i] = (self.quantum_rotations[i] + delta_theta) % (2*np.pi)
                
                # Measurement
                measured = np.array([self._quantum_measure(angle) for angle in self.quantum_rotations[i]])
                new_pos = np.clip(
                    self.population[i] + measured * (NUM_EDGE_NODES-1),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
                
                # Migration behavior
                if np.random.random() < 0.3:
                    new_pos = np.clip(
                        new_pos + np.round(np.random.normal(0, 1, size=NUM_TASKS)).astype(int),
                        0, NUM_EDGE_NODES-1
                    )
                
                new_fitness = self._compute_fitness(new_pos)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_pos.copy()
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve