import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridAVO:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self.second_best_position = None
        self.second_best_fitness = float('inf')
        self._initialize_population()

    def _initialize_population(self):
        for i in range(POP_SIZE):
            self.fitness[i] = self._compute_fitness(self.population[i])
            if self.fitness[i] < self.best_fitness:
                # Initialize both best and second best on first iteration
                self.second_best_fitness = self.best_fitness
                # Only copy if not None
                if self.best_position is not None:
                    self.second_best_position = self.best_position.copy()
                else:
                    self.second_best_position = self.population[i].copy()
                    
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()
            elif self.fitness[i] < self.second_best_fitness:
                self.second_best_fitness = self.fitness[i]
                self.second_best_position = self.population[i].copy()

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

    def _calculate_P(self, iter):
        w = np.random.uniform(-2, 2)
        alpha = np.random.uniform(0, 1)
        return (1 - iter/MAX_ITER) ** (w * alpha)

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            P1 = self._calculate_P(iter)
            P2 = self._calculate_P(iter)
            P3 = self._calculate_P(iter)
            
            for i in range(POP_SIZE):
                rand_vulture = np.random.choice([0, 1])  # 0: best, 1: second best
                selected_vulture = self.best_position if rand_vulture == 0 else self.second_best_position
                
                F = P1 * (2 * np.random.random() - 1)
                if abs(F) >= 1:  # Exploration
                    Levy = np.random.standard_cauchy(size=NUM_TASKS)
                    new_pos = np.clip(
                        selected_vulture - np.abs(
                            (2 * np.random.random() * selected_vulture) - self.population[i]
                        ) * Levy,
                        0, NUM_EDGE_NODES-1
                    ).astype(int)
                else:  # Exploitation
                    if np.random.random() < P2:  # Phase 1
                        S = selected_vulture - (
                            (np.random.random() * self.population[i]) * 
                            (selected_vulture - self.population[i]) / (selected_vulture + 1e-10)
                        )
                        new_pos = np.clip(S, 0, NUM_EDGE_NODES-1).astype(int)
                    else:  # Phase 2
                        A = P3 * (2 * np.random.random() - 1)
                        if np.random.random() < 0.5:
                            new_pos = np.clip(
                                selected_vulture - np.abs(selected_vulture - self.population[i]) * A * 0.5,
                                0, NUM_EDGE_NODES-1
                            ).astype(int)
                        else:
                            Q = np.random.random(size=NUM_TASKS)
                            new_pos = np.clip(
                                selected_vulture - A * Q + np.random.random() * (
                                    self.best_position - self.second_best_position
                                ),
                                0, NUM_EDGE_NODES-1
                            ).astype(int)
                
                new_fitness = self._compute_fitness(new_pos)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.second_best_fitness = self.best_fitness
                        self.second_best_position = self.best_position.copy()
                        self.best_fitness = new_fitness
                        self.best_position = new_pos.copy()
                    elif new_fitness < self.second_best_fitness:
                        self.second_best_fitness = new_fitness
                        self.second_best_position = new_pos.copy()
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve