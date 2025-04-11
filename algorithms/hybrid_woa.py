import numpy as np
from config import ALPHA, BANDWIDTH, GAMMA, NUM_EDGE_NODES, NUM_TASKS, POP_SIZE, MAX_ITER

class HybridWOA:
    def __init__(self, tasks, edge_nodes):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.population = np.random.randint(0, NUM_EDGE_NODES, size=(POP_SIZE, NUM_TASKS))
        self.fitness = np.zeros(POP_SIZE)
        self.best_position = None
        self.best_fitness = float('inf')
        self.b = 1  # Spiral shape constant
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

    def _encircle_prey(self, a, A, C, leader_pos):
        D = np.abs(C * leader_pos - self.population)
        return np.clip(leader_pos - A * D, 0, NUM_EDGE_NODES-1).astype(int)

    def _bubble_net_attack(self, a, current_pos):
        l = np.random.uniform(-1, 1, size=(POP_SIZE, NUM_TASKS))
        D_prime = np.abs(self.best_position - current_pos)
        return np.clip(D_prime * np.exp(self.b * l) * np.cos(2*np.pi*l) + self.best_position, 
                      0, NUM_EDGE_NODES-1).astype(int)

    def _search_prey(self, a, A):
        rand_leader = self.population[np.random.randint(POP_SIZE)]
        D = np.abs(A * rand_leader - self.population)
        return np.clip(rand_leader - A * D, 0, NUM_EDGE_NODES-1).astype(int)

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            a = 2 - iter * (2 / MAX_ITER)
            a2 = -1 + iter * (-1 / MAX_ITER)
            
            for i in range(POP_SIZE):
                A = 2 * a * np.random.random() - a
                C = 2 * np.random.random()
                p = np.random.random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        self.population[i] = self._encircle_prey(a, A, C, self.best_position)[i]
                    else:
                        self.population[i] = self._search_prey(a, A)[i]
                else:
                    self.population[i] = self._bubble_net_attack(a, self.population[i])[i]
                
                new_fitness = self._compute_fitness(self.population[i])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = self.population[i].copy()
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve