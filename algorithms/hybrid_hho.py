import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH

class HybridHHO:
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

    def _exploration_phase(self, rabbit_pos):
        q = np.random.random()
        if q < 0.5:
            rand_idx = np.random.randint(POP_SIZE)
            return np.clip(
                rabbit_pos - np.random.random() * abs(rabbit_pos - 2*np.random.random()*self.population[rand_idx]),
                0, NUM_EDGE_NODES-1
            ).astype(int)
        else:
            return np.clip(
                (rabbit_pos - np.mean(self.population, axis=0)) - 
                np.random.random(size=NUM_TASKS) * (NUM_EDGE_NODES-1),
                0, NUM_EDGE_NODES-1
            ).astype(int)

    def _exploitation_phase(self, rabbit_pos, E):
        J = 2 * (1 - np.random.random(size=NUM_TASKS))
        delta_pos = rabbit_pos - self.population
        
        if abs(E) >= 0.5:  # Soft besiege
            new_pos = np.clip(rabbit_pos - delta_pos * abs(J * rabbit_pos - self.population), 
                             0, NUM_EDGE_NODES-1)
        else:  # Hard besiege
            new_pos = np.clip(rabbit_pos - E * abs(delta_pos), 
                             0, NUM_EDGE_NODES-1)
        return new_pos.astype(int)

    def optimize(self):
        convergence_curve = []
        for iter in range(MAX_ITER):
            E0 = 2 * np.random.random() - 1
            E = 2 * E0 * (1 - (iter / MAX_ITER))
            
            for i in range(POP_SIZE):
                if abs(E) >= 1:
                    self.population[i] = self._exploration_phase(self.best_position)
                else:
                    self.population[i] = self._exploitation_phase(self.best_position, E)[i]
                
                new_fitness = self._compute_fitness(self.population[i])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = self.population[i].copy()
            
            convergence_curve.append(self.best_fitness)
        return self.best_position, convergence_curve