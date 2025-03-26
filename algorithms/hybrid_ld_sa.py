import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDSA(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.min_temperature = 1e-3

    def _compute_fitness(self, solution):
        utility = sum(self._compute_utility(i, node_idx) 
                   for i, node_idx in enumerate(solution))
        penalty = -np.sum(np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10))
        return utility + penalty

    def _generate_neighbor(self, current_solution):
        neighbor = current_solution.copy()
        task_idx = np.random.randint(0, NUM_TASKS)
        neighbor[task_idx] = np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[task_idx])
        return neighbor

    def optimize(self):
        current_solution = np.array([np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[i]) 
                                    for i in range(NUM_TASKS)])
        current_fitness = self._compute_fitness(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        convergence = [best_fitness]

        while self.temperature > self.min_temperature:
            neighbor = self._generate_neighbor(current_solution)
            neighbor_fitness = self._compute_fitness(neighbor)
            
            if neighbor_fitness < current_fitness or \
               np.random.rand() < np.exp((current_fitness - neighbor_fitness)/self.temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if neighbor_fitness < best_fitness:
                    best_solution = neighbor
                    best_fitness = neighbor_fitness
            
            convergence.append(best_fitness)
            self.temperature *= self.cooling_rate

        return best_solution, convergence