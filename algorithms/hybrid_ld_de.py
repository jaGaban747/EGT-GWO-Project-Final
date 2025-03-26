import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDDE(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.F = 0.8  # Differential weight
        self.CR = 0.9 # Crossover probability

    def _compute_fitness(self, solution):
        utility = sum(self._compute_utility(i, node_idx) 
                   for i, node_idx in enumerate(solution))
        penalty = -np.sum(np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10))
        return utility + penalty

    def optimize(self):
        self._initialize_population()
        convergence = []
        
        for _ in range(MAX_ITER):
            for i in range(POP_SIZE):
                # Mutation
                idxs = [idx for idx in range(POP_SIZE) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(
                    self.population[a] + self.F * (self.population[b] - self.population[c]),
                    0, NUM_EDGE_NODES-1
                ).astype(int)
                
                # Crossover
                cross_points = np.random.rand(NUM_TASKS) < self.CR
                trial = np.where(cross_points, mutant, self.population[i])
                
                # Selection
                trial_fitness = self._compute_fitness(trial)
                current_fitness = self._compute_fitness(self.population[i])
                if trial_fitness < current_fitness:
                    self.population[i] = trial
            
            best_fitness = min(self._compute_fitness(ind) for ind in self.population)
            convergence.append(best_fitness)
        
        best_idx = np.argmin([self._compute_fitness(ind) for ind in self.population])
        return self.population[best_idx], convergence