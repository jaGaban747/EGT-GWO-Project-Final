import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDPSO(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.velocity = np.random.uniform(-1, 1, (POP_SIZE, NUM_TASKS))
        self.pbest = np.copy(self.population)
        self.pbest_fitness = np.full(POP_SIZE, np.inf)
        self.gbest = None
        self.gbest_fitness = np.inf
    
    def optimize(self):
        self._initialize_population()
        convergence = []
        
        for iter in range(MAX_ITER):
            # Evaluate fitness
            for i in range(POP_SIZE):
                fitness = self._compute_fitness(self.population[i])
                if fitness < self.pbest_fitness[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_fitness[i] = fitness
                if fitness < self.gbest_fitness:
                    self.gbest = np.copy(self.population[i])
                    self.gbest_fitness = fitness
            
            # Update velocity and position
            for i in range(POP_SIZE):
                r1, r2 = np.random.rand(2)
                cognitive = 1.5 * r1 * (self.pbest[i] - self.population[i])
                social = 1.5 * r2 * (self.gbest - self.population[i])
                self.velocity[i] = 0.7 * self.velocity[i] + cognitive + social
                self.population[i] = np.clip(
                    np.round(self.population[i] + self.velocity[i]), 
                    0, NUM_EDGE_NODES-1
                ).astype(int)
            
            convergence.append(self.gbest_fitness)
        
        return self.gbest, convergence
    
    def _compute_fitness(self, solution):
        """Fitness with strategy probability penalty."""
        utility = sum(self._compute_utility(i, node_idx) for i, node_idx in enumerate(solution))
        penalty = -np.sum(np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10))
        return utility + penalty