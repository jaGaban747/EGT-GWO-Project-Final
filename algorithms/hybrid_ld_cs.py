import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDCS(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.pa = 0.25  # Discovery probability
        self.alpha = 1.0  # Step size
        self.lambda_ = 1.5  # Levy exponent

    def _compute_fitness(self, solution):
        utility = sum(self._compute_utility(i, node_idx) 
                   for i, node_idx in enumerate(solution))
        penalty = -np.sum(np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10))
        return utility + penalty

    def _levy_flight(self, size):
        beta = self.lambda_
        sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        return u/(np.abs(v)**(1/beta))

    def optimize(self):
        self._initialize_population()
        convergence = []
        best_idx = np.argmin([self._compute_fitness(ind) for ind in self.population])
        best_solution = np.copy(self.population[best_idx])
        best_fitness = self._compute_fitness(best_solution)
        
        for _ in range(MAX_ITER):
            # Generate new solutions via Levy flights
            for i in range(POP_SIZE):
                step = self.alpha * self._levy_flight(NUM_TASKS)
                new_solution = np.clip(
                    (self.population[i] + step).astype(int),
                    0, NUM_EDGE_NODES-1
                )
                
                # Apply Logit Dynamics with probability
                for j in range(NUM_TASKS):
                    if np.random.rand() < 0.1:  # 10% chance to use LD
                        new_solution[j] = np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
                
                new_fitness = self._compute_fitness(new_solution)
                if new_fitness < self._compute_fitness(self.population[i]):
                    self.population[i] = new_solution
            
            # Discovery and replacement
            for i in range(POP_SIZE):
                if np.random.rand() < self.pa:
                    self.population[i] = np.array([
                        np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
                        for j in range(NUM_TASKS)
                    ])
            
            # Update best solution
            current_fitness = [self._compute_fitness(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitness)
            if current_fitness[current_best_idx] < best_fitness:
                best_solution = np.copy(self.population[current_best_idx])
                best_fitness = current_fitness[current_best_idx]
            
            convergence.append(best_fitness)
        
        return best_solution, convergence