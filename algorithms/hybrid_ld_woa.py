import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDWOA(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.a = 2  # Exploration parameter
        self.b = 1  # Spiral shape parameter

    def _compute_fitness(self, solution):
        utility = sum(self._compute_utility(i, node_idx) 
                   for i, node_idx in enumerate(solution))
        penalty = -np.sum(np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10))
        return utility + penalty

    def optimize(self):
        self._initialize_population()
        convergence = []
        best_idx = np.argmin([self._compute_fitness(ind) for ind in self.population])
        best_solution = np.copy(self.population[best_idx])
        best_fitness = self._compute_fitness(best_solution)
        
        for iter in range(MAX_ITER):
            self.a = 2 - iter * (2 / MAX_ITER)  # Decrease a linearly
            
            for i in range(POP_SIZE):
                r1, r2 = np.random.rand(2)
                A = 2 * self.a * r1 - self.a
                C = 2 * r2
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey
                        D = abs(C * best_solution - self.population[i])
                        self.population[i] = np.clip(
                            (best_solution - A * D).astype(int),
                            0, NUM_EDGE_NODES-1
                        )
                    else:
                        # Exploration
                        rand_idx = np.random.randint(0, POP_SIZE)
                        D = abs(C * self.population[rand_idx] - self.population[i])
                        self.population[i] = np.clip(
                            (self.population[rand_idx] - A * D).astype(int),
                            0, NUM_EDGE_NODES-1
                        )
                else:
                    # Spiral updating
                    D = abs(best_solution - self.population[i])
                    L = np.random.uniform(-1, 1, size=NUM_TASKS)
                    self.population[i] = np.clip(
                        (D * np.exp(self.b * L) * np.cos(2 * np.pi * L) + best_solution).astype(int),
                        0, NUM_EDGE_NODES-1
                    )
                
                # Apply strategy probabilities
                for j in range(NUM_TASKS):
                    if np.random.rand() < 0.1:  # 10% chance to use Logit Dynamics
                        self.population[i,j] = np.random.choice(
                            NUM_EDGE_NODES, 
                            p=self.strategy_probs[j]
                        )
            
            # Update best solution
            current_fitness = [self._compute_fitness(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitness)
            if current_fitness[current_best_idx] < best_fitness:
                best_solution = np.copy(self.population[current_best_idx])
                best_fitness = current_fitness[current_best_idx]
            
            convergence.append(best_fitness)
        
        return best_solution, convergence