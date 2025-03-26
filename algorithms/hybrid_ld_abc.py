import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDABC(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.limit = 100  # Abandonment limit
        self.trials = np.zeros(POP_SIZE)

    def _compute_fitness(self, solution):
        utility = sum(self._compute_utility(i, node_idx) 
                   for i, node_idx in enumerate(solution))
        penalty = -np.sum(np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10))
        return utility + penalty

    def _generate_neighbor(self, current, partner):
        phi = np.random.uniform(-1, 1, NUM_TASKS)
        new_pos = np.clip(
            (current + phi * (current - partner)).astype(int),
            0, NUM_EDGE_NODES-1
        )
        
        # Apply Logit Dynamics to some dimensions
        for j in range(NUM_TASKS):
            if np.random.rand() < 0.1:
                new_pos[j] = np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
        
        return new_pos

    def optimize(self):
        self._initialize_population()
        fitness = np.array([self._compute_fitness(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        best_solution = np.copy(self.population[best_idx])
        best_fitness = fitness[best_idx]
        convergence = [best_fitness]
        
        for _ in range(MAX_ITER):
            # Employed bees phase
            for i in range(POP_SIZE):
                partner = np.random.choice([x for x in range(POP_SIZE) if x != i])
                new_solution = self._generate_neighbor(self.population[i], self.population[partner])
                new_fitness = self._compute_fitness(new_solution)
                
                if new_fitness < fitness[i]:
                    self.population[i] = new_solution
                    fitness[i] = new_fitness
                    self.trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                else:
                    self.trials[i] += 1
            
            # Onlooker bees phase
            probs = (1/(fitness + 1e-10)) / np.sum(1/(fitness + 1e-10))
            for _ in range(POP_SIZE):
                i = np.random.choice(POP_SIZE, p=probs)
                partner = np.random.choice([x for x in range(POP_SIZE) if x != i])
                new_solution = self._generate_neighbor(self.population[i], self.population[partner])
                new_fitness = self._compute_fitness(new_solution)
                
                if new_fitness < fitness[i]:
                    self.population[i] = new_solution
                    fitness[i] = new_fitness
                    self.trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                else:
                    self.trials[i] += 1
            
            # Scout bees phase
            for i in range(POP_SIZE):
                if self.trials[i] >= self.limit:
                    self.population[i] = np.array([
                        np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
                        for j in range(NUM_TASKS)
                    ])
                    fitness[i] = self._compute_fitness(self.population[i])
                    self.trials[i] = 0
            
            convergence.append(best_fitness)
        
        return best_solution, convergence