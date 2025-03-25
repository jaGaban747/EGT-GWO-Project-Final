import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDGA(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.mutation_rate = 0.1
    
    def _compute_fitness(self, solution):
        """Calculate fitness for a solution using utility + strategy penalty."""
        utility = sum(
            self._compute_utility(i, node_idx) 
            for i, node_idx in enumerate(solution)
        )
        penalty = -np.sum(
            np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10)
        )
        return utility + penalty

    def optimize(self):
        self._initialize_population()
        convergence = []
        
        for _ in range(MAX_ITER):
            fitness = np.array([self._compute_fitness(ind) for ind in self.population])
            parents = self._select_parents(fitness)
            offspring = self._crossover(parents)
            self.population = self._mutate(offspring)
            convergence.append(np.min(fitness))
        
        best_idx = np.argmin([self._compute_fitness(ind) for ind in self.population])
        return self.population[best_idx], convergence

    def _select_parents(self, fitness, tournament_size=3):
        selected = []
        for _ in range(POP_SIZE):
            candidates = np.random.choice(POP_SIZE, tournament_size)
            winner = candidates[np.argmin(fitness[candidates])]
            selected.append(self.population[winner])
        return np.array(selected)
    
    def _crossover(self, parents):
        offspring = np.zeros_like(parents)
        for i in range(0, POP_SIZE, 2):
            if i+1 >= POP_SIZE: break
            crossover_point = np.random.randint(1, NUM_TASKS)
            offspring[i] = np.concatenate([parents[i][:crossover_point], parents[i+1][crossover_point:]])
            offspring[i+1] = np.concatenate([parents[i+1][:crossover_point], parents[i][crossover_point:]])
        return offspring
    
    def _mutate(self, offspring):
        for i in range(POP_SIZE):
            for j in range(NUM_TASKS):
                if np.random.rand() < self.mutation_rate:
                    offspring[i, j] = np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
        return offspring