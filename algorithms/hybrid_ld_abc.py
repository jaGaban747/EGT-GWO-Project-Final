import numpy as np
from .base_gt import BaseGameTheory
from config import *

class HybridLDABC(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.limit = 50  # Reduced abandonment limit for better exploration
        self.trials = np.zeros(POP_SIZE)
        self.best_fitness_history = []

    def _normalize_utility(self, utility):
        """Normalize utility to prevent extreme values"""
        max_possible_utility = NUM_TASKS * MAX_UTILITY if hasattr(self, 'MAX_UTILITY') else NUM_TASKS * 100
        return utility / (max_possible_utility + 1e-10)

    def _compute_fitness(self, solution):
        """Improved fitness function with balanced utility and penalty"""
        # Calculate utility
        utility = sum(self._compute_utility(i, node_idx) 
                   for i, node_idx in enumerate(solution))
        utility = self._normalize_utility(utility)
        
        # Calculate penalty (with safeguards)
        valid_probs = np.clip(self.strategy_probs[np.arange(NUM_TASKS), solution], 1e-10, 1.0)
        penalty = -np.sum(np.log(valid_probs))
        penalty = penalty / NUM_TASKS  # Normalize penalty
        
        # Weighted sum (adjust weights as needed)
        return 0.7 * utility + 0.3 * penalty

    def _generate_neighbor(self, current, partner):
        """Generate neighbor solution with controlled exploration"""
        phi = np.random.uniform(-0.5, 0.5, NUM_TASKS)  # Reduced phi range for stability
        new_pos = np.clip(
            (current + phi * (current - partner)).astype(int),
            0, NUM_EDGE_NODES-1
        )
        
        # Apply Logit Dynamics with adaptive probability
        ld_prob = 0.2 * (1 - (len(self.best_fitness_history)/MAX_ITER) ) # Decreases over time
        for j in range(NUM_TASKS):
            if np.random.rand() < ld_prob:
                new_pos[j] = np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
        
        return new_pos

    def optimize(self):
        """Main optimization loop with enhanced convergence tracking"""
        self._initialize_population()
        fitness = np.array([self._compute_fitness(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        best_solution = np.copy(self.population[best_idx])
        best_fitness = fitness[best_idx]
        self.best_fitness_history = [best_fitness]
        
        for iteration in range(MAX_ITER):
            # Employed bees phase
            for i in range(POP_SIZE):
                partner_idx = np.random.choice([x for x in range(POP_SIZE) if x != i])
                new_solution = self._generate_neighbor(self.population[i], self.population[partner_idx])
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
            
            # Onlooker bees phase (fitness-proportionate selection)
            fitness_probs = np.clip(1.0 / (fitness - np.min(fitness) + 1e-10), 0, 1)
            fitness_probs /= np.sum(fitness_probs)
            
            for _ in range(POP_SIZE):
                i = np.random.choice(POP_SIZE, p=fitness_probs)
                partner_idx = np.random.choice([x for x in range(POP_SIZE) if x != i])
                new_solution = self._generate_neighbor(self.population[i], self.population[partner_idx])
                new_fitness = self._compute_fitness(new_solution)
                
                if new_fitness < fitness[i]:
                    self.population[i] = new_solution
                    fitness[i] = new_fitness
                    self.trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
            
            # Scout bees phase (adaptive abandonment)
            avg_fitness = np.mean(fitness)
            for i in range(POP_SIZE):
                if self.trials[i] >= self.limit and fitness[i] > avg_fitness:
                    self.population[i] = np.array([
                        np.random.choice(NUM_EDGE_NODES, p=self.strategy_probs[j])
                        for j in range(NUM_TASKS)
                    ])
                    fitness[i] = self._compute_fitness(self.population[i])
                    self.trials[i] = 0
            
            # Convergence tracking
            self.best_fitness_history.append(best_fitness)
            
            # Early stopping if converged
            if len(self.best_fitness_history) > 20:
                if np.std(self.best_fitness_history[-20:]) < 1e-6:
                    break
        
        return best_solution, self.best_fitness_history