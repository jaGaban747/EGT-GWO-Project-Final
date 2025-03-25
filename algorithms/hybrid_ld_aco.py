import numpy as np
from .base_gt import BaseGameTheory
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE

class HybridLDACO(BaseGameTheory):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.pheromone = np.ones((NUM_TASKS, NUM_EDGE_NODES))
        self.evaporation = 0.5
        self.alpha = 1.0  # Pheromone weight
        self.beta = 2.0   # Heuristic weight
    
    def _compute_fitness(self, solution):
        """Calculate fitness using utility + strategy penalty"""
        utility = sum(
            self._compute_utility(i, node_idx) 
            for i, node_idx in enumerate(solution)
        )
        # Use a small epsilon to avoid log(0)
        penalty = -np.sum(
            np.log(self.strategy_probs[np.arange(NUM_TASKS), solution] + 1e-10)
        )
        return utility + penalty
    
    def optimize(self):
        best_solution = None
        best_fitness = np.inf
        convergence = []
        
        for _ in range(100):  # ACO iterations
            solutions = []
            
            for _ in range(POP_SIZE):
                solution = []
                
                for task_idx in range(NUM_TASKS):
                    # Improved probability calculation
                    # Ensure non-negative values and handle potential NaNs
                    pheromone = np.abs(np.nan_to_num(
                        self.pheromone[task_idx] ** self.alpha, 
                        nan=1.0, 
                        posinf=1.0, 
                        neginf=1.0
                    ))
                    
                    strategy = np.abs(np.nan_to_num(
                        self.strategy_probs[task_idx] ** self.beta, 
                        nan=1.0, 
                        posinf=1.0, 
                        neginf=1.0
                    ))
                    
                    # Element-wise multiplication
                    probs = pheromone * strategy
                    
                    # Normalize probabilities
                    probs_sum = np.sum(probs)
                    
                    # Fallback to uniform distribution if sum is not positive
                    if probs_sum <= 0:
                        probs = np.ones(NUM_EDGE_NODES) / NUM_EDGE_NODES
                    else:
                        probs = probs / probs_sum
                    
                    # Ensure probabilities are non-negative and sum to 1
                    probs = np.abs(probs)
                    probs = probs / np.sum(probs)
                    
                    # Robust random choice
                    solution.append(np.random.choice(NUM_EDGE_NODES, p=probs))
                
                solutions.append(solution)
            
            # Update pheromones
            self.pheromone *= self.evaporation
            
            for solution in solutions:
                fitness = self._compute_fitness(solution)
                
                # Update best solution
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution
                
                # Pheromone update
                for task_idx, node_idx in enumerate(solution):
                    self.pheromone[task_idx, node_idx] += 1 / (1 + fitness)
            
            convergence.append(best_fitness)
        
        return best_solution, convergence