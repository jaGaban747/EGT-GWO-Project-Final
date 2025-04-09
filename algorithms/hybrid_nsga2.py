import numpy as np
from config import NUM_TASKS, NUM_EDGE_NODES, POP_SIZE, MAX_ITER, ALPHA, GAMMA, BANDWIDTH
from .base_gwo import BaseGWO

class HybridNSGA2(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        self.fronts = []

    def optimize(self):
        self.convergence = []
        
        for generation in range(MAX_ITER):
            # Evaluate objectives
            objectives = np.array([self._compute_fitness(ind) for ind in self.population])
            
            # Non-dominated sorting
            self.fronts = self._non_dominated_sort(objectives)
            
            # Crowding distance
            for front in self.fronts:
                self._crowding_distance(front, objectives)
            
            # Track convergence (average of first front)
            if self.fronts:
                avg_fitness = np.mean([obj[0]+obj[1] for obj in objectives[self.fronts[0]]])
                self.convergence.append(avg_fitness)
            
            # Selection and reproduction
            self._selection()
            
        # Return best solution from first front
        best_idx = self.fronts[0][0]
        return self.population[best_idx], self.convergence

    def _non_dominated_sort(self, objectives):
        # Implementation of NSGA-II non-dominated sorting
        fronts = []
        remaining = set(range(len(objectives)))
        
        while remaining:
            current_front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if (objectives[j][0] <= objectives[i][0] and 
                        objectives[j][1] <= objectives[i][1] and
                        (objectives[j][0] < objectives[i][0] or 
                         objectives[j][1] < objectives[i][1])):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(i)
            
            fronts.append(current_front)
            remaining -= set(current_front)
        
        return fronts

    def _crowding_distance(self, front, objectives):
        # Calculate crowding distance for solutions in a front
        pass  # Implementation omitted for brevity

    def _selection(self):
        # Tournament selection based on front rank and crowding distance
        pass  # Implementation omitted for brevity

    def _compute_fitness(self, solution):
        latency, energy = 0, 0
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            latency += ALPHA * (proc_time + tx_time)
            energy += GAMMA * node['energy_cost'] * task['cpu']
        return (latency, energy)