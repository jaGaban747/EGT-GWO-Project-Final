import numpy as np
from config import MAX_ITER, NUM_TASKS, NUM_EDGE_NODES, ALPHA, GAMMA, BANDWIDTH, EPSILON, LEARNING_RATE, DISCOUNT_FACTOR
from .base_gwo import BaseGWO

class HybridQlearning(BaseGWO):
    def __init__(self, tasks, edge_nodes):
        super().__init__(tasks, edge_nodes)
        # Initialize Q-table: states are tasks, actions are edge nodes
        self.q_table = np.zeros((NUM_TASKS, NUM_EDGE_NODES))
        self.epsilon = EPSILON  # Exploration rate
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR

    def optimize(self):
        self.convergence = []
        best_solution = None
        best_fitness = float('inf')
        
        for episode in range(MAX_ITER):
            # Generate solution using Q-learning policy
            solution = self._generate_solution()
            
            # Calculate fitness (single scalar value)
            current_fitness = self._compute_fitness(solution)
            self.convergence.append(current_fitness)
            
            # Update best solution
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = solution.copy()
            
            # Update Q-values based on the solution's performance
            self._update_q_values(solution, current_fitness)
            
            # Decay exploration rate
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return best_solution, self.convergence

    def _generate_solution(self):
        solution = np.zeros(NUM_TASKS, dtype=int)
        for task_idx in range(NUM_TASKS):
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Explore: random action
                solution[task_idx] = np.random.randint(0, NUM_EDGE_NODES)
            else:
                # Exploit: best known action
                solution[task_idx] = np.argmax(self.q_table[task_idx])
        return solution

    def _update_q_values(self, solution, fitness):
        # Convert fitness to reward (higher is better)
        reward = 1 / (1 + fitness)  # Simple reward scaling
        
        for task_idx, node_idx in enumerate(solution):
            current_q = self.q_table[task_idx, node_idx]
            
            # Q-learning update rule
            max_future_q = np.max(self.q_table[task_idx])
            new_q = (1 - self.learning_rate) * current_q + \
                    self.learning_rate * (reward + self.discount_factor * max_future_q)
            
            self.q_table[task_idx, node_idx] = new_q

    def _compute_fitness(self, solution):
        """
        Compute combined fitness metric (lower is better)
        Returns a SINGLE scalar value for convergence tracking
        """
        total_latency = 0
        total_energy = 0
        
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            
            # Calculate processing time
            proc_time = task['cpu'] / node['cpu_cap']
            
            # Calculate transmission time
            distance = np.linalg.norm(node['loc'] - task['loc'])
            tx_time = (task['data'] / BANDWIDTH) * distance
            
            # Accumulate totals
            total_latency += proc_time + tx_time
            total_energy += node['energy_cost'] * task['cpu']
        
        # Return weighted sum as a single value
        return ALPHA * total_latency + GAMMA * total_energy