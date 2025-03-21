import numpy as np
import matplotlib.pyplot as plt
from algorithms import HybridLDGWO, HybridRDGWO, HybridFPGWO, HybridBRDGWO
from utils.metrics import compute_metrics
from config import *

# Generate tasks and edge nodes (with mission-critical flag)
tasks = [
    {
        'cpu': np.random.randint(*TASK_CPU_RANGE), 
        'deadline': np.random.randint(*TASK_DEADLINE_RANGE), 
        'data': np.random.randint(*TASK_DATA_RANGE),
        'loc': np.random.rand(2) * 100,
        'mission_critical': (i < NUM_TASKS * MISSION_CRITICAL_RATIO)
    }
    for i in range(NUM_TASKS)
]

edge_nodes = [
    {
        'cpu_cap': np.random.randint(*EDGE_CPU_CAP_RANGE), 
        'loc': np.random.rand(2) * 100, 
        'energy_cost': np.random.uniform(*EDGE_ENERGY_COST_RANGE)
    }
    for _ in range(NUM_EDGE_NODES)
]

# Define all hybrid algorithms
algorithms = {
    'LD-GWO': HybridLDGWO(tasks, edge_nodes),
    'RD-GWO': HybridRDGWO(tasks, edge_nodes),
    'FP-GWO': HybridFPGWO(tasks, edge_nodes),
    'BRD-GWO': HybridBRDGWO(tasks, edge_nodes)
}

# Run comparisons
results = {}
for name, algo in algorithms.items():
    print(f"Running {name}...")
    solution, convergence = algo.optimize()
    metrics = compute_metrics(solution, tasks, edge_nodes, ALPHA, GAMMA, BANDWIDTH)
    results[name] = {'solution': solution, 'metrics': metrics, 'convergence': convergence}

# Plot convergence
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.plot(result['convergence'], label=name)
plt.title('Hybrid Algorithm Convergence')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot QoS differentiation
plt.figure(figsize=(12, 6))
qos_diffs = [result['metrics']['qos_differentiation'] for result in results.values()]
plt.bar(results.keys(), qos_diffs, color=['red' if diff > 0 else 'green' for diff in qos_diffs])
plt.title('QoS Differentiation (Lower is Better)')
plt.ylabel('Latency Difference (Mission-Critical - Normal)')
plt.grid(True)
plt.show()