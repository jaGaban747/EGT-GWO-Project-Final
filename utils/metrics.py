# metrics.py (replace compute_metrics)
from config import (MAX_ITER, NUM_TASKS, NUM_EDGE_NODES, BANDWIDTH, ALPHA, GAMMA,
                    RELIABILITY_CRITICAL, RELIABILITY_NORMAL, NODE_FAILURE_PROB, 
                    PACKET_LOSS_RATE, ENERGY_SCALE, TASK_DEADLINE_MIN, TASK_DEADLINE_MAX,
                    BETA10, DISTANCE_SCALE_FACTOR)

import numpy as np

def compute_metrics(solution, tasks, edge_nodes, convergence=None):
    latency = energy = 0
    node_loads = np.zeros(len(edge_nodes))
    mission_critical_latency = []
    normal_latency = []
    reliable_tasks = completed_tasks = 0
    
    for task_idx, node_idx in enumerate(solution):
        task = tasks[task_idx]
        node = edge_nodes[node_idx]
        proc_time = task['cpu'] / node['cpu_cap']
        distance = np.linalg.norm(node['loc'] - task['loc']) / DISTANCE_SCALE_FACTOR
        tx_time = (task['data'] / BANDWIDTH) * distance
        total_time = proc_time + tx_time
        
        if task.get('mission_critical', False):
            mission_critical_latency.append(total_time)
        else:
            normal_latency.append(total_time)
        
        latency += total_time
        energy += node['energy_cost'] * task['cpu']
        node_loads[node_idx] += task['cpu']
        
        if total_time <= task['deadline']:
            completed_tasks += 1
        
        reliability_req = RELIABILITY_CRITICAL if task.get('mission_critical', False) else RELIABILITY_NORMAL
        node_reliability = 1 - NODE_FAILURE_PROB
        network_reliability = 1 - PACKET_LOSS_RATE
        effective_reliability = node_reliability * network_reliability
        weight = 2.0 if task.get('mission_critical', False) else 1.0
        if total_time <= task['deadline'] and effective_reliability >= reliability_req:
            reliable_tasks += weight * BETA10  # Weight critical tasks
    
    total_weight = sum(2.0 if task.get('mission_critical', False) else 1.0 for task in tasks)
    fitness = ALPHA * (latency / len(tasks)) + GAMMA * (energy / len(tasks))
    response_time = latency / len(tasks)
    qos_diff = (np.mean(mission_critical_latency) - np.mean(normal_latency) 
                if mission_critical_latency and normal_latency else 0)
    utilization = node_loads / [n['cpu_cap'] for n in edge_nodes]
    resource_utilization = np.mean(utilization)
    eace = (completed_tasks / len(tasks)) / (energy * ENERGY_SCALE + 1e-10)
    
    # Calculate Convergence Speed Index (CSI)
    csi = 1.0  # Default if no convergence data
    if convergence and len(convergence) > 1:
        initial_fitness = convergence[0]
        final_fitness = convergence[-1]
        improvement = initial_fitness - final_fitness  # Assuming minimization
        if improvement > 0:
            threshold = final_fitness + 0.05 * improvement  # 95% of improvement
            for i, fitness in enumerate(convergence):
                if fitness <= threshold:  # Reached 95% improvement
                    csi = (i + 1) / MAX_ITER  # Normalize by MAX_ITER
                    break

    return {
        'fitness': fitness,
        'latency': latency / len(tasks),
        'energy': energy / len(tasks),
        'response_time': response_time,
        'qos_differentiation': qos_diff,
        'resource_utilization': resource_utilization,
        'energy_aware_completion_efficiency': eace,
    }