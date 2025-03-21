import numpy as np

def compute_metrics(solution, tasks, edge_nodes, ALPHA, GAMMA, BANDWIDTH):
    latency = energy = overhead = 0
    node_loads = np.zeros(len(edge_nodes))
    mission_critical_latency = []
    normal_latency = []
    
    for task_idx, node_idx in enumerate(solution):
        task = tasks[task_idx]
        node = edge_nodes[node_idx]
        proc_time = task['cpu'] / node['cpu_cap']
        tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
        total_time = proc_time + tx_time
        
        # Track mission-critical vs normal tasks
        if task.get('mission_critical', False):
            mission_critical_latency.append(total_time)
        else:
            normal_latency.append(total_time)
        
        latency += total_time
        energy += node['energy_cost'] * task['cpu']
        overhead += 1
        node_loads[node_idx] += task['cpu']
        
    # Basic metrics
    throughput = len(tasks) / (latency + 1e-10)
    fairness = (np.sum(node_loads) ** 2) / (len(edge_nodes) * np.sum(node_loads ** 2) + 1e-10)
    
    # New metrics
    response_time = latency / len(tasks)
    offloading_ratio = len(np.unique(solution)) / len(edge_nodes)  # Diversity of nodes used
    
    # QoS differentiation (mission-critical vs normal)
    qos_diff = np.mean(mission_critical_latency) - np.mean(normal_latency) if mission_critical_latency else 0
    
    # Resource utilization fairness (Jain's index for CPU utilization)
    utilization = node_loads / [n['cpu_cap'] for n in edge_nodes]
    resource_fairness = (np.sum(utilization) ** 2) / (len(edge_nodes) * np.sum(utilization ** 2) + 1e-10)
    
    return {
        'throughput': throughput,
        'latency': latency / len(tasks),
        'energy': energy / len(tasks),
        'overhead': overhead,
        'fairness': fairness,
        'response_time': response_time,
        'offloading_ratio': offloading_ratio,
        'qos_differentiation': qos_diff,
        'resource_utilization': np.mean(utilization),
        'resource_fairness': resource_fairness
    }