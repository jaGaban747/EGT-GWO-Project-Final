import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from algorithms import (
    HybridLDGWO, HybridRDGWO, HybridFPGWO, HybridBRDGWO,
    HybridLDPSO, HybridLDGA, HybridLDACO
)
from utils.metrics import compute_metrics
from config import *

# =====================================
# Directory Setup
# =====================================
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# =====================================
# Data Generation
# =====================================
np.random.seed(RANDOM_SEED)
tasks = [{
    'cpu': np.random.randint(*TASK_CPU_RANGE),
    'deadline': np.random.randint(*TASK_DEADLINE_RANGE),
    'data': np.random.randint(*TASK_DATA_RANGE),
    'loc': np.random.rand(2) * 100,
    'mission_critical': (i < NUM_TASKS * MISSION_CRITICAL_RATIO)
} for i in range(NUM_TASKS)]

edge_nodes = [{
    'cpu_cap': np.random.randint(*EDGE_CPU_CAP_RANGE),
    'loc': np.random.rand(2) * 100,
    'energy_cost': np.random.uniform(*EDGE_ENERGY_COST_RANGE)
} for _ in range(NUM_EDGE_NODES)]

# =====================================
# Algorithm Groups
# =====================================
game_theory_algorithms = {
    'LD-GWO': HybridLDGWO(tasks, edge_nodes),
    'RD-GWO': HybridRDGWO(tasks, edge_nodes),
    'FP-GWO': HybridFPGWO(tasks, edge_nodes),
    'BRD-GWO': HybridBRDGWO(tasks, edge_nodes)
}

metaheuristic_algorithms = {
    'LD-PSO': HybridLDPSO(tasks, edge_nodes),
    'LD-GA': HybridLDGA(tasks, edge_nodes),
    'LD-ACO': HybridLDACO(tasks, edge_nodes)
}

# =====================================
# Experiment Runner
# =====================================
def run_experiments(algorithms, group_name):
    results = {}
    print(f"\n=== Running {group_name} Algorithms ===")
    for name, algo in algorithms.items():
        try:
            solution, convergence = algo.optimize()
            metrics = compute_metrics(solution, tasks, edge_nodes)
            results[name] = {
                'solution': solution,
                'metrics': metrics,
                'convergence': convergence
            }
            print(f"{name}:")
            print(f"  Fitness: {ALPHA*metrics['latency'] + GAMMA*metrics['energy']:.2f}")
            print(f"  Response Time: {metrics['response_time']:.2f} ms")
            print(f"  Energy: {metrics['energy']:.2f} J")
        except Exception as e:
            print(f"Error running {name}: {str(e)}")
    return results

# =====================================
# Visualization Functions
# =====================================
def save_plot(figure, filename):
    try:
        figure.savefig(os.path.join(results_dir, filename))
    except Exception as e:
        print(f"Warning: Could not save {filename}: {str(e)}")

def plot_group_results(results, group_name):
    metrics = ['response_time', 'offloading_ratio', 
               'resource_fairness', 'qos_differentiation', 
               'resource_utilization']
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{group_name} Algorithms Performance', fontsize=16)
    
    # Convergence Plot
    axs[0,0].set_title('Convergence')
    for name, data in results.items():
        axs[0,0].plot(data['convergence'], label=name)
    axs[0,0].legend()
    axs[0,0].grid(True)
    
    # Metric Plots
    for i, metric in enumerate(metrics):
        row = (i+1) // 3
        col = (i+1) % 3
        values = [r['metrics'][metric] for r in results.values()]
        
        if metric == 'qos_differentiation':
            colors = ['red' if v > 0 else 'green' for v in values]
            axs[row,col].bar(results.keys(), values, color=colors)
        else:
            axs[row,col].bar(results.keys(), values)
            
        axs[row,col].set_title(metric.replace('_', ' ').title())
        axs[row,col].grid(True)
        plt.setp(axs[row,col].xaxis.get_majorticklabels(), rotation=45)
    
    fig.delaxes(axs[1,2])
    plt.tight_layout()
    save_plot(fig, f'{group_name.lower()}_performance.png')
    plt.show()

def plot_ldgwo_comparison(ldgwo_result, mh_results):
    """Special comparison showing LD-GWO vs each metaheuristic"""
    metrics = ['latency', 'energy', 'resource_utilization']
    fig = plt.figure(figsize=(15, 5))
    
    # Prepare data - LD-GWO vs each metaheuristic
    comparisons = {
        'LD-GWO': ldgwo_result,
        **mh_results
    }
    
    for i, metric in enumerate(metrics, 1):
        ax = plt.subplot(1, 3, i)
        names = list(comparisons.keys())
        values = [r['metrics'][metric] for r in comparisons.values()]
        
        # Calculate p-values (LD-GWO vs each)
        p_values = []
        for mh_name, mh_result in mh_results.items():
            _, p = mannwhitneyu(
                [ldgwo_result['metrics'][metric]],
                [mh_result['metrics'][metric]]
            )
            p_values.append(p)
        
        # Plot with LD-GWO highlighted
        colors = ['gold' if name == 'LD-GWO' else 'skyblue' for name in names]
        bars = ax.bar(names, values, color=colors)
        
        # Annotate with p-values
        for j, (name, p) in enumerate(zip(names[1:], p_values)):
            height = max(values[0], values[j+1]) * 1.05
            ax.text(j+1, height, f"p={p:.3f}", ha='center')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel(metric)
        ax.grid(True)
        plt.xticks(rotation=45)
    
    plt.suptitle("LD-GWO vs Metaheuristics Comparison", y=1.05)
    plt.tight_layout()
    save_plot(fig, 'ldgwo_vs_metaheuristics.png')
    plt.show()

def plot_cross_comparison(gt_res, mh_res):
    key_metrics = ['latency', 'energy', 'resource_utilization']
    
    fig = plt.figure(figsize=(15, 5))
    for i, metric in enumerate(key_metrics, 1):
        ax = plt.subplot(1, 3, i)
        
        # Use the smaller number of algorithms as reference
        num_comparisons = min(len(gt_res), len(mh_res))
        positions = np.arange(num_comparisons)
        width = 0.35
        
        gt_vals = [r['metrics'][metric] for r in list(gt_res.values())[:num_comparisons]]
        mh_vals = [r['metrics'][metric] for r in list(mh_res.values())[:num_comparisons]]
        
        stat, p = mannwhitneyu(gt_vals, mh_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        
        ax.bar(positions - width/2, gt_vals, width, label='Game Theory')
        ax.bar(positions + width/2, mh_vals, width, label='Metaheuristics')
        
        ax.set_title(f"{metric.replace('_', ' ').title()}\n(p={p:.3f} {sig})")
        ax.set_xticks(positions)
        ax.set_xticklabels(list(gt_res.keys())[:num_comparisons])
        ax.grid(True)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, 'cross_group_comparison.png')
    plt.show()

# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    # Run experiments
    gt_results = run_experiments(game_theory_algorithms, "Game Theory")
    mh_results = run_experiments(metaheuristic_algorithms, "Metaheuristics")
    
    # Extract LD-GWO specifically
    ldgwo_result = gt_results['LD-GWO']
    
    # Generate plots
    plot_group_results(gt_results, "Game Theory")
    plot_group_results(mh_results, "Metaheuristics")
    plot_ldgwo_comparison(ldgwo_result, mh_results)  # New focused comparison
    plot_cross_comparison(gt_results, mh_results)
    
    # Statistical comparison
    print("\n=== LD-GWO vs Metaheuristics Detailed Comparison ===")
    for name, result in mh_results.items():
        print(f"\nComparison: LD-GWO vs {name}")
        for metric in ['latency', 'energy', 'resource_utilization']:
            _, p = mannwhitneyu(
                [ldgwo_result['metrics'][metric]],
                [result['metrics'][metric]]
            )
            effect = ldgwo_result['metrics'][metric] - result['metrics'][metric]
            print(f"{metric.title():<20} p = {p:.4f} ({'*' if p < 0.05 else 'ns'})")
            print(f"Effect size: {effect:.2f} ({'LD-GWO better' if effect < 0 else 'Other better'})")