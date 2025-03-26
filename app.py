import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from algorithms import (
    HybridLDGWO, HybridRDGWO, HybridFPGWO, HybridBRDGWO,
    HybridLDPSO, HybridLDGA, HybridLDACO, HybridLDWOA, 
    HybridLDSA, HybridLDCS, HybridLDABC, HybridLDDE
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
    'LD-ACO': HybridLDACO(tasks, edge_nodes),
    'LD-WOA': HybridLDWOA(tasks, edge_nodes),
    'LD-SA': HybridLDSA(tasks, edge_nodes),
    'LD-CS': HybridLDCS(tasks, edge_nodes),
    'LD-ABC': HybridLDABC(tasks, edge_nodes),
    'LD-DE': HybridLDDE(tasks, edge_nodes)
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
        figure.savefig(os.path.join(results_dir, filename), bbox_inches='tight', dpi=300)
        plt.close(figure)
    except Exception as e:
        print(f"Warning: Could not save {filename}: {str(e)}")

def plot_group_results(results, group_name):
    # Set Seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl", len(results))
    
    metrics = ['response_time', 'offloading_ratio', 
               'resource_fairness', 'qos_differentiation', 
               'resource_utilization']
    
    fig, axs = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f'{group_name} Performance Comparison', fontsize=18, y=1.02)
    
    # Enhanced Convergence Plot
    axs[0,0].set_title('Algorithm Convergence', fontsize=14, pad=10)
    
    # Order algorithms with LD-GWO first
    algo_names = sorted(results.keys())
    if 'LD-GWO' in algo_names:
        algo_names.remove('LD-GWO')
        algo_names.insert(0, 'LD-GWO')
    
    # Plot convergence with Seaborn's color palette
    for i, name in enumerate(algo_names):
        linewidth = 3 if name == 'LD-GWO' else 2
        linestyle = '-' if name == 'LD-GWO' else '--'
        axs[0,0].plot(results[name]['convergence'], 
                     label=name,
                     linewidth=linewidth,
                     linestyle=linestyle)
    
    axs[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axs[0,0].set_xlabel('Iterations', fontsize=12)
    axs[0,0].set_ylabel('Fitness Value', fontsize=12)
    
    # Metric Plots
    metric_titles = {
        'response_time': 'Response Time (ms)',
        'offloading_ratio': 'Offloading Ratio',
        'resource_fairness': 'Resource Fairness',
        'qos_differentiation': 'QoS Differentiation',
        'resource_utilization': 'Resource Utilization'
    }
    
    for i, metric in enumerate(metrics):
        row = (i+1) // 3
        col = (i+1) % 3
        values = [r['metrics'][metric] for r in results.values()]
        
        if metric == 'qos_differentiation':
            colors = ['#e63946' if v > 0 else '#2a9d8f' for v in values]
            axs[row,col].bar(algo_names, values, color=colors)
        else:
            data = {'Algorithm': algo_names, 'Value': values}
            sns.barplot(data=data, x='Algorithm', y='Value', ax=axs[row,col], 
                       hue='Algorithm', palette='Blues', legend=False)
        
        axs[row,col].set_title(metric_titles[metric], fontsize=14, pad=10)
        axs[row,col].tick_params(axis='x', rotation=45)
    
    fig.delaxes(axs[1,2])
    plt.tight_layout()
    save_plot(fig, f'{group_name.lower()}_performance.png')

def plot_ldgwo_comparison(ldgwo_result, mh_results):
    """Special comparison showing LD-GWO vs each metaheuristic"""
    sns.set_style("whitegrid")
    metrics = ['latency', 'energy', 'resource_utilization']
    fig = plt.figure(figsize=(18, 6))
    
    comparisons = {
        'LD-GWO': ldgwo_result,
        **mh_results
    }
    
    for i, metric in enumerate(metrics, 1):
        ax = plt.subplot(1, 3, i)
        names = list(comparisons.keys())
        values = [r['metrics'][metric] for r in comparisons.values()]
        
        # Calculate p-values
        p_values = []
        for mh_name, mh_result in mh_results.items():
            _, p = mannwhitneyu(
                [ldgwo_result['metrics'][metric]],
                [mh_result['metrics'][metric]]
            )
            p_values.append(p)
        
        data = {'Algorithm': names, 'Value': values}
        palette = ['gold' if name == 'LD-GWO' else 'skyblue' for name in names]
        sns.barplot(data=data, x='Algorithm', y='Value', ax=ax,
                   hue='Algorithm', palette=palette, legend=False)
        
        # Annotate p-values
        for j, (name, p) in enumerate(zip(names[1:], p_values)):
            height = max(values[0], values[j+1]) * 1.05
            ax.text(j+1, height, f"p={p:.3f}", ha='center')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("LD-GWO vs Metaheuristics Comparison", y=1.05)
    plt.tight_layout()
    save_plot(fig, 'ldgwo_vs_metaheuristics.png')

def plot_cross_comparison(gt_res, mh_res):
    sns.set_style("whitegrid")
    key_metrics = ['latency', 'energy', 'resource_utilization']
    
    fig = plt.figure(figsize=(18, 6))
    for i, metric in enumerate(key_metrics, 1):
        ax = plt.subplot(1, 3, i)
        
        num_comparisons = min(len(gt_res), len(mh_res))
        positions = np.arange(num_comparisons)
        width = 0.35
        
        gt_vals = [r['metrics'][metric] for r in list(gt_res.values())[:num_comparisons]]
        mh_vals = [r['metrics'][metric] for r in list(mh_res.values())[:num_comparisons]]
        
        stat, p = mannwhitneyu(gt_vals, mh_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        
        data = pd.DataFrame({
            'Group': ['Game Theory']*num_comparisons + ['Metaheuristics']*num_comparisons,
            'Value': gt_vals + mh_vals,
            'Position': np.concatenate([positions - width/2, positions + width/2])
        })
        
        # Only create legend for the first subplot
        sns.barplot(data=data, x='Position', y='Value', ax=ax, 
                   hue='Group', palette=['skyblue', 'salmon'], 
                   width=width, legend=i==1)
        
        # If legend was created (on first subplot), handle it
        if i == 1:
            ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Remove legend for other subplots
            ax.get_legend().remove() if ax.get_legend() is not None else None
        
        ax.set_title(f"{metric.replace('_', ' ').title()}\n(p={p:.3f} {sig})")
        ax.set_xticks(positions)
        ax.set_xticklabels(list(gt_res.keys())[:num_comparisons])
    
    plt.tight_layout()
    save_plot(fig, 'cross_group_comparison.png')

# =====================================
# Statistical Analysis
# =====================================
def print_statistical_comparison(ldgwo_result, mh_results):
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

# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    # Run experiments
    gt_results = run_experiments(game_theory_algorithms, "Game Theory")
    mh_results = run_experiments(metaheuristic_algorithms, "Metaheuristics")
    
    # Combine results
    combined_results = {**gt_results, **mh_results}
    
    # Generate plots
    plot_group_results(combined_results, "All Algorithms")
    plot_ldgwo_comparison(gt_results['LD-GWO'], mh_results)
    plot_cross_comparison(gt_results, mh_results)
    
    # Statistical comparison
    print_statistical_comparison(gt_results['LD-GWO'], mh_results)