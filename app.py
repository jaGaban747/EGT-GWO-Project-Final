import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import os
import sys

# Add parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms import (
    HybridLDGWO, HybridRDGWO, HybridFPGWO, HybridBRDGWO,
    HybridLDPSO, HybridLDGA, HybridLDACO, HybridLDWOA, 
    HybridLDSA, HybridLDCS, HybridLDABC, HybridLDDE
)
from utils.metrics import compute_metrics
from config import *

# Page setup
st.set_page_config(
    page_title="Edge Computing Algorithm Comparison",
    page_icon="📊",
    layout="wide"
)

# Helper functions
@st.cache_data
def generate_data(seed):
    """Generate task and edge node data with specified seed"""
    np.random.seed(seed)
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
    
    return tasks, edge_nodes

def run_algorithm(algo_class, tasks, edge_nodes):
    """Run a specific algorithm and return results"""
    try:
        algo = algo_class(tasks, edge_nodes)
        solution, convergence = algo.optimize()
        metrics = compute_metrics(solution, tasks, edge_nodes)
        return {
            'solution': solution,
            'metrics': metrics,
            'convergence': convergence
        }
    except Exception as e:
        st.error(f"Error running {algo_class.__name__}: {str(e)}")
        return None

def normalize_metrics(results):
    """Normalize metrics across all algorithms for fair comparison"""
    normalized_results = {}
    
    # Extract all metric values for normalization
    metric_values = {}
    for algo, metrics in results.items():
        for metric_name, values in metrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].extend(values)
    
    # Calculate min and max for each metric
    metric_ranges = {}
    for metric_name, values in metric_values.items():
        metric_ranges[metric_name] = {
            'min': min(values),
            'max': max(values)
        }
    
    # Normalize metrics for each algorithm
    for algo, metrics in results.items():
        normalized_results[algo] = {}
        for metric_name, values in metrics.items():
            min_val = metric_ranges[metric_name]['min']
            max_val = metric_ranges[metric_name]['max']
            
            # Skip normalization if min equals max (all values are the same)
            if min_val == max_val:
                normalized_results[algo][metric_name] = [1.0 for _ in values]
            else:
                # For metrics where lower is better (like latency, energy)
                # We invert the normalization so higher values are better
                normalized_results[algo][metric_name] = [
                    1 - ((val - min_val) / (max_val - min_val)) 
                    for val in values
                ]
    
    return normalized_results

# Initialize session state
if 'all_results' not in st.session_state:
    st.session_state.all_results = {}

if 'normalized_results' not in st.session_state:
    st.session_state.normalized_results = {}

if 'all_convergence' not in st.session_state:
    st.session_state.all_convergence = {}

if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = ['latency', 'energy', 'resource_utilization']

# Dashboard layout
st.title("Edge Computing Algorithm Comparison Dashboard")

# Sidebar options
st.sidebar.header("Settings")

# Algorithm selection
st.sidebar.subheader("Algorithms")
algo_exp = st.sidebar.expander("Select Algorithms", expanded=True)
algos = {
    'LD-GWO': algo_exp.checkbox('LD-GWO', value=True),
    'RD-GWO': algo_exp.checkbox('RD-GWO', value=True),
    'FP-GWO': algo_exp.checkbox('FP-GWO', value=True),
    'BRD-GWO': algo_exp.checkbox('BRD-GWO', value=True),
    'LD-PSO': algo_exp.checkbox('LD-PSO', value=True),
    'LD-GA': algo_exp.checkbox('LD-GA', value=True),
    'LD-ACO': algo_exp.checkbox('LD-ACO', value=True),
    'LD-WOA': algo_exp.checkbox('LD-WOA', value=True),
    'LD-SA': algo_exp.checkbox('LD-SA', value=True),
    'LD-CS': algo_exp.checkbox('LD-CS', value=True),
    'LD-ABC': algo_exp.checkbox('LD-ABC', value=True),
    'LD-DE': algo_exp.checkbox('LD-DE', value=True)
}

# Experiment settings
st.sidebar.subheader("Experiment Configuration")
num_trials = st.sidebar.number_input("Number of Trials", 1, 50, 30)
selected_metrics = st.sidebar.multiselect(
    "Metrics for Statistical Tests",
    options=['fitness', 'latency', 'energy', 'resource_utilization'],
    default=['fitness']
)

# Visualization options
st.sidebar.subheader("Visualization")
show_convergence = st.sidebar.checkbox("Show convergence plots", value=True)
show_metrics = st.sidebar.checkbox("Show performance metrics", value=True)
use_normalized = st.sidebar.checkbox("Use normalized metrics", value=True)

# Algorithm mapping
algo_mapping = {
    'LD-GWO': HybridLDGWO,
    'RD-GWO': HybridRDGWO,
    'FP-GWO': HybridFPGWO,
    'BRD-GWO': HybridBRDGWO,
    'LD-PSO': HybridLDPSO,
    'LD-GA': HybridLDGA,
    'LD-ACO': HybridLDACO,
    'LD-WOA': HybridLDWOA,
    'LD-SA': HybridLDSA,
    'LD-CS': HybridLDCS,
    'LD-ABC': HybridLDABC,
    'LD-DE': HybridLDDE
}

# Run experiments
if st.sidebar.button("Run Experiments"):
    selected_algos = {name: algo_mapping[name] for name, selected in algos.items() if selected}
    if len(selected_algos) < 2:
        st.error("Please select at least 2 algorithms for comparison")
        st.stop()
    
    all_results = {algo: {metric: [] for metric in selected_metrics} for algo in selected_algos}
    all_convergence = {algo: [] for algo in selected_algos}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for trial in range(num_trials):
        status_text.text(f"Running trial {trial+1}/{num_trials}...")
        tasks, edge_nodes = generate_data(RANDOM_SEED + trial)
        
        for algo_name, algo_class in selected_algos.items():
            result = run_algorithm(algo_class, tasks, edge_nodes)
            if result:
                metrics = result['metrics']
                metrics['fitness'] = ALPHA*metrics['latency'] + GAMMA*metrics['energy']
                
                # Store metrics
                for metric in selected_metrics:
                    all_results[algo_name][metric].append(metrics[metric])
                
                # Store convergence history
                all_convergence[algo_name].append(result['convergence'])
        
        progress_bar.progress((trial + 1) / num_trials)
    
    # Normalize results for fair comparison
    normalized_results = normalize_metrics(all_results)
    
    # Save results to session state
    st.session_state.all_results = all_results
    st.session_state.normalized_results = normalized_results
    st.session_state.all_convergence = all_convergence
    
    progress_bar.empty()
    status_text.text("Experiments completed!")

# Display results
if st.session_state.all_results:
    st.success(f"Analysis complete! ({num_trials} trials)")
    
    # Select which dataset to use (normalized or original)
    results_to_use = st.session_state.normalized_results if use_normalized else st.session_state.all_results
    
    tabs = ["Performance Summary", "Convergence Analysis", "Statistical Tests"]
    if show_metrics:
        tabs.insert(2, "Detailed Metrics")
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # Performance Summary Tab
    with tab1:
        st.header("Performance Summary")
        
        # Note about normalization
        if use_normalized:
            st.info("Metrics are normalized to [0-1] scale for fair comparison. Higher values indicate better performance.")
        
        # Calculate mean values
        mean_results = []
        for algo, metrics in results_to_use.items():
            mean_vals = {f"Mean {k}": np.mean(v) for k, v in metrics.items()}
            mean_vals['Algorithm'] = algo
            mean_results.append(mean_vals)
        
        perf_df = pd.DataFrame(mean_results)
        cols = ['Algorithm'] + [f"Mean {m}" for m in selected_metrics]
        
        if use_normalized:
            # For normalized values, higher is better
            st.dataframe(
                perf_df[cols].style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
        else:
            # For raw values, lower is better
            st.dataframe(
                perf_df[cols].style.highlight_min(axis=0, color='lightgreen'),
                use_container_width=True
            )
        
        # Box plots
        st.subheader("Distribution of Results")
        metric_to_show = st.selectbox("Select metric for distribution", selected_metrics)
        
        plot_data = []
        for algo, metrics in results_to_use.items():
            for value in metrics[metric_to_show]:
                plot_data.append({
                    'Algorithm': algo,
                    'Value': value,
                    'Metric': metric_to_show
                })
        
        df_plot = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_plot, x='Algorithm', y='Value', ax=ax)
        if use_normalized:
            ax.set_title(f'Distribution of Normalized {metric_to_show.capitalize()} Values (higher is better)')
            ax.set_ylim(0, 1)
        else:
            ax.set_title(f'Distribution of {metric_to_show.capitalize()} Values (lower is better)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    # Convergence Analysis Tab
    with tab2:
        st.header("Convergence Analysis")
        
        if show_convergence and st.session_state.all_convergence:
            # Select specific trial to show or average
            trial_options = ["Average"] + [f"Trial {i+1}" for i in range(num_trials)]
            selected_trial = st.selectbox("Select trial for convergence plot", trial_options)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for algo_name, convergence_list in st.session_state.all_convergence.items():
                if selected_trial == "Average":
                    # Calculate average convergence across all trials
                    # First, find the minimum length among all convergence histories
                    min_length = min(len(conv) for conv in convergence_list)
                    # Truncate all convergence histories to the minimum length
                    truncated = [conv[:min_length] for conv in convergence_list]
                    # Calculate average
                    avg_convergence = np.mean(truncated, axis=0)
                    
                    # Plot average convergence
                    ax.plot(range(1, len(avg_convergence) + 1), 
                            avg_convergence, 
                            label=f"{algo_name}")
                else:
                    # Plot specific trial
                    trial_idx = int(selected_trial.split(" ")[1]) - 1
                    if trial_idx < len(convergence_list):
                        ax.plot(range(1, len(convergence_list[trial_idx]) + 1), 
                                convergence_list[trial_idx], 
                                label=f"{algo_name}")
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness Value')
            ax.set_title('Convergence Curves')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Add convergence statistics
            st.subheader("Convergence Statistics")
            
            # Calculate iterations to reach various convergence thresholds
            convergence_stats = []
            for algo_name, convergence_list in st.session_state.all_convergence.items():
                # Calculate statistics for each algorithm
                stats = {}
                stats['Algorithm'] = algo_name
                
                # Find iterations to reach 90%, 95%, and 99% of final fitness
                iterations_to_90 = []
                iterations_to_95 = []
                iterations_to_99 = []
                
                for conv in convergence_list:
                    final_value = conv[-1]
                    
                    # For 90% convergence
                    threshold_90 = final_value * 1.1  # 10% more than final value
                    iter_90 = next((i for i, v in enumerate(conv) if v <= threshold_90), len(conv))
                    iterations_to_90.append(iter_90)
                    
                    # For 95% convergence
                    threshold_95 = final_value * 1.05  # 5% more than final value
                    iter_95 = next((i for i, v in enumerate(conv) if v <= threshold_95), len(conv))
                    iterations_to_95.append(iter_95)
                    
                    # For 99% convergence
                    threshold_99 = final_value * 1.01  # 1% more than final value
                    iter_99 = next((i for i, v in enumerate(conv) if v <= threshold_99), len(conv))
                    iterations_to_99.append(iter_99)
                
                stats['90% Convergence (iterations)'] = np.mean(iterations_to_90)
                stats['95% Convergence (iterations)'] = np.mean(iterations_to_95)
                stats['99% Convergence (iterations)'] = np.mean(iterations_to_99)
                stats['Final Fitness (avg)'] = np.mean([conv[-1] for conv in convergence_list])
                
                convergence_stats.append(stats)
            
            # Display convergence statistics
            st.dataframe(
                pd.DataFrame(convergence_stats).style.highlight_min(
                    ['90% Convergence (iterations)', '95% Convergence (iterations)', 
                     '99% Convergence (iterations)', 'Final Fitness (avg)'],
                    color='lightgreen'
                ),
                use_container_width=True
            )
    
    # Detailed Metrics Tab
    if show_metrics:
        with tab3:
            st.header("Detailed Metrics Analysis")
            
            # Allow user to choose metric
            metric_choice = st.selectbox("Select metric to analyze", selected_metrics)
            
            # Create radar chart for normalized metrics
            st.subheader("Algorithm Performance Comparison (Radar Chart)")
            
            # Get mean values for each algorithm and metric
            radar_data = {}
            for algo, metrics in st.session_state.normalized_results.items():
                radar_data[algo] = {}
                for metric, values in metrics.items():
                    radar_data[algo][metric] = np.mean(values)
            
            # Create radar chart
            categories = selected_metrics
            N = len(categories)
            
            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Add grid lines and category labels
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            
            # Draw y-axis lines
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
            plt.ylim(0, 1)
            
            # Plot each algorithm
            for algo, metrics in radar_data.items():
                values = [metrics[cat] for cat in categories]
                values += values[:1]  # Close the loop
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo)
                ax.fill(angles, values, alpha=0.1)
            
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            st.pyplot(fig)
            
            # Show detailed metrics for selected metric
            st.subheader(f"Detailed {metric_choice.capitalize()} Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart for mean values
                mean_vals = {algo: np.mean(metrics[metric_choice]) for algo, metrics in results_to_use.items()}
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(mean_vals.keys(), mean_vals.values())
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
                
                if use_normalized:
                    ax.set_title(f'Mean Normalized {metric_choice.capitalize()} (higher is better)')
                    ax.set_ylim(0, 1)
                else:
                    ax.set_title(f'Mean {metric_choice.capitalize()} (lower is better)')
                
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
            
            with col2:
                # Violin plot for distribution
                plot_data = []
                for algo, metrics in results_to_use.items():
                    for value in metrics[metric_choice]:
                        plot_data.append({
                            'Algorithm': algo,
                            'Value': value
                        })
                
                df_plot = pd.DataFrame(plot_data)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.violinplot(data=df_plot, x='Algorithm', y='Value', ax=ax)
                
                if use_normalized:
                    ax.set_title(f'Distribution of Normalized {metric_choice.capitalize()} (higher is better)')
                    ax.set_ylim(0, 1)
                else:
                    ax.set_title(f'Distribution of {metric_choice.capitalize()} (lower is better)')
                
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
    
    # Statistical Tests Tab
    with tab4:
        st.header("Statistical Analysis")
        
        if use_normalized:
            st.info("Statistical tests are being performed on normalized metrics for fair comparison.")
        
        selected_test_metric = st.selectbox("Select metric for statistical tests", selected_metrics)
        
        # Prepare data for tests
        algorithms = list(results_to_use.keys())
        data_matrix = np.array([results_to_use[algo][selected_test_metric] 
                              for algo in algorithms]).T
        
        # Friedman Test
        st.subheader("Friedman Test")
        friedman_stat, friedman_p = friedmanchisquare(*data_matrix.T)
        st.write(f"Friedman chi-square statistic: {friedman_stat:.3f}")
        st.write(f"p-value: {friedman_p:.5f}")
        if friedman_p < 0.05:
            st.success("Significant differences detected (p < 0.05)")
        else:
            st.warning("No significant differences detected (p >= 0.05)")
        
        # Wilcoxon Pairwise Tests with Holm correction
        if friedman_p < 0.05:
            st.subheader("Post-hoc Pairwise Comparisons (Wilcoxon signed-rank test)")
            
            # Generate all possible pairs
            pairs = [(i, j) for i in range(len(algorithms)) 
                    for j in range(i+1, len(algorithms))]
            
            results = []
            p_values = []
            for i, j in pairs:
                stat, p = wilcoxon(data_matrix[:, i], data_matrix[:, j])
                results.append({
                    'Algorithm 1': algorithms[i],
                    'Algorithm 2': algorithms[j],
                    'Statistic': stat,
                    'p-value': p
                })
                p_values.append(p)
            
            # Apply Holm-Bonferroni correction
            reject, p_adjusted, _, _ = multipletests(p_values, method='holm')
            
            # Add adjusted p-values to results
            for idx, res in enumerate(results):
                res['Adjusted p-value'] = p_adjusted[idx]
                res['Significant'] = reject[idx]
            
            results_df = pd.DataFrame(results)
            
            # Formatting
            def highlight_significant(row):
                color = 'lightgreen' if row['Significant'] else 'white'
                return ['background-color: {}'.format(color)] * len(row)
            
            st.dataframe(
                results_df.style.apply(highlight_significant, axis=1)
                .format({'p-value': '{:.5f}', 'Adjusted p-value': '{:.5f}'}),
                use_container_width=True
            )
            
            # Create ranking visualization
            if len(algorithms) > 2:
                st.subheader("Algorithm Ranking")
                
                # Calculate the number of wins for each algorithm
                wins = {algo: 0 for algo in algorithms}
                for result in results:
                    if result['Significant']:
                        algo1_mean = np.mean(results_to_use[result['Algorithm 1']][selected_test_metric])
                        algo2_mean = np.mean(results_to_use[result['Algorithm 2']][selected_test_metric])
                        
                        if use_normalized:
                            # For normalized results, higher is better
                            winner = result['Algorithm 1'] if algo1_mean > algo2_mean else result['Algorithm 2']
                        else:
                            # For raw results, lower is better
                            winner = result['Algorithm 1'] if algo1_mean < algo2_mean else result['Algorithm 2']
                        
                        wins[winner] += 1
                
                # Create ranking bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(wins.keys(), wins.values())
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                ax.set_title(f'Number of Significant Wins for {selected_test_metric.capitalize()}')
                ax.set_ylabel('Number of Wins')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)

else:
    st.info("Select algorithms and configuration in the sidebar, then click 'Run Experiments'")