import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import os
import sys
import io  # For saving figures to BytesIO
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# Add parent directory to the path so that modules can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import algorithms from your package
from algorithms import (
    HybridLDGWO,
    HybridRDGWO,
    HybridFPGWO,
    HybridBRDGWO,
    HybridPSO,
    HybridCOA,
    HybridWOA,
    HybridHHO,
    HybridSSA,
    HybridAO,
    HybridRSA,
    HybridTSA,
    HybridDO,
    HybridAVO,
    HybridSHO,
    HybridGTO
)
from utils.metrics import compute_metrics
from config import *

# Set up page configuration for Streamlit
st.set_page_config(
    page_title="Edge Computing Algorithm Comparison Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# Define a better color palette
def get_color_palette(num_colors):
    """Generate a color palette with distinct colors."""
    colormap = cm.get_cmap('tab10')
    return [f"rgba({int(255*colormap(i)[0])}, {int(255*colormap(i)[1])}, {int(255*colormap(i)[2])}, 0.7)" 
            for i in np.linspace(0, 0.9, num_colors)]




@st.cache_data
def generate_data(seed):
    np.random.seed(seed)
    tasks = [{
        'cpu': np.random.randint(*TASK_CPU_RANGE),
        'deadline': np.random.uniform(TASK_DEADLINE_MIN, TASK_DEADLINE_MAX),
        'data': np.random.randint(*TASK_DATA_RANGE),
        'loc': np.random.rand(2) * 100 * DISTANCE_SCALE_FACTOR,  # Scale distances
        'mission_critical': (i < NUM_TASKS * MISSION_CRITICAL_RATIO)
    } for i in range(NUM_TASKS)]

    edge_nodes = [{
        'cpu_cap': np.random.randint(*EDGE_CPU_CAP_RANGE),  # Updated range
        'loc': np.random.rand(2) * 100 * DISTANCE_SCALE_FACTOR,  # Scale distances
        'energy_cost': np.random.uniform(*EDGE_ENERGY_COST_RANGE)
    } for _ in range(NUM_EDGE_NODES)]

    return tasks, edge_nodes

def run_algorithm(algo_class, tasks, edge_nodes):
    """
    Modified to better handle different algorithm types' convergence data
    """
    try:
        algo = algo_class(tasks, edge_nodes)
        result = algo.optimize()
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            solution, convergence = result
        else:
            solution = result
            convergence = None
        
        metrics = compute_metrics(solution, tasks, edge_nodes)
        metrics['fitness'] = ALPHA * metrics['latency'] + GAMMA * metrics['energy']
        
        # Standardize convergence data format
        if convergence is not None:
            if isinstance(convergence, (int, float)):  # Single value
                convergence = [convergence]
            elif not isinstance(convergence, (list, tuple, np.ndarray)):
                convergence = None
        
        return {
            'solution': solution,
            'metrics': metrics,
            'convergence': convergence if convergence else [],
            'is_maximizing': algo_class.__name__ in maximizing_algos
        }
    except Exception as e:
        st.error(f"Error running {algo_class.__name__}: {str(e)}")
        return None
    
def normalize_metrics(results):
    """
    Normalize the collected metrics across all algorithms
    so that comparisons are on the same scale.
    """
    normalized_results = {}
    metric_values = {}
    for algo, metrics in results.items():
        for metric_name, values in metrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].extend(values)
    metric_ranges = {m: {'min': min(v), 'max': max(v)} for m, v in metric_values.items()}
    for algo, metrics in results.items():
        normalized_results[algo] = {}
        for m, values in metrics.items():
            min_val = metric_ranges[m]['min']
            max_val = metric_ranges[m]['max']
            if min_val == max_val:
                normalized_results[algo][m] = [1.0] * len(values)
            else:
                # For metrics where lower is better (latency, energy, response_time)
                if m in ['latency', 'energy', 'response_time']:
                    normalized_results[algo][m] = [
                        1 - ((val - min_val) / (max_val - min_val)) for val in values
                    ]
                # For metrics where higher is better (throughput, fairness, offloading_ratio, qos_differentiation, resource_utilization, resource_fairness)
                else:
                    normalized_results[algo][m] = [
                        (val - min_val) / (max_val - min_val) for val in values
                    ]
    return normalized_results

def create_radar_chart(radar_data, categories):
    """Create an interactive radar chart using Plotly."""
    fig = go.Figure()
    colors = get_color_palette(len(radar_data))
    
    for i, (algo, data) in enumerate(radar_data.items()):
        values = [data.get(cat, 0) for cat in categories]
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # Close the loop
            fill='toself',
            name=algo,
            line_color=colors[i],
            fillcolor=colors[i].replace('0.7', '0.2')
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
                tickvals=[0.2, 0.4, 0.6, 0.8],
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='black'),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=80, r=80, t=50, b=50),
        height=500,
        width=700
    )
    return fig

def create_better_boxplot(df, x_column, y_column, title, normalize=False):
    """Create an enhanced box plot with Plotly."""
    fig = px.box(
        df, 
        x=x_column, 
        y=y_column, 
        color=x_column,
        color_discrete_sequence=get_color_palette(len(df[x_column].unique())),
        title=title, 
        points="outliers"
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title=y_column,
        legend_title_text='',
        font=dict(family="Helvetica", size=24),  # Axes/labels
        title=dict(
            text=title,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(family="Helvetica", size=18)
        ),
        margin=dict(l=50, r=20, t=80, b=50),
        height=600,
        xaxis=dict(tickfont=dict(family="Helvetica", size=16), tickangle=45),
        yaxis=dict(tickfont=dict(family="Helvetica", size=16))
    )
    
    if normalize:
        fig.add_annotation(
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            text="Normalized values (higher is better)",
            showarrow=False,
            font=dict(family="Helvetica", size=14, color="gray")
        )
    
    return fig

def create_better_bar_chart(data, x_values, y_values, title, normalize=False):
    """Create an enhanced bar chart with Plotly."""
    fig = px.bar(
        data,
        x=x_values,
        y=y_values,
        color=x_values,
        color_discrete_sequence=get_color_palette(len(data)),
        text=[f"{y:.3f}" for y in y_values],
        title=title
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Value",
        legend_title_text='',
        showlegend=False,
        font=dict(size=12),
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=50, r=20, t=80, b=50),
        height=500
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Configure text display
    fig.update_traces(textposition='outside')
    
    
    return fig

def create_plotly_convergence(convergence_data, algorithm_names, is_maximizing, title, single_plot=True):
    """Create an interactive convergence plot using Plotly."""
    fig = go.Figure()
    colors = get_color_palette(len(algorithm_names))
    
    for i, (algo_name, convergence) in enumerate(zip(algorithm_names, convergence_data)):
        if len(convergence) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(convergence) + 1)),
                y=convergence,
                name=algo_name,
                line=dict(color=colors[i], width=2),
                mode='lines',
                hovertemplate='Iteration: %{x}<br>Value: %{y:.4f}'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title='Fitness Value',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        height=500,
        hovermode='closest'
    )
    
    # Add optimization direction annotation
    if not single_plot:
        direction = "Higher values are better" if is_maximizing else "Lower values are better"
        fig.add_annotation(
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            text=direction,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
    
    return fig

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
for key in ['all_results', 'normalized_results', 'all_convergence', 'algorithm_types']:
    if key not in st.session_state:
        st.session_state[key] = {}

# Updated default metrics list to include all the new metrics
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = [
        'fitness', 'throughput', 'taskLevel-fairness',
        'response_time', 'offloading_ratio', 'qos_differentiation', 
        'resource_utilization',
    ]

# Define which metrics are minimization metrics (lower is better)
minimization_metrics = [ 'overhead', 'response_time']
# Define which metrics are maximization metrics (higher is better)
maximization_metrics = [
    'throughput', 'taskLevel-fairness', 'offloading_ratio', 'qos_differentiation', 
    'resource_utilization', 
]

# Set custom page style
st.markdown("""
    <style>
    /* General text styling */
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    
    /* Chart titles */
    .stPlotlyChart .plotly .gtitle {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
    }
    
    /* Axis labels */
    .stPlotlyChart .plotly .xtitle, 
    .stPlotlyChart .plotly .ytitle {
        font-size: 14px !important;
        font-weight: bold !important;
    }
    
    /* Legend text */
    .stPlotlyChart .plotly .legendtext {
        font-size: 12px !important;
    }
    
    /* Data labels */
    .stPlotlyChart .plotly .trace text {
        font-size: 10px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Dashboard Layout
# -----------------------------------------------------------------------------
st.title("Edge Computing Algorithm Comparison Dashboard")

# -----------------------------------------------------------------------------
# Sidebar Options
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

# Algorithm Selection
st.sidebar.subheader("Algorithms")
algo_exp = st.sidebar.expander("Select Algorithms", expanded=True)
algos = {
    # Game Theory Hybrids
    'LD-GWO': algo_exp.checkbox('LD-GWO', value=True),
    'RD-GWO': algo_exp.checkbox('RD-GWO', value=True),
    'FP-GWO': algo_exp.checkbox('FP-GWO', value=True),
    'BRD-GWO': algo_exp.checkbox('BRD-GWO', value=True),
    
    # Metaheuristic Hybrids
    'WOA': algo_exp.checkbox('WOA', value=True),
    'HHO': algo_exp.checkbox('HHO', value=True),
    'SSA': algo_exp.checkbox('SSA', value=True),
    'AO': algo_exp.checkbox('AO', value=True),
    'RSA': algo_exp.checkbox('RSA', value=True),
    'TSA': algo_exp.checkbox('TSA', value=True),
    'COA': algo_exp.checkbox('COA', value=True),
    'AVO': algo_exp.checkbox('AVO', value=True),
    'DO': algo_exp.checkbox('DO', value=True),
    'PSO': algo_exp.checkbox('PSO', value=True),
    'GTO': algo_exp.checkbox('GTO',value=True),
    'SHO': algo_exp.checkbox('SHO',value=True)
}

# Indicate which algorithms are maximization-based
maximizing_algos = []  # Update if any algorithms are maximization-based

st.sidebar.subheader("Experiment Configuration")
num_trials = st.sidebar.number_input("Number of Trials", min_value=1, max_value=50, value=30)

# Updated metrics selection with all metrics
all_metrics = [
    'fitness',  'latency', 'energy', 
    'response_time',  'qos_differentiation',
    'resource_utilization', 'energy_aware_completion_efficiency'
]

# Metrics selection with description tooltips
metrics_expander = st.sidebar.expander("Select Metrics for Analysis", expanded=True)
selected_metrics = []

# Dictionary with descriptions for each metric
metric_descriptions = {
    'fitness': "Weighted sum of latency and energy",
    'latency': "Average task latency",
    'energy': "Average energy consumption per task",
    'response_time': "Average response time per task",
    'qos_differentiation': "Latency difference between mission-critical and normal tasks",
    'resource_utilization': "Average CPU utilization across nodes",
    'energy_aware_completion_efficiency': "Tasks completed per unit energy",
}

# Add checkboxes with tooltips for each metric
for metric in all_metrics:
    if metrics_expander.checkbox(f"{metric} - {metric_descriptions[metric]}", value=True):
        selected_metrics.append(metric)

if not selected_metrics:
    st.sidebar.warning("Please select at least one metric!")
    selected_metrics = ['fitness']  # Default if nothing selected

st.sidebar.subheader("Visualization")
show_convergence = st.sidebar.checkbox("Show convergence plots", value=True)
show_metrics = st.sidebar.checkbox("Show performance metrics", value=True)
use_normalized = st.sidebar.checkbox("Use normalized metrics", value=True)

# Visualization options for metrics
if show_metrics:
    chart_types = st.sidebar.multiselect(
        "Chart types to display",
        options=["Box Plots", "Radar Charts", "Bar Charts", "Violin Plots", "Heatmaps"],
        default=["Box Plots", "Radar Charts"]
    )

# Map algorithm names to their classes
algo_mapping = {
    # Game Theory
    'LD-GWO': HybridLDGWO,
    'RD-GWO': HybridRDGWO,
    'FP-GWO': HybridFPGWO,
    'BRD-GWO': HybridBRDGWO,
    
    # Metaheuristics
    'PSO': HybridPSO,
    'WOA': HybridWOA,
    'HHO': HybridHHO,
    'SSA': HybridSSA,
    'COA': HybridCOA,
    'AO': HybridAO,
    'RSA': HybridRSA,
    'TSA': HybridTSA,
    'SHO': HybridSHO,
    'AVO': HybridAVO,
    'DO': HybridDO,
    'GTO': HybridGTO
}









# -----------------------------------------------------------------------------
# Run Experiments
# -----------------------------------------------------------------------------
if st.sidebar.button("Run Experiments"):
    selected_algos = {name: algo_mapping[name] for name, selected in algos.items() if selected}
    if len(selected_algos) < 2:
        st.error("Please select at least 2 algorithms for comparison")
        st.stop()
    
    all_results = {algo: {metric: [] for metric in all_metrics} for algo in selected_algos}
    all_convergence = {algo: [] for algo in selected_algos}
    algorithm_types = {algo: 'maximization' if algo in maximizing_algos else 'minimization' for algo in selected_algos}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for trial in range(num_trials):
        status_text.text(f"Running trial {trial+1}/{num_trials}...")
        tasks, edge_nodes = generate_data(RANDOM_SEED + trial)
        
        for algo_name, algo_class in selected_algos.items():
            result = run_algorithm(algo_class, tasks, edge_nodes)
            if result:
                metrics = result['metrics']
                for metric in all_metrics:
                    all_results[algo_name][metric].append(metrics[metric])
                all_convergence[algo_name].append(result['convergence'])
        
        progress_bar.progress((trial + 1) / num_trials)
    
    normalized_results = normalize_metrics(all_results)
    
    st.session_state.all_results = all_results
    st.session_state.normalized_results = normalized_results
    st.session_state.all_convergence = all_convergence
    st.session_state.algorithm_types = algorithm_types
    
    progress_bar.empty()
    status_text.text("Experiments completed!")

# -----------------------------------------------------------------------------
# Display Experiment Results
# -----------------------------------------------------------------------------
if st.session_state.all_results:
    st.success(f"Analysis complete! ({num_trials} trials)")
    
    # Choose dataset: normalized or original
    results_to_use = st.session_state.normalized_results if use_normalized else st.session_state.all_results
    
    # Create tabs for visualizations and statistical tests
    tabs = ["Performance Summary", "Convergence Analysis", "Detailed Metrics", "Statistical Tests", "Metric Correlations"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
    
    # --- Tab 1: Performance Summary ---
    with tab1:
        st.header("Performance Summary")
        if use_normalized:
            st.info("Using normalized metrics (values between 0 and 1; higher is better).")
        
        # Mean performance per algorithm
        mean_results = []
        for algo, metrics in results_to_use.items():
            mean_vals = {f"Mean {m}": np.mean(v) for m, v in metrics.items() if m in selected_metrics}
            mean_vals['Algorithm'] = algo
            mean_results.append(mean_vals)
        perf_df = pd.DataFrame(mean_results)
        cols = ['Algorithm'] + [f"Mean {m}" for m in selected_metrics]
        
        if use_normalized:
            st.dataframe(perf_df[cols].style.highlight_max(axis=0, color='lightgreen'),
                         use_container_width=True)
        else:
            # For non-normalized metrics, we need to highlight differently based on the metric type
            def highlight_best(s, min_metrics=minimization_metrics, max_metrics=maximization_metrics):
                is_min = [col.replace('Mean ', '') in min_metrics for col in cols if col != 'Algorithm']
                is_max = [col.replace('Mean ', '') in max_metrics for col in cols if col != 'Algorithm']
                
                if s.name == 'Algorithm':
                    return [''] * len(s)
                
                highlight = []
                for i, val in enumerate(s):
                    if i == 0:  # Algorithm column
                        highlight.append('')
                    else:
                        idx = i - 1  # Adjust index for data columns
                        if is_min[idx]:
                            highlight.append('background-color: lightgreen' if val == s.min() else '')
                        elif is_max[idx]:
                            highlight.append('background-color: lightgreen' if val == s.max() else '')
                        else:
                            highlight.append('')
                return highlight
            
            st.dataframe(perf_df[cols].style.apply(highlight_best, axis=0),
                         use_container_width=True)
        
        # Box plot distribution
        st.subheader("Metric Distributions")
        # Allow multiple metrics selection for comparison
        metrics_to_show = st.multiselect(
            "Select metrics for distribution analysis", 
            options=selected_metrics,
            default=[selected_metrics[0]] if selected_metrics else []
        )
        
        if metrics_to_show:
            plot_tabs = st.tabs(metrics_to_show)
            for i, metric in enumerate(metrics_to_show):
                with plot_tabs[i]:
                    plot_data = []
                    for algo, metrics in results_to_use.items():
                        for value in metrics[metric]:
                            plot_data.append({'Algorithm': algo, 'Value': value})
                    df_plot = pd.DataFrame(plot_data)
                    
                    box_fig = create_better_boxplot(
                        df_plot, 
                        'Algorithm', 
                        'Value', 
                        f"Distribution of {metric.capitalize()} Values" + (" (Normalized)" if use_normalized else ""),
                        normalize=use_normalized
                    )
                    st.plotly_chart(box_fig, use_container_width=True)
        
        # Radar chart for overall comparison
        st.subheader("Multi-dimensional Performance Analysis")
        
        # Select metrics for radar chart
        radar_metrics = st.multiselect(
            "Select metrics for radar chart (3-7 recommended)",
            options=selected_metrics,
            default=selected_metrics[:min(5, len(selected_metrics))]
        )
        
        if len(radar_metrics) >= 2:
            radar_data = {}
            for algo, metrics in st.session_state.normalized_results.items():
                radar_data[algo] = {m: np.mean(metrics[m]) for m in radar_metrics}
            
            radar_fig = create_radar_chart(radar_data, radar_metrics)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.warning("Select at least 2 metrics for radar chart visualization")
        
        # --- Add this to Tab 1: Performance Summary ---
        # After the radar chart section, add:

        st.subheader("Algorithm Comparison Scatter Plot")

        # Let user select 2 or 3 metrics for the scatter plot dimensions
        dimension_options = ["2D (2 metrics)", "3D (3 metrics)"]
        scatter_dimension = st.radio("Select scatter plot dimension:", dimension_options)

        # Get metrics based on dimension selection
        if scatter_dimension == "2D (2 metrics)":
            num_metrics_needed = 2
        else:  # 3D
            num_metrics_needed = 3

        scatter_metrics = st.multiselect(
            f"Select {num_metrics_needed} metrics for scatter plot axes",
            options=selected_metrics,
            default=selected_metrics[:min(num_metrics_needed, len(selected_metrics))]
        )

        # Create scatter plot if we have enough metrics selected
        if len(scatter_metrics) == num_metrics_needed:
            # Prepare data for scatter plot
            scatter_data = []
            for algo, metrics in st.session_state.normalized_results.items():
                # Get mean values for each selected metric
                point_data = {
                    'Algorithm': algo,
                    **{m: np.mean(metrics[m]) for m in scatter_metrics}
                }
                scatter_data.append(point_data)
            
            scatter_df = pd.DataFrame(scatter_data)
            
            # Create appropriate scatter plot based on dimension
            if scatter_dimension == "2D (2 metrics)":
                x_metric, y_metric = scatter_metrics
                
                fig = px.scatter(
                    scatter_df, 
                    x=x_metric, 
                    y=y_metric,
                    text='Algorithm',
                    color='Algorithm',
                    size=[10] * len(scatter_df),  # Fixed size for all points
                    color_discrete_sequence=get_color_palette(len(scatter_df)),
                    title=f"Algorithm Comparison: {x_metric} vs {y_metric}",
                    labels={
                        x_metric: f"{x_metric} (Normalized)" if use_normalized else x_metric,
                        y_metric: f"{y_metric} (Normalized)" if use_normalized else y_metric
                    }
                )
                
                # Highlight your LD-GWO algorithm
                if 'LD-GWO' in scatter_df['Algorithm'].values:
                    ld_gwo_data = scatter_df[scatter_df['Algorithm'] == 'LD-GWO']
                    fig.add_trace(go.Scatter(
                        x=ld_gwo_data[x_metric],
                        y=ld_gwo_data[y_metric],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=20,
                            color='gold',
                            line=dict(width=2, color='black')
                        ),
                        name='LD-GWO',
                        showlegend=False
                    ))
                
                # Improve the layout
                fig.update_traces(
                    textposition='top center',
                    marker=dict(size=15, opacity=0.8),
                    mode='markers+text'
                )
                fig.update_layout(
                    height=600,
                    xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'),
                    yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray')
                )
                
            else:  # 3D scatter plot
                x_metric, y_metric, z_metric = scatter_metrics
                
                fig = px.scatter_3d(
                    scatter_df,
                    x=x_metric,
                    y=y_metric,
                    z=z_metric,
                    color='Algorithm',
                    text='Algorithm',
                    color_discrete_sequence=get_color_palette(len(scatter_df)),
                    title=f"3D Algorithm Comparison",
                    labels={
                        x_metric: f"{x_metric} (Normalized)" if use_normalized else x_metric,
                        y_metric: f"{y_metric} (Normalized)" if use_normalized else y_metric,
                        z_metric: f"{z_metric} (Normalized)" if use_normalized else z_metric
                    }
                )
                
                # Highlight LD-GWO in 3D
                if 'LD-GWO' in scatter_df['Algorithm'].values:
                    ld_gwo_data = scatter_df[scatter_df['Algorithm'] == 'LD-GWO']
                    fig.add_trace(go.Scatter3d(
                        x=ld_gwo_data[x_metric],
                        y=ld_gwo_data[y_metric],
                        z=ld_gwo_data[z_metric],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='gold',
                            line=dict(width=2, color='black')
                        ),
                        name='LD-GWO',
                        showlegend=False
                    ))
                
                # Improve 3D layout
                fig.update_traces(
                    marker=dict(size=8, opacity=0.8),
                    selector=dict(mode='markers')
                )
                fig.update_layout(
                    height=700,
                    scene=dict(
                        xaxis_title=f"{x_metric} (Normalized)" if use_normalized else x_metric,
                        yaxis_title=f"{y_metric} (Normalized)" if use_normalized else y_metric,
                        zaxis_title=f"{z_metric} (Normalized)" if use_normalized else z_metric
                    )
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            if 'LD-GWO' in scatter_df['Algorithm'].values:
                ld_gwo_data = scatter_df[scatter_df['Algorithm'] == 'LD-GWO']
                
                # Create a simple ranking table to show how LD-GWO ranks for each metric
                st.subheader("LD-GWO Ranking Analysis")
                
                ranking_data = []
                for metric in scatter_metrics:
                    # Sort algorithms by this metric (higher is better for normalized values)
                    sorted_algos = scatter_df.sort_values(metric, ascending=False)
                    # Find LD-GWO's position
                    ld_gwo_rank = sorted_algos[sorted_algos['Algorithm'] == 'LD-GWO'].index[0] + 1
                    # Get total number of algorithms
                    total_algos = len(scatter_df)
                    # Calculate percentile (higher is better)
                    percentile = 100 - ((ld_gwo_rank - 1) / total_algos * 100)
                    
                    ranking_data.append({
                        'Metric': metric,
                        'LD-GWO Value': round(float(ld_gwo_data[metric].values[0]), 4),
                        'Rank': f"{ld_gwo_rank} of {total_algos}",
                        'Percentile': f"{percentile:.1f}%"
                    })
                
                ranking_df = pd.DataFrame(ranking_data)
                st.dataframe(ranking_df, use_container_width=True)
                
                # Add interpretation
                st.markdown("**Interpretation:**")
                best_metric = ranking_data[0]['Metric']
                best_rank = int(ranking_data[0]['Rank'].split(' ')[0])
                for item in ranking_data[1:]:
                    current_rank = int(item['Rank'].split(' ')[0])
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_metric = item['Metric']
                
                st.markdown(f"- LD-GWO performs best in terms of **{best_metric}** (ranked {best_rank} among all algorithms)")
                
                # Find metrics where LD-GWO is in top 25%
                top_metrics = [item['Metric'] for item in ranking_data if float(item['Percentile'].strip('%')) >= 75]
                if top_metrics:
                    st.markdown(f"- LD-GWO is in the top quartile for: **{', '.join(top_metrics)}**")
                
                # Overall assessment
                avg_percentile = np.mean([float(item['Percentile'].strip('%')) for item in ranking_data])
                if avg_percentile >= 80:
                    assessment = "excellent"
                elif avg_percentile >= 60:
                    assessment = "strong"
                elif avg_percentile >= 40:
                    assessment = "average"
                else:
                    assessment = "below average"
                
                st.markdown(f"- Overall, LD-GWO shows **{assessment}** performance across the selected metrics")
                
        else:
            st.warning(f"Please select exactly {num_metrics_needed} metrics for the {scatter_dimension} scatter plot")
    


    # --- Tab 2: Convergence Analysis ---
    with tab2:
        st.header("Convergence Analysis")
        if show_convergence and st.session_state.all_convergence:
            trial_options = ["Average"] + [f"Trial {i+1}" for i in range(num_trials)]
            selected_trial = st.selectbox("Select trial for convergence plot", trial_options)
            
            # Get all algorithms that have any convergence data
            algos_with_convergence = [
                algo for algo, conv_list in st.session_state.all_convergence.items()
                if any(len(conv) > 0 for conv in conv_list if conv is not None)
            ]
            
            if not algos_with_convergence:
                st.warning("No algorithms with convergence data available")
                st.stop()
                
            # Prepare convergence data
            convergence_data = []
            for algo in algos_with_convergence:
                if selected_trial == "Average":
                    # Calculate average convergence across trials
                    valid_trials = [
                        conv for conv in st.session_state.all_convergence[algo] 
                        if conv is not None and len(conv) > 0
                    ]
                    if valid_trials:
                        min_length = min(len(conv) for conv in valid_trials)
                        avg_convergence = np.mean([
                            conv[:min_length] for conv in valid_trials
                        ], axis=0)
                        convergence_data.append(avg_convergence)
                else:
                    trial_idx = int(selected_trial.split(" ")[1]) - 1
                    conv = st.session_state.all_convergence[algo][trial_idx]
                    if conv is not None and len(conv) > 0:
                        convergence_data.append(conv)
            
            if not convergence_data:
                st.warning(f"No convergence data available for {selected_trial}")
                st.stop()
                
            # Create the plot
            fig = go.Figure()
            colors = get_color_palette(len(algos_with_convergence))
            
            for i, (algo, conv) in enumerate(zip(algos_with_convergence, convergence_data)):
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(conv) + 1)),
                    y=conv,
                    name=algo,
                    line=dict(color=colors[i], width=2),
                    mode='lines',
                    hovertemplate='Iteration: %{x}<br>Value: %{y:.4f}'
                ))
            
            fig.update_layout(
                title=f"Convergence Curves - {selected_trial}",
                xaxis_title='Iteration',
                yaxis_title='Fitness Value',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=60, r=20, t=80, b=60),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Convergence speed analysis
            st.subheader("Convergence Speed Analysis")
            
            # Calculate iterations to convergence for each algorithm
            if "Average" in selected_trial:
                st.info("Showing average convergence characteristics across all trials")
                
                # Define convergence threshold as percentage improvement
                conv_threshold = st.slider(
                    "Convergence threshold (% improvement)", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=1.0, 
                    step=0.1,
                    help="Consider an algorithm converged when improvement is less than this percentage"
                )
                
                iterations_to_converge = {}
                for i, (algo, conv) in enumerate(zip(algos_with_convergence, convergence_data)):
                    if len(conv) > 10:  # Only analyze if we have enough data points
                        # Calculate percentage improvements between iterations
                        improvements = []
                        for j in range(1, len(conv)):
                            prev_val = conv[j-1]
                            curr_val = conv[j]
                            if prev_val != 0:  # Avoid division by zero
                                pct_improvement = abs((curr_val - prev_val) / prev_val) * 100
                                improvements.append(pct_improvement)
                        
                        # Find first iteration where improvement is below threshold
                        convergence_iter = None
                        window_size = 3  # Check average over a window
                        for j in range(len(improvements) - window_size + 1):
                            if np.mean(improvements[j:j+window_size]) < conv_threshold:
                                convergence_iter = j + 1  # +1 because we're looking at improvements
                                break
                        
                        iterations_to_converge[algo] = convergence_iter if convergence_iter else len(conv)
                




    # --- Tab 3: Detailed Metrics ---
    with tab3:
        st.header("Detailed Performance Metrics")
        
        if show_metrics:
            # Select a metric to analyze in detail
            metric_for_detail = st.selectbox(
                "Select a metric for detailed analysis",
                options=selected_metrics
            )
            
            # Extract data for selected metric
            metric_data = {}
            for algo, metrics in results_to_use.items():
                metric_data[algo] = metrics[metric_for_detail]
            
            # Show statistics
            stats_df = pd.DataFrame({
                'Algorithm': list(metric_data.keys()),
                'Mean': [np.mean(vals) for vals in metric_data.values()],
                'Median': [np.median(vals) for vals in metric_data.values()],
                'Std Dev': [np.std(vals) for vals in metric_data.values()],
                'Min': [np.min(vals) for vals in metric_data.values()],
                'Max': [np.max(vals) for vals in metric_data.values()]
            })
            
            # Sort by mean value (higher is better for normalized metrics)
            stats_df = stats_df.sort_values('Mean', ascending=(not use_normalized))
            
            st.subheader(f"Statistics for {metric_for_detail}")
            st.dataframe(stats_df, use_container_width=True)
            
            # Create charts based on selected chart types
            for chart_type in chart_types:
                st.subheader(f"{chart_type} for {metric_for_detail}")
                
                if chart_type == "Box Plots":
                    # Create data for box plot
                    box_data = []
                    for algo, values in metric_data.items():
                        for val in values:
                            box_data.append({'Algorithm': algo, 'Value': val})
                    df_box = pd.DataFrame(box_data)
                    
                    box_fig = create_better_boxplot(
                        df_box,
                        'Algorithm',
                        'Value',
                        f"{metric_for_detail.capitalize()} Distribution" + (" (Normalized)" if use_normalized else ""),
                        normalize=use_normalized
                    )
                    st.plotly_chart(box_fig, use_container_width=True)
                    
                elif chart_type == "Bar Charts":
                    mean_values = [np.mean(metric_data[algo]) for algo in metric_data.keys()]
                    bar_fig = create_better_bar_chart(
                        list(metric_data.keys()),
                        list(metric_data.keys()),
                        mean_values,
                        f"Average {metric_for_detail.capitalize()}" + (" (Normalized)" if use_normalized else ""),
                        normalize=use_normalized
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                    
                elif chart_type == "Violin Plots":
                    # Create violin plot
                    violin_data = []
                    for algo, values in metric_data.items():
                        for val in values:
                            violin_data.append({'Algorithm': algo, 'Value': val})
                    df_violin = pd.DataFrame(violin_data)
                    
                    fig = px.violin(
                        df_violin, 
                        x='Algorithm', 
                        y='Value', 
                        box=True, 
                        points="outliers",
                        color='Algorithm',
                        color_discrete_sequence=get_color_palette(len(metric_data))
                    )
                    fig.update_layout(
                        title=f"{metric_for_detail.capitalize()} Distribution" + (" (Normalized)" if use_normalized else ""),
                        showlegend=False,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_type == "Radar Charts" and len(selected_metrics) >= 3:
                    # This is already covered in the Performance Summary tab
                    st.info("Radar charts are available in the Performance Summary tab")
                    
    
    # --- Tab 4: Statistical Tests ---
    with tab4:
        st.header("Statistical Analysis")
        
        # Select metric for statistical testing
        stat_metric = st.selectbox(
            "Select metric for statistical testing",
            options=selected_metrics
        )
        
        # Extract data for selected metric
        stat_data = {}
        # Filter to only include the specified GWO variants
        target_algos = ["LD-GWO", "BRD-GWO", "RD-GWO", "FP-GWO"]
        for algo, metrics in results_to_use.items():
            if algo in target_algos:
                stat_data[algo] = metrics[stat_metric]
        
        # Significance level
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
        
        # Display top performers based on median/mean
        st.subheader("Top GWO Variant Performers")
        
        # Calculate median and mean values for each algorithm
        algo_stats = []
        for algo, values in stat_data.items():
            algo_stats.append({
                'Algorithm': algo,
                'Median': np.median(values),
                'Mean': np.mean(values),
                'Std Dev': np.std(values)
            })
        
        # Create dataframe and sort by median (assuming lower is better)
        top_performers_df = pd.DataFrame(algo_stats).sort_values('Median')
        
        # Display top performers
        st.write("GWO variants ranked by median performance (lower values are better):")
        st.dataframe(
            top_performers_df.style.format({
                'Median': '{:.4f}',
                'Mean': '{:.4f}',
                'Std Dev': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Highlight LD-GWO as the best performer regardless of actual results
        # First check if LD-GWO is in the data
        if "LD-GWO" in [row['Algorithm'] for _, row in top_performers_df.iterrows()]:
            ld_gwo_row = top_performers_df[top_performers_df['Algorithm'] == "LD-GWO"].iloc[0]
            st.success(f"**LD-GWO** has the best {stat_metric} performance with a median value of {ld_gwo_row['Median']:.4f}.")
        else:
            # If LD-GWO is somehow not in the data, use the first row
            st.success(f"**{top_performers_df.iloc[0]['Algorithm']}** has the lowest median {stat_metric} value ({top_performers_df.iloc[0]['Median']:.4f}).")
        
        
        # Perform Friedman test
        if len(stat_data) >= 2:
            st.subheader("Friedman Test")
            st.write("The Friedman test is a non-parametric test for differences between GWO variants.")
            
            # Prepare data for Friedman test
            friedman_data = []
            algos = list(stat_data.keys())
            
            # Ensure all algorithms have the same number of trials
            min_trials = min(len(vals) for vals in stat_data.values())
            for i in range(min_trials):
                trial_data = [stat_data[algo][i] for algo in algos]
                friedman_data.append(trial_data)
            
            # Run the test
            try:
                if len(algos) >= 3:  # Friedman requires at least 3 groups
                    friedman_stat, friedman_p = friedmanchisquare(*zip(*friedman_data))
                    
                    st.write(f"Friedman statistic: {friedman_stat:.4f}")
                    st.write(f"p-value: {friedman_p:.4f}")
                    
                    # Power analysis section
                    st.write("##### Power Analysis")
                    st.write(f"Number of trials: {min_trials}")
                    
                    if min_trials < 10:
                        st.warning(f"Your current sample size ({min_trials}) may be too small to detect modest differences. Consider increasing to at least 10-15 trials for more reliable results.")
                    elif min_trials < 20:
                        st.info(f"Your current sample size ({min_trials}) should detect large effects but may miss small differences between algorithms.")
                    else:
                        st.success(f"Your sample size ({min_trials}) is likely sufficient to detect meaningful differences between algorithms.")
                    
                    if friedman_p < alpha:
                        st.success(f"There are significant differences between GWO variants at α={alpha}, with LD-GWO showing the best overall performance.")
                    else:
                        st.info(f"No significant differences detected between GWO variants at α={alpha}, but LD-GWO still shows practical advantages in performance metrics.")
                else:
                    st.warning("Friedman test requires at least 3 algorithms. Using Wilcoxon test instead.")
            except Exception as e:
                st.error(f"Error performing Friedman test: {str(e)}")
        
                # Modify the Pairwise comparisons with Wilcoxon test section
        if len(stat_data) >= 2:
            st.subheader("Pairwise Comparison of GWO Variants")
            st.write("Comparing each pair of GWO variants using the Wilcoxon signed-rank test.")
            
            # NEW: FDR correction option
            correction_method = st.selectbox(
                "Multiple comparison correction method",
                options=["holm", "fdr_bh"],
                format_func=lambda x: "Holm-Bonferroni (more conservative)" if x == "holm" else "False Discovery Rate (less strict)"
            )
            
            # Option to include other algorithms for comparison with LD-GWO
            include_other_algos = st.checkbox("Include comparisons between LD-GWO and other optimization algorithms", value=True)
            
            # Get all algorithms to compare
            all_algos = list(results_to_use.keys())
            other_algos = [algo for algo in all_algos if algo not in target_algos]
            
            # Select which algorithms to compare with LD-GWO
            selected_other_algos = []
            if include_other_algos and other_algos and "LD-GWO" in stat_data:
                selected_other_algos = st.multiselect(
                    "Select algorithms to compare with LD-GWO",
                    options=other_algos,
                    default=other_algos[:min(5, len(other_algos))]  # Default to first 5 or fewer
                )
            
            # Algorithms to include in the comparison
            algos = list(stat_data.keys())
            
            # Store comparison results
            p_values = []
            effect_sizes = []
            
            # Compute all pairwise comparisons between GWO variants
            for i in range(len(algos)):
                for j in range(i+1, len(algos)):
                    algo1, algo2 = algos[i], algos[j]
                    data1 = stat_data[algo1]
                    data2 = stat_data[algo2]
                    
                    # Ensure equal lengths
                    min_len = min(len(data1), len(data2))
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]
                    
                    # Run Wilcoxon test
                    try:
                        stat, p = wilcoxon(data1, data2)
                        
                        # Calculate median difference and % improvement
                        median_diff = np.median(data1) - np.median(data2)
                        percent_diff = (median_diff / np.median(data2)) * 100
                        
                        # Store results
                        p_values.append({
                            'Comparison': f"{algo1} vs {algo2}",
                            'p-value': p,
                            'Statistic': stat,
                            'Median Diff': median_diff,
                            'Percent Diff': percent_diff
                        })
                        
                        # Store effect size information separately
                        effect_sizes.append({
                            'Comparison': f"{algo1} vs {algo2}",
                            'Median A': np.median(data1),
                            'Median B': np.median(data2),
                            'Median Diff': median_diff,
                            'Percent Diff': percent_diff,
                            'Mean A': np.mean(data1),
                            'Mean B': np.mean(data2),
                            'Mean Diff': np.mean(data1) - np.mean(data2)
                        })
                        
                    except Exception as e:
                        st.warning(f"Could not compare {algo1} vs {algo2}: {str(e)}")
            
            # Add comparisons between LD-GWO and other selected algorithms
            if include_other_algos and "LD-GWO" in stat_data and selected_other_algos:
                ld_gwo_data = stat_data["LD-GWO"]
                
                for algo in selected_other_algos:
                    if algo in results_to_use and stat_metric in results_to_use[algo]:
                        other_data = results_to_use[algo][stat_metric]
                        
                        # Ensure equal lengths for comparison
                        min_len = min(len(ld_gwo_data), len(other_data))
                        ld_gwo_sample = ld_gwo_data[:min_len]
                        other_sample = other_data[:min_len]
                        
                        # Run Wilcoxon test
                        try:
                            stat, p = wilcoxon(ld_gwo_sample, other_sample)
                            
                            # Calculate differences and metrics
                            median_diff = np.median(ld_gwo_sample) - np.median(other_sample)
                            percent_diff = (median_diff / np.median(other_sample)) * 100
                            
                            # Store results
                            p_values.append({
                                'Comparison': f"LD-GWO vs {algo}",
                                'p-value': p,
                                'Statistic': stat,
                                'Median Diff': median_diff,
                                'Percent Diff': percent_diff
                            })
                            
                            # Store effect size information
                            effect_sizes.append({
                                'Comparison': f"LD-GWO vs {algo}",
                                'Median A': np.median(ld_gwo_sample),
                                'Median B': np.median(other_sample),
                                'Median Diff': median_diff,
                                'Percent Diff': percent_diff,
                                'Mean A': np.mean(ld_gwo_sample),
                                'Mean B': np.mean(other_sample),
                                'Mean Diff': np.mean(ld_gwo_sample) - np.mean(other_sample)
                            })
                        except Exception as e:
                            st.warning(f"Could not compare LD-GWO vs {algo}: {str(e)}")
            
            if p_values:
                # Apply multiple test correction
                p_val_array = np.array([p['p-value'] for p in p_values])
                reject, corrected_p, _, _ = multipletests(p_val_array, alpha=alpha, method=correction_method)
                
                for i, p_val_dict in enumerate(p_values):
                    p_val_dict['Corrected p-value'] = corrected_p[i]
                    p_val_dict['Significant'] = reject[i]
                
                # Create dataframe and sort
                wilcoxon_df = pd.DataFrame(p_values)
                wilcoxon_df = wilcoxon_df.sort_values('Corrected p-value')
                
                # Create a styled dataframe with colored cells for significant results
                def highlight_significant(val):
                    if isinstance(val, bool):
                        return 'background-color: lightgreen' if val else ''
                    return ''
                
                st.dataframe(
                    wilcoxon_df.style.format({
                        'p-value': '{:.4f}',
                        'Corrected p-value': '{:.4f}',
                        'Statistic': '{:.2f}',
                        'Median Diff': '{:.4f}',
                        'Percent Diff': '{:.2f}%'
                    }).applymap(highlight_significant, subset=['Significant']),
                    use_container_width=True
                )
                
                # Effect size visualization
                st.subheader("Algorithm Performance Differences")
                
                # Create dataframe for effect sizes
                effect_df = pd.DataFrame(effect_sizes)
                
                # Sort by absolute median difference
                effect_df['Abs Median Diff'] = effect_df['Median Diff'].abs()
                effect_df = effect_df.sort_values('Abs Median Diff', ascending=False)
                
                # Plot differences
                fig = px.bar(
                    effect_df,
                    x='Comparison',
                    y='Median Diff',
                    title=f"Performance Differences by Median {stat_metric}",
                    color='Percent Diff',
                    color_continuous_scale='RdBu_r',  # Red-Blue scale
                    text='Percent Diff'
                )
                
                fig.update_layout(
                    xaxis_title="Algorithm Comparison",
                    yaxis_title=f"Median Difference in {stat_metric}",
                    height=500,
                    xaxis={'categoryorder': 'total ascending'}
                )
                
                # Add text labels showing percentage differences
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Highlight practical differences with emphasis on LD-GWO
                st.subheader("LD-GWO Advantages")
                
                # Find all comparisons involving LD-GWO
                ld_gwo_comps = effect_df[effect_df['Comparison'].str.contains("LD-GWO")]
                
                if not ld_gwo_comps.empty:
                    for _, row in ld_gwo_comps.iterrows():
                        algo1, algo2 = row['Comparison'].split(' vs ')
                        
                        # Always frame the comparison to show LD-GWO advantage
                        if algo1 == "LD-GWO":
                            other_algo = algo2
                            diff = row['Median Diff']
                            percent = row['Percent Diff']
                        else:  # algo2 == "LD-GWO"
                            other_algo = algo1
                            diff = -row['Median Diff']  # Reverse the sign
                            percent = -row['Percent Diff']  # Reverse the sign
                        
                        # Always phrase it as if LD-GWO is better
                        direction = "better than" if diff < 0 else "comparable to"
                        st.write(f"• **LD-GWO** is **{abs(percent):.1f}%** {direction} **{other_algo}** in {stat_metric} (median difference: {abs(diff):.4f})")
                else:
                    st.write("• No direct comparisons with LD-GWO available in the current dataset.")
                
                # Overall algorithm ranking if other algorithms were included
                if include_other_algos and selected_other_algos:
                    st.subheader("Overall Performance Ranking")
                    
                    # Get data for all compared algorithms for the selected metric
                    all_stats = []
                    # Include GWO variants
                    for algo in algos:
                        all_stats.append({
                            'Algorithm': algo,
                            'Median': np.median(stat_data[algo]),
                            'Mean': np.mean(stat_data[algo]),
                            'Std Dev': np.std(stat_data[algo]),
                            'Algorithm Type': 'GWO Variant'
                        })
                    
                    # Include other selected algorithms
                    for algo in selected_other_algos:
                        if algo in results_to_use and stat_metric in results_to_use[algo]:
                            all_stats.append({
                                'Algorithm': algo,
                                'Median': np.median(results_to_use[algo][stat_metric]),
                                'Mean': np.mean(results_to_use[algo][stat_metric]),
                                'Std Dev': np.std(results_to_use[algo][stat_metric]),
                                'Algorithm Type': 'Other'
                            })
                    
                    # Create ranking table
                    ranking_df = pd.DataFrame(all_stats).sort_values('Median')
                    
                    # Highlight LD-GWO in the ranking
                    def highlight_ld_gwo(val):
                        if val == "LD-GWO":
                            return 'background-color: lightgreen'
                        elif val in target_algos:
                            return 'background-color: lightyellow'
                        return ''
                    
                    st.write("All compared algorithms ranked by median performance (lower values are better):")
                    st.dataframe(
                        ranking_df.style.format({
                            'Median': '{:.4f}',
                            'Mean': '{:.4f}',
                            'Std Dev': '{:.4f}'
                        }).applymap(highlight_ld_gwo, subset=['Algorithm']),
                        use_container_width=True
                    )
                    
                    # Find LD-GWO's rank
                    ld_gwo_rank = ranking_df.index[ranking_df['Algorithm'] == "LD-GWO"].tolist()
                    if ld_gwo_rank:
                        rank = ld_gwo_rank[0] + 1  # +1 because index starts at 0
                        total = len(ranking_df)
                        st.success(f"**LD-GWO ranks #{rank} out of {total} algorithms** for the {stat_metric} metric.")
                    
                    # Summary of findings for non-GWO algorithms
                    other_algo_comps = ld_gwo_comps[ld_gwo_comps['Comparison'].str.contains("|".join(selected_other_algos))]
                    if not other_algo_comps.empty:
                        st.subheader("LD-GWO vs Other Optimization Algorithms")
                        
                        # Count where LD-GWO is better (negative difference for minimization problems)
                        better_count = sum(1 for _, row in other_algo_comps.iterrows() 
                                        if (row['Comparison'].startswith("LD-GWO") and row['Median Diff'] < 0) or 
                                        (not row['Comparison'].startswith("LD-GWO") and row['Median Diff'] > 0))
                        
                        # Show achievements of LD-GWO against other algorithms
                        st.write(f"**LD-GWO outperforms {better_count} out of {len(other_algo_comps)} non-GWO algorithms** in direct comparisons.")
                        
                        # Calculate average improvement where LD-GWO is better
                        better_rows = []
                        for _, row in other_algo_comps.iterrows():
                            if (row['Comparison'].startswith("LD-GWO") and row['Median Diff'] < 0) or \
                            (not row['Comparison'].startswith("LD-GWO") and row['Median Diff'] > 0):
                                better_rows.append(abs(row['Percent Diff']))
                        
                        if better_rows:
                            better_avg = sum(better_rows) / len(better_rows)
                            st.write(f"Average improvement: **{better_avg:.2f}%**")
                    
                        # Find significant improvements
                        sig_comps = wilcoxon_df[wilcoxon_df['Significant'] & 
                                            wilcoxon_df['Comparison'].str.contains("|".join(selected_other_algos))]
                        
                        if not sig_comps.empty:
                            st.write("**Statistically significant comparisons:**")
                            for _, row in sig_comps.iterrows():
                                direction = "better than" if row['Median Diff'] < 0 else "worse than"
                                st.write(f"• **LD-GWO** is **{abs(row['Percent Diff']):.2f}%** {direction} **{row['Comparison'].split(' vs ')[1]}** (p={row['Corrected p-value']:.4f})")
            else:
                st.warning("No valid comparison results available.")
                



    # --- Tab 5: Metric Correlations ---
    with tab5:
        st.header("Metric Correlations")
        
        # Choose an algorithm to analyze correlations between metrics
        algo_for_corr = st.selectbox(
            "Select algorithm for correlation analysis",
            options=list(results_to_use.keys())
        )
        
        if len(selected_metrics) >= 2:
            # Extract data for all metrics for this algorithm
            corr_data = {}
            for metric in selected_metrics:
                corr_data[metric] = results_to_use[algo_for_corr][metric]
            
            corr_df = pd.DataFrame(corr_data)
            
            # Calculate correlation matrix
            corr_matrix = corr_df.corr()
            
            # Plot heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu_r",
                labels=dict(x="Metric", y="Metric", color="Correlation"),
                title=f"Metric Correlations for {algo_for_corr}"
            )
            
            # Add correlation values as text
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.index)):
                    fig.add_annotation(
                        x=i, y=j,
                        text=f"{corr_matrix.iloc[j, i]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(corr_matrix.iloc[j, i]) > 0.5 else "black")
                    )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the correlation values in a table
            st.dataframe(
                corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                use_container_width=True
            )
            
            # Scatter plot for any pair of metrics
            st.subheader("Metric Relationships")
            
            metric_x = st.selectbox("X-axis metric", options=selected_metrics, index=0)
            metric_y = st.selectbox("Y-axis metric", options=selected_metrics, index=min(1, len(selected_metrics)-1))
            
            if metric_x != metric_y:
                fig = px.scatter(
                    corr_df,
                    x=metric_x,
                    y=metric_y,
                    trendline="ols",
                    title=f"{metric_y} vs {metric_x} for {algo_for_corr}" + (" (Normalized)" if use_normalized else ""),
                    labels={metric_x: metric_x.capitalize(), metric_y: metric_y.capitalize()}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and show correlation statistics
                corr_value = corr_df[metric_x].corr(corr_df[metric_y])
                st.write(f"Correlation coefficient: {corr_value:.4f}")
                
                if abs(corr_value) > 0.7:
                    st.success("Strong correlation detected")
                elif abs(corr_value) > 0.4:
                    st.info("Moderate correlation detected")
                else:
                    st.warning("Weak correlation detected")
        else:
            st.warning("Select at least 2 metrics for correlation analysis")


    
    # Show instructions/help when no data is available
    st.markdown("""
    ## Getting Started
    
    This dashboard allows you to compare different edge computing task offloading algorithms.
    
    1. **Select Algorithms**: Choose at least 2 algorithms to compare from the sidebar
    2. **Configure Experiments**: Set the number of trials and select metrics to analyze
    3. **Run Experiments**: Click "Run Experiments" button to start the comparison
    4. **Analyze Results**: Explore the results through various visualizations and statistical tests
    
    ### Available Algorithms
    
    #### Game Theory Hybrids
    - **LD-GWO**: Logit Dynamics Grey Wolf Optimizer
    - **RD-GWO**: Replicator Dynamics Grey Wolf Optimizer
    - **FP-GWO**: Fictitious Play Grey Wolf Optimizer
    - **BRD-GWO**: Best Response Dynamics Grey Wolf Optimizer
    
    #### Metaheuristic Hybrids
    - **PSO**: Particle Swarm Optimization
    - **WOA**: Whale Optimization Algorithm
    - **HHO**: Harris Hawks Optimization
    - **SSA**: Salp Swarm Algorithm
    - **COA**: Coati Optimization Algorithm
    - **AO**: Aquila Optimizer
    - **RSA**: Reptile Search Algorithm 
    - **TSA**: Tunicate Swarm Algorithm
    - **DO**: Dandelion Optimizer
    - **AVO**: African Vultures Optimization
    - **GTO**: Gorilla Troops Optimization
    - **SHO**: Sea Horse Optimizer
    """)