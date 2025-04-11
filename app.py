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
    HybridWOA,
    HybridHHO,
    HybridSSA,
    HybridAO,
    HybridRSA,
    HybridTSA,
    HybridGBO,
    HybridAVO,
    HybridQANA
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
    """Generate task and edge node data with a specified seed."""
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
                # For metrics where lower is better (e.g., latency, energy),
                # invert the normalization so higher normalized values are better.
                normalized_results[algo][m] = [
                    1 - ((val - min_val) / (max_val - min_val)) for val in values
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
        points="outliers"  # Only show outlier points
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title=y_column,
        legend_title_text='',
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
    
    # Add a note about normalization if applicable
    if normalize:
        fig.add_annotation(
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            text="Normalized values (higher is better)",
            showarrow=False,
            font=dict(size=10, color="gray")
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
    
    # Add a note about normalization if applicable
    if normalize:
        fig.add_annotation(
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            text="Normalized values (higher is better)",
            showarrow=False,
            font=dict(size=10, color="gray")
        )
    
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

if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = ['latency', 'energy', 'resource_utilization']

# Set custom page style
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
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
    'GBO': algo_exp.checkbox('GBO', value=True),
    'AVO': algo_exp.checkbox('AVO', value=True),
    'QANA': algo_exp.checkbox('QANA', value=True),
    'PSO': algo_exp.checkbox('PSO', value=True)
}

# Indicate which algorithms are maximization-based
maximizing_algos = []  # Update if any algorithms are maximization-based

st.sidebar.subheader("Experiment Configuration")
num_trials = st.sidebar.number_input("Number of Trials", min_value=1, max_value=50, value=30)
selected_metrics = st.sidebar.multiselect(
    "Metrics for Statistical Tests",
    options=['fitness', 'latency', 'energy', 'resource_utilization'],
    default=['fitness']
)

st.sidebar.subheader("Visualization")
show_convergence = st.sidebar.checkbox("Show convergence plots", value=True)
show_metrics = st.sidebar.checkbox("Show performance metrics", value=True)
use_normalized = st.sidebar.checkbox("Use normalized metrics", value=True)

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
    'AO': HybridAO,
    'RSA': HybridRSA,
    'TSA': HybridTSA,
    'GBO': HybridGBO,
    'AVO': HybridAVO,
    'QANA': HybridQANA
}

# -----------------------------------------------------------------------------
# Run Experiments
# -----------------------------------------------------------------------------
if st.sidebar.button("Run Experiments"):
    selected_algos = {name: algo_mapping[name] for name, selected in algos.items() if selected}
    if len(selected_algos) < 2:
        st.error("Please select at least 2 algorithms for comparison")
        st.stop()
    
    all_results = {algo: {metric: [] for metric in selected_metrics} for algo in selected_algos}
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
                for metric in selected_metrics:
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
    tabs = ["Performance Summary", "Convergence Analysis", "Detailed Metrics", "Statistical Tests"]
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # --- Tab 1: Performance Summary ---
    with tab1:
        st.header("Performance Summary")
        if use_normalized:
            st.info("Using normalized metrics (values between 0 and 1; higher is better).")
        
        # Mean performance per algorithm
        mean_results = []
        for algo, metrics in results_to_use.items():
            mean_vals = {f"Mean {m}": np.mean(v) for m, v in metrics.items()}
            mean_vals['Algorithm'] = algo
            mean_results.append(mean_vals)
        perf_df = pd.DataFrame(mean_results)
        cols = ['Algorithm'] + [f"Mean {m}" for m in selected_metrics]
        
        if use_normalized:
            st.dataframe(perf_df[cols].style.highlight_max(axis=0, color='lightgreen'),
                         use_container_width=True)
        else:
            st.dataframe(perf_df[cols].style.highlight_min(axis=0, color='lightgreen'),
                         use_container_width=True)
        
        # Box plot distribution
        st.subheader("Metric Distributions")
        metric_to_show = st.selectbox("Select metric for distribution", selected_metrics)
        plot_data = []
        for algo, metrics in results_to_use.items():
            for value in metrics[metric_to_show]:
                plot_data.append({'Algorithm': algo, 'Value': value})
        df_plot = pd.DataFrame(plot_data)
        
        box_fig = create_better_boxplot(
            df_plot, 
            'Algorithm', 
            'Value', 
            f"Distribution of {metric_to_show.capitalize()} Values" + (" (Normalized)" if use_normalized else ""),
            normalize=use_normalized
        )
        st.plotly_chart(box_fig, use_container_width=True)
    
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
    
    # --- Tab 3: Detailed Metrics ---
    with tab3:
        st.header("Detailed Metrics Analysis")
        metric_choice = st.selectbox("Select metric to analyze", selected_metrics)
        
        # Radar Chart
        st.subheader("Radar Chart Comparison")
        radar_data = {}
        for algo, metrics in st.session_state.normalized_results.items():
            radar_data[algo] = {m: np.mean(metrics[m]) for m in selected_metrics}
        
        categories = selected_metrics
        radar_fig = create_radar_chart(radar_data, categories)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Detailed analysis by metric
        st.subheader(f"Detailed {metric_choice.capitalize()} Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            mean_vals = {algo: np.mean(results_to_use[algo][metric_choice]) for algo in results_to_use}
            bar_fig = create_better_bar_chart(
                list(mean_vals.keys()),
                list(mean_vals.keys()),
                list(mean_vals.values()),
                f"Mean {metric_choice.capitalize()}" + (" (Normalized)" if use_normalized else ""),
                normalize=use_normalized
            )
            st.plotly_chart(bar_fig)
            
        with col2:
            plot_data = []
            for algo, metrics in results_to_use.items():
                for value in metrics[metric_choice]:
                    plot_data.append({'Algorithm': algo, 'Value': value})
            df_plot = pd.DataFrame(plot_data)
            
            violin_fig = px.violin(
                df_plot,
                x="Algorithm", 
                y="Value",
                color="Algorithm",
                box=True,
                color_discrete_sequence=get_color_palette(len(df_plot["Algorithm"].unique())),
                title=f"Distribution of {metric_choice.capitalize()}" + (" (Normalized)" if use_normalized else "")
            )
            violin_fig.update_layout(
                xaxis_title="",
                legend_title_text='',
                xaxis_tickangle=45,
                height=500,
                showlegend=False
            )
            
            if use_normalized:
                violin_fig.add_annotation(
                    x=0.5, y=1.05,
                    xref="paper", yref="paper",
                    text="Normalized values (higher is better)",
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
                
            st.plotly_chart(violin_fig)
    
    # --- Tab 4: Statistical Tests ---
    with tab4:
        st.header("Statistical Analysis")
        st.info("Statistical tests are performed on raw (un-normalized) metrics for fairness.")
        selected_test_metric = st.selectbox("Select metric for statistical tests", selected_metrics)
        
        # Prepare data for tests
        algo_values = {}
        for algo, metrics in st.session_state.all_results.items():
            if selected_test_metric in metrics:
                algo_values[algo] = metrics[selected_test_metric]
                
        # Friedman test
        if len(algo_values) >= 3:
            algos_list = list(algo_values.keys())
            values_list = [algo_values[algo] for algo in algos_list]
            try:
                friedman_stat, friedman_p = friedmanchisquare(*values_list)
                
                st.subheader("Friedman Test")
                col1, col2 = st.columns(2)
                with col1:
                    friedman_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=1 - friedman_p,  # Invert p-value for visual purposes
                        title={'text': "Confidence Level", 'font': {'size': 24}},
                        number={'suffix': "%", 'font': {'size': 26}, 'valueformat': '.2f'},
                        gauge={
                            'axis': {'range': [0, 1], 'tickvals': [0, 0.05, 0.5, 0.95, 1], 
                                    'ticktext': ['0%', '5%', '50%', '95%', '100%']},
                            'bar': {'color': "green" if friedman_p < 0.05 else "gray"},
                            'steps': [
                                {'range': [0, 0.05], 'color': "lightgray"},
                                {'range': [0.05, 0.95], 'color': "lightblue"},
                                {'range': [0.95, 1], 'color': "royalblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.95
                            }
                        }
                    ))
                    
                    friedman_fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    st.plotly_chart(friedman_fig)
                
                with col2:
                    st.markdown(f"""
                    ### Friedman Test Results
                    
                    **Chi-Square Value:** {friedman_stat:.3f}
                    
                    **p-value:** {friedman_p:.5f}
                    
                    **Conclusion:** {"Significant differences detected between algorithms (p < 0.05)" if friedman_p < 0.05 else "No significant differences detected between algorithms (p ≥ 0.05)"}
                    """)
            except Exception as e:
                st.error(f"Error performing Friedman test: {e}")
        else:
            st.warning("Friedman test requires at least 3 algorithms.")
        
        # Pairwise Wilcoxon tests
        st.subheader("Pairwise Wilcoxon Tests")
        if len(algo_values) >= 2:
            algo_pairs = [(a1, a2) for idx, a1 in enumerate(algo_values.keys()) 
                          for a2 in list(algo_values.keys())[idx+1:]]
            results = []
            p_values = []
            for a1, a2 in algo_pairs:
                try:
                    stat, p = wilcoxon(algo_values[a1], algo_values[a2])
                    results.append({
                        'Algorithm 1': a1,
                        'Algorithm 2': a2,
                        'Statistic': stat,
                        'p-value': p
                    })
                    p_values.append(p)
                except Exception as e:
                    st.error(f"Error performing Wilcoxon test between {a1} and {a2}: {e}")
            
            # Adjust p-values for multiple comparisons
            if p_values:
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
                
                # Update results with corrected p-values
                for i, result in enumerate(results):
                    results[i]['Corrected p-value'] = pvals_corrected[i]
                    results[i]['Significant'] = reject[i]
                
                # Create a DataFrame for display
                results_df = pd.DataFrame(results)
                
                # Apply conditional formatting
                styled_df = results_df.style.apply(
                    lambda row: ['background-color: lightgreen' if row['Significant'] else 'background-color: white' 
                                for _ in row], axis=1
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Create a heatmap for p-values
                st.subheader("P-value Heatmap")
                
                # Prepare matrix data for heatmap
                algos = list(algo_values.keys())
                p_matrix = pd.DataFrame(index=algos, columns=algos)
                
                # Fill diagonal with 1.0 (same algorithm)
                for algo in algos:
                    p_matrix.loc[algo, algo] = 1.0
                
                # Fill with computed p-values
                for result in results:
                    a1, a2 = result['Algorithm 1'], result['Algorithm 2']
                    p_matrix.loc[a1, a2] = result['Corrected p-value']
                    p_matrix.loc[a2, a1] = result['Corrected p-value']  # Mirror value
                
                # Create heatmap
                fig = ff.create_annotated_heatmap(
                    z=p_matrix.values.tolist(),
                    x=p_matrix.columns.tolist(),
                    y=p_matrix.index.tolist(),
                    annotation_text=[[f"{val:.3f}" if val is not None else "" for val in row] 
                                     for row in p_matrix.values],
                    colorscale='YlGnBu_r',
                    showscale=True
                )
                
                fig.update_layout(
                    title="Corrected P-values (Lower values indicate stronger evidence of difference)",
                    height=500,
                    margin=dict(l=50, r=20, t=80, b=50),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation of results
                significant_pairs = [(r['Algorithm 1'], r['Algorithm 2']) for r in results if r['Significant']]
                if significant_pairs:
                    st.markdown(f"""
                    ### Significant Differences Detected
                    
                    The following algorithm pairs show statistically significant differences in {selected_test_metric}:
                    """)
                    for a1, a2 in significant_pairs:
                        mean1 = np.mean(algo_values[a1])
                        mean2 = np.mean(algo_values[a2])
                        better = a1 if mean1 < mean2 else a2
                        if selected_test_metric in ['fitness', 'latency', 'energy']:  # Lower is better
                            st.markdown(f"- **{a1}** vs **{a2}**: {better} performs better")
                        else:  # Higher is better
                            better = a1 if mean1 > mean2 else a2
                            st.markdown(f"- **{a1}** vs **{a2}**: {better} performs better")
                else:
                    st.info("No statistically significant differences were found between any algorithm pairs after correction for multiple comparisons.")
            else:
                st.warning("No valid test results to display.")
        else:
            st.warning("Pairwise tests require at least 2 algorithms.")

# -----------------------------------------------------------------------------
# Conclusions Section
# -----------------------------------------------------------------------------
if st.session_state.all_results:
    st.header("Conclusions and Recommendations")
    
    # Calculate average ranks
    ranks = {}
    
    for metric in selected_metrics:
        metric_ranks = {}
        for algo in st.session_state.all_results:
            # Get mean values for each metric (use non-normalized for fairness)
            mean_val = np.mean(st.session_state.all_results[algo][metric])
            metric_ranks[algo] = mean_val
        
        # Sort algorithms based on the metric (lower is better for most metrics)
        sorted_algos = sorted(metric_ranks.items(), key=lambda x: x[1])
        
        # Assign ranks
        for rank, (algo, _) in enumerate(sorted_algos):
            if algo not in ranks:
                ranks[algo] = {}
            ranks[algo][metric] = rank + 1  # +1 because ranks start at 1, not 0
    
    # Calculate average rank across metrics
    avg_ranks = {}
    for algo, metric_ranks in ranks.items():
        avg_ranks[algo] = sum(metric_ranks.values()) / len(metric_ranks)
    
    # Sort algorithms by average rank
    sorted_avg_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    # Display ranking as a horizontal bar chart
    st.subheader("Algorithm Rankings")
    
    fig = px.bar(
        x=[rank for _, rank in sorted_avg_ranks],
        y=[algo for algo, _ in sorted_avg_ranks],
        orientation='h',
        color=[algo for algo, _ in sorted_avg_ranks],
        color_discrete_sequence=get_color_palette(len(sorted_avg_ranks)),
        labels={"x": "Average Rank (Lower is Better)", "y": "Algorithm"},
        title="Overall Algorithm Performance Ranking"
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=50, r=20, t=80, b=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate recommendations
    best_algo = sorted_avg_ranks[0][0]
    worst_algo = sorted_avg_ranks[-1][0]
    
    st.markdown(f"""
    ### Key Findings
    
    Based on the experiments across {num_trials} trials:
    
    1. **{best_algo}** achieves the best overall performance ranking.
    
    2. The performance gap between **{best_algo}** and **{worst_algo}** is substantial.
    
    3. For latency-critical applications, **{sorted(ranks.items(), key=lambda x: x[1].get('latency', float('inf')))[0][0]}** 
       shows the best performance.
    
    4. For energy efficiency, **{sorted(ranks.items(), key=lambda x: x[1].get('energy', float('inf')))[0][0]}** 
       is the recommended choice.
    
    5. Statistical analysis {"supports" if len([r for r in results if r['Significant']]) > 0 else "does not strongly support"} 
       the conclusion that there are significant differences between the algorithms.
    """)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Edge Computing Algorithm Comparison Dashboard • Created with Streamlit</p>
    </div>
""", unsafe_allow_html=True)