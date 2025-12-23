"""
Web-based Monitoring Dashboard

Streamlit-based interactive dashboard for visualizing reward signals and training metrics.

Run with: streamlit run src/monitoring/web_dashboard.py -- --data-dir ./monitoring_data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
import sys


def load_monitoring_data(data_dir: str) -> Dict[str, Any]:
    """Load monitoring data from JSON files."""
    data_path = Path(data_dir)

    if not data_path.exists():
        return {'steps': [], 'metrics': []}

    # Load summary
    summary_file = data_path / "summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

    # Load step data
    step_files = sorted(data_path.glob("step_*.json"))

    steps = []
    metrics_data = []

    for step_file in step_files:
        with open(step_file, 'r') as f:
            step_data = json.load(f)
            steps.append(step_data['step'])
            metrics_data.append(step_data['metrics'])

    return {
        'steps': steps,
        'metrics': metrics_data,
        'summary': summary,
    }


def create_reward_overview_chart(steps: List[int], metrics: List[Dict]) -> go.Figure:
    """Create overview chart of total rewards."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean Reward Over Time', 'Reward Distribution',
                       'Signal-to-Noise Ratio', 'Reward Variance'),
        specs=[[{'secondary_y': False}, {'type': 'box'}],
               [{'secondary_y': False}, {'secondary_y': False}]],
    )

    # Extract data
    mean_rewards = [m.get('rewards/mean', 0) for m in metrics]
    std_rewards = [m.get('rewards/std', 0) for m in metrics]
    snr = [m.get('rewards/signal_to_noise', 0) for m in metrics]
    variance = [m.get('rewards/variance', 0) for m in metrics]

    # Mean reward with std bands
    fig.add_trace(
        go.Scatter(x=steps, y=mean_rewards, name='Mean Reward',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Add std bands
    upper_bound = [m + s for m, s in zip(mean_rewards, std_rewards)]
    lower_bound = [m - s for m, s in zip(mean_rewards, std_rewards)]

    fig.add_trace(
        go.Scatter(x=steps, y=upper_bound, fill=None, mode='lines',
                  line=dict(width=0), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=steps, y=lower_bound, fill='tonexty', mode='lines',
                  line=dict(width=0), name='±1 Std Dev',
                  fillcolor='rgba(0, 0, 255, 0.2)'),
        row=1, col=1
    )

    # Box plot of reward distribution
    fig.add_trace(
        go.Box(y=mean_rewards, name='Reward Distribution'),
        row=1, col=2
    )

    # Signal-to-noise ratio
    fig.add_trace(
        go.Scatter(x=steps, y=snr, name='SNR',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )

    # Variance
    fig.add_trace(
        go.Scatter(x=steps, y=variance, name='Variance',
                  line=dict(color='orange', width=2)),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=True, title_text="Reward Signal Overview")

    return fig


def create_function_breakdown_chart(steps: List[int], metrics: List[Dict]) -> go.Figure:
    """Create per-function reward breakdown chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Reward Functions Over Time', 'Reward Function Contributions (%)'),
        specs=[[{'secondary_y': False}], [{'type': 'bar'}]],
        row_heights=[0.6, 0.4],
    )

    # Identify all reward functions
    function_names = set()
    for m in metrics:
        for key in m.keys():
            if key.startswith('rewards/functions/') and not key.endswith('_std'):
                func_name = key.replace('rewards/functions/', '')
                function_names.add(func_name)

    function_names = sorted(function_names)

    # Extract per-function data
    colors = px.colors.qualitative.Plotly

    for i, func_name in enumerate(function_names):
        values = [m.get(f'rewards/functions/{func_name}', 0) for m in metrics]

        fig.add_trace(
            go.Scatter(x=steps, y=values, name=func_name,
                      line=dict(color=colors[i % len(colors)], width=2),
                      mode='lines+markers'),
            row=1, col=1
        )

    # Contributions (use latest step)
    if metrics:
        latest_metrics = metrics[-1]
        contributions = {}

        for key, value in latest_metrics.items():
            if key.startswith('contributions/'):
                func_name = key.replace('contributions/', '')
                contributions[func_name] = value

        if contributions:
            fig.add_trace(
                go.Bar(x=list(contributions.keys()), y=list(contributions.values()),
                      name='Contribution %',
                      marker=dict(color=colors[:len(contributions)])),
                row=2, col=1
            )

    fig.update_layout(height=900, showlegend=True, title_text="Reward Function Analysis")
    fig.update_yaxes(title_text="Reward Value", row=1, col=1)
    fig.update_yaxes(title_text="Contribution (%)", row=2, col=1)

    return fig


def create_training_dynamics_chart(steps: List[int], metrics: List[Dict]) -> go.Figure:
    """Create training dynamics chart (KL, entropy, advantages)."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('KL Divergence', 'Policy Entropy',
                       'Advantage Mean', 'Advantage Std'),
    )

    # Extract data
    kl_div = [m.get('training/kl_divergence', None) for m in metrics]
    policy_ent = [m.get('training/policy_entropy', None) for m in metrics]
    adv_mean = [m.get('training/advantage_mean', None) for m in metrics]
    adv_std = [m.get('training/advantage_std', None) for m in metrics]

    # Filter out None values
    kl_steps = [s for s, v in zip(steps, kl_div) if v is not None]
    kl_values = [v for v in kl_div if v is not None]

    ent_steps = [s for s, v in zip(steps, policy_ent) if v is not None]
    ent_values = [v for v in policy_ent if v is not None]

    adv_mean_steps = [s for s, v in zip(steps, adv_mean) if v is not None]
    adv_mean_values = [v for v in adv_mean if v is not None]

    adv_std_steps = [s for s, v in zip(steps, adv_std) if v is not None]
    adv_std_values = [v for v in adv_std if v is not None]

    # Add traces
    if kl_values:
        fig.add_trace(
            go.Scatter(x=kl_steps, y=kl_values, name='KL Divergence',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )

    if ent_values:
        fig.add_trace(
            go.Scatter(x=ent_steps, y=ent_values, name='Policy Entropy',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )

    if adv_mean_values:
        fig.add_trace(
            go.Scatter(x=adv_mean_steps, y=adv_mean_values, name='Advantage Mean',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )

    if adv_std_values:
        fig.add_trace(
            go.Scatter(x=adv_std_steps, y=adv_std_values, name='Advantage Std',
                      line=dict(color='orange', width=2)),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=True, title_text="Training Dynamics")

    return fig


def create_quality_metrics_chart(steps: List[int], metrics: List[Dict]) -> go.Figure:
    """Create quality metrics chart."""
    fig = go.Figure()

    # Extract quality metrics
    snr = [m.get('rewards/signal_to_noise', 0) for m in metrics]
    entropy = [m.get('rewards/entropy', 0) for m in metrics]
    variance = [m.get('rewards/variance', 0) for m in metrics]

    # Normalize variance for visualization
    if max(variance) > 0:
        variance_normalized = [v / max(variance) * max(snr) for v in variance]
    else:
        variance_normalized = variance

    fig.add_trace(
        go.Scatter(x=steps, y=snr, name='Signal-to-Noise Ratio',
                  line=dict(color='green', width=2))
    )

    fig.add_trace(
        go.Scatter(x=steps, y=entropy, name='Reward Entropy',
                  line=dict(color='blue', width=2))
    )

    fig.add_trace(
        go.Scatter(x=steps, y=variance_normalized, name='Variance (normalized)',
                  line=dict(color='orange', width=2, dash='dash'))
    )

    fig.update_layout(
        height=500,
        title_text="Reward Quality Metrics",
        xaxis_title="Training Step",
        yaxis_title="Metric Value",
        showlegend=True,
    )

    return fig


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="GRPO Reward Monitoring Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("🎯 GRPO Reward Signal Monitoring Dashboard")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Data directory selection
    data_dir = st.sidebar.text_input(
        "Monitoring Data Directory",
        value="./monitoring_data",
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=10,
    )

    if auto_refresh:
        st.sidebar.info(f"Auto-refreshing every {refresh_interval}s")
        # Note: Streamlit doesn't support true auto-refresh without external tools
        # This is a placeholder for future implementation

    # Load data
    with st.spinner("Loading monitoring data..."):
        data = load_monitoring_data(data_dir)

    if not data['steps']:
        st.error(f"No monitoring data found in {data_dir}")
        st.info("Start training with monitoring enabled to generate data.")
        return

    # Display summary metrics
    st.header("📈 Summary Statistics")

    if data.get('summary'):
        summary = data['summary']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Steps Tracked",
                summary.get('total_steps_tracked', 0),
            )

        with col2:
            current = summary.get('current_metrics', {})
            st.metric(
                "Current Mean Reward",
                f"{current.get('mean_reward', 0):.3f}",
            )

        with col3:
            st.metric(
                "Signal-to-Noise Ratio",
                f"{current.get('signal_to_noise', 0):.3f}",
            )

        with col4:
            st.metric(
                "Anomalies Detected",
                summary.get('anomalies_detected', 0),
            )

    # Main charts
    st.header("📊 Reward Signal Overview")
    fig_overview = create_reward_overview_chart(data['steps'], data['metrics'])
    st.plotly_chart(fig_overview, use_container_width=True)

    st.header("🔍 Reward Function Breakdown")
    fig_breakdown = create_function_breakdown_chart(data['steps'], data['metrics'])
    st.plotly_chart(fig_breakdown, use_container_width=True)

    st.header("🎓 Training Dynamics")
    fig_dynamics = create_training_dynamics_chart(data['steps'], data['metrics'])
    st.plotly_chart(fig_dynamics, use_container_width=True)

    st.header("✨ Quality Metrics")
    fig_quality = create_quality_metrics_chart(data['steps'], data['metrics'])
    st.plotly_chart(fig_quality, use_container_width=True)

    # Data table
    with st.expander("📋 View Raw Data"):
        if data['metrics']:
            df = pd.DataFrame(data['metrics'])
            df.insert(0, 'step', data['steps'])
            st.dataframe(df, use_container_width=True)

    # Summary details
    if data.get('summary'):
        with st.expander("📄 Detailed Summary"):
            st.json(data['summary'])

    # Footer
    st.markdown("---")
    st.markdown(
        "**GRPO Reward Monitoring Dashboard** | "
        "Built with Streamlit and Plotly | "
        f"Last updated: {len(data['steps'])} steps tracked"
    )


if __name__ == "__main__":
    main()
