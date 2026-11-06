"""Interactive dark-theme Plotly visualizations for SecOps and Data Science modes."""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# SecOps Cyber Color Palette
DARK_BG = "#0b0f19"
PANEL_BG = "#111927"
CYBER_NEON_TEAL = "#00ffcc"
CYBER_NEON_BLUE = "#00e5ff"
CYBER_NEON_PINK = "#ff0055"
CYBER_NEON_PURPLE = "#7928ca"
CYBER_NEON_AMBER = "#ffaa00"
GRID_COLOR = "rgba(255, 255, 255, 0.07)"
TEXT_COLOR = "#e2e8f0"


def apply_secops_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply uniform SecOps dark-mode styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": CYBER_NEON_BLUE, "family": "Inter, sans-serif"},
        },
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "showgrid": True,
        },
        yaxis={
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "showgrid": True,
        },
        legend={
            "bgcolor": "rgba(17, 25, 39, 0.8)",
            "bordercolor": "rgba(0, 229, 255, 0.2)",
            "borderwidth": 1,
            "font": {"size": 11, "color": TEXT_COLOR},
        },
    )
    return fig


def create_model_comparison_radar_chart(
    model_metrics: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
) -> go.Figure:
    """Generate a multi-model radar chart comparing performance metrics."""
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig = go.Figure()
    palette = [CYBER_NEON_TEAL, CYBER_NEON_BLUE, CYBER_NEON_PINK, CYBER_NEON_PURPLE, CYBER_NEON_AMBER]

    # Categories capitalized
    categories = [k.replace("_", " ").upper() for k in metric_keys]
    # Close radar loop
    radar_categories = categories + [categories[0]]

    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        color = palette[idx % len(palette)]
        values = [metrics.get(k, 0.0) for k in metric_keys]
        values.append(values[0])  # Close the loop

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_categories,
                fill="toself",
                name=model_name,
                line={"color": color, "width": 2},
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
            )
        )

    fig.update_layout(
        polar={
            "bgcolor": PANEL_BG,
            "radialaxis": {
                "visible": True,
                "range": [0, 1.0],
                "gridcolor": GRID_COLOR,
                "linecolor": GRID_COLOR,
                "tickfont": {"color": "#94a3b8", "size": 9},
            },
            "angularaxis": {
                "gridcolor": GRID_COLOR,
                "linecolor": GRID_COLOR,
                "tickfont": {"color": TEXT_COLOR, "size": 11, "family": "Inter, sans-serif"},
            },
        },
    )

    return apply_secops_layout(fig, "🧠 Neural & Model Arena Efficacy Benchmark", height=450)


def create_neural_loss_curve(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    architecture_name: str = "Deep MLP Neural Network",
) -> go.Figure:
    """Plot epoch-by-epoch loss convergence curves for Neural Networks."""
    fig = go.Figure()
    epochs = list(range(1, len(train_loss) + 1))

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines+markers",
            name="Training Loss (MSE/BCE)",
            line={"color": CYBER_NEON_TEAL, "width": 2.5},
            marker={"size": 5, "color": CYBER_NEON_TEAL},
        )
    )

    if val_loss and len(val_loss) == len(train_loss):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_loss,
                mode="lines+markers",
                name="Validation Loss",
                line={"color": CYBER_NEON_PINK, "width": 2.5, "dash": "dash"},
                marker={"size": 5, "color": CYBER_NEON_PINK},
            )
        )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    return apply_secops_layout(fig, f"📈 Loss Convergence: {architecture_name}", height=380)


def create_adversarial_decay_curve(decay_points: List[Dict[str, Any]]) -> go.Figure:
    """Plot adversarial robustness decay curve across perturbation epsilons."""
    fig = go.Figure()

    epsilons = [p["epsilon"] for p in decay_points]
    scores = [p["score"] * 100.0 for p in decay_points]
    drops = [p.get("accuracy_drop", 0.0) * 100.0 for p in decay_points]

    fig.add_trace(
        go.Scatter(
            x=epsilons,
            y=scores,
            mode="lines+markers",
            name="Model Evasion Resistance (%)",
            line={"color": CYBER_NEON_BLUE, "width": 3},
            marker={"size": 8, "symbol": "diamond", "color": CYBER_NEON_TEAL},
            fill="tozeroy",
            fillcolor="rgba(0, 229, 255, 0.1)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epsilons,
            y=drops,
            mode="lines+markers",
            name="Adversary Efficacy Gain / Degradation (%)",
            line={"color": CYBER_NEON_PINK, "width": 2.5, "dash": "dot"},
            marker={"size": 6, "color": CYBER_NEON_PINK},
        )
    )

    fig.update_xaxes(title_text="Perturbation Magnitude (Epsilon ε)")
    fig.update_yaxes(title_text="Percentage (%)", range=[0, 105])
    return apply_secops_layout(fig, "🛡️ Adversarial Stress Test: Robustness Decay Curve", height=400)


def create_shap_feature_importance_bar(
    features: List[str],
    importances: List[float],
    title: str = "🔍 XAI Feature Contribution (SHAP / Sensitivity)",
) -> go.Figure:
    """Generate horizontal bar chart of top SHAP feature importances."""
    fig = go.Figure()

    # Sort descending
    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1])
    sorted_feat = [p[0] for p in sorted_pairs]
    sorted_val = [p[1] for p in sorted_pairs]

    fig.add_trace(
        go.Bar(
            x=sorted_val,
            y=sorted_feat,
            orientation="h",
            marker={
                "color": sorted_val,
                "colorscale": [[0, CYBER_NEON_BLUE], [1, CYBER_NEON_PINK]],
                "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
            },
        )
    )

    fig.update_xaxes(title_text="Mean |SHAP Value| / Importance Weight")
    fig.update_yaxes(title_text="Log Feature")
    return apply_secops_layout(fig, title, height=max(360, len(features) * 28))


def create_anomaly_reconstruction_plot(
    errors: np.ndarray,
    threshold: float,
    anomaly_indices: List[int],
) -> go.Figure:
    """Plot Autoencoder reconstruction error distribution and threshold cutoff."""
    fig = go.Figure()
    indices = np.arange(len(errors))
    is_anomaly = np.isin(indices, anomaly_indices)

    # Normal samples
    fig.add_trace(
        go.Scatter(
            x=indices[~is_anomaly],
            y=errors[~is_anomaly],
            mode="markers",
            name="Normal Network Traffic",
            marker={"size": 5, "color": CYBER_NEON_TEAL, "opacity": 0.7},
        )
    )

    # Anomaly samples
    fig.add_trace(
        go.Scatter(
            x=indices[is_anomaly],
            y=errors[is_anomaly],
            mode="markers",
            name="🚨 Zero-Day / High-Risk Anomaly",
            marker={"size": 8, "symbol": "cross", "color": CYBER_NEON_PINK},
        )
    )

    # Cutoff threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=CYBER_NEON_AMBER,
        annotation_text=f"Anomaly Threshold ({threshold:.4f})",
        annotation_position="top left",
        annotation_font_color=CYBER_NEON_AMBER,
    )

    fig.update_xaxes(title_text="Log Event Sequence Index")
    fig.update_yaxes(title_text="Reconstruction Error (MSE)")
    return apply_secops_layout(fig, "🔬 Deep Autoencoder Reconstruction Anomaly Distribution", height=420)
