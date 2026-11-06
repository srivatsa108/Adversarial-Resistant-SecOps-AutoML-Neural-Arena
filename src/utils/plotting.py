"""Theme-aware Plotly visualizations supporting light and dark themes."""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go


def get_theme_tokens(theme: str = "dark") -> Dict[str, str]:
    """Retrieve theme-specific palette adhering to WCAG AA contrast."""
    if theme.lower() == "light":
        return {
            "paper_bg": "#F7F7F8",
            "plot_bg": "#FFFFFF",
            "text": "#0A0A0A",
            "muted": "#52525B",
            "grid": "#E5E5E7",
            "border": "#D4D4D8",
            "accent": "#2F6BFF",
            "accent_fill": "rgba(47, 107, 255, 0.10)",
            "danger": "#DC2626",
            "success": "#16A34A",
            "warning": "#D97706",
            "font_family": "Geist, sans-serif",
            "mono_family": "Geist Mono, monospace",
        }
    return {
        "paper_bg": "#0A0A0A",
        "plot_bg": "#141414",
        "text": "#EDEDED",
        "muted": "#A1A1AA",
        "grid": "#262626",
        "border": "#333333",
        "accent": "#5B9DF5",
        "accent_fill": "rgba(91, 157, 245, 0.12)",
        "danger": "#E5484D",
        "success": "#46A758",
        "warning": "#F5A524",
        "font_family": "Geist, sans-serif",
        "mono_family": "Geist Mono, monospace",
    }


def apply_theme_layout(
    fig: go.Figure,
    title: str,
    height: int = 380,
    theme: str = "dark",
) -> go.Figure:
    """Apply unified theme styling to Plotly figures."""
    t = get_theme_tokens(theme)
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 14, "color": t["text"], "family": t["font_family"]},
        },
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font={"color": t["text"], "family": t["font_family"]},
        height=height,
        margin={"l": 40, "r": 30, "t": 55, "b": 35},
        xaxis={
            "gridcolor": t["grid"],
            "zerolinecolor": t["grid"],
            "showgrid": True,
            "tickfont": {"color": t["muted"], "size": 11, "family": t["mono_family"]},
        },
        yaxis={
            "gridcolor": t["grid"],
            "zerolinecolor": t["grid"],
            "showgrid": True,
            "tickfont": {"color": t["muted"], "size": 11, "family": t["mono_family"]},
        },
        legend={
            "bgcolor": t["plot_bg"],
            "bordercolor": t["border"],
            "borderwidth": 1,
            "font": {"size": 11, "color": t["muted"]},
        },
    )
    return fig


# Backwards compatibility alias
def apply_secops_layout(fig: go.Figure, title: str, height: int = 420, theme: str = "dark") -> go.Figure:
    return apply_theme_layout(fig, title, height=height, theme=theme)


def create_model_comparison_radar_chart(
    model_metrics: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
    theme: str = "dark",
) -> go.Figure:
    """Generate a multi-model radar chart comparing evaluation metrics."""
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    t = get_theme_tokens(theme)
    fig = go.Figure()

    palette = [t["accent"], t["success"], t["warning"], t["danger"], t["muted"]]
    categories = [k.replace("_", " ").upper() for k in metric_keys]
    radar_categories = categories + [categories[0]]

    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        color = palette[idx % len(palette)]
        values = [metrics.get(k, 0.0) for k in metric_keys]
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_categories,
                fill="toself",
                name=model_name,
                line={"color": color, "width": 1.5},
            )
        )

    fig.update_layout(
        polar={
            "bgcolor": t["plot_bg"],
            "radialaxis": {
                "visible": True,
                "range": [0, 1.0],
                "gridcolor": t["grid"],
                "linecolor": t["grid"],
                "tickfont": {"color": t["muted"], "size": 9, "family": t["mono_family"]},
            },
            "angularaxis": {
                "gridcolor": t["grid"],
                "linecolor": t["grid"],
                "tickfont": {"color": t["text"], "size": 11},
            },
        },
    )

    return apply_theme_layout(fig, "Model comparison benchmark", height=420, theme=theme)


def create_neural_loss_curve(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    architecture_name: str = "Deep MLP Neural Network",
    theme: str = "dark",
) -> go.Figure:
    """Plot training and validation loss curves."""
    t = get_theme_tokens(theme)
    fig = go.Figure()
    epochs = list(range(1, len(train_loss) + 1))

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines",
            name="Training loss",
            line={"color": t["accent"], "width": 2},
        )
    )

    if val_loss and len(val_loss) == len(train_loss):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_loss,
                mode="lines",
                name="Validation loss",
                line={"color": t["danger"], "width": 1.5, "dash": "dash"},
            )
        )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    return apply_theme_layout(fig, f"Loss convergence: {architecture_name}", height=360, theme=theme)


def create_adversarial_decay_curve(
    decay_points: List[Dict[str, Any]],
    theme: str = "dark",
) -> go.Figure:
    """Plot adversarial robustness decay across perturbation magnitude."""
    t = get_theme_tokens(theme)
    fig = go.Figure()

    epsilons = [p["epsilon"] for p in decay_points]
    scores = [p["score"] * 100.0 for p in decay_points]
    drops = [p.get("accuracy_drop", 0.0) * 100.0 for p in decay_points]

    fig.add_trace(
        go.Scatter(
            x=epsilons,
            y=scores,
            mode="lines+markers",
            name="Evasion resistance (%)",
            line={"color": t["accent"], "width": 2},
            marker={"size": 6, "color": t["accent"]},
            fill="tozeroy",
            fillcolor=t["accent_fill"],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epsilons,
            y=drops,
            mode="lines",
            name="Evasion gain (%)",
            line={"color": t["danger"], "width": 1.5, "dash": "dot"},
        )
    )

    fig.update_xaxes(title_text="Perturbation epsilon (ε)")
    fig.update_yaxes(title_text="Retention percentage", range=[0, 105])
    return apply_theme_layout(fig, "Adversarial evasion resistance decay", height=380, theme=theme)


def create_shap_feature_importance_bar(
    features: List[str],
    importances: List[float],
    title: str = "Feature attribution ranking",
    theme: str = "dark",
) -> go.Figure:
    """Generate horizontal bar chart of top SHAP feature contributions."""
    t = get_theme_tokens(theme)
    fig = go.Figure()

    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1])
    sorted_feat = [p[0] for p in sorted_pairs]
    sorted_val = [p[1] for p in sorted_pairs]

    fig.add_trace(
        go.Bar(
            x=sorted_val,
            y=sorted_feat,
            orientation="h",
            marker={"color": t["accent"]},
        )
    )

    fig.update_xaxes(title_text="Mean absolute attribution")
    fig.update_yaxes(title_text="Feature")
    return apply_theme_layout(fig, title, height=max(320, len(features) * 26), theme=theme)


def create_anomaly_reconstruction_plot(
    errors: np.ndarray,
    threshold: float,
    anomaly_indices: List[int],
    theme: str = "dark",
) -> go.Figure:
    """Plot Autoencoder reconstruction error distribution and threshold cutoff."""
    t = get_theme_tokens(theme)
    fig = go.Figure()
    indices = np.arange(len(errors))
    is_anomaly = np.isin(indices, anomaly_indices)

    fig.add_trace(
        go.Scatter(
            x=indices[~is_anomaly],
            y=errors[~is_anomaly],
            mode="markers",
            name="Baseline traffic",
            marker={"size": 4, "color": t["accent"], "opacity": 0.6},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=indices[is_anomaly],
            y=errors[is_anomaly],
            mode="markers",
            name="Zero-day outlier",
            marker={"size": 6, "color": t["danger"]},
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=t["warning"],
        annotation_text=f"Cutoff ({threshold:.4f})",
        annotation_position="top left",
        annotation_font_color=t["warning"],
        annotation_font_size=10,
    )

    fig.update_xaxes(title_text="Telemetry record index")
    fig.update_yaxes(title_text="Reconstruction error (MSE)")
    return apply_theme_layout(fig, "Reconstruction error distribution", height=380, theme=theme)


def create_verification_latency_chart(
    test_results: List[Dict[str, Any]],
    theme: str = "dark",
) -> go.Figure:
    """Plot verification test suite latencies with pass/fail indicators."""
    t = get_theme_tokens(theme)
    fig = go.Figure()

    ids = [r["id"] for r in test_results]
    names = [r["name"] for r in test_results]
    durations = [r["duration_ms"] for r in test_results]
    colors = [t["success"] if r["status"] == "PASSED" else t["danger"] for r in test_results]

    fig.add_trace(
        go.Bar(
            x=durations,
            y=ids,
            orientation="h",
            text=[f"{d:.1f} ms" for d in durations],
            textposition="auto",
            textfont={"family": t["mono_family"], "size": 11, "color": "#FFFFFF"},
            hovertext=names,
            marker={"color": colors, "line": {"color": t["border"], "width": 1}},
        )
    )

    fig.update_xaxes(title_text="Execution latency (ms)", type="log")
    fig.update_yaxes(autorange="reversed")
    return apply_theme_layout(fig, "System invariant verification latency (log ms)", height=380, theme=theme)


def create_telemetry_distribution_plot(
    feature_counts: Dict[str, int],
    title: str = "Telemetry distribution breakdown",
    theme: str = "dark",
) -> go.Figure:
    """Plot distribution of telemetry categories or protocols."""
    t = get_theme_tokens(theme)
    fig = go.Figure()

    labels = list(feature_counts.keys())
    values = list(feature_counts.values())

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": t["accent"], "line": {"color": t["border"], "width": 1}},
            text=values,
            textposition="auto",
            textfont={"family": t["mono_family"], "size": 11, "color": "#FFFFFF"},
        )
    )

    fig.update_xaxes(title_text="Category / Protocol")
    fig.update_yaxes(title_text="Sample count")
    return apply_theme_layout(fig, title, height=320, theme=theme)

