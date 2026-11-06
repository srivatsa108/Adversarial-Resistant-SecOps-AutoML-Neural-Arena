"""Math-to-English translation helpers and Threat Severity Badging for SecOps Mode."""

from typing import Any, Dict, List, Tuple


def get_threat_badge(score: float, metric_type: str = "risk", theme: str = "dark") -> Dict[str, str]:
    """Translate numerical risk or performance scores into restrained SecOps badges.

    Args:
        score: Normalized metric value (0.0 to 1.0 or 0 to 100).
        metric_type: 'risk', 'hackability', 'confidence', or 'evasion_resistance'.
        theme: 'dark' or 'light'.

    Returns:
        Dict with 'level' (Critical/High/Medium/Low), 'color', 'icon', 'badge_html'.
    """
    # Normalize score to 0.0 - 1.0
    val = score / 100.0 if score > 1.0 else score
    is_dark = theme.lower() == "dark"

    if metric_type in ("risk", "hackability"):
        if val >= 0.75:
            level = "Critical"
            color = "#E5484D" if is_dark else "#DC2626"
            bg = "rgba(229, 72, 77, 0.12)" if is_dark else "rgba(220, 38, 38, 0.08)"
            border = "rgba(229, 72, 77, 0.3)" if is_dark else "rgba(220, 38, 38, 0.25)"
        elif val >= 0.50:
            level = "High risk"
            color = "#F5A524" if is_dark else "#D97706"
            bg = "rgba(245, 165, 36, 0.12)" if is_dark else "rgba(217, 119, 6, 0.08)"
            border = "rgba(245, 165, 36, 0.3)" if is_dark else "rgba(217, 119, 6, 0.25)"
        elif val >= 0.25:
            level = "Moderate risk"
            color = "#5B9DF5" if is_dark else "#2563EB"
            bg = "rgba(91, 157, 245, 0.12)" if is_dark else "rgba(37, 99, 235, 0.08)"
            border = "rgba(91, 157, 245, 0.3)" if is_dark else "rgba(37, 99, 235, 0.25)"
        else:
            level = "Low risk"
            color = "#46A758" if is_dark else "#16A34A"
            bg = "rgba(70, 167, 88, 0.12)" if is_dark else "rgba(22, 163, 74, 0.08)"
            border = "rgba(70, 167, 88, 0.3)" if is_dark else "rgba(22, 163, 74, 0.25)"
    else:  # evasion_resistance or confidence (higher is safer)
        if val >= 0.85:
            level = "Optimal defense"
            color = "#46A758" if is_dark else "#16A34A"
            bg = "rgba(70, 167, 88, 0.12)" if is_dark else "rgba(22, 163, 74, 0.08)"
            border = "rgba(70, 167, 88, 0.3)" if is_dark else "rgba(22, 163, 74, 0.25)"
        elif val >= 0.70:
            level = "Moderate defense"
            color = "#5B9DF5" if is_dark else "#2563EB"
            bg = "rgba(91, 157, 245, 0.12)" if is_dark else "rgba(37, 99, 235, 0.08)"
            border = "rgba(91, 157, 245, 0.3)" if is_dark else "rgba(37, 99, 235, 0.25)"
        elif val >= 0.50:
            level = "Elevated risk"
            color = "#F5A524" if is_dark else "#D97706"
            bg = "rgba(245, 165, 36, 0.12)" if is_dark else "rgba(217, 119, 6, 0.08)"
            border = "rgba(245, 165, 36, 0.3)" if is_dark else "rgba(217, 119, 6, 0.25)"
        else:
            level = "Vulnerable"
            color = "#E5484D" if is_dark else "#DC2626"
            bg = "rgba(229, 72, 77, 0.12)" if is_dark else "rgba(220, 38, 38, 0.08)"
            border = "rgba(229, 72, 77, 0.3)" if is_dark else "rgba(220, 38, 38, 0.25)"

    badge_html = (
        f'<span style="background-color: {bg}; color: {color}; '
        f'border: 1px solid {border}; padding: 3px 10px; border-radius: 4px; '
        f'font-weight: 500; font-size: 13px; line-height: 16px; display: inline-block;">'
        f'{level}</span>'
    )

    return {
        "level": level,
        "color": color,
        "icon": "",
        "badge_html": badge_html,
    }


def translate_metrics_to_secops(
    metrics: Dict[str, float],
    problem_type: str = "classification",
    theme: str = "dark",
) -> Dict[str, Any]:
    """Translate raw machine learning mathematics into plain-English SecOps metrics."""
    secops_view: Dict[str, Any] = {}

    if problem_type == "classification":
        f1 = metrics.get("f1", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)

        detection_rate_pct = rec * 100.0
        false_alarm_rate_pct = (1.0 - prec) * 100.0
        overall_reliability_pct = f1 * 100.0

        secops_view["headline"] = f"{overall_reliability_pct:.1f}% catch rate"
        secops_view["detection_rate"] = f"{detection_rate_pct:.1f}%"
        secops_view["false_positive_risk"] = f"{false_alarm_rate_pct:.1f}%"
        secops_view["operational_readiness"] = "Operational" if overall_reliability_pct >= 85 else "Needs training"
        secops_view["badge"] = get_threat_badge(overall_reliability_pct, "confidence", theme=theme)
        secops_view["plain_summary"] = (
            f"The detection model identifies {detection_rate_pct:.1f}% of validated attacks "
            f"with a {false_alarm_rate_pct:.1f}% false-positive rate on background telemetry."
        )

    elif problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        mae = metrics.get("mae", 0.0)
        rmse = metrics.get("rmse", 0.0)

        confidence_pct = max(0.0, min(100.0, r2 * 100.0))
        secops_view["headline"] = f"{confidence_pct:.1f}% prediction confidence"
        secops_view["average_deviation"] = f"±{mae:.2f} units"
        secops_view["peak_uncertainty"] = f"±{rmse:.2f} units"
        secops_view["badge"] = get_threat_badge(confidence_pct, "confidence", theme=theme)
        secops_view["plain_summary"] = (
            f"The risk score model captures {confidence_pct:.1f}% of network baseline trends, "
            f"with an average severity deviation of ±{mae:.2f} points."
        )

    elif problem_type == "unsupervised":
        anomalies_found = int(metrics.get("anomaly_count", 0))
        total_logs = int(metrics.get("total_logs", 0))
        contamination = metrics.get("contamination", 0.05) * 100.0
        mean_deviation = metrics.get("mean_deviation_pct", 100.0)

        secops_view["headline"] = f"{anomalies_found} zero-day outliers identified"
        secops_view["anomaly_rate"] = f"{contamination:.1f}% of ingress traffic"
        secops_view["outlier_deviation"] = f"+{mean_deviation:.1f}% baseline deviation"
        secops_view["badge"] = get_threat_badge(mean_deviation, "risk", theme=theme)
        secops_view["plain_summary"] = (
            f"The deep autoencoder flagged {anomalies_found} unusual log patterns deviating by "
            f"+{mean_deviation:.1f}% from baseline traffic, indicating potential zero-day activity."
        )

    return secops_view


def translate_shap_to_secops_root_causes(
    feature_importances: List[Dict[str, Any]],
    top_n: int = 4,
) -> str:
    """Format top SHAP feature attributions into plain-English root causes."""
    if not feature_importances:
        return "Model decision matches baseline distribution."

    sorted_features = sorted(feature_importances, key=lambda x: x.get("importance", 0.0), reverse=True)[:top_n]
    total_imp = sum(f.get("importance", 0.0) for f in sorted_features)
    if total_imp == 0:
        total_imp = 1.0

    narrative_tokens = []
    for f in sorted_features:
        name = f.get("feature", "unknown_attribute").replace("_", " ").lower()
        pct = (f.get("importance", 0.0) / total_imp) * 100.0
        narrative_tokens.append(f"{name} ({pct:.1f}% impact)")

    return f"Flagged primarily by {', '.join(narrative_tokens)}."


def translate_adversarial_to_secops_warning(
    resistance_score: float,
    drop_pct: float,
    tested_epsilon: float,
    theme: str = "dark",
) -> Tuple[Dict[str, str], str]:
    """Generate calm, precise SecOps advisory on adversarial robustness."""
    badge = get_threat_badge(resistance_score, "evasion_resistance", theme=theme)

    if resistance_score >= 80.0:
        advice = (
            f"Defensive model demonstrates {resistance_score:.1f}% evasion resistance. "
            f"Perturbations up to {tested_epsilon * 100:.0f}% feature noise produced "
            f"{drop_pct:.1f}% degradation. Boundary remains resilient."
        )
    elif resistance_score >= 50.0:
        advice = (
            f"Defensive model shows moderate resilience ({resistance_score:.1f}%). "
            f"Feature noise reduced detection accuracy by {drop_pct:.1f}%. "
            f"Adversarial retraining recommended."
        )
    else:
        advice = (
            f"Critical evasion vulnerability: Evasion resistance is {resistance_score:.1f}%. "
            f"Feature mutations ({tested_epsilon * 100:.0f}% noise) bypass detection in "
            f"{drop_pct:.1f}% of test vectors. Retraining and rate limiting required."
        )

    return badge, advice
