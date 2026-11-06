"""Math-to-English translation helpers and Threat Severity Badging for SecOps Mode."""

from typing import Any, Dict, List, Tuple


def get_threat_badge(score: float, metric_type: str = "risk") -> Dict[str, str]:
    """Translate numerical risk or performance scores into SecOps Threat Badges.

    Args:
        score: Normalized metric value (0.0 to 1.0 or 0 to 100).
        metric_type: 'risk', 'hackability', 'confidence', or 'evasion_resistance'.

    Returns:
        Dict with 'level' (Critical/High/Medium/Low), 'color', 'icon', 'badge_html'.
    """
    # Normalize score to 0.0 - 1.0
    val = score / 100.0 if score > 1.0 else score

    if metric_type in ("risk", "hackability"):
        if val >= 0.75:
            level = "CRITICAL"
            color = "#ff0055"
            icon = "🚨"
        elif val >= 0.50:
            level = "HIGH"
            color = "#ff5500"
            icon = "⚠️"
        elif val >= 0.25:
            level = "MEDIUM"
            color = "#ffaa00"
            icon = "⚡"
        else:
            level = "LOW"
            color = "#00ffcc"
            icon = "🛡️"
    else:  # evasion_resistance or confidence (higher is safer)
        if val >= 0.85:
            level = "OPTIMAL / FORTIFIED"
            color = "#00ffcc"
            icon = "🛡️"
        elif val >= 0.70:
            level = "MODERATE DEFENSE"
            color = "#00e5ff"
            icon = "⚖️"
        elif val >= 0.50:
            level = "ELEVATED RISK"
            color = "#ffaa00"
            icon = "⚠️"
        else:
            level = "VULNERABLE / COMPROMISED"
            color = "#ff0055"
            icon = "🚨"

    badge_html = (
        f'<span style="background-color: {color}22; color: {color}; '
        f'border: 1px solid {color}; padding: 4px 12px; border-radius: 6px; '
        f'font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; display: inline-block;">'
        f'{icon} {level}</span>'
    )

    return {
        "level": level,
        "color": color,
        "icon": icon,
        "badge_html": badge_html,
    }


def translate_metrics_to_secops(
    metrics: Dict[str, float],
    problem_type: str = "classification",
) -> Dict[str, Any]:
    """Translate raw machine learning mathematics into plain-English SecOps metrics."""
    secops_view: Dict[str, Any] = {}

    if problem_type == "classification":
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)

        # Threat Detection Efficacy (derived from F1 & Recall)
        detection_rate_pct = rec * 100.0
        false_alarm_rate_pct = (1.0 - prec) * 100.0
        overall_reliability_pct = f1 * 100.0

        secops_view["headline"] = f"{overall_reliability_pct:.1f}% Threat Catch Rate"
        secops_view["detection_rate"] = f"{detection_rate_pct:.1f}%"
        secops_view["false_positive_risk"] = f"{false_alarm_rate_pct:.1f}%"
        secops_view["operational_readiness"] = "Combat Ready" if overall_reliability_pct >= 85 else "Needs Retraining"
        secops_view["badge"] = get_threat_badge(overall_reliability_pct, "confidence")
        secops_view["plain_summary"] = (
            f"The detection engine accurately flags {detection_rate_pct:.1f}% of genuine cyber threats "
            f"while sustaining a low {false_alarm_rate_pct:.1f}% false-positive alert burden on tier-1 analysts."
        )

    elif problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        mae = metrics.get("mae", 0.0)
        rmse = metrics.get("rmse", 0.0)

        # Variance Explained translated to Predictive Precision Confidence
        confidence_pct = max(0.0, min(100.0, r2 * 100.0))
        secops_view["headline"] = f"{confidence_pct:.1f}% Telemetry Predictive Confidence"
        secops_view["average_deviation"] = f"±{mae:.2f} units"
        secops_view["peak_uncertainty"] = f"±{rmse:.2f} units"
        secops_view["badge"] = get_threat_badge(confidence_pct, "confidence")
        secops_view["plain_summary"] = (
            f"The continuous risk scoring model captures {confidence_pct:.1f}% of network baseline trends, "
            f"with an average severity deviation of only ±{mae:.2f} points."
        )

    elif problem_type == "unsupervised":
        anomalies_found = int(metrics.get("anomaly_count", 0))
        total_logs = int(metrics.get("total_logs", 0))
        contamination = metrics.get("contamination", 0.05) * 100.0
        mean_deviation = metrics.get("mean_deviation_pct", 100.0)

        secops_view["headline"] = f"{anomalies_found} Zero-Day Log Outliers Flagged"
        secops_view["anomaly_rate"] = f"{contamination:.1f}% of ingress traffic"
        secops_view["outlier_deviation"] = f"+{mean_deviation:.1f}% over baseline"
        secops_view["badge"] = get_threat_badge(mean_deviation, "risk")
        secops_view["plain_summary"] = (
            f"The Deep Autoencoder isolated {anomalies_found} unusual log sequences deviating by "
            f"+{mean_deviation:.1f}% from standard traffic profiles, indicating potential zero-day exploit attempts."
        )

    return secops_view


def translate_shap_to_secops_root_causes(
    feature_importances: List[Dict[str, Any]],
    top_n: int = 4,
) -> str:
    """Format top SHAP feature attributions into plain-English cyber root causes."""
    if not feature_importances:
        return "Model decision driven by standard baseline distribution."

    sorted_features = sorted(feature_importances, key=lambda x: x.get("importance", 0.0), reverse=True)[:top_n]
    total_imp = sum(f.get("importance", 0.0) for f in sorted_features)
    if total_imp == 0:
        total_imp = 1.0

    narrative_tokens = []
    for f in sorted_features:
        name = f.get("feature", "unknown_attribute").replace("_", " ").title()
        pct = (f.get("importance", 0.0) / total_imp) * 100.0
        narrative_tokens.append(f"{name} [{pct:.1f}% impact]")

    return f"Flagged primarily due to anomalous {', '.join(narrative_tokens)}."


def translate_adversarial_to_secops_warning(
    resistance_score: float,
    drop_pct: float,
    tested_epsilon: float,
) -> Tuple[Dict[str, str], str]:
    """Generate SecOps advisory warning on adversarial robustness and evasion resistance."""
    badge = get_threat_badge(resistance_score, "evasion_resistance")

    if resistance_score >= 80.0:
        advice = (
            f"Defensive model demonstrates Fortified Evasion Resistance ({resistance_score:.1f}%). "
            f"Synthetic payload mutations up to {tested_epsilon * 100:.0f}% feature noise caused negligible "
            f"efficacy erosion ({drop_pct:.1f}% degradation). No immediate filter bypass threat detected."
        )
    elif resistance_score >= 50.0:
        advice = (
            f"Defensive model displays Moderate Resilience ({resistance_score:.1f}%). "
            f"Adversary feature jittering (e.g. packet size padding, timing delays) reduced detection "
            f"efficacy by {drop_pct:.1f}%. Recommend activating adversarial data augmentation."
        )
    else:
        advice = (
            f"🚨 CRITICAL MODEL HACKABILITY WARNING: Evasion resistance dropped to {resistance_score:.1f}%. "
            f"Subtle feature perturbations ({tested_epsilon * 100:.0f}% noise) allow malicious payloads to bypass "
            f"detection in {drop_pct:.1f}% of evasion attempts! Immediate adversarial retraining, rate-limiting, "
            f"and secondary packet-inspection rules required."
        )

    return badge, advice
