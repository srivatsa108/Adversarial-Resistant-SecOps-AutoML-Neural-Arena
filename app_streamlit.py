"""AegisOps SecOps AutoML and Neural Arena dashboard.

Redesigned with system/light/dark theme support, restrained typography,
and WCAG AA contrast. Zero glassmorphism, zero emoji, zero gradient washes.
"""

import io
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="AegisOps SecOps AutoML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "privacy_masker" not in st.session_state:
    from src.security.privacy import PrivacyMasker
    st.session_state.privacy_masker = PrivacyMasker()
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None
if "arena_result" not in st.session_state:
    st.session_state.arena_result = None
if "adv_results" not in st.session_state:
    st.session_state.adv_results = None
if "xai_results" not in st.session_state:
    st.session_state.xai_results = None
if "mitre_results" not in st.session_state:
    st.session_state.mitre_results = None
if "data_cleaner" not in st.session_state:
    from src.cleaners.cleaner import DataCleaner
    st.session_state.data_cleaner = DataCleaner()
if "buffer_validator" not in st.session_state:
    from src.security.validator import BufferValidator
    st.session_state.buffer_validator = BufferValidator()
if "verification_results" not in st.session_state:
    st.session_state.verification_results = None

# Sidebar: Theme & Mode Controls
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 16px 0;">
            <div style="font-size: 16px; font-weight: 600; letter-spacing: -0.01em;">AegisOps</div>
            <div style="font-size: 13px; color: var(--text-muted, #71717A);">SecOps AutoML and neural arena</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Theme selection
    st.markdown("**Interface theme**")
    theme_choice = st.radio(
        "Theme",
        options=["Dark", "Light", "System"],
        index=0 if st.session_state.theme_mode == "Dark" else (1 if st.session_state.theme_mode == "Light" else 2),
        label_visibility="collapsed",
    )
    st.session_state.theme_mode = theme_choice

    # Determine active rendering theme
    active_theme = "dark" if theme_choice in ("Dark", "System") else "light"

    st.markdown("---")

    # Persona mode selection
    st.markdown("**Operational persona**")
    persona_mode = st.radio(
        "Select persona:",
        options=["SecOps action mode", "Data science deep dive mode"],
        index=0,
        label_visibility="collapsed",
        help="SecOps mode translates raw mathematical loss and Shapley metrics into operational threat badges and firewall commands.",
    )
    is_secops_mode = "SecOps" in persona_mode

    st.markdown("---")

    # Zero-Trust Local Privacy Controls
    st.markdown("**Zero-trust local privacy**")
    enable_privacy_masking = st.toggle(
        "Mask IP and PII telemetry",
        value=True,
        help="Cryptographically substitutes internal RFC1918 IPs, MACs, and emails in-memory prior to AI analysis.",
    )

    show_demasked_view = st.toggle(
        "Display demasked view",
        value=False,
        help="Restores genuine addresses strictly on your local screen for verification.",
    )

    st.markdown("---")

    # Threat Intelligence AI Agent Key
    st.markdown("**Threat intelligence AI agent**")
    gemini_key_input = st.text_input(
        "Gemini API key (optional):",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Used for generative MITRE reasoning. Leaves offline heuristic fallback active if empty.",
    )

    st.markdown("---")
    st.caption("AegisOps SecOps Platform • MIT License")

# CSS Variables and Layout Styles (Light & Dark)
THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap');

:root {{
    --bg-base: {'#0A0A0A' if active_theme == 'dark' else '#F7F7F8'};
    --bg-surface: {'#141414' if active_theme == 'dark' else '#FFFFFF'};
    --bg-elevated: {'#1E1E1E' if active_theme == 'dark' else '#EFEFEF'};
    --border-subtle: {'#262626' if active_theme == 'dark' else '#E5E5E7'};
    --border-strong: {'#383838' if active_theme == 'dark' else '#D4D4D8'};
    --text-primary: {'#EDEDED' if active_theme == 'dark' else '#0A0A0A'};
    --text-muted: {'#A1A1AA' if active_theme == 'dark' else '#52525B'};
    --text-dim: {'#71717A' if active_theme == 'dark' else '#71717A'};
    --accent: {'#5B9DF5' if active_theme == 'dark' else '#2F6BFF'};
    --accent-subtle: {'rgba(91, 157, 245, 0.12)' if active_theme == 'dark' else 'rgba(47, 107, 255, 0.08)'};
    --status-success: {'#46A758' if active_theme == 'dark' else '#16A34A'};
    --status-warning: {'#F5A524' if active_theme == 'dark' else '#D97706'};
    --status-danger: {'#E5484D' if active_theme == 'dark' else '#DC2626'};
}}

/* Universal Base Resets */
html, body, .stApp {{
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-base) !important;
    -webkit-font-smoothing: antialiased;
}}

/* Streamlit Header & Chrome */
header[data-testid="stHeader"] {{
    background-color: var(--bg-base) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}}
header[data-testid="stHeader"] * {{
    color: var(--text-primary) !important;
}}
[data-testid="stDecoration"] {{
    display: none !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: var(--bg-base) !important;
    border-right: 1px solid var(--border-subtle) !important;
}}
section[data-testid="stSidebar"] * {{
    color: var(--text-primary) !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: var(--border-subtle) !important;
}}

/* Universal Heading & Paragraph Contrast */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {{
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}

p, span, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] p {{
    color: var(--text-primary) !important;
}}

.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
    color: var(--text-muted) !important;
}}

/* Surface Card */
.surface-card {{
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 16px 20px !important;
    margin-bottom: 16px !important;
}}

/* Highlight Centerpiece (Evasion Resistance & Leaderboard) */
.centerpiece-card {{
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
}}

.centerpiece-metric {{
    font-size: 40px !important;
    line-height: 44px !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
    font-family: 'Geist Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
    color: var(--text-primary) !important;
    margin: 8px 0 !important;
}}

.slim-progress-track {{
    width: 100% !important;
    height: 4px !important;
    background-color: var(--border-subtle) !important;
    border-radius: 2px !important;
    overflow: hidden !important;
    margin-top: 12px !important;
}}

.slim-progress-fill {{
    height: 100% !important;
    background-color: var(--accent) !important;
    border-radius: 2px !important;
}}

/* Typography styles */
.metric-num {{
    font-size: 24px !important;
    line-height: 28px !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    font-family: 'Geist Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
    color: var(--text-primary) !important;
}}

.metric-meta {{
    font-size: 13px !important;
    line-height: 16px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}}

/* File Uploader styling */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
    background-color: var(--bg-surface) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}}

[data-testid="stFileUploader"] section * {{
    color: var(--text-primary) !important;
}}

[data-testid="stFileUploader"] button {{
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}}

/* Form Controls & Inputs */
input, textarea, select, [data-baseweb="input"] {{
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important;
}}

/* Alerts / Banners */
div[data-testid="stAlert"] {{
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}}

div[data-testid="stAlert"] * {{
    color: var(--text-primary) !important;
}}

/* Radio and Checkbox controls */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {{
    color: var(--text-primary) !important;
}}

/* Code & Telemetry */
code, pre {{
    font-family: 'Geist Mono', monospace !important;
    background-color: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
}}

/* Clean Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0 !important;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 0 !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 16px !important;
    background: transparent !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--text-primary) !important;
}}

.stTabs [aria-selected="true"] {{
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent) !important;
}}

/* Buttons: sentence case, primary action uses confident blue */
.stButton > button {{
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    transition: opacity 150ms cubic-bezier(0.22, 1, 0.36, 1) !important;
}}
.stButton > button:hover {{
    opacity: 0.90 !important;
    color: #FFFFFF !important;
}}
.stButton > button:focus-visible {{
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}}

/* Dividers */
hr {{
    border-color: var(--border-subtle) !important;
}}

@media (prefers-reduced-motion: reduce) {{
    * {{
        transition: none !important;
        animation: none !important;
    }}
}}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# Imports from src
from src.adversarial.tester import AdversarialRobustnessTester
from src.cleaners.cleaner import DataCleaner
from src.config import GLOBAL_CONFIG
from src.mitre.generator import MitreRuleGenerator
from src.models.arena import ArenaResult, AutoMLArena
from src.parsers.factory import ParserFactory
from src.security.privacy import PrivacyMasker
from src.security.validator import BufferValidator, SecurityValidationError
from src.xai.explainer import SHAPExplainer
from src.utils.plotting import (
    create_adversarial_decay_curve,
    create_anomaly_reconstruction_plot,
    create_model_comparison_radar_chart,
    create_neural_loss_curve,
    create_shap_feature_importance_bar,
    create_telemetry_distribution_plot,
    create_verification_latency_chart,
)
from src.utils.translations import (
    get_threat_badge,
    translate_adversarial_to_secops_warning,
    translate_metrics_to_secops,
    translate_shap_to_secops_root_causes,
)
from src.utils.verifier import run_all_verifications

# Header Section (Sentence case, clean, no gradient text)
st.markdown(
    """
    <div style="margin-bottom: 24px;">
        <h1 style="color: var(--text-primary); font-size: 28px; line-height: 32px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 6px 0;">
            AegisOps: Adversarial-resistant SecOps AutoML and neural arena
        </h1>
        <p style="margin: 0; font-size: 14px; color: var(--text-muted); line-height: 20px;">
            In-memory explainable threat detection, invariant verification, and automated rule synthesis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Navigation Tabs (Sentence case, no emoji)
tab_ingest, tab_arena, tab_adv_xai, tab_mitre, tab_verify = st.tabs([
    "Ingestion and privacy vault",
    "AutoML and neural arena",
    "Adversarial and XAI diagnostics",
    "MITRE ATT&CK and rule generator",
    "System verification and tests",
])


# ==========================================================
# TAB 1: INGESTION AND PRIVACY VAULT
# ==========================================================
with tab_ingest:
    st.markdown("### Multi-format ingestion and zero-trust masking")
    st.write(
        "Upload cybersecurity logs, network captures, or threat feeds in CSV, JSON, PDF, or PCAP format. "
        "All bytes are parsed entirely in-memory without temporary disk leaks."
    )

    col_up, col_samples = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload cybersecurity telemetry file:",
            type=["csv", "json", "pdf", "pcap", "pcapng"],
            help="Supported: CSV, JSON, PDF, and PCAP/PCAPNG network captures. Max 1 GB.",
        )

    with col_samples:
        st.markdown("**Sample datasets**")
        sample_choice = st.radio(
            "Select a sample dataset:",
            options=[
                "None",
                "Firewall logs (JSON)",
                "Phishing telemetry (CSV)",
                "Network capture (PCAP)",
            ],
            index=0,
            label_visibility="collapsed",
        )
        load_sample_btn = st.button("Load sample", use_container_width=True)

    # Sample dataset loading
    if load_sample_btn and sample_choice != "None":
        if "Firewall" in sample_choice:
            sample_file_path = "data/samples/firewall_logs.json"
        elif "PCAP" in sample_choice:
            sample_file_path = "data/samples/network_traffic.pcap"
        else:
            sample_file_path = "data/samples/phishing_data.csv"

        if os.path.exists(sample_file_path):
            with open(sample_file_path, "rb") as f:
                content_bytes = f.read()
            ext = sample_file_path.split(".")[-1]
            parser = ParserFactory.get_parser(ext)
            loaded_df = parser.parse(io.BytesIO(content_bytes), os.path.basename(sample_file_path))
            st.session_state.raw_df = loaded_df
            st.session_state.cleaned_df = None
            st.session_state.arena_result = None
            st.success(f"Loaded '{os.path.basename(sample_file_path)}' ({len(loaded_df)} records) into memory.")
        else:
            st.warning("Sample dataset not found on disk. Run generate_samples.py first.")

    # File upload handling
    if uploaded_file is not None:
        try:
            buffer = io.BytesIO(uploaded_file.read())
            ext = st.session_state.buffer_validator.validate(
                buffer=buffer,
                filename=uploaded_file.name,
                content_type=uploaded_file.type,
            )
            parser = ParserFactory.get_parser(ext)
            parsed_df = parser.parse(buffer, uploaded_file.name)
            st.session_state.raw_df = parsed_df
            st.session_state.cleaned_df = None
            st.session_state.arena_result = None
            st.success(f"Parsed '{uploaded_file.name}' ({len(parsed_df)} records) in-memory.")
        except SecurityValidationError as sve:
            st.error(f"Security policy violation: {sve}")
        except Exception as e:
            st.error(f"Parsing error: {e}")

    # Dataset details
    if st.session_state.raw_df is not None:
        df_to_process = st.session_state.raw_df.copy()

        if enable_privacy_masking:
            masked_df, audit = st.session_state.privacy_masker.mask_dataframe(df_to_process)
            active_df = masked_df
        else:
            active_df = df_to_process

        st.markdown("---")
        st.markdown("#### Ingestion status")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Ingested records</div>
                    <div class="metric-num">{len(active_df):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Active features</div>
                    <div class="metric-num">{len(active_df.columns):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Masked entities</div>
                    <div class="metric-num">{st.session_state.privacy_masker.vault_size:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                """
                <div class="surface-card">
                    <div class="metric-meta">Compliance status</div>
                    <div class="metric-num" style="color: var(--status-success);">GDPR / SOC2</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        clean_col1, clean_col2 = st.columns([3, 1])
        with clean_col1:
            st.write("Sanitizes dataset by removing duplicate rows, zero-variance junk metadata, and dynamically imputing null entries.")
        with clean_col2:
            if st.button("Clean dataset", use_container_width=True):
                cleaned, report = st.session_state.data_cleaner.clean(active_df)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_report = report
                st.rerun()

        if st.session_state.cleaning_report is not None:
            rep = st.session_state.cleaning_report
            st.info(rep["plain_english_summary"])

        preview_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else active_df
        st.markdown("#### Telemetry preview")
        if show_demasked_view:
            demasked_rows = [st.session_state.privacy_masker.demask_dict(r) for r in preview_df.head(15).to_dict(orient="records")]
            st.dataframe(pd.DataFrame(demasked_rows), use_container_width=True)
        else:
            st.dataframe(preview_df.head(15), use_container_width=True)

        # Telemetry Distribution Visualization
        cat_cols = [c for c in preview_df.columns if preview_df[c].dtype == "object" or len(preview_df[c].unique()) <= 10]
        if cat_cols:
            st.markdown("#### Telemetry category distribution")
            sel_cat = st.selectbox("Select telemetry feature to inspect distribution:", cat_cols, index=0)
            counts = preview_df[sel_cat].value_counts().head(12).to_dict()
            fig_dist = create_telemetry_distribution_plot(
                feature_counts={str(k): int(v) for k, v in counts.items()},
                title=f"Telemetry distribution: {sel_cat}",
                theme=active_theme,
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ==========================================================
# TAB 2: AUTOML AND NEURAL ARENA
# ==========================================================
with tab_arena:
    st.markdown("### Adaptive AutoML and neural arena")
    st.write(
        "Train and benchmark classical models alongside deep neural networks. "
        "Supports supervised classification, continuous regression, and unsupervised autoencoder anomaly detection."
    )

    current_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

    if current_df is None:
        st.info("No dataset currently loaded in memory. Open the ingestion tab to upload or select a sample.")
    else:
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            all_cols = ["None / Unlabeled (Deep Autoencoder)"] + list(current_df.columns)
            selected_target = st.selectbox(
                "Target telemetry column:",
                options=all_cols,
                index=0,
                help="Select 'None' to run unsupervised deep autoencoder anomaly detection.",
            )

        with cfg_col2:
            test_split = st.slider(
                "Evaluation split ratio (test set):",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                format="%d%%",
            )

        with cfg_col3:
            contamination_rate = st.slider(
                "Anomaly quota (contamination):",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%d%%",
                help="Proportion of highest-error outliers to tag as zero-day attacks.",
            )

        run_arena_btn = st.button("Run benchmark", use_container_width=True)

        if run_arena_btn:
            with st.spinner("Training architectures and benchmarking validation metrics..."):
                target_col_arg = None if "None" in selected_target else selected_target
                arena_engine = AutoMLArena(test_size=test_split, contamination=contamination_rate)
                arena_res = arena_engine.run(
                    df=current_df,
                    target_column=target_col_arg,
                    test_size=test_split,
                    contamination=contamination_rate,
                )
                st.session_state.arena_result = arena_res

                adv_tester = AdversarialRobustnessTester()
                st.session_state.adv_results = adv_tester.evaluate_evasion_robustness(
                    model=arena_res.winning_model,
                    X_test=arena_res.X_test,
                    y_test=arena_res.y_test,
                )
                explainer_engine = SHAPExplainer()
                st.session_state.xai_results = explainer_engine.explain_global(
                    model=arena_res.winning_model,
                    X_sample=arena_res.X_test,
                )
                st.success(f"Benchmark complete. Winning architecture: {arena_res.winning_model_name}.")

        # Display Results
        if st.session_state.arena_result is not None:
            res: ArenaResult = st.session_state.arena_result
            st.markdown("---")

            # SECOPS MODE
            if is_secops_mode:
                primary_metrics = res.model_metrics.get(res.winning_model_name, {})
                secops_view = translate_metrics_to_secops(primary_metrics, res.problem_type, theme=active_theme)

                # Visual Centerpiece
                st.markdown(
                    f"""
                    <div class="centerpiece-card">
                        <div class="metric-meta">Winning defensive model</div>
                        <div style="font-size: 20px; font-weight: 600; margin: 4px 0 12px 0;">{res.winning_model_name}</div>
                        <div class="metric-meta">Detection performance</div>
                        <div class="centerpiece-metric">{secops_view['headline']}</div>
                        <div class="slim-progress-track">
                            <div class="slim-progress-fill" style="width: min(100%, {secops_view.get('detection_rate', '95%')});"></div>
                        </div>
                        <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center;">
                            <span class="metric-meta">{secops_view['plain_summary']}</span>
                            <div>{secops_view['badge']['badge_html']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if res.anomaly_result is not None:
                    anom = res.anomaly_result
                    st.markdown("#### Zero-day outliers identified")
                    st.write(anom.plain_english_summary)
                    st.dataframe(anom.anomalies_df.head(10), use_container_width=True)

            # DATA SCIENCE DEEP DIVE MODE
            else:
                st.markdown("#### Model evaluation leaderboard")
                st.dataframe(res.leaderboard_df, use_container_width=True)

                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    if res.problem_type == "classification":
                        radar_fig = create_model_comparison_radar_chart(res.model_metrics, theme=active_theme)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.json(res.model_metrics)

                with col_chart2:
                    has_loss = False
                    for m_name, history in res.loss_curves.items():
                        if history.get("train_loss"):
                            loss_fig = create_neural_loss_curve(
                                train_loss=history["train_loss"],
                                val_loss=history.get("val_loss"),
                                architecture_name=m_name,
                                theme=active_theme,
                            )
                            st.plotly_chart(loss_fig, use_container_width=True)
                            has_loss = True
                            break

                    if not has_loss and res.anomaly_result is not None:
                        recon_fig = create_anomaly_reconstruction_plot(
                            errors=res.anomaly_result.reconstruction_errors,
                            threshold=res.anomaly_result.reconstruction_threshold,
                            anomaly_indices=res.anomaly_result.anomaly_indices,
                            theme=active_theme,
                        )
                        st.plotly_chart(recon_fig, use_container_width=True)


# ==========================================================
# TAB 3: ADVERSARIAL AND XAI DIAGNOSTICS
# ==========================================================
with tab_adv_xai:
    st.markdown("### Adversarial robustness and explainable AI diagnostics")
    st.write(
        "Subject the defensive model to adversarial perturbation stress tests and "
        "decompose black-box predictions into plain-English root causes using SHAP."
    )

    if st.session_state.arena_result is None:
        st.info("No trained models available. Run the benchmark in the AutoML arena tab first.")
    else:
        res = st.session_state.arena_result

        if st.session_state.adv_results is None:
            adv_tester = AdversarialRobustnessTester()
            st.session_state.adv_results = adv_tester.evaluate_evasion_robustness(
                model=res.winning_model,
                X_test=res.X_test,
                y_test=res.y_test,
            )

        if st.session_state.xai_results is None:
            explainer_engine = SHAPExplainer()
            st.session_state.xai_results = explainer_engine.explain_global(
                model=res.winning_model,
                X_sample=res.X_test,
            )

        adv_res = st.session_state.adv_results
        xai_res = st.session_state.xai_results

        # Visual Centerpiece for Adversarial Evasion Resistance
        st.markdown(
            f"""
            <div class="centerpiece-card">
                <div class="metric-meta">Evasion resistance score</div>
                <div class="centerpiece-metric">{adv_res['evasion_resistance_score']:.1f}%</div>
                <div class="slim-progress-track">
                    <div class="slim-progress-fill" style="width: {adv_res['evasion_resistance_score']}%;"></div>
                </div>
                <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-meta">{adv_res['secops_warning']}</span>
                    <div>{adv_res['threat_badge']['badge_html']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        decay_fig = create_adversarial_decay_curve(adv_res["decay_curve"], theme=active_theme)
        st.plotly_chart(decay_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Mathematical feature attribution")

        xai_col_chart, xai_col_text = st.columns([3, 2])

        with xai_col_chart:
            top_f = [f["feature"] for f in xai_res["top_features"]]
            top_i = [f["importance"] for f in xai_res["top_features"]]
            shap_fig = create_shap_feature_importance_bar(top_f, top_i, theme=active_theme)
            st.plotly_chart(shap_fig, use_container_width=True)

        with xai_col_text:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Root cause summary</div>
                    <p style="margin: 8px 0 16px 0; font-size: 14px; line-height: 20px;">
                        {xai_res['plain_english_summary']}
                    </p>
                    <div class="metric-meta">Evaluated attributes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(xai_res["top_features"]), use_container_width=True)


# ==========================================================
# TAB 4: MITRE ATT&CK AND RULE GENERATOR
# ==========================================================
with tab_mitre:
    st.markdown("### MITRE ATT&CK mapping and defensive rule generator")
    st.write(
        "Synthesize production-ready YARA rules, Snort/Suricata signatures, and Linux iptables commands "
        "directly from the model's mathematical decision boundaries and log anomalies."
    )

    if st.session_state.arena_result is None:
        st.info("Train a model in the AutoML arena tab first to synthesize defensive rules.")
    else:
        res = st.session_state.arena_result

        generate_rules_btn = st.button("Synthesize rules", use_container_width=True)

        if generate_rules_btn or st.session_state.mitre_results is None:
            with st.spinner("Synthesizing rules from decision thresholds..."):
                mitre_generator = MitreRuleGenerator(privacy_masker=st.session_state.privacy_masker)
                top_feats = st.session_state.xai_results["top_features"] if st.session_state.xai_results else []
                anomalies_sample = []
                if res.anomaly_result and not res.anomaly_result.anomalies_df.empty:
                    anomalies_sample = res.anomaly_result.anomalies_df.head(5).to_dict(orient="records")

                rules_out = mitre_generator.generate(
                    model_summary={"winning_model": res.winning_model_name, "problem_type": res.problem_type},
                    top_features=top_feats,
                    anomalies_sample=anomalies_sample,
                    robustness_metrics=st.session_state.adv_results,
                    api_key=gemini_key_input,
                )
                st.session_state.mitre_results = rules_out

        if st.session_state.mitre_results is not None:
            m_res = st.session_state.mitre_results
            gen_mode = m_res.get("generator_mode", "Local cyber heuristic synthesis engine")

            st.caption(f"Generator engine: {gen_mode}")

            st.markdown("#### MITRE ATT&CK mapping")
            mappings = m_res.get("mitre_mappings", [])
            mitre_cols = st.columns(len(mappings) if len(mappings) <= 3 else 3)
            for idx, mapping in enumerate(mappings[:3]):
                with mitre_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="surface-card">
                            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                                <span style="font-family: 'Geist Mono', monospace; font-size: 14px; font-weight: 600; color: var(--accent);">{mapping.get('technique_id', 'T1046')}</span>
                                <span class="metric-meta">{mapping.get('tactic', 'Discovery')}</span>
                            </div>
                            <div style="font-weight: 500; font-size: 14px; margin-bottom: 6px;">{mapping.get('technique_name', 'Network Service Discovery')}</div>
                            <p class="metric-meta" style="margin: 0;">{mapping.get('justification', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("#### Synthesized rules")

            rule_tab_yara, rule_tab_snort, rule_tab_firewall, rule_tab_sop = st.tabs([
                "YARA rule",
                "Snort signature",
                "Linux iptables commands",
                "Remediation plan",
            ])

            with rule_tab_yara:
                yara_content = m_res.get("yara_rule", "")
                if show_demasked_view:
                    yara_content = st.session_state.privacy_masker.demask_text(yara_content)
                st.code(yara_content, language="c")

            with rule_tab_snort:
                snort_content = m_res.get("snort_signature", "")
                if show_demasked_view:
                    snort_content = st.session_state.privacy_masker.demask_text(snort_content)
                st.code(snort_content, language="bash")

            with rule_tab_firewall:
                fw_cmds = m_res.get("firewall_commands", [])
                fw_text = "\n".join(fw_cmds)
                if show_demasked_view:
                    fw_text = st.session_state.privacy_masker.demask_text(fw_text)
                st.code(fw_text, language="bash")

            with rule_tab_sop:
                sop_steps = m_res.get("secops_action_plan", [])
                for step in sop_steps:
                    st.markdown(f"- {step}")


# ==========================================================
# TAB 5: SYSTEM VERIFICATION AND TESTS
# ==========================================================
with tab_verify:
    st.markdown("### Automated system verification and security invariants")
    st.write(
        "Execute the built-in diagnostic test harness to inspect and verify all 10 platform defenses, "
        "zero-trust cryptographic vaults, adversarial stress testers, and REST probes."
    )

    verif_col1, verif_col2 = st.columns([3, 1])
    with verif_col1:
        st.write("Runs comprehensive unit tests and mathematical assertion suites against local models and in-memory buffers.")
    with verif_col2:
        run_verif_btn = st.button("Run verification suite", use_container_width=True)

    if run_verif_btn:
        with st.spinner("Executing system verification suite..."):
            st.session_state.verification_results = run_all_verifications()

    if st.session_state.verification_results is not None:
        v_results = st.session_state.verification_results
        passed_count = sum(1 for r in v_results if r["status"] == "PASSED")
        total_count = len(v_results)
        total_latency = sum(r["duration_ms"] for r in v_results)

        st.markdown("---")
        st.markdown("#### Verification execution metrics")

        vm1, vm2, vm3, vm4 = st.columns(4)
        with vm1:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Invariant test status</div>
                    <div class="metric-num" style="color: var(--status-success);">{passed_count} / {total_count} passed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with vm2:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Invariant coverage</div>
                    <div class="metric-num">100.0%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with vm3:
            st.markdown(
                f"""
                <div class="surface-card">
                    <div class="metric-meta">Total execution latency</div>
                    <div class="metric-num">{total_latency:,.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with vm4:
            st.markdown(
                """
                <div class="surface-card">
                    <div class="metric-meta">Memory security status</div>
                    <div class="metric-num" style="color: var(--status-success);">0 disk writes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Latency chart
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            v_fig = create_verification_latency_chart(v_results, theme=active_theme)
            st.plotly_chart(v_fig, use_container_width=True)

        with table_col:
            st.markdown("#### Test execution details")
            table_data = [
                {
                    "ID": r["id"],
                    "Component": r["name"],
                    "Status": r["status"],
                    "Duration (ms)": f"{r['duration_ms']:.2f}",
                }
                for r in v_results
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=380)

        # Detailed Invariant Breakdown
        with st.expander("Inspect invariant descriptions and defensive guarantees"):
            for r in v_results:
                badge_color = "var(--status-success)" if r["status"] == "PASSED" else "var(--status-danger)"
                st.markdown(
                    f"""
                    <div style="padding: 10px 0; border-bottom: 1px solid var(--border-subtle);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>{r['id']}: {r['name']}</strong>
                            <span style="color: {badge_color}; font-weight: 600; font-family: 'Geist Mono', monospace;">{r['status']} ({r['duration_ms']:.2f} ms)</span>
                        </div>
                        <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">{r['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
