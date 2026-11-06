"""Modern SecOps Dark-Mode Streamlit Dashboard for Adversarial-Resistant AutoML.

Features:
- Sleek SecOps Dark Mode with Glassmorphic UI, glowing borders, and neon cyber accents.
- Persona Mode Toggle: '🛡️ SecOps Action Mode' (Default) vs '🔬 Data Science Deep-Dive Mode'.
- Multi-Tab Navigation:
  [📁 Ingestion & Zero-Trust Privacy] -> [🧠 AutoML & Neural Arena]
  -> [🛡️ Adversarial & XAI Diagnostics] -> [🤖 MITRE ATT&CK & Rule Generator].
- Interactive Plotly dark-theme charts.
- Google Antigravity SDK / Gemini integration with local cyber heuristic fallback.
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
    page_title="Antigravity SecOps AutoML & Neural Arena",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Glassmorphic SecOps Dark Mode styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global Dark Theme Settings */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #080c14 100%);
}

/* Glassmorphism Cards */
.cyber-card {
    background: rgba(17, 25, 39, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.cyber-card:hover {
    border-color: rgba(0, 229, 255, 0.45);
    box-shadow: 0 10px 40px 0 rgba(0, 229, 255, 0.15);
}

.cyber-card-danger {
    background: rgba(30, 10, 20, 0.75);
    border: 1px solid rgba(255, 0, 85, 0.35);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.cyber-card-success {
    background: rgba(10, 30, 25, 0.75);
    border: 1px solid rgba(0, 255, 204, 0.35);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

/* Metrics and glowing numbers */
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00ffcc 0%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 4px;
}

/* Glowing Neon Accents */
.neon-teal { color: #00ffcc; text-shadow: 0 0 10px rgba(0,255,204,0.4); }
.neon-blue { color: #00e5ff; text-shadow: 0 0 10px rgba(0,229,255,0.4); }
.neon-pink { color: #ff0055; text-shadow: 0 0 10px rgba(255,0,85,0.4); }
.neon-amber { color: #ffaa00; text-shadow: 0 0 10px rgba(255,170,0,0.4); }

/* Monospace Code blocks */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Tab Active states */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(17, 25, 39, 0.6);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(0, 229, 255, 0.15) !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0052cc 0%, #0099ff 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    box-shadow: 0 0 15px rgba(0, 153, 255, 0.5);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
)
from src.utils.translations import (
    get_threat_badge,
    translate_adversarial_to_secops_warning,
    translate_metrics_to_secops,
    translate_shap_to_secops_root_causes,
)

# Session state initialization
if "privacy_masker" not in st.session_state:
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
    st.session_state.data_cleaner = DataCleaner()
if "buffer_validator" not in st.session_state:
    st.session_state.buffer_validator = BufferValidator()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <span style="font-size: 2rem;">🛡️</span>
            <div>
                <h3 style="margin: 0; padding: 0; font-size: 1.2rem; font-weight: 800; color: #00e5ff;">ANTIGRAVITY</h3>
                <span style="font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">SecOps AutoML & Deep Net Arena</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 1. PERSONA MODE TOGGLE
    st.markdown("#### 👤 Operational Persona")
    persona_mode = st.radio(
        "Select Active Interface Persona:",
        options=["🛡️ SecOps Action Mode", "🔬 Data Science Deep-Dive Mode"],
        index=0,
        help=(
            "SecOps Action Mode abstracts raw mathematics into Threat Badges and remediation rules. "
            "Data Science Mode exposes loss curves, hyperparameter controls, and raw validation metrics."
        ),
    )
    is_secops_mode = "SecOps" in persona_mode

    st.markdown("---")

    # 2. ZERO-TRUST LOCAL PRIVACY CONTROLS
    st.markdown("#### 🔒 Zero-Trust Local Privacy")
    enable_privacy_masking = st.toggle(
        "Enable PII & IP Obfuscation",
        value=True,
        help="Cryptographically anonymizes IPv4, IPv6, MAC, emails, and sensitive URLs in-memory prior to AI/ML ingestion.",
    )

    show_demasked_view = st.toggle(
        "Display De-Masked Telemetry",
        value=False,
        help="Restores real entities strictly on your local browser screen (GDPR/SOC2 compliant local de-masking).",
    )

    st.markdown("---")

    # 3. GOOGLE ANTIGRAVITY / GEMINI API INTEGRATION
    st.markdown("#### 🤖 Google Antigravity Agent")
    gemini_key_input = st.text_input(
        "Gemini API Key (Optional):",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Enables Google Antigravity LLM Agent for enhanced MITRE reasoning. Leaves fallback to local rule synthesis if omitted.",
    )

    st.markdown("---")
    st.caption("SecOps AutoML Platform v1.0.0 • Open Source MIT License")


# ==========================================
# MAIN TITLE BANNER
# ==========================================
st.markdown(
    f"""
    <div style="margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Adversarial-Resistant SecOps AutoML & Neural Arena
                </h1>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                    Explainable AI (XAI) & Deep Neural Network Platform for Cybersecurity Data • <span class="neon-teal">{'Active: 🛡️ SecOps Action Mode' if is_secops_mode else 'Active: 🔬 Data Science Deep-Dive Mode'}</span>
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0, 229, 255, 0.3); color: #00e5ff; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    🔒 Zero-Trust In-Memory Sandbox
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Multi-Tab Navigation
tab_ingest, tab_arena, tab_adv_xai, tab_mitre = st.tabs([
    "📁 Ingestion & Zero-Trust Privacy",
    "🧠 AutoML & Neural Arena",
    "🛡️ Adversarial & XAI Diagnostics",
    "🤖 MITRE ATT&CK & Rule Generator",
])


# ==========================================================
# TAB 1: INGESTION & ZERO-TRUST PRIVACY
# ==========================================================
with tab_ingest:
    st.markdown("### 📥 Multi-Format Ingestion & Zero-Trust Cryptographic Anonymizer")
    st.write(
        "Upload cybersecurity logs, network captures, or threat feeds in **CSV**, **JSON**, **PDF**, or raw **PCAP / PCAPNG** format. "
        "All bytes are parsed entirely in-memory (`io.BytesIO`) without temporary disk leaks."
    )

    col_up, col_samples = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Choose a cybersecurity telemetry file:",
            type=["csv", "json", "pdf", "pcap", "pcapng"],
            help="Supported: CSV, JSON (records/arrays/ndjson), PDF (tables & regex text mining), and PCAP/PCAPNG packet captures. Max 50MB.",
        )

    with col_samples:
        st.markdown("**Quick-Load Built-in Sample Datasets:**")
        sample_choice = st.radio(
            "Select a demo threat dataset:",
            options=[
                "None",
                "Firewall Logs (Unsupervised Anomaly)",
                "Phishing Data (Supervised Target)",
                "Network Packet Capture (PCAP Anomaly)",
            ],
            index=0,
        )

        load_sample_btn = st.button("Load Selected Sample", use_container_width=True)

    # Handle sample dataset loading
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
            st.success(f"Successfully loaded '{os.path.basename(sample_file_path)}' ({len(loaded_df)} records) into memory!")
        else:
            st.warning("Sample dataset not found on disk. Run `python generate_samples.py` first.")

    # Handle file upload
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
            st.success(f"Parsed '{uploaded_file.name}' via {ext.upper()} in-memory parser ({len(parsed_df)} records).")
        except SecurityValidationError as sve:
            st.error(f"Security Policy Violation: {sve}")
        except Exception as e:
            st.error(f"Parsing error: {e}")

    # Display dataset details if loaded
    if st.session_state.raw_df is not None:
        df_to_process = st.session_state.raw_df.copy()

        # Zero-Trust Cryptographic Anonymizer step
        if enable_privacy_masking:
            masked_df, audit = st.session_state.privacy_masker.mask_dataframe(df_to_process)
            active_df = masked_df
        else:
            active_df = df_to_process
            audit = {"total_vault_entities": 0, "new_entities_masked": 0, "category_breakdown": {}}

        st.markdown("---")
        st.markdown("#### 🔒 Preprocessing: Zero-Trust Local Masking & Junk Sanitization")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div class="metric-label">Ingested Logs</div>
                    <div class="metric-value">{len(active_df):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div class="metric-label">Raw Features</div>
                    <div class="metric-value">{len(active_df.columns):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div class="metric-label">Masked Entities</div>
                    <div class="metric-value neon-teal">{st.session_state.privacy_masker.vault_size:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div class="metric-label">Compliance Guard</div>
                    <div class="metric-value neon-blue">SOC2 / GDPR</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Trigger Cleaning Module
        clean_col1, clean_col2 = st.columns([3, 1])
        with clean_col1:
            st.markdown(
                """
                **Automated Data Cleaning & Metadata Junk Pruning:**
                Removes duplicate rows, strips zero-variance constant junk columns, and dynamically imputes nulls.
                """
            )
        with clean_col2:
            if st.button("🧹 Clean & Sanitize Dataset", use_container_width=True):
                cleaned, report = st.session_state.data_cleaner.clean(active_df)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_report = report
                st.rerun()

        # Display Cleaning Report if executed
        if st.session_state.cleaning_report is not None:
            rep = st.session_state.cleaning_report
            st.markdown(
                f"""
                <div class="cyber-card-success">
                    <div style="font-weight: 700; color: #00ffcc; margin-bottom: 6px;">
                        🛡️ Zero-Trust Sanitization Audit
                    </div>
                    <p style="margin: 0; font-size: 0.95rem; color: #cbd5e1;">
                        {rep['plain_english_summary']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Ingestion Data Table Preview (Masked vs De-Masked)
        preview_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else active_df

        st.markdown("#### 🔍 Active In-Memory Telemetry Preview")
        if show_demasked_view:
            st.info("🔓 Local De-Masking Active: Sensitive internal hostnames and IP addresses are restored on your local display.")
            # Restore entities for local view
            demasked_rows = [st.session_state.privacy_masker.demask_dict(r) for r in preview_df.head(15).to_dict(orient="records")]
            st.dataframe(pd.DataFrame(demasked_rows), use_container_width=True)
        else:
            st.dataframe(preview_df.head(15), use_container_width=True)


# ==========================================================
# TAB 2: AUTOML & NEURAL ARENA
# ==========================================================
with tab_arena:
    st.markdown("### 🧠 Adaptive AutoML & Deep Neural Network Arena")
    st.write(
        "Train and benchmark classical models and PyTorch Deep Neural Networks. "
        "Supports Supervised Classification, Continuous Regression, and Unsupervised Autoencoder Anomaly Detection."
    )

    current_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

    if current_df is None:
        st.warning("⚠️ No dataset currently loaded in memory. Go to the 'Ingestion' tab to upload or load a sample.")
    else:
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            all_cols = ["None / Unlabeled (Deep Autoencoder)"] + list(current_df.columns)
            selected_target = st.selectbox(
                "Select Target Telemetry Column:",
                options=all_cols,
                index=0,
                help="Select 'None' to run Unsupervised Deep Autoencoder Anomaly Detection.",
            )

        with cfg_col2:
            test_split = st.slider(
                "Evaluation Split Ratio (Train/Test):",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                format="%d%%",
            )

        with cfg_col3:
            contamination_rate = st.slider(
                "Autoencoder Anomaly Quota (Contamination):",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%d%%",
                help="Used when Target is 'None' to isolate the top X% reconstruction error outliers as Zero-Day anomalies.",
            )

        run_arena_btn = st.button("🚀 Launch AutoML & Neural Arena Benchmark", use_container_width=True)

        if run_arena_btn:
            with st.spinner("Executing neural training, cross-validation, and hyperparameter benchmarking..."):
                target_col_arg = None if "None" in selected_target else selected_target
                arena_engine = AutoMLArena(test_size=test_split, contamination=contamination_rate)
                arena_res = arena_engine.run(
                    df=current_df,
                    target_column=target_col_arg,
                    test_size=test_split,
                    contamination=contamination_rate,
                )
                st.session_state.arena_result = arena_res
                # Trigger downstream tests automatically
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
                st.success(f"Arena benchmark finished! Winning Architecture: **{arena_res.winning_model_name}**.")

        # DISPLAY RESULTS
        if st.session_state.arena_result is not None:
            res: ArenaResult = st.session_state.arena_result
            st.markdown("---")

            # ---------------------------------------------
            # SECOPS ACTION MODE DISPLAY
            # ---------------------------------------------
            if is_secops_mode:
                st.markdown("#### 🛡️ SecOps Threat Defense Scorecard")

                # Plain-English metrics
                primary_metrics = res.model_metrics.get(res.winning_model_name, {})
                secops_view = translate_metrics_to_secops(primary_metrics, res.problem_type)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(
                        f"""
                        <div class="cyber-card">
                            <div class="metric-label">Winning Defense Engine</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #00e5ff;">{res.winning_model_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div class="cyber-card">
                            <div class="metric-label">Threat Detection Efficacy</div>
                            <div class="metric-value neon-teal">{secops_view['headline']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"""
                        <div class="cyber-card">
                            <div class="metric-label">Threat Severity Status</div>
                            <div style="margin-top: 8px;">{secops_view['badge']['badge_html']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.markdown(
                        f"""
                        <div class="cyber-card">
                            <div class="metric-label">Operational Status</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #00ffcc;">{secops_view.get('operational_readiness', 'Active')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div class="cyber-card">
                        <div style="font-weight: 700; color: #00e5ff; margin-bottom: 6px;">
                            📋 Plain-English Defense Summary
                        </div>
                        <p style="margin: 0; font-size: 1.05rem; color: #f1f5f9;">
                            {secops_view['plain_summary']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if res.anomaly_result is not None:
                    st.markdown("#### 🚨 Zero-Day / High-Risk Anomaly Telemetry")
                    anom = res.anomaly_result
                    st.markdown(
                        f"""
                        <div class="cyber-card-danger">
                            <div style="font-weight: 700; color: #ff0055; margin-bottom: 6px;">
                                ⚠️ {anom.anomaly_count} Zero-Day Outliers Surpassed Reconstruction Threshold
                            </div>
                            <p style="margin: 0; color: #e2e8f0;">
                                {anom.plain_english_summary}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.dataframe(anom.anomalies_df.head(10), use_container_width=True)

            # ---------------------------------------------
            # DATA SCIENCE DEEP-DIVE MODE DISPLAY
            # ---------------------------------------------
            else:
                st.markdown("#### 🔬 Data Science Neural & Model Leaderboard")
                st.dataframe(res.leaderboard_df, use_container_width=True)

                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    if res.problem_type == "classification":
                        radar_fig = create_model_comparison_radar_chart(res.model_metrics)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.markdown("**Metric Distribution Matrix**")
                        st.json(res.model_metrics)

                with col_chart2:
                    # Neural Network Loss Curves
                    has_loss = False
                    for m_name, history in res.loss_curves.items():
                        if history.get("train_loss"):
                            loss_fig = create_neural_loss_curve(
                                train_loss=history["train_loss"],
                                val_loss=history.get("val_loss"),
                                architecture_name=m_name,
                            )
                            st.plotly_chart(loss_fig, use_container_width=True)
                            has_loss = True
                            break

                    if not has_loss and res.anomaly_result is not None:
                        recon_fig = create_anomaly_reconstruction_plot(
                            errors=res.anomaly_result.reconstruction_errors,
                            threshold=res.anomaly_result.reconstruction_threshold,
                            anomaly_indices=res.anomaly_result.anomaly_indices,
                        )
                        st.plotly_chart(recon_fig, use_container_width=True)


# ==========================================================
# TAB 3: ADVERSARIAL & XAI DIAGNOSTICS
# ==========================================================
with tab_adv_xai:
    st.markdown("### 🛡️ Adversarial Robustness & Explainable AI (XAI) Diagnostics")
    st.write(
        "Subject the winning model to adversarial evasion noise injection (FGSM/PGD) and decompose "
        "black-box decision boundaries into plain-English root causes using SHAP."
    )

    if st.session_state.arena_result is None:
        st.warning("⚠️ No trained models available. Launch the AutoML Arena in the previous tab first.")
    else:
        res = st.session_state.arena_result

        # Run adversarial test if not present
        if st.session_state.adv_results is None:
            adv_tester = AdversarialRobustnessTester()
            st.session_state.adv_results = adv_tester.evaluate_evasion_robustness(
                model=res.winning_model,
                X_test=res.X_test,
                y_test=res.y_test,
            )

        # Run XAI if not present
        if st.session_state.xai_results is None:
            explainer_engine = SHAPExplainer()
            st.session_state.xai_results = explainer_engine.explain_global(
                model=res.winning_model,
                X_sample=res.X_test,
            )

        adv_res = st.session_state.adv_results
        xai_res = st.session_state.xai_results

        # ADVERSARIAL ROBUSTNESS SECTION
        st.markdown("#### 1. Adversarial Evasion & Perturbation Resistance")

        adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 2])
        with adv_col1:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div class="metric-label">Evasion Resistance Score</div>
                    <div class="metric-value neon-blue">{adv_res['evasion_resistance_score']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with adv_col2:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div class="metric-label">Model Hackability Status</div>
                    <div style="margin-top: 8px;">{adv_res['threat_badge']['badge_html']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with adv_col3:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div style="font-weight: 700; color: #ffaa00; margin-bottom: 4px;">⚠️ SecOps Threat Advisory</div>
                    <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">{adv_res['secops_warning']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Adversarial Decay Plot
        decay_fig = create_adversarial_decay_curve(adv_res["decay_curve"])
        st.plotly_chart(decay_fig, use_container_width=True)

        st.markdown("---")

        # EXPLAINABLE AI (XAI) SECTION
        st.markdown("#### 2. Explainable AI: SHAP Mathematical Feature Contributions")

        xai_col_chart, xai_col_text = st.columns([3, 2])

        with xai_col_chart:
            top_f = [f["feature"] for f in xai_res["top_features"]]
            top_i = [f["importance"] for f in xai_res["top_features"]]
            shap_fig = create_shap_feature_importance_bar(top_f, top_i)
            st.plotly_chart(shap_fig, use_container_width=True)

        with xai_col_text:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div style="font-weight: 700; color: #00ffcc; margin-bottom: 8px;">
                        🔍 Plain-English Root Cause Attribution
                    </div>
                    <p style="margin: 0 0 12px 0; font-size: 0.95rem; color: #f1f5f9; line-height: 1.6;">
                        {xai_res['plain_english_summary']}
                    </p>
                    <div class="metric-label" style="margin-top: 15px;">Evaluated Feature Weights:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Display clean feature weight breakdown
            st.dataframe(pd.DataFrame(xai_res["top_features"]), use_container_width=True)


# ==========================================================
# TAB 4: MITRE ATT&CK & RULE GENERATOR
# ==========================================================
with tab_mitre:
    st.markdown("### 🤖 Automated MITRE ATT&CK Mapping & Live Defensive Rule Generation")
    st.write(
        "Synthesize production-ready **YARA rules**, **Snort/Suricata IDS signatures**, and **Linux iptables commands** "
        "directly from the model's mathematical decision boundaries and zero-day log anomalies."
    )

    if st.session_state.arena_result is None:
        st.warning("⚠️ Train a model in the 'AutoML Arena' first to generate live defensive rules.")
    else:
        res = st.session_state.arena_result

        generate_rules_btn = st.button("⚡ Synthesize MITRE Mappings & Defensive Rules", use_container_width=True)

        if generate_rules_btn or st.session_state.mitre_results is None:
            with st.spinner("Compiling threat telemetry and synthesizing firewall & IDS rules..."):
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

            # Generator engine badge
            gen_mode = m_res.get("generator_mode", "Local Cyber Heuristic Synthesis Engine")
            st.markdown(
                f"""
                <div style="margin-bottom: 15px;">
                    <span style="background: rgba(0, 255, 204, 0.15); border: 1px solid #00ffcc; color: #00ffcc; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">
                        ⚡ Rule Generator: {gen_mode}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # MITRE ATT&CK Cards
            st.markdown("#### 🎯 MITRE ATT&CK Matrix Mapping")
            mappings = m_res.get("mitre_mappings", [])
            mitre_cols = st.columns(len(mappings) if len(mappings) <= 3 else 3)
            for idx, mapping in enumerate(mappings[:3]):
                with mitre_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="cyber-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-weight: 800; color: #00e5ff; font-size: 1.1rem;">{mapping.get('technique_id', 'T1046')}</span>
                                <span style="background: rgba(255,0,85,0.2); color: #ff0055; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; font-weight: 700;">{mapping.get('tactic', 'Discovery')}</span>
                            </div>
                            <div style="font-weight: 700; color: #ffffff; margin-bottom: 6px;">{mapping.get('technique_name', 'Network Service Discovery')}</div>
                            <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                                {mapping.get('justification', '')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # LIVE DEFENSIVE CODE GENERATION
            st.markdown("#### 🛡️ Production Defensive Rules & Command Generator")

            rule_tab_yara, rule_tab_snort, rule_tab_firewall, rule_tab_sop = st.tabs([
                "📜 YARA Rule",
                "📡 Snort / Suricata IDS Signature",
                "🔥 Linux iptables / UFW Commands",
                "📋 SOC Analyst Remediation Plan",
            ])

            with rule_tab_yara:
                st.markdown("**Copy-Pasteable YARA Detection Rule:**")
                yara_content = m_res.get("yara_rule", "")
                if show_demasked_view:
                    yara_content = st.session_state.privacy_masker.demask_text(yara_content)
                st.code(yara_content, language="c")

            with rule_tab_snort:
                st.markdown("**Copy-Pasteable Snort/Suricata Network Signature:**")
                snort_content = m_res.get("snort_signature", "")
                if show_demasked_view:
                    snort_content = st.session_state.privacy_masker.demask_text(snort_content)
                st.code(snort_content, language="bash")

            with rule_tab_firewall:
                st.markdown("**Linux Kernel iptables & UFW Enforcement Commands:**")
                fw_cmds = m_res.get("firewall_commands", [])
                fw_text = "\n".join(fw_cmds)
                if show_demasked_view:
                    fw_text = st.session_state.privacy_masker.demask_text(fw_text)
                st.code(fw_text, language="bash")

            with rule_tab_sop:
                st.markdown("**SOC Analyst Tier-1/Tier-2 Action Plan:**")
                sop_steps = m_res.get("secops_action_plan", [])
                for step in sop_steps:
                    st.markdown(f"- **{step}**")
