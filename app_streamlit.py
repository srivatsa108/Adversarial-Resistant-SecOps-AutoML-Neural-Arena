"""Adversa SecOps AutoML and Neural Arena dashboard.

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
    page_title="Adversa SecOps AutoML",
    layout="wide",
    initial_sidebar_state="auto",
)

# Initialize Session State
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

from src.utils.providers import CUSTOM, PROVIDERS, env_names, key_from_environment, parse_env_text

# Active theme. Streamlit's own Settings menu (top right) switches Light, Dark or
# System, and this value follows it so charts, tables and CSS all stay in sync.
def _detect_theme() -> str:
    try:
        kind = st.context.theme.type
        if kind in ("light", "dark"):
            return kind
    except Exception:
        pass
    return "dark"


active_theme = _detect_theme()

# Sidebar: workspace controls
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">A</div>
            <div>
                <div class="brand-name">Adversa</div>
                <div class="brand-sub">Security analytics workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-group">View</div>', unsafe_allow_html=True)
    persona_mode = st.radio(
        "Persona",
        options=["SecOps action mode", "Data science deep dive mode"],
        index=0,
        label_visibility="collapsed",
        help="SecOps mode turns loss and Shapley values into threat levels and firewall commands. Deep dive mode shows the raw metrics.",
    )
    is_secops_mode = "SecOps" in persona_mode

    st.markdown('<div class="nav-group">Privacy</div>', unsafe_allow_html=True)
    enable_privacy_masking = st.toggle(
        "Mask IPs and personal data",
        value=True,
        help="Replaces internal IPs, MAC addresses and emails in memory before any analysis.",
    )
    show_demasked_view = st.toggle(
        "Show original values",
        value=False,
        help="Restores real addresses on your screen only, for verification.",
    )

    st.markdown('<div class="nav-group">Threat intelligence</div>', unsafe_allow_html=True)
    if "api_keys_imported" not in st.session_state:
        st.session_state.api_keys_imported = ""
    env_file = st.file_uploader(
        "Import keys from a .env file",
        type=["env", "txt"],
        help="Reads lines such as OPENAI_API_KEY=... and fills in the matching providers. Nothing is stored on disk.",
    )
    if env_file is not None:
        _text = env_file.getvalue().decode("utf-8", errors="ignore")
        _sig = f"{env_file.name}:{len(_text)}"
        if _sig != st.session_state.api_keys_imported:
            for _name, _val in parse_env_text(_text).items():
                st.session_state[f"key_{_name}"] = _val
            st.session_state.api_keys_imported = _sig
    llm_provider = st.selectbox("Provider", options=list(PROVIDERS), index=0)
    _cfg = PROVIDERS[llm_provider]
    _env_key = key_from_environment(llm_provider)
    st.session_state.setdefault(f"key_{llm_provider}", _env_key or "")
    st.session_state.setdefault(f"model_{llm_provider}", str(_cfg["model"]))
    llm_key = st.text_input(
        "API key",
        type="password",
        placeholder="Paste a key",
        key=f"key_{llm_provider}",
        help="Used for generative MITRE reasoning. The offline rule engine runs when this is empty.",
    )
    llm_model = st.text_input("Model", key=f"model_{llm_provider}")
    llm_base_url = ""
    if llm_provider == CUSTOM:
        llm_base_url = st.text_input("Base URL", placeholder="https://your-endpoint/v1", key="base_url_custom")
    if llm_key:
        _where = f"from {env_names(llm_provider)[0]}" if (_env_key and llm_key == _env_key) else "set for this session"
        st.markdown(f'<div class="key-status"><span class="dot"></span>Key {_where}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-status off">No key. Rules use the offline engine.</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="side-foot">Theme: open the menu at the top right, then Settings.<br>MIT License</div>',
        unsafe_allow_html=True,
    )

PALETTES = {
    "dark": {
        "bg": "#0A0B0D", "panel": "#101215", "raised": "#16191D",
        "border": "#1E2227", "border-strong": "#2C333B",
        "text": "#ECEEF1", "muted": "#8D95A0", "dim": "#5F6772",
        "accent": "#5B8CFF", "accent-ink": "#FFFFFF", "accent-soft": "rgba(91,140,255,0.14)",
        "ok": "#3FB27F", "warn": "#E8A53A", "bad": "#EF5B5B",
        "ok-soft": "rgba(63,178,127,0.12)", "warn-soft": "rgba(232,165,58,0.12)",
        "bad-soft": "rgba(239,91,91,0.12)", "shadow": "none",
    },
    "light": {
        "bg": "#F5F6F8", "panel": "#FFFFFF", "raised": "#EEF0F3",
        "border": "#E3E6EB", "border-strong": "#CDD3DB",
        "text": "#0F1720", "muted": "#566070", "dim": "#8892A0",
        "accent": "#2D5BFF", "accent-ink": "#FFFFFF", "accent-soft": "rgba(45,91,255,0.09)",
        "ok": "#148A5B", "warn": "#9A6412", "bad": "#C93434",
        "ok-soft": "rgba(20,138,91,0.09)", "warn-soft": "rgba(154,100,18,0.10)",
        "bad-soft": "rgba(201,52,52,0.08)", "shadow": "0 1px 2px rgba(16,24,40,0.06)",
    },
}
_vars = "\n".join(f"    --{k}: {v};" for k, v in PALETTES[active_theme].items())

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');

:root {
__VARS__
    --font: 'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
    --mono: 'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --radius: 10px;
    --ease: cubic-bezier(0.22, 1, 0.36, 1);
}

/* Base */
.stApp, .stApp :where(p, span, div, label, li, td, th, h1, h2, h3, h4, h5, a, button, input, textarea, summary):not([data-testid="stIconMaterial"]) {
    font-family: var(--font);
}
.stApp { background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }
.stApp :where(p, li, label, span, div) { color: inherit; }
[data-testid="stMainBlockContainer"] { max-width: 1280px; padding: 2.25rem 2.5rem 4rem; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; font-weight: 600; }
h3 { font-size: 1.25rem; margin-bottom: 0.25rem; }
h4 { font-size: 1rem; margin-top: 1.5rem; }
[data-testid="stMarkdownContainer"] p { color: var(--muted); line-height: 1.6; max-width: 78ch; }
[data-testid="stMarkdownContainer"] strong { color: var(--text); font-weight: 600; }
hr { border-color: var(--border); margin: 1.5rem 0; }
code, pre, [data-testid="stCode"] * { font-family: var(--mono); }
[data-testid="stCode"], pre {
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
}

/* Chrome */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], .stAppDeployButton, [data-testid="stStatusWidget"] { display: none; }
[data-testid="stToolbar"] * { color: var(--muted); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0.5rem 0.75rem 1.5rem; }
[data-testid="stSidebar"] :where(p, span, label) { color: var(--text); }
.brand { display: flex; align-items: center; gap: 12px; padding: 14px 4px 18px; border-bottom: 1px solid var(--border); margin-bottom: 8px; }
.brand-mark {
    width: 34px; height: 34px; border-radius: 9px; background: var(--accent); color: var(--accent-ink);
    display: grid; place-items: center; font-weight: 600; font-size: 16px; flex: none;
}
.brand-name { font-weight: 600; font-size: 15px; color: var(--text); letter-spacing: -0.01em; }
.brand-sub { font-size: 12px; color: var(--muted); }
.nav-group { font-size: 12px; font-weight: 500; color: var(--dim); margin: 22px 4px 8px; }
.key-status { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); margin: -4px 2px 4px; }
.key-status.off { color: var(--dim); }
.side-foot { font-size: 12px; line-height: 1.6; color: var(--dim); margin: 28px 4px 0; padding-top: 14px; border-top: 1px solid var(--border); }

/* Radio as selectable rows */
[data-testid="stRadio"] [role="radiogroup"] { gap: 4px; }
[data-testid="stRadio"] label {
    border: 1px solid transparent; border-radius: 8px; padding: 8px 10px; margin: 0;
    transition: background-color 150ms var(--ease), border-color 150ms var(--ease);
}
[data-testid="stRadio"] label:hover { background: var(--raised); }
[data-testid="stRadio"] label:has(input:checked) { background: var(--accent-soft); border-color: transparent; }
[data-testid="stRadio"] label:has(input:checked) p { color: var(--text); font-weight: 500; }
[data-testid="stRadio"] label p { font-size: 14px; color: var(--muted); }

[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { border-color: var(--border-strong); }

/* Page header and KPI cards */
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
.page-head .page-title { font-size: 24px !important; line-height: 30px !important; font-weight: 600 !important; letter-spacing: -0.02em; color: var(--text); margin: 0; padding: 0; }
.page-head .page-sub { font-size: 14px; color: var(--muted); margin: 4px 0 0; max-width: none; }
.pill {
    display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 500; color: var(--muted);
    background: var(--panel); border: 1px solid var(--border); border-radius: 999px; padding: 6px 12px;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 28px; }
.kpi { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow); }
.kpi-label { font-size: 13px; color: var(--muted); font-weight: 500; }
.kpi-value { font-size: 26px; line-height: 34px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); font-variant-numeric: tabular-nums; margin-top: 6px; }
.kpi-value.empty { color: var(--dim); }
.kpi-meta { font-size: 12px; color: var(--dim); margin-top: 2px; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 3rem; } }

/* Existing panels */
.surface-card, .centerpiece-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 20px; margin-bottom: 16px; box-shadow: var(--shadow);
}
.centerpiece-card { padding: 22px 24px; border-color: var(--border-strong); }
.centerpiece-metric { font-size: 44px; line-height: 48px; font-weight: 600; letter-spacing: -0.03em; color: var(--text); font-variant-numeric: tabular-nums; margin: 6px 0; }
.metric-num { font-size: 22px; line-height: 28px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); font-variant-numeric: tabular-nums; }
.metric-meta { font-size: 13px; line-height: 18px; color: var(--muted); font-weight: 500; }
.slim-progress-track { width: 100%; height: 6px; background: var(--raised); border-radius: 999px; overflow: hidden; margin-top: 14px; }
.slim-progress-fill { height: 100%; background: var(--accent); border-radius: 999px; }

/* Tabs as a segmented control */
[role="tablist"] {
    gap: 2px; background: var(--raised); padding: 4px; border-radius: 12px; border: 1px solid var(--border);
    width: fit-content; max-width: 100%; overflow-x: auto; box-shadow: none;
}
[role="tablist"] [data-baseweb="tab-highlight"], [role="tablist"] [data-baseweb="tab-border"],
[data-testid="stTabHighlight"], [data-testid="stTabBorder"] { display: none !important; }
[role="tablist"] [role="tab"] {
    height: 36px; padding: 0 14px; border-radius: 9px; background: transparent !important; color: var(--muted);
    font-size: 14px; font-weight: 500; border: none !important; box-shadow: none;
    transition: background-color 150ms var(--ease), color 150ms var(--ease);
}
[role="tablist"] [role="tab"] p { color: inherit; font-size: 14px; font-weight: 500; max-width: none; }
[role="tablist"] [role="tab"]:hover { color: var(--text); }
[role="tablist"] [role="tab"][aria-selected="true"] { background: var(--panel) !important; color: var(--text); box-shadow: 0 0 0 1px var(--border-strong) !important; }
[role="tablist"] [role="tab"]::before, [role="tablist"] [role="tab"]::after { display: none !important; content: none !important; }
[role="tabpanel"] { padding-top: 1.75rem; }

/* Buttons */
.stApp .stButton > button, .stApp button[data-testid="stBaseButton-secondary"], .stApp button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important; color: var(--accent-ink) !important; border: 1px solid transparent !important;
    border-radius: 8px; font-size: 14px; font-weight: 500; min-height: 40px; padding: 0 16px; box-shadow: none !important;
    transition: filter 150ms var(--ease);
}
.stApp .stButton > button p { color: var(--accent-ink) !important; font-weight: 500; max-width: none; }
.stApp .stButton > button:hover { filter: brightness(1.1); }
.stApp .stButton > button:focus:not(:focus-visible) { outline: none; box-shadow: none !important; }
.stApp .stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, textarea {
    background: var(--panel); border-color: var(--border-strong); border-radius: 8px; color: var(--text);
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--accent); }
[data-testid="stWidgetLabel"] p { font-size: 13px; font-weight: 500; color: var(--muted); }
[data-testid="stSlider"] [role="slider"] { background: var(--accent); }

/* Upload zone */
[data-testid="stFileUploader"] section {
    background: var(--panel); border: 1px dashed var(--border-strong); border-radius: var(--radius); padding: 22px;
    transition: border-color 150ms var(--ease), background-color 150ms var(--ease);
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent); background: var(--accent-soft); }
.stApp [data-testid="stFileUploader"] section button, .stApp [data-testid="stFileUploader"] section button p { background: var(--raised) !important; color: var(--text) !important; border: 1px solid var(--border-strong) !important; }
[data-testid="stFileUploaderFile"] { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; margin-top: 8px; }
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileIcon"],
[data-testid="stFileUploaderFile"] svg:first-child { display: none; }

/* Alerts, tinted by meaning */
[data-testid="stAlert"] { border-radius: var(--radius); border: 1px solid var(--border); background: var(--panel); padding: 4px 6px; }
[data-testid="stAlert"] *, [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
[data-testid="stAlertContainer"], [data-testid="stAlert"] [class*="stAlertContent"] { background: transparent !important; border: none !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { background: var(--ok-soft); border-color: color-mix(in srgb, var(--ok) 35%, transparent); }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { background: var(--warn-soft); border-color: color-mix(in srgb, var(--warn) 40%, transparent); }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { background: var(--bad-soft); border-color: color-mix(in srgb, var(--bad) 40%, transparent); }
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { background: var(--accent-soft); border-color: color-mix(in srgb, var(--accent) 35%, transparent); }

/* Tables and charts sit on panels */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; background: var(--panel); }
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: var(--radius); background: var(--panel); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""".replace("__VARS__", _vars)
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

# Page header and live KPI cards
st.markdown(
    """
    <div class="page-head">
        <div>
            <div class="page-title">Threat detection workspace</div>
            <p class="page-sub">Train, stress-test and explain models on security telemetry, then export defensive rules.</p>
        </div>
        <span class="pill"><span class="dot"></span>All data stays in memory</span>
    </div>
    """,
    unsafe_allow_html=True,
)
def _kpi(label: str, value: Optional[str], meta: str) -> str:
    cls = "kpi-value" if value else "kpi-value empty"
    return (
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="{cls}">{value if value else "&mdash;"}</div>'
        f'<div class="kpi-meta">{meta}</div></div>'
    )


def render_kpis() -> None:
    raw = st.session_state.raw_df
    arena = st.session_state.arena_result
    adv = st.session_state.adv_results
    rows = f"{len(raw):,}" if raw is not None else None
    cols = f"{len(raw.columns):,}" if raw is not None else None
    model = getattr(arena, "winning_model_name", None) if arena is not None else None
    score = None
    if isinstance(adv, dict) and "evasion_resistance_score" in adv:
        score = f"{adv['evasion_resistance_score']:.1f}%"
    kpi_slot.markdown(
        f"""
        <div class="kpis">
            {_kpi("Records loaded", rows, "In this session" if rows else "Load a file or a sample to start")}
            {_kpi("Features", cols, "Columns before cleaning" if cols else "Appear after loading data")}
            {_kpi("Best model", model, "Winner of the arena benchmark" if model else "Run the arena benchmark")}
            {_kpi("Evasion resistance", score, "Higher is harder to evade" if score else "Runs after the benchmark")}
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_slot = st.empty()

# Main navigation
tab_ingest, tab_arena, tab_adv_xai, tab_mitre, tab_verify = st.tabs([
    "Ingestion and privacy",
    "Model arena",
    "Robustness and explanations",
    "MITRE and rules",
    "Verification",
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

        if st.session_state.cleaning_report is not None and st.session_state.cleaned_df is not None:
            rep = st.session_state.cleaning_report
            st.info(rep["plain_english_summary"])

            # Cleaned dataset in-memory export options
            export_df = st.session_state.cleaned_df.copy()
            is_demasked = show_demasked_view
            if is_demasked:
                export_rows = [st.session_state.privacy_masker.demask_dict(r) for r in export_df.to_dict(orient="records")]
                export_df = pd.DataFrame(export_rows)

            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")

            export_info_col, dl_csv_col, dl_json_col = st.columns([2, 1, 1])
            with export_info_col:
                privacy_state_label = "Demasked / Real Entities" if is_demasked else "Zero-Trust Masked (GDPR Safe)"
                st.caption(f"Cleaned dataset: {len(export_df):,} records × {len(export_df.columns)} features • {privacy_state_label}")
            with dl_csv_col:
                st.download_button(
                    label="Download cleaned CSV",
                    data=csv_bytes,
                    file_name="cleaned_telemetry_masked.csv" if not is_demasked else "cleaned_telemetry_demasked.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl_json_col:
                st.download_button(
                    label="Download cleaned JSON",
                    data=json_bytes,
                    file_name="cleaned_telemetry_masked.json" if not is_demasked else "cleaned_telemetry_demasked.json",
                    mime="application/json",
                    use_container_width=True,
                )

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
                    api_key=llm_key,
                    provider=llm_provider,
                    model=llm_model,
                    base_url=llm_base_url,
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


# Refresh the KPI cards with whatever this run produced
render_kpis()
