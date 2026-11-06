# Adversa: Adversarial-Resistant SecOps AutoML & Neural Arena

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Streamlit-1.40+-ff4b4b.svg)](https://streamlit.io/)
[![API Backend](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

> **A production-ready, open-source Adversarial-Resistant, Explainable Automated Machine Learning (AutoML) and Deep Neural Network platform engineered for cybersecurity telemetry, threat hunting, and incident response.**

---

## 🏛️ System Architecture

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                  INGRESS: MULTI-FORMAT PARSING                         │
   │           [CSV In-Memory]  •  [JSON / NDJSON]  •  [PDF Mining]         │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │ (io.BytesIO stream)
   ┌──────────────────────────────────▼─────────────────────────────────────┐
   │               ZERO-TRUST LOCAL CRYPTOGRAPHIC ANONYMIZER                │
   │    • Masks RFC1918 IPv4/IPv6 (e.g. 192.168.1.50 -> INTERNAL_HOST_001) │
   │    • Obfuscates MAC, emails, sensitive intranet URLs                   │
   │    • In-memory Bidirectional Vault (Demasking strictly on local UI)    │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Sanitized & Masked Data)
   ┌──────────────────────────────────▼─────────────────────────────────────┐
   │                    CLEANING & JUNK METADATA FILTER                     │
   │  • Deduplication  • Zero-Variance Pruning  • Imputation  • Scaler/OHE  │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼─────────────────────────────────────┐
   │                     ADAPTIVE NEURAL & AUTOML ARENA                     │
   │ ┌───────────────────────────┬─────────────────────────┬──────────────┐ │
   │ │  Mode A: Classification   │   Mode B: Regression    │   Mode C:    │ │
   │ │ • Logistic Regression     │ • Ridge Regression      │ Unsupervised │ │
   │ │ • Random Forest           │ • Random Forest Reg.    │ Deep PyTorch │ │
   │ │ • XGBoost Gradient Boost  │ • Deep MLP Regressor    │ Autoencoder  │ │
   │ │ • Deep PyTorch MLP Net    │                         │ Anomaly Det. │ │
   │ └───────────────────────────┴─────────────────────────┴──────────────┘ │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Winning Model & Decision Boundaries)
   ┌──────────────────────────────────▼─────────────────────────────────────┐
   │                   NEXT-GEN ADVERSARIAL & XAI ENGINES                   │
   │  • Adversarial Stress Test: FGSM/PGD Evasion Attack Simulation         │
   │  • Evasion Resistance Score (0-100%) & "Hackability Warning"           │
   │  • SHAP Explainer: Exact feature contributions -> Plain English causes │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │ (Sanitized Cyber Telemetry Payload)
   ┌──────────────────────────────────▼─────────────────────────────────────┐
   │         AI THREAT INTEL AGENT / MITRE DEFENSIVE RULE GENERATOR         │
   │  • MITRE ATT&CK Matrix Mapping (Tactic & Technique IDs: T1046, T1071)  │
   │  • Synthesizes Live YARA Rules                                         │
   │  • Synthesizes Snort / Suricata IDS Signatures                         │
   │  • Generates Linux Kernel iptables & UFW firewall block commands       │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 The 5 Core Innovations

### 1. Dual Persona Interface: "SecOps Action Mode" vs "Data Science Deep-Dive Mode"
- **🛡️ SecOps Action Mode (Default):** Abstract away all raw mathematical jargon ($R^2$, MSE, log loss, Shapley equations) and translate metrics into plain-English **Threat Severity Badges (Critical/High/Medium/Low)**, percentage impact bars, MITRE ATT&CK cards, and copy-paste remediation commands.
- **🔬 Data Science Deep-Dive Mode:** Reveal raw model leaderboards, loss convergence curves, ROC-AUC matrices, hyperparameter controls, and raw SHAP attributions.

### 2. Zero-Trust Local Privacy Mode (PII & IP Obfuscation)
- Preprocessing layer cryptographic anonymizer (`PrivacyMasker`) automatically detects and masks internal RFC 1918 IPs (`192.168.1.50` -> `INTERNAL_HOST_001`), MAC addresses, user emails, and sensitive URLs before data reaches ML models or external LLM APIs.
- Features a **local de-masking toggle** that restores genuine hostnames strictly inside the user's browser, maintaining SOC2, HIPAA, and GDPR compliance.

### 3. Adversarial ML & Data Poisoning Defense
- Automatically subjects the winning model and Deep Neural Network to adversarial stress tests using **Fast Gradient Sign Method (FGSM)** and feature perturbation.
- Computes an **Evasion Resistance Score (0-100%)** alongside standard metrics, alerting analysts in plain English if subtle packet size or URL mutations could bypass network defenses.

### 4. Explainable AI (XAI) for Deep Neural Networks via SHAP
- Solves the deep learning "Black Box" problem with exact SHAP (SHapley Additive exPlanations).
- Translates mathematical gradient vectors into actionable cyber narratives (e.g., *"Flagged primarily due to unusual RDP Port 3389 traffic [64% impact] and prolonged connection duration [24% impact]"*).

### 5. Automated MITRE ATT&CK Mapping & Live Defensive Rule Generation
- Packages model boundaries, Autoencoder zero-day anomaly vectors, and SHAP weights into a sanitized payload for the **AI Threat Intelligence Agent** (with an offline cybersecurity heuristic fallback).
- Outputs production-ready, copy-pasteable:
  - **YARA Rules** with exact string tokens and hex patterns.
  - **Snort / Suricata IDS Signatures** with tailored flow and port targets.
  - **Linux `iptables` and `ufw` block commands**.

---

## Quickstart & Local Installation

### Prerequisites
- Python 3.10+ (macOS, Linux, or Windows)
- Git

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/cyber-automl-arena.git
cd cyber-automl-arena

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env to add your GEMINI_API_KEY if desired (Optional)
```

### 4. Generate Built-in Sample Datasets

```bash
python generate_samples.py
```
*Generates messy `phishing_data.csv` (supervised) and `firewall_logs.json` (unsupervised) in `data/samples/`.*

---

## 🖥️ Running the Platform

### Option A: Modern Streamlit Dashboard (SecOps Dark Mode)

```bash
streamlit run app_streamlit.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option B: Decoupled FastAPI Backend

```bash
uvicorn app_fastapi.py --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger/OpenAPI documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## ☁️ Streamlit Cloud Deployment Guide

1. Push this repository to your GitHub account (see automated script below).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New App**.
3. Select your repository, set the branch to `main`, and specify `app_streamlit.py` as the **Main file path**.
4. In **Advanced Settings**, add your environment variables:
   - `GEMINI_API_KEY`: *(Optional threat intelligence API key)*
5. Click **Deploy!**

---

## Security & Privacy Disclaimer

> [!CAUTION]
> **CRITICAL SECURITY NOTICE FOR SECURITY ANALYSTS:**
> **DO NOT** upload classified, unredacted production network pcaps or proprietary internal logs to public hosted demo links without ensuring that **Zero-Trust Privacy Mode** is toggled ON. When enabled, all internal IPv4, IPv6, MAC addresses, and corporate email addresses are cryptographically masked in-memory before entering analysis pipelines. All file uploads are handled strictly in volatile RAM (`io.BytesIO`) and are never written to disk storage.

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.
