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

## 📊 Datasets & Public Origins

Adversa comes pre-configured with 5 cybersecurity telemetry datasets located in `data/samples/` (and reproducibly generated via `generate_samples.py`), modeled after standard public benchmarks:

1. **Phishing Email & URL Telemetry (`phishing_data.csv`)**:
   - *Origin / Source*: Modeled after the **UCI Machine Learning Repository Phishing Websites Dataset** and **Kaggle Phishing Email Classification Dataset**.
   - *Features*: URL length, subdomain depth, entropy, IP in URL, SSL age, suspicious keywords, and target threat label.
   - *Task*: Supervised classification.
2. **Network Firewall Logs (`firewall_logs.json`)**:
   - *Origin / Source*: Modeled after the **Kaggle Internet Firewall Data (UCI)** and Palo Alto firewall telemetry logs.
   - *Features*: Source/destination IP, port, action (ALLOW/DENY), bytes sent/received, packet rate, duration.
   - *Task*: Unsupervised anomaly detection & classification.
3. **Authentication & SSH Syslog Records (`auth_syslog.log`)**:
   - *Origin / Source*: Modeled after **SecRepo Linux Auth & Syslog RFC 3164/5424** intrusion log benchmarks.
   - *Features*: Timestamp, hostname, PID, raw syslog message, source IP, action.
   - *Task*: Unsupervised zero-day outlier detection with high-cardinality log parsing.
4. **Network Packet Capture (`network_traffic.pcap`)**:
   - *Origin / Source*: Synthesized binary PCAP network flows adhering to **Wireshark / Libpcap** packet specifications.
   - *Features*: Ethernet frame, IP packet header, TCP/UDP ports, payload byte entropy, flags.
   - *Task*: Binary protocol inspection and network intrusion classification.
5. **Security Data Lake Flows (`security_telemetry.parquet`)**:
   - *Origin / Source*: Apache Parquet columnar data lake telemetry based on **AWS VPC Flow Logs**.
   - *Features*: Flow index, duration, bytes, packet count, protocol, TCP flags.
   - *Task*: High-throughput columnar security stream analysis.

---

## 💡 What I Learned (Reflective AI Engineering Takeaways)

Working with an AI coding assistant (Google Antigravity) to architect and deploy a production-ready application provided several fundamental insights:

1. **Iterative Problem Specification beats Generic Prompts**:
   - Giving the AI specific architectural constraints (e.g. `io.BytesIO` RAM-only buffers, RFC 1918 HMAC masking, 128 max features in `float32`) prevented fragile solutions and ensured security by design.
2. **Never Accept AI Output Blindly (The `ArrayMemoryError` Lesson)**:
   - When ingesting raw syslog files, the AI initially used standard `pd.get_dummies()`. For high-cardinality text (UUIDs, timestamps, raw messages), this attempted to allocate 106 Terabytes of RAM, crashing NumPy. Reviewing the stack trace and guiding the AI to extract structural signals (`_len` and `_digits`) and apply top-15 category binning eliminated the memory spike entirely.
3. **Design System & Contrast Rigor**:
   - Initial AI-generated light themes suffered from faint headings and white text inside inputs. Enforcing explicit CSS variables, `-webkit-text-fill-color`, and WCAG AA contrast standards transformed the UI into a clean, modern analytics workspace.
4. **Air-Gapped Reliability with Heuristic Fallbacks**:
   - Relying solely on cloud LLM APIs makes security software brittle. Implementing deterministic offline heuristic fallbacks guaranteed 100% operational readiness even without internet connectivity or API keys.

---

## Quickstart & Local Installation

### Prerequisites
- Python 3.10+ (macOS, Linux, or Windows)
- Git

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/srivatsa108/Adversarial-Resistant-SecOps-AutoML-Neural-Arena.git
cd Adversarial-Resistant-SecOps-AutoML-Neural-Arena

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
# Edit .env to add your threat intelligence API key if desired (Optional)
```

### 4. Generate Built-in Sample Datasets

```bash
python generate_samples.py
```

---

## 🖥️ Running the Platform

### Option A: Modern Streamlit Dashboard (Dark & Light Theme)

```bash
streamlit run app_streamlit.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option B: Decoupled FastAPI Backend

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger/OpenAPI documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Option C: Run Full Automated Verification Suite

```bash
PYTHONPATH=. pytest tests/ -v
```

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
