# Project Submission: AegisOps - Adversarial-Resistant SecOps AutoML & Neural Arena

**GitHub Repository:** https://github.com/srivatsa108/Adversarial-Resistant-SecOps-AutoML-Neural-Arena
**Primary Author:** Srivatsa (https://github.com/srivatsa108)
**License:** MIT
**Tech Stack:** Python 3.10+, PyTorch, Scikit-Learn, XGBoost, Scapy, SHAP, Streamlit, FastAPI, Plotly

---

## Summary and problem statement

Classical intrusion detection systems and rule-based firewalls rely on static signatures that attackers evade by slightly changing packet sizes, entropy or request timing. Cloud-based machine learning avoids that, but sends raw internal network data to third parties, which creates GDPR and SOC 2 liability.

AegisOps is an open-source AutoML and deep neural network platform built for cybersecurity telemetry. It runs entirely in memory (`io.BytesIO`) with no temporary files, masks sensitive entities before any model sees them, stress-tests models against adversarial noise, explains decisions with SHAP, and generates Snort, YARA and iptables rules mapped to MITRE ATT&CK.

---

## Key features

1. **In-memory ingestion (CSV, JSON, PDF, PCAP/PCAPNG).** Parses network, transport and payload-entropy features directly from RAM up to 1 GB.
2. **Local anonymizer (`PrivacyMasker`).** Replaces internal IPs (`192.168.1.50` becomes `INTERNAL_HOST_001`), MAC addresses, emails and URLs before tensor preparation. An optional toggle restores real values on the local screen only.
3. **Two operating modes.**
   - *SecOps action mode (default):* plain-English severity labels (Critical, High, Medium, Low), impact summaries and ready-to-run firewall commands.
   - *Data science deep-dive mode:* model leaderboard, PyTorch loss curves, radar comparison of metrics, and raw validation results.
4. **Adaptive model arena.** Compares Logistic Regression, Random Forest, XGBoost and a multi-layer perceptron for supervised detection. On unlabeled logs it trains a PyTorch autoencoder and flags high reconstruction error as possible zero-day activity.
5. **Adversarial robustness testing (`AdversarialRobustnessTester`).** Adds bounded FGSM and Laplacian noise to continuous features and reports an Evasion Resistance Score (0 to 100%).
6. **Explainability with SHAP.** Converts feature contributions into plain-English causes, for example: "Flagged due to unusual RDP port 3389 traffic (64.2% impact) and long connection duration (24.1% impact)."
7. **MITRE ATT&CK mapping and rule generation.** Produces YARA rules, Snort/Suricata signatures and iptables/UFW commands, using an AI agent with an offline heuristic fallback.
8. **Automated verification suite.** Live in-dashboard test runner profiling all 10 defensive and mathematical invariants with exact millisecond latencies.

---

## Interface

The dashboard and landing page share one design: black background, Geist and Geist Mono type, thin 1 px borders, flat surfaces, white primary buttons, and muted green/amber/red used only for status. There are no emoji, glow effects, glass blur or gradient text. Charts use the same palette.

---

## Quickstart

### 1. Set up
```bash
git clone https://github.com/srivatsa108/Adversarial-Resistant-SecOps-AutoML-Neural-Arena.git
cd Adversarial-Resistant-SecOps-AutoML-Neural-Arena
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_samples.py
```

### 2. Run the dashboard
```bash
streamlit run app_streamlit.py
```
Open http://localhost:8501.

### 3. Run the FastAPI backend
```bash
uvicorn app_fastapi:app --port 8000 --reload
```
API docs are at http://localhost:8000/docs.

### 4. Run the tests
```bash
PYTHONPATH=. pytest tests/ -v
```
