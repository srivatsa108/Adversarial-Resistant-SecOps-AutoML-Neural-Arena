# Adversa: 3-Minute Demo Video Walkthrough Script

Use this script to record your 2-to-5 minute screen recording demo.

---

### Minute 0:00 - 0:45 | Introduction & Problem Overview
1. **Open the browser at `http://localhost:8501`.**
2. **Narration:**
   > "Hi everyone. Today I'm demonstrating **Adversa**, an Adversarial-Resistant, Explainable Automated Machine Learning and Neural Network platform built for cybersecurity telemetry.
   > 
   > Traditional intrusion detection systems rely on static signatures that attackers easily evade with slight packet mutations. At the same time, sending raw network logs to external cloud APIs creates serious GDPR and SOC2 compliance risks.
   > 
   > Adversa solves this by running an end-to-end telemetry pipeline entirely in memory with zero disk leaks, cryptographic privacy masking, adversarial stress testing, and automated MITRE ATT&CK defensive rule synthesis."

---

### Minute 0:45 - 1:30 | Ingestion, Privacy Masking & Cleaned Data Export
1. **Show the Sidebar controls:**
   - Click the **Theme Toggle** to show smooth switching between **Dark mode** and **Light mode** with full contrast compliance.
   - Select a sample dataset (e.g. `Phishing telemetry (CSV)` or `Firewall logs (JSON)`), and click **Load sample**.
2. **Show Tab 1 ("Ingest & inspect"):**
   - Point out the 4 live KPI cards populated dynamically at the top (Records, Features, Missing Cells, Memory Footprint).
   - Show the raw preview vs sanitized preview.
   - Toggle **Mask IPs and personal data** in the sidebar to show RFC 1918 internal IPs and emails replaced with HMAC pseudonyms.
   - Toggle **Show original values** to show local client-side verification.
   - Click **Download cleaned CSV** or **Download cleaned JSON** to demonstrate in-memory sanitized data export.

---

### Minute 1:30 - 2:15 | AutoML Arena & Adversarial Robustness
1. **Click Tab 2 ("AutoML & neural arena"):**
   - Click **Run automated arena search**.
   - Show the models competing: Logistic Regression, Random Forest, XGBoost, and PyTorch Deep MLP.
   - Toggle between **SecOps action mode** (showing Threat Severity Badges, detection rates, and firewall summaries) and **Data science deep dive mode** (showing ROC-AUC radar comparison and loss curves).
2. **Click Tab 3 ("Robustness & XAI"):**
   - Show the **Adversarial Robustness Curve** (evaluating evasion resistance under FGSM perturbation).
   - Point out the **Evasion Resistance Score** (e.g. 98.3% resilient).
   - Scroll down to the **SHAP Feature Importance Bar Chart** and show the plain-English explanation of why the threat was flagged.

---

### Minute 2:15 - 3:00 | MITRE ATT&CK Rules & Automated System Verification
1. **Click Tab 4 ("MITRE rules & export"):**
   - Point out the automatically mapped MITRE techniques (e.g., T1046 Network Service Discovery, T1059 Command Injection).
   - Show the synthesized **Sigma Rule**, **Snort / Suricata IDS Signature**, and **Linux iptables firewall block command**.
2. **Click Tab 5 ("System verification and tests"):**
   - Click **Run automated system verification**.
   - Show all 10 system invariants passing (0 disk writes, zero data leaks, sub-second latency chart).
3. **Closing Remarks:**
   > "In conclusion, Adversa demonstrates how an AI coding assistant can be guided to build a complete, production-grade cybersecurity application from scratch—combining security, explainability, and adversarial resilience. Thank you!"
