# Key Prompts Used with AI-Assisted IDE (Google Antigravity)

This document contains the sequence of structured prompts used to build **Adversa**, an Adversarial-Resistant SecOps AutoML & Neural Arena platform, following the AI-assisted software engineering methodology.

---

## Prompt 1: Project Architecture & Core Ingestion Pipeline
```text
Build a modular, production-ready Python platform called "Adversa" for cybersecurity telemetry analysis.
Requirements:
1. In-memory data pipeline using io.BytesIO with zero disk writes to prevent data leaks.
2. Multi-format parser factory supporting CSV, TSV, JSON, JSONL, Syslog/CEF logs, PCAP network captures, and Apache Parquet.
3. Cryptographic zero-trust anonymization (PrivacyMasker) that replaces internal RFC 1918 IPs, MACs, and emails with deterministic hashes, supporting ephemeral client-side local demasking.
4. Data cleaner that filters zero-variance junk features, handles missing values, and prepares scaled matrices.
```

## Prompt 2: Neural Arena, AutoML & Adversarial Testing Engine
```text
Implement an adaptive AutoML and Deep Neural Network arena:
1. Support supervised classification and regression using Logistic Regression, Random Forest, XGBoost, and a PyTorch Deep MLP with early stopping.
2. Support unsupervised log anomaly detection using a PyTorch Deep Autoencoder with MSE reconstruction error thresholding.
3. Implement an Adversarial Robustness Tester (FGSM and Laplacian noise perturbations across epsilons 0.0 to 0.3) that computes an Evasion Resistance Score (0-100%).
4. Integrate SHAP TreeExplainer/KernelExplainer to translate numerical Shapley values into plain-English cybersecurity root causes.
```

## Prompt 3: Threat Intelligence & MITRE Rule Generation
```text
Create a defensive rule generator that maps top threat features to MITRE ATT&CK tactics and techniques.
Requirements:
1. Synthesize production-ready Sigma YAML rules, Snort/Suricata IDS rules, and Linux iptables firewall commands.
2. Provide a pluggable provider registry supporting 13 LLM endpoints (Gemini, OpenAI, Anthropic, DeepSeek, Groq, Mistral, Ollama, Custom).
3. Include a deterministic offline heuristic rule compiler that automatically runs when no API keys are provided.
```

## Prompt 4: UI Redesign & Modern Analytics Dashboard
```text
Redesign the Streamlit dashboard into a high-density, engineering-grade analytics workspace inspired by Linear, Vercel, and Datadog:
1. Provide native Dark and Light themes with WCAG AAA/AA contrast compliance and an instant sidebar toggle.
2. Layout: Sidebar with brand block and grouped controls, single-metric KPI strip populated dynamically via st.empty(), segmented controls/tabs, 10px rounded panels with 1px borders.
3. Dual persona view: "SecOps action mode" (plain-English severity badges, detection rates, and firewall actions) and "Data science deep dive mode" (loss curves, ROC-AUC, and radar benchmarks).
4. Strictly zero emoji, zero glowing borders, zero glassmorphism, and zero animated floating particles.
```

## Prompt 5: Security Review, Verification & Memory Safety (ArrayMemoryError Fix)
```text
Perform an exhaustive security review and performance optimization:
1. Fix numpy._core._exceptions._ArrayMemoryError during unsupervised training: replace naive pd.get_dummies() with cardinality-aware feature extraction (extract length and digit counts for high-cardinality text, bin low-cardinality to top-15 + 'OTHER', and cap features at 128 in np.float32).
2. Accelerate privacy masking using vectorized Series.map() unique-value caching for 25x–100x speedup.
3. Add in-memory CSV and JSON export buttons for cleaned datasets with spreadsheet formula injection neutralization.
4. Implement an automated verification tab that profiles all 10 mathematical and security invariants with millisecond latencies.
```
