# Adversa: Master Engineering Prompt (Final Iteration)

Build, optimize, and deploy "Adversa": a modular, production-ready, open-source Adversarial-Resistant, Explainable Automated Machine Learning (AutoML) & Deep Neural Network cybersecurity telemetry platform.

Target GitHub Repository:
https://github.com/srivatsa108/Adversarial-Resistant-SecOps-AutoML-Neural-Arena

================================================================================
1. BRAND IDENTITY & MODERN ANALYTICS DESIGN SYSTEM
================================================================================
- Platform Name: "Adversa" (strictly zero mention of "Antigravity").
- Design Aesthetic: High-density, engineering-grade SecOps analytics dashboard inspired by Linear, Vercel, and Datadog.
- Palette & Theming:
  * Native dual-theme engine supporting seamless Dark mode and Light mode toggling via an instant segmented control at the top of the sidebar.
  * Light Mode Palette: Background #F6F8FA, panel/card #FFFFFF, raised hover #EFF1F4, border #E1E4E8, strong border #CFD3D8, primary text #111827 (slate-900, >14:1 contrast ratio), muted text #4B5563 (>6.5:1 ratio), primary accent #2563EB, success #059669, warning #D97706, error #DC2626.
  * Dark Mode Palette: Background #0A0B0D, panel #101215, raised #16191D, border #1E2227, strong border #2C333B, text #ECEEF1, muted #8D95A0, accent #5B8CFF, success #3FB27F, warning #E8A53A, error #EF5B5B.
  * Strict WCAG AAA/AA compliance across all typography, form inputs (-webkit-text-fill-color enabled to prevent faint text in WebKit/Chrome), buttons, tables, alerts, and Plotly charts.
  * Visual Restraint: 10px rounded card containers with 1px border dividers, subtle box shadows, monospace tabular numerals, single-metric KPI strip populated dynamically via st.empty(), segmented tab controls. Clean enterprise workspace footer ("Adversa SecOps Telemetry · In-Memory Pipeline · v2.4"). Strictly zero emoji, zero glowing borders, zero glassmorphism, zero animated floating particles, and zero open-source license clutter in the UI.

================================================================================
2. GENERALIZED MULTI-FORMAT INGESTION & DATA LAKE ENGINE
================================================================================
- Extensible ParserFactory supporting unified in-memory parsing across 11+ telemetry formats:
  * Delimited text: CSV, TSV (RFC 4180).
  * Structured data: JSON arrays, JSONL / NDJSON streaming lines.
  * Unstructured security logs: Syslog (RFC 3164 / RFC 5424), Common Event Format (CEF), and Key-Value firewall logs with automatic header and kv-pair extraction.
  * Binary captures: PCAP and PCAPNG network packet captures (parsing Ethernet, IP, TCP, UDP, ICMP headers, sequence flags, TTL, and payload entropy).
  * Data lakes: Columnar Apache Parquet files (.parquet).
  * Document intelligence: PDF security reports with table and metadata extraction.
- Ingestion Safety: Magic-byte file signature validation (PAR1, PCAP microsecond/nanosecond headers, %PDF-), rejecting spoofed extensions before buffer processing.
- Quota: 1 GB in-memory quota ceiling configured across Streamlit and backend validators.

================================================================================
3. ZERO-TRUST SECURITY, PRIVACY & IN-MEMORY ARCHITECTURE
================================================================================
- RAM-Only Lifecycle: Zero unencrypted temporary disk writes (`io.BytesIO` exclusively throughout upload, parsing, cleaning, feature engineering, and inference).
- Zero-Trust Cryptographic Privacy Masking:
  * RFC 1918 cryptographic pseudonymization (`PrivacyMasker`) replacing internal IPv4 addresses, IPv6, MAC addresses, and corporate email addresses with deterministic keyed HMAC hashes.
  * Local Client-Side Demasking: HMAC secret keys remain resident in local session memory, enabling analysts to toggle ephemeral client-side verification views without exposing raw PII upstream.
- Injection Neutralization: Sanitize scalar strings to neutralize spreadsheet formula injection (=, +, -, @, \t, \r) and strip XSS scripts before data rendering.
- In-Memory Dataset Export: Direct client-side downloads of sanitized, masked, and cleaned datasets in both CSV and JSON formats generated on the fly via `io.StringIO`.

================================================================================
4. PERFORMANCE ACCELERATION & MEMORY SAFETY (OOM PREVENTION)
================================================================================
- Cardinality-Aware Feature Partitioning:
  * Intelligent detection of high-cardinality text, raw syslog strings, hashes, UUIDs, and timestamps (columns where unique values > 30 or cardinality ratio > 15%).
  * Convert high-cardinality columns into structural numerical features (`_len` and `_digits`), completely replacing unbounded `pd.get_dummies()` and permanently eliminating `numpy._core._exceptions._ArrayMemoryError`.
  * Frequency-bin low-to-medium categorical columns into the top 15 most frequent categories, grouping all remaining infrequent entries into "OTHER".
  * Cap total one-hot encoded feature matrix at 128 highest-variance columns.
  * Strict `np.float32` matrix allocation throughout StandardScaler and neural network loaders (cutting memory footprint by 50%).
- Vectorized Privacy Masking:
  * Unique-value caching with `Series.map()` replacing cell-by-cell regex operations, achieving a 25x–100x masking speedup on large dataframes.
- Session-Level Cache Guards:
  * Hash-keyed Streamlit caching preventing redundant privacy re-masking on unrelated widget reruns.
- Sub-Second Diagnostics:
  * SHAP TreeExplainer / KernelExplainer bounded to a representative 100-row sample.
  * Adversarial robustness perturbations (FGSM / PGD / boundary noise) evaluated across 300 test instances across 7 perturbation steps (ε = 0.0 to 0.3).

================================================================================
5. ADAPTIVE AUTOML & NEURAL ARENA
================================================================================
- Dual-Engine Task Determination:
  * Supervised Classification & Regression: Auto-trains and benchmarks Logistic Regression, Random Forest, XGBoost, and a custom PyTorch Deep Multi-Layer Perceptron (MLP) with adaptive early stopping and learning rate scheduling.
  * Unsupervised Anomaly Detection: PyTorch Deep Autoencoder with bottleneck compression and dynamic MSE reconstruction thresholding to isolate zero-day threat outliers.
- Dual-Persona Translation:
  * Data Science Mode: Displays raw loss curves, confusion matrices, F1 scores, precision-recall trade-offs, and Shapley feature attribution.
  * SecOps Action Mode: Converts mathematical metrics into actionable threat intelligence—catch rates, false alarm percentages, adversarial evasion resilience badges, and recommended firewall filtering commands.

================================================================================
6. THREAT INTELLIGENCE & RULE COMPILATION
================================================================================
- Sigma & MITRE ATT&CK Generation:
  * Automatically maps top contributing adversarial features to MITRE ATT&CK techniques (e.g., T1046 Network Service Discovery, T1059 Command Injection, T1110 Brute Force).
  * Exports production-ready YAML Sigma rules directly compatible with SIEM/SOAR platforms (Splunk, Elastic, Sentinel).
- Hybrid LLM / Heuristic Engine:
  * Pluggable provider registry for 13 LLM threat intelligence endpoints (OpenAI, Anthropic, Gemini, DeepSeek, Groq, Mistral, Ollama, Custom v1).
  * Built-in offline deterministic rule compiler that activates automatically when API keys are absent, guaranteeing 100% air-gapped usability.
