# Key prompts

The prompts that shaped the project, in the order I used them. The two long prompts are saved in full in this repo (`ANTIGRAVITY_PROMPT.md`, `ADVERSA_PROMPT.md`).

## 1. Build the app (Antigravity)

Original brief given to the AI IDE: build a Streamlit security-ML workspace that ingests security telemetry (CSV, JSON, PCAP, logs, Parquet), masks personal data in memory, trains and compares several models, attacks the best one with adversarial inputs, explains it with SHAP, and generates MITRE ATT&CK-mapped detection rules.

**Why it worked:** it named the exact stages and outputs, so the generated modules (`src/cleaners`, `src/models`, `src/adversarial`, `src/xai`, `src/mitre`) match the pipeline.

## 2. Make the interface look like a modern dashboard (Antigravity)

Full text: `ANTIGRAVITY_PROMPT.md` and `ADVERSA_PROMPT.md`. Key instructions: flat surfaces with one thin border, KPI cards across the top, a left sidebar for settings, dark and light themes from one set of tokens, no emoji, glow, glass or gradients.

**What went wrong first:** my first, shorter prompt produced a cluttered interface. The prompt above fixed it by being specific about layout and what to leave out.

## 3. Review and fix (my follow-up prompts)

- "make the ui as per the latest edition and remove the ai elements like symbols etc"
- "the antigravity ui is so bad" followed by "cant it look any modern app with dashboards"
- "can u help me add that as a prompt as well. also add more options for importing api keys i mean from various platforms"
- "can we rename from aegis to something else" (renamed to Adversa)

**What I changed after reviewing the output:** fixed sliders that showed 0%, stopped the results from vanishing when switching view mode, corrected the `uvicorn app_fastapi:app` command in the README, and repainted native widgets that stayed dark in light mode.

## 4. Dataset and demo

- Chose the public PhishTank / UNB phishing URL dataset, cited in `data/public/DATASET.md`.
- Prompt for the walkthrough: record a 2 to 5 minute demo that uploads the data, cleans it, runs the benchmark, shows robustness and SHAP, generates rules and runs the verification suite.

> Edit this file so it lists the prompts exactly as you typed them in Antigravity. The AI IDE's own history is the best source.
