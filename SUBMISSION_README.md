# Adversa: adversarial-resistant security ML workspace

**Student:** Srivatsa Shashikumar (PES1PGX25CS072), Executive M.Tech Cybersecurity Engineering
**Course:** AI/ML for Cybersecurity
**Built with:** Google Antigravity (AI IDE), guided by written prompts (see `KEY_PROMPTS.md`)

## Problem

Security detectors that score well on a clean test set can still be fooled by small, deliberate changes to their inputs. Adversa is a general tool for any tabular security data (CSV, JSON, logs, PCAP, Parquet): upload a dataset, pick a target column (or none, for unlabeled data), and it asks three questions of the detector it builds. Phishing URL detection is the example used in the demo.

1. Which model detects phishing best on this data?
2. How easily can an attacker evade it?
3. Why does it flag a site, and can that be turned into a defensive rule?

## Dataset

Demo dataset (the app accepts any dataset; see "Works on other data" below).

`data/public/phishing_urls_10k.csv`: 10,000 rows (5,000 phishing, 5,000 legitimate), 16 binary URL, domain and page features plus a `Label` column.

- Source: Shreya Gopal, *Phishing Website Detection by Machine Learning Techniques* (GitHub, `DataFiles/5.urldata.csv`).
- Phishing URLs come from PhishTank. Legitimate URLs come from the University of New Brunswick URL 2016 dataset. Feature definitions follow the UCI "Phishing Websites" dataset.
- The file has many repeated rows. The app's cleaner removes them, leaving 4,374 unique records.
- Details and links: `data/public/DATASET.md`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Then, in the browser:

1. **Ingestion and privacy**: upload `data/public/phishing_urls_10k.csv`, click **Clean dataset**.
2. **Model arena**: choose `Label` as the target, click **Run benchmark**.
3. **Robustness and explanations**: runs the adversarial attack and SHAP automatically.
4. **MITRE and rules**: click **Synthesize rules**. No API key is needed, an offline generator is used when none is set.
5. **Verification**: click **Run verification suite** (10 checks).

Optional: add an API key in the sidebar (Gemini, Anthropic, OpenAI and other compatible providers) to generate rules with an LLM. Run the tests with `pytest`.

## What the app does

- Masks IPs and personal data in memory, removes duplicates.
- Trains and compares a PyTorch neural network, XGBoost and an anomaly detector.
- Attacks the winner with small input perturbations and reports an evasion resistance score.
- Explains decisions with SHAP.
- Maps findings to MITRE ATT&CK and writes YARA, Snort and firewall rules.
- Dark and light themes.

## Works on other data

Besides the phishing file, I ran a different synthetic network-log dataset through the app (`data/public/second_dataset_network_logs.csv`: IP, protocol, bytes, duration, failed logins, country, with a three-class `attack_type` target and missing values). It trained and benchmarked end to end (winner: Deep MLP, 97.3% catch rate). Screenshot: `screenshots/12_second_dataset_network_logs.png`. The sidebar also has sample datasets in other formats, and an unlabeled option that uses an autoencoder for anomaly detection.

## Results from my run

In the recorded demo run: winning model XGBoost Classifier, 93.5% catch rate with a 6.6% false positive rate, evasion resistance score 93.1%, and 10 of 10 verification checks passed. Numbers vary a little between runs (an earlier run gave an evasion score near 88% and a different winner). See `screenshots/`.

## What I learned

Replace this with your own words. Points from the real work:

- Vague prompts gave a poor interface. Prompts with exact design rules (colours, spacing, what to remove) gave a usable one.
- I could not accept the generated code as it was. Reviewing it found a slider that displayed 0%, results that disappeared when switching views, a wrong `uvicorn` command in the README, and a light theme that left some widgets dark.
- The duplicated rows in a public dataset can inflate scores, so cleaning first matters.
- A model with a high test score is not necessarily robust. The evasion score shows a different picture.
- Tests (10 verification checks) caught problems I would have missed by eye.

## Files

| Item | Where |
|---|---|
| Code | `app_streamlit.py`, `src/` |
| Dataset | `data/public/` |
| Screenshots | `screenshots/` |
| Demo video | `Adversa_demo.mp4` |
| Key prompts | `KEY_PROMPTS.md` |
