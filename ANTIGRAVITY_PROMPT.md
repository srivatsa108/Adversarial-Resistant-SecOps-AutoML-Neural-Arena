# Task: Make the Adversa UI look like a modern analytics dashboard, with light and dark themes and multi-provider API keys

## Context
Python security-ML platform. UI files: `app_streamlit.py` (Streamlit dashboard), `src/utils/plotting.py` (Plotly), `src/utils/translations.py` (severity badges), `landing/index.html` (landing page), `.streamlit/config.toml`. Users are SOC analysts (default "SecOps action mode") and data scientists ("Data science deep dive mode"). The current UI is a plain Streamlit page with oversized headings, empty space, default radio buttons and faded buttons. Rebuild it so it reads like a product such as Linear, Vercel or Datadog: an app shell, KPI cards, panels, segmented tabs.

## Layout (build exactly this structure)
1. **Sidebar (app shell).** Brand block at the top: a 34 px rounded square with the letter "A" in the accent colour, product name, and a one-line descriptor. Below it, small muted group labels (sentence case, 12 px): View, Privacy, Threat intelligence. Persona choice is a vertical list of selectable rows with a tinted background on the selected row. Footer note at the bottom, separated by a border.
2. **Page header.** Title "Threat detection workspace" (24 px, weight 600), one-line subtitle, and a right-aligned pill with a small green dot: "All data stays in memory".
3. **KPI strip.** Four equal cards under the header: Records loaded, Features, Best model, Evasion resistance. Each has a label, a 26 px tabular-number value, and one line of helper text. Empty values show a dash. The strip must be rendered at the end of the script run (use `st.empty()` and fill it last) so it shows data loaded or computed in the same run.
4. **Tabs.** Segmented control: a raised track with a 1 px border, the selected tab as a panel-coloured pill. Tabs: Ingestion and privacy, Model arena, Robustness and explanations, MITRE and rules, Verification.
5. **Content.** Charts, tables and code blocks sit on bordered panels with 10 px radius. Primary button solid accent with white text, 40 px tall, no heavy focus ring after a click (ring on keyboard focus only). File uploader is a dashed drop zone that tints with the accent on hover. Alerts are tinted by meaning (success, warning, error, info) on a single box with no inner box.

## Themes
- Define every colour as a CSS variable generated from one Python dict per theme. No hex values in component markup.
- Follow Streamlit's built-in theme setting: read `st.context.theme.type` ("light" or "dark") to pick the palette, and pass the same value to every Plotly function. The user switches Light, Dark or System from Streamlit's own Settings menu. Do not add a second custom theme switch, because Streamlit's tables and inputs would not follow it.
- Also set `[theme.light]` and `[theme.dark]` in `.streamlit/config.toml` with the same colours.
- Dark palette: bg `#0A0B0D`, panel `#101215`, raised `#16191D`, border `#1E2227`, text `#ECEEF1`, muted `#8D95A0`, accent `#5B8CFF`, ok `#3FB27F`, warn `#E8A53A`, bad `#EF5B5B`.
- Light palette: bg `#F5F6F8`, panel `#FFFFFF`, raised `#EEF0F3`, border `#E3E6EB`, text `#0F1720`, muted `#566070`, accent `#2D5BFF`, ok `#148A5B`, warn `#9A6412`, bad `#C93434`. Light panels get one soft shadow, dark panels none.
- Body text must meet WCAG AA contrast in both themes.

## Type and style rules
- Geist for text, Geist Mono for IPs, hashes and rule code. Scale 12, 13, 14, 16, 20, 24, 26. Weights 400, 500, 600 only. Tabular numerals on every metric.
- Keep Streamlit's Material icon font intact (do not apply the text font to icon spans).
- Sentence case everywhere. Buttons start with a verb. Errors say what happened and how to fix it.
- No emoji, glow, glass blur, gradient text or washes, decorative icons, all-caps eyebrow labels, or arrows in button text. Motion only in response to a click or hover, 150 ms, opacity and colour only, and honour `prefers-reduced-motion`.
- Works from 390 px to 1440 px with no horizontal page scroll. The sidebar must start collapsed on phones (`initial_sidebar_state="auto"`). KPI cards go two per row under 900 px.

## API keys from many providers
Add `src/utils/providers.py` with a `PROVIDERS` table (name, kind, env var names, default model, base URL) covering: Google Gemini, Anthropic, OpenAI, OpenRouter, Groq, Mistral, Together AI, DeepSeek, xAI, Cohere, Hugging Face, Perplexity, and "Custom (OpenAI-compatible)" for Azure, Ollama, LM Studio and vLLM (user supplies the base URL).
- Sidebar, under "Threat intelligence": a `.env` file importer (reads lines such as `OPENAI_API_KEY=...`, fills the matching providers, writes nothing to disk), a Provider select, a password API key field, an editable Model field, and a Base URL field only for the custom provider.
- Prefill the key from the environment when a matching variable exists, and show a status line: "Key from OPENAI_API_KEY", "Key set for this session", or "No key. Rules use the offline engine."
- Use `st.session_state.setdefault` to prefill widget values. Do not pass `value=` together with a `key=` whose state you also set.
- `MitreRuleGenerator.generate()` gets `provider`, `model` and `base_url` arguments. Gemini uses the google-genai SDK, Anthropic uses the Messages API, everything else uses an OpenAI-compatible `/chat/completions` call through `requests`. Keep `api_key` working on its own for the FastAPI app. On any failure, fall back to the offline rule engine.
- Never log, print or persist a key. The payload sent to any provider must stay masked by `PrivacyMasker`.
- Add the provider variables to `.env.example` and raise `streamlit` to `>=1.46.0` in `requirements.txt`.

## Do not change
Model, parsing, privacy masking, SHAP and verification logic, and every key the app reads from returned dictionaries (`badge_html`, `icon`, `plain_english_summary`, and so on).

## How to work
1. Read the files above and list the current colour and component inventory.
2. Implement the layout, themes and provider panel described here.
3. Run `python -m py_compile` on changed files and `PYTHONPATH=. pytest tests/ -v`.
4. Run `streamlit run app_streamlit.py`. Check every tab in light and dark at 1440 px and 390 px, and fix low contrast, overflow, or charts that do not follow the theme. Take a screenshot of each.
5. Finish with a short summary of what changed and anything you could not verify.

## Definition of done
- Switching theme in Streamlit's Settings menu changes every surface, chart, table, input and badge, with no leftover dark or light patches.
- The KPI strip shows real numbers after loading a sample.
- Choosing any provider and pasting or importing a key changes the generator label to that provider, and a bad key falls back to the offline engine without an error.
- No emoji, glow, glass blur or gradient text anywhere. Tests pass.
