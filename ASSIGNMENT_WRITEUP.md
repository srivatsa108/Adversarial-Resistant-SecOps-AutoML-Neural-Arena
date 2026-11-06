# UI redesign of Adversa (Adversarial-Resistant SecOps AutoML)

## Goal
Take the dashboard that an AI coding agent (Google Antigravity) produced from my first prompt, find what was wrong with its UI, and rebuild it as a modern analytics dashboard with light and dark themes and support for several LLM providers.

## What was wrong with the first result
- Oversized headings and large empty areas, with no summary of what the user had loaded or run.
- Default-looking radio buttons, faded buttons and a heavy focus ring after a click.
- The theme switch was a sidebar radio that only changed some colours. Tables and inputs stayed dark in light mode, and "System" was really just dark.
- Only one provider (Gemini) could supply the API key.

## What I changed
1. **Layout.** Sidebar with brand block and grouped settings, page header with a status pill, four live KPI cards (records, features, best model, evasion resistance), segmented tabs, bordered panels, tinted alerts.
2. **Light and dark.** One colour palette per theme, chosen from Streamlit's own theme setting so charts, tables and inputs all switch together. System follows the operating system.
3. **Visual rules.** Geist type, one accent colour, thin borders, no emoji, glow, glass blur or gradient text, sentence-case copy, motion only on interaction, reduced-motion respected, usable from 390 px to 1440 px.
4. **API keys.** Provider picker (Gemini, Anthropic, OpenAI, OpenRouter, Groq, Mistral, Together, DeepSeek, xAI, Cohere, Hugging Face, Perplexity, custom OpenAI-compatible), paste a key, read it from environment variables, or import a `.env` file. Keys stay in memory. If a call fails, the app falls back to the offline rule engine.

## How I worked
I wrote a structured prompt for the coding agent (layout, palettes, type scale, rules, definition of done), reviewed the result in a browser at desktop and phone widths in both themes, found the problems above, and fixed them directly. The final prompt that describes the finished design is in `ANTIGRAVITY_PROMPT.md`.

## Verification and limits
- Checked visually in a browser in light and dark at desktop width, and the sidebar at phone width.
- Tested the `.env` parsing and the fallback to the offline engine with an invalid key.
- Not tested: a real API key against each provider, and the model, robustness and rules tabs in the redesigned UI (those need PyTorch, SHAP and the full environment).

## Files
`app_streamlit.py`, `src/utils/providers.py`, `src/mitre/generator.py`, `.streamlit/config.toml`, `requirements.txt`, `.env.example`, `ANTIGRAVITY_PROMPT.md`.
