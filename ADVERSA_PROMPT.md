# Task: Redesign the UI of Antigravity SecOps AutoML with light and dark themes

## Context
This repo is a Python security-ML platform. The UI has two parts:
- `app_streamlit.py` is the main dashboard (Streamlit), with helpers in `src/utils/plotting.py` (Plotly charts) and `src/utils/translations.py` (severity badges).
- `landing/index.html` is the static landing page.

Users are SOC analysts (default "SecOps action" mode) and data scientists ("Data science deep dive" mode). The interface should feel like a precise, calm professional tool, not a template.

## What to build
1. **Two complete themes, light and dark.**
   - Define all colours as CSS variables on `:root` and override them for dark mode. No hard-coded hex values inside components.
   - Default to the user's system setting (`prefers-color-scheme`). Add a theme switch (System / Light / Dark) in the Streamlit sidebar and in the landing page header, and remember the choice for the session (`st.session_state` in Streamlit, `localStorage` in a try/catch on the landing page).
   - Both themes must meet WCAG AA contrast (4.5:1 for body text, 3:1 for large text and UI borders).
   - Plotly charts, dataframes, code blocks and severity badges must follow the active theme. Build one theme-aware Plotly layout function that takes the theme name; do not leave dark-only colours in `plotting.py`.
2. **A more stylish, distinctive look, still restrained.**
   - Dark: true black base, raised surfaces one step lighter, thin 1 px borders.
   - Light: cool near-white base (`#F7F7F8`), white surfaces, ink text (`#0A0A0A`), same border logic. Avoid cream and terracotta palettes.
   - One accent colour only (a confident blue, for example `#2F6BFF` light and `#5B9DF5` dark), used for the primary action, focus rings and the selected tab. Status colours (green, amber, red) are used only to show state.
   - Type: Geist for text, Geist Mono for IPs, rule code, hashes and numbers in tables. Use a clear scale (12, 14, 16, 20, 28, 40), weights 400/500/600 only, tight letter-spacing on headings, tabular numerals for metrics.
   - Layout: generous spacing on a 4/8 px grid, content width capped around 1120 px on the landing page, left-aligned text, line length under 80 characters.
   - Make one thing memorable: the Evasion Resistance Score and the model leaderboard should be the visual centre of the dashboard (large, clean, well-spaced figures with a slim progress bar), with everything else quiet.
   - Depth comes from borders and surface steps first, with at most one soft shadow on floating elements (menus, toasts). No glassmorphism, no glow, no gradient text or washes, no emoji, no decorative icons, no all-caps eyebrow labels, no arrows appended to button text.
3. **Motion.** Only motion that responds to a click (tab change, expand, theme switch at 150 to 200 ms). Animate only opacity and colour. Honour `prefers-reduced-motion`. No scroll-reveal or hover-lift effects.
4. **Copy.** Sentence case. Buttons start with a verb ("Clean dataset", "Run benchmark", "Generate rules"). Errors say what happened and how to fix it, and never blame the user. Empty states say what to do next, with one button.
5. **Keep behaviour intact.** Do not change model, parsing, privacy-masking, SHAP or rule-generation logic. Keep all function names and return keys used by the app (for example `badge_html`, `icon`, `plain_english_summary`).

## Responsive and accessibility floor
- Works from 375 px to 1440 px with no horizontal page scroll; tables scroll inside their own container.
- Visible keyboard focus on every interactive element, logical tab order, labels on all inputs, touch targets at least 40 px.
- Landing page: semantic landmarks, a skip link, and a sensible `<title>` and meta description.

## Files to change
`app_streamlit.py`, `src/utils/plotting.py`, `src/utils/translations.py`, `landing/index.html`, and `.streamlit/config.toml` (so Streamlit's own widgets follow the theme). Put shared tokens in one place and reuse them.

## How to work
1. Read the files above and list the current colour, type and component inventory.
2. Propose the token set (colours for both themes, type scale, spacing, radii) in a short table before writing code.
3. Implement, then run `python -m py_compile` on the Python files and `PYTHONPATH=. pytest tests/ -v`.
4. Run `streamlit run app_streamlit.py` and the landing page, and check every tab in both themes at 375 px and 1440 px. Screenshot each and fix anything with low contrast, overflow or mismatched chart colours.
5. Finish with a short summary of what changed and anything you could not verify.

## Definition of done
- Switching theme changes every surface, chart, table and badge with no leftover dark or light patches.
- No emoji, glow, glass blur or gradient text anywhere in the app, charts or landing page.
- Tests pass and the dashboard runs without errors.
