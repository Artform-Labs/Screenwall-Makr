# Screenwall Makr (Beta — First Draft)

**Version:** `0.1.0` · **Release:** Beta — First Draft · **Date:** 2026-05-15  

Flat-pattern DXF generator for perforated architectural screenwall panels with L/J flanges, install slots, and bend-aware shop geometry. Built for Artform / Ermaksan Speed Bend Pro workflow.

**Live app:** [share.streamlit.io](https://share.streamlit.io/) — repo `jkl-rgb/Screenwall-Makr`, entry `streamlit_app.py`.

---

## Beta scope (first draft)

This milestone is intended for **team testing over time**, not production sign-off.

| In scope | Not yet |
|----------|---------|
| CSV import → ZIP of panel DXFs | Sheet nesting (`nest_panels` stub) |
| L4S, J4S, L2TB/J2TB, L2LR/J2LR, MIX, RT4S, RT4J | Duplicate `panel_id` detection in UI |
| Perforations (straight / staggered) | Hard cap on batch size (memory/time limits apply) |
| Shape infill: DXF / SVG / AI-PDF upload → hole-pattern infill of arbitrary outlines | EPS import (use SVG / PDF instead) |
| Install slots (L + J shop rules) | |
| Bend + cut + finished_face + text layers | |
| Panel ID (ortho + RT parallel flange) | |

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

1. Download the CSV template from the app.  
2. Fill one row per panel; use a **unique `panel_id`** per row.  
3. Upload CSV → **Generate DXFs** → download ZIP.

See [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md) for Community Cloud deploy and version checks.

---

## Documentation map

| File | Purpose |
|------|---------|
| [CURRENT_STATE.md](CURRENT_STATE.md) | Short handoff — what works, what’s next |
| [KnownTruths](KnownTruths) | Full engineering reference (bend math, layers, CSV) |
| [.cursor/rules/screenwall.mdc](.cursor/rules/screenwall.mdc) | Agent/IDE project rules |
| `screenwall_generator.py` | DXF engine (`APP_VERSION`, `GENERATOR_ARTWORK_TAG`) |

---

## CSV tips for testers

- **`flange_code`** must be exact: `L4S`, `J4S`, `L2TB`, `J2TB`, `L2LR`, `J2LR`, `MIX`, `RT4S`, `RT4J` — not `J4L` (common typo).  
- **Unique `panel_id`** per row — duplicates overwrite the same `.dxf` in the ZIP.  
- Save as **CSV (comma)**; confirm in Notepad that headers include `flange_code` (no space in the name).  
- **RT4S / RT4J:** set `rt_opposing_edge`, `rt_leg_left`, `rt_leg_right`.  
- Large batches (100+ panels) work in code but may be slow or hit Streamlit Cloud memory/time limits.

---

## Verify deployed build

Under the template download, the caption should show:

`Release: Beta v0.1.0 · Beta — First Draft · 2026-05-15`  
and a `DXF engine:` tag matching `GENERATOR_ARTWORK_TAG` in `screenwall_generator.py`.

---

## Tests

```powershell
python -m unittest discover -s tests -q
```

---

## License / repo

GitHub: [jkl-rgb/Screenwall-Makr](https://github.com/jkl-rgb/Screenwall-Makr)
