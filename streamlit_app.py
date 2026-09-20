import base64
import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from screenwall_generator import (
    APP_RELEASE_DATE,
    APP_RELEASE_LABEL,
    APP_VERSION,
    GENERATOR_ARTWORK_TAG,
    INSTALL_SLOT_EXTRA,
    parse_csv,
    build_panel_document,
)
from gcode_export import GCodeConfig, doc_to_gcode, doc_to_gcode_combo
from gcode_import import parse_gcode, paths_to_document, summarize
from pdf_preview import doc_to_pdf, paths_to_pdf
from shape_infill import (
    MIN_PITCH_FACTOR,
    MM_PER_IN,
    PDF_PT_PER_IN,
    SVG_PX_PER_IN,
    analyze_strokes,
    build_infill_document,
    infill_hole_centers,
    load_shape,
    scale_rings,
    strip_bounding_rect,
    ShapeGeometry,
)
from sample_data import TEMPLATE_CSV, build_sample_files, zip_files

st.set_page_config(page_title="Screenwall Makr Beta", layout="wide")
LOGO_PATH = Path(__file__).parent / "assets" / "artform_logo.png"
APP_DISPLAY_VERSION = f"Beta v{APP_VERSION} · {APP_RELEASE_LABEL} · {APP_RELEASE_DATE}"


def _inline_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _show_pdf(data: bytes, height: int = 640) -> None:
    """Inline PDF viewer: st.pdf when available, base64 embed otherwise."""
    if hasattr(st, "pdf"):
        try:
            st.pdf(data, height=height)
            return
        except Exception:
            pass
    b64 = base64.b64encode(data).decode("ascii")
    st.markdown(
        f'<embed src="data:application/pdf;base64,{b64}" type="application/pdf" '
        f'width="100%" height="{height}px" />',
        unsafe_allow_html=True,
    )


def _nc_preview(data: bytes, max_lines: int = 36) -> str:
    lines = data.decode("ascii").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + (
        f"\n... (+{len(lines) - max_lines} more lines - full file in the ZIP download)"
    )


@st.cache_data(show_spinner=False)
def _sample_bundle():
    spec, files = build_sample_files()
    return spec, files, zip_files(files)


def _zip_outputs(folder: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in ("*.dxf", "*.nc"):
            for f in sorted(Path(folder).glob(pattern)):
                z.write(f, f.name)
    mem.seek(0)
    return mem.read()


# CSV template data lives in sample_data.py (shared with the built-in sample).
template_csv = TEMPLATE_CSV

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    .stApp {
        background: #ffffff;
        color: #1d1d1f;
    }
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .artform-header {
        padding-top: 0.35rem;
    }
    .artform-title {
        font-size: clamp(2.3rem, 5vw, 4rem);
        line-height: 1.02;
        font-weight: 300;
        letter-spacing: -0.03em;
        color: #1d1d1f;
        margin: 0;
    }
    .artform-beta {
        display: inline-block;
        margin-left: 0.35em;
        padding: 0.12em 0.45em 0.18em;
        font-size: 0.38em;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        vertical-align: middle;
        color: #1d1d1f;
        background: #f2f2f7;
        border: 1px solid #d2d2d7;
        border-radius: 0.35em;
    }
    .artform-subtitle {
        max-width: 42rem;
        margin-top: 0.95rem;
        font-size: 1.05rem;
        line-height: 1.6;
        font-weight: 300;
        color: #6e6e73;
    }
    .artform-build {
        margin-top: 0.8rem;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
        color: #8e8e93;
    }
    .artform-rule {
        height: 1px;
        background: linear-gradient(90deg, rgba(29,29,31,0.14), rgba(29,29,31,0.05));
        margin: 1.35rem 0 1.8rem 0;
    }
    .artform-logo {
        display: block;
        width: 75%;
        max-width: 165px;
        min-width: 120px;
        margin: 0.85rem auto 0;
        height: auto;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e5e5e7;
        border-radius: 18px;
        background: #fbfbfd;
        overflow: hidden;
    }
    [data-testid="stFileUploader"] {
        border-radius: 18px;
        border: 1px dashed #c7c7cc;
        background: #fbfbfd;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid #d2d2d7;
        background: #ffffff;
        color: #1d1d1f;
        font-weight: 500;
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #1d1d1f;
        color: #1d1d1f;
    }
</style>
""", unsafe_allow_html=True)

logo_b64 = _inline_image_base64(LOGO_PATH) if LOGO_PATH.exists() else None

header_left, header_right = st.columns([4.8, 0.8], vertical_alignment="top")
with header_left:
    st.markdown(
        f"""
        <div class="artform-header">
            <h1 class="artform-title">Screenwall Makr<span class="artform-beta">Beta</span></h1>
            <div class="artform-subtitle">
                Flat pattern DXF generator for perforated screenwall panels,
                with bend-aware flange geometry, install-slot placement, and
                fabrication-ready CSV import controls. Team testing release.
            </div>
            <div class="artform-build">{APP_DISPLAY_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    if logo_b64:
        st.markdown(
            f'<img class="artform-logo" src="data:image/png;base64,{logo_b64}" alt="Artform logo" />',
            unsafe_allow_html=True,
        )

st.markdown('<div class="artform-rule"></div>', unsafe_allow_html=True)

with st.expander("Flange code reference", expanded=False):
    st.markdown("""
| Code | Description |
|------|-------------|
| `L4S` | L-flange, all 4 sides, same depth |
| `J4S` | J-flange (leg + return lip), all 4 sides, same depth |
| `L2TB` | L-flange, top + bottom only |
| `J2TB` | J-flange, top + bottom only |
| `L2LR` | L-flange, left + right only |
| `J2LR` | J-flange, left + right only |
| `RT4S` | Right trapezoid, L-flange all sides — parallel top or bottom, vertical legs, angled opposite edge |
| `RT4J` | Right trapezoid, J-flange all sides (same `flange2_depth` as `J4S`) |
| `MIX` | Per-side type and depth — fill in the `top_*`, `bottom_*`, `left_*`, `right_*` columns |

**Notes**
- `flange1_depth` = nominal leg depth (outside mold line). Used for all non-MIX codes.
- `flange2_depth` = nominal return lip depth for J codes. Leave blank for L codes.
- `material` defaults to `aluminum`; `alloy` defaults to `3003`.
- `fastening_pair` is required in the CSV. Use `tb`, `lr`, `t`, `b`, `l`, `r`, or `none`.
  Typical practice is to place install slots on the long sides only.
- `fastener_dia` is the install slot width.
- Accepted width aliases on import: `slot_width`, `fastener_slot_width`, `install_slot_width`.
- `slot_length` is optional total install slot length. Leave it blank to use the default
  `fastener_dia + 0.50"`.
- Accepted length aliases on import: `fastener_slot_length`, `install_slot_length`.
- `shop_flat_mode`: `auto` (default) blends shop cut-line flats (5052 / 0.1875″ calibration) with bend theory across all table materials; `off` uses pure `_flat_leg_L` / `_flat_leg_J` / `_flat_lip` only.
- DXF layer `finished_face` (gray): **CSV `width` × `height`** = nominal face opening at developed `f1+f2` runouts (before BD corner arc). Holes / slots / margin use this. Layer `bend`: **F1** bend-1 CL is **inset** `_f1_bend_inset_from_face` from the finished face (0.168″ toward the panel interior at 5052 / 0.1875″ ref; scales with bend table). **J** bend-2 CL is **outward** at developed f1 from the face (e.g. 1.497″ at ref F1=2″). **L** outer / **J** lip use shop-calibrated developed flats (`shop_flat_mode` `auto`). **0.195″** = face corner to **inside flange leg edge** (cut datum; no relief square); scale from the single ref until more coupons exist. Exclude `finished_face` from perimeter cut in CAM.
- **Inside face up:** flat patterns are generated as if you are looking at the **inside** of the cassette (perforated face up). Panel ID stick text is placed for that view; on an **L** flange the leg that receives the ID may be the one that ends up **hidden after forming**.
- **J install slots** (return lip, `fastening` layer): slot center to **blank outer lip** = **nominal `flange2_depth` / MIX `*_f2`** minus the OC from the aligned perforation to the inner **finished_face** edge. Only the **outermost** perforation line toward that flange is used (so stagger does not pull slots to an inner row). Along-flange spacing is **≤ 12″** c–c (never greater). **J+J** miters: slot stations trim from each corner by developed `f2` so holes stay in material outside the miter.
- For `MIX`: set `top_type` / `bottom_type` / `left_type` / `right_type` to `L` or `J`.
  Set `*_f1` (leg depth) and `*_f2` (return lip, J only). A side with `*_f1 = 0` is a straight cut.
- For `RT4S` / `RT4J`: set `rt_opposing_edge` to `top` or `bottom` (which parallel edge is straight in plan).
  Set `rt_leg_left` and `rt_leg_right` to the vertical face heights at the left and right hard corners
  (inches). `width` is still the parallel span between the vertical legs. Leave `rt_*` blank for other codes.
- **Corner fold clearance:** at every blank corner where two flanges fold, each flange end edge is
  inset **`thickness / 2`** (3/32″ per side on 3/16″ material → 3/16″ total gap after folding) so the
  folded edges never clash. Square ends step back through the hard corner (staircase); **J+J** miter
  edges offset perpendicular so the folded lips keep a uniform gap. Scales 1:1 with gauge for all
  materials. Override per row with `corner_gap_override` (total gap in inches; `0` disables). Corners
  where only one side has a flange are untouched.
""")

with st.expander("Beta testing notes (READ FIRST)", expanded=False):
    st.markdown("""
**Release:** Beta v0.1.0 — First Draft (team testing, not production sign-off).

- Use a **unique `panel_id`** on every row. Duplicate IDs overwrite the same DXF in the ZIP.
- **`flange_code`** must be exact (`J4S`, `L4S`, …). Common typo: **`J4L`** → use **`J4S`**.
- Save as **CSV (comma)**. In Notepad, confirm the header column is `flange_code` (no space).
- **150+ panels** often work locally (~0.2 s/panel); Streamlit Cloud may time out or run out of memory on very large ZIPs.
- Report issues with the **Release** and **DXF engine** strings shown in the caption below (proves which build ran).
""")

with st.expander("Material & gauge rules", expanded=False):
    st.markdown("""
**Thickness column** accepts either a decimal (e.g. `0.1875`) or a gauge string (`16 ga`, `14 ga`, `11 ga`).

Gauge strings decode **differently per material** to match the shop bend table (Known Truths Section 2):

| Gauge | Aluminum | Steel |
|------|---------|-------|
| `16 ga` | 0.0625″ | 0.0600″ |
| `14 ga` | 0.0800″ | 0.0750″ |
| `11 ga` | 0.1250″ | 0.1200″ |

**For STEEL panels:**
- `material = steel` drives both gauge decoding and the bend-table lookup.
- `alloy` may still be `steel` (or a steel descriptor) for clarity, but it no longer
  has to be `steel` just to get steel k-factor / radius values.

**For ALUMINUM panels:**
- `material = aluminum`
- `alloy = 3003` (default), `5052`, or `6061` (6061 needs larger bend radii — verify with shop).

**Power-user override:** if you want to force specific values regardless of material, set `k_factor_override` and `bend_radius_override` columns and they take precedence over the table.
""")

st.download_button(
    "⬇ Download CSV template",
    data=template_csv,
    file_name="screenwall_template.csv",
    mime="text/csv",
)
st.caption(
    "Populate the template and upload below. All dimensions in inches. Import guidance lives in the expanders above. "
    f"**DXF engine:** `{GENERATOR_ARTWORK_TAG}` · **Release:** `{APP_DISPLAY_VERSION}`"
)

if st.button("Show me a sample file"):
    st.session_state["show_sample"] = True

if st.session_state.get("show_sample"):
    spec, sample_files, sample_zip = _sample_bundle()
    n_holes = sum(
        1 for line in sample_files[f"{spec.panel_id}_punch.nc"].decode("ascii").splitlines()
        if line.startswith("X")
    )
    st.info(
        f"Sample panel **`{spec.panel_id}`** — {spec.face_width}″ × {spec.face_height}″, "
        f"{spec.thickness}″ {spec.material} {spec.alloy}, `{spec.flange_code}`, "
        f"Ø{spec.hole_dia}″ holes @ {spec.pitch}″ {spec.pattern}, install slots top+bottom. "
        f"Generated with the same engine as a CSV upload: the shop DXF, a single-machine "
        f"laser program, and the punch + laser combo pair ({n_holes} punch hits)."
    )
    tab_pdf, tab_laser, tab_punch, tab_combo_laser = st.tabs([
        "PDF preview — geometry",
        f"{spec.panel_id}.nc — laser (all features)",
        f"{spec.panel_id}_punch.nc — turret punch hits",
        f"{spec.panel_id}_laser.nc — combo laser (etch + perimeter)",
    ])
    with tab_pdf:
        st.caption(
            "Flat-pattern geometry, one color per DXF layer: red = perimeter cut, "
            "orange = perforations, blue = install slots, magenta = panel-ID etch, "
            "dashed green = bend centerlines, dashed gray = finished face (reference)."
        )
        _show_pdf(sample_files[f"{spec.panel_id}.pdf"])
    with tab_laser:
        st.caption(
            "One machine does everything: panel-ID etch at mark power, then perforations, "
            "install slots, and the blank perimeter last. Each path is annotated with "
            "`(op=… layer=… shape=…)`."
        )
        st.code(_nc_preview(sample_files[f"{spec.panel_id}.nc"]), language="gcode")
    with tab_punch:
        st.caption(
            "Two-machine combo, file 1: every perforation and install slot as a single punch "
            "hit. The tool table in the header maps T numbers to round (RD) and obround (OB) "
            "tools — remap to your turret stations."
        )
        st.code(_nc_preview(sample_files[f"{spec.panel_id}_punch.nc"]), language="gcode")
    with tab_combo_laser:
        st.caption(
            "Two-machine combo, file 2: only the panel-ID etch and the perimeter cut remain "
            "for the laser — the punched features are gone."
        )
        st.code(_nc_preview(sample_files[f"{spec.panel_id}_laser.nc"]), language="gcode")
    st.markdown("**Download individual files**")
    dl_specs = [
        (f"{spec.panel_id}.pdf", "application/pdf"),
        (f"{spec.panel_id}.dxf", "application/octet-stream"),
        (f"{spec.panel_id}.nc", "text/plain"),
        (f"{spec.panel_id}_punch.nc", "text/plain"),
        (f"{spec.panel_id}_laser.nc", "text/plain"),
    ]
    for col, (fname, mime) in zip(st.columns(len(dl_specs)), dl_specs):
        with col:
            st.download_button(
                f"⬇ {fname}",
                sample_files[fname],
                fname,
                mime=mime,
                key=f"dl_{fname}",
            )
    zip_col, hide_col = st.columns([1.2, 5])
    with zip_col:
        st.download_button(
            "⬇ Everything (.zip)",
            sample_zip,
            "screenwall_sample.zip",
            mime="application/zip",
        )
    with hide_col:
        if st.button("Hide sample"):
            st.session_state["show_sample"] = False
            st.rerun()

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "input.csv"
        csv_path.write_bytes(uploaded.getvalue())
        try:
            panels = parse_csv(str(csv_path))
            st.success(f"Loaded {len(panels)} panel(s).")

            # Show parsed summary
            summary = []
            for p in panels:
                slot_length = p.slot_length if p.slot_length is not None else (p.fastener_dia + INSTALL_SLOT_EXTRA)
                summary.append({
                    "ID": p.panel_id,
                    "W×H": f"{p.face_width}″ × {p.face_height}″",
                    "Thickness": f"{p.thickness}″",
                    "Alloy": f"{p.material} {p.alloy}",
                    "Code": p.flange_code,
                    "Hole Ø": f"{p.hole_dia}″",
                    "Pitch": f"{p.pitch}″",
                    "Pattern": p.pattern,
                    "Install Sides": p.fastening_pair,
                    "Slot W×L": f"{p.fastener_dia}″ × {slot_length}″",
                })
            st.dataframe(summary, use_container_width=True)

            want_gcode = st.checkbox(
                "Also export G-code (.nc) — one program per panel, same geometry as the DXF",
                value=False,
            )
            gcode_config = None
            gcode_combo = False
            if want_gcode:
                with st.expander("G-code options", expanded=False):
                    st.caption(
                        "Generic RS-274 2D cut program: panel-ID etch first, then perforations "
                        "and install slots, blank perimeter last. `finished_face` is never machined; "
                        "bend centerlines are reference only unless etched below. Post-check feeds/"
                        "power against your controller before running. "
                        "**Punch + laser combo** writes two files per panel for two machines: "
                        "`_punch.nc` (perforations + install slots as single hits, tool table in the "
                        "header for turret-station remap) and `_laser.nc` (panel-ID etch + perimeter)."
                    )
                    mode_label = st.radio(
                        "Machine style",
                        [
                            "Laser / plasma (M3–M5 head)",
                            "Router / mill (Z plunge)",
                            "Turret punch + laser combo (two files per panel)",
                        ],
                        horizontal=True,
                    )
                    if mode_label.startswith("Turret"):
                        mode = "combo"
                    elif mode_label.startswith("Laser"):
                        mode = "laser"
                    else:
                        mode = "mill"
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        feed_cut = st.number_input("Cut feed (in/min)", value=60.0, min_value=1.0)
                    with c2:
                        feed_mark = st.number_input("Mark feed (in/min)", value=120.0, min_value=1.0)
                    with c3:
                        etch_bend = st.checkbox("Etch bend centerlines", value=False)
                    gcode_combo = mode == "combo"
                    gcode_config = GCodeConfig(
                        mode="laser" if gcode_combo else mode,
                        feed_cut=feed_cut,
                        feed_mark=feed_mark,
                        include_bend_marks=etch_bend,
                    )

            button_label = "Generate DXFs + G-code" if want_gcode else "Generate DXFs"
            if st.button(button_label, type="primary"):
                errors = []
                for panel in panels:
                    try:
                        doc = build_panel_document(panel)
                        doc.saveas(str(Path(d) / f"{panel.panel_id}.dxf"))
                        if want_gcode and gcode_combo:
                            programs = doc_to_gcode_combo(doc, panel.panel_id, panel.thickness, gcode_config)
                            for machine, program in programs.items():
                                (Path(d) / f"{panel.panel_id}_{machine}.nc").write_text(program, encoding="ascii")
                        elif want_gcode:
                            program = doc_to_gcode(doc, panel.panel_id, panel.thickness, gcode_config)
                            (Path(d) / f"{panel.panel_id}.nc").write_text(program, encoding="ascii")
                    except Exception as e:
                        errors.append(f"{panel.panel_id}: {e}")
                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    z = _zip_outputs(d)
                    st.download_button(
                        "⬇ Download panels (.zip)",
                        z,
                        "screenwall_panels.zip",
                        mime="application/zip",
                    )
        except Exception as e:
            st.error(str(e))

# ---------------------------------------------------------------------------
# Build artwork from G-code (import)
# ---------------------------------------------------------------------------
st.markdown('<div class="artform-rule"></div>', unsafe_allow_html=True)
st.subheader("Build artwork from G-code")
st.caption(
    "Upload `.nc` programs to rebuild layered artwork. Screenwall-generated files "
    "(laser, mill, or punch) re-import losslessly via their `(op=… layer=… shape=…)` "
    "annotations; other 2D G-code is classified by shape — circles → holes, rounded "
    "slots → install slots, closed loops → cut, long lines → bend, short strokes → text. "
    "Supports G0–G3 (I/J or R arcs), inch/mm, absolute/incremental."
)
nc_uploads = st.file_uploader(
    "Upload G-code (.nc)", type=["nc", "gcode", "tap", "txt"], accept_multiple_files=True
)
for nc_file in nc_uploads or []:
    try:
        text = nc_file.getvalue().decode("utf-8", errors="replace")
        result = parse_gcode(text)
        doc = paths_to_document(result.paths)
        summary = summarize(result)
        stem = Path(nc_file.name).stem
        panel_name = summary["panel_id"] or stem
        with st.expander(f"{nc_file.name} → `{panel_name}`", expanded=True):
            if not result.paths:
                st.warning("No machinable paths found in this program.")
                for w in summary["warnings"]:
                    st.warning(w)
                continue
            counts = ", ".join(f"{v} × `{k}`" for k, v in sorted(summary["paths_per_layer"].items()))
            st.markdown(
                f"**Dialect:** `{summary['dialect']}`"
                f"{' (annotated — lossless)' if summary['annotated'] else ' (heuristic classification)'} · "
                f"**Extents:** {summary['extents_in'][0]}″ × {summary['extents_in'][1]}″ · "
                f"**Paths:** {counts}"
                + (f" · **Hole Ø:** {', '.join(f'{d}″' for d in summary['hole_diameters'])}"
                   if summary["hole_diameters"] else "")
            )
            for w in summary["warnings"]:
                st.warning(w)
            pdf_bytes = paths_to_pdf(result.paths, f"{panel_name} - imported from {nc_file.name}")
            _show_pdf(pdf_bytes)
            with tempfile.TemporaryDirectory() as nc_d:
                dxf_path = Path(nc_d) / f"{panel_name}.dxf"
                doc.saveas(str(dxf_path))
                dxf_bytes = dxf_path.read_bytes()
            c1, c2, _ = st.columns([1.2, 1.2, 3])
            with c1:
                st.download_button(
                    f"⬇ {panel_name}.dxf", dxf_bytes, f"{panel_name}.dxf",
                    mime="application/octet-stream", key=f"imp_dxf_{nc_file.name}",
                )
            with c2:
                st.download_button(
                    f"⬇ {panel_name}.pdf", pdf_bytes, f"{panel_name}.pdf",
                    mime="application/pdf", key=f"imp_pdf_{nc_file.name}",
                )
    except Exception as e:
        st.error(f"{nc_file.name}: {e}")

# ---------------------------------------------------------------------------
# Shape infill (DXF / SVG / AI / PDF upload → perforated artwork)
# ---------------------------------------------------------------------------
st.markdown('<div class="artform-rule"></div>', unsafe_allow_html=True)
st.subheader("Infill a shape with perforations")
st.caption(
    "Upload **DXF**, **SVG**, or **AI / PDF** files of closed outlines — letters, "
    "logos, arbitrary polygons — and fill them with the same straight / staggered "
    "hole pattern the panels use. One file can hold many graphics (a whole word); "
    "interior voids like the counter of an *O* stay empty. Every hole keeps the "
    "edge margin from the outline. For crisp letterforms: the **perimeter outline "
    "row** traces each outline with corner-anchored holes at per-edge-justified "
    "spacing, **auto-align grid** slides the fill grid (never its spacing) to fit "
    "the most holes, and **edge nudge** lets an almost-fitting hole slide slightly "
    "inward (same Ø, capped shift). "
    "Convert text to outlines before export (Illustrator: Type → Create Outlines). "
    "EPS isn't supported — use SVG, PDF, or PDF-compatible .ai instead."
)
shape_uploads = st.file_uploader(
    "Upload shape files (.dxf / .svg / .ai / .pdf)",
    type=["dxf", "svg", "ai", "pdf"],
    accept_multiple_files=True,
)

if shape_uploads:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        s_hole_dia = st.number_input("Hole Ø (in)", value=0.25, min_value=0.01, step=0.0625, format="%.4f")
    with c2:
        s_pitch = st.number_input("Pitch c–c (in)", value=0.75, min_value=0.02, step=0.125, format="%.4f")
    with c3:
        s_pattern = st.selectbox("Pattern", ["staggered", "straight"])
    with c4:
        s_angle = st.number_input("Stagger angle (°)", value=60.0, min_value=15.0, max_value=90.0, step=5.0)
    with c5:
        s_margin = st.number_input("Edge margin (in)", value=0.25, min_value=0.0, step=0.0625, format="%.4f")
    with c6:
        s_flex_pct = st.number_input(
            "Spacing flex (% of pitch)", value=15, min_value=0, max_value=25, step=5,
            help="Enables letter-aware fill: strokes are detected and filled "
                 "with rows that follow them (equal stroke widths get identical "
                 "fill; row/hole spacing may flex by this much), wider areas get "
                 "the standard lattice. Hole size never changes. 0 = rigid grid "
                 "only (holes that don't fit are simply dropped).",
        )
    t1, t2, t3, t4 = st.columns([1.3, 1.4, 1.5, 1.2])
    with t1:
        s_perimeter = st.checkbox(
            "Perimeter outline row", value=False,
            help="Adds a dedicated row of holes tracing every outline, corner-"
                 "anchored, justified per edge. Best for LARGE graphics with wide "
                 "strokes (4+ rows); on letter-width strokes it can read as a "
                 "racetrack — the elastic lattice already hugs the edges there.",
        )
    with t2:
        s_narrow = st.checkbox(
            "Fill narrow strokes", value=True,
            help="Strokes too narrow for lattice rows get a chain of holes down "
                 "the stroke's centerline. In the very narrowest zones the edge "
                 "margin may relax to half so the form stays covered.",
        )
    with t3:
        s_autofit = st.checkbox(
            "Auto-fit pitch to strokes", value=False,
            help="Measures the typical stroke width and tightens the pitch (never "
                 f"below {MIN_PITCH_FACTOR}× hole Ø, never looser than entered) so "
                 "at least 3 hole rows span the typical stroke.",
        )
    with t4:
        st.empty()

for shape_file in shape_uploads or []:
    fkey = shape_file.name
    with st.expander(f"{shape_file.name}", expanded=True):
        try:
            is_svg = fkey.lower().endswith(".svg")
            is_pdfish = fkey.lower().endswith((".ai", ".pdf"))
            if is_svg:
                unit_options = ["px (96 / inch)", "millimeters", "inches", "scale to target width"]
            elif is_pdfish:
                unit_options = ["points (72 / inch)", "scale to target width"]
            else:
                unit_options = ["inches", "millimeters", "scale to target width"]
            u1, u2, u3 = st.columns([2.2, 1, 1.2])
            with u1:
                unit_label = st.radio(
                    "Drawing units", unit_options, horizontal=True, key=f"units_{fkey}"
                )
            target_width = None
            with u2:
                if unit_label.startswith("scale"):
                    target_width = st.number_input(
                        "Target width (in)", value=24.0, min_value=0.1, key=f"tw_{fkey}"
                    )
            with u3:
                drop_bg = st.checkbox(
                    "Drop background rect",
                    value=is_pdfish,
                    key=f"bg_{fkey}",
                    help="Remove one outer rectangle spanning the whole canvas "
                         "(artboard/background). Uncheck for border/stencil designs.",
                )

            if unit_label.startswith("px"):
                unit_scale = 1.0 / SVG_PX_PER_IN
            elif unit_label.startswith("points"):
                unit_scale = 1.0 / PDF_PT_PER_IN
            elif unit_label.startswith("mill"):
                unit_scale = 1.0 / MM_PER_IN
            elif unit_label.startswith("scale"):
                unit_scale = None  # parser default; rescaled to target width below
            else:
                unit_scale = 1.0

            shape = load_shape(shape_file.getvalue(), fkey, unit_scale=unit_scale)
            if drop_bg:
                shape, removed = strip_bounding_rect(shape)
                if removed:
                    st.info("Removed one full-canvas rectangle (artboard/background).")
            if target_width is not None:
                w, _ = shape.extents
                if w > 1e-9:
                    shape = ShapeGeometry(scale_rings(shape.rings, target_width / w), shape.warnings)
            for w_msg in shape.warnings:
                st.warning(w_msg)

            ew, eh = shape.extents

            file_pitch = s_pitch
            stroke = analyze_strokes(shape, s_hole_dia, s_pitch, s_margin)
            if stroke is not None:
                if s_autofit and stroke.fit_pitch is not None and stroke.fit_pitch < s_pitch:
                    file_pitch = stroke.fit_pitch
                    st.info(
                        f"Auto-fit: pitch tightened {s_pitch}″ → **{file_pitch}″** so "
                        f"~3 rows span the typical {stroke.median_width}″ stroke."
                    )
                stroke_now = analyze_strokes(shape, s_hole_dia, file_pitch, s_margin)
                msg = (
                    f"**Stroke analysis:** typical {stroke_now.median_width}″ wide "
                    f"(thin {stroke_now.thin_width}″) → ~**{stroke_now.rows_typical} rows** "
                    f"across the typical stroke, ~{stroke_now.rows_thin} across thin ones."
                )
                if stroke_now.rows_typical < 3:
                    if stroke.fit_pitch is not None and stroke.fit_pitch < file_pitch:
                        msg += (
                            f" For 3 rows: set pitch ≤ **{stroke.fit_pitch}″** "
                            "(or enable Auto-fit pitch)."
                        )
                    elif stroke.fit_scale is not None:
                        msg += (
                            f" Even at the tightest pitch this hole Ø can't reach 3 rows — "
                            f"scale the artwork up ~**{stroke.fit_scale}×**"
                        )
                        if stroke.fit_dia is not None:
                            msg += f" or drop to hole **Ø ≤ {stroke.fit_dia}″** (at pitch 2×Ø)"
                        msg += "."
                    st.warning(msg)
                else:
                    st.caption(msg)

            result = infill_hole_centers(
                shape, s_hole_dia, file_pitch, s_pattern, s_angle, s_margin,
                nudge_max=file_pitch * s_flex_pct / 100.0,
                perimeter_row=s_perimeter, optimize_grid=True,
                narrow_fill=s_narrow, spacing_flex=s_flex_pct / 100.0,
            )
            nudge_note = (
                f" · **Perimeter row:** {result.contour}" if result.contour else ""
            ) + (
                f" · **Centerline (narrow strokes):** {result.midline}" if result.midline else ""
            ) + (
                f" · **Adjusted within flex:** {result.nudged}" if result.nudged else ""
            ) + (
                f" · **Dropped at edge:** {result.dropped}" if result.dropped else ""
            )
            st.markdown(
                f"**Extents:** {ew:.2f}″ × {eh:.2f}″ · **Outlines:** {len(shape.rings)} · "
                f"**Holes:** {len(result.holes)} × Ø{s_hole_dia}″{nudge_note}"
            )
            if not result.holes:
                st.warning(
                    "No holes fit. Reduce hole Ø / pitch / margin, scale the shape up, "
                    "or check the units."
                )
            shape_name = Path(fkey).stem
            doc = build_infill_document(shape, result.holes, s_hole_dia)
            pdf_bytes = doc_to_pdf(
                doc, f"{shape_name} - {len(result.holes)} x {s_hole_dia}\" holes"
            )
            _show_pdf(pdf_bytes)
            with tempfile.TemporaryDirectory() as sh_d:
                dxf_path = Path(sh_d) / f"{shape_name}_perf.dxf"
                doc.saveas(str(dxf_path))
                dxf_bytes = dxf_path.read_bytes()
            d1, d2, _ = st.columns([1.2, 1.2, 3])
            with d1:
                st.download_button(
                    f"⬇ {shape_name}_perf.dxf", dxf_bytes, f"{shape_name}_perf.dxf",
                    mime="application/octet-stream", key=f"shape_dxf_{fkey}",
                )
            with d2:
                st.download_button(
                    f"⬇ {shape_name}_perf.pdf", pdf_bytes, f"{shape_name}_perf.pdf",
                    mime="application/pdf", key=f"shape_pdf_{fkey}",
                )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"{shape_file.name}: {e}")
