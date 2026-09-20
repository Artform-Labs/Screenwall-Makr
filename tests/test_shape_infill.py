"""Shape infill: DXF/SVG/AI-PDF import + hole-pattern infill of arbitrary outlines."""
import math

import ezdxf
import pytest

from pdf_preview import doc_to_pdf
from shape_infill import (
    InfillResult,
    MIN_PITCH_FACTOR,
    ShapeGeometry,
    analyze_strokes,
    build_infill_document,
    contour_hole_centers,
    infill_hole_centers,
    load_shape,
    rings_from_dxf,
    rings_from_pdf,
    rings_from_svg,
    scale_rings,
    strip_bounding_rect,
    _classify_centers,
    _segment_arrays,
)

import numpy as np


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _dxf_bytes(tmp_path, build):
    doc = ezdxf.new(dxfversion="R2010")
    doc.units = 1
    build(doc.modelspace())
    p = tmp_path / "shape.dxf"
    doc.saveas(str(p))
    return p.read_bytes()


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _min_boundary_dist(shape, x, y):
    _, d, _, _ = _classify_centers(np.array([[x, y]]), _segment_arrays(shape.rings))
    return float(d[0])


def _inside(shape, x, y):
    ins, _, _, _ = _classify_centers(np.array([[x, y]]), _segment_arrays(shape.rings))
    return bool(ins[0])


# ---------------------------------------------------------------------------
# DXF import
# ---------------------------------------------------------------------------
def test_dxf_rect_with_void(tmp_path):
    data = _dxf_bytes(tmp_path, lambda msp: (
        msp.add_lwpolyline(_rect(0, 0, 10, 6), close=True),
        msp.add_lwpolyline(_rect(4, 2, 6, 4), close=True),
    ))
    shape = rings_from_dxf(data)
    assert len(shape.rings) == 2
    assert shape.extents == pytest.approx((10.0, 6.0))
    # even-odd: void interior is "outside"
    assert _inside(shape, 1.0, 1.0)
    assert not _inside(shape, 5.0, 3.0)


def test_dxf_open_lines_joined_into_ring(tmp_path):
    data = _dxf_bytes(tmp_path, lambda msp: (
        msp.add_line((0, 0), (8, 0)),
        msp.add_line((8, 0), (4, 6)),
        msp.add_line((4, 6), (0, 0)),
    ))
    shape = rings_from_dxf(data)
    assert len(shape.rings) == 1
    assert _inside(shape, 4.0, 1.0)


def test_dxf_circle_and_mm_units(tmp_path):
    data = _dxf_bytes(tmp_path, lambda msp: msp.add_circle((0, 0), 127.0))
    shape = rings_from_dxf(data, unit_scale=1.0 / 25.4)  # 127 mm radius = 5"
    w, h = shape.extents
    assert w == pytest.approx(10.0, abs=0.01)
    assert h == pytest.approx(10.0, abs=0.01)


def test_dxf_no_closed_outline_raises(tmp_path):
    data = _dxf_bytes(tmp_path, lambda msp: msp.add_line((0, 0), (5, 5)))
    with pytest.raises(ValueError, match="No closed outlines"):
        rings_from_dxf(data)


def test_dxf_skips_text_with_warning(tmp_path):
    data = _dxf_bytes(tmp_path, lambda msp: (
        msp.add_lwpolyline(_rect(0, 0, 5, 5), close=True),
        msp.add_text("HI"),
    ))
    shape = rings_from_dxf(data)
    assert len(shape.rings) == 1
    assert any("TEXT" in w for w in shape.warnings)


# ---------------------------------------------------------------------------
# SVG import
# ---------------------------------------------------------------------------
SVG_DONUT = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="480">'
    b'<path d="M0,0 H480 V480 H0 Z M144,144 H336 V336 H144 Z" fill-rule="evenodd"/>'
    b"</svg>"
)


def test_svg_path_subpaths_and_px_scale():
    shape = rings_from_svg(SVG_DONUT)  # 480 px = 5"
    assert len(shape.rings) == 2
    assert shape.extents == pytest.approx((5.0, 5.0))
    assert _inside(shape, 0.5, 0.5)
    assert not _inside(shape, 2.5, 2.5)  # inner void


def test_svg_circle_element():
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">'
        b'<circle cx="48" cy="48" r="48"/></svg>'
    )
    shape = rings_from_svg(svg)
    w, h = shape.extents
    assert w == pytest.approx(1.0, abs=0.01)
    assert h == pytest.approx(1.0, abs=0.01)


def test_svg_multiple_disjoint_graphics():
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480">'
        b'<rect x="0" y="0" width="384" height="480"/>'
        b'<rect x="576" y="0" width="384" height="480"/></svg>'
    )
    shape = rings_from_svg(svg)  # two 4"x5" blocks, 2" apart
    assert len(shape.rings) == 2
    assert _inside(shape, 2.0, 2.5)
    assert _inside(shape, 8.0, 2.5)
    assert not _inside(shape, 5.0, 2.5)  # gap between graphics


def test_svg_live_text_warns():
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">'
        b'<rect x="0" y="0" width="96" height="96"/><text x="10" y="50">A</text></svg>'
    )
    shape = rings_from_svg(svg)
    assert any("text" in w.lower() for w in shape.warnings)


# ---------------------------------------------------------------------------
# AI / PDF import
# ---------------------------------------------------------------------------
def _pdf_bytes(content: bytes) -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\ntrailer\n%%EOF\n"
    )


def test_pdf_paths_clip_discarded_donut_kept():
    content = (
        b"0 0 720 720 re\nW n\n"                     # clip = artboard, must be discarded
        b"72 72 m 648 72 l 648 648 l 72 648 l h f\n"  # 8" outer square
        b"288 288 m 432 288 l 432 432 l 288 432 l h f\n"  # 2" inner void
    )
    shape = rings_from_pdf(_pdf_bytes(content))
    assert len(shape.rings) == 2
    assert shape.extents == pytest.approx((8.0, 8.0))
    assert not _inside(shape, 4.0, 4.0)  # void center


def test_pdf_cm_transform_and_curves():
    content = (
        b"q 2 0 0 2 0 0 cm\n"
        b"36 0 m 72 36 l 0 36 l h f\n"  # 1" wide triangle, scaled 2x by cm → 2"
        b"Q\n"
    )
    shape = rings_from_pdf(_pdf_bytes(content))
    assert len(shape.rings) == 1
    w, h = shape.extents
    assert w == pytest.approx(2.0, abs=0.01)


def test_pdf_compressed_stream():
    import zlib
    raw = b"0 0 144 144 re f\n"
    content = zlib.compress(raw)
    shape = rings_from_pdf(_pdf_bytes(content))
    assert shape.extents == pytest.approx((2.0, 2.0))


def test_load_shape_eps_rejected():
    with pytest.raises(ValueError, match="EPS"):
        load_shape(b"%!PS-Adobe-3.0", "logo.eps")


# ---------------------------------------------------------------------------
# strip_bounding_rect
# ---------------------------------------------------------------------------
def test_strip_bounding_rect():
    shape = ShapeGeometry([_rect(0, 0, 20, 10), _rect(3, 3, 6, 6)])
    stripped, removed = strip_bounding_rect(shape)
    assert removed
    assert len(stripped.rings) == 1
    assert stripped.extents == pytest.approx((3.0, 3.0))
    # single ring: never stripped
    single = ShapeGeometry([_rect(0, 0, 20, 10)])
    _, removed2 = strip_bounding_rect(single)
    assert not removed2


# ---------------------------------------------------------------------------
# Infill
# ---------------------------------------------------------------------------
def test_infill_rect_all_holes_clear():
    shape = ShapeGeometry([_rect(0, 0, 10, 6)])
    res = infill_hole_centers(shape, 0.25, 0.75, "staggered", 60.0, 0.25)
    clearance = 0.25 + 0.125
    assert len(res.holes) > 50
    for x, y in res.holes:
        assert _inside(shape, x, y)
        assert _min_boundary_dist(shape, x, y) >= clearance - 1e-9


def test_infill_respects_interior_void():
    shape = ShapeGeometry([_rect(0, 0, 10, 6), _rect(4, 2, 6, 4)])
    res = infill_hole_centers(shape, 0.25, 0.5, "straight", 60.0, 0.2)
    clearance = 0.2 + 0.125
    assert res.holes
    for x, y in res.holes:
        # not inside the void, and clear of its walls too
        assert not (4.0 < x < 6.0 and 2.0 < y < 4.0)
        assert _min_boundary_dist(shape, x, y) >= clearance - 1e-9


def test_infill_multi_graphic_fills_each_glyph():
    shape = ShapeGeometry([_rect(0, 0, 4, 5), _rect(6, 0, 10, 5)])
    res = infill_hole_centers(shape, 0.25, 0.5, "straight", 60.0, 0.2)
    left = [h for h in res.holes if h[0] < 4.5]
    right = [h for h in res.holes if h[0] > 5.5]
    between = [h for h in res.holes if 4.5 <= h[0] <= 5.5]
    assert left and right
    assert not between


def test_infill_nudge_recovers_edge_holes():
    # 45° hypotenuse cuts the grid at varying clearances → some holes miss by
    # a hair without nudge and fit with it.
    shape = ShapeGeometry([[(0.0, 0.0), (9.0, 0.0), (0.0, 9.0)]])
    kw = dict(hole_dia=0.5, pitch=0.6, pattern="straight", stagger_angle=60.0, margin=0.25)
    plain = infill_hole_centers(shape, **kw, nudge_max=0.0)
    nudged = infill_hole_centers(shape, **kw, nudge_max=0.15)
    assert plain.nudged == 0
    assert nudged.nudged > 0
    assert len(nudged.holes) == len(plain.holes) + nudged.nudged
    clearance = 0.25 + 0.25
    for x, y in nudged.holes:
        assert _inside(shape, x, y)
        assert _min_boundary_dist(shape, x, y) >= clearance - 1e-6
    # nudged holes never crowd an existing hole
    pts = nudged.holes
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            assert d >= 0.5 * 1.05 - 1e-9


def test_infill_nudge_capped():
    # Same shape: an unlimited budget would admit more holes than a small one.
    shape = ShapeGeometry([[(0.0, 0.0), (9.0, 0.0), (0.0, 9.0)]])
    kw = dict(hole_dia=0.5, pitch=0.6, pattern="straight", stagger_angle=60.0, margin=0.25)
    small = infill_hole_centers(shape, **kw, nudge_max=0.05)
    large = infill_hole_centers(shape, **kw, nudge_max=0.30)
    assert small.nudged <= large.nudged


def test_infill_validation():
    shape = ShapeGeometry([_rect(0, 0, 5, 5)])
    with pytest.raises(ValueError, match="pitch"):
        infill_hole_centers(shape, 0.5, 0.25, "straight", 60.0, 0.1)
    with pytest.raises(ValueError, match="candidate holes"):
        infill_hole_centers(ShapeGeometry([_rect(0, 0, 5000, 5000)]),
                            0.05, 0.1, "staggered", 60.0, 0.1)


def test_contour_row_rectangle_corners_and_edges():
    shape = ShapeGeometry([_rect(0, 0, 10, 6)])
    centers, dropped = contour_hole_centers(shape, 0.25, 0.75, 0.2)
    clearance = 0.2 + 0.125
    assert centers
    # every ring hole hugs the boundary at exact clearance (within nudge tol)
    for x, y in centers:
        assert _min_boundary_dist(shape, x, y) == pytest.approx(clearance, abs=0.02)
    # all four corners anchored: a hole near each corner's bisector point
    for cx, cy in [(0, 0), (10, 0), (10, 6), (0, 6)]:
        ex = cx + (clearance if cx == 0 else -clearance) * (1 if cx == 0 else 1)
        assert any(math.hypot(x - cx, y - cy) <= clearance * math.sqrt(2) + 0.03
                   for x, y in centers), f"no corner hole at {cx},{cy}"
    # per-edge justified spacing: gaps along the bottom edge are uniform
    bottom = sorted(x for x, y in centers if abs(y - clearance) < 0.02)
    gaps = [b - a for a, b in zip(bottom, bottom[1:])]
    assert max(gaps) - min(gaps) < 0.02
    assert all(abs(g - 0.75) <= 0.75 * 0.5 for g in gaps)


def test_contour_row_traces_smooth_void():
    # donut: ring follows both the outer square and the circular counter
    shape = ShapeGeometry([_rect(0, 0, 8, 8)])
    import ezdxf as _e  # circle void via flattened ring
    theta = [i * math.tau / 72 for i in range(72)]
    void = [(4 + 1.5 * math.cos(t), 4 + 1.5 * math.sin(t)) for t in theta]
    shape = ShapeGeometry([_rect(0, 0, 8, 8), void])
    centers, _ = contour_hole_centers(shape, 0.25, 0.6, 0.15)
    clearance = 0.15 + 0.125
    near_void = [(x, y) for x, y in centers if 1.0 < math.hypot(x - 4, y - 4) < 2.2]
    assert len(near_void) >= 10  # a ring of holes hugs the counter
    for x, y in near_void:
        # just outside the void at clearance
        assert math.hypot(x - 4, y - 4) == pytest.approx(1.5 + clearance, abs=0.03)


def test_perimeter_row_integrated_no_crowding():
    shape = ShapeGeometry([[(0.0, 0.0), (9.0, 0.0), (0.0, 9.0)]])
    res = infill_hole_centers(shape, 0.25, 0.6, "staggered", 60.0, 0.15,
                              nudge_max=0.15, perimeter_row=True)
    assert res.contour > 0
    assert len(res.holes) > res.contour  # interior fill still present
    clearance = 0.15 + 0.125
    ring = res.holes[:res.contour]
    grid = res.holes[res.contour:]
    for x, y in res.holes:
        assert _inside(shape, x, y)
        assert _min_boundary_dist(shape, x, y) >= clearance - 1e-6
    # grid holes keep visual separation from the ring
    for gx, gy in grid:
        dmin = min(math.hypot(gx - rx, gy - ry) for rx, ry in ring)
        assert dmin >= 0.8 * 0.6 - 1e-9
    # apex corner (sharpest point of the triangle) is anchored
    assert any(math.hypot(x - 0, y - 9) < 1.2 for x, y in ring)


def test_optimize_grid_never_worse_and_valid():
    shape = ShapeGeometry([[(0.0, 0.0), (9.0, 0.0), (0.0, 9.0)]])
    kw = dict(hole_dia=0.5, pitch=0.6, pattern="straight", stagger_angle=60.0, margin=0.25)
    default = infill_hole_centers(shape, **kw)
    optimized = infill_hole_centers(shape, **kw, optimize_grid=True)
    assert len(optimized.holes) >= len(default.holes)
    clearance = 0.25 + 0.25
    for x, y in optimized.holes:
        assert _inside(shape, x, y)
        assert _min_boundary_dist(shape, x, y) >= clearance - 1e-6
    # spacing untouched: all pairwise x-deltas within a row are multiples of pitch
    rows = {}
    for x, y in optimized.holes:
        rows.setdefault(round(y, 6), []).append(x)
    for xs in rows.values():
        xs = sorted(xs)
        for a, b in zip(xs, xs[1:]):
            k = (b - a) / 0.6
            assert abs(k - round(k)) < 1e-6


def _diag_strip(width, length=8.0):
    """45° strip of the given perpendicular width — a synthetic thin stroke."""
    d = length / math.sqrt(2)
    off = width / math.sqrt(2)
    return [(0.0, 0.0), (d, d), (d - off, d + off), (-off, off)]


def test_analyze_strokes_measures_width():
    # 10 × 1.2 bar: chords are 1.2 across the long walls (dominant sample)
    shape = ShapeGeometry([_rect(0, 0, 10, 1.2)])
    a = analyze_strokes(shape, 0.25, 0.75, 0.15)
    assert a is not None
    assert a.median_width == pytest.approx(1.2, abs=0.05)
    # fit_pitch reaches 3 rows: (w - 2c) / 1.6 with c = 0.275
    assert a.fit_pitch == pytest.approx((1.2 - 0.55) / 1.6, abs=0.01)
    assert a.fit_scale is None


def test_analyze_strokes_recommends_scale_when_hole_too_big():
    # 0.8" stroke with a 0.25" hole and 0.15" margin can never host 3 rows
    shape = ShapeGeometry([_rect(0, 0, 10, 0.8)])
    a = analyze_strokes(shape, 0.25, 0.75, 0.15)
    assert a.fit_pitch is None
    assert a.fit_scale is not None and a.fit_scale > 1.0
    # applying the recommended scale makes 3 rows reachable
    scaled = ShapeGeometry(scale_rings(shape.rings, a.fit_scale))
    a2 = analyze_strokes(scaled, 0.25, 0.75, 0.15)
    assert a2.fit_pitch is not None
    assert a2.fit_pitch >= MIN_PITCH_FACTOR * 0.25 - 1e-9


def test_analyze_strokes_rows_estimate_wide_shape():
    shape = ShapeGeometry([_rect(0, 0, 10, 6)])
    a = analyze_strokes(shape, 0.25, 0.75, 0.15)
    assert a.rows_typical >= 3


def test_midline_rescue_covers_narrow_diagonal_stroke():
    # 0.7" diagonal stroke: too narrow for grid rows between the perimeter
    # rings — without rescue the body is bare, with rescue a centerline chain
    # runs down the stroke.
    shape = ShapeGeometry([_diag_strip(0.7)])
    kw = dict(hole_dia=0.25, pitch=0.6, pattern="staggered", stagger_angle=60.0,
              margin=0.12, nudge_max=0.1, perimeter_row=True)
    rescued = infill_hole_centers(shape, **kw, narrow_fill=True)
    assert rescued.midline >= 8
    # the centerline row owns the narrow zone: no off-center nudged holes
    assert rescued.nudged == 0
    # midline holes hug the stroke's medial line: boundary distance ≈ w/2
    mid = rescued.holes[len(rescued.holes) - rescued.midline:]
    for x, y in mid:
        assert _inside(shape, x, y)
        assert _min_boundary_dist(shape, x, y) == pytest.approx(0.35, abs=0.03)
    # chain spacing is uniform (consistent to the eye) and never overlapping
    gaps = [math.hypot(mid[i + 1][0] - mid[i][0], mid[i + 1][1] - mid[i][1])
            for i in range(len(mid) - 1)]
    assert max(gaps) - min(gaps) < 0.05
    pts = rescued.holes
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            assert d >= 0.25 * 1.05 - 1e-9


def test_midline_rescue_relaxes_margin_in_narrowest_zones():
    # 0.45" stroke: full clearance (0.12 + 0.125 = 0.245 each side → needs
    # 0.49") is impossible, but relaxed clearance (0.125 + 0.06) fits.
    shape = ShapeGeometry([_diag_strip(0.45)])
    res = infill_hole_centers(shape, 0.25, 0.6, "staggered", 60.0, 0.12,
                              perimeter_row=True, narrow_fill=True)
    assert res.midline >= 5
    relaxed = 0.125 + 0.06
    for x, y in res.holes[len(res.holes) - res.midline:]:
        assert _min_boundary_dist(shape, x, y) >= relaxed - 1e-6


def test_midline_rescue_skips_wide_shapes():
    shape = ShapeGeometry([_rect(0, 0, 10, 6)])
    res = infill_hole_centers(shape, 0.25, 0.6, "staggered", 60.0, 0.15,
                              perimeter_row=True, narrow_fill=True)
    wide_only = infill_hole_centers(shape, 0.25, 0.6, "staggered", 60.0, 0.15,
                                    perimeter_row=True)
    assert res.midline == 0
    assert len(res.holes) == len(wide_only.holes)


def test_scale_rings():
    shape = ShapeGeometry(scale_rings([_rect(0, 0, 10, 6)], 2.4))
    assert shape.extents == pytest.approx((24.0, 14.4))


# ---------------------------------------------------------------------------
# Output document + preview
# ---------------------------------------------------------------------------
def test_build_document_layers_and_preview(tmp_path):
    shape = ShapeGeometry([_rect(0, 0, 10, 6), _rect(4, 2, 6, 4)])
    res = infill_hole_centers(shape, 0.25, 0.75, "staggered", 60.0, 0.25)
    doc = build_infill_document(shape, res.holes, 0.25)
    msp = doc.modelspace()
    polys = [e for e in msp if e.dxftype() == "LWPOLYLINE"]
    circles = [e for e in msp if e.dxftype() == "CIRCLE"]
    assert len(polys) == 2 and all(p.dxf.layer == "cut" and p.closed for p in polys)
    assert len(circles) == len(res.holes)
    assert all(c.dxf.layer == "holes" for c in circles)
    # every hole keeps the drawn diameter — nudge moves, never resizes
    assert all(abs(c.dxf.radius - 0.125) < 1e-12 for c in circles)
    pdf = doc_to_pdf(doc, "shape test")
    assert pdf.startswith(b"%PDF")
    out = tmp_path / "out.dxf"
    doc.saveas(str(out))
    assert out.stat().st_size > 0


def test_end_to_end_dxf_roundtrip(tmp_path):
    data = _dxf_bytes(tmp_path, lambda msp: (
        msp.add_lwpolyline(_rect(0, 0, 12, 8), close=True),
        msp.add_circle((6, 4), 1.5),
    ))
    shape = load_shape(data, "panel_logo.dxf")
    res = infill_hole_centers(shape, 0.25, 0.6, "staggered", 60.0, 0.2, nudge_max=0.15)
    assert res.holes
    doc = build_infill_document(shape, res.holes, 0.25)
    assert doc_to_pdf(doc, "roundtrip").startswith(b"%PDF")
