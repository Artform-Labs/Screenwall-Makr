"""Shape infill: DXF/SVG/AI-PDF import + hole-pattern infill of arbitrary outlines."""
import math

import ezdxf
import pytest

from pdf_preview import doc_to_pdf
from shape_infill import (
    InfillResult,
    ShapeGeometry,
    build_infill_document,
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
