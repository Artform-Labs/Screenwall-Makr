"""Infill an uploaded shape (DXF, SVG, or AI/PDF) with the screenwall hole pattern.

Takes arbitrary closed outlines — letters, logos, any polygon, including
shapes with interior voids (the counter of an "O", for example) and multiple
disjoint graphics in one file (a whole word) — and fills them with the same
straight / staggered perforation grid the panel generator uses, honoring hole
diameter, pitch, stagger angle, and an edge-clearance margin.

Geometry model
--------------
A shape is a list of *rings*: closed point loops in inches, y-up. Insideness
uses the even-odd rule across all rings, so interior rings are voids and
multiple disjoint outlines work naturally. A hole center is kept when it is
inside (even-odd) AND its distance to every boundary segment is at least
``margin + hole_radius`` — the exact "circle fits fully inside with the
requested clearance" condition for any polygon, convex or not.

Edge nudge: a hole that ALMOST fits may optionally slide a small, capped
distance inward (never resized) until it clears the outline — so glyph edges
read cleanly instead of dropping a whole row of boundary holes.

No shapely/GEOS: containment and clearance are computed with numpy
(already required by ezdxf). SVG parsing uses ``svgelements`` (pure Python);
AI/PDF parsing is a best-effort stdlib content-stream reader.
"""
from __future__ import annotations

import io
import math
import re as _re
import zlib
from dataclasses import dataclass, field

import ezdxf
import numpy as np
from ezdxf import path as _ezpath
from ezdxf import recover as _ezrecover

from screenwall_generator import _hole_centers

# Flattening tolerances (inches): max sag for curve → polyline conversion.
FLATTEN_DISTANCE_IN = 0.005
# Max chord length when sampling SVG/PDF curves, in final (scaled) inches.
CHORD_IN = 0.02
# Endpoint snap tolerance when joining open DXF entities into loops (inches).
JOIN_TOL_IN = 0.005
# Guard rails so a huge graphic with a tiny pitch cannot melt the server.
MAX_GRID_CANDIDATES = 250_000
MAX_HOLES = 60_000

# Unit conversions.
SVG_PX_PER_IN = 96.0   # SVG user units are CSS pixels
PDF_PT_PER_IN = 72.0   # PDF / AI user units are points
MM_PER_IN = 25.4

_DXF_PATH_TYPES = {
    "LWPOLYLINE", "POLYLINE", "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
}


@dataclass
class ShapeGeometry:
    """Closed rings (inches, y-up, translated to the origin) + import notes."""
    rings: list[list[tuple[float, float]]]
    warnings: list[str] = field(default_factory=list)

    @property
    def bbox(self):
        xs = [x for r in self.rings for x, _ in r]
        ys = [y for r in self.rings for _, y in r]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def extents(self):
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0, y1 - y0)


@dataclass
class InfillResult:
    holes: list[tuple[float, float]]
    nudged: int          # how many of `holes` were shifted to fit
    dropped: int         # candidates near the edge that could not fit
    contour: int = 0     # holes in the perimeter outline row (first in `holes`)
    midline: int = 0     # centerline rescue holes in narrow strokes (last in `holes`)


@dataclass
class StrokeAnalysis:
    """Chord-based stroke width statistics + coverage guidance.

    Widths are perpendicular chords cast inward from the boundary, so they
    measure the actual body of each stroke (letter stems, bars, ring bands).
    """
    median_width: float          # typical stroke width (inches)
    thin_width: float            # 10th-percentile stroke width (inches)
    rows_typical: int            # estimated hole rows across the typical stroke
    rows_thin: int               # estimated hole rows across the thin strokes
    fit_pitch: float | None      # pitch that reaches 3 rows across the typical
                                 # stroke, or None if the hole is too big for that
    fit_scale: float | None      # or: scale-up factor for the current pitch
    fit_dia: float | None = None  # or: hole Ø that reaches 3 rows at pitch = 2×Ø


# ---------------------------------------------------------------------------
# Ring helpers
# ---------------------------------------------------------------------------
def _ring_area(ring) -> float:
    a = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def _dedupe_ring(ring, tol=1e-9):
    out = []
    for p in ring:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def _clean_rings(rings, min_area=1e-6):
    cleaned = []
    for ring in rings:
        r = _dedupe_ring(ring)
        if len(r) >= 3 and abs(_ring_area(r)) > min_area:
            cleaned.append(r)
    return cleaned


def _normalize_rings(rings):
    """Translate so the shape's bbox min corner sits at the origin."""
    if not rings:
        return rings
    min_x = min(x for r in rings for x, _ in r)
    min_y = min(y for r in rings for _, y in r)
    return [[(x - min_x, y - min_y) for x, y in r] for r in rings]


def scale_rings(rings, factor: float):
    return [[(x * factor, y * factor) for x, y in r] for r in rings]


def strip_bounding_rect(shape: ShapeGeometry) -> tuple[ShapeGeometry, bool]:
    """Drop one outer ring that spans (almost) the whole bbox.

    AI/PDF and some SVG exports include an artboard/background rectangle that
    would flip the even-odd fill of everything inside it. Only removes a ring
    when other rings remain.
    """
    if len(shape.rings) < 2:
        return shape, False
    x0, y0, x1, y1 = shape.bbox
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    for i, ring in enumerate(shape.rings):
        rx0 = min(x for x, _ in ring); rx1 = max(x for x, _ in ring)
        ry0 = min(y for _, y in ring); ry1 = max(y for _, y in ring)
        covers = (rx1 - rx0) >= 0.99 * w and (ry1 - ry0) >= 0.99 * h
        boxy = abs(abs(_ring_area(ring)) - (rx1 - rx0) * (ry1 - ry0)) <= 0.02 * w * h
        if covers and boxy:
            rest = shape.rings[:i] + shape.rings[i + 1:]
            return ShapeGeometry(_normalize_rings(rest), list(shape.warnings)), True
    return shape, False


def _join_open_chains(chains, tol=JOIN_TOL_IN):
    """Greedily join open polylines end-to-end into closed rings.

    Real-world DXF text/logos often arrive exploded into LINE/ARC/SPLINE
    fragments; this stitches them back into loops. Returns (rings, leftover).
    """
    pool = [list(c) for c in chains if len(c) >= 2]
    rings, leftover = [], 0
    while pool:
        chain = pool.pop()
        grew = True
        while grew:
            head, tail = chain[0], chain[-1]
            if math.hypot(head[0] - tail[0], head[1] - tail[1]) <= tol and len(chain) >= 3:
                break
            grew = False
            for i, cand in enumerate(pool):
                c0, c1 = cand[0], cand[-1]
                if math.hypot(tail[0] - c0[0], tail[1] - c0[1]) <= tol:
                    chain += cand[1:]
                elif math.hypot(tail[0] - c1[0], tail[1] - c1[1]) <= tol:
                    chain += cand[-2::-1]
                elif math.hypot(head[0] - c1[0], head[1] - c1[1]) <= tol:
                    chain = cand[:-1] + chain
                elif math.hypot(head[0] - c0[0], head[1] - c0[1]) <= tol:
                    chain = cand[::-1][:-1] + chain
                else:
                    continue
                pool.pop(i)
                grew = True
                break
        head, tail = chain[0], chain[-1]
        if math.hypot(head[0] - tail[0], head[1] - tail[1]) <= tol and len(chain) >= 4:
            rings.append(chain[:-1])
        else:
            leftover += 1
    return rings, leftover


# ---------------------------------------------------------------------------
# DXF import
# ---------------------------------------------------------------------------
def rings_from_dxf(data: bytes, unit_scale: float = 1.0) -> ShapeGeometry:
    """Extract closed rings from DXF bytes.

    ``unit_scale`` multiplies drawing units into inches (1.0 for inch
    drawings, 1/25.4 for millimeter drawings).
    """
    doc, _auditor = _ezrecover.read(io.BytesIO(data))
    msp = doc.modelspace()

    closed, open_chains, skipped = [], [], {}
    for e in msp:
        kind = e.dxftype()
        if kind not in _DXF_PATH_TYPES:
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        try:
            p = _ezpath.make_path(e)
        except Exception:
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        pts = [(v.x * unit_scale, v.y * unit_scale)
               for v in p.flattening(distance=FLATTEN_DISTANCE_IN / max(unit_scale, 1e-9))]
        pts = _dedupe_ring(pts)
        if len(pts) < 2:
            continue
        if p.is_closed and len(pts) >= 3:
            closed.append(pts)
        else:
            open_chains.append(pts)

    warnings = []
    joined, leftover = _join_open_chains(open_chains)
    if leftover:
        warnings.append(
            f"{leftover} open path(s) could not be joined into closed loops and were ignored."
        )
    for kind, count in sorted(skipped.items()):
        warnings.append(
            f"Skipped {count} × {kind} entit{'y' if count == 1 else 'ies'} "
            "(only line/arc/polyline/spline geometry is used — explode blocks and text to outlines)."
        )

    rings = _clean_rings(closed + joined)
    if not rings:
        raise ValueError(
            "No closed outlines found in the DXF. The shape must be closed "
            "polylines/splines/circles (explode text and blocks first)."
        )
    return ShapeGeometry(_normalize_rings(rings), warnings)


# ---------------------------------------------------------------------------
# SVG import
# ---------------------------------------------------------------------------
def rings_from_svg(data: bytes, unit_scale: float = 1.0 / SVG_PX_PER_IN) -> ShapeGeometry:
    """Extract closed rings from SVG bytes.

    ``unit_scale`` multiplies SVG user units into inches (default treats user
    units as CSS px at 96/inch). The y-axis is flipped to y-up for DXF.
    """
    from svgelements import SVG, Path as SvgPath, Shape, Text, Close, Line, Move

    svg = SVG.parse(io.BytesIO(data), reify=True, ppi=SVG_PX_PER_IN)

    closed, open_chains, warnings = [], [], []
    text_count = 0
    for element in svg.elements():
        if isinstance(element, Text):
            text_count += 1
            continue
        if not isinstance(element, Shape):
            continue
        try:
            path = element if isinstance(element, SvgPath) else SvgPath(element)
            path.reify()
        except Exception:
            continue
        for sub in path.as_subpaths():
            sub = SvgPath(sub)
            pts = []
            is_closed = False
            for seg in sub:
                if isinstance(seg, Move):
                    if seg.end is not None:
                        pts.append((seg.end.x, seg.end.y))
                    continue
                if isinstance(seg, Close):
                    is_closed = True
                    continue
                if seg.end is None or seg.start is None:
                    continue
                if isinstance(seg, Line):
                    pts.append((seg.end.x, seg.end.y))
                    continue
                try:
                    length = seg.length(error=1e-4)
                except Exception:
                    length = 0.0
                n = max(2, int(math.ceil((length * unit_scale) / CHORD_IN)))
                for i in range(1, n + 1):
                    p = seg.point(i / n)
                    pts.append((p.x, p.y))
            # inches, y-up
            pts = _dedupe_ring([(x * unit_scale, -y * unit_scale) for x, y in pts])
            if len(pts) < 2:
                continue
            if not is_closed and len(pts) >= 3:
                if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= JOIN_TOL_IN:
                    is_closed = True
            if is_closed and len(pts) >= 3:
                closed.append(pts)
            else:
                open_chains.append(pts)

    if text_count:
        warnings.append(
            f"Skipped {text_count} live <text> element(s) — convert text to outlines/paths "
            "in your design tool before exporting."
        )
    joined, leftover = _join_open_chains(open_chains)
    if leftover:
        warnings.append(
            f"{leftover} open path(s) could not be joined into closed loops and were ignored."
        )

    rings = _clean_rings(closed + joined)
    if not rings:
        raise ValueError(
            "No closed outlines found in the SVG. The shape must be closed paths "
            "(convert text/strokes to outlines first)."
        )
    return ShapeGeometry(_normalize_rings(rings), warnings)


# ---------------------------------------------------------------------------
# AI / PDF import (best effort)
# ---------------------------------------------------------------------------
_PDF_TOKEN = _re.compile(
    rb"(-?(?:\d+\.?\d*|\.\d+))"          # number
    rb"|/[^\s/<>\[\]()%]*"               # name (skipped)
    rb"|(BT.*?ET)"                        # text blocks (skipped)
    rb"|(\((?:\\.|[^\\()])*\))"          # string (skipped)
    rb"|(<[^>]*>)"                        # hex string / dict-ish (skipped)
    rb"|(\[[^\]]*\])"                     # array (skipped)
    rb"|([A-Za-z'\"*]{1,3})",            # operator
    _re.DOTALL,
)
_PDF_PATH_HINT = _re.compile(rb"\b(re|m|l|c|v|y)\b")


def _mostly_text(blob: bytes) -> bool:
    sample = blob[:4096]
    if not sample:
        return False
    printable = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
    return printable / len(sample) > 0.9


def _pdf_content_streams(data: bytes):
    """Yield candidate vector content streams from raw PDF/AI bytes."""
    for m in _re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        if end < 0:
            continue
        blob = data[start:end].rstrip(b"\r\n")
        try:
            candidate = zlib.decompress(blob)
        except zlib.error:
            candidate = blob
        if _mostly_text(candidate) and _PDF_PATH_HINT.search(candidate):
            yield candidate


def _flatten_cubic(p0, p1, p2, p3, unit_scale):
    poly_len = (
        math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        + math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        + math.hypot(p3[0] - p2[0], p3[1] - p2[1])
    )
    n = min(64, max(2, int(math.ceil((poly_len * unit_scale) / CHORD_IN))))
    pts = []
    for i in range(1, n + 1):
        t = i / n
        u = 1.0 - t
        x = (u**3) * p0[0] + 3 * (u**2) * t * p1[0] + 3 * u * (t**2) * p2[0] + (t**3) * p3[0]
        y = (u**3) * p0[1] + 3 * (u**2) * t * p1[1] + 3 * u * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((x, y))
    return pts


def rings_from_pdf(data: bytes, unit_scale: float = 1.0 / PDF_PT_PER_IN) -> ShapeGeometry:
    """Extract closed vector paths from PDF or PDF-compatible Adobe ``.ai`` bytes.

    Best-effort content-stream reader: follows m/l/c/v/y/re/h path construction
    with q/Q/cm transforms, keeps painted paths (fill or stroke), and discards
    pure clipping paths (``W n``) and text. PDF user space is points (72/in),
    already y-up. Save Illustrator files with "Create PDF Compatible File" on
    (the default) or use Save As → PDF.
    """
    closed, open_chains = [], []
    paint_ops = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"S", b"s", b"n"}
    close_first = {b"b", b"b*", b"s"}

    found_stream = False
    for content in _pdf_content_streams(data):
        found_stream = True
        ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        stack: list = []
        nums: list[float] = []
        subpaths: list[tuple[list, bool]] = []
        cur: list = []
        cur_closed = False
        clip_pending = False

        def xf(x, y, _m=None):
            a, b, c, d, e, f = _m if _m is not None else ctm
            return (a * x + c * y + e, b * x + d * y + f)

        def commit_cur():
            nonlocal cur, cur_closed
            if len(cur) >= 2:
                subpaths.append((cur, cur_closed))
            cur, cur_closed = [], False

        for tok in _PDF_TOKEN.finditer(content):
            num, _txt, _s, _hx, _arr, op = tok.group(1), tok.group(2), tok.group(3), tok.group(4), tok.group(5), tok.group(6)
            if num is not None:
                try:
                    nums.append(float(num))
                except ValueError:
                    nums = []
                continue
            if op is None:
                nums = []
                continue
            try:
                if op == b"q":
                    stack.append(ctm)
                elif op == b"Q":
                    ctm = stack.pop() if stack else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                elif op == b"cm" and len(nums) >= 6:
                    a, b, c, d, e, f = nums[-6:]
                    A, B, C, D, E, F = ctm
                    ctm = (a * A + b * C, a * B + b * D,
                           c * A + d * C, c * B + d * D,
                           e * A + f * C + E, e * B + f * D + F)
                elif op == b"m" and len(nums) >= 2:
                    commit_cur()
                    cur = [xf(nums[-2], nums[-1])]
                elif op == b"l" and len(nums) >= 2 and cur:
                    cur.append(xf(nums[-2], nums[-1]))
                elif op in (b"c", b"v", b"y") and cur:
                    p0 = cur[-1]
                    if op == b"c" and len(nums) >= 6:
                        p1 = xf(nums[-6], nums[-5]); p2 = xf(nums[-4], nums[-3]); p3 = xf(nums[-2], nums[-1])
                    elif op == b"v" and len(nums) >= 4:
                        p1 = p0; p2 = xf(nums[-4], nums[-3]); p3 = xf(nums[-2], nums[-1])
                    elif op == b"y" and len(nums) >= 4:
                        p1 = xf(nums[-4], nums[-3]); p3 = xf(nums[-2], nums[-1]); p2 = p3
                    else:
                        nums = []
                        continue
                    cur.extend(_flatten_cubic(p0, p1, p2, p3, unit_scale))
                elif op == b"re" and len(nums) >= 4:
                    commit_cur()
                    x, y, w, h = nums[-4:]
                    subpaths.append(([xf(x, y), xf(x + w, y), xf(x + w, y + h), xf(x, y + h)], True))
                elif op == b"h":
                    cur_closed = True
                elif op in (b"W", b"W*"):
                    clip_pending = True
                elif op in paint_ops:
                    if op in close_first:
                        cur_closed = True
                    commit_cur()
                    if op == b"n" and clip_pending:
                        subpaths = []          # pure clipping path — discard
                    elif op != b"n":
                        for pts, was_closed in subpaths:
                            scaled = _dedupe_ring([(x * unit_scale, y * unit_scale) for x, y in pts])
                            if len(scaled) < 2:
                                continue
                            if (was_closed or op not in (b"S",)) and len(scaled) >= 3:
                                closed.append(scaled)
                            else:
                                open_chains.append(scaled)
                        subpaths = []
                    else:
                        subpaths = []
                    clip_pending = False
            finally:
                nums = []

    if not found_stream:
        raise ValueError(
            "No vector content found. Save the Illustrator file with "
            "'Create PDF Compatible File' checked, use Save As → PDF, or export SVG."
        )

    warnings = [
        "AI/PDF import is best-effort — check the preview. For exact results export "
        "SVG from Illustrator (text converted to outlines)."
    ]
    joined, leftover = _join_open_chains(open_chains)
    if leftover:
        warnings.append(
            f"{leftover} open path(s) could not be joined into closed loops and were ignored."
        )
    rings = _clean_rings(closed + joined)
    if not rings:
        raise ValueError(
            "No closed outlines found in the AI/PDF. Convert text/strokes to outlines "
            "and make sure shapes are filled paths, or export SVG instead."
        )
    return ShapeGeometry(_normalize_rings(rings), warnings)


def load_shape(data: bytes, filename: str, unit_scale: float | None = None) -> ShapeGeometry:
    """Dispatch on extension. ``unit_scale`` = drawing/user units → inches."""
    name = (filename or "").lower()
    if name.endswith(".svg"):
        return rings_from_svg(data, 1.0 / SVG_PX_PER_IN if unit_scale is None else unit_scale)
    if name.endswith(".dxf"):
        return rings_from_dxf(data, 1.0 if unit_scale is None else unit_scale)
    if name.endswith(".ai") or name.endswith(".pdf"):
        return rings_from_pdf(data, 1.0 / PDF_PT_PER_IN if unit_scale is None else unit_scale)
    if name.endswith(".eps"):
        raise ValueError(
            "EPS (PostScript) is not supported. From Illustrator use "
            "File → Save As → SVG or PDF (or .ai with PDF compatibility, the default)."
        )
    raise ValueError(f"Unsupported file type: {filename} (upload .dxf, .svg, .ai, or .pdf)")


# ---------------------------------------------------------------------------
# Infill
# ---------------------------------------------------------------------------
def _segment_arrays(rings):
    x0, y0, x1, y1 = [], [], [], []
    for ring in rings:
        n = len(ring)
        for i in range(n):
            ax, ay = ring[i]
            bx, by = ring[(i + 1) % n]
            x0.append(ax); y0.append(ay); x1.append(bx); y1.append(by)
    return (np.asarray(x0), np.asarray(y0), np.asarray(x1), np.asarray(y1))


def _classify_centers(pts, segs):
    """(inside, min_dist, nearest_x, nearest_y) for each point vs the boundary."""
    sx0, sy0, sx1, sy1 = segs
    n_seg = len(sx0)
    dx = sx1 - sx0
    dy = sy1 - sy0
    seg_len2 = np.maximum(dx * dx + dy * dy, 1e-18)

    inside = np.zeros(len(pts), dtype=bool)
    min_d = np.zeros(len(pts))
    near_x = np.zeros(len(pts))
    near_y = np.zeros(len(pts))
    # Chunked broadcasting keeps peak memory bounded (~8 arrays × chunk × n_seg).
    chunk = max(64, int(2_000_000 / max(n_seg, 1)))
    for start in range(0, len(pts), chunk):
        px = pts[start:start + chunk, 0][:, None]   # (P,1)
        py = pts[start:start + chunk, 1][:, None]
        # Even-odd ray cast toward +x.
        straddle = (sy0[None, :] > py) != (sy1[None, :] > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = sx0[None, :] + (py - sy0[None, :]) * dx[None, :] / np.where(
                dy[None, :] == 0.0, np.inf, dy[None, :]
            )
        crossings = np.sum(straddle & (px < x_at), axis=1)
        # Min distance point → segment (and the nearest boundary point).
        t = ((px - sx0[None, :]) * dx[None, :] + (py - sy0[None, :]) * dy[None, :]) / seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        qx = sx0[None, :] + t * dx[None, :]
        qy = sy0[None, :] + t * dy[None, :]
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        idx = np.argmin(d2, axis=1)
        rows = np.arange(d2.shape[0])
        sl = slice(start, start + d2.shape[0])
        inside[sl] = (crossings % 2) == 1
        min_d[sl] = np.sqrt(d2[rows, idx])
        near_x[sl] = qx[rows, idx]
        near_y[sl] = qy[rows, idx]
    return inside, min_d, near_x, near_y


def _try_nudge(px, py, segs, clearance, budget):
    """Slide a point inward (small, capped steps) until the hole clears.

    Returns the new (x, y) or None. Direction is always along the local inward
    normal (away from the nearest boundary point when inside, toward and past
    it when outside). Re-classifies after each step so narrow strokes where
    both sides conflict are rejected, not force-fitted.
    """
    moved = 0.0
    for _ in range(4):
        inside, d, qx, qy = _classify_centers(np.array([[px, py]]), segs)
        if inside[0] and d[0] >= clearance - 1e-9:
            return (px, py)
        vx, vy = px - qx[0], py - qy[0]
        norm = math.hypot(vx, vy)
        if norm < 1e-12:
            return None  # exactly on the outline — no stable normal
        if not inside[0]:
            vx, vy = -vx, -vy
        signed = d[0] if inside[0] else -d[0]
        need = (clearance - signed) + 1e-6
        step = min(need, budget - moved)
        if step <= 1e-9:
            return None
        px += vx / norm * step
        py += vy / norm * step
        moved += step
    inside, d, _, _ = _classify_centers(np.array([[px, py]]), segs)
    return (px, py) if inside[0] and d[0] >= clearance - 1e-9 else None


# Corner detection: direction change sharper than this is an anchored corner.
CORNER_TURN_DEG = 40.0
# Per-edge justified spacing may deviate from pitch by at most this fraction
# before an edge gains/loses a hole instead.
CONTOUR_SPACING_FLEX = 0.5


def _fits_one(px, py, segs, clearance):
    inside, d, _, _ = _classify_centers(np.array([[px, py]]), segs)
    return bool(inside[0]) and float(d[0]) >= clearance - 1e-9


def _inward_dir(px, py, nx, ny, segs, probe):
    """Orient a unit normal so it points into the filled region (even-odd probe)."""
    inside, _, _, _ = _classify_centers(np.array([[px + nx * probe, py + ny * probe]]), segs)
    return (nx, ny) if inside[0] else (-nx, -ny)


def _ring_corners(ring):
    """Indices of vertices whose direction change exceeds CORNER_TURN_DEG."""
    n = len(ring)
    corners = []
    for i in range(n):
        ax, ay = ring[i - 1]
        bx, by = ring[i]
        cx, cy = ring[(i + 1) % n]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        l1 = math.hypot(*v1)
        l2 = math.hypot(*v2)
        if l1 < 1e-12 or l2 < 1e-12:
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        if math.degrees(math.acos(cosang)) > CORNER_TURN_DEG:
            corners.append(i)
    return corners


def _corner_candidate(ring, i, segs, clearance):
    """Hole center anchoring a sharp corner + its along-edge footprint.

    The center sits on the inward angle bisector, deep enough to clear both
    edges (clearance / sin(half-angle), capped). Returns
    ``((x, y), along_trim)`` where ``along_trim`` is how far along each edge
    the corner hole projects — edge stations are justified over the span
    *between* corner holes so every gap on the side reads uniform.
    """
    n = len(ring)
    ax, ay = ring[i - 1]
    bx, by = ring[i]
    cx, cy = ring[(i + 1) % n]
    u = (bx - ax, by - ay)
    v = (cx - bx, cy - by)
    lu, lv = math.hypot(*u), math.hypot(*v)
    if lu < 1e-12 or lv < 1e-12:
        return None, 0.0
    # Inward normals of the two edges meeting at the corner.
    nu = _inward_dir((ax + bx) / 2, (ay + by) / 2, -u[1] / lu, u[0] / lu, segs, clearance * 0.25)
    nv = _inward_dir((bx + cx) / 2, (by + cy) / 2, -v[1] / lv, v[0] / lv, segs, clearance * 0.25)
    bx_, by_ = nu[0] + nv[0], nu[1] + nv[1]
    lb = math.hypot(bx_, by_)
    if lb < 1e-9:
        return None, 0.0
    bis = (bx_ / lb, by_ / lb)
    # Interior half-angle between the edges: sin(half) from the bisector geometry.
    cosang = max(-1.0, min(1.0, (-(u[0]) * v[0] - u[1] * v[1]) / (lu * lv)))
    half = max(math.acos(cosang) / 2.0, 1e-3)
    depth = min(clearance / max(math.sin(half), 1.0 / 3.0), 3.0 * clearance)
    along_trim = depth * math.cos(half)
    return (bx + bis[0] * depth, by + bis[1] * depth), along_trim


def _edge_stations(ring, i0, i1, spacing, trim0=0.0, trim1=0.0):
    """Justified stations along ring vertices i0→i1.

    Stations span the arc between the two corner holes' footprints
    (``trim0``/``trim1`` in from each end) at ``effective / round(effective /
    spacing)`` so the corner-to-first-hole gap matches the interior gaps.
    """
    n_ring = len(ring)
    pts = [ring[i0]]
    j = i0
    while j % n_ring != i1 % n_ring:
        j += 1
        pts.append(ring[j % n_ring])
    seg_len = [math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
               for k in range(len(pts) - 1)]
    total = sum(seg_len)
    effective = total - trim0 - trim1
    if effective < spacing * (1.0 - CONTOUR_SPACING_FLEX):
        return []
    n = max(1, round(effective / spacing))
    out = []
    for k in range(1, n):
        target = trim0 + effective * k / n
        acc = 0.0
        for s, L in enumerate(seg_len):
            if acc + L >= target - 1e-12:
                t = (target - acc) / max(L, 1e-12)
                x = pts[s][0] + (pts[s + 1][0] - pts[s][0]) * t
                y = pts[s][1] + (pts[s + 1][1] - pts[s][1]) * t
                # local inward normal
                dx, dy = pts[s + 1][0] - pts[s][0], pts[s + 1][1] - pts[s][1]
                out.append((x, y, -dy / max(L, 1e-12), dx / max(L, 1e-12)))
                break
            acc += L
    return out


def _loop_stations(ring, spacing):
    """Evenly spaced stations around a smooth closed loop (no corners)."""
    n_ring = len(ring)
    seg_len = [math.hypot(ring[(k + 1) % n_ring][0] - ring[k][0],
                          ring[(k + 1) % n_ring][1] - ring[k][1]) for k in range(n_ring)]
    total = sum(seg_len)
    if total < spacing * (1.0 - CONTOUR_SPACING_FLEX):
        return []
    n = max(3, round(total / spacing))
    out = []
    for k in range(n):
        target = total * k / n
        acc = 0.0
        for s in range(n_ring):
            L = seg_len[s]
            if acc + L >= target - 1e-12:
                t = (target - acc) / max(L, 1e-12)
                p0, p1 = ring[s], ring[(s + 1) % n_ring]
                x = p0[0] + (p1[0] - p0[0]) * t
                y = p0[1] + (p1[1] - p0[1]) * t
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                out.append((x, y, -dy / max(L, 1e-12), dx / max(L, 1e-12)))
                break
            acc += L
    return out


def contour_hole_centers(shape: ShapeGeometry, hole_dia: float, spacing: float,
                         margin: float, nudge_max: float = 0.0):
    """Perimeter outline row: one hole ring hugging every outline at exact
    clearance, corner-anchored, spacing justified per edge.

    Sharp corners (letter apexes, stem/crossbar joints) always get a hole —
    that is what makes the form read. Between corners, holes are spaced
    evenly at ``total_edge_length / round(length / spacing)`` so each side
    looks uniform; the per-side deviation from nominal spacing is what a
    human eye tolerates best. Returns (centers, dropped_count).
    """
    segs = _segment_arrays(shape.rings)
    clearance = margin + hole_dia / 2.0
    min_cc = max(1.05 * hole_dia, 0.55 * spacing)

    anchors, stations = [], []
    for ring in shape.rings:
        corners = _ring_corners(ring)
        if corners:
            trims = {}
            for ci in corners:
                cand, trim = _corner_candidate(ring, ci, segs, clearance)
                trims[ci] = trim
                if cand is not None:
                    anchors.append(cand)
            for a, b in zip(corners, corners[1:] + [corners[0] + len(ring)]):
                stations.extend(_edge_stations(
                    ring, a % len(ring), b % len(ring), spacing,
                    trims[a % len(ring)], trims[b % len(ring)],
                ))
        else:
            # Smooth loop (circle / O counter): even stations all the way around.
            stations.extend(_loop_stations(ring, spacing))

    kept, dropped = [], 0
    budget = nudge_max if nudge_max > 0 else 0.35 * spacing

    def _admit(cand):
        nonlocal dropped
        if cand is None:
            dropped += 1
            return
        x, y = cand
        if not _fits_one(x, y, segs, clearance):
            moved = _try_nudge(x, y, segs, clearance, budget)
            if moved is None:
                dropped += 1
                return
            x, y = moved
        if any((x - kx) ** 2 + (y - ky) ** 2 < min_cc * min_cc for kx, ky in kept):
            dropped += 1
            return
        kept.append((x, y))

    # Two facing wall rows need this much local stroke width; below it a
    # single centerline row (narrow_fill) reads better than colliding rings.
    two_row_min = 2.0 * clearance + 1.05 * hole_dia

    for cand in anchors:                      # corners first: they anchor the form
        _admit(cand)
    for sx, sy, nx, ny in stations:
        ix, iy = _inward_dir(sx, sy, nx, ny, segs, clearance * 0.25)
        w = _chord_width(sx, sy, ix, iy, segs)
        if w is not None and w < two_row_min:
            continue                          # leave the zone to the midline row
        _admit((sx + ix * clearance, sy + iy * clearance))
    return kept, dropped


def _grid_candidates(shape: ShapeGeometry, hole_dia, pitch, pattern, stagger_angle,
                     margin, phase=(0.0, 0.0)):
    """Grid over the shape bbox with an explicit phase offset (for alignment sweeps)."""
    x0, y0, x1, y1 = shape.bbox
    hr = hole_dia / 2.0
    min_x, max_x = x0 + margin + hr, x1 - margin - hr
    min_y, max_y = y0 + margin + hr, y1 - margin - hr
    if pattern == "straight":
        row_step, col_off = pitch, 0.0
    else:
        alpha = math.radians(stagger_angle)
        row_step, col_off = pitch * math.sin(alpha), pitch * math.cos(alpha)
    out = []
    y = min_y + (phase[1] % row_step) - row_step
    row = 0
    while y <= max_y + 1e-9:
        if y >= min_y - 1e-9:
            offset = col_off if row % 2 else 0.0
            x = min_x + ((phase[0] + offset) % pitch) - pitch
            while x <= max_x + 1e-9:
                if x >= min_x - 1e-9:
                    out.append((x, y))
                x += pitch
        y += row_step
        row += 1
    return out


# Tightest manufacturable pitch, as a multiple of hole diameter (web ≈ 0.2×Ø).
MIN_PITCH_FACTOR = 1.2
# Relaxation iterations for the elastic lattice.
ELASTIC_ITERATIONS = 8


def _neighbor_pairs(pts, active, r):
    """Index pairs of active points closer than r (uniform-grid hash)."""
    cell = max(r, 1e-9)
    buckets: dict[tuple[int, int], list[int]] = {}
    idxs = np.nonzero(active)[0]
    for i in idxs:
        buckets.setdefault((int(pts[i, 0] // cell), int(pts[i, 1] // cell)), []).append(i)
    r2 = r * r
    pairs = []
    for (cx, cy), members in buckets.items():
        cand = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cand.extend(buckets.get((cx + ox, cy + oy), []))
        for i in members:
            for j in cand:
                if j <= i:
                    continue
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                if dx * dx + dy * dy < r2:
                    pairs.append((i, j))
    return pairs


def elastic_lattice_fill(shape: ShapeGeometry, segs, hole_dia: float, pitch: float,
                         pattern: str, stagger_angle: float, margin: float,
                         flex: float, avoid=None, avoid_dist: float = 0.0):
    """Fill with ONE lattice that keeps its geometry but breathes within ±flex.

    The answer to "even fill that still reads as a 60° pattern":

    1. **Global fit** — sweep the lattice's uniform scale (±flex, exact
       stagger angle preserved) and phase; keep the configuration where the
       most holes are inside or within reach of the boundary.
    2. **Local relaxation** — iterate: holes violating edge clearance slide
       along the boundary normal until they clear; crowded holes push apart.
       Every hole is tethered to its original lattice node by ``flex×pitch``,
       so the field can never drift into visible distortion.
    3. Holes that cannot reach validity inside their tether drop out.

    Edge holes end up hugging the outline at exact clearance (tight
    perimeter) while interior holes stay on-lattice — one continuous system,
    no ring/grid seams. Returns (holes, moved_count, dropped_count).
    """
    clearance = margin + hole_dia / 2.0
    straight = pattern == "straight"
    alpha = math.radians(stagger_angle)

    scales = sorted({1.0, 1.0 - flex, 1.0 - flex / 2.0, 1.0 + flex / 2.0, 1.0 + flex})
    best = None
    for s in scales:
        p_s = pitch * s
        if p_s < MIN_PITCH_FACTOR * hole_dia:
            continue
        row_step = p_s if straight else p_s * math.sin(alpha)
        for i in range(3):
            for j in range(3):
                cand = _grid_candidates(shape, hole_dia, p_s, pattern, stagger_angle,
                                        margin, phase=(p_s * i / 3.0, row_step * j / 3.0))
                if not cand:
                    continue
                pts = np.asarray(cand, dtype=float)
                inside, d, _, _ = _classify_centers(pts, segs)
                viable = inside & (d >= clearance - flex * p_s)
                strict = inside & (d >= clearance)
                score = (int(viable.sum()), int(strict.sum()), -abs(s - 1.0))
                if best is None or score > best[0]:
                    best = (score, pts[viable], p_s)
    if best is None or len(best[1]) == 0:
        return [], 0, 0

    pts = best[1].copy()
    p_s = best[2]
    orig = pts.copy()
    n0 = len(pts)
    cap = flex * p_s
    r_min = max(1.05 * hole_dia, (1.0 - flex) * p_s * 0.95)
    active = np.ones(n0, dtype=bool)
    if avoid is not None and len(avoid) and avoid_dist > 0:
        av = np.asarray(avoid, dtype=float)

        def _too_close(p):
            return np.min((av[:, 0] - p[0]) ** 2 + (av[:, 1] - p[1]) ** 2) < avoid_dist ** 2
    else:
        av = None

    for _ in range(ELASTIC_ITERATIONS):
        inside, d, qx, qy = _classify_centers(pts, segs)
        signed = np.where(inside, d, -d)
        need = clearance - signed
        viol = active & (need > 1e-9)
        if np.any(viol):
            vx = pts[viol, 0] - qx[viol]
            vy = pts[viol, 1] - qy[viol]
            norm = np.maximum(np.hypot(vx, vy), 1e-12)
            sign = np.where(inside[viol], 1.0, -1.0)
            step = need[viol] + 1e-4
            pts[viol, 0] += sign * vx / norm * step
            pts[viol, 1] += sign * vy / norm * step
        # crowding: push apart pairs closer than r_min
        for i, j in _neighbor_pairs(pts, active, r_min):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                continue
            push = (r_min - dist) / 2.0
            pts[i, 0] += dx / dist * push
            pts[i, 1] += dy / dist * push
            pts[j, 0] -= dx / dist * push
            pts[j, 1] -= dy / dist * push
        # tether: nothing strays farther than the flex budget from its node
        disp = pts - orig
        dn = np.hypot(disp[:, 0], disp[:, 1])
        over = active & (dn > cap)
        if np.any(over):
            pts[over] = orig[over] + disp[over] * (cap / dn[over])[:, None]

    inside, d, _, _ = _classify_centers(pts, segs)
    valid = active & inside & (d >= clearance - 1e-6)
    disp = np.hypot(pts[:, 0] - orig[:, 0], pts[:, 1] - orig[:, 1])
    # Final admission: least-moved first, enforcing separation (and ring gap).
    kept: list[tuple[float, float]] = []
    moved = 0
    admit_r = max(1.05 * hole_dia, r_min * 0.95)
    for i in np.argsort(disp):
        if not valid[i]:
            continue
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if any((x - kx) ** 2 + (y - ky) ** 2 < admit_r * admit_r for kx, ky in kept):
            continue
        if av is not None and _too_close((x, y)):
            continue
        kept.append((x, y))
        if disp[i] > 1e-6:
            moved += 1
    return kept, moved, n0 - len(kept)
# Grid rows must sit at least this fraction of pitch from the perimeter ring
# (shared by the exclusion zone and the rows-across estimate).
RING_EXCLUSION_FACTOR = 0.8


def _chord_width(px, py, dx, dy, segs):
    """Length of the inward chord from boundary point (px,py) along (dx,dy):
    the local stroke width. Returns None if the ray never exits (bad normal)."""
    sx0, sy0, sx1, sy1 = segs
    ex = sx1 - sx0
    ey = sy1 - sy0
    denom = dx * ey - dy * ex
    ax = sx0 - px
    ay = sy0 - py
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ax * ey - ay * ex) / denom
        s = (ax * dy - ay * dx) / denom
    valid = (np.abs(denom) > 1e-12) & (s >= -1e-9) & (s <= 1.0 + 1e-9) & (t > 1e-6)
    if not np.any(valid):
        return None
    return float(np.min(t[valid]))


def _boundary_chords(shape: ShapeGeometry, segs, step: float):
    """(x, y, nx, ny, width) chords sampled around every ring at ~step spacing."""
    out = []
    for ring in shape.rings:
        for sx, sy, nx, ny in _loop_stations(ring, step):
            ix, iy = _inward_dir(sx, sy, nx, ny, segs, 1e-4)
            w = _chord_width(sx, sy, ix, iy, segs)
            if w is not None:
                out.append((sx, sy, ix, iy, w))
    return out


def analyze_strokes(shape: ShapeGeometry, hole_dia: float, pitch: float,
                    margin: float) -> StrokeAnalysis | None:
    """Measure stroke widths and derive what it takes to get 3 rows across.

    Rows across a stroke of width ``w``: the two perimeter-ring rows sit at
    ``clearance`` from each wall, and interior rows need
    ``RING_EXCLUSION_FACTOR × pitch`` of room, so
    ``rows ≈ 2 + floor((w − 2·clearance) / (RING_EXCLUSION_FACTOR·pitch)) − 1``
    (bounded below by what physically fits).
    """
    segs = _segment_arrays(shape.rings)
    step = max(min(pitch, 0.5), sum(shape.extents) / 400.0)
    widths = np.asarray([c[4] for c in _boundary_chords(shape, segs, step)])
    if len(widths) == 0:
        return None
    med = float(np.median(widths))
    thin = float(np.percentile(widths, 10))
    clearance = margin + hole_dia / 2.0

    def rows(w):
        if w < hole_dia + margin:            # not even one relaxed hole
            return 0
        if w < 2.0 * clearance + 1.05 * hole_dia:
            return 1                          # single (center) row territory
        return 2 + max(0, int((w - 2.0 * clearance) / (RING_EXCLUSION_FACTOR * pitch)) - 1)

    fit_pitch = (med - 2.0 * clearance) / (2.0 * RING_EXCLUSION_FACTOR)
    floor_pitch = MIN_PITCH_FACTOR * hole_dia
    fit_dia = None
    if fit_pitch >= floor_pitch:
        fit_pitch = round(min(fit_pitch, pitch), 4)
        fit_scale = None
    else:
        # Even the tightest pitch can't reach 3 rows: the artwork must grow —
        # or the hole must shrink. 3 rows at the aesthetic pitch of 2×Ø needs
        # w ≥ 2·margin + Ø + 2·0.8·(2Ø)  →  Ø ≤ (w − 2·margin) / 4.2.
        need_w = 2.0 * clearance + 2.0 * RING_EXCLUSION_FACTOR * floor_pitch
        fit_pitch = None
        fit_scale = round(need_w / max(med, 1e-9), 2)
        dia = (med - 2.0 * margin) / 4.2
        if dia > 0.02:
            fit_dia = round(dia, 3)
    return StrokeAnalysis(round(med, 3), round(thin, 3), rows(med), rows(thin),
                          fit_pitch, fit_scale, fit_dia)


def _midline_rescue(shape: ShapeGeometry, segs, ring_holes, other_holes,
                    hole_dia: float, pitch: float, margin: float):
    """Centerline row through strokes too narrow for interior grid rows.

    Anywhere the local stroke width leaves no room between the perimeter-ring
    rows, chain holes down the stroke's medial line at ~pitch spacing. Edge
    clearance may relax down to ``hole_r + margin/2`` in the narrowest zones —
    coverage of the form beats nominal margin there. Returns (added, dropped).
    """
    clearance = margin + hole_dia / 2.0
    relaxed = hole_dia / 2.0 + margin * 0.5
    hr = hole_dia / 2.0
    # Only strokes too narrow for a grid row between the rings need rescue.
    narrow_limit = 2.0 * clearance + 2.0 * RING_EXCLUSION_FACTOR * pitch
    ring_arr = np.asarray(ring_holes, dtype=float) if ring_holes else None
    other_arr = np.asarray(other_holes, dtype=float) if other_holes else None
    min_cc = 1.05 * hole_dia

    added, dropped = [], 0
    step = min(pitch / 2.0, 0.5)
    for ring in shape.rings:
        for sx, sy, nx, ny in _loop_stations(ring, step):
            ix, iy = _inward_dir(sx, sy, nx, ny, segs, 1e-4)
            w = _chord_width(sx, sy, ix, iy, segs)
            if w is None or w >= narrow_limit or w < hole_dia + margin:
                continue
            mx, my = sx + ix * w / 2.0, sy + iy * w / 2.0
            # Skip zones the grid already covers.
            if other_arr is not None and len(other_arr) and np.min(
                (other_arr[:, 0] - mx) ** 2 + (other_arr[:, 1] - my) ** 2
            ) < (0.75 * pitch) ** 2:
                continue
            req = clearance if w >= 2.0 * clearance + min_cc else max(relaxed, hr + 1e-4)
            if not _fits_one(mx, my, segs, req):
                dropped += 1
                continue
            if any((mx - axx) ** 2 + (my - ayy) ** 2 < (0.85 * pitch) ** 2
                   for axx, ayy in added):
                continue
            if ring_arr is not None and len(ring_arr) and np.min(
                (ring_arr[:, 0] - mx) ** 2 + (ring_arr[:, 1] - my) ** 2
            ) < min_cc * min_cc:
                dropped += 1
                continue
            added.append((mx, my))
    return added, dropped


def _corner_anchor_pass(shape: ShapeGeometry, segs, existing, hole_dia: float,
                        pitch: float, margin: float, flex: float):
    """Add a hole at any sharp corner (letter apex) the fill left bare."""
    clearance = margin + hole_dia / 2.0
    min_gap = max(1.05 * hole_dia, 0.6 * pitch)
    added = []
    for ring in shape.rings:
        for ci in _ring_corners(ring):
            cand, _ = _corner_candidate(ring, ci, segs, clearance)
            if cand is None:
                continue
            x, y = cand
            if not _fits_one(x, y, segs, clearance):
                moved = _try_nudge(x, y, segs, clearance, max(flex, 0.2) * pitch)
                if moved is None:
                    continue
                x, y = moved
            if any((x - hx) ** 2 + (y - hy) ** 2 < min_gap * min_gap
                   for hx, hy in existing + added):
                continue
            added.append((x, y))
    return added


def infill_hole_centers(shape: ShapeGeometry, hole_dia: float, pitch: float,
                        pattern: str, stagger_angle: float, margin: float,
                        nudge_max: float = 0.0, perimeter_row: bool = False,
                        optimize_grid: bool = False, narrow_fill: bool = False,
                        spacing_flex: float = 0.0) -> InfillResult:
    """Hole grid over the shape bbox, filtered to holes that fully fit inside.

    Same grid math and margin semantics as the panel face (`_hole_centers`):
    ``margin`` is the clearance from the boundary to the hole *edge*.

    ``nudge_max`` (inches) lets an almost-fitting boundary hole slide up to
    that distance inward — same diameter, slightly off-grid — so letter edges
    stay readable. Nudged holes are rejected if they would come within one
    hole diameter (center-to-center, +5%) of another hole.

    ``perimeter_row`` adds a corner-anchored, per-edge-justified hole ring
    hugging every outline at exact clearance (see `contour_hole_centers`);
    interior grid holes that would crowd the ring are excluded.

    ``optimize_grid`` sweeps the grid phase (a 4×4 lattice of offsets plus
    the default centered grid) and keeps the alignment that fits the most
    holes — spacing is untouched, so there is zero visual cost.

    ``narrow_fill`` chains a centerline row through strokes too narrow for
    grid rows between the perimeter rings (see `_midline_rescue`); use
    `analyze_strokes` to see measured widths and pitch/scale/Ø guidance.

    ``spacing_flex`` > 0 switches the fill to the elastic lattice
    (`elastic_lattice_fill`): one lattice that keeps the stagger angle but may
    scale/slide/deform locally within ±flex, giving even fill with edge holes
    hugging the outline. Replaces the grid + nudge stack (``nudge_max`` and
    ``optimize_grid`` are ignored); the fraction is of pitch (e.g. 0.15).
    """
    if hole_dia <= 0 or pitch <= 0:
        raise ValueError("hole diameter and pitch must be positive")
    if pitch < hole_dia:
        raise ValueError("pitch must be >= hole diameter (holes would overlap)")
    x0, y0, x1, y1 = shape.bbox
    w, h = x1 - x0, y1 - y0

    # Candidate-count guard before allocating anything.
    est = (w / pitch + 2.0) * (h / (pitch * math.sin(math.radians(stagger_angle)) or pitch) + 2.0)
    if pattern == "staggered":
        est *= 2.0
    if est > MAX_GRID_CANDIDATES:
        raise ValueError(
            f"Pattern would generate ~{int(est):,} candidate holes "
            f"(limit {MAX_GRID_CANDIDATES:,}). Increase the pitch or shrink the shape."
        )

    segs = _segment_arrays(shape.rings)
    clearance = margin + hole_dia / 2.0

    contour, dropped = [], 0
    if perimeter_row:
        contour, c_dropped = contour_hole_centers(shape, hole_dia, pitch, margin, nudge_max)
        dropped += c_dropped
    ring_pts = np.asarray(contour, dtype=float) if contour else None
    # Grid holes must keep visual separation from the perimeter ring.
    ring_excl = 0.8 * pitch

    if spacing_flex > 0.0:
        # Elastic lattice mode: one lattice, globally scaled/aligned (±flex,
        # angle preserved) and locally relaxed, replaces the grid/nudge stack.
        fill, moved, e_dropped = elastic_lattice_fill(
            shape, segs, hole_dia, pitch, pattern, stagger_angle, margin,
            spacing_flex, avoid=contour if contour else None,
            avoid_dist=ring_excl if contour else 0.0,
        )
        dropped += e_dropped
        holes = list(contour) + fill
        holes += _corner_anchor_pass(shape, segs, holes, hole_dia, pitch, margin, spacing_flex)
        midline = []
        if narrow_fill:
            midline, m_dropped = _midline_rescue(
                shape, segs, contour, holes[len(contour):], hole_dia, pitch, margin
            )
            dropped += m_dropped
            holes.extend(midline)
        if len(holes) > MAX_HOLES:
            raise ValueError(
                f"Pattern produced {len(holes):,} holes (limit {MAX_HOLES:,}). "
                "Increase the pitch or shrink the shape."
            )
        return InfillResult(holes, moved, dropped, contour=len(contour), midline=len(midline))

    def _grid_eval(raw):
        """(fits mask, inside, min_d, pts) for a candidate grid, ring-aware."""
        if not raw:
            return None
        pts = np.asarray(raw, dtype=float)
        inside, min_d, _, _ = _classify_centers(pts, segs)
        fits = inside & (min_d >= clearance)
        if ring_pts is not None and len(pts):
            d2 = ((pts[:, None, 0] - ring_pts[None, :, 0]) ** 2
                  + (pts[:, None, 1] - ring_pts[None, :, 1]) ** 2)
            near_ring = np.min(d2, axis=1) < ring_excl * ring_excl
        else:
            near_ring = np.zeros(len(pts), dtype=bool)
        return raw, pts, inside, min_d, fits & ~near_ring, near_ring

    candidates = [_hole_centers(x0, y0, w, h, hole_dia, pitch, pattern, stagger_angle, margin)]
    if optimize_grid:
        row_step = pitch if pattern == "straight" else pitch * math.sin(math.radians(stagger_angle))
        for i in range(4):
            for j in range(4):
                candidates.append(_grid_candidates(
                    shape, hole_dia, pitch, pattern, stagger_angle, margin,
                    phase=(pitch * i / 4.0, row_step * j / 4.0),
                ))

    best = None
    for raw in candidates:
        ev = _grid_eval(raw)
        if ev is None:
            continue
        if best is None or int(np.sum(ev[4])) > int(np.sum(best[4])):
            best = ev

    # Local stroke width below which the centerline row owns the zone: grid
    # and nudged holes there sit off-center and read as jitter.
    narrow_limit = 2.0 * clearance + 2.0 * RING_EXCLUSION_FACTOR * pitch

    def _in_narrow_zone(i, pts, inside, min_d, qx, qy):
        if min_d[i] >= narrow_limit / 2.0:
            return False
        vx, vy = pts[i, 0] - qx[i], pts[i, 1] - qy[i]
        norm = math.hypot(vx, vy)
        if norm < 1e-12:
            return False
        if not inside[i]:
            vx, vy = -vx, -vy
        w = _chord_width(float(qx[i]), float(qy[i]), vx / norm, vy / norm, segs)
        return w is not None and w < narrow_limit

    if best is not None:
        raw, pts, inside, min_d, fits, near_ring = best
        qx = qy = None
        if narrow_fill:
            inside, min_d, qx, qy = _classify_centers(pts, segs)
            for i in np.nonzero(fits)[0]:
                if _in_narrow_zone(i, pts, inside, min_d, qx, qy):
                    fits[i] = False
        holes = list(contour) + [(float(x), float(y)) for (x, y), ok in zip(raw, fits) if ok]
    else:
        holes = list(contour)

    nudged = 0
    if best is not None and nudge_max > 0.0:
        signed = np.where(inside, min_d, -min_d)
        needed = clearance - signed
        eligible = (~fits) & (~near_ring) & (needed <= nudge_max + 1e-9)
        min_cc = hole_dia * 1.05  # never let a nudged hole overlap a neighbor
        for i in np.nonzero(eligible)[0]:
            if narrow_fill and _in_narrow_zone(i, pts, inside, min_d, qx, qy):
                continue                      # centerline row owns this zone
            moved = _try_nudge(pts[i, 0], pts[i, 1], segs, clearance, nudge_max)
            if moved is None:
                dropped += 1
                continue
            mx, my = moved
            if any((mx - hx) ** 2 + (my - hy) ** 2 < min_cc * min_cc for hx, hy in holes):
                dropped += 1
                continue
            if ring_pts is not None and len(ring_pts) and np.min(
                (ring_pts[:, 0] - mx) ** 2 + (ring_pts[:, 1] - my) ** 2
            ) < (ring_excl * ring_excl):
                dropped += 1
                continue
            holes.append((mx, my))
            nudged += 1
        dropped += int(np.sum((~fits) & (~near_ring) & (needed > nudge_max + 1e-9) & (signed > -clearance)))

    midline = []
    if narrow_fill:
        midline, m_dropped = _midline_rescue(
            shape, segs, contour, holes[len(contour):], hole_dia, pitch, margin
        )
        dropped += m_dropped
        holes.extend(midline)

    if len(holes) > MAX_HOLES:
        raise ValueError(
            f"Pattern produced {len(holes):,} holes (limit {MAX_HOLES:,}). "
            "Increase the pitch or shrink the shape."
        )
    return InfillResult(holes, nudged, dropped, contour=len(contour), midline=len(midline))


# ---------------------------------------------------------------------------
# Output document
# ---------------------------------------------------------------------------
def build_infill_document(shape: ShapeGeometry, holes, hole_dia: float):
    """ezdxf doc matching panel conventions: outline on `cut`, holes on `holes`."""
    doc = ezdxf.new(dxfversion="R2010")
    doc.units = 1  # inches
    msp = doc.modelspace()
    for name, color in [("cut", 1), ("holes", 2)]:
        if name not in doc.layers:
            doc.layers.add(name=name, color=color)
    for ring in shape.rings:
        msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": "cut"})
    for x, y in holes:
        msp.add_circle((x, y), hole_dia / 2.0, dxfattribs={"layer": "holes"})
    return doc
