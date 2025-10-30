#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crack generator (vector, mm units)

- Generates a naturalistic crack whose skeleton starts at the center of the image area
  and grows in BOTH directions along a random heading.
- Growth stops per arm when either its length quota is reached or the image border is hit.
- The crack width varies along the skeleton using 1D fBm, clamped to [MIN_CRACK_WIDTH_MM, MAX_CRACK_WIDTH_MM].
- The exact same geometry is rendered into:
    1) an image-only SVG (180×180 mm), and
    2) an A4 layout SVG with scale bars and positioning.
"""

import os, shutil, math, random
from secrets import randbits
from typing import Dict, Any, List, Tuple

import numpy as np
import svgwrite

# =======================
# Configuration
# =======================

# Sampling / batch
STEP = 0.1                      # mm, advance per skeleton step
NUM_ITERATIONS = 10
MASTER_SEED = None              # set int for reproducible series; None => random each run

# Crack length (per figure)
MIN_CRACK_LENGTH_MM = 50.0
MAX_CRACK_LENGTH_MM = 300.0

# Width model (fBm parameters)
MIN_CRACK_WIDTH_MM = 0.10
MAX_CRACK_WIDTH_MM = 5.00
BASE_WIDTH_MM      = 0.50
WIDTH_FBM_AMP_MM   = 1.00
FBM_OCTAVES        = 5
FBM_LACUNARITY     = 2.0
FBM_H              = 0.8

# Heading model (kinked random walk)
P_KINK          = 0.05          # probability of a discrete “kink” per step
THETA_KINK_DEG  = 12.0          # mean kink magnitude (deg)
KINK_TAIL       = 0.5           # Laplace tail for kink magnitudes
SIGMA_SMALL_DEG = 0.8           # small Gaussian jitter (deg)
MAX_TURN_DEG    = 15.0          # hard clamp per-step turn (deg)

# Visual toggles
SHOW_SKELETON_SEGMENTS = False
SHOW_SKELETON_MARKERS  = False
SHOW_CRACK_OUTLINE     = False
SHOW_CRACK_FILL        = True

# Page & layout (mm)
A4_W = 210.0
A4_H = 297.0
MARGIN = 10.0

IMG_SIZE = 180.0                # image area is 180×180 mm in its own SVG
IMG_X = MARGIN + 5.0            # image placement in A4 layout
IMG_Y = A4_H - MARGIN - 180.0
IMG_W = IMG_SIZE
IMG_H = IMG_SIZE

# Origin placement as a fraction of the box (0..1)
ORIGIN_MIN_FRAC = 0.20
ORIGIN_MAX_FRAC = 0.80


# Scale bars (checker pattern)
SCALEBAR_SEG     = 10.0
SCALEBAR_W       = 2.0
SCALEBAR_COLOR_A = "#000000"    # black
SCALEBAR_COLOR_B = "#ffff00"    # yellow

# Style
SKELETON_COLOR      = "#ff0000"
SKELETON_W_MM       = 0.25
MARKER_R_MM         = 0.25
CRACK_FILL_COLOR    = "#000000"
CRACK_FILL_OPACITY  = 1.0
CRACK_STROKE_COLOR  = "#000000"
CRACK_STROKE_W_MM   = 0.3

FRAME_STROKE   = "#000000"
DATABOX_STROKE = "#333333"

# Numerics
EPS = 1e-12


# =======================
# Geometry helpers
# =======================

def fbm_1d(n: int, *, octaves=5, lacunarity=2.0, H=0.8, seed=0) -> np.ndarray:
    """1D fBm-like signal over n samples in [0,1], zero-mean and unit-std."""
    import random as _random
    rng = _random.Random(int(seed) & 0xFFFFFFFF)
    s = np.linspace(0.0, 1.0, n)
    y = np.zeros(n, float)
    freq, amp = 1.0, 1.0
    for _ in range(octaves):
        phase = 2*math.pi * rng.random()
        y += amp * np.cos(2*math.pi*freq*s + phase)
        freq *= lacunarity
        amp  *= lacunarity ** (-H)
    y -= y.mean()
    std = y.std()
    if std > 0:
        y /= std
    return y


def path_length(points: List[Tuple[float, float]]) -> float:
    """Polyline length in mm."""
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(points[i][0] - points[i-1][0],
                          points[i][1] - points[i-1][1]) for i in range(1, len(points)))


def offset_polyline(points: List[Tuple[float, float]], offset: float) -> List[Tuple[float, float]]:
    """
    Simple per-vertex offset using local averaged tangent.
    Works well for thin bands. For robust thick buffering, use a geometry kernel.
    """
    if len(points) < 2:
        return []
    out = []
    for i, (x, y) in enumerate(points):
        if i == 0:
            x2, y2 = points[i+1]
            dx, dy = x2 - x, y2 - y
        elif i == len(points) - 1:
            x1, y1 = points[i-1]
            dx, dy = x - x1, y - y1
        else:
            x1, y1 = points[i-1]
            x2, y2 = points[i+1]
            dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        nx, ny = (-dy / L, dx / L) if L > EPS else (0.0, 0.0)  # left normal
        out.append((x + offset * nx, y + offset * ny))
    return out


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    """Signed polygon area (mm^2). Positive for CCW winding."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


# =======================
# Scale bar factories
# =======================

def make_checker_bar_h(dwg, *, length=IMG_SIZE, seg=SCALEBAR_SEG, width=SCALEBAR_W,
                       colors=(SCALEBAR_COLOR_A, SCALEBAR_COLOR_B)):
    """Horizontal checker bar (2 rows: YB / BY / ...)."""
    nseg = int(round(length / seg))
    g = dwg.g(id="checker_bar_h")
    half_h = width / 2.0
    for i in range(nseg):
        x0 = i * seg
        top_color = colors[i % 2]
        bot_color = colors[(i + 1) % 2]
        g.add(dwg.rect(insert=(x0, 0),          size=(seg,  half_h), fill=top_color, stroke="none"))
        g.add(dwg.rect(insert=(x0, half_h),     size=(seg,  half_h), fill=bot_color, stroke="none"))
    return g


def make_checker_bar_v(dwg, *, length=IMG_SIZE, seg=SCALEBAR_SEG, width=SCALEBAR_W,
                       colors=(SCALEBAR_COLOR_A, SCALEBAR_COLOR_B)):
    """Vertical checker bar (2 columns: YB / BY / ...)."""
    nseg = int(round(length / seg))
    g = dwg.g(id="checker_bar_v")
    half_w = width / 2.0
    for i in range(nseg):
        y0 = i * seg
        left_color  = colors[i % 2]
        right_color = colors[(i + 1) % 2]
        g.add(dwg.rect(insert=(0,      y0), size=(half_w, seg), fill=left_color,  stroke="none"))
        g.add(dwg.rect(insert=(half_w, y0), size=(half_w, seg), fill=right_color, stroke="none"))
    return g


# =======================
# Renderer (same group is reused in both outputs)
# =======================

def render_crack_group(dwg: svgwrite.Drawing, geom: Dict[str, Any]) -> svgwrite.container.Group:
    """Create an SVG group for the crack (polygon + optional skeleton)."""
    g = dwg.g(id="crack_content")

    # Polygon body
    if geom["poly"] and (SHOW_CRACK_FILL or SHOW_CRACK_OUTLINE):
        fill   = CRACK_FILL_COLOR if SHOW_CRACK_FILL else "none"
        stroke = CRACK_STROKE_COLOR if SHOW_CRACK_OUTLINE else "none"
        g.add(dwg.polygon(points=geom["poly"],
                          fill=fill, fill_opacity=CRACK_FILL_OPACITY,
                          stroke=stroke, stroke_width=CRACK_STROKE_W_MM))

    # Skeleton (may include breaks; here it's a single subpath)
    if SHOW_SKELETON_SEGMENTS and geom.get("skeleton_subpaths"):
        for sub in geom["skeleton_subpaths"]:
            if len(sub) < 2:
                continue
            x0, y0 = sub[0]
            path = dwg.path(d=f"M {x0} {y0}", stroke=SKELETON_COLOR, fill="none",
                            stroke_width=SKELETON_W_MM,
                            **{"stroke-linecap": "round", "stroke-linejoin": "round"})
            for x, y in sub[1:]:
                path.push(f"L {x} {y}")
            g.add(path)

    # Markers at skeleton samples (if enabled)
    if SHOW_SKELETON_MARKERS and geom.get("skeleton_subpaths"):
        for sub in geom["skeleton_subpaths"]:
            for x, y in sub:
                g.add(dwg.circle(center=(x, y), r=MARKER_R_MM,
                                 fill="none", stroke=SKELETON_COLOR, stroke_width=0.2))
    return g


# =======================
# A) Image-only SVG (180×180 mm)
# =======================

def build_image_only_svg(index: int, out_dir_img: str, geom: Dict[str, Any]):
    """Write the 180×180 mm crack image (no page layout)."""
    fn = os.path.join(out_dir_img, f"img_180x180_{index:03d}.svg")
    dwg = svgwrite.Drawing(fn, size=(f"{IMG_SIZE}mm", f"{IMG_SIZE}mm"),
                           viewBox=f"0 0 {IMG_SIZE} {IMG_SIZE}")
    dwg.add(render_crack_group(dwg, geom))
    dwg.save()
    print(f"[{index}] Wrote image-only SVG: {fn}")


# =======================
# B) A4 page with layout & scale bars; embeds the same crack group
# =======================

def build_a4_layout_svg(index: int, out_dir_a4: str, geom: Dict[str, Any]):
    """Write an A4 page that positions the same crack into a framed layout."""
    fn = os.path.join(out_dir_a4, f"crack_sheet_{index:03d}.svg")
    dwg = svgwrite.Drawing(fn, size=(f"{A4_W}mm", f"{A4_H}mm"),
                           viewBox=f"0 0 {A4_W} {A4_H}")

    # Data box (top area placeholder)
    dwg.add(dwg.rect(insert=(MARGIN + 0.1, MARGIN + 0.1),
                     size=(190.0 - 0.2, 82.0 - 0.2),
                     fill="none", stroke=DATABOX_STROKE, stroke_width=0.2))

    # Corner squares of the scale frame
    dwg.add(dwg.rect(insert=(MARGIN, A4_H - MARGIN - 190),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_A, stroke="none"))
    dwg.add(dwg.rect(insert=(MARGIN, A4_H - MARGIN - SCALEBAR_W),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_A, stroke="none"))
    dwg.add(dwg.rect(insert=(A4_W - MARGIN - SCALEBAR_W, A4_H - MARGIN - 190),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_A, stroke="none"))
    dwg.add(dwg.rect(insert=(MARGIN + 190 - SCALEBAR_W + 0.1,
                             A4_H - MARGIN - SCALEBAR_W + 0.1),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_B, stroke=SCALEBAR_COLOR_A, stroke_width=0.2))

    # Checker bars (defs + uses)
    bar_h = make_checker_bar_h(dwg)
    bar_v = make_checker_bar_v(dwg)
    dwg.defs.add(bar_h); dwg.defs.add(bar_v)

    dwg.add(dwg.use(bar_h, insert=(MARGIN + 5,               A4_H - MARGIN - 190)))          # top
    dwg.add(dwg.use(bar_h, insert=(MARGIN + 5,               A4_H - MARGIN - SCALEBAR_W)))   # bottom
    dwg.add(dwg.use(bar_v, insert=(MARGIN,                   A4_H - MARGIN - 185)))          # left
    dwg.add(dwg.use(bar_v, insert=(A4_W - MARGIN - SCALEBAR_W, A4_H - MARGIN - 185)))        # right

    # Place the SAME crack group inside the image area
    g_crack = render_crack_group(dwg, geom)  # made in image coords (0..180)
    g_crack.translate(IMG_X, IMG_Y)
    dwg.add(g_crack)

    dwg.save()
    print(f"[{index}] Wrote A4 SVG: {fn}")


# =======================
# Crack synthesis (center-origin, bidirectional, kinks + fBm width)
# =======================

def generate_crack_from_center_kink_fbm(seed: int, box_size: float) -> Dict[str, Any]:
    """
    Build a skeleton from the image center, growing in two opposite directions.
    Heading evolves via a kinked random walk; width varies by fBm.
    Returns a dict with:
      - skeleton_subpaths: [polyline]
      - poly: filled polygon vertices
      - metrics: length_mm, area_mm2, mean_width_mm
    """
    rng = np.random.default_rng(seed)

    # Total length (quantized to STEP), split equally per arm
    L_total = random.randrange(int(MIN_CRACK_LENGTH_MM / STEP),
                               int(MAX_CRACK_LENGTH_MM / STEP) + 1) * STEP
    L_arm = 0.5 * L_total

    # Center origin
    cx = float(rng.uniform(ORIGIN_MIN_FRAC, ORIGIN_MAX_FRAC) * box_size)
    cy = float(rng.uniform(ORIGIN_MIN_FRAC, ORIGIN_MAX_FRAC) * box_size)

    # Forward and backward initial headings
    theta0 = rng.uniform(0.0, 2 * math.pi)

    def in_bounds(x: float, y: float) -> bool:
        return 0.0 <= x <= box_size and 0.0 <= y <= box_size

    def step_heading(theta: float) -> float:
        """Per-step heading update: sparse kinks + small Gaussian jitter, clamped."""
        if rng.random() < P_KINK:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            dtheta = sign * (math.radians(THETA_KINK_DEG) *
                             np.random.laplace(loc=1.0, scale=KINK_TAIL))
        else:
            dtheta = math.radians(SIGMA_SMALL_DEG) * rng.normal()
        dtheta = max(-math.radians(MAX_TURN_DEG), min(math.radians(MAX_TURN_DEG), dtheta))
        return theta + dtheta

    def grow_arm(x0: float, y0: float, theta_init: float, target_len: float) -> List[Tuple[float, float]]:
        """Advance by STEP while inside the box and below target_len."""
        pts = [(x0, y0)]
        theta = theta_init
        acc = 0.0
        while acc + STEP <= target_len:
            theta = step_heading(theta)
            nx = pts[-1][0] + STEP * math.cos(theta)
            ny = pts[-1][1] + STEP * math.sin(theta)
            if not in_bounds(nx, ny):
                break
            pts.append((nx, ny))
            acc += STEP
        return pts

    # Grow both arms; stitch while avoiding duplicate center point
    arm_f = grow_arm(cx, cy, theta0,           L_arm)
    arm_b = grow_arm(cx, cy, theta0 + math.pi, L_arm)
    skel = (list(reversed(arm_b))[:-1] if len(arm_b) > 0 else []) + arm_f
    if len(skel) < 2:  # pathological edge case (tiny stub)
        skel = [(cx, cy), (cx + STEP, cy)]

    # Width via fBm, clamped
    n_pts = len(skel)
    fbm = fbm_1d(n_pts, octaves=FBM_OCTAVES, lacunarity=FBM_LACUNARITY, H=FBM_H,
                 seed=(int(seed) ^ 0x5A5A5A5A))
    width = np.clip(BASE_WIDTH_MM + WIDTH_FBM_AMP_MM * fbm,
                    MIN_CRACK_WIDTH_MM, MAX_CRACK_WIDTH_MM)
    halfw = 0.5 * width

    # Offset polygon (variable width)
    left, right = [], []
    for i, (x, y) in enumerate(skel):
        if i == 0:
            x2, y2 = skel[i + 1]; dx, dy = x2 - x, y2 - y
        elif i == n_pts - 1:
            x1, y1 = skel[i - 1]; dx, dy = x - x1, y - y1
        else:
            x1, y1 = skel[i - 1]; x2, y2 = skel[i + 1]; dx, dy = x2 - x1, y2 - y1
        Ld = math.hypot(dx, dy)
        nx, ny = (-dy / Ld, dx / Ld) if Ld > EPS else (0.0, 0.0)  # left normal
        hw = float(halfw[i])
        left.append((x + hw * nx, y + hw * ny))
        right.append((x - hw * nx, y - hw * ny))

    poly: List[Tuple[float, float]] = []
    if len(left) >= 2 and len(right) >= 2:
        poly = left + list(reversed(right))
        if polygon_area(poly) < 0:
            poly.reverse()

    # Metrics (already within the image box; no clipping needed)
    length_mm = path_length(skel)
    area_mm2 = abs(polygon_area(poly)) if len(poly) >= 3 else 0.0
    mean_w = (area_mm2 / length_mm) if length_mm > 0 else 0.0

    return {
        "skeleton_subpaths": [skel],
        "poly": poly,
        "metrics": {
            "length_mm": length_mm,
            "area_mm2": area_mm2,
            "mean_width_mm": mean_w,
        },
    }


# =======================
# Driver
# =======================

def main():
    # Fresh output roots
    root = "out"
    out_img = os.path.join(root, "img_180x180_mm")
    out_a4  = os.path.join(root, "A4_print")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_a4,  exist_ok=True)

    for idx in range(1, NUM_ITERATIONS + 1):
        # seed selection
        used_seed = (MASTER_SEED + idx) if isinstance(MASTER_SEED, int) else randbits(64)
        # geometry once
        geom = generate_crack_from_center_kink_fbm(seed=used_seed, box_size=IMG_SIZE)
        # render same geometry twice
        build_image_only_svg(idx, out_img, geom)
        build_a4_layout_svg(idx, out_a4, geom)


if __name__ == "__main__":
    main()
