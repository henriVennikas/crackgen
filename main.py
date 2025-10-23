import math, random, json, os, shutil
import numpy as np
from secrets import randbits

# Main configuration parameters for crack rendering
STEP = 0.1                      # mm, step size along skeleton
NUM_ITERATIONS = 1              # how many different renderings to produce
MASTER_SEED = None              # random seed for reproducibility

# Toggles for visual elements
SHOW_SKELETON_SEGMENTS = True   # centerline polyline
SHOW_SKELETON_MARKERS  = False  # red dots at skeleton samples
SHOW_WHISKER_SEGMENTS  = False  # perpendicular half-whiskers
SHOW_WHISKER_MARKERS   = False  # dots at whisker tips
SHOW_CRACK_OUTLINE     = False  # polygon stroke
SHOW_CRACK_FILL        = True   # polygon fill

# Crack dimension limits
MAX_CRACK_LENGTH_MM = 300.0  # mm, maximum crack length to consider
MIN_CRACK_LENGTH_MM = 50.0   # mm, minimum crack length to consider
MIN_CRACK_WIDTH_MM = 0.10
MAX_CRACK_WIDTH_MM = 2.00
BASE_WIDTH_MM      = 0.60
WIDTH_FBM_AMP_MM   = 0.30

# Kinked, persistent random walk parameters
P_KINK           = 0.07         # Proability of a kink at each step
THETA_KINK_DEG   = 25.0         # Mean kink angle in degrees
KINK_TAIL        = 0.6          # Temporal correlation of kink angles
SIGMA_SMALL_DEG  = 2.0          # Standard deviation of small-angle noise in degrees
THETA_PREF_DEG   = 90.0         # Preferred turning angle in degrees
DRIFT_GAIN       = 0.04         # Strength of drift toward preferred angle
MAX_TURN_DEG     = 50.0         # Maximum turn angle per step in degrees

# fBm width parameters
FBM_OCTAVES    = 5              # Number of fBm octaves
FBM_LACUNARITY = 2.0            # fBm lacunarity
FBM_H          = 0.8            # fBm Hurst exponent

# Page layout parameters
A4_W = 210.0                    # mm
A4_H = 297.0                    # mm
MARGIN = 10.0                   # mm
IMG_SIZE = 180.0                # mm
BAR_SIZE = 5.0                  # mm, measurement scale bar thickness
BUFFER = 1.0                    # mm, gap between image and scale bar
DATA_BOX_X = MARGIN             # mm
DATA_BOX_Y = MARGIN             # mm
DATA_BOX_WIDTH = 190            # mm
DATA_BOX_HEIGHT = 82            # mm, space for image and gap

# Color and style parameters
SKELETON_COLOR      = "#ff0000"
SKELETON_W_MM       = max(0.1, STEP/20)
MARKER_R_MM         = max(0.1, STEP/20)
MARKER_STROKE       = "#ff0000"
MARKER_FILL         = "none"
MARKER_STROKE_W     = max(0.1, STEP/20)

WHISKER_COLOR       = "#1a8f1a"
WHISKER_STROKE_W_MM = max(0.1, STEP/20)
WHISKER_END_R_MM    = max(0.1, STEP/20)

CRACK_FILL_COLOR    = "#000000"   # black fill
CRACK_FILL_OPACITY  = 1.0
CRACK_STROKE_COLOR  = "#000000"
CRACK_STROKE_W_MM   = 0.3

SCALEBAR_COLOR_A      = "#000000"
SCALEBAR_COLOR_B      = "#ffff00"
SCALEBAR_SEG         = 10.0       # mm, length of each segment
SCALEBAR_W           = 2.0        # mm, width of scale bar
FRAME_STROKE        = "#000000"
DATABOX_STROKE      = "#333333"

# A small epsilon value for numerical stability
EPS = 1e-12


def crackgen(index: int, seed_override: int | None = None):
    
    crack_length = random.randrange(int(MIN_CRACK_LENGTH_MM / STEP),
                     int(MAX_CRACK_LENGTH_MM / STEP)+1) * STEP
    n = max(1, int(round(crack_length / STEP))) # number of segments
        
    if seed_override is not None:
        used_seed = int(seed_override)
    else:
        if MASTER_SEED is None:
            used_seed = randbits(64)
        else:
            used_seed = int(MASTER_SEED) + index  # vary by index
    print(f"[{index}] Segments: {n} (~{crack_length:.2f} mm) | seed: {used_seed}")

    # SVG (1 mm units)
    dwg = svgwrite.Drawing(
        f"crack_sheet_{index:03d}.svg",
        size=(f"{A4_W}mm", f"{A4_H}mm"),
        viewBox=f"0 0 {A4_W} {A4_H}"
    )

     # DATA BOX FRAME
    dwg.add(dwg.rect(insert=(MARGIN+0.1, MARGIN+0.1),
                     size=(DATA_BOX_WIDTH-0.2, DATA_BOX_HEIGHT-0.2),
                     fill="none", stroke=FRAME_STROKE, stroke_width=0.2))
    
    # SCALE BAR CORNERS
    # 1 - top-left
    dwg.add(dwg.rect(insert=(MARGIN, A4_H-MARGIN-190),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_A, stroke="none"))
    # 2 - bottom-left
    dwg.add(dwg.rect(insert=(MARGIN, A4_H - MARGIN - SCALEBAR_W),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_A, stroke="none"))
    # 3 - top-right
    dwg.add(dwg.rect(insert=(A4_W - MARGIN - SCALEBAR_W, A4_H - MARGIN - 190),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_A, stroke="none"))
    # 4 - bottom-right
    dwg.add(dwg.rect(insert=(MARGIN + 190 - SCALEBAR_W + 0.1 , A4_H - MARGIN - SCALEBAR_W + 0.1),
                     size=(SCALEBAR_W, SCALEBAR_W),
                     fill=SCALEBAR_COLOR_B, stroke=SCALEBAR_COLOR_A, stroke_width=0.2))
    
    def make_checker_bar_h(
        dwg, *, length=IMG_SIZE, seg=SCALEBAR_SEG, width=SCALEBAR_W,
        colors=(SCALEBAR_COLOR_A, SCALEBAR_COLOR_B)  # e.g. ("#FFD400", "#000000")
    ):
        nseg = int(round(length / seg))
        g = dwg.g(id="checker_bar_h")
        half_h = width / 2.0  # top/bottom halves
        for i in range(nseg):
            x0 = i * seg
            top_color = colors[i % 2]          # Y, B, Y, B...
            bot_color = colors[(i + 1) % 2]    # B, Y, B, Y...
            g.add(dwg.rect(insert=(x0, 0),          size=(seg,  half_h), fill=top_color, stroke="none"))
            g.add(dwg.rect(insert=(x0, half_h),     size=(seg,  half_h), fill=bot_color, stroke="none"))
        return g

    def make_checker_bar_v(
        dwg, *, length=IMG_SIZE, seg=SCALEBAR_SEG, width=SCALEBAR_W,
        colors=(SCALEBAR_COLOR_A, SCALEBAR_COLOR_B)  # e.g. ("#FFD400", "#000000")
    ):
        nseg = int(round(length / seg))
        g = dwg.g(id="checker_bar_v")
        half_w = width / 2.0  # left/right halves
        for i in range(nseg):
            y0 = i * seg
            left_color  = colors[i % 2]         # Y, B, Y, B...
            right_color = colors[(i + 1) % 2]   # B, Y, B, Y...
            g.add(dwg.rect(insert=(0,      y0), size=(half_w, seg), fill=left_color,  stroke="none"))
            g.add(dwg.rect(insert=(half_w, y0), size=(half_w, seg), fill=right_color, stroke="none"))
        return g


    # build once and reuse
    bar_h = make_checker_bar_h(dwg)  # 180×SCALEBAR_W horizontal
    bar_v = make_checker_bar_v(dwg)  # SCALEBAR_W×180 vertical
    dwg.defs.add(bar_h)
    dwg.defs.add(bar_v)

    # place around image area
    dwg.add(dwg.use(bar_h, insert=(MARGIN + 5,          A4_H - MARGIN - 190)))  # top
    dwg.add(dwg.use(bar_h, insert=(MARGIN + 5,          A4_H - MARGIN - SCALEBAR_W)))    # bottom
    dwg.add(dwg.use(bar_v, insert=(MARGIN,              A4_H - MARGIN - 185)))  # left
    dwg.add(dwg.use(bar_v, insert=(A4_W - MARGIN - SCALEBAR_W,   A4_H - MARGIN - 185)))  # right

    # Save SVG
    svg_path = f"crack_sheet_{index:03d}.svg"
    dwg.saveas(svg_path)
    print(f"[{index}] Wrote {svg_path}")


def main():

     # === setup output directory ===
    outdir = "out"
    if os.path.exists(outdir):
        print(f"Clearing previous contents in '{outdir}' ...")
        shutil.rmtree(outdir)  # removes folder and all contents
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    print(f"Output directory ready: {os.getcwd()}")


    # batch generate
    for idx in range(1, NUM_ITERATIONS + 1):
        crackgen(idx, seed_override=None)

if __name__ == "__main__":
    main()