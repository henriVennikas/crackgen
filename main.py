import math, random, json, os, shutil
import numpy as np
import svgwrite
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
A4_WIDTH = 210.0                # mm
A4_HEIGHT = 297.0               # mm
MARGIN = 10.0                   # mm
IMG_SIZE = 180.0                # mm
BAR_SIZE = 5.0                  # mm, measurement scale bar thickness
DATA_BOX_X = MARGIN             # mm
DATA_BOX_Y = MARGIN             # mm
DATA_BOX_WIDTH = A4_WIDTH - 2 * MARGIN # mm
DATA_BOX_HEIGHT = A4_HEIGHT - 2 * MARGIN - IMG_SIZE - BAR_SIZE # mm, space for image and gap

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
SCALEBAR_COLOR_B      = "#ffffff"
FRAME_STROKE        = "#ffff00"
DATABOX_STROKE      = "#333333"

# A small epsilon value for numerical stability
EPS = 1e-12


def crackgen(index: int, seed_override: int | None = None):
    
    crack_length = random.uniform(MIN_CRACK_LENGTH_MM, MAX_CRACK_LENGTH_MM)
    n = max(1, int(round(L / STEP))) # number of segments
        
    if seed_override is not None:
        used_seed = int(seed_override)
    else:
        if MASTER_SEED is None:
            used_seed = randbits(64)
        else:
            used_seed = int(MASTER_SEED) + index  # vary by index
    print(f"[{index}] Segments: {n} (~{L_nominal:.1f} mm) | seed: {used_seed}")


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
    for idx in range(1, NUM_SHEETS + 1):
        crack_sheet_once(idx, seed_override=None)

if __name__ == "__main__":
    main()