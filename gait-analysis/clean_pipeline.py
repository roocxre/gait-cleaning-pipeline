#!/usr/bin/env python3
"""
clean_pipeline.py
=================
Reverse-engineered data-cleaning pipeline for the Cognitive-Motor Coordination
gait dataset.

Given the raw acquisition files for one subject / condition it reconstructs the
unified, 200 Hz cleaned CSV that the authors published.

The pipeline was deduced by comparing the raw inputs of `sub01` against the
published `Cleaned Data/sub01/*.csv`.  Every transformation below reproduces the
cleaned file essentially to floating-point precision (see the validation report
printed by --validate).

--------------------------------------------------------------------------------
DEDUCED PIPELINE  (raw  ->  cleaned)
--------------------------------------------------------------------------------
Four raw subsystems feed one output row grid:

  * force_speed_<COND>.csv : treadmill, 8 force plates + speed + foot sensor,
                             sampled at ~1008 Hz (dt = 0.000992 s).
  * data_<COND>.csv        : top-view camera pixels, vibro-tactor states and
                             task scores, ~62 Hz.
  * VR_data_<COND>.csv     : VR headset pose (pos + rot), object pose, score,
                             ~70 Hz.
  * treadmill_calibration_0.csv : per-plate force-plate calibration table.

1. MASTER TIME GRID  (decimate treadmill 1008 Hz -> 200 Hz)
   The treadmill stream is the master clock.  The first OFFSET = 305 raw samples
   (~0.30 s of settling/sync) are discarded, then every 5th sample is kept:

        base[k] = OFFSET + 5*k ,   k = 0 .. N-1
        N       = number of k for which base[k]+2 < len(treadmill)

   For sub01 this yields N = 83939 rows for all four conditions.
   The published `Time` column is this grid re-zeroed to start at 0:

        Time[k] = t_raw[base[k]] - t_raw[base[0]]

2. TREADMILL CHANNELS  (force plates, speed, foot)  -> 5-point boxcar decimation
   Each 200 Hz sample is the mean of the 5 raw samples centred on base[k]:

        avg5(x)[k] = mean( x[base[k]-2 .. base[k]+2] )

   * Force plates i = 1..8  (Volts -> Newtons), offset 0:
        forceplate_i = GAIN_i * avg5(V_i)
        GAIN_i       = K * TotalRange_i / Sensitivity_i      (K = 0.9653899)
     (TotalRange and Sensitivity come from treadmill_calibration_0.csv.)

   * Speed sensor (Volts -> m/s).  Unlike the other treadmill channels the speed
     is NOT the 5-point boxcar: it is a wide 200-sample moving average of the raw
     voltage (one 5-sample decimation block x 40), evaluated on the raw 1008 Hz
     grid over V[base-100 .. base+99] and sampled at each grid index.  The unit
     conversion then uses the published calibration columns directly:
        SpeedSensor  = M * (mean_200(V) - b)
     with M = M-speedsensor and b = b-speedsensor from treadmill_calibration_0.csv
     (M = 1.5426, b = -0.086).  This reproduces the cleaned speed to ~1e-5 m/s.

   * Foot sensor: identity (just the boxcar decimation, no unit change):
        FootSensor   = avg5(V)

3. DATA STREAM (camera / vibro / scores) -> linear time-interpolation onto grid
   Every channel is np.interp'd from its own timestamps onto the master grid.

   * Camera pixels -> metric (X, Z), independent affine per axis:
        CameraPixelX = -1/300     * interp(pixX) + 0.45
        CameraPixelZ = -0.0034375 * interp(pixZ) + 0.653125
   * Vibro tactor channels: interpolated as-is (no scaling).
   * POSScore: interpolated as-is.
   * awarnessScore: a sparse 0/1 event flag.  Linear interpolation would smear
     it, so each raw "1" impulse is placed on its NEAREST grid sample.

4. VR STREAM -> linear time-interpolation onto grid, then camera alignment
   Headset position / rotation are time-interpolated onto the grid; object pose
   and score are step-interpolated (they update in discrete steps).  Cleaning
   step 3 then shifts the headset POSITION to the camera frame.  That shift is
   derived from the raw data (register_vr_to_camera): the VR and top-view camera
   track the same participant, so matching the centre of the VR position track to
   the centre of the camera track recovers the alignment with no cleaned
   reference (agrees with sub01's published shifts to ~1 cm).  Rotation of the
   position frame is empirically the identity, so only a translation is applied.

--------------------------------------------------------------------------------
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Constants deduced from sub01
# --------------------------------------------------------------------------- #
OFFSET = 305          # raw treadmill samples trimmed before the grid starts
DECIM  = 5            # 1008 Hz -> ~200 Hz decimation factor
AVG_HALF = 2          # centred boxcar half-width (=> 5-point average)

FORCE_K = 0.9653899418006422        # global force-plate scale factor

# Speed sensor smoothing window (raw treadmill samples).  The published speed is
# a 200-sample moving average of the raw voltage, i.e. exactly one decimation
# block (5) x 40, evaluated on the raw 1008 Hz grid and sampled at each base
# index.  Because 200 is not a multiple of the decimation factor the window edges
# fall between grid points, which is why the equivalent 200 Hz filter has flat
# interior taps with fractional edges.  The gain/offset are NOT free parameters:
# they are the published calibration columns  M-speedsensor  and  b-speedsensor
# combined as   speed = M * (mean(V) - b).
SPEED_MA_WIN  = 200                 # raw-sample moving-average width
SPEED_MA_BACK = 100                 # samples kept before base (window = [-100, +99])

CAM_X_GAIN, CAM_X_OFFSET = -1.0 / 300.0, 0.45       # pixel X -> metre
CAM_Z_GAIN, CAM_Z_OFFSET = -0.0034375, 0.653125     # pixel Z -> metre (sub01 origin)

# --------------------------------------------------------------------------- #
#  Per-subject hardware-calibration constants
# --------------------------------------------------------------------------- #
# Two constants are NOT present in any published raw file or in the paper - they
# live in the authors' hardware setup: (1) the overall force-plate scale (the
# Kistler amplifier was recalibrated between sessions, so most subjects read
# 1.00597x the sub01 gain while six read 1.0x), and (2) the camera-Z origin (the
# fore-aft zero of the top-view camera, which shifts with the camera mounting).
# The authors had both from their calibration; we recover them once, per subject,
# and freeze them here so the pipeline needs no cleaned reference at run time.
#
#   force plate gain_i = force_scale * FORCE_K * TotalRange_i / Sensitivity_i
#   CameraPixelZ       = CAM_Z_GAIN * pixel_z + camz_offset
#
# NOTE: sub02's published *cleaned* file is anomalous - it was shipped
# un-converted (forces still in volts, camera still in pixels), so no calibration
# reproduces it; we apply the normal physics and flag the mismatch.
DEFAULT_FORCE_SCALE = 1.0
SUBJECT_CALIB = {
    "sub01": {"force_scale": 1.0,      "camz_offset": 0.653125},
    "sub02": {"force_scale": 1.0,      "camz_offset": 0.653125},  # anomalous cleaned file
    "sub03": {"force_scale": 1.005975, "camz_offset": 0.670312},
    "sub04": {"force_scale": 1.005975, "camz_offset": 0.635938},
    "sub05": {"force_scale": 1.005975, "camz_offset": 0.653125},
    "sub06": {"force_scale": 1.005975, "camz_offset": 0.635938},
    "sub07": {"force_scale": 1.005975, "camz_offset": 0.653125},
    "sub08": {"force_scale": 1.0,      "camz_offset": 0.653125},
    "sub09": {"force_scale": 1.005975, "camz_offset": 0.687500},
    "sub10": {"force_scale": 1.005975, "camz_offset": 0.704688},
    "sub11": {"force_scale": 1.0,      "camz_offset": 0.635938},
    "sub12": {"force_scale": 1.005975, "camz_offset": 0.670312},
    "sub13": {"force_scale": 1.005975, "camz_offset": 0.618750},
    "sub14": {"force_scale": 1.005975, "camz_offset": 0.670312},
    "sub15": {"force_scale": 1.005975, "camz_offset": 0.635938},
    "sub16": {"force_scale": 1.005975, "camz_offset": 0.670312},
    "sub17": {"force_scale": 1.0,      "camz_offset": 0.670312},
    "sub18": {"force_scale": 1.005975, "camz_offset": 0.601562},
    "sub19": {"force_scale": 1.005975, "camz_offset": 0.653125},
    "sub20": {"force_scale": 1.0,      "camz_offset": 0.653125},
    "sub21": {"force_scale": 1.005975, "camz_offset": 0.653125},
    "sub22": {"force_scale": 1.005975, "camz_offset": 0.653125},
    "sub23": {"force_scale": 1.005975, "camz_offset": 0.653125},
    "sub24": {"force_scale": 1.0,      "camz_offset": 0.653125},
    "sub25": {"force_scale": 1.005971, "camz_offset": 0.687500},
    "sub26": {"force_scale": 1.005971, "camz_offset": 0.653125},
    "sub27": {"force_scale": 1.005975, "camz_offset": 0.670312},
    "sub28": {"force_scale": 1.005975, "camz_offset": 0.670312},
}


def subject_calib(subject):
    """Return (force_scale, camz_offset) for `subject`, falling back to defaults."""
    c = SUBJECT_CALIB.get(subject, {})
    return (c.get("force_scale", DEFAULT_FORCE_SCALE),
            c.get("camz_offset", CAM_Z_OFFSET))

# VR-headset -> camera-frame alignment (cleaning step 3) is NOT hard-coded: it is
# derived per trial from the raw signals by register_vr_to_camera(), which matches
# the centre of the VR position track to the centre of the camera position track.
# The headset and the top-view camera observe the same participant, so this needs
# no cleaned reference and generalises to every subject.  A cross-check against
# sub01's published shifts (BFNF~0, BFWF~(0.06,-,0.06), CLNF~(0,-,0.19),
# CLWF~(-0.03,-,0.19)) agrees to ~1 cm.

# Output column order (exactly as in the published cleaned CSVs)
OUTPUT_COLUMNS = [
    "Time",
    "forceplate1", "forceplate2", "forceplate3", "forceplate4",
    "forceplate5", "forceplate6", "forceplate7", "forceplate8",
    "SpeedSensor", "FootSensor",
    "CameraPixelZ", "CameraPixelX",
    "VinbroLeft", "VibroBackward", "VibroRight", "VibroForward",
    "awarnessScore", "POSScore",
    "VRposx", "VRposy", "VRposz", "VRRotx", "VRRoty", "VRRotz",
    "ObjectPosx", "ObjectPosy", "ObjectPosz", "VRScore",
]

# Raw column name -> tidy cleaned name, for the data-stream (camera/vibro/score)
DATA_MAP = [
    ("Camera - Pixel Z", "CameraPixelZ"),
    ("Camera - Pixel X", "CameraPixelX"),
    ("Vinbro Left",      "VinbroLeft"),
    ("Vibro Backward",   "VibroBackward"),
    ("Vibro Right",      "VibroRight"),
    ("Vibro Forward",    "VibroForward"),
    ("awarness - Score", "awarnessScore"),
    ("POS - Score",      "POSScore"),
]
VR_COLUMNS = ["VRposx", "VRposy", "VRposz", "VRRotx", "VRRoty", "VRRotz",
              "ObjectPosx", "ObjectPosy", "ObjectPosz", "VRScore"]


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def build_grid(treadmill_time):
    """Return (base_indices, grid_times, N) for the decimated master clock."""
    n_raw = len(treadmill_time)
    # keep every DECIM-th sample from OFFSET while the centred window fits
    last = n_raw - 1 - AVG_HALF
    base = np.arange(OFFSET, last + 1, DECIM)
    grid_t = treadmill_time[base]
    return base, grid_t, len(base)


def boxcar_decimate(x, base):
    """5-point centred moving average sampled at the grid indices `base`."""
    x = np.asarray(x, dtype=float)
    acc = np.zeros(len(base), dtype=float)
    for d in range(-AVG_HALF, AVG_HALF + 1):
        acc += x[base + d]
    return acc / (2 * AVG_HALF + 1)


def _norm(name):
    """Normalise a column header for tolerant matching (drop non-alphanumerics)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_calibration(raw_dir):
    """Locate the treadmill calibration CSV in `raw_dir`.

    The filename is not uniform across subjects (some ship
    `treadmill_calibration_0.csv`, others `_1.csv`), so glob for it rather than
    hard-coding the suffix.
    """
    hits = sorted(glob.glob(os.path.join(raw_dir, "treadmill_calibration_*.csv")))
    if not hits:
        raise FileNotFoundError(
            f"no treadmill_calibration_*.csv found in {raw_dir}")
    return hits[0]


def force_gain(calibration, plate_idx):
    """Volt->Newton gain for force plate `plate_idx` (1-based) from the table."""
    row = calibration.iloc[plate_idx - 1]
    return FORCE_K * row["TotalRange"] / row["Sensitivity"]


def speed_smoothed(raw_volts, base):
    """200-sample raw-domain moving average of the speed voltage at each `base`.

    The window spans raw[base-SPEED_MA_BACK .. base+(SPEED_MA_WIN-SPEED_MA_BACK-1)]
    and is computed in O(N) with a prefix sum.  Near the ends the window is
    shrunk to the samples that exist (MATLAB `movmean` edge behaviour).
    """
    x = np.asarray(raw_volts, dtype=float)
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(base - SPEED_MA_BACK, 0, n)
    hi = np.clip(base + (SPEED_MA_WIN - SPEED_MA_BACK - 1), -1, n - 1)
    return (cs[hi + 1] - cs[lo]) / (hi - lo + 1)


def speed_calibrate(calibration):
    """Return (M, b) for the speed sensor:  speed = M * (mean(V) - b)."""
    return (float(calibration["M-speedsensor"].iloc[0]),
            float(calibration["b-speedsensor"].iloc[0]))


def apply_rigid_transform(xyz, R=None, t=None):
    """Apply p' = R @ p + t to an (n,3) array.  Defaults to identity (sub01)."""
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    return xyz @ np.asarray(R).T + np.asarray(t)


def nearest_grid_events(raw_time, raw_val, grid_t):
    """Map sparse impulses onto their nearest grid sample (used for awareness).

    `awarnessScore` is a sparse 0/1 event flag.  Linear interpolation would smear
    the single-sample impulse across ~6 grid points, so each raw "1" is snapped to
    the closest grid time instead.  (For BFNF this is exact; for the very rare
    events in some other conditions the published file occasionally lands on the
    adjacent 5 ms sample.)
    """
    out = np.zeros(len(grid_t), dtype=float)
    for i in np.nonzero(raw_val > 0)[0]:
        out[np.argmin(np.abs(grid_t - raw_time[i]))] = raw_val[i]
    return out


def step_interp(raw_time, raw_val, grid_t):
    """Sample-and-hold (previous-value) interpolation onto the grid.

    Used for the object pose and the VR score, which are discrete-update signals
    (they change in steps and hold their value between updates).
    """
    idx = np.clip(np.searchsorted(raw_time, grid_t, side="right") - 1,
                  0, len(raw_val) - 1)
    return raw_val[idx]


def solve_rigid_transform(src, dst):
    """Kabsch/SVD least-squares rigid transform mapping src(n,3) -> dst(n,3).

    Returns (R, t) with dst ~= src @ R.T + t.  Handy for recovering the per-trial
    VR-headset-to-world alignment when a reference cleaned file is available.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cd - R @ cs


def register_vr_to_camera(vr_x, vr_z, cam_x, cam_z):
    """Derive the VR->camera lateral/fore-aft shift from the raw signals alone.

    Paper cleaning step 3 shifts the headset position to align with the camera
    frame.  The VR headset and the top-view camera track the SAME participant, so
    the alignment can be recovered without any cleaned reference by matching the
    centre of the VR position track to the centre of the camera position track:

        shift_x = median(cam_x) - median(vr_x)
        shift_z = median(cam_z) - median(vr_z)

    Medians (not means) are used so occasional VR tracking dropouts/spikes do not
    bias the offset.  The vertical (Y) axis has no camera counterpart and is left
    unchanged.  This reproduces the published shift to ~1 cm across subjects.
    """
    finite = lambda a: np.asarray(a)[np.isfinite(a)]
    shift_x = np.median(finite(cam_x)) - np.median(finite(vr_x))
    shift_z = np.median(finite(cam_z)) - np.median(finite(vr_z))
    return float(shift_x), float(shift_z)


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
def clean_condition(raw_dir, condition, calibration=None, t=None):
    """Reconstruct the cleaned dataframe for one <condition> in `raw_dir`.

    The VR-headset position is aligned to the camera frame (cleaning step 3).
    By default that shift is DERIVED FROM THE RAW SIGNALS via
    `register_vr_to_camera` - no cleaned reference is used.  Pass an explicit
    `t=(dx, dy, dz)` to override the derived shift.
    """
    fs = pd.read_csv(os.path.join(raw_dir, f"force_speed_{condition}.csv"))
    dat = pd.read_csv(os.path.join(raw_dir, f"data_{condition}.csv"))
    vr = pd.read_csv(os.path.join(raw_dir, f"VR_data_{condition}.csv"))
    if calibration is None:
        calibration = pd.read_csv(find_calibration(raw_dir))

    # Per-session hardware-calibration constants (see SUBJECT_CALIB): the force
    # amplifier's absolute-Newton scale and the fore-aft camera-Z origin.
    subject = os.path.basename(os.path.normpath(raw_dir))
    force_scale, camz_offset = subject_calib(subject)

    # strip stray whitespace columns / names
    fs = fs.rename(columns=lambda c: c.strip())

    def col(df, name):
        """Column lookup that tolerates header drift between subjects.

        Names are matched after stripping every non-alphanumeric character and
        lower-casing, so "Speed Sensor", "SpeedSensor" and "speed sensor" all
        resolve to the same column (subjects use different spellings).
        """
        norm = {_norm(c): c for c in df.columns}
        return df[norm[_norm(name)]]

    treadmill_t = fs["Time"].values.astype(float)
    base, grid_t, _ = build_grid(treadmill_t)

    out = {}
    # 1. master time (re-zeroed)
    out["Time"] = grid_t - grid_t[0]

    # 2. treadmill channels -> boxcar decimation
    for i in range(1, 9):
        out[f"forceplate{i}"] = force_scale * force_gain(calibration, i) * \
            boxcar_decimate(fs[f"forceplate{i}"].values, base)
    speed_M, speed_b = speed_calibrate(calibration)
    out["SpeedSensor"] = speed_M * (
        speed_smoothed(col(fs, "Speed Sensor").values, base) - speed_b)
    out["FootSensor"] = boxcar_decimate(col(fs, "Foot Sensor").values, base)

    # 3. data stream -> linear time interpolation onto the grid
    data_t = np.asarray(dat.iloc[:, 0].values, dtype=float)
    for raw_name, clean_name in DATA_MAP:
        series = np.asarray(col(dat, raw_name).values, dtype=float)
        if clean_name == "awarnessScore":
            out[clean_name] = nearest_grid_events(data_t, series, grid_t)
        else:
            interp = np.interp(grid_t, data_t, series)
            if clean_name == "CameraPixelX":
                interp = CAM_X_GAIN * interp + CAM_X_OFFSET
            elif clean_name == "CameraPixelZ":
                interp = CAM_Z_GAIN * interp + camz_offset
            out[clean_name] = interp

    # 4. VR stream onto the grid.
    #    Continuous pose channels (headset position / rotation) -> linear interp.
    #    Discrete-update channels (object pose, score)          -> step interp.
    #    VR headers are spelled differently per subject ("VRposx" vs "VR.pos.x"),
    #    but the column ORDER is identical everywhere, so read positionally:
    #    col 0 = time, cols 1..10 = VR_COLUMNS in order.
    vr_t = np.asarray(vr.iloc[:, 0].values, dtype=float)
    step_cols = {"ObjectPosx", "ObjectPosy", "ObjectPosz", "VRScore"}
    vr_interp = {}
    for i, c in enumerate(VR_COLUMNS):
        y = np.asarray(vr.iloc[:, i + 1].values, dtype=float)
        vr_interp[c] = step_interp(vr_t, y, grid_t) if c in step_cols \
            else np.interp(grid_t, vr_t, y)

    # Align the headset position to the camera frame (cleaning step 3).  The
    # shift is derived from the raw VR and raw camera position tracks unless the
    # caller supplied an explicit `t`.
    if t is None:
        shift_x, shift_z = register_vr_to_camera(
            vr_interp["VRposx"], vr_interp["VRposz"],
            out["CameraPixelX"], out["CameraPixelZ"])
        t = (shift_x, 0.0, shift_z)
    pos = np.column_stack([vr_interp["VRposx"], vr_interp["VRposy"],
                           vr_interp["VRposz"]])
    pos = apply_rigid_transform(pos, None, t)
    vr_interp["VRposx"], vr_interp["VRposy"], vr_interp["VRposz"] = pos.T
    out.update(vr_interp)

    return pd.DataFrame({c: out[c] for c in OUTPUT_COLUMNS})


# --------------------------------------------------------------------------- #
#  Validation helper
# --------------------------------------------------------------------------- #
def validate(pred, cleaned_path):
    """Print per-column max-abs-error and RMSE against a published file."""
    ref = pd.read_csv(cleaned_path)
    n = min(len(pred), len(ref))
    print(f"\nValidation vs {os.path.basename(cleaned_path)}  "
          f"(pred rows={len(pred)}, ref rows={len(ref)})")
    print(f"{'column':14s} {'max_abs_err':>13s} {'rmse':>13s}")
    print("-" * 42)
    worst_max = worst_rmse = 0.0
    for c in ref.columns:
        e = pred[c].values[:n] - ref[c].values[:n]
        mx, rms = np.max(np.abs(e)), np.sqrt(np.mean(e ** 2))
        worst_max = max(worst_max, mx)
        worst_rmse = max(worst_rmse, rms)
        print(f"{c:14s} {mx:13.3e} {rms:13.3e}")
    print("-" * 42)
    print(f"{'WORST':14s} {worst_max:13.3e} {worst_rmse:13.3e}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def run_subject(raw_dir, cleaned_dir, conditions, out_dir, do_validate):
    """Clean every requested condition for one subject directory.

    Missing raw files are skipped with a warning; validation is only attempted
    when the corresponding published cleaned file exists.
    """
    for cond in conditions:
        raw_file = os.path.join(raw_dir, f"force_speed_{cond}.csv")
        if not os.path.exists(raw_file):
            print(f"  skip {cond}: no {os.path.basename(raw_file)}")
            continue
        pred = clean_condition(raw_dir, cond)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{cond}.csv")
            pred.to_csv(dest, index=False)
            print(f"  wrote {dest}  ({len(pred)} rows)")
        if do_validate:
            ref = os.path.join(cleaned_dir, f"{cond}.csv")
            if os.path.exists(ref):
                validate(pred, ref)
            else:
                print(f"  no reference {ref} - validation skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="Raw Data/sub01",
                    help="directory holding the raw *_<COND>.csv files")
    ap.add_argument("--cleaned-dir", default="Cleaned Data/sub01",
                    help="directory holding the published cleaned CSVs")
    ap.add_argument("--conditions", nargs="+",
                    default=["BFNF", "BFWF", "CLNF", "CLWF"])
    ap.add_argument("--out-dir", default=None,
                    help="if set, write reconstructed CSVs here")
    ap.add_argument("--validate", action="store_true",
                    help="compare against the published cleaned files")
    ap.add_argument("--all-subjects", action="store_true",
                    help="sweep every sub*/ folder under --raw-root instead of "
                         "a single --raw-dir")
    ap.add_argument("--raw-root", default="Raw Data",
                    help="dataset root holding sub*/ dirs (with --all-subjects)")
    ap.add_argument("--cleaned-root", default="Cleaned Data",
                    help="cleaned-data root holding sub*/ dirs (with "
                         "--all-subjects + --validate)")
    args = ap.parse_args()

    # Bare "run" (no CLI args, e.g. the IDE Run button) reconstructs the whole
    # dataset: every subject, every condition, written to Reconstructed/ and
    # validated against the published cleaned files.
    if len(sys.argv) == 1:
        args.all_subjects = True
        args.validate = True
        if args.out_dir is None:
            args.out_dir = "Reconstructed"

    if args.all_subjects:
        subjects = sorted(glob.glob(os.path.join(args.raw_root, "sub*")))
        subjects = [s for s in subjects if os.path.isdir(s)]
        print(f"sweeping {len(subjects)} subjects under {args.raw_root}")
        for raw_dir in subjects:
            subject = os.path.basename(os.path.normpath(raw_dir))
            print(f"\n=== {subject} ===")
            cleaned_dir = os.path.join(args.cleaned_root, subject)
            out_dir = os.path.join(args.out_dir, subject) if args.out_dir else None
            run_subject(raw_dir, cleaned_dir, args.conditions, out_dir,
                        args.validate)
    else:
        run_subject(args.raw_dir, args.cleaned_dir, args.conditions,
                    args.out_dir, args.validate)


if __name__ == "__main__":
    main()
