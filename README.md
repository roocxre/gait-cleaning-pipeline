# Gait Cleaning Pipeline

A reverse-engineered, standalone reproduction of the data-cleaning pipeline for
the **Cognitive-Motor Coordination during walking** gait dataset
([Scientific Data, s41597-026-08103-4](https://www.nature.com/sdata/)).

Given the raw acquisition files for a subject/condition, `clean_pipeline.py`
reconstructs the unified 200 Hz cleaned CSV that the original authors published —
using **only** the raw inputs and the per-session hardware-calibration constants,
never the published cleaned files themselves.

## What it does

Four raw subsystems are fused onto one 200 Hz output grid:

| Raw file | Contents | Rate |
|---|---|---|
| `force_speed_<COND>.csv` | 8 force plates + treadmill speed + foot sensor | ~1008 Hz |
| `data_<COND>.csv` | top-view camera pixels, vibro-tactor states, task scores | ~62 Hz |
| `VR_data_<COND>.csv` | VR headset pose, object pose, VR score | ~70 Hz |
| `treadmill_calibration_*.csv` | per-plate force-plate calibration table | — |

Key transformations:

- **Master 200 Hz grid** by decimating the treadmill clock (5-point centred
  boxcar; grid starts at raw sample 305).
- **SpeedSensor** = `M·(MA200(V) − b)` — a 200-sample raw-domain moving average
  with gain/offset from the calibration table (reproduces to ~1e-5 RMS).
- **Force plates** → Newtons via the calibration table × per-session amplifier
  scale.
- **Camera** pixels → metres (per-subject fore-aft origin).
- **VR headset** position registered to the camera frame directly from the raw
  tracks (no cleaned reference needed).

## Usage

```bash
# Reconstruct everything: all subjects, all conditions, written to Reconstructed/,
# validated against Cleaned Data/ when present.
python clean_pipeline.py

# One subject / condition, with a validation report:
python clean_pipeline.py --raw-dir "Raw Data/sub01" \
    --cleaned-dir "Cleaned Data/sub01" --conditions CLNF --validate

# Sweep all subjects, write only (no validation):
python clean_pipeline.py --all-subjects --out-dir Reconstructed
```

Reconstruction depends only on `Raw Data/`; the `Cleaned Data/` folder is used
solely for the optional `--validate` accuracy report. If it is absent, files are
still reconstructed and validation is skipped.

## Data

The `Raw Data/`, `Cleaned Data/`, and `Reconstructed/` folders (~9 GB) are **not**
included in this repository — obtain the dataset from its original publication and
place `Raw Data/` and `Cleaned Data/` alongside `clean_pipeline.py`.

## Accuracy

For calibrated subjects most channels reproduce to floating-point or
near-instrument precision (force plates ~1e-3 N RMS, SpeedSensor ~1e-5, camera
often ~1e-7). Residual notes:

- A few camera-Z files retain ~3 cm RMS — the linear pixel→metre model does not
  fully capture the authors' image-processing correction on those files.
- `sub02`'s published cleaned file is un-converted (forces in volts, camera in
  pixels) — a defect in the dataset, so it cannot be matched by a physically
  correct reconstruction.

## Requirements

Python 3 with `numpy`, `pandas`, `scipy`.

## Dependencies

```bash
pip install numpy pandas scipy
```
