# EEG Analysis Reproduction

Reproduction, in Python, of the EEG analyses in Shin et al. 2018, "Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset", *Scientific Data* 5:180003 ([sdata20183.pdf](sdata20183.pdf)). The paper's analysis was done in MATLAB (EEGLAB, the AAR toolbox and the BBCI toolbox). Only the EEG part is reproduced; NIRS is out of scope by design.

The results table is in [results/paper_vs_ours.md](results/paper_vs_ours.md), and the figures are in [results/figures/](results/figures/).

## Methods

Two versions of the complete pipeline were kept: the original published MATLAB scripts (ported to python) and a set of modified scripts to address gaps between the published pipeline and the parent paper.

The MATLAB functions the analysis depends on were ported to Python from their source code, which is copied under [reference/](reference/) with commit hashes in [reference/SOURCES.txt](reference/SOURCES.txt). The ported functions are:

- **BBCI toolbox:** segmentation, signed r², the time-interval heuristic, shrinkage LDA, stratified K-fold cross-validation, CSP with median-variance scoring, band selection and variance-based trial rejection.
- **AAR toolbox:** iWASOBI, the fractal-dimension component criterion and the sliding-window driver.
- **EEGLAB:** `newtimef` and the helper functions it calls.

### Data facts

- **The data is not cleaned.** `Cleaned Data/` is the raw recording converted to MATLAB and downsampled to 200 Hz. All filtering and cleaning happens in the analysis.
- **Channels:** 28 EEG channels, plus HEOG and VEOG. Marker times are in milliseconds.
- **Sessions:** each file holds 3 sessions concatenated. Every session's first marker is exactly 30 s after the session start, which is how the boundaries are recovered.
- **Behavior:** the authors' `behavior.zip` is unzipped into [behavior/](behavior/). It lines up with the EEG markers for all 14,040 n-back and 9,360 DSR trials, and it is used to keep correct trials only, as the paper does.
- **Trial counts:** DSR has 360 trials per subject, not the 180 the paper states.

### Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_preprocess.py` | Per session: 1–40 Hz, 6th-order zero-phase Butterworth filter, then AAR ocular removal (iWASOBI with the `eog_fd` criterion at `pop_autobsseog` defaults). Keeps the 28 EEG channels and caches them. |
| 2 | `02_erp_classification.py` | ERP classification. Epochs run −0.1…1 s with a −0.1…0 s baseline, correct trials only. Features are mean amplitudes in 5 heuristically chosen intervals (28 × 5), classified with shrinkage LDA under 10×10-fold cross-validation. Tables 5 and 6. |
| 3 | `03_nback_contrast.py` | Determines which n-back contrast Table 5 reports (see below). |
| 4 | `04_ersp.py` | ERD/ERS maps with the EEGLAB `newtimef` port (FFT with a 10.24 s Hanning window, single-trial baseline normalization, −5…−2 s baseline). Figs 2c, 4c, 6a,b and S3. |
| 5 | `05_wg_classification.py` | WG vs BL classification. Per-subject band selection, then CSP and shrinkage LDA on 5 s windows stepped by 1 s, under 10×5-fold cross-validation. Fig 7 EEG curve. |
| 6 | `06_figures.py` | Figures in the paper's layout. |
| 7 | `07_report.py` | Paper-vs-ours table. |

## Results

The full table is in [results/paper_vs_ours.md](results/paper_vs_ours.md). Accuracies are mean ± SD over 26 subjects.

| Target | Paper | Paper pipeline | Released pipeline |
|---|---|---|---|
| DSR, O vs X (Table 6) | 86.8 ± 5.9 | 86.8 ± 6.4 | 85.6 ± 7.1 |
| 2-back (Table 5) | 76.9 ± 8.0 | 77.7 ± 7.7 | 76.7 ± 7.1 |
| 3-back (Table 5) | 76.0 ± 7.9 | 76.1 ± 7.4 | 76.1 ± 6.9 |
| 2- vs 3-back, Wilcoxon | P > 0.05 | p = 0.12 | p = 0.76 |
| WG EEG peak (Fig 7) | 76.9 % at 10 s | 73.7 % at 6 s (72.0 % at 10 s) | 78.8 % at 5 s (77.1 % at 10 s) |

- **ERP classification (Tables 5 and 6)** reproduces. All accuracies are within one SD of the paper's, and the 2- vs 3-back difference is not significant.
- **ERD/ERS maps (Figs 2c, 4c, 6)** reproduce qualitatively, in [results/figures/](results/figures/):
  - 2-back: strong alpha ERD and weak low-beta ERD.
  - WG: alpha ERD starting at the −2 s instruction, followed by beta and gamma ERD.
  - DSR: gamma ERD strongest at Pz and delta ERD at AFz.
- **ERP waveforms (Figs 2a,b and 4a,b)** reproduce: N100, P200 and a delayed P300 that is larger for targets in 2-back.
- **WG classification (Fig 7)** reproduces the shape: accuracy rises before onset, plateaus for windows ending 5–10 s, then falls to ~55 % after 13–16 s. Its peak sits earlier than the paper's, at 5–6 s instead of 10 s, and it is broad. The ocular-cleaned pipeline is 4–5 points lower than the unfiltered one.
- **Nested selection.** Choosing the ERP intervals inside cross-validation costs 2–5 points for the Table 5 contrasts and about 1 for DSR. The paper's numbers match the non-nested version.

## Workflow

Python 3.11+ with numpy, scipy, matplotlib, mne and pytest ([requirements.txt](requirements.txt)):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
for s in 01_preprocess 02_erp_classification 03_nback_contrast 04_ersp 05_wg_classification 06_figures 07_report; do
    .venv/bin/python scripts/$s.py
done
```

Stage 1 writes about 2.8 GB to `cache/` (git-ignored); later stages read it. A full run takes about an hour on 8 cores.

## Data

Place the MATLAB-format EEG data (`VP001-EEG` … `VP026-EEG`) in `Cleaned Data/`. It is available from the dataset page, http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/.

## Licenses

The released scripts and the BBCI toolbox are MIT-licensed, and EEGLAB is BSD-licensed. The AAR toolbox is GPL-2, so [eegrepro/aar.py](eegrepro/aar.py), as a translation of it, is GPL-2 as well.
