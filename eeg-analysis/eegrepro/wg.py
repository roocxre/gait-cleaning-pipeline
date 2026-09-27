"""WG vs BL EEG classification over sliding windows (dataset C, Fig. 7 EEG curve).

Follows the EEG part of Dataset_C_meta-classification.m:

1. epochs -10..25 s around task onset, trialwise baseline -5..-2 s
2. participant-specific band: select_bandnarrow on the -2..10 s epochs of all
   trials (see bbci.select_bandnarrow for the montage-forced changes)
3. zero-phase butter(3, band) on the epochs
4. 5 s windows, 1 s step, window ends -5..25 s (31 windows)
5. per window and fold: CSP (14 filters per class = all 28) on the training
   fold, log-variance features, shrinkage LDA; accuracy from the confusion
   matrix pooled over folds

Variants
--------
paper     cleaned data (1-40 Hz + AAR), 10 x 5-fold CV as the paper states
released  unfiltered data, EOG dropped, 1 x 5-fold (nShift = 1 in the script)

``nested_band=True`` repeats the band selection on each training fold (the
released code selects it on all trials, a mild leak).
"""
from __future__ import annotations

import numpy as np

from . import bbci, data, erp
from .filters import bandpass

WINDOW_ENDS_S = np.arange(-5, 26)
VARIANTS = {"paper": {"source": "clean", "n_rep": 10},
            "released": {"source": "raw", "n_rep": 1}}


def prepare(subject: str, variant: str):
    rec = erp.load(subject, "wg", "paper" if VARIANTS[variant]["source"] == "clean" else "released")
    tr = data.trial_table(rec)
    y = (tr.label == "BL").astype(int)                    # 0 = WG (class 1), 1 = BL
    epo, t = bbci.segment(rec.x, rec.fs, tr.sample, (-10_000, 25_000))
    epo = bbci.baseline(epo, t, (-5_000, -2_000))
    return epo, t, y, rec.clab, rec.fs


def band_for(epo, t, y, clab, fs, idx=None):
    sel = (t > -2_000) & (t <= 10_000)
    e = epo[sel] if idx is None else epo[sel][..., idx]
    yy = y if idx is None else y[idx]
    return bbci.select_bandnarrow(e, yy, clab, fs)


def window_covs(epo_f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n_windows, n_tr, n_ch, n_ch) covariances of each 5 s window."""
    out = []
    for end in WINDOW_ENDS_S * 1000:
        sel = (t > end - 5_000) & (t <= end)            # proc_selectIval
        out.append(bbci.trial_covariances(epo_f[sel]))
    return np.stack(out)


def classify(subject: str, variant: str, nested_band: bool = False):
    seed = data.SUBJECTS.index(subject) + 1
    rng = np.random.default_rng(seed)
    epo, t, y, clab, fs = prepare(subject, variant)
    n_rep = VARIANTS[variant]["n_rep"]
    band = band_for(epo, t, y, clab, fs)
    covs = window_covs(bandpass(epo, fs, band, axis=0), t) if not nested_band else None
    acc = np.zeros((n_rep, len(WINDOW_ENDS_S)))
    for r in range(n_rep):
        folds = list(bbci.sample_kfold(y, 5, rng))
        correct = np.zeros(len(WINDOW_ENDS_S))
        for tr, te in folds:
            if nested_band:
                fb = band_for(epo, t, y, clab, fs, tr)
                c = window_covs(bandpass(epo, fs, fb, axis=0), t)
            else:
                c = covs
            for w in range(len(WINDOW_ENDS_S)):
                W = bbci.csp_auto_cov(c[w, tr], y[tr], patterns=14)
                clf = bbci.train_rlda_shrink(bbci.log_variance_cov(c[w, tr], W), y[tr])
                correct[w] += np.sum(clf.predict(bbci.log_variance_cov(c[w, te], W)) == y[te])
        acc[r] = correct / len(y)
    return acc.mean(axis=0), band
