"""ERP epochs and single-trial ERP classification (datasets A and B).

Variants
--------
paper     what the paper's Methods describe: 1-40 Hz band-pass + AAR ocular
          removal (scripts/01_preprocess.py), epochs -0.1..1 s, baseline
          -0.1..0 s, correct responses only.
released  Dataset_B_ERP_classification.m as published: unfiltered data, EOG
          channels dropped, no ocular removal, epochs -0.2..1 s, baseline
          -0.1..0 s, correct responses only.  The authors released no n-back
          classifier; the same recipe is applied to n-back for comparison.

Classification (both variants): signed r^2 -> 5 intervals chosen by the BBCI
heuristic -> mean amplitude per interval and channel (5 x 28 = 140 features)
-> shrinkage LDA -> 10 x 10-fold stratified CV, seeded per subject like
``rng(vp)``.  The released script picks the intervals on all trials before CV;
``nested=True`` repeats the selection inside every training fold instead.
"""
from __future__ import annotations

import numpy as np

from . import bbci, data

CONTRASTS = {
    # name: task, (class 0, class 1); a class is (label, n-back level or None)
    # Table 5 contrast, decided in scripts/03_nback_contrast.py: marker class
    # "0-back target" vs "2-back non-target" (resp. 3-back)
    "2-back": ("nback", (("target", 0), ("non-target", 2))),
    "3-back": ("nback", (("target", 0), ("non-target", 3))),
    # the Methods' literal reading: target vs non-target within one level
    "2-back T/NT": ("nback", (("non-target", 2), ("target", 2))),
    "3-back T/NT": ("nback", (("non-target", 3), ("target", 3))),
    "DSR": ("dsr", (("X", None), ("O", None))),
}
# Further readings of Table 5's "(versus 0-back)" row (scripts/03_nback_contrast.py)
ALT_CONTRASTS = {
    "2-back target vs 0-back": ("nback", (("target", 0), ("target", 2))),
    "3-back target vs 0-back": ("nback", (("target", 0), ("target", 3))),
    "2-back all vs 0-back": ("nback", (("target", 0), ("any", 2))),
    "3-back all vs 0-back": ("nback", (("target", 0), ("any", 3))),
}
VARIANTS = {
    "paper": {"source": "clean", "ival": (-100, 1000), "base": (-100, 0)},
    "released": {"source": "raw", "ival": (-200, 1000), "base": (-100, 0)},
}


def load(subject: str, task: str, variant: str) -> data.Recording:
    if VARIANTS[variant]["source"] == "clean":
        return data.load_clean(subject, task)
    rec = data.load_raw(subject, task)
    eeg = rec.channels(data.EEG_CLAB)
    rec.x, rec.clab = rec.x[:, eeg], list(data.EEG_CLAB)
    return rec


def epochs(rec: data.Recording, contrast: str, variant: str, correct_only: bool = True):
    """Returns (epo (T, 28, n), t_ms, y) with y = index into the contrast's classes."""
    task, classes = {**CONTRASTS, **ALT_CONTRASTS}[contrast]
    tr = data.trial_table(rec)
    y = np.full(len(tr.label), -1)
    for k, (label, n) in enumerate(classes):
        m = (tr.label == label) if label != "any" else np.ones(len(y), bool)
        if n is not None:
            m &= tr.nback == n
        y[m] = k
    mask = y >= 0
    if correct_only:
        mask &= tr.correct
    tr, y = tr.select(mask), y[mask]
    cfg = VARIANTS[variant]
    epo, t = bbci.segment(rec.x, rec.fs, tr.sample, cfg["ival"])
    epo = bbci.baseline(epo, t, cfg["base"])
    return epo, t, y


def features(epo: np.ndarray, t: np.ndarray, ivals: np.ndarray) -> np.ndarray:
    fv = bbci.jumping_means(epo, t, ivals)
    return fv.reshape(-1, fv.shape[-1])


def classify(epo: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int,
             nested: bool = False, n_rep: int = 10, n_folds: int = 10) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    ivals = bbci.select_time_intervals(bbci.r_square_signed(epo, y), t, 5)
    if not nested:
        return bbci.crossvalidate(features(epo, t, ivals), y, n_rep, n_folds, rng), ivals
    reps = []
    for _ in range(n_rep):
        accs = []
        for tr, te in bbci.sample_kfold(y, n_folds, rng):
            iv = bbci.select_time_intervals(bbci.r_square_signed(epo[..., tr], y[tr]), t, 5)
            clf = bbci.train_rlda_shrink(features(epo[..., tr], t, iv), y[tr])
            accs.append(np.mean(clf.predict(features(epo[..., te], t, iv)) == y[te]))
        reps.append(np.mean(accs))
    return float(np.mean(reps)), ivals
