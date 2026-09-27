"""Stage 4: ERD/ERS spectral power maps (paper Figs 2c, 4c, 6a,b; Suppl. S3).

Per subject, condition and channel: EEGLAB newtimef port (eegrepro.ersp) on
task-onset epochs, then the mean over subjects (as Dataset_*_ERSP.m).

Epochs are -15..60 s (n-back, DSR) and -15..30 s (WG) as in the released
scripts, not the paper's -5 s start: with newtimef's default 10.24 s window the
first output time is 5.1 s after the epoch start, so a -5 s epoch would leave
no output time inside the -5..-2 s baseline.

Variants: paper (1-40 Hz + AAR data) and released (unfiltered, EOG dropped).
Output: results/ersp_<variant>.npz, grand averages at AFz, Cz and Pz (the
channels the paper shows; all 28 are computed, only these are kept to save disk).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegrepro import bbci, data, erp, ersp  # noqa: E402

CONDITIONS = {"0-back": ("nback", (-15_000, 60_000)), "2-back": ("nback", (-15_000, 60_000)),
              "3-back": ("nback", (-15_000, 60_000)), "DSR": ("dsr", (-15_000, 60_000)),
              "BL": ("wg", (-15_000, 30_000)), "WG": ("wg", (-15_000, 30_000))}
SHOW = ["AFz", "Cz", "Pz"]


def run(job):
    subject, variant = job
    out = {}
    for task in ("nback", "dsr", "wg"):
        rec = erp.load(subject, task, variant)
        samples, labels = data.series_table(rec)
        for cond, (ctask, ival) in CONDITIONS.items():
            if ctask != task:
                continue
            s = samples[labels == cond]
            # BBCI proc_segmentation drops epochs that run past the recording
            s = s[s + int(ival[1] / 1000 * rec.fs) < rec.x.shape[0]]
            epo, _ = bbci.segment(rec.x, rec.fs, s, ival)
            maps = []
            for ch in range(epo.shape[1]):
                p, times, freqs = ersp.newtimef(epo[:, ch, :], ival, rec.fs)
                maps.append(p.astype(np.float32))
            out[cond] = np.stack(maps)                       # (ch, f, t)
            out[f"{cond}_n"] = len(s)
    return subject, variant, out, times, freqs


def main():
    idx = [data.EEG_CLAB.index(c) for c in SHOW]
    for variant in ("paper", "released"):
        jobs = [(s, variant) for s in data.SUBJECTS]
        total = {c: 0.0 for c in CONDITIONS}      # running sum -> grand average
        axes = {}
        with ProcessPoolExecutor(max_workers=4) as ex:
            for subject, _, out, t, f in ex.map(run, jobs):
                for c in CONDITIONS:
                    total[c] = total[c] + out[c][idx]
                    axes[CONDITIONS[c][0]] = (t, f)
                print(variant, subject, {c: out[f"{c}_n"] for c in CONDITIONS}, flush=True)
        res = {"channels": SHOW}
        for c in CONDITIONS:
            t, f = axes[CONDITIONS[c][0]]
            res[c] = (total[c] / len(jobs)).astype(np.float32)          # (3, f, t)
            res[f"{c}_times"], res[f"{c}_freqs"] = t, f
        np.savez_compressed(data.RESULTS_DIR / f"ersp_{variant}.npz", **res)


if __name__ == "__main__":
    main()
