"""Stage 1: paper preprocessing, cached per subject/task.

raw (200 Hz) -> 1-40 Hz 6th-order zero-phase Butterworth (per session)
             -> AAR ocular removal (iWASOBI + eog_fd, per session, all 30 channels)
             -> keep the 28 EEG channels -> cache/paper/<VP>_<task>.npz (float32)

Also writes results/qc_preprocess.csv: frontal-channel |r| with the original
VEOG/HEOG before and after cleaning, power ratio, components removed, fallbacks.
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegrepro import aar, data, filters  # noqa: E402

OUT = data.CACHE_DIR / "paper"
FRONTAL = ["Fp1", "Fp2", "AFz", "AFF5h", "AFF6h"]


def process(job):
    subject, task = job
    rec = data.load_raw(subject, task)
    xf = filters.bandpass(rec.x, rec.fs, (1, 40), sessions=rec.sessions)
    clean = np.empty_like(xf)
    info = []
    for a, b in rec.sessions:
        clean[a:b], inf = aar.autobss_eog(xf[a:b], rec.fs)
        info += inf
    eeg = rec.channels(data.EEG_CLAB)
    np.savez(OUT / f"{subject}_{task}.npz", x=clean[:, eeg].astype(np.float32),
             fs=rec.fs, mrk_ms=rec.mrk_ms, mrk_code=rec.mrk_code,
             sessions=np.array(rec.sessions))
    fr = rec.channels(FRONTAL)
    veog, heog = xf[:, rec.clab.index("VEOG")], xf[:, rec.clab.index("HEOG")]

    def mean_abs_r(z, ref):
        return float(np.mean([abs(np.corrcoef(z[:, c], ref)[0, 1]) for c in fr]))

    return {"subject": subject, "task": task,
            "veog_r_before": mean_abs_r(xf, veog), "veog_r_after": mean_abs_r(clean, veog),
            "heog_r_before": mean_abs_r(xf, heog), "heog_r_after": mean_abs_r(clean, heog),
            "power_ratio": float(np.sum(clean[:, eeg] ** 2) / np.sum(xf[:, eeg] ** 2)),
            "n_windows": len(info),
            "components_removed": " ".join(str(i["n_removed"]) for i in info),
            "fallbacks": sum(i["fallback"] for i in info)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data.RESULTS_DIR.mkdir(exist_ok=True)
    jobs = [(s, t) for s in data.SUBJECTS for t in data.TASKS]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for row in ex.map(process, jobs):
            rows.append(row)
            print(f"{row['subject']} {row['task']:5s} VEOG |r| {row['veog_r_before']:.2f}->"
                  f"{row['veog_r_after']:.2f}  power {row['power_ratio']:.2f}  "
                  f"removed [{row['components_removed']}]  fallbacks {row['fallbacks']}", flush=True)
    with open(data.RESULTS_DIR / "qc_preprocess.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
