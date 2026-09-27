"""Stage 5: WG vs BL EEG classification accuracy over time (paper Fig. 7, EEG).

Writes results/wg_classification.csv (subject x variant x window) and
results/wg_bands.csv, prints the grand-average peak.
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegrepro import data, wg  # noqa: E402

RUNS = [("paper", False), ("paper", True), ("released", False)]


def run(job):
    subject, variant, nested = job
    acc, band = wg.classify(subject, variant, nested_band=nested)
    return subject, variant, nested, acc, band


def main():
    jobs = [(s, v, n) for v, n in RUNS for s in data.SUBJECTS]
    rows, bands = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for subject, variant, nested, acc, band in ex.map(run, jobs):
            name = f"{variant}{'_nested_band' if nested else ''}"
            for end, a in zip(wg.WINDOW_ENDS_S, acc):
                rows.append({"subject": subject, "run": name, "window_end_s": int(end), "accuracy": a})
            bands.append({"subject": subject, "run": name, "band_lo": band[0], "band_hi": band[1]})
            print(subject, name, f"peak {acc.max():.3f} at {wg.WINDOW_ENDS_S[acc.argmax()]} s, band {band}",
                  flush=True)
    for fname, rs in (("wg_classification.csv", rows), ("wg_bands.csv", bands)):
        with open(data.RESULTS_DIR / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    for v, n in RUNS:
        name = f"{v}{'_nested_band' if n else ''}"
        m = np.array([[r["accuracy"] for r in rows if r["run"] == name and r["subject"] == s]
                      for s in data.SUBJECTS]).mean(axis=0) * 100
        i = int(np.argmax(m))
        print(f"{name:20s} grand-average peak {m[i]:.1f}% at t = {wg.WINDOW_ENDS_S[i]} s "
              f"(paper: 76.9% at 10 s); acc at 0 s {m[5]:.1f}%, at -5 s {m[0]:.1f}%")


if __name__ == "__main__":
    main()
