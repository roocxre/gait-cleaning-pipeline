"""Stage 2: ERP single-trial classification (paper Tables 5 and 6).

Writes results/erp_classification.csv (one row per subject x contrast x
variant x selection mode) and prints the grand averages and the 2-back vs
3-back Wilcoxon signed-rank test.
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegrepro import data, erp  # noqa: E402

PAPER = {"2-back": (76.9, 8.0), "3-back": (76.0, 7.9), "DSR": (86.8, 5.9)}


def run_subject(job):
    subject, variant = job
    seed = data.SUBJECTS.index(subject) + 1                  # rng(vp)
    rows = []
    for task in ("nback", "dsr"):
        rec = erp.load(subject, task, variant)
        for contrast, (ctask, _) in erp.CONTRASTS.items():
            if ctask != task:
                continue
            epo, t, y = erp.epochs(rec, contrast, variant)
            for nested in (False, True):
                acc, ivals = erp.classify(epo, t, y, seed, nested=nested)
                rows.append({"subject": subject, "contrast": contrast, "variant": variant,
                             "selection": "nested" if nested else "before_cv",
                             "n_class0": int(np.sum(y == 0)), "n_class1": int(np.sum(y == 1)),
                             "accuracy": acc,
                             "intervals_ms": " ".join(f"{a:.0f}-{b:.0f}" for a, b in ivals)})
    return rows


def main():
    jobs = [(s, v) for v in erp.VARIANTS for s in data.SUBJECTS]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run_subject, jobs):
            rows += r
            print(r[0]["subject"], r[0]["variant"], flush=True)
    out = data.RESULTS_DIR / "erp_classification.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def accs(contrast, variant, selection):
        return np.array([r["accuracy"] for r in rows if r["contrast"] == contrast
                         and r["variant"] == variant and r["selection"] == selection]) * 100

    for variant in erp.VARIANTS:
        for selection in ("before_cv", "nested"):
            print(f"\n{variant} / intervals selected {selection}")
            for c in erp.CONTRASTS:
                a = accs(c, variant, selection)
                ref = f"(paper {PAPER[c][0]} +- {PAPER[c][1]})" if c in PAPER else ""
                print(f"  {c:12s} {a.mean():5.1f} +- {a.std(ddof=1):4.1f}   {ref}")
            p = wilcoxon(accs("2-back", variant, selection), accs("3-back", variant, selection)).pvalue
            print(f"  2-back vs 3-back Wilcoxon signed-rank p = {p:.3f}  (paper: P > 0.05)")


if __name__ == "__main__":
    main()
