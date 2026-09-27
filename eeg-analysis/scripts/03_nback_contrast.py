"""Stage 3: which n-back contrast is behind Table 5's "Classification acc.
(versus 0-back)" row?

The same pipeline reproduces the DSR accuracy exactly (86.8 %) but gives only
~68 % (= majority-class rate) for 2-/3-back target vs non-target (paper 76.9 /
76.0).  Table 6 lays out "symbol O ... (versus symbol X)" = O vs X, so by the
same layout Table 5 reads "2-back (versus 0-back)".  This script scores every
candidate reading with the paper pipeline.

Result: only "0-back target" vs "2-back non-target" (resp. 3-back) matches all
three Table 5 facts -- means, SDs and a non-significant 2- vs 3-back
difference.  Those are the literal marker class names, which reconciles the
text ("target vs non-target") with the table ("versus 0-back").
Writes results/nback_contrast_check.csv.
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegrepro import data, erp  # noqa: E402

CANDIDATES = ["2-back T/NT", "3-back T/NT", "2-back", "3-back", *erp.ALT_CONTRASTS]


def run(subject):
    seed = data.SUBJECTS.index(subject) + 1
    rec = erp.load(subject, "nback", "paper")
    rows = []
    for c in CANDIDATES:
        epo, t, y = erp.epochs(rec, c, "paper")
        acc, _ = erp.classify(epo, t, y, seed)
        rows.append({"subject": subject, "contrast": c, "n_class0": int(np.sum(y == 0)),
                     "n_class1": int(np.sum(y == 1)), "accuracy": acc,
                     "majority_rate": max(np.mean(y == 0), np.mean(y == 1))})
    return rows


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run, data.SUBJECTS):
            rows += r
    with open(data.RESULTS_DIR / "nback_contrast_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    def acc(c):
        return np.array([r["accuracy"] for r in rows if r["contrast"] == c]) * 100

    print(f"{'contrast':30s} {'acc':>12s} {'majority':>9s}   paper: 2-back 76.9 +- 8.0, 3-back 76.0 +- 7.9")
    for c in CANDIDATES:
        m = np.mean([r["majority_rate"] for r in rows if r["contrast"] == c]) * 100
        print(f"{c:30s} {acc(c).mean():5.1f} +- {acc(c).std(ddof=1):4.1f} {m:8.1f}")
    print("\n2-back vs 3-back Wilcoxon signed-rank (paper: P > 0.05)")
    for c2 in [c for c in CANDIDATES if c.startswith("2-back")]:
        c3 = c2.replace("2-back", "3-back")
        print(f"  {c2:30s} p = {wilcoxon(acc(c2), acc(c3)).pvalue:.3f}")


if __name__ == "__main__":
    main()
