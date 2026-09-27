"""Stage 6: figures in the layout of the paper's EEG figures.

fig_erp_<contrast>.png   Fig 2a,b (2-back), S2 (3-back), 4a,b (DSR): grand
                         average at Cz/Pz +- SEM and scalp maps per 200 ms.
                         As in Dataset_{A,B}_ERP.m, correct trials of all
                         subjects are pooled before averaging.
fig_ersp_<cond>.png      Fig 2c / S3 (0/2/3-back), 4c (DSR), 6a,b (BL, WG).
fig_wg_accuracy.png      Fig 7 EEG curve.
All from the paper variant unless the file name says otherwise.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegrepro import data, erp, plots  # noqa: E402
from eegrepro.plots import plt  # noqa: E402

FIG = data.RESULTS_DIR / "figures"
SCALP_IVALS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
ERP_FIGS = {  # figure contrast: (erp contrast, class labels as in the paper, ylim, clim)
    "2-back": ("2-back T/NT", ["non-target", "target"], 6, 8),
    "3-back": ("3-back T/NT", ["non-target", "target"], 6, 8),
    "DSR": ("DSR", ["symbol X", "symbol O"], 8, 8),
}


def erp_figure(name, variant="paper"):
    contrast, labels, ylim, clim = ERP_FIGS[name]
    task = erp.CONTRASTS[contrast][0]
    epos, ys = [], []
    for s in data.SUBJECTS:
        epo, t, y = erp.epochs(erp.load(s, task, variant), contrast, variant)
        epos.append(epo)
        ys.append(y)
    epo, y = np.concatenate(epos, axis=2), np.concatenate(ys)
    cls = [epo[..., y == k] for k in (0, 1)]
    means = [c.mean(axis=2) for c in cls]
    sems = [c.std(axis=2, ddof=1) / np.sqrt(c.shape[2]) for c in cls]
    diff = means[0] - means[1]
    pos, clab = data.load_montage()

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(4, 5, height_ratios=[1.5, 1, 1, 1], hspace=0.7)
    for j, ch in enumerate(["Cz", "Pz"]):
        ax = fig.add_subplot(gs[0, j * 2 + (1 if j else 0):j * 2 + 2 + (1 if j else 0)])
        ci = clab.index(ch)
        plots.waveform(ax, t, [m[:, ci] for m in means], [s[:, ci] for s in sems], labels, ch,
                       ylim=(-ylim, ylim), shade=SCALP_IVALS[::2])
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
    rows = [(labels[0], means[0]), (labels[1], means[1]), (f"{labels[0]} − {labels[1]}", diff)]
    for r, (lab, m) in enumerate(rows):
        for k, (a, b) in enumerate(SCALP_IVALS):
            ax = fig.add_subplot(gs[r + 1, k])
            im = plots.topomap(ax, m[(t > a) & (t <= b)].mean(axis=0), pos, (-clim, clim))
            if r == 0:
                ax.set_title(f"{a}–{b} ms", fontsize=8, fontweight="normal")
            if k == 0:
                ax.text(-1.55, 0, lab, rotation=90, va="center", ha="center", fontsize=8,
                        color=plots.TEXT2)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.35])
    fig.colorbar(im, cax=cax, label="µV")
    n = [int(np.sum(y == k)) for k in (0, 1)]
    fig.suptitle(f"{name}: grand average ERP ({variant} pipeline, correct trials, "
                 f"n = {n[0]} / {n[1]} pooled over 26 subjects)", x=0.08, ha="left",
                 fontsize=10, color=plots.TEXT)
    fig.savefig(FIG / f"fig_erp_{name}.png")
    plt.close(fig)


def ersp_figures(variant="paper"):
    z = np.load(data.RESULTS_DIR / f"ersp_{variant}.npz")
    for cond in ("0-back", "2-back", "3-back", "DSR", "BL", "WG"):
        t, f = z[f"{cond}_times"] / 1000, z[f"{cond}_freqs"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), sharey=True)
        xlim = (-5, t[-1])
        for ax, ch in zip(axes, ["AFz", "Cz", "Pz"]):
            im = plots.tf_image(ax, t, f, z[cond][list(z["channels"]).index(ch)], ch, xlim=xlim)
            if ch != "AFz":
                ax.set_ylabel("")
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="dB vs −5…−2 s")
        fig.suptitle(f"{cond}: grand-average ERD/ERS ({variant} pipeline, 26 subjects)",
                     x=0.06, ha="left", fontsize=10)
        fig.savefig(FIG / f"fig_ersp_{cond}{'' if variant == 'paper' else '_' + variant}.png")
        plt.close(fig)


def wg_figure():
    rows = list(csv.DictReader(open(data.RESULTS_DIR / "wg_classification.csv")))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axhline(50, color=plots.GRID, lw=1)
    ax.axvline(0, color=plots.TEXT2, lw=1, ls="--")
    runs = [("paper", "paper pipeline (10×5 CV)"),
            ("paper_nested_band", "paper pipeline, band chosen inside CV"),
            ("released", "released script (unfiltered, 1×5 CV)")]
    for k, (run, label) in enumerate(runs):
        acc = np.array([[float(r["accuracy"]) for r in rows if r["run"] == run and r["subject"] == s]
                        for s in data.SUBJECTS]) * 100
        ends = sorted({int(r["window_end_s"]) for r in rows})
        m, se = acc.mean(0), acc.std(0, ddof=1) / np.sqrt(acc.shape[0])
        ax.plot(ends, m, color=plots.SERIES[k], lw=2, label=label)
        if run == "paper":
            ax.fill_between(ends, m - se, m + se, color=plots.SERIES[k], alpha=0.15, lw=0)
            i = int(np.argmax(m))
            ax.annotate(f"peak {m[i]:.1f}% at {ends[i]} s", (ends[i], m[i]), xytext=(8, 8),
                        textcoords="offset points", fontsize=8, color=plots.TEXT)
    ax.plot([10], [76.9], marker="D", ms=8, color=plots.TEXT, ls="none",
            label="paper: EEG peak 76.9% at 10 s")
    ax.set_xlabel("right end of 5 s window (s)")
    ax.set_ylabel("WG vs BL accuracy (%)")
    ax.set_xlim(-5, 25)
    ax.set_ylim(40, 90)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Dataset C: EEG classification over time (mean of 26 subjects, ±SEM for paper pipeline)",
                 loc="left")
    fig.savefig(FIG / "fig_wg_accuracy.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for name in ERP_FIGS:
        erp_figure(name)
        print("erp", name, flush=True)
    for v in ("paper", "released"):
        ersp_figures(v)
    print("ersp", flush=True)
    wg_figure()
    print("wg", flush=True)


if __name__ == "__main__":
    main()
