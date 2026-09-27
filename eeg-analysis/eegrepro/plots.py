"""Figure helpers (matplotlib).  Colours: categorical slots in fixed order for
classes/series, a blue <-> gray <-> red diverging map for signed amplitudes and
dB changes (neutral midpoint = no change)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # blue, orange, aqua, yellow
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#123d73", "#2a78d6", "#9cc3f1", "#f0efec", "#f3a9a8", "#e34948", "#8f1f1e"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.facecolor": "white", "savefig.dpi": 160, "savefig.bbox": "tight",
})


def waveform(ax, t, means, sems, labels, title, ylim=None, shade=None, base=(-100, 0)):
    """ERP at one channel: class means (2 px lines) with dashed +-SEM bounds."""
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvspan(*base, color=GRID, zorder=0)
    for a, b in shade or []:
        ax.axvspan(a, b, color="#f4f3f0", zorder=0)
    for k, (m, s, lab) in enumerate(zip(means, sems, labels)):
        ax.plot(t, m, color=SERIES[k], lw=2, label=lab)
        ax.plot(t, m + s, color=SERIES[k], lw=0.8, ls="--")
        ax.plot(t, m - s, color=SERIES[k], lw=0.8, ls="--")
    ax.set_title(title, loc="left")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude (µV)")
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(t[0], t[-1])


def topomap(ax, values, pos, vlim):
    im, _ = mne.viz.plot_topomap(values, pos, axes=ax, show=False, cmap=DIVERGING, vlim=vlim,
                                 contours=0, sensors=True, sphere=(0, 0, 0, 1.0),
                                 extrapolate="head")
    return im


def tf_image(ax, times_s, freqs, ersp_db, title, clim=3.0, xlim=None):
    im = ax.pcolormesh(times_s, freqs, ersp_db, cmap=DIVERGING, vmin=-clim, vmax=clim,
                       shading="auto", rasterized=True)
    ax.axvline(0, color=TEXT, lw=1, ls="--")
    ax.set_title(title, loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    if xlim:
        ax.set_xlim(xlim)
    return im
