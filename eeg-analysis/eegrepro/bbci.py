"""Ports of the BBCI toolbox functions used by the authors' scripts.

Each function names its MATLAB original (sources in reference/bbci/).  Arrays
use the BBCI layout: epochs are (time, channel, trial); feature matrices for
classifiers are (feature, trial).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal.windows import kaiser


# --------------------------------------------------------------------------
# epochs
# --------------------------------------------------------------------------
def segment(x: np.ndarray, fs: float, samples: np.ndarray, ival_ms: tuple[float, float]):
    """proc_segmentation.  x: (n_samples, n_ch); samples: 0-based marker samples.
    Returns (epo (T, n_ch, n_trials), t_ms).  As in BBCI the interval is
    left-open: [-100, 1000] ms at 200 Hz gives t = -95 ... 1000 ms."""
    si = 1000.0 / fs
    a, b = ival_ms
    len_sa = int(round((b - a) / si))
    core = (int(np.ceil(a / si)), int(np.floor(b / si)))
    addone = int(core[1] - core[0] + 1 < len_sa)
    offsets = np.arange(-len_sa + 1, 1) + int(np.floor(b / si)) + addone
    idx = samples[None, :] + offsets[:, None]
    if idx.min() < 0 or idx.max() >= x.shape[0]:
        raise IndexError("epoch outside recording")
    t = np.linspace(si * (core[0] + 1), si * (core[1] + addone), len_sa)
    return np.transpose(x[idx], (0, 2, 1)), t


def baseline(epo: np.ndarray, t: np.ndarray, ival_ms: tuple[float, float]) -> np.ndarray:
    """proc_baseline (trialwise, 'beginning_exact'): subtract the mean over
    samples with a < t <= b."""
    m = (t > ival_ms[0]) & (t <= ival_ms[1])
    return epo - epo[m].mean(axis=0, keepdims=True)


def jumping_means(epo: np.ndarray, t: np.ndarray, ivals: np.ndarray) -> np.ndarray:
    """proc_jumpingMeans with explicit intervals (inclusive).  -> (n_ival, n_ch, n_tr)"""
    return np.stack([epo[(t >= lo) & (t <= hi)].mean(axis=0) for lo, hi in ivals])


def r_square_signed(epo: np.ndarray, y: np.ndarray) -> np.ndarray:
    """proc_rSquareSigned: sign(r) r^2 of the point-biserial r (class 0 minus class 1)."""
    c1, c2 = epo[..., y == 0], epo[..., y == 1]
    n1, n2 = c1.shape[-1], c2.shape[-1]
    div = epo.std(axis=-1, ddof=1)
    div[div == 0] = 1
    r = (c1.mean(-1) - c2.mean(-1)) * np.sqrt(n1 * n2) / (div * (n1 + n2))
    return r * np.abs(r)


# --------------------------------------------------------------------------
# procutil_selectTimeIntervals (heuristic of Blankertz et al. 2011)
# --------------------------------------------------------------------------
def select_time_intervals(rsq: np.ndarray, t: np.ndarray, n_ivals: int = 5,
                          score_factor_for_max: float = 3.0, r_rel_threshold: float = 0.5,
                          c_threshold: float = 0.75) -> np.ndarray:
    """rsq: (T, n_ch) signed r^2.  Greedy: pick the time of the highest score,
    grow while score > 0.5*peak and the scalp pattern correlates > 0.75 with
    the peak pattern, blank the interval, repeat."""
    X = rsq.copy()
    ivals = []
    for _ in range(n_ivals):
        Xpos, Xneg = np.maximum(X, 0), np.minimum(X, 0)
        score = (Xpos.mean(1) + Xpos.max(1) * score_factor_for_max
                 - (Xneg.mean(1) + Xneg.min(1) * score_factor_for_max))
        ti = int(np.argmax(score))
        peak = score[ti]
        if peak == 0:
            break
        lo = _enlarge(X, score, ti, -1, peak * r_rel_threshold, c_threshold)
        hi = _enlarge(X, score, ti, +1, peak * r_rel_threshold, c_threshold)
        ivals.append((t[lo], t[hi]))
        X[lo:hi + 1] = 0
    return np.array(ivals)


def _enlarge(X, score, ti, di, r_thr, c_thr) -> int:
    top = X[ti]
    bti, goon = ti, True
    while goon and 0 < bti < X.shape[0] - 1:
        bti += di
        row = X[bti]
        if np.any(row):
            corr = top @ row / np.sqrt(top @ top) / np.sqrt(row @ row)
            goon = score[bti] > r_thr and corr > c_thr
        else:
            goon = False
    return bti - di if not goon else bti


# --------------------------------------------------------------------------
# shrinkage LDA (train_RLDAshrink + clsutil_shrinkage, target 'B')
# --------------------------------------------------------------------------
@dataclass
class LDA:
    w: np.ndarray
    b: float

    def decision(self, X: np.ndarray) -> np.ndarray:        # X: (n_feat, n)
        return self.w @ X + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """0/1 labels.  BBCI loss_0_1: out > 0 -> class 2; out == 0 counts as wrong."""
        out = self.decision(X)
        return np.where(out > 0, 1, np.where(out < 0, 0, -1))


def shrinkage_cov(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Schaefer-Strimmer analytic shrinkage toward nu*I.  X: (p, n)."""
    p, n = X.shape
    Xn = X - X.mean(axis=1, keepdims=True)
    S = Xn @ Xn.T
    nu = np.mean(np.diag(S))
    T = nu * np.eye(p)
    Xn2 = Xn ** 2
    V = (Xn2 @ Xn2.T - S ** 2 / n) / (n - 1)
    gamma = float(np.clip(n * V.sum() / np.sum((S - T) ** 2), 0, 1))
    return (gamma * T + (1 - gamma) * S) / (n - 1), gamma


def train_rlda_shrink(X: np.ndarray, y: np.ndarray) -> LDA:
    """X: (n_feat, n), y in {0,1}; equal priors (BBCI default)."""
    means = np.stack([X[:, y == c].mean(axis=1) for c in (0, 1)], axis=1)
    Xc = np.concatenate([X[:, y == c] - means[:, [c]] for c in (0, 1)], axis=1)
    C, _ = shrinkage_cov(Xc)
    W = np.linalg.pinv(C) @ means
    b = -0.5 * np.sum(means * W, axis=0)
    return LDA(W[:, 1] - W[:, 0], float(b[1] - b[0]))


# --------------------------------------------------------------------------
# cross-validation (sample_KFold + crossvalidation, loss_0_1)
# --------------------------------------------------------------------------
def sample_kfold(y: np.ndarray, n_folds: int, rng: np.random.Generator):
    """One stratified split: per class, a random permutation cut at
    round(linspace(0, n_c, K+1)).  Yields (train_idx, test_idx)."""
    parts = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)[rng.permutation(np.sum(y == c))]
        div = np.round(np.linspace(0, len(idx), n_folds + 1)).astype(int)
        parts.append([idx[div[k]:div[k + 1]] for k in range(n_folds)])
    all_idx = np.arange(len(y))
    for k in range(n_folds):
        te = np.sort(np.concatenate([p[k] for p in parts]))
        yield np.setdiff1d(all_idx, te), te


def crossvalidate(X: np.ndarray, y: np.ndarray, n_rep: int, n_folds: int,
                  rng: np.random.Generator, fit=train_rlda_shrink) -> float:
    """crossvalidation(fv, @train_RLDAshrink, 'SampleFcn', {@sample_KFold,[R K]}):
    accuracy = mean over repetitions of mean over folds of fold accuracy."""
    reps = []
    for _ in range(n_rep):
        accs = [np.mean(fit(X[:, tr], y[tr]).predict(X[:, te]) == y[te])
                for tr, te in sample_kfold(y, n_folds, rng)]
        reps.append(np.mean(accs))
    return float(np.mean(reps))


# --------------------------------------------------------------------------
# CSP (proc_cspAuto with 'normalize',1,'score','medianvar','selectPolicy','equalperclass')
# --------------------------------------------------------------------------
def csp_auto(epo: np.ndarray, y: np.ndarray, patterns: int) -> np.ndarray:
    """epo: (T, n_ch, n_tr).  Returns spatial filters W (n_ch, 2*patterns)."""
    T, nch, _ = epo.shape
    R = []
    for c in (0, 1):
        covs = [np.cov(epo[:, :, i], rowvar=False) for i in np.flatnonzero(y == c)]
        Rc = np.mean(covs, axis=0)
        R.append(Rc / np.trace(Rc))
    from scipy.linalg import eigh
    _, W = eigh(R[1], R[0] + R[1])
    var = np.einsum("tci,cf->tfi", epo, W).var(axis=0, ddof=1)      # (n_filt, n_tr)
    v1 = np.median(var[:, y == 0], axis=1)
    v2 = np.median(var[:, y == 1], axis=1)
    score = v2 / (v1 + v2)
    di = np.argsort(score, kind="stable")
    fi = np.concatenate([di[:patterns], di[::-1][:patterns]])
    return W[:, fi]


def log_variance(epo: np.ndarray, W: np.ndarray) -> np.ndarray:
    """proc_linearDerivation + proc_variance + proc_logarithm -> (n_filt, n_tr)."""
    return np.log(np.einsum("tci,cf->tfi", epo, W).var(axis=0, ddof=1))


def trial_covariances(epo: np.ndarray) -> np.ndarray:
    """Per-trial channel covariance (ddof=1) -> (n_tr, n_ch, n_ch).  Everything
    csp_auto/log_variance need, so CV folds can reuse it."""
    xc = epo - epo.mean(axis=0, keepdims=True)
    return np.einsum("tci,tdi->icd", xc, xc) / (epo.shape[0] - 1)


def csp_auto_cov(covs: np.ndarray, y: np.ndarray, patterns: int) -> np.ndarray:
    """csp_auto computed from per-trial covariances (identical result)."""
    from scipy.linalg import eigh
    R = [covs[y == c].mean(axis=0) for c in (0, 1)]
    R = [r / np.trace(r) for r in R]
    _, W = eigh(R[1], R[0] + R[1])
    var = np.einsum("cf,icd,df->fi", W, covs, W)
    v1 = np.median(var[:, y == 0], axis=1)
    v2 = np.median(var[:, y == 1], axis=1)
    di = np.argsort(v2 / (v1 + v2), kind="stable")
    return W[:, np.concatenate([di[:patterns], di[::-1][:patterns]])]


def log_variance_cov(covs: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.log(np.einsum("cf,icd,df->fi", W, covs, W))


# --------------------------------------------------------------------------
# select_bandnarrow (participant-specific band for CSP)
# --------------------------------------------------------------------------
MOTOR_AREAS = [["FC5", "FC3", "CFC5", "CFC3", "C5", "C3", "CCP5", "CCP3", "CP5", "CP3"],
               ["FC1", "FCz", "FC2", "CFC1", "CFC2", "C1", "Cz", "C2", "CCP1", "CCP2",
                "CP1", "CPz", "CP2"],
               ["FC4", "FC6", "CFC4", "CFC6", "C4", "C6", "CCP4", "CCP6", "CP4", "CP6"]]


def select_bandnarrow(epo: np.ndarray, y: np.ndarray, clab: list[str], fs: float,
                      areas: list[list[str]] | None = MOTOR_AREAS,
                      band=(5, 35), band_topscore=(7, 35),
                      threshold=1 / 3, threshold_stop=1 / 20, threshold_ext=1 / 2):
    """epo: (T, n_ch, n_tr), already cut to the selection interval ([-2, 10] s).

    Differences from BBCI (both forced by this montage, see PLAN.md):
    the Laplacian is skipped because no channel of the 28-channel cap has the
    four grid neighbours the 'small' Laplacian requires, and areas are
    matched against the channels that exist (FC5 C3 CP5 | FC1 Cz FC2 CP1 CP2 |
    FC4.. FC6 C4 CP6)."""
    if areas is None:
        areas = [clab]
    win = kaiser(int(fs), 2)
    N = len(win)
    freqs = np.arange(N // 2 + 1) * fs / N
    ib = max(np.flatnonzero(freqs <= band[0]))
    ie = min(np.append(np.flatnonzero(freqs >= band[1]), len(freqs) - 1))
    step = N // 2
    T = epo.shape[0]
    n_win = 1 + max(0, (T - N) // step)
    X = np.zeros((N,) + epo.shape[1:])
    for k in range(n_win):
        seg = epo[k * step:k * step + N] * win[:, None, None]
        X += np.abs(np.fft.fft(seg, N, axis=0)) ** 2
    spec = 10 * np.log10(X[ib:ie + 1] / (n_win * np.sum(win ** 2)) + np.finfo(float).eps)
    f = freqs[ib:ie + 1]
    # proc_rSquareSigned with classes in mrk order: class 1 = WG (y==0), class 2 = BL
    score = r_square_signed(spec, y)
    score = _moving_average_centered(score, np.array([0.5, 1.0, 0.5]))
    idx = np.flatnonzero((f >= band_topscore[0]) & (f <= band_topscore[1]))
    chanscore = np.sqrt(np.sum(score[idx] ** 2, axis=0))
    chansel = []
    for area in areas:
        ci = [clab.index(c) for c in area if c in clab]
        if ci:
            chansel.append(ci[int(np.argmax(chanscore[ci]))])
    xx = score[:, chansel]
    topfreq = idx[int(np.argmax(np.mean(np.abs(xx[idx]), axis=1)))]
    xx = xx * np.sign(xx[topfreq])
    freqscore = xx.mean(axis=1)
    top = int(np.argmax(freqscore))
    topscore = freqscore[top]
    sel, ext = [top, top], [0.0, 0.0]
    while sel[0] > 0:                           # extend downwards
        fsc = freqscore[sel[0] - 1]
        if fsc >= topscore * threshold:
            sel[0] -= 1
        else:
            if fsc < topscore * threshold_stop:
                ext[0] = 0
            elif np.mean(freqscore[sel[0] - 1:sel[0] + 1]) >= topscore * threshold_ext:
                ext[0] = -1
            else:
                ext[0] = -0.5
            break
    while sel[1] < len(freqscore) - 1:          # extend upwards
        fsc = freqscore[sel[1] + 1]
        if fsc >= topscore * threshold:
            sel[1] += 1
        else:
            if fsc < topscore * threshold_stop:
                ext[1] = 0
            elif np.mean(freqscore[sel[1]:sel[1] + 2]) >= topscore * threshold_ext:
                ext[1] = 1
            else:
                ext[1] = 0.5
            break
    return (f[sel[0]] + ext[0], f[sel[1]] + ext[1])


def _moving_average_centered(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    """procutil_movingAverage(x, 3, 'centered', 'Window', [.5 1 .5]'):
    weighted centred average along axis 0, renormalised at the edges."""
    n = len(window)
    half = n // 2
    out = np.empty_like(x)
    T = x.shape[0]
    for k in range(T):
        lo, hi = max(0, k - half), min(T, k + half + 1)
        w = window[lo - k + half:hi - k + half]
        out[k] = np.tensordot(w, x[lo:hi], axes=(0, 0)) / w.sum()
    return out


# --------------------------------------------------------------------------
# reject_varEventsAndChannels (default options) -- used for ERP figures only
# --------------------------------------------------------------------------
def _percentiles(v: np.ndarray, p: tuple[float, float]) -> tuple[float, float]:
    """stat_percentiles: p<50 -> x[1+floor(p N)], p>=50 -> x[ceil(p N)] (1-based)."""
    xs = np.sort(v[~np.isnan(v)])
    N = len(xs)
    out = []
    for q in p:
        q = q / 100
        pos = int(np.ceil(q * N)) if q >= 0.5 else 1 + int(np.floor(q * N))
        out.append(xs[pos - 1])
    return out[0], out[1]


def reject_var_events(x: np.ndarray, fs: float, samples: np.ndarray,
                      ival_ms=(-100, 1000)) -> np.ndarray:
    """Boolean mask of events kept.  Variance of 5-40 Hz (causal butter(5))
    filtered epochs; whisker threshold perc90 + 3*(perc90-perc10)."""
    b, a = butter(5, np.array([5, 40]) / fs * 2, btype="bandpass")
    xf = lfilter(b, a, x, axis=0)
    epo, _ = segment(xf, fs, samples, ival_ms)
    V = epo.var(axis=0, ddof=1)                                     # (n_ch, n_ev)
    n_ev = V.shape[1]
    ch_good = np.arange(V.shape[0])
    ev_good = np.arange(n_ev)
    rejected = []
    silent = np.flatnonzero(np.mean(V < 0.5, axis=1) > 0.1)
    V = np.delete(V, silent, axis=0)
    ch_good = np.delete(ch_good, silent)

    def thresh(M):
        lo, hi = _percentiles(M.ravel(), (10, 90))
        return hi + 3 * (hi - lo)

    th = thresh(V)
    r = np.flatnonzero(np.mean(V > th, axis=0) > 0.2)
    rejected += list(ev_good[r])
    V, ev_good = np.delete(V, r, axis=1), np.delete(ev_good, r)
    th = thresh(V)
    isout = V > th
    if isout.sum() > 0.05 * n_ev:
        qu = isout.sum(1) / isout.sum()
        rc = np.flatnonzero((qu > 0.1) & (isout.mean(1) > 0.05))
        V, ch_good = np.delete(V, rc, axis=0), np.delete(ch_good, rc)
        th = thresh(V)
    r = np.flatnonzero(np.any(V > th, axis=0))
    rejected += list(ev_good[r])
    keep = np.ones(n_ev, bool)
    keep[rejected] = False
    return keep
