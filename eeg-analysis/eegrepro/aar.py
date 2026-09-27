"""Ocular artifact removal: port of the AAR toolbox (Gomez-Herrero) as called by
``pop_autobsseog(EEG, [], [], 'iwasobi')`` with all other options at their
defaults.

Copyright of the original MATLAB code: German Gomez-Herrero (AAR, GPL-2) and
Petr Tichavsky (iWASOBI).  This file is a translation and is therefore
distributed under the GNU GPL v2 or later.  Sources: reference/aar/.

Why these settings
------------------
The paper: "EEGLAB was used for automatic artifact removal (AAR) toolbox ...
Improved weight-adjusted second-order blind identification (iWASOBI) method
implemented in AAR toolbox was applied" and cites Gomez-Herrero et al. 2006,
"Automatic removal of ocular artifacts in the EEG *without an EOG reference
channel*".  That paper's method is AAR's default component criterion,
``eog_fd`` (low fractal dimension = ocular source).  Nothing else is specified,
so the ``pop_autobsseog`` defaults are used:

* analysis window = shift = 0.5 * n_channels^2 seconds (non-overlapping),
* iWASOBI: PCA pre-whitening (eigen-ratio 1e6), AR order 10, rmax 0.99,
  eps0 5e-7,
* eog_fd: Sevcik fractal dimension averaged over windows of 10 % of the data,
  remove between min(2, n/9) and max(.., n/3) components.

Two deliberate deviations, both so that the output matches the paper's claim
of "ocular artifact-free EEG":

1. ``cover_tail``: AAR drops a final window that would extend more than half a
   window past the data and then leaves that tail *uncleaned*.  Here the tail
   is cleaned with one extra window aligned to the end of the data.
2. Divergence fallback (see ``_clean_window``): if iWASOBI's weighted stage
   produces an unmixing whose removal increases total power, the window uses
   iWASOBI's uniformly weighted initial solution instead.  Counts are logged.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz


# --------------------------------------------------------------------------
# PCA pre-whitening (pca.m)
# --------------------------------------------------------------------------
def pca_whiten(x: np.ndarray, nbc: int | None = None, eigratio: float = 1e6):
    """x: (d, N).  Returns (W, W @ x) with W = D^-1/2 V' over retained PCs."""
    d = x.shape[0]
    nbc = d if nbc is None else nbc
    evals, evecs = np.linalg.eigh(np.cov(x))
    order = np.argsort(np.abs(evals))[::-1]
    val = np.abs(evals)[order]
    while val[0] / val[nbc - 1] > eigratio:
        nbc -= 1
    V = evecs[:, order[:nbc]]
    W = np.diag(evals[order[:nbc]] ** -0.5) @ V.T
    return W, W @ x


# --------------------------------------------------------------------------
# iWASOBI (Tichavsky, Yeredor & Nielsen 2008)
# --------------------------------------------------------------------------
def iwasobi(x: np.ndarray, ar_order: int = 10, rmax: float = 0.99, eps0: float = 5.0e-7,
            n_iter: int = 3, return_init: bool = False):
    """Demixing matrix W (d x d) for x (d, N).  With ``return_init`` also the
    uniformly weighted (SOBI-like) initial solution."""
    d, N = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    T = N - ar_order
    C0 = np.stack([x[:, :T] @ x[:, k:T + k].T / T for k in range(ar_order + 1)])
    C0[1:] = 0.5 * (C0[1:] + np.transpose(C0[1:], (0, 2, 1)))
    W0, Ms = _uwajd(C0, 20)
    W = W0
    for _ in range(n_iter):
        H = _weights(Ms, rmax, eps0)
        W, Ms = _wajd(C0, H, W, 5)
    return (W, W0) if return_init else W


def _transform(M: np.ndarray, W: np.ndarray):
    Ms = np.einsum("ij,ljk,mk->lim", W, M, W)
    Rs = np.einsum("lii->il", Ms)          # (d, L) diagonals
    return Ms, Rs


def _update(W, M, d1, d2):
    d = W.shape[0]
    A0 = np.eye(d)
    m = 0
    for i in range(1, d):
        A0[i, :i] = d1[m:m + i]
        A0[:i, i] = d2[m:m + i]
        m += i
    W = np.linalg.solve(A0, W)
    aux = 1.0 / np.sqrt(np.diag(W @ M[0] @ W.T))
    return aux[:, None] * W


def _uwajd(M: np.ndarray, maxnumiter: int = 20):
    """Uniformly weighted AJD (initial separation).  M: (L, d, d) lagged covs."""
    L, d, _ = M.shape
    E, Hv = np.linalg.eigh(M[0])
    W = np.diag(1.0 / np.sqrt(E)) @ Hv.T
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    Ms, Rs = _transform(M, W)
    crit = np.sum(Ms ** 2) - np.sum(Rs ** 2)
    improve, it = 10.0, 0
    while improve > 1e-7 and it < maxnumiter:
        b11, b12, b22, c1, c2 = [], [], [], [], []
        for i in range(1, d):
            Yim = Ms[:, :i, i].T                                   # (i, L)
            b22.append(np.full(i, np.sum(Rs[i] ** 2)))
            b12.append(Rs[:i] @ Rs[i])
            b11.append(np.sum(Rs[:i] ** 2, axis=1))
            c2.append(Yim @ Rs[i])
            c1.append(np.sum(Rs[:i] * Yim, axis=1))
        b11, b12, b22, c1, c2 = map(np.concatenate, (b11, b12, b22, c1, c2))
        det0 = b11 * b22 - b12 ** 2
        W = _update(W, M, (c1 * b22 - b12 * c2) / det0, (b11 * c2 - b12 * c1) / det0)
        Ms, Rs = _transform(M, W)
        critic = np.sum(Ms ** 2) - np.sum(Rs ** 2)
        improve, crit = abs(critic - crit), critic
        it += 1
    return W, Ms


def _wajd(M: np.ndarray, H: np.ndarray, W: np.ndarray, maxnumit: int):
    """Weighted AJD.  H: (n_pairs, L, L) weight blocks, pair order (i>k) row-major."""
    L, d, _ = M.shape
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    Ms, Rs = _transform(M, W)
    ii, kk = _pairs(d)
    for _ in range(maxnumit):
        Yim = Ms[:, ii, kk].T                                       # (n_pairs, L)
        Wlam1 = np.einsum("pab,pb->pa", H, Rs[ii])
        Wlam2 = np.einsum("pab,pb->pa", H, Rs[kk])
        b11 = np.sum(Rs[kk] * Wlam2, axis=1)
        b12 = np.sum(Rs[ii] * Wlam2, axis=1)
        b22 = np.sum(Rs[ii] * Wlam1, axis=1)
        c1 = np.sum(Wlam2 * Yim, axis=1)
        c2 = np.sum(Wlam1 * Yim, axis=1)
        det0 = b11 * b22 - b12 ** 2
        W = _update(W, M, (c1 * b22 - b12 * c2) / det0, (b11 * c2 - b12 * c1) / det0)
        Ms, Rs = _transform(M, W)
    return W, Ms


def _pairs(d: int):
    ii, kk = [], []
    for i in range(1, d):
        for k in range(i):
            ii.append(i)
            kk.append(k)
    return np.array(ii), np.array(kk)


def _weights(Ms: np.ndarray, rmax: float, eps0: float) -> np.ndarray:
    L, d, _ = Ms.shape
    R = np.einsum("lii->li", Ms)                                    # (L, d)
    ARC, sigmy = _armodel(R, rmax)
    ii, kk = _pairs(d)
    AR3 = np.stack([np.convolve(ARC[:, i], ARC[:, k]) for i, k in zip(ii, kk)], axis=1)
    phi = _ar2r(AR3)
    H = _thinv5(phi, L, len(ii), eps0 * phi[0])
    fact = 1.0 / (sigmy[ii] * sigmy[kk])
    return H * fact[:, None, None]


def _armodel(R: np.ndarray, rmax: float):
    M, d = R.shape
    AR = np.zeros((M, d))
    for j in range(d):
        a = np.concatenate([[1.0], -np.linalg.solve(toeplitz(R[:M - 1, j]), R[1:M, j])])
        v = np.roots(a)
        outside = np.abs(v) > 1                                     # polystab
        v = np.where(outside, 1.0 / np.conj(v), v)
        vmax = np.max(np.abs(v))
        if vmax > rmax:
            v = v * rmax / vmax
        AR[:, j] = np.real(np.poly(v))
    Rs = _ar2r(AR)
    return AR, R[0] / Rs[0]


def _ar2r(a: np.ndarray) -> np.ndarray:
    """Autocovariance (rows = lags 0..p) of unit-variance-innovation AR models
    given by the columns of ``a``.  Indices kept 1-based as in ar2r.m."""
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        a = a.T
    p1, m = a.shape
    A = np.zeros((p1 + 1, m))
    A[1:] = a
    alfa = A.copy()
    K = np.zeros((p1 + 1, m))
    p = p1 - 1
    for n in range(p, 0, -1):
        K[n] = -A[n + 1]
        for k in range(1, n):
            alfa[k + 1] = (A[k + 1] + K[n] * A[n - k + 1]) / (1 - K[n] ** 2)
        A = alfa.copy()
    r = np.zeros((p + 2, m))
    r[1] = 1.0 / np.prod(1 - K[1:] ** 2, axis=0)
    f = r.copy()
    b = f.copy()
    for k in range(1, p + 1):
        for n in range(k, 0, -1):
            Kn = K[n]
            f[n] = f[n + 1] + Kn * b[k - n + 1]
            b[k - n + 1] = -Kn * f[n + 1] + (1 - Kn ** 2) * b[k - n + 1]
        b[k + 1] = f[1]
        r[k + 1] = f[1]
    return r[1:]


def _thinv5(phi: np.ndarray, K: int, M: int, eps: np.ndarray) -> np.ndarray:
    """Inverses of M symmetric Toeplitz-plus-Hankel K x K matrices (THinv5.m).
    Returns (M, K, K).  1-based row indices via a padded leading row."""
    P = np.zeros((2 * K + 1, M))
    P[1:phi.shape[0] + 1] = phi[:2 * K]
    z = np.zeros((1, M))
    one = np.ones((1, M))
    x1, x2, x3, x4 = (np.zeros((K + 1, M)) for _ in range(4))
    almold = 2 * P[1] + eps
    C0 = 1.0 / almold
    x1[1] = C0
    x2[1] = C0
    x3[1] = -C0 * P[2]
    x4[1] = -2 * C0 * P[2]
    x4old = np.zeros((0, M))
    lalold = 2 * P[2] / almold
    for k in range(1, K):
        f2o = P[k + 1:1:-1] + P[k + 1:2 * k + 1]                    # rows k+1..2 / k+1..2k
        alm = np.sum(f2o * x4[1:k + 1], axis=0) + P[1] + eps + P[2 * k + 1]
        a0 = P[k + 2] if k < K - 1 else np.zeros(M)
        gam1 = np.sum(f2o * x1[1:k + 1], axis=0)
        gam3 = np.sum(f2o * x3[1:k + 1], axis=0) + a0 + P[k]
        x4[k + 1] = 1.0
        b1m = np.sum((np.vstack([P[2:k + 2], a0]) + np.vstack([z, P[1:k + 1]])) * x4[1:k + 2], axis=0)
        b2m = np.sum((np.vstack([a0, P[k + 1:1:-1]]) + P[k + 2:2 * k + 3]) * x4[1:k + 2], axis=0)
        latemp = b2m / alm
        b2m = latemp - lalold
        lalold = latemp
        bom = alm / almold
        x2[1:k + 2] = x4[1:k + 2] / alm
        x1[1:k + 2] = np.vstack([x1[1:k + 1], z]) - gam1 * x2[1:k + 2]
        x3[1:k + 2] = np.vstack([x3[1:k + 1], z]) - gam3 * x2[1:k + 2]
        x4temp = x4[1:k + 1].copy()
        x4[1:k + 2] = (np.vstack([z, x4[1:k + 1]]) + np.vstack([x4[2:k + 1], one, z])
                       - bom * np.vstack([x4old, one, z])
                       - b2m * x4[1:k + 2] - b1m * x1[1:k + 2] - x4[1] * x3[1:k + 2])
        x4old = x4temp
        almold = alm
    x1, x2, x3, x4 = x1[1:], x2[1:], x3[1:], x4[1:]
    Q = phi.copy() if phi.shape[0] >= 2 * K else np.vstack([phi, np.zeros((2 * K - phi.shape[0], M))])
    f1 = np.vstack([Q[1:K], z]) + np.vstack([z, Q[:K - 1]])
    f2 = np.vstack([z, Q[K - 1:0:-1]]) + np.vstack([Q[K:2 * K - 1], z])
    G = np.zeros((K, M, K))                                          # G[:, m, k]
    G[:, :, 0] = x1
    clast = np.zeros((K, M))
    for k in range(1, K):
        ck = G[:, :, k - 1]
        G[:, :, k] = (np.vstack([ck[1:], z]) + np.vstack([z, ck[:-1]]) - clast
                      - np.sum(f1 * ck, axis=0) * x1 - np.sum(f2 * ck, axis=0) * x2
                      - ck[0] * x3 - ck[K - 1] * x4)
        clast = ck
    return np.transpose(G, (1, 0, 2))


# --------------------------------------------------------------------------
# component selection: eog_fd (fractal dimension)
# --------------------------------------------------------------------------
def sevcik_fd(wave: np.ndarray) -> float:
    n = len(wave)
    span = wave.max() - wave.min()
    if span < 1e-6:
        return 1.0
    y = (wave - wave.max()) / span
    L = np.sum(np.sqrt((1.0 / (n - 1)) ** 2 + np.diff(y) ** 2))
    return 1 + np.log(L) / np.log(2 * (n - 1))


def eog_fd(Y: np.ndarray, rng_: tuple[int, int] | None = None) -> np.ndarray:
    """Indices of components judged ocular (lowest mean Sevcik FD)."""
    d, N = Y.shape
    lo, hi = rng_ if rng_ is not None else (min(2, d // 9), d // 3)
    wl = int(np.floor(0.1 * N))
    starts = np.arange(0, N, wl)
    fdim = np.array([np.mean([sevcik_fd(y[s:min(s + wl, N)]) for s in starts]) for y in Y])
    order = np.argsort(fdim, kind="stable")
    fsorted = fdim[order]
    dist = np.diff(fsorted)
    index = order[:1]
    for j in range(1, len(dist)):
        if np.mean(np.cumsum(dist[j:])) < np.mean(np.cumsum(dist[:j])):
            index = order[:j]
            break
    if len(index) < lo:
        index = order[:lo]
    if len(index) > hi:
        index = order[:hi]
    return index


# --------------------------------------------------------------------------
# sliding-window driver (autobss.m / pop_autobsseog.m defaults)
# --------------------------------------------------------------------------
def _remove(Xc: np.ndarray, W: np.ndarray, fd_range):
    A = np.linalg.pinv(W)
    Y = W @ Xc
    idx = eog_fd(Y, fd_range)
    if len(idx) >= Xc.shape[0]:
        return np.zeros_like(Xc), idx
    return Xc - A[:, idx] @ Y[idx], idx


def _clean_window(Xw: np.ndarray, fd_range) -> tuple[np.ndarray, dict]:
    """One AAR analysis window.  Safeguard (not in AAR): on some windows of this
    dataset the weighted iterations of iWASOBI diverge (the AR-based weight
    matrices are nearly singular for strongly coloured EEG) and removing the
    selected "components" *adds* power -- up to 3x per channel.  Removing
    sources can only lower total power, so in that case the window falls back
    to iWASOBI's own uniformly weighted initial solution (SOBI-like AJD)."""
    mu = Xw.mean(axis=1, keepdims=True)
    Xc = Xw - mu
    Wp, Xp = pca_whiten(Xc, eigratio=1e6)
    W, W0 = iwasobi(Xp, return_init=True)
    out, idx = _remove(Xc, W @ Wp, fd_range)
    fallback = np.sum(out ** 2) > np.sum(Xc ** 2)
    if fallback:
        out, idx = _remove(Xc, W0 @ Wp, fd_range)
    return out + mu, {"n_removed": int(len(idx)), "fallback": bool(fallback)}


def autobss_eog(x: np.ndarray, fs: float, wl_seconds: float | None = None,
                cover_tail: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Clean continuous data x (n_samples, n_channels).  Returns the cleaned data
    and per-window info (components removed, whether the fallback was used)."""
    X = x.T
    d, L = X.shape
    wl = int(np.floor(fs * 0.5 * d ** 2)) if wl_seconds is None else int(np.floor(wl_seconds * fs))
    wl = min(wl, L)
    fd_range = (min(2, d // 9), max(min(2, d // 9), d // 3))
    Y = X.copy()
    removed = []
    start = 0
    while start < L:
        stop = start + wl
        if stop > L:
            if stop - L > 0.5 * wl:
                break
            stop = L
        Y[:, start:stop], nrm = _clean_window(X[:, start:stop], fd_range)
        removed.append(nrm)
        start = stop
    if start < L and cover_tail:
        tail, nrm = _clean_window(X[:, L - wl:L], fd_range)
        Y[:, start:] = tail[:, start - (L - wl):]
        removed.append(nrm)
    return Y.T, removed
