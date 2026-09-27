"""Checks of the MATLAB ports against independent formulations (no MATLAB here)."""
import numpy as np
from scipy.linalg import hankel, toeplitz
from scipy.signal import lfilter

from eegrepro import aar, bbci


def test_ar2r_satisfies_yule_walker():
    a = np.array([[1.0, -0.9, 0.2], [1.0, 0.5, 0.1]]).T          # two AR(2) models
    r = aar._ar2r(a)
    for j in range(a.shape[1]):
        # sum_i a_i r(|k-i|) = 0 for k >= 1 and = sigma^2 = 1 for k = 0
        for k in range(3):
            s = sum(a[i, j] * r[abs(k - i), j] for i in range(3))
            assert np.isclose(s, 1.0 if k == 0 else 0.0)


def test_thinv5_inverts_toeplitz_plus_hankel():
    rng = np.random.default_rng(0)
    K, M = 6, 4
    arcs = [np.concatenate([[1], rng.uniform(-0.3, 0.3, 2)]) for _ in range(2 * M)]
    ar3 = np.stack([np.convolve(arcs[2 * m], arcs[2 * m + 1])
                    for m in range(M)], axis=1)
    ar3 = np.vstack([ar3, np.zeros((2 * K - 1 - ar3.shape[0], M))])
    phi = aar._ar2r(ar3[:5])
    phi = np.vstack([phi, np.zeros((2 * K - 1 - phi.shape[0], M))])
    eps = 1e-3 * phi[0]
    H = aar._thinv5(phi, K, M, eps)
    for m in range(M):
        p = np.concatenate([phi[:, m], [0.0]])
        A = toeplitz(p[:K]) + hankel(p[:K], p[K - 1:2 * K - 1]) + eps[m] * np.eye(K)
        assert np.allclose(H[m] @ A, np.eye(K), atol=1e-8)


def test_iwasobi_separates_ar_sources():
    rng = np.random.default_rng(1)
    N, d = 20000, 4
    poles = [0.95, 0.5, -0.6, 0.0]
    S = np.stack([lfilter([1], [1, -p], rng.standard_normal(N)) for p in poles])
    A = rng.standard_normal((d, d))
    X = A @ S
    Wp, Xw = aar.pca_whiten(X)
    W = aar.iwasobi(Xw) @ Wp
    G = W @ A                               # should be scaled permutation
    G = np.abs(G) / np.abs(G).max(axis=1, keepdims=True)
    assert np.all(np.sort(G, axis=1)[:, -2] < 0.05)


def test_segmentation_matches_bbci_convention():
    x = np.arange(1000, dtype=float)[:, None]
    epo, t = bbci.segment(x, 200.0, np.array([100]), (-100, 1000))
    assert epo.shape == (220, 1, 1)
    assert t[0] == -95 and t[-1] == 1000
    assert epo[np.flatnonzero(t == 0)[0], 0, 0] == 100
    b = bbci.baseline(epo, t, (-100, 0))
    assert np.isclose(b[(t > -100) & (t <= 0)].mean(), 0)


def test_shrinkage_lda_separates():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((10, 200))
    y = np.repeat([0, 1], 100)
    X[0, y == 1] += 2
    clf = bbci.train_rlda_shrink(X, y)
    assert np.mean(clf.predict(X) == y) > 0.8
    assert clf.w[0] > 0


def test_kfold_is_stratified_partition():
    y = np.array([0] * 30 + [1] * 70)
    seen = []
    for tr, te in bbci.sample_kfold(y, 10, np.random.default_rng(0)):
        assert np.sum(y[te] == 0) == 3 and np.sum(y[te] == 1) == 7
        assert len(np.intersect1d(tr, te)) == 0
        seen.append(te)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(100))
