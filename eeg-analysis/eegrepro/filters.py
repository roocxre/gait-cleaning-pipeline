"""Zero-phase Butterworth filtering.

MATLAB ``butter(3, band/fs*2, 'bandpass')`` + ``filtfilt`` gives a 6th-order
band-pass applied forward and backward (the paper's "6th order zero-phase
Butterworth").  Second-order sections are used for numerical stability at the
1 Hz / 200 Hz corner; the response is the same filter.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 3,
             sessions: list[tuple[int, int]] | None = None, axis: int = 0) -> np.ndarray:
    """Band-pass ``x`` along ``axis``.  With ``sessions`` each [start, stop) segment
    is filtered separately, so the concatenation jumps cannot ring into the data."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    if sessions is None:
        return sosfiltfilt(sos, x, axis=axis)
    if axis != 0:
        raise ValueError("per-session filtering expects time on axis 0")
    out = np.empty_like(x)
    for a, b in sessions:
        out[a:b] = sosfiltfilt(sos, x[a:b], axis=0)
    return out
