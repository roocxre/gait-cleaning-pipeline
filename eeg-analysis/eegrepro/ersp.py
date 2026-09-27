"""ERD/ERS maps: port of the FFT branch of EEGLAB ``newtimef`` as called by
Dataset_{A,B,C}_ERSP.m:

    newtimef(chdata, frames, ival_epo, fs, 0, 'type','phasecoher',
             'freqs',[1 40], 'baseline',[-5000 -2000],
             'trialbase','full', 'rmerp','on')

Resulting behaviour (sources in reference/eeglab/):

* ``rmerp``: the across-trial mean (the evoked response) is subtracted from
  every trial, so only induced power remains.
* cycles = 0: FFT with a Hanning taper, window 2^(nextpow2(frames)-3) samples
  (2048 samples = 10.24 s for these epochs), zero-padded 2x (4096-point FFT,
  0.049 Hz bins), 200 output times spaced between tmin+win/2 and tmax-win/2.
* ``trialbase='full'``: each trial's power is divided by its mean over all
  output times (single-trial normalization); then the "standard baseline
  removal" divides by the mean over the baseline times of the *trial-averaged*
  power; trials are averaged and converted to dB.

Version matters.  The authors ran EEGLAB of 2016-17, whose newtimefbaseln
(reference/eeglab/newtimefbaseln_2015.m) takes the baseline from the
trial-averaged power whenever ``singletrials`` is on.  The current EEGLAB
develop branch (reference/eeglab/newtimefbaseln.m) computes it *per trial* for
``trialbase='full'``.  Dividing every trial by its own noisy single-bin baseline
inflates the mean ratio without bound (E[1/X] for a chi-square variable), which
put a spurious +3-4 dB on every frequency and time when tried on this data, so
the older behaviour is the one implemented.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann


def newtimef(data: np.ndarray, tlimits_ms: tuple[float, float], fs: float,
             freqs=(1.0, 40.0), baseline_ms=(-5000.0, -2000.0),
             n_times: int = 200, padratio: int = 2):
    """data: (frames, n_trials) one channel.  Returns (ersp_db (n_freq, n_times),
    times_ms, freqs_hz)."""
    frames, _ = data.shape
    data = data - data.mean(axis=1, keepdims=True)                       # rmerp
    winsize = int(2 ** (int(np.ceil(np.log2(frames))) - 3))
    nfft = padratio * winsize
    # gettimes: EEGLAB spreads `frames` points evenly over tlimits
    srate_t = 1000.0 * (frames - 1) / (tlimits_ms[1] - tlimits_ms[0])
    wintime = 500.0 * winsize / srate_t
    times = np.linspace(tlimits_ms[0] + wintime, tlimits_ms[1] - wintime, n_times)
    centre = np.round((times - tlimits_ms[0]) / 1000.0 * srate_t + 1).astype(int) - 1
    offsets = np.arange(-winsize // 2 + 1, winsize // 2 + 1)
    idx = centre[None, :] + offsets[:, None]                              # (win, n_times)
    fftf = np.linspace(0, fs / 2, nfft // 2 + 1)[1:]
    lo = fftf[np.argmin(np.abs(fftf - freqs[0]))]
    hi = fftf[np.argmin(np.abs(fftf - freqs[1]))]
    fsel = (fftf >= lo) & (fftf <= hi)
    win = hann(winsize + 2)[1:-1]                                          # MATLAB hanning()
    seg = data[idx]                                                        # (win, n_times, n_tr)
    seg = (seg - seg.mean(axis=0, keepdims=True)) * win[:, None, None]
    X = np.fft.rfft(seg, nfft, axis=0)[1:][fsel]                           # (n_f, n_times, n_tr)
    P = np.abs(2 / 0.375 * X / winsize) ** 2
    P = P / P.mean(axis=1, keepdims=True)                                  # trialbase 'full'
    base = (times >= baseline_ms[0]) & (times <= baseline_ms[1])
    if not base.any():
        raise ValueError("no output times inside the baseline window")
    mbase = P.mean(axis=2)[:, base].mean(axis=1, keepdims=True)            # trial-averaged baseline
    return 10 * np.log10(P.mean(axis=2) / mbase), times, fftf[fsel]
