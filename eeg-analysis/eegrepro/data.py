"""Loading the MATLAB-format EEG data, markers and behavioural summaries.

Conventions follow the BBCI toolbox that produced the files:

* ``cnt.x`` is samples x channels in microvolts at 200 Hz; channels 1-28 are EEG,
  29-30 are HEOG/VEOG.
* ``mrk.time`` is in milliseconds, where sample ``k`` (1-based) sits at
  ``k * 1000 / fs`` ms.  A marker therefore lands on 0-based sample
  ``ceil((t - eps) / si) - 1`` (BBCI ``proc_segmentation``).
* Each task file is three recording sessions concatenated with
  ``proc_appendCnt``.  ``cnt.T`` only stores the first session's length, so the
  other boundaries are recovered from the long marker gaps between sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Cleaned Data"
BEHAVIOR_DIR = ROOT / "behavior"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]
TASKS = ("nback", "dsr", "wg")
EEG_CLAB = ["Fp1", "AFF5h", "AFz", "F1", "FC5", "FC1", "T7", "C3", "Cz", "CP5",
            "CP1", "P7", "P3", "Pz", "POz", "O1", "Fp2", "AFF6h", "F2", "FC2",
            "FC6", "C4", "T8", "CP2", "CP6", "P4", "P8", "O2"]
EOG_CLAB = ["HEOG", "VEOG"]

# marker codes (Dataset description_MATLAB.pdf)
NBACK_SERIES = {112: 0, 128: 2, 144: 3}           # series marker -> n
NBACK_TRIAL = {16: (0, "target"), 48: (2, "target"), 64: (2, "non-target"),
               80: (3, "target"), 96: (3, "non-target")}
DSR_SERIES = 48
DSR_TRIAL = {16: "O", 32: "X"}                   # O = Go/target, X = No-go
WG_TRIAL = {16: "WG", 32: "BL"}

# a gap between consecutive markers longer than this separates two sessions
# (within a session: <= ~31 s; between sessions: ~58-60 s)
_SESSION_GAP_MS = 45_000
N_TRIALS = {"nback": 567, "dsr": 378, "wg": 60}   # markers per file incl. series


@dataclass
class Recording:
    subject: str
    task: str
    fs: float
    clab: list[str]
    x: np.ndarray            # (n_samples, n_channels) float64, microvolts
    mrk_ms: np.ndarray       # (n_markers,) int
    mrk_code: np.ndarray     # (n_markers,) int
    sessions: list[tuple[int, int]]   # [start, stop) sample ranges

    @property
    def mrk_sample(self) -> np.ndarray:
        si = 1000.0 / self.fs
        return (np.ceil((self.mrk_ms - si / 100) / si) - 1).astype(int)

    def channels(self, names: list[str]) -> np.ndarray:
        return np.array([self.clab.index(c) for c in names])


def _load_struct(path: Path, name: str):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)[name]


def load_raw(subject: str, task: str) -> Recording:
    """Load one subject/task exactly as stored (no processing)."""
    folder = DATA_DIR / f"{subject}-EEG"
    cnt = _load_struct(folder / f"cnt_{task}.mat", f"cnt_{task}")
    mrk = _load_struct(folder / f"mrk_{task}.mat", f"mrk_{task}")
    x = np.asarray(cnt.x, dtype=np.float64)
    rec = Recording(subject=subject, task=task, fs=float(cnt.fs),
                    clab=[str(c) for c in cnt.clab], x=x,
                    mrk_ms=np.asarray(mrk.time, dtype=np.int64),
                    mrk_code=np.asarray(mrk.event.desc, dtype=np.int64).ravel(),
                    sessions=[])
    if rec.clab != EEG_CLAB + EOG_CLAB or len(rec.mrk_ms) != N_TRIALS[task]:
        raise ValueError(f"{subject} {task}: unexpected channels or marker count")
    rec.sessions = _find_sessions(rec, int(np.atleast_1d(cnt.T)[0]))
    return rec


def load_clean(subject: str, task: str) -> Recording:
    """Paper-preprocessed data from scripts/01_preprocess.py (28 EEG channels)."""
    z = np.load(CACHE_DIR / "paper" / f"{subject}_{task}.npz")
    return Recording(subject=subject, task=task, fs=float(z["fs"]), clab=list(EEG_CLAB),
                     x=z["x"].astype(np.float64), mrk_ms=z["mrk_ms"], mrk_code=z["mrk_code"],
                     sessions=[tuple(s) for s in z["sessions"]])


def load_montage() -> tuple[np.ndarray, list[str]]:
    """2-D scalp positions (BBCI mnt.x / mnt.y) of the 28 EEG channels."""
    mnt = _load_struct(DATA_DIR / "VP001-EEG" / "mnt_nback.mat", "mnt_nback")
    clab = [str(c) for c in mnt.clab]
    idx = [clab.index(c) for c in EEG_CLAB]
    return np.column_stack([mnt.x, mnt.y])[idx], EEG_CLAB


def _find_sessions(rec: Recording, first_len: int) -> list[tuple[int, int]]:
    """Session boundaries.  Every session's first marker is exactly 30 s after
    the session start (verified for sessions 1-2 of all 78 files, where cnt.T
    gives the true boundary), so boundary = first marker - 30 s."""
    gaps = np.where(np.diff(rec.mrk_ms) > _SESSION_GAP_MS)[0]
    if len(gaps) != 2 or rec.mrk_ms[0] != 30_000:
        raise ValueError(f"{rec.subject} {rec.task}: unexpected session structure")
    si = 1000.0 / rec.fs
    bounds = [int(round((rec.mrk_ms[g + 1] - 30_000) / si)) for g in gaps]
    if bounds[0] != first_len:
        raise ValueError(f"{rec.subject} {rec.task}: boundary {bounds[0]} != cnt.T {first_len}")
    edges = [0, *bounds, rec.x.shape[0]]
    return [(edges[i], edges[i + 1]) for i in range(3)]


# --------------------------------------------------------------------------
# behaviour (behavior.zip from the authors' repository)
# --------------------------------------------------------------------------
def _behavior_dir(task: str) -> Path:
    return BEHAVIOR_DIR / {"nback": "n-back", "dsr": "dsr"}[task] / "summary"


@lru_cache(maxsize=None)
def load_behavior(subject: str, task: str) -> dict[str, np.ndarray]:
    """Per-trial behaviour in chronological order (series-major), aligned with the
    trial markers once series markers are dropped.

    result: 1 correct, 0 wrong button, -1 no response (the first n trials of each
    n-back series are always -1 because no answer is possible yet).
    """
    parts = [_load_struct(_behavior_dir(task) / f"{subject}-EEG" / f"summary{k}.mat",
                          f"summary{k}") for k in (1, 2, 3)]
    out = {key: np.concatenate([np.asarray(getattr(p, key)).reshape(-1) for p in parts])
           for key in ("result", "response", "reaction_time")}
    out["result"] = out["result"].astype(int)
    if task == "nback":
        out["digit"] = np.concatenate([np.asarray(p.flag).reshape(-1) for p in parts])
        out["series_n"] = np.concatenate([np.asarray(p.nback).ravel() for p in parts])
    else:
        out["go"] = np.concatenate([np.asarray(p.flag).reshape(-1) for p in parts])
    return out


# --------------------------------------------------------------------------
# trial tables
# --------------------------------------------------------------------------
@dataclass
class Trials:
    sample: np.ndarray       # 0-based marker sample
    label: np.ndarray        # str class label
    series: np.ndarray       # series (block) index, -1 for WG
    correct: np.ndarray      # bool (all True where behaviour does not apply)
    nback: np.ndarray | None = None   # n of the series (n-back only)

    def select(self, mask: np.ndarray) -> "Trials":
        return Trials(self.sample[mask], self.label[mask], self.series[mask],
                      self.correct[mask], None if self.nback is None else self.nback[mask])


def trial_table(rec: Recording) -> Trials:
    """Single-stimulus trials (series markers removed) with behavioural outcome."""
    code, samp = rec.mrk_code, rec.mrk_sample
    if rec.task == "wg":
        label = np.array([WG_TRIAL[c] for c in code])
        n = len(code)
        return Trials(samp, label, np.full(n, -1), np.ones(n, bool))

    is_series = np.isin(code, list(NBACK_SERIES) if rec.task == "nback" else [DSR_SERIES])
    series = np.cumsum(is_series) - 1
    keep = ~is_series
    beh = load_behavior(rec.subject, rec.task)
    if rec.task == "nback":
        label = np.array([NBACK_TRIAL[c][1] for c in code[keep]])
        nb = np.array([NBACK_TRIAL[c][0] for c in code[keep]])
        _check_nback_alignment(code[keep], beh)
    else:
        label = np.array([DSR_TRIAL[c] for c in code[keep]])
        nb = None
        if not np.array_equal(np.where(beh["go"] == 1, 16, 32), code[keep]):
            raise ValueError(f"{rec.subject}: DSR behaviour/marker mismatch")
    return Trials(samp[keep], label, series[keep], beh["result"] == 1, nb)


def _check_nback_alignment(codes: np.ndarray, beh: dict) -> None:
    """Re-derive target/non-target from the displayed digits and compare."""
    digits = beh["digit"].reshape(27, 20)
    expected = []
    for s, n in enumerate(beh["series_n"]):
        for j in range(20):
            if n == 0:
                expected.append(16)
            else:
                tgt = j >= n and digits[s, j - n] == digits[s, j]
                expected.append({2: (48, 64), 3: (80, 96)}[n][0 if tgt else 1])
    if not np.array_equal(expected, codes):
        raise ValueError("n-back behaviour/marker mismatch")


def series_table(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Task-onset markers used for ERD/ERS epochs: (sample, label)."""
    code, samp = rec.mrk_code, rec.mrk_sample
    if rec.task == "nback":
        m = np.isin(code, list(NBACK_SERIES))
        return samp[m], np.array([f"{NBACK_SERIES[c]}-back" for c in code[m]])
    if rec.task == "dsr":
        m = code == DSR_SERIES
        return samp[m], np.array(["DSR"] * int(m.sum()))
    return samp, np.array([WG_TRIAL[c] for c in code])
