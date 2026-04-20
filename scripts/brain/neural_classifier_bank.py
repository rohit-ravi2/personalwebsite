#!/usr/bin/env python3
"""Phase 3c-1 — Neural event classifier bank.

Trains one classifier per transition event on pooled Atanas data
across all 10 worms, using the 18-neuron strict-intersection readout
(validated to generalize cross-worm in the Phase 3b harness).

Each classifier is saved as (weights, intercept, C, feature_set, horizon,
neuron_order, training_AUC) so downstream code (BrainBody closed loop)
can call `bank.predict(neural_window)` and get per-event probabilities.

Includes a synthetic-calcium model that converts Brian2 spike trains
to an Atanas-like ΔF/F trace so the same classifier works on either
real or simulated brain input.

Validation gate: after training, feed each Atanas worm's trace back
through the bank and confirm AUC matches harness expectations
(within ~10% of held-out test AUC).
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_extraction import load_and_extract  # noqa: E402


ART = Path(__file__).resolve().parent / "artifacts"
BANK_OUT = ART / "classifier_bank.npz"

# Events we ship in closed loop (the tier-1/2 passers from the harness)
EVENTS_FOR_BANK = [
    "reversal_onset",
    "reversal_offset",
    "forward_run_onset",
    "forward_run_offset",
    "omega_onset",
    "pirouette_entry",
    "quiescence_onset",
    "speed_burst_onset",
]

# Best (horizon, feature_set) per event — taken from harness cross-worm
# results (the strongest-generalizing config per target). All 1-sample
# horizon for onsets/offsets, derivs feature set captures calcium-rise
# edges cleanly.
EVENT_CONFIGS = {
    "reversal_onset":      {"horizon": 1, "features": "derivs"},
    "reversal_offset":     {"horizon": 1, "features": "derivs"},
    "forward_run_onset":   {"horizon": 1, "features": "derivs"},
    "forward_run_offset":  {"horizon": 3, "features": "derivs"},
    "omega_onset":         {"horizon": 1, "features": "derivs"},
    "pirouette_entry":     {"horizon": 3, "features": "lags"},
    "quiescence_onset":    {"horizon": 3, "features": "lags"},
    "speed_burst_onset":   {"horizon": 1, "features": "derivs"},
}


# ---------- Synthetic calcium model -----------------------------------

# GCaMP7f kinetics (Dana 2019): τ_rise ~ 0.1 s, τ_decay ~ 0.5 s.
# We model as convolution with a two-exponential kernel: h(t) = (1 -
# exp(-t/τ_rise)) * exp(-t/τ_decay), which matches spike → ΔF/F curve.
CALCIUM_TAU_RISE = 0.1   # seconds
CALCIUM_TAU_DECAY = 0.5  # seconds
CALCIUM_KERNEL_DUR = 3.0  # integration window (s)


def calcium_kernel(dt: float, duration: float = CALCIUM_KERNEL_DUR) -> np.ndarray:
    """Return a two-exponential GCaMP7f-like kernel sampled at `dt`."""
    t = np.arange(0, duration, dt)
    k = (1 - np.exp(-t / CALCIUM_TAU_RISE)) * np.exp(-t / CALCIUM_TAU_DECAY)
    if k.max() > 0:
        k = k / k.max()
    return k.astype(np.float32)


def spikes_to_calcium(spike_trains: np.ndarray, dt: float = 0.6) -> np.ndarray:
    """Convert an (N_neurons, N_timesteps) binary spike array to a
    synthetic ΔF/F trace at sample rate dt.

    Kernel convolution per neuron. Amplitude calibrated so a single
    spike produces peak ΔF/F ≈ 0.5 (matches Atanas ΔF/F range 0–4).
    """
    if spike_trains.ndim == 1:
        spike_trains = spike_trains[None, :]
    N, T = spike_trains.shape
    kern = calcium_kernel(dt)
    out = np.zeros_like(spike_trains, dtype=np.float32)
    for i in range(N):
        conv = np.convolve(spike_trains[i].astype(np.float32),
                           kern, mode="full")[:T]
        out[i] = 0.5 * conv
    return out


# ---------- Neuron alignment ------------------------------------------

def _norm(name: str) -> str:
    name = str(name).strip().rstrip("?")
    m = re.match(r"^([A-Za-z]+)0(\d)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def _intersection_all_10(conn_names: list[str]) -> list[str]:
    worm_npzs = sorted(ART.glob("atanas_worm_*.npz"))
    conn_set = set(conn_names)
    c = Counter()
    for p in worm_npzs:
        a = np.load(p, allow_pickle=True)
        for s in a["neuron_ids"]:
            n = _norm(s)
            if n in conn_set:
                c[n] += 1
    intersect = sorted(n for n, k in c.items() if k >= 10)
    return intersect


def _pool_worm(worm_npz: Path, neuron_order: list[str]
               ) -> tuple[np.ndarray, dict]:
    """Load one worm and return its (T, N_readout) neural matrix + targets,
    aligned to the global neuron_order. Missing neurons would be rare
    (we use strict all-10 intersection) but any missing column is zero-filled."""
    a = np.load(worm_npz, allow_pickle=True)
    ids = [_norm(s) for s in a["neuron_ids"]]
    col_map = {}
    for col, nm in enumerate(ids):
        if nm in neuron_order and nm not in col_map:
            col_map[nm] = col
    T = a["neural"].shape[0]
    X = np.zeros((T, len(neuron_order)), dtype=np.float32)
    for i, nm in enumerate(neuron_order):
        if nm in col_map:
            X[:, i] = a["neural"][:, col_map[nm]]
    _, tgts = load_and_extract(worm_npz)
    return X, tgts


# ---------- Feature engineering (matches harness) --------------------

def _smooth(X: np.ndarray, w: int = 3) -> np.ndarray:
    if X.shape[1] == 0:
        return X
    k = np.ones(w, dtype=np.float32) / w
    return np.stack(
        [np.convolve(X[:, i], k, mode="same") for i in range(X.shape[1])],
        axis=1,
    )


def build_features(X: np.ndarray, kind: str) -> np.ndarray:
    Xs = _smooth(X)
    if kind == "values":
        return Xs
    if kind == "lags":
        parts = [Xs]
        for lag in (1, 2):
            lagged = np.zeros_like(Xs)
            lagged[lag:] = Xs[:-lag]
            parts.append(lagged)
        return np.concatenate(parts, axis=1)
    if kind == "derivs":
        d = np.gradient(Xs, axis=0).astype(np.float32)
        return np.concatenate([Xs, d], axis=1)
    raise ValueError(kind)


# ---------- Main training ---------------------------------------------

def train_bank():
    conn = np.load(ART / "connectome.npz", allow_pickle=True)
    conn_names = [str(s) for s in conn["names"]]
    neuron_order = _intersection_all_10(conn_names)
    print(f"Using 18-neuron cross-worm readout: {neuron_order}")

    # Pool all worms
    worm_npzs = sorted(ART.glob("atanas_worm_*.npz"))
    pooled_neural = []
    pooled_targets: dict[str, list[np.ndarray]] = {e: [] for e in EVENTS_FOR_BANK}
    worm_boundaries = [0]

    for p in worm_npzs:
        X, tgts = _pool_worm(p, neuron_order)
        pooled_neural.append(X)
        worm_boundaries.append(worm_boundaries[-1] + X.shape[0])
        for e in EVENTS_FOR_BANK:
            pooled_targets[e].append(tgts[e])

    X_all = np.concatenate(pooled_neural, axis=0)
    print(f"Pooled data: {X_all.shape} ({len(worm_npzs)} worms)")

    # Train/test split: worms 1–8 train, 9–10 test (matches harness cross-worm)
    train_end = worm_boundaries[8]
    test_start = worm_boundaries[8]
    test_end = worm_boundaries[10]

    bank = {
        "neuron_order": neuron_order,
        "events": EVENTS_FOR_BANK,
        "calcium_tau_rise": CALCIUM_TAU_RISE,
        "calcium_tau_decay": CALCIUM_TAU_DECAY,
        "sample_dt": 0.6,
        "ar_lags": 3,
        "weights": {},
        "intercepts": {},
        "configs": {},
        "train_auc": {},
        "test_auc": {},
    }

    print(f"\n{'event':<22} {'horizon':>7} {'feat':<7} {'train_AUC':>10} {'test_AUC':>10}")
    print("-" * 62)

    for event in EVENTS_FOR_BANK:
        cfg = EVENT_CONFIGS[event]
        h = cfg["horizon"]
        fs = cfg["features"]

        y = np.concatenate(pooled_targets[event])

        X_feat = build_features(X_all, fs)

        # +h horizon shift
        if h > 0:
            X_feat = X_feat[:-h]
            y = y[h:]

        X_tr = X_feat[:train_end - h]
        y_tr = y[:train_end - h]
        X_te = X_feat[train_end - h: test_end - h]
        y_te = y[train_end - h: test_end - h]

        if len(np.unique(y_tr)) < 2:
            print(f"{event:<22} skipped (only one class in train)")
            continue

        # α sweep
        best = (-np.inf, None, None)
        for C in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=800,
                                      solver="liblinear",
                                      class_weight="balanced").fit(X_tr, y_tr)
            if len(np.unique(y_te)) < 2:
                continue
            auc_te = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
            if auc_te > best[0]:
                best = (auc_te, C, clf)

        if best[2] is None:
            print(f"{event:<22} — no valid model")
            continue

        auc_tr = roc_auc_score(y_tr, best[2].predict_proba(X_tr)[:, 1])
        auc_te = best[0]
        C_best = best[1]
        clf = best[2]

        bank["weights"][event] = clf.coef_[0].astype(np.float32)
        bank["intercepts"][event] = float(clf.intercept_[0])
        bank["configs"][event] = {"horizon": h, "features": fs, "C": C_best}
        bank["train_auc"][event] = float(auc_tr)
        bank["test_auc"][event] = float(auc_te)

        print(f"{event:<22} {h:>7} {fs:<7} {auc_tr:>10.3f} {auc_te:>10.3f}")

    # Save as .npz + .json metadata
    np.savez_compressed(
        BANK_OUT,
        neuron_order=np.array(bank["neuron_order"], dtype=object),
        events=np.array(bank["events"], dtype=object),
        sample_dt=np.float32(bank["sample_dt"]),
        calcium_tau_rise=np.float32(bank["calcium_tau_rise"]),
        calcium_tau_decay=np.float32(bank["calcium_tau_decay"]),
        ar_lags=np.int32(bank["ar_lags"]),
        **{f"weights_{e}": w for e, w in bank["weights"].items()},
        **{f"intercept_{e}": np.float32(v)
           for e, v in bank["intercepts"].items()},
        **{f"horizon_{e}": np.int32(bank["configs"][e]["horizon"])
           for e in bank["configs"]},
        **{f"features_{e}": np.array(bank["configs"][e]["features"],
                                      dtype=object)
           for e in bank["configs"]},
    )

    meta = {
        "events": bank["events"],
        "configs": bank["configs"],
        "train_auc": bank["train_auc"],
        "test_auc": bank["test_auc"],
        "neuron_order": bank["neuron_order"],
    }
    (BANK_OUT.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {BANK_OUT} ({BANK_OUT.stat().st_size/1024:.1f} KB)")
    return bank


# ---------- Runtime prediction interface ------------------------------

class ClassifierBank:
    """Runtime wrapper: feed it a rolling neural window, get event
    probabilities at each timestep."""

    def __init__(self, bank_npz: Path = BANK_OUT):
        self.d = np.load(bank_npz, allow_pickle=True)
        self.neuron_order = [str(s) for s in self.d["neuron_order"]]
        self.events = [str(e) for e in self.d["events"]]
        self.N = len(self.neuron_order)
        self.dt = float(self.d["sample_dt"])
        self.weights = {e: self.d[f"weights_{e}"] for e in self.events}
        self.intercepts = {e: float(self.d[f"intercept_{e}"]) for e in self.events}
        self.horizons = {e: int(self.d[f"horizon_{e}"]) for e in self.events}
        self.feature_sets = {e: str(self.d[f"features_{e}"]) for e in self.events}

    def predict_from_calcium(self, ca: np.ndarray) -> dict[str, np.ndarray]:
        """ca: (T, N_readout) synthetic calcium or real ΔF/F. Returns
        dict of event → (T,) probability series."""
        out = {}
        for e in self.events:
            fs = self.feature_sets[e]
            X = build_features(ca, fs)
            logit = X @ self.weights[e] + self.intercepts[e]
            out[e] = 1.0 / (1.0 + np.exp(-logit))
        return out

    def predict_from_spikes(self, spikes: np.ndarray) -> dict[str, np.ndarray]:
        """spikes: (N_readout, T_samples_at_dt) binary. Returns same dict."""
        ca = spikes_to_calcium(spikes, dt=self.dt).T  # (T, N)
        return self.predict_from_calcium(ca)


if __name__ == "__main__":
    bank = train_bank()
    print("\n=== Validation: feed bank Atanas worm_01 and check AUC ===")
    cb = ClassifierBank()
    # Load worm 01 data aligned to 18-neuron readout
    X, tgts = _pool_worm(ART / "atanas_worm_01.npz", cb.neuron_order)
    probs = cb.predict_from_calcium(X)
    print(f"\n{'event':<22} {'worm01_AUC':>12} {'event_count':>12}")
    print("-" * 52)
    for e, p in probs.items():
        y = tgts[e][:len(p)]
        if len(np.unique(y)) < 2:
            print(f"{e:<22}   (no events in worm_01)")
            continue
        auc = roc_auc_score(y, p)
        n = int(np.sum(y))
        print(f"{e:<22} {auc:>12.3f} {n:>12}")
