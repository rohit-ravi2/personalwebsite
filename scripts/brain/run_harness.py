#!/usr/bin/env python3
"""Phase 3b multi-event evaluation harness.

Runs the full cross-product:
    worms × targets × horizons × feature-sets × models
with AR baselines, per-worm fits, and a cross-worm generalization
split (train on worms 1–8, test on 9–10) using the 59-neuron ≥80%-
intersection feature set.

Success criteria are tier-stratified and fixed in advance:
  Tier 1 (rare events):     AUC lift over AR ≥ +0.05
  Tier 2 (state transitions): AUC lift over AR ≥ +0.03
  Tier 3 (derivatives):     R²_neural ≥ 0.15 AND lift ≥ +0.08
  Tier 4 (multi-class):     accuracy ≥ majority + 10 pp
  Tier 5 (long horizon):    lift ≥ +0.08 over horizon-matched AR

Horizons tested: +1, +3, +8, +16 samples (~0.6, 1.8, 4.8, 9.6 s).

Output: artifacts/harness_results.csv + harness_summary.md
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

# Local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_extraction import load_and_extract, TARGET_META  # noqa: E402


ART = Path(__file__).resolve().parent / "artifacts"
OUT_CSV = ART / "harness_results.csv"
OUT_MD = ART / "harness_summary.md"

HORIZONS = [1, 3, 8, 16]
FEATURE_SETS = ["values", "lags", "derivs"]
ALPHAS_RIDGE = [0.1, 1.0, 10.0, 100.0, 1000.0]
CS_LOGIT = [0.01, 0.1, 1.0, 10.0]

EMBARGO_SECONDS = 60.0
SAMPLE_DT = 0.6  # ~1.67 Hz
EMBARGO_SAMPLES = int(round(EMBARGO_SECONDS / SAMPLE_DT))  # 100
TRAIN_FRAC = 0.70
AR_LAGS = 3

TIER_THRESHOLDS = {
    1: {"event_auc_lift": 0.05, "continuous_r2_lift": 0.05},
    2: {"event_auc_lift": 0.03, "continuous_r2_lift": 0.03},
    3: {"event_auc_lift": 0.05, "continuous_r2_floor": 0.15,
        "continuous_r2_lift": 0.08},
    4: {"accuracy_over_majority": 0.10},
    5: {"event_auc_lift": 0.08, "continuous_r2_lift": 0.08},
}


# ---------- Feature engineering ----------------------------------------

def _normalise(name: str) -> str:
    name = str(name).strip().rstrip("?")
    m = re.match(r"^([A-Za-z]+)0(\d)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def _align_worm_to_connectome(worm_npz: Path, conn_names: list[str],
                               restrict_to: set[str] | None = None
                               ) -> tuple[np.ndarray, list[str]]:
    a = np.load(worm_npz, allow_pickle=True)
    neural_raw = a["neural"]
    ids = [_normalise(s) for s in a["neuron_ids"]]
    idx_map = {n: i for i, n in enumerate(conn_names)}
    T = neural_raw.shape[0]

    columns = []
    neural_cols = []
    seen = set()
    for col, nm in enumerate(ids):
        if nm not in idx_map:
            continue
        if nm in seen:
            continue
        if restrict_to is not None and nm not in restrict_to:
            continue
        seen.add(nm)
        columns.append(nm)
        neural_cols.append(neural_raw[:, col])

    X = np.stack(neural_cols, axis=1).astype(np.float32) if neural_cols else \
        np.zeros((T, 0), np.float32)
    return X, columns


def build_features(X: np.ndarray, kind: str) -> np.ndarray:
    """Return smoothed+engineered feature matrix."""
    # 3-sample rolling mean to align calcium rise with behavior
    k = np.ones(3, dtype=np.float32) / 3
    X_smooth = np.stack(
        [np.convolve(X[:, i], k, mode="same") for i in range(X.shape[1])],
        axis=1,
    ) if X.shape[1] > 0 else X

    if kind == "values":
        return X_smooth
    if kind == "lags":
        # [X(t), X(t-1), X(t-2)]
        parts = [X_smooth]
        for lag in (1, 2):
            lagged = np.zeros_like(X_smooth)
            lagged[lag:] = X_smooth[:-lag]
            parts.append(lagged)
        return np.concatenate(parts, axis=1)
    if kind == "derivs":
        d = np.gradient(X_smooth, axis=0).astype(np.float32)
        return np.concatenate([X_smooth, d], axis=1)
    raise ValueError(f"unknown feature kind: {kind}")


def ar_features(y: np.ndarray, lags: int) -> np.ndarray:
    """AR(lags) design matrix. Shape (T - lags, lags)."""
    T = len(y)
    out = np.zeros((T - lags, lags), dtype=np.float32)
    for k in range(1, lags + 1):
        out[:, k - 1] = y[lags - k:T - k]
    return out


def _align_for_horizon(X_neural: np.ndarray, y: np.ndarray, horizon: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (neural_rows, ar_rows, y_target) aligned for +horizon prediction.

    For row r:
      ar_X[r] = [y(r+AR_LAGS-1), ..., y(r)]  (the AR_LAGS values up to
                                              position r+AR_LAGS-1)
      neural_X[r] = neural[r+AR_LAGS-1]
      y_target[r]  = y[r+AR_LAGS-1 + horizon]
    We throw away rows where the target index ≥ T.
    """
    T = len(y)
    ar = ar_features(y, AR_LAGS)          # (T - AR_LAGS, AR_LAGS)
    neural_rows = X_neural[AR_LAGS - 1:-1] if AR_LAGS - 1 > 0 else X_neural[:-1]
    # Row r of ar corresponds to predicting y[r + AR_LAGS - 1 + horizon]
    target_idx = np.arange(AR_LAGS - 1, T - horizon)
    n_rows = len(target_idx)
    ar_aligned = ar[:n_rows] if n_rows > 0 else ar
    n_aligned = X_neural[AR_LAGS - 1:AR_LAGS - 1 + n_rows]
    y_t = y[target_idx + horizon]
    return n_aligned, ar_aligned, y_t


def _split(T: int) -> tuple[slice, slice]:
    tr_end = int(T * TRAIN_FRAC)
    te_start = tr_end + EMBARGO_SAMPLES
    if te_start >= T - 5:
        te_start = tr_end + max(1, (T - tr_end) // 3)
    return slice(0, tr_end), slice(te_start, T)


def _best_ridge(X_tr, y_tr, X_te, y_te, alphas=ALPHAS_RIDGE):
    n = len(X_tr)
    v_cut = int(0.88 * n)
    if v_cut < 10 or n - v_cut < 5 or X_tr.shape[1] == 0:
        return float("nan"), float("nan")
    best = (-np.inf, None, None)
    for a in alphas:
        try:
            m = Ridge(alpha=a).fit(X_tr[:v_cut], y_tr[:v_cut])
            s = r2_score(y_tr[v_cut:], m.predict(X_tr[v_cut:]))
            if s > best[0]:
                best = (s, a, m)
        except Exception:
            continue
    if best[1] is None:
        return float("nan"), float("nan")
    final = Ridge(alpha=best[1]).fit(X_tr, y_tr)
    try:
        r2 = r2_score(y_te, final.predict(X_te))
    except Exception:
        r2 = float("nan")
    return r2, best[1]


def _best_logit(X_tr, y_tr, X_te, y_te, Cs=CS_LOGIT):
    if X_tr.shape[1] == 0 or len(np.unique(y_tr)) < 2:
        return float("nan"), float("nan"), float("nan")
    n = len(X_tr)
    v_cut = int(0.88 * n)
    if v_cut < 20 or n - v_cut < 5:
        return float("nan"), float("nan"), float("nan")
    best = (-np.inf, None, None)
    for C in Cs:
        try:
            clf = LogisticRegression(C=C, max_iter=500,
                                     solver="liblinear").fit(
                X_tr[:v_cut], y_tr[:v_cut])
            if len(np.unique(y_tr[v_cut:])) < 2:
                continue
            s = roc_auc_score(y_tr[v_cut:],
                              clf.predict_proba(X_tr[v_cut:])[:, 1])
            if s > best[0]:
                best = (s, C, clf)
        except Exception:
            continue
    if best[1] is None:
        return float("nan"), float("nan"), float("nan")
    try:
        final = LogisticRegression(
            C=best[1], max_iter=500, solver="liblinear"
        ).fit(X_tr, y_tr)
        if len(np.unique(y_te)) < 2:
            return float("nan"), float("nan"), float("nan")
        auc = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
        acc = accuracy_score(y_te, final.predict(X_te))
        return auc, acc, best[1]
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _best_multiclass(X_tr, y_tr, X_te, y_te, Cs=CS_LOGIT):
    if X_tr.shape[1] == 0:
        return float("nan"), float("nan"), float("nan")
    mask_tr = y_tr >= 0
    mask_te = y_te >= 0
    if mask_tr.sum() < 50 or mask_te.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_te, y_te = X_te[mask_te], y_te[mask_te]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), float("nan"), float("nan")
    n = len(X_tr)
    v_cut = int(0.88 * n)
    if v_cut < 30:
        return float("nan"), float("nan"), float("nan")
    best = (-np.inf, None)
    for C in Cs:
        try:
            clf = LogisticRegression(
                C=C, max_iter=500, solver="lbfgs",
                multi_class="multinomial"
            ).fit(X_tr[:v_cut], y_tr[:v_cut])
            s = accuracy_score(y_tr[v_cut:], clf.predict(X_tr[v_cut:]))
            if s > best[0]:
                best = (s, C)
        except Exception:
            continue
    if best[1] is None:
        return float("nan"), float("nan"), float("nan")
    try:
        final = LogisticRegression(
            C=best[1], max_iter=500, solver="lbfgs",
            multi_class="multinomial"
        ).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, final.predict(X_te))
        # majority baseline
        vals, counts = np.unique(y_tr, return_counts=True)
        maj = vals[np.argmax(counts)]
        maj_acc = accuracy_score(y_te, np.full_like(y_te, maj))
        return acc, maj_acc, best[1]
    except Exception:
        return float("nan"), float("nan"), float("nan")


# ---------- Per-(worm, target, horizon, featureset) evaluation --------

def eval_regression(neural_X, ar_X, y_target, tr_sl, te_sl):
    """Return row with AR-only, neural-only, combined R²s."""
    # AR-only
    ar_r2, ar_alpha = _best_ridge(
        ar_X[tr_sl], y_target[tr_sl], ar_X[te_sl], y_target[te_sl])
    # Neural-only
    n_r2, n_alpha = _best_ridge(
        neural_X[tr_sl], y_target[tr_sl], neural_X[te_sl], y_target[te_sl])
    # Combined
    comb_X = np.concatenate([ar_X, neural_X], axis=1)
    c_r2, c_alpha = _best_ridge(
        comb_X[tr_sl], y_target[tr_sl], comb_X[te_sl], y_target[te_sl])
    return dict(ar_score=ar_r2, neural_score=n_r2, combined_score=c_r2,
                neural_alpha=n_alpha, ar_alpha=ar_alpha, comb_alpha=c_alpha)


def eval_binary(neural_X, ar_X, y_target, tr_sl, te_sl):
    y_int = y_target.astype(int)
    if len(np.unique(y_int[tr_sl])) < 2 or len(np.unique(y_int[te_sl])) < 2:
        return dict(ar_score=float("nan"), neural_score=float("nan"),
                    combined_score=float("nan"), neural_alpha=float("nan"),
                    ar_alpha=float("nan"), comb_alpha=float("nan"),
                    neural_acc=float("nan"))
    ar_auc, ar_acc, _ = _best_logit(
        ar_X[tr_sl], y_int[tr_sl], ar_X[te_sl], y_int[te_sl])
    n_auc, n_acc, n_c = _best_logit(
        neural_X[tr_sl], y_int[tr_sl], neural_X[te_sl], y_int[te_sl])
    comb_X = np.concatenate([ar_X, neural_X], axis=1)
    c_auc, c_acc, _ = _best_logit(
        comb_X[tr_sl], y_int[tr_sl], comb_X[te_sl], y_int[te_sl])
    return dict(ar_score=ar_auc, neural_score=n_auc, combined_score=c_auc,
                neural_alpha=n_c, ar_alpha=float("nan"),
                comb_alpha=float("nan"), neural_acc=n_acc)


def eval_multiclass(neural_X, ar_X, y_target, tr_sl, te_sl):
    y_int = y_target.astype(int)
    # AR-only multiclass
    ar_acc, ar_maj, _ = _best_multiclass(
        ar_X[tr_sl], y_int[tr_sl], ar_X[te_sl], y_int[te_sl])
    n_acc, n_maj, _ = _best_multiclass(
        neural_X[tr_sl], y_int[tr_sl], neural_X[te_sl], y_int[te_sl])
    comb_X = np.concatenate([ar_X, neural_X], axis=1)
    c_acc, c_maj, _ = _best_multiclass(
        comb_X[tr_sl], y_int[tr_sl], comb_X[te_sl], y_int[te_sl])
    return dict(ar_score=ar_acc, neural_score=n_acc, combined_score=c_acc,
                majority=n_maj, neural_alpha=float("nan"),
                ar_alpha=float("nan"), comb_alpha=float("nan"))


def run_one_worm(worm_npz: Path, conn_names: list[str],
                 restrict_to: set[str] | None, worm_label: str) -> list[dict]:
    X_raw, used_names = _align_worm_to_connectome(
        worm_npz, conn_names, restrict_to
    )
    if X_raw.shape[1] == 0:
        return []
    data, targets = load_and_extract(worm_npz)
    T = X_raw.shape[0]

    rows = []
    for fs in FEATURE_SETS:
        X_full = build_features(X_raw, fs)
        for tgt_name, y_full in targets.items():
            tier, kind = TARGET_META[tgt_name]
            for h in HORIZONS:
                nX, aX, y_t = _align_for_horizon(X_full, y_full, h)
                T_eff = len(y_t)
                if T_eff < 200:
                    continue
                tr_sl, te_sl = _split(T_eff)
                row = dict(
                    worm=worm_label, target=tgt_name, tier=tier, kind=kind,
                    horizon=h, feature_set=fs, n_neurons=len(used_names),
                    n_train=tr_sl.stop - tr_sl.start,
                    n_test=te_sl.stop - te_sl.start,
                )
                try:
                    if kind == "continuous":
                        r = eval_regression(nX, aX, y_t, tr_sl, te_sl)
                    elif kind == "event" or kind == "state":
                        r = eval_binary(nX, aX, y_t, tr_sl, te_sl)
                    elif kind == "multiclass":
                        r = eval_multiclass(nX, aX, y_t, tr_sl, te_sl)
                    else:
                        continue
                    row.update(r)
                except Exception as e:
                    row["error"] = str(e)[:100]
                rows.append(row)
    return rows


def run_crossworm(worm_npzs: list[Path], conn_names: list[str],
                  intersect_set: set[str]) -> list[dict]:
    """Train on worms 1–8, test on 9–10 using the ≥80% intersection set."""
    rows = []
    # Load all worms, align to intersection feature set
    loaded = []
    for p in worm_npzs:
        X_raw, names = _align_worm_to_connectome(p, conn_names, intersect_set)
        if X_raw.shape[1] == 0:
            continue
        # Ensure same column order across worms by building with intersect order
        ordered = sorted(intersect_set)
        name_to_col = {n: i for i, n in enumerate(names)}
        col_idx = [name_to_col[n] for n in ordered if n in name_to_col]
        if len(col_idx) != len(ordered):
            continue
        X_aligned = X_raw[:, col_idx]
        data, targets = load_and_extract(p)
        loaded.append((X_aligned, targets, p.stem))

    if len(loaded) < 10:
        print(f"  cross-worm: only {len(loaded)} worms aligned, skipping")
        return rows

    # Worms 1-8 train, 9-10 test (by file sorting)
    train_X = np.concatenate([x for x, _, _ in loaded[:8]], axis=0)
    test_X = np.concatenate([x for x, _, _ in loaded[8:]], axis=0)
    train_T = train_X.shape[0]
    test_T = test_X.shape[0]

    for fs in FEATURE_SETS:
        Xtr = build_features(train_X, fs)
        Xte = build_features(test_X, fs)

        for tgt_name in TARGET_META.keys():
            tier, kind = TARGET_META[tgt_name]
            # Concatenate y across worms
            y_tr = np.concatenate([t[tgt_name] for _, t, _ in loaded[:8]])
            y_te = np.concatenate([t[tgt_name] for _, t, _ in loaded[8:]])
            for h in HORIZONS:
                # Align for horizon — simpler: just shift target
                if h >= len(y_tr) or h >= len(y_te):
                    continue
                y_tr_h = y_tr[h:]
                y_te_h = y_te[h:]
                Xtr_h = Xtr[:-h] if h > 0 else Xtr
                Xte_h = Xte[:-h] if h > 0 else Xte
                # AR features (using same-target history)
                ar_tr = ar_features(y_tr[:-h] if h > 0 else y_tr, AR_LAGS)
                ar_te = ar_features(y_te[:-h] if h > 0 else y_te, AR_LAGS)
                y_tr_align = y_tr_h[AR_LAGS:]
                y_te_align = y_te_h[AR_LAGS:]
                Xtr_align = Xtr_h[AR_LAGS:]
                Xte_align = Xte_h[AR_LAGS:]

                if len(y_tr_align) < 300 or len(y_te_align) < 30:
                    continue

                row = dict(
                    worm="CROSS", target=tgt_name, tier=tier, kind=kind,
                    horizon=h, feature_set=fs,
                    n_neurons=len(intersect_set),
                    n_train=len(y_tr_align), n_test=len(y_te_align),
                )
                try:
                    if kind == "continuous":
                        # AR-only
                        ar_r2, _ = _best_ridge(
                            ar_tr, y_tr_align, ar_te, y_te_align)
                        n_r2, _ = _best_ridge(
                            Xtr_align, y_tr_align, Xte_align, y_te_align)
                        comb_tr = np.concatenate([ar_tr, Xtr_align], axis=1)
                        comb_te = np.concatenate([ar_te, Xte_align], axis=1)
                        c_r2, _ = _best_ridge(
                            comb_tr, y_tr_align, comb_te, y_te_align)
                        row["ar_score"] = ar_r2
                        row["neural_score"] = n_r2
                        row["combined_score"] = c_r2
                    elif kind in ("event", "state"):
                        yi_tr = y_tr_align.astype(int)
                        yi_te = y_te_align.astype(int)
                        ar_auc, _, _ = _best_logit(
                            ar_tr, yi_tr, ar_te, yi_te)
                        n_auc, _, _ = _best_logit(
                            Xtr_align, yi_tr, Xte_align, yi_te)
                        comb_tr = np.concatenate([ar_tr, Xtr_align], axis=1)
                        comb_te = np.concatenate([ar_te, Xte_align], axis=1)
                        c_auc, _, _ = _best_logit(
                            comb_tr, yi_tr, comb_te, yi_te)
                        row["ar_score"] = ar_auc
                        row["neural_score"] = n_auc
                        row["combined_score"] = c_auc
                    elif kind == "multiclass":
                        yi_tr = y_tr_align.astype(int)
                        yi_te = y_te_align.astype(int)
                        ar_acc, _, _ = _best_multiclass(
                            ar_tr, yi_tr, ar_te, yi_te)
                        n_acc, n_maj, _ = _best_multiclass(
                            Xtr_align, yi_tr, Xte_align, yi_te)
                        comb_tr = np.concatenate([ar_tr, Xtr_align], axis=1)
                        comb_te = np.concatenate([ar_te, Xte_align], axis=1)
                        c_acc, _, _ = _best_multiclass(
                            comb_tr, yi_tr, comb_te, yi_te)
                        row["ar_score"] = ar_acc
                        row["neural_score"] = n_acc
                        row["combined_score"] = c_acc
                        row["majority"] = n_maj
                except Exception as e:
                    row["error"] = str(e)[:120]
                rows.append(row)
    return rows


def compute_intersection_set(worm_npzs, conn_names, min_worms=8):
    """Return set of neurons that appear in ≥min_worms/10 worms."""
    c = Counter()
    conn_set = set(conn_names)
    for p in worm_npzs:
        a = np.load(p, allow_pickle=True)
        nm = set(_normalise(s) for s in a["neuron_ids"])
        nm = nm & conn_set
        for n in nm:
            c[n] += 1
    return {n for n, k in c.items() if k >= min_worms}


def main() -> None:
    t0 = time.time()
    conn = np.load(ART / "connectome.npz", allow_pickle=True)
    conn_names = [str(s) for s in conn["names"]]
    worm_npzs = sorted(ART.glob("atanas_worm_*.npz"))
    print(f"Harness starting: {len(worm_npzs)} worms, "
          f"{len(TARGET_META)} targets, {len(HORIZONS)} horizons, "
          f"{len(FEATURE_SETS)} feature sets")

    # For cross-worm generalization we need a feature set present in
    # EVERY worm (otherwise columns shift). Use the strict all-10
    # intersection (≥10/10) — smaller but consistent.
    intersect = compute_intersection_set(worm_npzs, conn_names, min_worms=10)
    print(f"Cross-worm strict intersection (all 10/10): {len(intersect)} neurons")

    all_rows = []
    for p in worm_npzs:
        t_worm = time.time()
        label = p.stem.replace("atanas_", "")
        rows = run_one_worm(p, conn_names, restrict_to=None, worm_label=label)
        all_rows.extend(rows)
        print(f"  {label}: {len(rows)} rows in {time.time()-t_worm:.1f}s")

    print("\nCross-worm generalization (train 1-8, test 9-10)…")
    t_cw = time.time()
    cw_rows = run_crossworm(worm_npzs, conn_names, intersect)
    all_rows.extend(cw_rows)
    print(f"  CROSS: {len(cw_rows)} rows in {time.time()-t_cw:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows, "
          f"{OUT_CSV.stat().st_size/1024:.1f} KB)")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
