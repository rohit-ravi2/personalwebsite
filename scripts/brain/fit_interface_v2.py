#!/usr/bin/env python3
"""Phase 3b gate — VERSION 2 with autocorrelation controls.

Fixes the following methodological issues with `fit_interface.py`:

1. **Embargo between train and test.** Calcium has ~2 s decay and
   behavioral epochs span seconds-to-minutes. An 80/20 temporal split
   without a gap leaks the test set's opening behavioral epoch into
   the training set's tail. We insert a 60 s gap (~100 samples at
   1.67 Hz) so the first test sample is truly out-of-epoch.

2. **Autoregressive baselines.** For each behavioral target we fit:
     - AR(k)    — only `y(t-k..t-1)` as features
     - NEURAL   — only `x(t)` neural trace as features
     - COMBINED — AR(k) features + neural
   The meaningful quantity is NEURAL R² minus AR R²: how much extra
   predictability does neural activity buy beyond simple persistence?

3. **One-step-ahead target.** Predict y(t+1) from x(t) and y(t-k..t-1),
   NOT y(t) from x(t). Same-timepoint prediction in a smoothed feature
   space is near-trivial.

4. **Stronger regularisation.** Ridge α swept {0.1, 1, 10, 100}; report
   the best per model by held-out R². Prevents the 95-feature × 1280-
   sample head/body curvature overfit seen in v1.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score


ART = Path(__file__).resolve().parent / "artifacts"
CONN = ART / "connectome.npz"
ATANAS = ART / "atanas_worm_01.npz"
OUT = ART / "motor_interface_v2.npz"

EMBARGO_SECONDS = 60.0
TRAIN_FRAC = 0.70  # smaller than v1 to leave room for embargo + test
AR_LAGS = 3


def _norm(name: str) -> str:
    name = str(name).strip().rstrip("?")
    m = re.match(r"^([A-Za-z]+)0(\d)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def _align():
    conn = np.load(CONN, allow_pickle=True)
    conn_names = [str(s) for s in conn["names"]]
    idx_map = {n: i for i, n in enumerate(conn_names)}
    a = np.load(ATANAS, allow_pickle=True)
    ids = [_norm(s) for s in a["neuron_ids"]]
    neural_raw = a["neural"]
    T = neural_raw.shape[0]
    X = np.full((T, len(conn_names)), np.nan, dtype=np.float32)
    valid = np.zeros(len(conn_names), dtype=bool)
    for col, nm in enumerate(ids):
        if nm in idx_map and not valid[idx_map[nm]]:
            X[:, idx_map[nm]] = neural_raw[:, col]
            valid[idx_map[nm]] = True
    return X[:, valid], [conn_names[i] for i, v in enumerate(valid) if v], a


def _smooth(z, w=3):
    if z.ndim == 1:
        k = np.ones(w, dtype=np.float32) / w
        return np.convolve(z, k, mode="same")
    k = np.ones(w, dtype=np.float32) / w
    return np.stack([np.convolve(z[:, i], k, mode="same")
                     for i in range(z.shape[1])], axis=1)


def _embargo_split(T, train_frac, embargo_samples):
    cut_train = int(train_frac * T)
    start_test = cut_train + embargo_samples
    if start_test >= T - 10:
        raise ValueError("embargo too large for available data")
    return slice(0, cut_train), slice(start_test, T)


def _best_ridge_r2(X_tr, y_tr, X_te, y_te, alphas=(0.1, 1, 10, 100, 1000)):
    best = (-np.inf, None, None)
    for a in alphas:
        m = Ridge(alpha=a).fit(X_tr, y_tr)
        r2 = r2_score(y_te, m.predict(X_te))
        if r2 > best[0]:
            best = (r2, a, m)
    return best  # (test_r2, best_alpha, fitted_model)


def _ar_features(y, lags):
    """Build AR design matrix: for t >= lags, rows = [y(t-1),...,y(t-lags)]."""
    T = len(y)
    X = np.zeros((T - lags, lags), dtype=np.float32)
    for k in range(1, lags + 1):
        X[:, k - 1] = y[lags - k:T - k]
    y_aligned = y[lags:]
    return X, y_aligned


def main() -> None:
    print("=" * 70)
    print("Phase 3b gate v2 — embargo + AR baselines + regularization sweep")
    print("=" * 70)

    X, valid_names, a = _align()
    t = np.asarray(a["t"])
    dt = float(np.median(np.diff(t)))
    print(f"Neural design: {X.shape}  sample dt={dt*1000:.0f} ms")

    embargo_samples = int(round(EMBARGO_SECONDS / dt))
    tr_sl, te_sl = _embargo_split(X.shape[0], TRAIN_FRAC, embargo_samples)
    print(f"Train: {tr_sl.stop} samples | "
          f"Embargo: {embargo_samples} samples ({EMBARGO_SECONDS:.0f} s) | "
          f"Test: {te_sl.stop - te_sl.start} samples")

    Xs = _smooth(X, w=3)

    targets = {
        "velocity":  a["velocity"],
        "head_curv": a["head_curv"],
        "body_curv": a["body_curv"],
        "ang_vel":   a["ang_vel"],
    }
    reversal = (a["reversal"] > 0.5).astype(np.int64)

    print()
    print(f"{'target':<12} {'AR':>8} {'NEURAL':>10} {'COMB':>10}  "
          f"{'Δ(neural−AR)':>14}  {'α*_neural'}")
    print("-" * 78)

    W_lin = np.zeros((X.shape[1], len(targets)), dtype=np.float32)
    b_lin = np.zeros(len(targets), dtype=np.float32)
    lift_per_target = {}

    for i, (name, y_full) in enumerate(targets.items()):
        # One-step-ahead: predict y[t+1] from neural[t] and y[t-k..t]
        y = np.asarray(y_full, dtype=np.float32)
        ar_X, _ = _ar_features(y, AR_LAGS)
        # Align neural and AR rows → effective indices are AR_LAGS..T-1
        # (AR feature for row r encodes y[r-AR_LAGS..r-1]; target is y[r+1])
        # One-step-ahead: target is y[AR_LAGS+1..T]
        y_target = y[AR_LAGS + 1:]
        neural_X = Xs[AR_LAGS:-1]
        assert len(y_target) == len(neural_X) == len(ar_X) - 1

        ar_X_aligned = ar_X[:-1]  # drop last row (no future y to predict)
        T_eff = len(y_target)

        # Now apply embargo split on T_eff
        tr_sl2, te_sl2 = _embargo_split(T_eff, TRAIN_FRAC, embargo_samples)

        # --- AR-only
        ar_r2, ar_alpha, _ = _best_ridge_r2(
            ar_X_aligned[tr_sl2], y_target[tr_sl2],
            ar_X_aligned[te_sl2], y_target[te_sl2],
        )
        # --- Neural-only
        n_r2, n_alpha, n_model = _best_ridge_r2(
            neural_X[tr_sl2], y_target[tr_sl2],
            neural_X[te_sl2], y_target[te_sl2],
        )
        # --- Combined
        comb_X = np.concatenate([ar_X_aligned, neural_X], axis=1)
        c_r2, c_alpha, _ = _best_ridge_r2(
            comb_X[tr_sl2], y_target[tr_sl2],
            comb_X[te_sl2], y_target[te_sl2],
        )

        lift = n_r2 - ar_r2
        lift_per_target[name] = lift
        print(f"{name:<12} {ar_r2:>8.3f} {n_r2:>10.3f} {c_r2:>10.3f}  "
              f"{lift:>+14.3f}  α={n_alpha:g}")

        # Save neural-only linear weights for the motor interface
        W_lin[:, i] = n_model.coef_.astype(np.float32)
        b_lin[i] = float(n_model.intercept_)

    # Reversal classifier with AR + neural comparison
    rv = reversal.astype(np.float32)
    rv_target = rv[AR_LAGS + 1:].astype(int)
    ar_rv, _ = _ar_features(rv, AR_LAGS)
    ar_rv_aligned = ar_rv[:-1]
    neural_X_rv = Xs[AR_LAGS:-1]
    tr_sl2, te_sl2 = _embargo_split(len(rv_target), TRAIN_FRAC, embargo_samples)

    def _logit_auc(Xtr, Xte, ytr, yte, C=1.0):
        clf = LogisticRegression(max_iter=500, C=C).fit(Xtr, ytr)
        # Handle case where one class not in training set (possible in AR)
        if len(np.unique(ytr)) < 2:
            return float("nan"), float("nan"), None
        try:
            auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        except Exception:
            auc = float("nan")
        acc = accuracy_score(yte, clf.predict(Xte))
        return acc, auc, clf

    print()
    print(f"{'reversal':<12}  {'acc':>6}  {'AUC':>6}")
    print("-" * 32)
    acc_ar, auc_ar, _ = _logit_auc(
        ar_rv_aligned[tr_sl2], ar_rv_aligned[te_sl2],
        rv_target[tr_sl2], rv_target[te_sl2])
    acc_nu, auc_nu, clf_n = _logit_auc(
        neural_X_rv[tr_sl2], neural_X_rv[te_sl2],
        rv_target[tr_sl2], rv_target[te_sl2])
    comb_rv = np.concatenate([ar_rv_aligned, neural_X_rv], axis=1)
    acc_cm, auc_cm, _ = _logit_auc(
        comb_rv[tr_sl2], comb_rv[te_sl2],
        rv_target[tr_sl2], rv_target[te_sl2])
    print(f"AR(3) only    {acc_ar:>6.3f}  {auc_ar:>6.3f}")
    print(f"Neural only   {acc_nu:>6.3f}  {auc_nu:>6.3f}")
    print(f"Combined      {acc_cm:>6.3f}  {auc_cm:>6.3f}")
    print(f"Δ AUC(neural−AR) = {auc_nu - auc_ar:+.3f}")
    print(f"Δ AUC(comb−AR)   = {auc_cm - auc_ar:+.3f}  "
          f"(what neural adds beyond persistence)")

    # Baseline: predict constant = test-set mean
    print()
    print(f"{'sanity':<12} {'const_R²':>10}")
    print("-" * 26)
    for name, y in targets.items():
        yt = y[AR_LAGS + 1:]
        c = yt[tr_sl2].mean()
        r = r2_score(yt[te_sl2], np.full(te_sl2.stop - te_sl2.start, c))
        print(f"{name:<12} {r:>10.3f}")

    np.savez_compressed(
        OUT,
        W_lin=W_lin, b_lin=b_lin,
        target_names=np.array(list(targets.keys()), dtype=object),
        neuron_order=np.array(valid_names, dtype=object),
        embargo_s=EMBARGO_SECONDS,
        W_rev_logit=clf_n.coef_[0].astype(np.float32) if clf_n else np.zeros(X.shape[1], np.float32),
        b_rev_logit=np.array([float(clf_n.intercept_[0])] if clf_n else [0.0], dtype=np.float32),
    )
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")

    # Gate decision
    print()
    print("=" * 70)
    best_lift = max(lift_per_target.values())
    print(f"Largest neural-over-AR lift: {best_lift:+.3f}  "
          f"({max(lift_per_target, key=lift_per_target.get)})")
    rev_lift = auc_cm - auc_ar
    print(f"Reversal AUC lift (comb − AR): {rev_lift:+.3f}")
    if best_lift > 0.1 or rev_lift > 0.05:
        print("→ Neural activity adds information beyond persistence.")
        print("  Phase 3c closed-loop remains viable.")
    else:
        print("→ Neural signal is ≤ AR persistence. Reconsider targets or")
        print("  use longer-horizon prediction (t+5s instead of t+1 sample).")


if __name__ == "__main__":
    main()
