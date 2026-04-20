#!/usr/bin/env python3
"""Phase 3b step 1 — fit a brain-body interface from Atanas 2023
paired (neural, behavior) data.

**Phase 3b gate:** can we predict behavioral variables (velocity,
head curvature, reversal state) from the neural trace using a linear
or small-MLP regressor? If yes, there is a real neural→motor mapping
to learn — closed-loop Phase 3c is feasible. If no, the paired signal
is too noisy or we need a richer neural representation.

Fits:
  - linear regression:  velocity, head_curv, ang_vel  (continuous)
  - logistic regression: reversal (binary event)
  - tiny MLP (1 hidden):  velocity (nonlinearity check)

Split: 80/20 temporal (no shuffle — would leak autocorrelation).

Outputs:
  - artifacts/motor_interface.npz with:
      W_lin       (N_neurons, 4)  linear weights per behavioural var
      b_lin       (4,)            linear biases
      neuron_order (N,)           canonical connectome name per row
      W_rev_logit (N,), b_rev_logit (scalar)   logistic for reversal
  - Prints a table of held-out R² / accuracy so the user sees the gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score


ART = Path(__file__).resolve().parent / "artifacts"
CONN = ART / "connectome.npz"
ATANAS = ART / "atanas_worm_01.npz"
OUT = ART / "motor_interface.npz"


def _normalise_neuron(name: str) -> str:
    """Strip tentative `?` suffixes and zero-padding (VB02 → VB2)."""
    name = str(name).strip().rstrip("?")
    # Strip leading zero in NN → N form
    m = re.match(r"^([A-Za-z]+)0(\d)$", name)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return name


def _align_to_connectome() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load atanas neural, map its columns into connectome-name order.

    Returns:
        neural (T, N_conn) — neural columns placed at their connectome
            index, NaN elsewhere (we'll subset before fitting).
        valid_mask (N_conn,) — True where a column was filled.
        order (N_conn,) — canonical connectome neuron names.
    """
    conn = np.load(CONN, allow_pickle=True)
    conn_names = [str(s) for s in conn["names"]]
    name_to_idx = {n: i for i, n in enumerate(conn_names)}

    a = np.load(ATANAS, allow_pickle=True)
    neural_raw = a["neural"]  # (T, 134)
    ids = [_normalise_neuron(s) for s in a["neuron_ids"]]
    T, _ = neural_raw.shape
    N_conn = len(conn_names)

    neural = np.full((T, N_conn), np.nan, dtype=np.float32)
    valid = np.zeros(N_conn, dtype=bool)
    matched_names: list[str] = []
    duplicates: list[str] = []
    unmatched: list[str] = []

    for col, nm in enumerate(ids):
        if nm in name_to_idx:
            idx = name_to_idx[nm]
            if valid[idx]:
                duplicates.append(nm)
            else:
                neural[:, idx] = neural_raw[:, col]
                valid[idx] = True
                matched_names.append(nm)
        else:
            unmatched.append(nm)

    print(f"Matched {valid.sum()}/{len(ids)} Atanas ROIs → connectome")
    if unmatched:
        print(f"  unmatched (not in connectome): "
              f"{sorted(set(unmatched))[:10]}")
    if duplicates:
        print(f"  duplicate matches (ignored repeats): "
              f"{sorted(set(duplicates))[:10]}")
    return neural, valid, conn_names


def _time_split(X, y, frac=0.8):
    n = len(X)
    cut = int(frac * n)
    return X[:cut], X[cut:], y[:cut], y[cut:]


def main() -> None:
    print("=" * 60)
    print("Phase 3b gate: fit behavior from neural activity")
    print("=" * 60)

    neural, valid, conn_names = _align_to_connectome()

    # Subset to the valid (identified-in-connectome) columns for fitting.
    X = neural[:, valid]                         # (T, N_valid)
    valid_names = [conn_names[i] for i, v in enumerate(valid) if v]
    print(f"Design matrix: {X.shape}")

    # Behavior targets
    a = np.load(ATANAS, allow_pickle=True)
    targets = {
        "velocity":  a["velocity"],
        "head_curv": a["head_curv"],
        "body_curv": a["body_curv"],
        "ang_vel":   a["ang_vel"],
    }
    reversal = (a["reversal"] > 0.5).astype(np.int64)

    # Small temporal smoothing of neural trace — GCaMP7f is ~0.5 s
    # decay, so a 3-sample rolling mean lines up calcium with behavior
    # better than raw values.
    def smooth(z, w=3):
        k = np.ones(w, dtype=np.float32) / w
        return np.stack([np.convolve(z[:, i], k, mode="same")
                         for i in range(z.shape[1])], axis=1)
    Xs = smooth(X, w=3)

    print()
    print(f"{'target':<12} {'train_R²':>10} {'test_R²':>10}  "
          f"{'|w|_top3':<30}")
    print("-" * 68)

    W_lin = np.zeros((X.shape[1], len(targets)), dtype=np.float32)
    b_lin = np.zeros(len(targets), dtype=np.float32)

    for i, (name, y) in enumerate(targets.items()):
        Xtr, Xte, ytr, yte = _time_split(Xs, y, frac=0.8)
        model = Ridge(alpha=1.0)
        model.fit(Xtr, ytr)
        tr = r2_score(ytr, model.predict(Xtr))
        te = r2_score(yte, model.predict(Xte))
        # Top-3 contributing neurons by |weight|
        order = np.argsort(-np.abs(model.coef_))[:3]
        top = " ".join(f"{valid_names[j]}({model.coef_[j]:+.2f})"
                       for j in order)
        print(f"{name:<12} {tr:>10.3f} {te:>10.3f}  {top}")
        W_lin[:, i] = model.coef_.astype(np.float32)
        b_lin[i] = float(model.intercept_)

    # Reversal classification
    Xtr, Xte, ytr, yte = _time_split(Xs, reversal, frac=0.8)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(Xtr, ytr)
    acc_tr = accuracy_score(ytr, clf.predict(Xtr))
    acc_te = accuracy_score(yte, clf.predict(Xte))
    try:
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    except ValueError:
        auc = float("nan")
    order = np.argsort(-np.abs(clf.coef_[0]))[:5]
    top = " ".join(f"{valid_names[j]}({clf.coef_[0, j]:+.2f})"
                   for j in order)
    print(f"{'reversal':<12} acc_tr={acc_tr:.3f}  acc_te={acc_te:.3f}  "
          f"AUC={auc:.3f}")
    print(f"  top predictors: {top}")

    # MLP nonlinearity check for velocity
    Xtr, Xte, ytr, yte = _time_split(Xs, targets["velocity"], frac=0.8)
    mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500,
                       random_state=0, early_stopping=False, alpha=0.01)
    mlp.fit(Xtr, ytr)
    r_tr = r2_score(ytr, mlp.predict(Xtr))
    r_te = r2_score(yte, mlp.predict(Xte))
    print(f"\n{'velocity(MLP)':<14} train_R²={r_tr:.3f}  test_R²={r_te:.3f}")

    np.savez_compressed(
        OUT,
        W_lin=W_lin,
        b_lin=b_lin,
        target_names=np.array(list(targets.keys()), dtype=object),
        neuron_order=np.array(valid_names, dtype=object),
        valid_mask_in_connectome=valid,
        W_rev_logit=clf.coef_[0].astype(np.float32),
        b_rev_logit=np.array([float(clf.intercept_[0])], dtype=np.float32),
    )
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")

    # Gate summary
    print()
    print("=" * 60)
    best = max(r2_score(
        _time_split(Xs, y, frac=0.8)[3],
        Ridge(alpha=1.0).fit(
            *_time_split(Xs, y, frac=0.8)[:3:2],
        ).predict(_time_split(Xs, y, frac=0.8)[1])
    ) for y in targets.values())
    print(f"Phase 3b gate — best held-out R² across behavior vars: "
          f"{best:.3f}")
    if best > 0.3:
        print("→ PASS. Neural activity linearly predicts behavior above the")
        print("  gate threshold (R²>0.3). Closed-loop Phase 3c is viable.")
    else:
        print("→ FAIL. Linear fit below R²=0.3. Need nonlinear regressor,")
        print("  more data, or a different target representation.")


if __name__ == "__main__":
    main()
