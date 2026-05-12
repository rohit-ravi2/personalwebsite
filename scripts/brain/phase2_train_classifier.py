#!/usr/bin/env python3
"""Phase 2 — fresh classifier bank training under A2-balanced readout.

Per Phase 2 pre-flight (`docs/phase2_preflight.md`):
  Architecture: A2-balanced (21 cells) + B1 (per-event LogReg) + C1 (real Atanas only)
  + D2 (leave-one-worm-out 10-fold CV).

Readout (A2-balanced, ≥7-worm threshold):
  legacy 18 (cells in all 10 worms) + AVAL (9/10) + AVAR (8/10) + AVDL (9/10)
  = 21 cells total. Missing-cell handling: zero-fill (existing _pool_worm
  behavior preserved). PVC pair, AVB pair, AVDR (4/10) excluded.

Differs from `neural_classifier_bank.py`:
  - 21-neuron readout instead of 18 (legacy strict cross-worm intersection)
  - 10-fold leave-one-worm-out CV instead of 8/10-train + 2/10-test split
  - Reports mean ± SEM AUC across folds + Brier score + calibration metric
  - Output: classifier_bank_v2_a2balanced.npz (legacy not modified)

Reuses: spikes_to_calcium, calcium_kernel, build_features, _pool_worm,
_norm, EVENTS_FOR_BANK, EVENT_CONFIGS, CALCIUM_TAU_RISE, CALCIUM_TAU_DECAY
from neural_classifier_bank.py (production-grade infrastructure).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from neural_classifier_bank import (  # noqa: E402
    EVENTS_FOR_BANK,
    EVENT_CONFIGS,
    CALCIUM_TAU_RISE,
    CALCIUM_TAU_DECAY,
    _pool_worm,
    build_features,
)
from event_extraction import load_and_extract  # noqa: E402

ART = THIS_DIR / "artifacts"
BANK_OUT = ART / "classifier_bank_v2_a2balanced.npz"

# A2-balanced readout: legacy 18 (≥10 worms) + AVAL + AVAR + AVDL (≥7 worms)
LEGACY_18 = sorted([
    "AIBL", "ASEL", "AUAL", "AVEL", "AVER", "CEPDL", "I3", "IL2DL",
    "M3L", "M3R", "NSML", "NSMR", "OLQDL", "OLQDR", "OLQVL", "RMER",
    "SMDVL", "URXL",
])
ADDED_COMMANDS = ["AVAL", "AVAR", "AVDL"]
A2_BALANCED_READOUT = sorted(set(LEGACY_18) | set(ADDED_COMMANDS))

# Per-cell coverage (from Phase 2 pre-flight verification)
A2_COVERAGE = {
    nm: 10 for nm in LEGACY_18  # all in all 10 worms
}
A2_COVERAGE.update({"AVAL": 9, "AVAR": 8, "AVDL": 9})

C_GRID = (0.01, 0.1, 1.0, 10.0)


def calibration_error_uniform(y_true, y_prob, n_bins=10):
    """Uniform-bin calibration error (mean |confidence - accuracy| over bins)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    err_total = 0.0
    n_total = len(y_prob)
    n_used = 0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        err_total += abs(conf - acc) * mask.sum()
        n_used += mask.sum()
    return err_total / n_used if n_used > 0 else float("nan")


def train_one_event(event, X_per_worm, y_per_worm):
    """Train a per-event LogReg with leave-one-worm-out CV.

    Returns dict with: best C, per-fold AUC + Brier + calibration error,
    final-model weights/intercept (trained on all 10 worms with best C).
    """
    cfg = EVENT_CONFIGS[event]
    h = cfg["horizon"]
    fs = cfg["features"]

    # Build per-worm features + labels with horizon shift
    feats_per_worm = []
    targs_per_worm = []
    for X, y in zip(X_per_worm, y_per_worm):
        Xf = build_features(X, fs)
        if h > 0:
            Xf = Xf[:-h]
            yf = y[h:]
        else:
            yf = y
        feats_per_worm.append(Xf)
        targs_per_worm.append(yf)

    n_worms = len(feats_per_worm)

    # CV per C: leave one worm out, train on others, score on held-out
    cv_results = {}
    for C in C_GRID:
        per_fold = []
        for held_out_idx in range(n_worms):
            X_tr = np.concatenate(
                [f for i, f in enumerate(feats_per_worm) if i != held_out_idx]
            )
            y_tr = np.concatenate(
                [t for i, t in enumerate(targs_per_worm) if i != held_out_idx]
            )
            X_te = feats_per_worm[held_out_idx]
            y_te = targs_per_worm[held_out_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                per_fold.append({
                    "held_out_worm": held_out_idx + 1,
                    "auc": float("nan"),
                    "brier": float("nan"),
                    "cal_err": float("nan"),
                    "n_pos_train": int(y_tr.sum()),
                    "n_pos_test": int(y_te.sum()),
                })
                continue

            clf = LogisticRegression(
                C=C, max_iter=800, solver="liblinear",
                class_weight="balanced",
            ).fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, probs)
            brier = brier_score_loss(y_te, probs)
            cal = calibration_error_uniform(y_te, probs)
            per_fold.append({
                "held_out_worm": held_out_idx + 1,
                "auc": float(auc),
                "brier": float(brier),
                "cal_err": float(cal),
                "n_pos_train": int(y_tr.sum()),
                "n_pos_test": int(y_te.sum()),
            })

        valid = [f for f in per_fold if not np.isnan(f["auc"])]
        if not valid:
            cv_results[C] = {
                "per_fold": per_fold,
                "mean_auc": float("nan"),
                "sem_auc": float("nan"),
            }
            continue
        aucs = np.array([f["auc"] for f in valid])
        briers = np.array([f["brier"] for f in valid])
        cals = np.array([f["cal_err"] for f in valid])
        cv_results[C] = {
            "per_fold": per_fold,
            "n_folds_valid": len(valid),
            "mean_auc": float(aucs.mean()),
            "sem_auc": float(aucs.std(ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else 0.0,
            "mean_brier": float(briers.mean()),
            "sem_brier": float(briers.std(ddof=1) / np.sqrt(len(briers))) if len(briers) > 1 else 0.0,
            "mean_cal_err": float(cals.mean()),
        }

    # Pick best C by mean CV AUC (highest)
    best_C = None
    best_auc = -np.inf
    for C, res in cv_results.items():
        if not np.isnan(res["mean_auc"]) and res["mean_auc"] > best_auc:
            best_auc = res["mean_auc"]
            best_C = C

    if best_C is None:
        return {
            "event": event,
            "horizon": h,
            "features": fs,
            "cv_results": cv_results,
            "best_C": None,
            "weights": None,
            "intercept": None,
            "final_train_auc": None,
        }

    # Train final model on ALL worms with best C
    X_full = np.concatenate(feats_per_worm)
    y_full = np.concatenate(targs_per_worm)
    final_clf = LogisticRegression(
        C=best_C, max_iter=800, solver="liblinear",
        class_weight="balanced",
    ).fit(X_full, y_full)
    final_train_auc = roc_auc_score(
        y_full, final_clf.predict_proba(X_full)[:, 1]
    )

    return {
        "event": event,
        "horizon": h,
        "features": fs,
        "cv_results": cv_results,
        "best_C": best_C,
        "weights": final_clf.coef_[0].astype(np.float32),
        "intercept": float(final_clf.intercept_[0]),
        "final_train_auc": float(final_train_auc),
    }


def main():
    print("=" * 72)
    print("Phase 2 classifier training — A2-balanced readout (21 cells)")
    print("=" * 72)
    print(f"Readout cells ({len(A2_BALANCED_READOUT)}):")
    for nm in A2_BALANCED_READOUT:
        cov = A2_COVERAGE.get(nm, "?")
        marker = "★ legacy18" if nm in LEGACY_18 else "+ command"
        print(f"  {nm}: {cov}/10 worms  {marker}")
    print()
    print(f"Events ({len(EVENTS_FOR_BANK)}): {EVENTS_FOR_BANK}")
    print(f"CV: leave-one-worm-out 10-fold")
    print(f"C grid: {C_GRID}")
    print()

    # Load all 10 worms with A2-balanced readout
    worm_npzs = sorted(ART.glob("atanas_worm_*.npz"))
    if len(worm_npzs) != 10:
        print(f"WARNING: expected 10 worm npz files, got {len(worm_npzs)}")
    print(f"Loading {len(worm_npzs)} worms...")
    X_per_worm = []
    y_per_worm = {e: [] for e in EVENTS_FOR_BANK}
    for p in worm_npzs:
        X, tgts = _pool_worm(p, A2_BALANCED_READOUT)
        X_per_worm.append(X)
        for e in EVENTS_FOR_BANK:
            y_per_worm[e].append(tgts[e])
    print(f"Loaded. Each worm shape: {X_per_worm[0].shape}")

    # Train each event
    print()
    print(f"{'event':<22} {'h':>3} {'feat':<7} {'best C':>8} "
          f"{'CV AUC mean':>13} {'± SEM':>8} {'CV Brier':>10}")
    print("-" * 80)

    results = {}
    for event in EVENTS_FOR_BANK:
        res = train_one_event(event, X_per_worm, y_per_worm[event])
        results[event] = res
        if res["best_C"] is None:
            print(f"{event:<22} — NO VALID MODEL")
            continue
        cv = res["cv_results"][res["best_C"]]
        print(
            f"{event:<22} {res['horizon']:>3} {res['features']:<7} "
            f"{res['best_C']:>8.2f} {cv['mean_auc']:>13.3f} "
            f"{cv['sem_auc']:>8.3f} {cv['mean_brier']:>10.4f}"
        )

    # Save bank
    save_dict = {
        "neuron_order": np.array(A2_BALANCED_READOUT, dtype=object),
        "events": np.array(EVENTS_FOR_BANK, dtype=object),
        "sample_dt": np.float32(0.6),
        "calcium_tau_rise": np.float32(CALCIUM_TAU_RISE),
        "calcium_tau_decay": np.float32(CALCIUM_TAU_DECAY),
        "ar_lags": np.int32(3),
    }
    for e, res in results.items():
        if res["weights"] is None:
            continue
        save_dict[f"weights_{e}"] = res["weights"]
        save_dict[f"intercept_{e}"] = np.float32(res["intercept"])
        save_dict[f"horizon_{e}"] = np.int32(res["horizon"])
        save_dict[f"features_{e}"] = np.array(res["features"], dtype=object)
    np.savez_compressed(BANK_OUT, **save_dict)

    # JSON metadata with full CV details
    meta = {
        "readout": A2_BALANCED_READOUT,
        "readout_label": "A2-balanced (≥7-worm threshold)",
        "readout_coverage": A2_COVERAGE,
        "events": EVENTS_FOR_BANK,
        "cv_folds": 10,
        "cv_method": "leave-one-worm-out",
        "C_grid": list(C_GRID),
        "per_event": {
            e: {
                "horizon": res["horizon"],
                "features": res["features"],
                "best_C": res["best_C"],
                "final_train_auc": res["final_train_auc"],
                "cv_results": {
                    str(C): {
                        k: v for k, v in cv.items() if k != "per_fold"
                    }
                    for C, cv in res["cv_results"].items()
                },
                "per_fold_at_best_C": (
                    res["cv_results"][res["best_C"]]["per_fold"]
                    if res["best_C"] is not None else []
                ),
            }
            for e, res in results.items()
        },
    }
    json_path = BANK_OUT.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, default=str))

    print()
    print(f"Bank written: {BANK_OUT} ({BANK_OUT.stat().st_size/1024:.1f} KB)")
    print(f"Metadata:     {json_path}")
    print()

    # Summary diagnostics
    print("=" * 72)
    print("Summary — CV AUC vs legacy classifier")
    print("=" * 72)
    legacy_meta = json.loads((ART / "classifier_bank.json").read_text())
    print(f"{'event':<22} {'legacy train_auc':>16} {'v2 CV mean':>13} "
          f"{'v2 CV SEM':>11} {'Δ vs legacy':>13}")
    for e in EVENTS_FOR_BANK:
        if e not in results or results[e]["best_C"] is None:
            continue
        legacy_auc = legacy_meta["train_auc"].get(e, float("nan"))
        v2_auc = results[e]["cv_results"][results[e]["best_C"]]["mean_auc"]
        v2_sem = results[e]["cv_results"][results[e]["best_C"]]["sem_auc"]
        delta = v2_auc - legacy_auc
        print(
            f"{e:<22} {legacy_auc:>16.3f} {v2_auc:>13.3f} "
            f"{v2_sem:>11.3f} {delta:>+13.3f}"
        )
    print()
    print("Caveat: legacy 'train_auc' is fit on training set (overfits);")
    print("v2 'CV AUC' is mean held-out worm AUC. Direct comparison is biased")
    print("toward legacy looking better than it would on held-out data.")


if __name__ == "__main__":
    main()
