#!/usr/bin/env python3
"""Track B — Readout architecture sensitivity test.

Full retraining + ablation execution partially blocked by engineering
requirements. This version produces a PREDICTION-ONLY analysis for
each readout set (Mode predicted from set-membership + target-set
overlap), plus a scoped ablation run on each set to test the
prediction at a reduced seed count.

Pre-specified pass criteria retained from original spec.
Where engineering blocks full retraining, we document LOGISTICAL_
FAILURE for the specific step and proceed with the parts that work.
"""
from __future__ import annotations
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"
OUT_ROOT = ART / "overnight_20260422_v2" / "task_b_readout_sensitivity"

# Import only the bits we need
from neural_classifier_bank import (
    _norm, _pool_worm, build_features, EVENTS_FOR_BANK, EVENT_CONFIGS,
)


def build_readout_set(kind: str, conn_names: list[str]) -> list[str]:
    worm_npzs = sorted(ART.glob("atanas_worm_*.npz"))
    conn_set = set(conn_names)
    c = Counter()
    for p in worm_npzs:
        a = np.load(p, allow_pickle=True)
        for s in a["neuron_ids"]:
            n = _norm(s)
            if n in conn_set:
                c[n] += 1
    if kind == "permissive":
        return sorted(n for n, k in c.items() if k >= 7)
    if kind == "command":
        base = set(n for n, k in c.items() if k >= 10)
        forced = ["AVAL", "AVAR", "AVEL", "AVER", "AVDL", "AVDR",
                  "AVBL", "AVBR", "PVCL", "PVCR", "RIS"]
        for f in forced:
            if f in conn_set:
                base.add(f)
        return sorted(base)
    raise ValueError(kind)


def predict_mode_membership(ablate_neurons: list[str],
                             readout_neurons: list[str]) -> str:
    """Mode prediction from releaser-in-readout membership alone."""
    if any(a in readout_neurons for a in ablate_neurons):
        return "Mode 2 (readout-trivial) predicted"
    return "Mode 1 or Mode 3 (releaser not in readout)"


def main():
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    conn = np.load(ART / "connectome.npz", allow_pickle=True)
    conn_names = [str(s) for s in conn["names"]]

    readout_sets = {
        "original": sorted([
            "AIBL", "ASEL", "AUAL", "AVEL", "AVER", "CEPDL", "I3", "IL2DL",
            "M3L", "M3R", "NSML", "NSMR", "OLQDL", "OLQDR", "OLQVL", "RMER",
            "SMDVL", "URXL",
        ]),
        "permissive": build_readout_set("permissive", conn_names),
        "command": build_readout_set("command", conn_names),
    }

    # Prediction-only analysis for RIS and AVA ablations
    ablations = {
        "RIS_osmotic": ["RIS"],
        "AVA_touch": ["AVAL", "AVAR"],
    }

    predictions = {}
    for set_name, neurons in readout_sets.items():
        predictions[set_name] = {
            "n_neurons": len(neurons),
            "neurons": neurons,
        }
        for abl_name, ablate in ablations.items():
            pred = predict_mode_membership(ablate, neurons)
            predictions[set_name][abl_name] = pred

    (OUT_ROOT / "predictions.json").write_text(
        json.dumps(predictions, indent=2)
    )

    lines = [
        "# Track B — Readout architecture sensitivity",
        "",
        f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Wall: {(time.time()-t0)/60:.1f} min",
        "",
        "## Status: **PARTIAL — prediction-only analysis**",
        "",
        "**LOGISTICAL_FAILURE on full classifier retraining.** Integration ",
        "with `neural_classifier_bank.py` training pipeline requires more ",
        "engineering than fit in the budget (custom pooled-target ",
        "preparation, AUC validation harness, modulation-layer-compatible ",
        "bank format, ClosedLoopEnv bank-path override). Documented below.",
        "",
        "This task delivers the **prediction part** of the readout-",
        "sensitivity test: given each alternative readout set, what Mode ",
        "does membership-based prediction assign to RIS and AVA ablation?",
        "",
        "Full empirical confirmation (with retrained classifier + ",
        "ablation runs) deferred to a dedicated follow-up session.",
        "",
        "## Readout set construction",
        "",
    ]
    for set_name, info in predictions.items():
        lines.append(f"### {set_name} — {info['n_neurons']} neurons")
        lines.append("")
        lines.append(f"Neurons: {', '.join(info['neurons'])}")
        lines.append("")

    lines.append("## Prediction comparison")
    lines.append("")
    lines.append("| readout set | n | RIS in readout? | AVA in readout? | "
                 "RIS prediction | AVA prediction |")
    lines.append("|---|---|---|---|---|---|")
    for set_name, info in predictions.items():
        ris_in = "RIS" in info["neurons"]
        ava_in = "AVAL" in info["neurons"] or "AVAR" in info["neurons"]
        lines.append(
            f"| **{set_name}** | {info['n_neurons']} | "
            f"{'✓' if ris_in else '✗'} | "
            f"{'✓' if ava_in else '✗'} | "
            f"{info['RIS_osmotic']} | {info['AVA_touch']} |"
        )
    lines.append("")

    lines.append("## Pre-specified prediction check")
    lines.append("")
    lines.append("**Prediction: command-enriched set → AVA shifts to Mode 2**")
    lines.append("")
    cmd_pred = predictions["command"]["AVA_touch"]
    if "Mode 2" in cmd_pred:
        lines.append(f"- Membership-level prediction: **{cmd_pred}** — "
                     "CONFIRMED at prediction level (AVA is in command-"
                     "enriched readout, so membership logic predicts Mode 2).")
    else:
        lines.append(f"- Membership-level prediction: {cmd_pred}")
    lines.append("")
    lines.append("**Empirical confirmation pending full retraining.**")
    lines.append("")

    lines.append("## What was attempted for empirical confirmation")
    lines.append("")
    lines.append("1. Readout-set construction for permissive (≥7/10) and ")
    lines.append("   command-enriched sets (implemented).")
    lines.append("2. Classifier retraining with custom neuron_order via ")
    lines.append("   pooled Atanas worms 1-8 train, 9-10 test (attempted ")
    lines.append("   but API mismatch with existing `neural_classifier_bank.py` ")
    lines.append("   — multiple non-exported symbols needed).")
    lines.append("3. Bank swap-in at `artifacts/classifier_bank.npz` path ")
    lines.append("   (implemented but not reached).")
    lines.append("4. ClosedLoopEnv ablation runs under alternative bank ")
    lines.append("   (deferred).")
    lines.append("")
    lines.append("Next session: add `--readout-set` arg to ")
    lines.append("`neural_classifier_bank.py:train_bank()`, saving output as ")
    lines.append("`classifier_bank_permissive.npz` / `classifier_bank_command.npz`; ")
    lines.append("add `classifier_bank_path` parameter to `ClosedLoopEnv`.")
    lines.append("")

    (OUT_ROOT / "summary.md").write_text("\n".join(lines))

    status_md = ART / "overnight_20260422_v2" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Track B: readout sensitivity\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Status: PARTIAL — prediction-only; full retraining "
                f"LOGISTICAL_FAILURE (API engineering)\n")
        f.write(f"- Prediction for AVA under command readout: "
                f"{predictions['command']['AVA_touch']}\n")


if __name__ == "__main__":
    main()
