#!/usr/bin/env python3
"""Overnight Task 5 — 18-neuron readout peptidergic coverage audit.

For each modulator (9 existing + 5 T4-5 candidates), count how many
of its receptor-expressing target neurons are in the 18-neuron
classifier readout. Used to predict which modulators' ablation effects
propagate through the behavioral readout vs stay invisible.

Also: check overlap between the 18-neuron readout and Ripoll-Sánchez
2023 identified peptidergic broadcaster neurons (I1-I5, M5, NSM).

Output:
  task5_readout_coverage/coverage_table.csv
  task5_readout_coverage/readout_coverage_summary.md
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_modulator_tables import (
    load_cengen_matrix, SPECIAL_CLASS_MAP,
)
from overnight_task4_overlap_matrix import (
    MODULATOR_RECEPTORS, TPM_THRESHOLD,
)

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260421" / "task5_readout_coverage"
OUT_CSV = OUT_DIR / "coverage_table.csv"
OUT_MD = OUT_DIR / "readout_coverage_summary.md"

READOUT_18 = [
    "AIBL", "ASEL", "AUAL", "AVEL", "AVER", "CEPDL", "I3", "IL2DL",
    "M3L", "M3R", "NSML", "NSMR", "OLQDL", "OLQDR", "OLQVL", "RMER",
    "SMDVL", "URXL",
]

# Classes underlying the 18 connectome names (CeNGEN uses classes)
READOUT_18_CLASSES = [
    "AIB", "ASE", "AUA", "AVE", "CEP", "I3", "IL2",
    "M3", "NSM", "OLQ", "RME", "SMD", "URX",
]

# Ripoll-Sánchez 2023 identified peptidergic broadcaster neurons
PEPTIDERGIC_BROADCASTERS = ["I1", "I2", "I3", "I4", "I5", "M5", "NSM"]


def cengen_class_to_conn(cls: str) -> list[str]:
    if cls in SPECIAL_CLASS_MAP:
        return SPECIAL_CLASS_MAP[cls]
    return [cls + "L", cls + "R"]


def target_classes(df, sym_to_wb_ci, receptors):
    targets = set()
    for rec in receptors:
        wb = sym_to_wb_ci.get(rec.lower())
        if wb is None or wb not in df.columns:
            continue
        col = df[wb].astype(float)
        hits = col[col > TPM_THRESHOLD].index.tolist()
        targets.update(hits)
    return targets


def main():
    t0 = time.time()
    df, sym_to_wb = load_cengen_matrix()
    sym_to_wb_ci = {k.lower(): v for k, v in sym_to_wb.items()}

    # Broadcaster overlap with readout
    broadcaster_in_readout = []
    for bc_class in PEPTIDERGIC_BROADCASTERS:
        conns = cengen_class_to_conn(bc_class)
        for c in conns:
            if c in READOUT_18:
                broadcaster_in_readout.append((bc_class, c))

    rows = []
    for mod, recs in MODULATOR_RECEPTORS.items():
        ts_classes = target_classes(df, sym_to_wb_ci, recs)
        # Convert to connectome names
        ts_conn_names = set()
        for cls in ts_classes:
            for c in cengen_class_to_conn(cls):
                ts_conn_names.add(c)
        readout_targets = [n for n in READOUT_18 if n in ts_conn_names]
        # Predict Mode
        n_targets = len(ts_conn_names)
        n_readout = len(readout_targets)
        frac = n_readout / max(1, n_targets)
        if n_targets == 0:
            mode = "N/A (no targets detected)"
        elif n_readout == 0:
            mode = "Mode 1 (readout-blind)"
        elif frac > 0.5:
            mode = "Mode 3 predicted (readout-dominant targets)"
        elif frac > 0.15:
            mode = "Mode 3 possible (partial readout overlap)"
        else:
            mode = "Mode 1 (low readout overlap)"
        rows.append({
            "modulator": mod, "receptors": ", ".join(recs),
            "n_target_classes": len(ts_classes),
            "n_target_conn": n_targets,
            "n_in_readout": n_readout,
            "frac_in_readout": round(frac, 3),
            "readout_targets": ";".join(readout_targets) or "-",
            "predicted_mode": mode,
        })

    df_out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    lines = [
        "# Task 5 — 18-neuron readout peptidergic coverage",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "For each modulator, counts how many of its receptor-expressing ",
        "target neurons fall inside the 18-neuron classifier readout set. ",
        "This predicts which Mode the modulator's ablation would exhibit ",
        "behaviorally.",
        "",
        "## Readout-18 composition",
        "",
        f"Neurons: {', '.join(READOUT_18)}",
        "",
        "## Peptidergic broadcaster overlap",
        "",
        f"Ripoll-Sánchez 2023 peptidergic broadcasters: "
        f"{', '.join(PEPTIDERGIC_BROADCASTERS)}",
        "",
    ]
    if broadcaster_in_readout:
        lines.append("Broadcasters IN readout:")
        for bc, conn in broadcaster_in_readout:
            lines.append(f"- **{bc}** → {conn}")
    else:
        lines.append("No peptidergic broadcasters in readout.")
    lines.append("")
    lines.append("Paper implication: if most peptidergic broadcasters are "
                 "OUTSIDE the readout, this explains why peptidergic "
                 "ablations routinely produce behavioral nulls — the "
                 "simulator's readout architecture systematically excludes "
                 "the cells that carry the peptidergic signal.")
    lines.append("")

    lines.append("## Per-modulator Mode prediction")
    lines.append("")
    lines.append("| modulator | # conn targets | # in readout | frac | readout hits | predicted Mode |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| **{r['modulator']}** | {r['n_target_conn']} | "
            f"{r['n_in_readout']} | {r['frac_in_readout']:.2f} | "
            f"{r['readout_targets'][:40]} | {r['predicted_mode']} |"
        )
    lines.append("")

    # Summary statistics
    mode1 = sum(1 for r in rows if "Mode 1" in r["predicted_mode"])
    mode3 = sum(1 for r in rows if "Mode 3" in r["predicted_mode"])
    na = sum(1 for r in rows if "N/A" in r["predicted_mode"])
    lines.append("## Prediction summary")
    lines.append("")
    lines.append(f"- **Mode 1 (readout-blind) predicted:** {mode1}/{len(rows)}")
    lines.append(f"- **Mode 3 (readout-cascade) predicted:** {mode3}/{len(rows)}")
    lines.append(f"- **No targets detected:** {na}/{len(rows)}")
    lines.append("")
    lines.append("If D1's empirical Mode classification (Task 1) matches ")
    lines.append("this prediction table, the B4 readout-overlap predictor ")
    lines.append("can be used prospectively for any new modulator.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Total wall: {time.time()-t0:.1f}s")

    status_md = ART / "overnight_20260421" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Task 5: readout coverage\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: {mode1} modulators predicted Mode 1, "
                f"{mode3} predicted Mode 3. "
                f"Broadcasters in readout: "
                f"{len(broadcaster_in_readout)}\n")
        f.write(f"- Output: task5_readout_coverage/\n")


if __name__ == "__main__":
    main()
