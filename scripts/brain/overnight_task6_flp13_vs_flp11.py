#!/usr/bin/env python3
"""Overnight Task 6 — FLP-13 vs FLP-11 target comparison.

Focused side-by-side analysis. FLP-13 is the new T4-5 candidate for
the ALA sleep pathway; FLP-11 is the existing RIS sleep peptide. Are
they parallel (distinct target sets, coordinated effects) or redundant
(largely same targets)?

Output:
  task6_flp13_vs_flp11/comparison.csv
  task6_flp13_vs_flp11/comparison_summary.md
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

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260421" / "task6_flp13_vs_flp11"
OUT_CSV = OUT_DIR / "comparison.csv"
OUT_MD = OUT_DIR / "comparison_summary.md"

TPM_THRESHOLD = 0.5

FLP_11_RECEPTORS = ["npr-1", "npr-22", "dmsr-1", "dmsr-7", "npr-11"]
FLP_13_RECEPTORS = ["dmsr-1", "dmsr-2"]

READOUT_18 = [
    "AIBL", "ASEL", "AUAL", "AVEL", "AVER", "CEPDL", "I3", "IL2DL",
    "M3L", "M3R", "NSML", "NSMR", "OLQDL", "OLQDR", "OLQVL", "RMER",
    "SMDVL", "URXL",
]


def cengen_class_to_conn(cls: str) -> list[str]:
    if cls in SPECIAL_CLASS_MAP:
        return SPECIAL_CLASS_MAP[cls]
    return [cls + "L", cls + "R"]


def target_classes(df, sym_to_wb_ci, receptors):
    targets = {}
    for rec in receptors:
        wb = sym_to_wb_ci.get(rec.lower())
        if wb is None or wb not in df.columns:
            continue
        col = df[wb].astype(float)
        for cls, tpm in col.items():
            if tpm > TPM_THRESHOLD:
                targets.setdefault(cls, []).append((rec, round(float(tpm), 2)))
    return targets


def main():
    t0 = time.time()
    df, sym_to_wb = load_cengen_matrix()
    sym_to_wb_ci = {k.lower(): v for k, v in sym_to_wb.items()}

    flp11_targets = target_classes(df, sym_to_wb_ci, FLP_11_RECEPTORS)
    flp13_targets = target_classes(df, sym_to_wb_ci, FLP_13_RECEPTORS)

    flp11_set = set(flp11_targets.keys())
    flp13_set = set(flp13_targets.keys())

    both = flp11_set & flp13_set
    only_11 = flp11_set - flp13_set
    only_13 = flp13_set - flp11_set

    jaccard = len(both) / max(1, len(flp11_set | flp13_set))

    rows = []
    for cls in sorted(flp11_set | flp13_set):
        in_11 = cls in flp11_set
        in_13 = cls in flp13_set
        r11 = ";".join(f"{r}={t}" for r, t in flp11_targets.get(cls, []))
        r13 = ";".join(f"{r}={t}" for r, t in flp13_targets.get(cls, []))
        conn_names = cengen_class_to_conn(cls)
        in_readout = any(c in READOUT_18 for c in conn_names)
        rows.append({
            "neuron_class": cls,
            "flp11_target": in_11,
            "flp13_target": in_13,
            "flp11_receptors": r11,
            "flp13_receptors": r13,
            "in_readout": in_readout,
        })
    df_out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Readout coverage specifically
    readout_11 = [cls for cls in flp11_set
                  if any(c in READOUT_18 for c in cengen_class_to_conn(cls))]
    readout_13 = [cls for cls in flp13_set
                  if any(c in READOUT_18 for c in cengen_class_to_conn(cls))]

    lines = [
        "# Task 6 — FLP-13 vs FLP-11 target comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"FLP-11 receptors: {', '.join(FLP_11_RECEPTORS)}",
        f"FLP-13 receptors: {', '.join(FLP_13_RECEPTORS)} "
        f"(per Nath 2016 ALA sleep pathway)",
        f"TPM threshold for target detection: {TPM_THRESHOLD}",
        "",
        "## Target-set comparison",
        "",
        f"- **FLP-11 targets:** {len(flp11_set)} classes "
        f"({', '.join(sorted(flp11_set))})",
        f"- **FLP-13 targets:** {len(flp13_set)} classes "
        f"({', '.join(sorted(flp13_set))})",
        f"- **Shared targets:** {len(both)} classes "
        f"({', '.join(sorted(both)) if both else 'none'})",
        f"- **FLP-11 only:** {len(only_11)} classes "
        f"({', '.join(sorted(only_11)) if only_11 else 'none'})",
        f"- **FLP-13 only:** {len(only_13)} classes "
        f"({', '.join(sorted(only_13)) if only_13 else 'none'})",
        f"- **Jaccard overlap:** {jaccard:.2f}",
        "",
        "## Verdict",
        "",
    ]
    if jaccard > 0.7:
        lines.append(f"- **REDUNDANT** (Jaccard = {jaccard:.2f}) — FLP-13 "
                     "and FLP-11 target largely the same neurons. "
                     "Adding FLP-13 to T4-5 may not provide distinct "
                     "empirical coverage.")
    elif jaccard > 0.3:
        lines.append(f"- **PARTIAL OVERLAP** (Jaccard = {jaccard:.2f}) — "
                     "FLP-13 and FLP-11 share some targets. Adding FLP-13 "
                     "adds distinct coverage on the non-overlapping targets "
                     "but doesn't fully decouple the pathways.")
    else:
        lines.append(f"- **DISTINCT PATHWAYS** (Jaccard = {jaccard:.2f}) — "
                     "FLP-13 and FLP-11 target largely different neurons. "
                     "Supports the 'parallel sleep pathways' framing "
                     "(RIS-FLP-11 and ALA-FLP-13 as separate modules).")

    lines.append("")
    lines.append("## Readout-18 coverage")
    lines.append("")
    lines.append(f"- FLP-11 target classes in readout: "
                 f"{len(readout_11)} ({', '.join(readout_11) if readout_11 else 'none'})")
    lines.append(f"- FLP-13 target classes in readout: "
                 f"{len(readout_13)} ({', '.join(readout_13) if readout_13 else 'none'})")
    lines.append("")

    lines.append("## Per-class target table")
    lines.append("")
    lines.append("| class | FLP-11 | FLP-13 | FLP-11 rec (TPM) | FLP-13 rec (TPM) | in readout |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df_out.iterrows():
        lines.append(
            f"| {r['neuron_class']} | {'✓' if r['flp11_target'] else '-'} | "
            f"{'✓' if r['flp13_target'] else '-'} | "
            f"{r['flp11_receptors'][:30]} | "
            f"{r['flp13_receptors'][:30]} | "
            f"{'✓' if r['in_readout'] else '-'} |"
        )
    lines.append("")
    lines.append("## T4-5 implication")
    lines.append("")
    if jaccard < 0.3:
        lines.append("- FLP-13 is a DEFENSIBLE T4-5 addition: targets "
                     "are non-redundant with FLP-11's.")
    elif jaccard > 0.7:
        lines.append("- FLP-13 is LIKELY REDUNDANT with FLP-11 at current "
                     "receptor-coverage level. Reconsider inclusion or "
                     "add only if phenotype evidence distinguishes them.")
    else:
        lines.append("- FLP-13 provides partial distinct coverage. "
                     "Including adds marginal information; justified only "
                     "if the phenotype evidence shows dissociable effects.")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Total wall: {time.time()-t0:.1f}s")

    status_md = ART / "overnight_20260421" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Task 6: FLP-13 vs FLP-11\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: Jaccard = {jaccard:.2f}, "
                f"FLP-11 unique={len(only_11)}, "
                f"FLP-13 unique={len(only_13)}, "
                f"shared={len(both)}\n")
        f.write(f"- Output: task6_flp13_vs_flp11/\n")


if __name__ == "__main__":
    main()
