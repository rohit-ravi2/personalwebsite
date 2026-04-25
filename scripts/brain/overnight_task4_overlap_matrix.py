#!/usr/bin/env python3
"""Overnight Task 4 — 14×14 modulator-target overlap matrix.

For 9 existing modulators + 5 locked T4-5 candidates, compute target
neuron sets (neurons expressing the relevant receptor above threshold)
and pairwise Jaccard overlap.

Output:
  task4_overlap_matrix/overlap_matrix.csv
  task4_overlap_matrix/overlap_summary.md
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
    load_gene_symbol_map, load_cengen_matrix, SPECIAL_CLASS_MAP,
)

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260421" / "task4_overlap_matrix"
OUT_CSV = OUT_DIR / "overlap_matrix.csv"
OUT_MD = OUT_DIR / "overlap_summary.md"

# Receptor mRNA is much lower abundance than peptide mRNA — use lower
# threshold for target-set construction. CeNGEN Taylor 2021's fixed
# threshold=4 is biased to peptide-magnitude genes; receptors at
# functional density typically sit at 0.3-3 TPM.
TPM_THRESHOLD = 0.5

# Modulator → receptor gene symbols (from biology literature)
MODULATOR_RECEPTORS = {
    # Existing 9
    "FLP-11": ["npr-1", "npr-22", "dmsr-1", "dmsr-7", "npr-11"],
    "FLP-1":  ["npr-4", "npr-5", "npr-11"],
    "FLP-2":  ["npr-30", "frpr-18"],
    "NLP-12": ["ckr-1", "ckr-2"],
    "PDF-1":  ["pdfr-1"],
    "5HT":    ["mod-1", "ser-1", "ser-4", "ser-5", "ser-6", "ser-7"],
    "DA":     ["dop-1", "dop-2", "dop-3", "dop-4"],
    "TA":     ["tyra-2", "tyra-3", "ser-2", "lgc-55"],
    "OA":     ["octr-1", "ser-3", "ser-6"],
    # 5 T4-5 candidates
    "FLP-13":  ["dmsr-1", "dmsr-2"],
    "FLP-18":  ["npr-1", "npr-4", "npr-5"],
    "FLP-21":  ["npr-1"],
    "NLP-40":  ["aex-2"],
    "DAF-28":  ["daf-2"],
}


def cengen_class_to_conn(cls: str) -> list[str]:
    if cls in SPECIAL_CLASS_MAP:
        return SPECIAL_CLASS_MAP[cls]
    return [cls + "L", cls + "R"]


def target_set(df, sym_to_wb_ci, receptors) -> set[str]:
    """Return set of neuron classes expressing any receptor above threshold."""
    targets = set()
    for rec in receptors:
        wb = sym_to_wb_ci.get(rec.lower())
        if wb is None or wb not in df.columns:
            continue
        col = df[wb].astype(float)
        hits = col[col > TPM_THRESHOLD].index.tolist()
        targets.update(hits)
    return targets


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    t0 = time.time()
    print("Loading CeNGEN...")
    df, sym_to_wb = load_cengen_matrix()
    sym_to_wb_ci = {k.lower(): v for k, v in sym_to_wb.items()}

    # Build target sets
    targets = {}
    for mod, recs in MODULATOR_RECEPTORS.items():
        ts = target_set(df, sym_to_wb_ci, recs)
        targets[mod] = ts
        print(f"  {mod:8s} receptors={recs} → {len(ts)} target classes")

    # Pairwise Jaccard
    mod_names = list(MODULATOR_RECEPTORS.keys())
    n = len(mod_names)
    J = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(mod_names):
        for j, b in enumerate(mod_names):
            J[i, j] = jaccard(targets[a], targets[b])

    df_J = pd.DataFrame(J, index=mod_names, columns=mod_names)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_J.to_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV}")

    # Flag high-overlap pairs
    high_overlaps = []
    distinct_pairs = []
    for i, a in enumerate(mod_names):
        for j, b in enumerate(mod_names):
            if i >= j:
                continue
            v = float(J[i, j])
            if v > 0.7:
                high_overlaps.append((a, b, v))
            if v < 0.1:
                distinct_pairs.append((a, b, v))

    lines = [
        "# Task 4 — Modulator target-set overlap matrix",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"14 modulators (9 existing + 5 T4-5 candidates) × 14. Pairwise ",
        f"Jaccard overlap of target neuron classes (TPM > {TPM_THRESHOLD}) ",
        "expressing each modulator's receptor set.",
        "",
        "## Target-set sizes",
        "",
        "| modulator | receptors | # target classes |",
        "|---|---|---|",
    ]
    for mod, recs in MODULATOR_RECEPTORS.items():
        lines.append(f"| **{mod}** | {', '.join(recs)} | {len(targets[mod])} |")
    lines.append("")

    lines.append("## Full overlap matrix (Jaccard)")
    lines.append("")
    lines.append("| | " + " | ".join(mod_names) + " |")
    lines.append("|---|" + "---|" * len(mod_names))
    for i, a in enumerate(mod_names):
        row = [f"**{a}**"]
        for j, b in enumerate(mod_names):
            row.append(f"{J[i,j]:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## High-overlap pairs (> 0.7 — potential redundancy)")
    lines.append("")
    if high_overlaps:
        for a, b, v in sorted(high_overlaps, key=lambda x: -x[2]):
            lines.append(f"- **{a} ↔ {b}**: Jaccard = {v:.2f}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Distinct pairs (< 0.1 — complementary coverage)")
    lines.append("")
    if distinct_pairs:
        for a, b, v in sorted(distinct_pairs, key=lambda x: x[2])[:10]:
            lines.append(f"- {a} ↔ {b}: Jaccard = {v:.2f}")
    else:
        lines.append("- None (all modulators overlap moderately).")
    lines.append("")

    # T4-5 candidate specific notes
    lines.append("## T4-5 candidate redundancy check vs existing 9")
    lines.append("")
    t45 = ["FLP-13", "FLP-18", "FLP-21", "NLP-40", "DAF-28"]
    existing = [m for m in mod_names if m not in t45]
    for cand in t45:
        overlaps = [(ex, float(J[mod_names.index(cand),
                                  mod_names.index(ex)]))
                    for ex in existing]
        overlaps.sort(key=lambda x: -x[1])
        top = overlaps[:3]
        rec_text = ", ".join(f"{ex}={v:.2f}" for ex, v in top)
        max_j = top[0][1] if top else 0.0
        status = ("REDUNDANT" if max_j > 0.7 else
                  "PARTIAL" if max_j > 0.3 else
                  "DISTINCT")
        lines.append(f"- **{cand}** → top overlaps: {rec_text} [{status}]")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Total wall: {time.time()-t0:.1f}s")

    # Append STATUS
    status_md = ART / "overnight_20260421" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Task 4: overlap matrix\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: 14×14 Jaccard matrix; "
                f"{len(high_overlaps)} high-overlap pairs (>0.7), "
                f"{len(distinct_pairs)} distinct pairs (<0.1)\n")
        f.write(f"- Output: task4_overlap_matrix/\n")


if __name__ == "__main__":
    main()
