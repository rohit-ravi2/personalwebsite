#!/usr/bin/env python3
"""Overnight Task 3 — Genome-wide neuropeptide expression survey.

Reference dataset (NOT inclusion filter) for the paper. Iterates over
all gene symbols matching FLP-*, NLP-*, INS-*, NPP-* in WormBase
association + any additional annotated peptide genes, pulls CeNGEN
TPM, and produces a structured catalog.

Output:
  task3_peptide_survey/peptide_expression_catalog.csv (full table)
  task3_peptide_survey/peptide_expression_summary.md  (grouped report)
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_modulator_tables import (
    load_gene_symbol_map, load_cengen_matrix,
)

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260421" / "task3_peptide_survey"
OUT_CSV = OUT_DIR / "peptide_expression_catalog.csv"
OUT_MD = OUT_DIR / "peptide_expression_summary.md"

TPM_THRESHOLD = 4.0
TPM_STRONG = 20.0

# Peptide gene family regex patterns
PEPTIDE_PATTERNS = [
    (r"^flp-\d+$", "FLP"),
    (r"^nlp-\d+$", "NLP"),
    (r"^ins-\d+$", "INS"),
    (r"^npp-\d+$", "NPP"),
    (r"^pdf-\d+$", "PDF"),
    (r"^daf-28$", "INS"),  # insulin family member
]


def main():
    t0 = time.time()
    print("Loading CeNGEN + gene association...")
    df, sym_to_wb = load_cengen_matrix()
    sym_to_wb_ci = {k.lower(): v for k, v in sym_to_wb.items()}

    # Iterate all symbols, filter to peptide families
    candidate_symbols = []
    for sym in sym_to_wb_ci.keys():
        for pat, family in PEPTIDE_PATTERNS:
            if re.match(pat, sym):
                candidate_symbols.append((sym, family))
                break
    candidate_symbols = sorted(set(candidate_symbols))
    print(f"Found {len(candidate_symbols)} peptide genes in WormBase "
          f"(FLP/NLP/INS/NPP families)")

    rows = []
    for sym, family in candidate_symbols:
        wb = sym_to_wb_ci.get(sym)
        if wb is None or wb not in df.columns:
            rows.append({
                "symbol": sym, "family": family, "wb_id": wb or "unresolved",
                "resolved": False, "max_tpm": 0.0,
                "n_expressing": 0, "n_strong": 0, "top_expressing": "",
                "category": "unresolved",
            })
            continue
        col = df[wb].astype(float)
        max_tpm = float(col.max())
        n_exp = int((col > TPM_THRESHOLD).sum())
        n_strong = int((col > TPM_STRONG).sum())
        top_classes = col[col > TPM_THRESHOLD].sort_values(
            ascending=False).index.tolist()[:8]
        if max_tpm < TPM_THRESHOLD:
            cat = "below_threshold"
        elif n_exp == 1:
            cat = "narrow"
        elif n_exp <= 3:
            cat = "narrow"
        elif n_exp >= 10:
            cat = "broad"
        else:
            cat = "moderate"
        rows.append({
            "symbol": sym, "family": family, "wb_id": wb,
            "resolved": True, "max_tpm": round(max_tpm, 2),
            "n_expressing": n_exp, "n_strong": n_strong,
            "top_expressing": ";".join(top_classes),
            "category": cat,
        })

    df_out = pd.DataFrame(rows).sort_values(
        ["family", "n_expressing", "max_tpm"], ascending=[True, False, False]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df_out)} rows)")

    # Summary counts
    lines = [
        "# Task 3 — Genome-wide peptide expression survey",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "**Reference dataset, not a T4-5 inclusion filter.** Every gene ",
        f"below has expression status in CeNGEN (TPM > {TPM_THRESHOLD} ",
        "threshold from Taylor 2021). Peptides passing this filter are ",
        "candidates for further literature/phenotype review; peptides ",
        "failing are below detection and currently untestable at this ",
        "CeNGEN resolution.",
        "",
        "## Overall counts",
        "",
    ]
    cnt = df_out.groupby(["family", "category"]).size().unstack(
        fill_value=0)
    lines.append("| family | broad | moderate | narrow | below_threshold | unresolved |")
    lines.append("|---|---|---|---|---|---|")
    for family in ["FLP", "NLP", "INS", "NPP", "PDF"]:
        if family not in cnt.index:
            continue
        row = cnt.loc[family]
        lines.append(
            f"| **{family}** | {row.get('broad', 0)} | "
            f"{row.get('moderate', 0)} | {row.get('narrow', 0)} | "
            f"{row.get('below_threshold', 0)} | "
            f"{row.get('unresolved', 0)} |"
        )
    lines.append("")

    # Top expressed peptides per family
    for family in ["FLP", "NLP", "INS", "NPP", "PDF"]:
        sub = df_out[df_out["family"] == family]
        if sub.empty:
            continue
        top = sub[sub["resolved"] & (sub["max_tpm"] > TPM_THRESHOLD)]
        top = top.sort_values("max_tpm", ascending=False)
        lines.append(f"## {family} family — expressed peptides")
        lines.append("")
        lines.append("| symbol | max TPM | # classes > 4 | top expressing |")
        lines.append("|---|---|---|---|")
        for _, r in top.head(30).iterrows():
            lines.append(
                f"| {r['symbol']} | {r['max_tpm']} | {r['n_expressing']} | "
                f"{r['top_expressing'][:60]} |"
            )
        lines.append("")
        # Below-threshold
        below = sub[sub["resolved"] & (sub["max_tpm"] <= TPM_THRESHOLD)]
        if len(below) > 0:
            lines.append(f"**{family} below threshold** "
                         f"({len(below)} peptides): "
                         f"{', '.join(below['symbol'].tolist())}")
            lines.append("")
        # Unresolved
        unres = sub[~sub["resolved"]]
        if len(unres) > 0:
            lines.append(f"**{family} unresolved** ({len(unres)}): "
                         f"{', '.join(unres['symbol'].tolist())}")
            lines.append("")

    lines.append("## Methodological note")
    lines.append("")
    lines.append("- TPM threshold follows Taylor 2021 (CeNGEN) convention.")
    lines.append("- This catalog is reference supplementary data; peptide ")
    lines.append("  inclusion in the simulator's T4-5 modulator expansion ")
    lines.append("  requires additional phenotype verification (C1) + ")
    lines.append("  receptor target coverage (A3) + Mode prediction (B4).")
    lines.append("- CeNGEN under-detects peptides with rapid mRNA turnover ")
    lines.append("  (e.g., NLP-22) and rate-limiting NT-synthesis enzymes ")
    lines.append("  (tph-1, cat-2, tbh-1). Absence here does not equal ")
    lines.append("  absence in biology.")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Total wall: {time.time()-t0:.1f}s")

    # Append status
    status_md = ART / "overnight_20260421" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Task 3: genome-wide peptide survey\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: {len(df_out)} peptides surveyed; "
                f"{df_out['resolved'].sum()} resolved, "
                f"{(df_out['n_expressing'] > 0).sum()} expressed above threshold\n")
        f.write(f"- Output: task3_peptide_survey/\n")


if __name__ == "__main__":
    main()
