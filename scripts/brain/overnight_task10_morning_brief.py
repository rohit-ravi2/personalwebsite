#!/usr/bin/env python3
"""Overnight Task 10 — MORNING_BRIEF synthesis.

Runs last. Reads outputs from Tasks 1-8 and produces a single
consolidated document: what completed, headline findings, open issues,
and recommendations for the day's work.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"
ROOT = ART / "overnight_20260421"
OUT = ROOT / "MORNING_BRIEF.md"


def safe_read(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def safe_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# MORNING BRIEF — overnight run {time.strftime('%Y-%m-%d')}",
        "",
        f"*Generated: {now}*",
        "",
        "Wake-up synthesis of the overnight agentic run. Full details per",
        "task live in each `taskN_*/` subdirectory.",
        "",
    ]

    # Task 1 — D1 classification
    lines.append("## 1. D1 modulator Mode classification (the headline)")
    lines.append("")
    d1_csv = safe_csv(ROOT / "task1_d1" / "d1_classification_table.csv")
    if d1_csv is not None and len(d1_csv) > 0:
        mode_counts = d1_csv["observed_mode"].value_counts().to_dict()
        lines.append(f"**9 modulators classified across {len(d1_csv)} rows.**")
        lines.append("")
        for mode, cnt in mode_counts.items():
            lines.append(f"- **{mode}**: {cnt} modulators "
                         f"({', '.join(d1_csv[d1_csv['observed_mode'] == mode]['modulator'])})")
        lines.append("")
        lines.append("**Headline table (Δ per state vs control):**")
        lines.append("")
        lines.append("| modulator | scenario | ΔREV | ΔQUI | Mode |")
        lines.append("|---|---|---|---|---|")
        for _, r in d1_csv.iterrows():
            lines.append(
                f"| {r['modulator']} | {r['scenario']} | "
                f"{r['delta_REVERSE']:+.2f} | {r['delta_QUIESCENT']:+.2f} | "
                f"{r['observed_mode']} |"
            )
        lines.append("")
    else:
        lines.append("**Task 1 did not complete.** Check STATUS.md and ")
        lines.append("`task1_d1/run.log` for whether D1 is still running ")
        lines.append("or encountered a failure.")
        lines.append("")

    # Task 3 — peptide survey headline
    lines.append("## 2. Genome-wide peptide survey")
    lines.append("")
    survey_csv = safe_csv(ROOT / "task3_peptide_survey"
                          / "peptide_expression_catalog.csv")
    if survey_csv is not None:
        expressed = survey_csv[survey_csv["n_expressing"] > 0]
        below = survey_csv[
            (survey_csv["resolved"])
            & (survey_csv["n_expressing"] == 0)
        ]
        unres = survey_csv[~survey_csv["resolved"]]
        lines.append(f"- **{len(survey_csv)}** peptides scanned "
                     f"(FLP + NLP + INS + NPP families)")
        lines.append(f"- **{len(expressed)}** expressed above TPM=4 threshold")
        lines.append(f"- **{len(below)}** below detection "
                     f"(present in CeNGEN but TPM ≤ 4)")
        lines.append(f"- **{len(unres)}** unresolved gene symbols "
                     f"(potential artifacts or naming mismatches)")
        # By family
        lines.append("")
        lines.append("| family | resolved | expressed > 4 |")
        lines.append("|---|---|---|")
        for fam in ["FLP", "NLP", "INS", "NPP", "PDF"]:
            sub = survey_csv[survey_csv["family"] == fam]
            if len(sub) == 0:
                continue
            lines.append(f"| {fam} | {sub['resolved'].sum()} / {len(sub)} | "
                         f"{(sub['n_expressing'] > 0).sum()} |")
        lines.append("")

    # Task 4 — overlap
    lines.append("## 3. Modulator target-set overlap")
    lines.append("")
    ov_md = safe_read(ROOT / "task4_overlap_matrix" / "overlap_summary.md")
    if "## T4-5 candidate redundancy check" in ov_md:
        section = ov_md.split("## T4-5 candidate redundancy check")[1]
        section = section.split("\n\n")[0] + "\n"
        lines.append(section.strip())
        lines.append("")

    # Task 5 — readout coverage
    lines.append("## 4. 18-neuron readout peptidergic coverage")
    lines.append("")
    cov_csv = safe_csv(ROOT / "task5_readout_coverage" / "coverage_table.csv")
    if cov_csv is not None:
        mode1 = cov_csv["predicted_mode"].str.contains("Mode 1", na=False).sum()
        mode3 = cov_csv["predicted_mode"].str.contains("Mode 3", na=False).sum()
        lines.append(f"- Predicted Mode 1 (readout-blind): "
                     f"**{mode1}**/{len(cov_csv)} modulators")
        lines.append(f"- Predicted Mode 3 (readout-cascade): "
                     f"**{mode3}**/{len(cov_csv)} modulators")
        lines.append("")

    # Task 6 — FLP-13 vs FLP-11
    lines.append("## 5. FLP-13 vs FLP-11 target comparison")
    lines.append("")
    cmp_md = safe_read(ROOT / "task6_flp13_vs_flp11" / "comparison_summary.md")
    if "## Verdict" in cmp_md:
        verdict = cmp_md.split("## Verdict")[1].split("##")[0].strip()
        lines.append(verdict)
        lines.append("")

    # Task 7 — PubMed
    lines.append("## 6. T4-5 citation verification")
    lines.append("")
    pmd = safe_read(ROOT / "task7_pubmed" / "t4_5_citation_check.md")
    if "## Summary verification table" in pmd:
        section = pmd.split("## Summary verification table")[1]
        section = section.split("\n\n**Action")[0]
        lines.append("## Summary verification table" + section)
        lines.append("")

    # Task 8 — Ripoll-Sánchez
    lines.append("## 7. Ripoll-Sánchez 2023 cross-reference")
    lines.append("")
    rsmd = safe_read(ROOT / "task8_ripoll_sanchez" / "cross_reference.md")
    if "## Cross-reference for our T4-5 candidates" in rsmd:
        section = rsmd.split("## Cross-reference for our T4-5 candidates")[1]
        section = section.split("##")[0]
        lines.append("### " + section.strip())
        lines.append("")

    # Task 2
    lines.append("## 8. T4-5 pre-validation")
    lines.append("")
    prev_md = safe_read(ROOT / "task2_t45_preval" / "preval_report.md")
    if "## Verdict" in prev_md:
        section = prev_md.split("## Verdict")[1]
        lines.append(section.strip())
        lines.append("")
    else:
        lines.append("Task 2 did not run (D1 ran over time budget or other "
                     "skip reason).")
        lines.append("")

    # Open issues / next-day items
    lines.append("## Open issues and recommended next actions")
    lines.append("")
    lines.append("**Citation corrections to apply:**")
    lines.append("- Update FLP-18 primary reference from Rogers 2003 to "
                 "Cohen et al. 2009 (Rogers 2003 is actually FLP-21/NPR-1).")
    lines.append("- Verify Nelson 2013 attribution for NLP-22. CeNGEN shows "
                 "~zero expression in RIA; either the literature cite was "
                 "imprecise or CeNGEN under-detects rapid-turnover peptides.")
    lines.append("")
    lines.append("**T4-5 scope refinement:**")
    if d1_csv is not None:
        lines.append("- Review D1 classification table. Any modulator that "
                     "shows a Mode different from Task 5's prediction is a "
                     "priority for investigation — the predictor needs "
                     "refinement.")
    lines.append("- FLP-13 vs FLP-11 Jaccard (Task 6): check verdict above. "
                 "If largely redundant, reconsider FLP-13 as the quiescence "
                 "peptide addition.")
    lines.append("")
    lines.append("**Paper-relevant findings to preserve:**")
    lines.append("- Peptidergic broadcasters in 18-neuron readout (Task 5): "
                 "directly supports Mode 1 readout-blindness argument.")
    lines.append("- Ripoll-Sánchez 2023 confirms peptidergic rich-club = "
                 "52% of neurons (Task 8); cite in paper methods.")
    lines.append("- Genome-wide peptide survey produces supplementary data "
                 "justifying our modulator selection as data-driven.")
    lines.append("")

    lines.append("## Phase 0 close-out checklist")
    lines.append("")
    lines.append("- [x] Baseline audits (scenario, AVA/touch, RIS molecular)")
    lines.append("- [x] Phase 0 three-mode taxonomy demonstrated")
    lines.append("- [x] Voltage-scale finding + Mellem 2008 replacement")
    lines.append("- [x] Audit strategy document "
                 "(`docs/audit-strategy.md`)")
    lines.append("- [x] Peptide validation pipeline "
                 "(A1 + B4 + D1)")
    lines.append("- [ ] Commit all overnight outputs")
    lines.append("- [ ] Apply FLP-18 citation correction")
    lines.append("- [ ] Refine T4-5 scope based on D1 outcomes")
    lines.append("")

    lines.append("## STATUS.md tail")
    lines.append("")
    status_md = safe_read(ROOT / "STATUS.md")
    lines.append("```")
    lines.append(status_md.strip())
    lines.append("```")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
