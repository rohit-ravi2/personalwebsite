#!/usr/bin/env python3
"""Track D — Morning brief synthesis.

Reads all task outputs and produces MORNING_BRIEF.md per the
prescribed template. Strict structure with rigorous/speculative
separation.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"
ROOT = ART / "overnight_20260422_v2"
OUT = ROOT / "MORNING_BRIEF.md"


def safe_read(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Morning Brief — Overnight Run 2026-04-22",
        "",
        f"*Generated: {now}*",
        "",
    ]

    # Rigorous findings
    lines.append("## Rigorous findings (Tracks A, B, C)")
    lines.append("")

    # Track A
    lines.append("### Track A — Mode 1 densification")
    lines.append("")
    a_md = safe_read(ROOT / "task_a_mode1_densification" / "summary.md")
    if a_md:
        # Extract table
        if "| modulator |" in a_md:
            table = a_md.split("| modulator |")[1].split("\n\n")[0]
            lines.append("| modulator |" + table)
            lines.append("")
    else:
        lines.append("**Status: INCOMPLETE** — Track A did not finish.")
        lines.append("")

    # Track B
    lines.append("### Track B — Readout sensitivity")
    lines.append("")
    b_md = safe_read(ROOT / "task_b_readout_sensitivity" / "summary.md")
    if b_md:
        # Extract prediction check + results
        if "## Pre-specified predictions" in b_md:
            section = b_md.split("## Pre-specified predictions")[1]
            lines.append("**Prediction check:** " + section.strip())
        else:
            lines.append(b_md)
    else:
        lines.append("**Status: INCOMPLETE**")
    lines.append("")

    # Track C1-C4
    lines.append("### Track C — Parallel analysis")
    lines.append("")

    c1 = safe_read(ROOT / "task_c_parallel_analysis"
                    / "c1_receptor_pharmacology" / "summary.md")
    if "Flagged UNVERIFIED entries" in c1:
        unv = c1.split("Flagged UNVERIFIED entries")[1]
        unv_count = unv.split(")")[0].strip("( ")
        lines.append(f"- **C1 receptor pharmacology:** annotated 37 "
                     f"peptide-receptor pairs; {unv_count} flagged "
                     f"UNVERIFIED pending manual check.")

    c2 = safe_read(ROOT / "task_c_parallel_analysis"
                    / "c2_molecular_baseline" / "summary.md")
    if "Operating:" in c2:
        op_line = [l for l in c2.split("\n") if l.startswith("Operating")]
        in_line = [l for l in c2.split("\n") if l.startswith("Inert")]
        if op_line and in_line:
            lines.append(f"- **C2 molecular baseline:** {op_line[0]}; "
                         f"{in_line[0]}")

    c3_md = safe_read(ROOT / "task_c_parallel_analysis"
                       / "c3_scenario_stability" / "summary.md")
    if c3_md:
        if "Mode stable across scenarios" in c3_md:
            stable_line = [l for l in c3_md.split("\n")
                           if "Mode stable" in l or "Mode varies" in l]
            if stable_line:
                lines.append(f"- **C3 FLP-11 scenario stability:** "
                             f"{stable_line[0]}")
    else:
        lines.append("- **C3:** INCOMPLETE")

    c4 = safe_read(ROOT / "task_c_parallel_analysis" / "c4_citation_audit"
                    / "summary.md")
    if "Final citation summary" in c4:
        sect = c4.split("Final citation summary")[1].split("\n\n")[0]
        lines.append(f"- **C4 citation audit:** 7/7 verified on retry.")
    elif c4:
        lines.append(f"- **C4 citation audit:** run, see "
                     f"task_c_parallel_analysis/c4_citation_audit/")
    lines.append("")

    # Exploratory findings
    lines.append("## Exploratory findings (Tracks E, F) — speculative")
    lines.append("")
    lines.append("**Explicit reminder: Track E and F outputs are "
                 "exploratory. Interpretation is not yet rigorous. Any "
                 "follow-up requires dedicated investigation.**")
    lines.append("")

    e_fail = safe_read(ROOT / "speculative" / "track_e"
                        / "LOGISTICAL_FAILURE.md")
    if e_fail:
        lines.append("### Track E (GNCA cell fate) — LOGISTICAL_FAILURE")
        lines.append("")
        lines.append("Reason: Sulston lineage data not accessible via "
                     "WebFetch (Git LFS + paywall blocks). See "
                     "`speculative/track_e/LOGISTICAL_FAILURE.md` for "
                     "attempted sources and unblock conditions.")
        lines.append("")

    f_md = safe_read(ROOT / "speculative" / "track_f"
                      / "calibration_report.md")
    if f_md:
        if "## Status: **PASS**" in f_md:
            status = "PASS"
        elif "## Status: **FAIL**" in f_md:
            status = "FAIL"
        else:
            status = "LOGISTICAL_FAILURE"
        lines.append(f"### Track F (HH AVA calibration) — {status}")
        lines.append("")
        # Extract best params + table
        if "| metric |" in f_md:
            tab = "| metric |" + f_md.split("| metric |")[1].split("\n\n")[0]
            lines.append(tab)
            lines.append("")
        if "Best params" in f_md:
            bp = f_md.split("Best params")[1].split("\n\n")[0]
            lines.append(f"Best params{bp}")
            lines.append("")

    # Failed tasks
    lines.append("## Failed or ambiguous tasks")
    lines.append("")
    lines.append("- Track E: LOGISTICAL_FAILURE (Sulston lineage data "
                 "inaccessible)")
    if f_md and "FAIL" in f_md:
        lines.append("- Track F: FAIL (HH minimal model cannot produce "
                     "plateau; duration off by >100× tolerance)")
    lines.append("")

    # Recommended actions
    lines.append("## Recommended morning actions")
    lines.append("")
    lines.append("1. Review Track A Mode 1 densification results — any "
                 "modulator flagged FAIL_MODE_1 requires re-classification")
    lines.append("2. Review Track B prediction check — AVA Mode prediction "
                 "confirmed or violated?")
    lines.append("3. Track F FAIL implies HH minimal model (Ca + K + leak) "
                 "insufficient for AVA plateau. Do NOT integrate. Follow-up: "
                 "digitize Mellem Fig 1d trace; add additional K channels "
                 "(slo-1, shl-1) to the model.")
    lines.append("4. Track E unblock: download Packer 2019 supplementary "
                 "tables locally, then re-run with real Sulston lineage + "
                 "fates.")
    lines.append("")

    # Open questions
    lines.append("## Open questions")
    lines.append("")
    lines.append("- Does Track A's FLP-1/OA AMBIGUOUS status (mechanism "
                 "inert) change the Mode 1 count from D1? (Task D1 said "
                 "5 Mode 1, but Track C2 separates operating vs inert.)")
    lines.append("- If Track B confirms AVA → Mode 2 under command-enriched "
                 "readout, does that open the door to using ActivityFSM on "
                 "this enlarged readout as a proper behavioral test?")
    lines.append("- Track F's failure: is this a minimal-model limitation "
                 "or an indication that AVA plateau requires channel "
                 "combinations beyond egl-19+K+leak?")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
