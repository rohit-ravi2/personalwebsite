#!/usr/bin/env python3
"""Phase 1 gauntlet — partial-results analyzer.

Loads whatever phase1_gauntlet_*_phenotype.csv and phase1_gauntlet_*_scenario.csv
files exist and runs the summary + decision matrix from `phase1_gauntlet.py`.
Used after a partial gauntlet run to build the decision matrix from completed
modes only.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from phase1_gauntlet import (  # noqa: E402
    summarize_phenotype,
    summarize_cascade,
    summarize_scenarios,
    write_decision_matrix,
    MODES,
    ART,
)


def main():
    tier = "screen"

    # Discover available CSVs
    phen_paths = sorted(ART.glob(f"phase1_gauntlet_*_{tier}_phenotype.csv"))
    scen_paths = sorted(ART.glob(f"phase1_gauntlet_*_{tier}_scenario.csv"))

    print(f"Found phenotype CSVs:")
    for p in phen_paths:
        print(f"  {p.name}")
    print(f"Found scenario CSVs:")
    for p in scen_paths:
        print(f"  {p.name}")

    phen_dfs = [pd.read_csv(p) for p in phen_paths]
    scen_dfs = [pd.read_csv(p) for p in scen_paths]

    if phen_dfs:
        phen_all = pd.concat(phen_dfs, ignore_index=True)
        phen_summary = summarize_phenotype(phen_all)
        cascade_summary = summarize_cascade(phen_all)
    else:
        phen_summary = {}
        cascade_summary = {}

    if scen_dfs:
        scen_all = pd.concat(scen_dfs, ignore_index=True)
        scen_summary = summarize_scenarios(scen_all)
    else:
        scen_summary = {}

    # JSON sidecar
    summary_path = ART / f"phase1_gauntlet_{tier}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "tier": tier,
            "modes_complete_phenotype": sorted(phen_all["mode"].unique().tolist()) if phen_dfs else [],
            "modes_complete_scenario": sorted(scen_all["mode"].unique().tolist()) if scen_dfs else [],
            "phenotype_summary": phen_summary,
            "cascade_summary": cascade_summary,
            "scenario_summary": scen_summary,
        }, f, indent=2, default=str)
    print(f"\n[summary JSON written] {summary_path}")

    # Decision matrix markdown
    decision_path = ART / f"phase1_gauntlet_{tier}_decision_matrix.md"
    write_decision_matrix(phen_summary, cascade_summary, scen_summary, decision_path)
    print(f"[decision matrix written] {decision_path}")


if __name__ == "__main__":
    main()
