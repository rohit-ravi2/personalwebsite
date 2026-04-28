#!/usr/bin/env python3
"""
Phase H — Empirical validation against 8 anchor predictions.

Status: SCAFFOLDED. Anchor evaluation IS implemented; runs once Phase G outputs exist.

Purpose
-------
Compare Phase G simulator predictions against 8 published wet-lab anchors.
Pass criterion: >= 4 of 8 anchors match within tolerance, AND the multi-target
lesion test (Gate G.1.5) holds at the program level.

8 anchors
---------
1. WT halothane EC50 ~3% atm (Crowder 1996 PMID 8855256), within 2x.
2. WT isoflurane EC50 ~5% atm (Morgan 1995), within 2x.
3. gas-1(fc21) iso EC50 leftward 2-3x (Morgan & Sedensky 1995 PMID 7549290), within 50%.
4. unc-79(e1068) halothane rightward 2-3x (Sedensky 1992 PMID 1346264), within 50%.
5. unc-80(e1069) similar to unc-79 (Sedensky 1992 PMID 1346264), within 50%.
6. twk-18(cn110) halothane resistance 2-3x (Sedensky 2001 PMID 11756669), within 50%.
7. unc-13(s69) halothane hypersensitivity 2-3x (van Swinderen 1999), within 50%.
8. propofol immobilization in uM range (Boddington 2017), order of magnitude.

Inputs
------
- artifacts/runs/aggregated_ec50.csv (Phase G)
- artifacts/runs/lesion_comparison.csv (Phase G)

Outputs
-------
- artifacts/validation/anchor_table.csv
- artifacts/validation/anchor_evaluation.md
- artifacts/validation/lesion_test_program_level.md
- artifacts/validation/program_verdict.md
- artifacts/validation/phase_h_completion.md

Reference: preregistration/phase_h_empirical_validation.md
"""

import argparse
import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "artifacts" / "runs"
VAL_DIR = ROOT / "artifacts" / "validation"
LOG_DIR = ROOT / "artifacts" / "logs"

# Anchor specifications: (name, anesthetic, genotype, ratio_target, tolerance, source)
ANCHORS = [
    {
        "id": 1, "name": "WT halothane EC50", "anesthetic": "halothane", "genotype": "WT",
        "expected_EC50_uM": 340, "tolerance_factor": 2.0,
        "source_PMID": "8855256", "source": "Crowder 1996 PNAS"
    },
    {
        "id": 2, "name": "WT isoflurane EC50", "anesthetic": "isoflurane", "genotype": "WT",
        "expected_EC50_uM": 290, "tolerance_factor": 2.0,
        "source_PMID": "PMID lookup needed", "source": "Morgan 1995"
    },
    {
        "id": 3, "name": "gas-1 iso hypersensitivity", "anesthetic": "isoflurane",
        "genotype_compare": ("gas1", "WT"), "expected_ratio_range": (0.25, 0.67),
        "tolerance_factor_on_ratio": 1.5,
        "source_PMID": "7549290", "source": "Morgan & Sedensky 1995"
    },
    {
        "id": 4, "name": "unc-79 halothane resistance", "anesthetic": "halothane",
        "genotype_compare": ("unc79", "WT"), "expected_ratio_range": (1.5, 3.0),
        "tolerance_factor_on_ratio": 1.5,
        "source_PMID": "1346264", "source": "Sedensky 1992"
    },
    {
        "id": 5, "name": "unc-80 similar to unc-79", "anesthetic": "halothane",
        "genotype_compare": ("unc80", "WT"), "expected_ratio_range": (1.5, 3.0),
        "tolerance_factor_on_ratio": 1.5,
        "source_PMID": "1346264", "source": "Sedensky 1992"
    },
    {
        "id": 6, "name": "twk-18 halothane resistance", "anesthetic": "halothane",
        "genotype_compare": ("twk18", "WT"), "expected_ratio_range": (1.5, 3.0),
        "tolerance_factor_on_ratio": 1.5,
        "source_PMID": "11756669", "source": "Sedensky 2001"
    },
    {
        "id": 7, "name": "unc-13 halothane hypersensitivity", "anesthetic": "halothane",
        "genotype_compare": ("unc13", "WT"), "expected_ratio_range": (0.33, 0.67),
        "tolerance_factor_on_ratio": 1.5,
        "source_PMID": "PMID lookup needed", "source": "van Swinderen 1999"
    },
    {
        "id": 8, "name": "propofol uM range", "anesthetic": "propofol", "genotype": "WT",
        "expected_EC50_uM": 1.0, "tolerance_factor": 10.0,
        "source_PMID": "PMID lookup needed", "source": "Boddington 2017"
    },
]


def setup_logger(verbose: bool = False) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"phase_h_{date.today().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("phase_h")


def load_aggregated_ec50(log: logging.Logger) -> list[dict] | None:
    path = RUN_DIR / "aggregated_ec50.csv"
    if not path.exists():
        log.error("Phase G aggregated_ec50.csv not found at %s", path)
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def evaluate_anchor(anchor: dict, ec50_rows: list[dict], log: logging.Logger) -> dict:
    """Evaluate a single anchor. Returns {anchor_id, pass, ratio, reason}."""
    if anchor.get("genotype_compare"):
        # Genotype comparison anchor (3, 4, 5, 6, 7)
        mut, wt = anchor["genotype_compare"]
        try:
            mut_row = next(r for r in ec50_rows
                           if r.get("anesthetic") == anchor["anesthetic"]
                           and r.get("genotype") == mut
                           and r.get("lesion_class", "full") == "full")
            wt_row = next(r for r in ec50_rows
                          if r.get("anesthetic") == anchor["anesthetic"]
                          and r.get("genotype") == wt
                          and r.get("lesion_class", "full") == "full")
        except StopIteration:
            return {"anchor_id": anchor["id"], "pass": False, "ratio": None,
                    "reason": "missing data"}
        try:
            ratio = float(mut_row["fitted_EC50"]) / float(wt_row["fitted_EC50"])
        except (ValueError, ZeroDivisionError):
            return {"anchor_id": anchor["id"], "pass": False, "ratio": None,
                    "reason": "invalid EC50 values"}
        lo_target, hi_target = anchor["expected_ratio_range"]
        tol = anchor["tolerance_factor_on_ratio"]
        # Allow ratio to be between lo_target/tol and hi_target*tol
        in_range = (lo_target / tol) <= ratio <= (hi_target * tol)
        return {"anchor_id": anchor["id"], "pass": in_range, "ratio": ratio,
                "reason": f"ratio={ratio:.2f}; target [{lo_target}, {hi_target}]; "
                          f"tolerated [{lo_target/tol:.2f}, {hi_target*tol:.2f}]"}
    else:
        # Absolute EC50 anchor (1, 2, 8)
        try:
            row = next(r for r in ec50_rows
                       if r.get("anesthetic") == anchor["anesthetic"]
                       and r.get("genotype") == anchor.get("genotype", "WT")
                       and r.get("lesion_class", "full") == "full")
        except StopIteration:
            return {"anchor_id": anchor["id"], "pass": False, "ratio": None,
                    "reason": "missing data"}
        try:
            sim_EC50 = float(row["fitted_EC50"])
        except (ValueError, KeyError):
            return {"anchor_id": anchor["id"], "pass": False, "ratio": None,
                    "reason": "invalid simulated EC50"}
        ratio = sim_EC50 / anchor["expected_EC50_uM"]
        tol = anchor["tolerance_factor"]
        in_range = (1.0 / tol) <= ratio <= tol
        return {"anchor_id": anchor["id"], "pass": in_range, "ratio": ratio,
                "reason": f"sim={sim_EC50:.1f} vs pub={anchor['expected_EC50_uM']:.1f}; "
                          f"ratio={ratio:.2f}, tolerance {tol}x"}


def evaluate_lesion_test(log: logging.Logger) -> dict:
    """Re-evaluate Gate G.1.5 at the program level."""
    path = RUN_DIR / "lesion_comparison.csv"
    if not path.exists():
        log.warning("lesion_comparison.csv not found; G.1.5 evaluation deferred")
        return {"overall": "PENDING", "reason": "missing lesion data"}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    full_row = next((r for r in rows if r.get("lesion_class") == "full"), None)
    if not full_row:
        return {"overall": "PENDING", "reason": "no full-effect row"}
    try:
        full_effect = float(full_row["fraction_immobilized_mean"])
    except (ValueError, KeyError):
        return {"overall": "PENDING", "reason": "invalid full-effect value"}
    single_class_reproduces = []
    for r in rows:
        if r.get("lesion_class") in ("full", ""):
            continue
        try:
            le = float(r["fraction_immobilized_mean"])
        except (ValueError, KeyError):
            continue
        if le >= 0.8 * full_effect:
            single_class_reproduces.append(r["lesion_class"])
    return {
        "single_classes_reproducing_80pct": single_class_reproduces,
        "G.1.5_pass": len(single_class_reproduces) == 0,
        "overall": "PASS" if len(single_class_reproduces) == 0 else "FAIL_MULTITARGET_FALSIFIED",
    }


def write_program_verdict(anchor_results: list[dict], lesion_result: dict,
                           log: logging.Logger) -> dict:
    n_pass = sum(1 for r in anchor_results if r["pass"])
    if n_pass >= 6:
        verdict_label = "STRONG_PASS"
    elif n_pass >= 4:
        verdict_label = "PASS"
    elif n_pass >= 2:
        verdict_label = "PARTIAL_FAIL"
    else:
        verdict_label = "FAIL"

    lesion_pass = lesion_result.get("G.1.5_pass", None)
    program_overall = "PASS" if (verdict_label in ("PASS", "STRONG_PASS")
                                  and lesion_pass) else "FAIL_OR_PARTIAL"

    verdict = {
        "n_pass_anchors": n_pass,
        "verdict_label": verdict_label,
        "lesion_test_pass": lesion_pass,
        "program_overall": program_overall,
        "anchor_results": anchor_results,
        "lesion_result": lesion_result,
    }
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(VAL_DIR / "program_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    md = ["# Wave P — Program verdict", "",
          f"**Anchors passed:** {n_pass}/8 ({verdict_label})",
          f"**Lesion test (G.1.5):** {'PASS' if lesion_pass else 'FAIL or PENDING'}",
          f"**Program overall:** {program_overall}", ""]
    md.append("## Per-anchor results")
    md.append("")
    md.append("| ID | Name | Pass | Ratio | Reason |")
    md.append("|---|---|---|---|---|")
    for r in anchor_results:
        anchor = next(a for a in ANCHORS if a["id"] == r["anchor_id"])
        md.append(f"| {r['anchor_id']} | {anchor['name']} | "
                  f"{'PASS' if r['pass'] else 'FAIL'} | "
                  f"{r.get('ratio')} | {r.get('reason')} |")
    md.append("")
    md.append("## Lesion test")
    md.append("")
    md.append(f"Single classes reproducing >=80% of full effect: "
              f"{lesion_result.get('single_classes_reproducing_80pct', 'PENDING')}")
    with open(VAL_DIR / "program_verdict.md", "w") as f:
        f.write("\n".join(md))
    log.info("Wrote program_verdict.md and .json")
    return verdict


def run(args: argparse.Namespace, log: logging.Logger) -> int:
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        log.info("[dry-run] would evaluate %d anchors + lesion test", len(ANCHORS))
        return 0
    ec50_rows = load_aggregated_ec50(log)
    if ec50_rows is None:
        log.error("Cannot evaluate without Phase G aggregated_ec50.csv")
        return 1
    results = [evaluate_anchor(a, ec50_rows, log) for a in ANCHORS]
    lesion = evaluate_lesion_test(log)
    write_program_verdict(results, lesion, log)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase H empirical validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    log = setup_logger(args.verbose)

    print("PHASE H SCAFFOLD — anchor evaluation IS implemented; needs Phase G output")
    print("See preregistration/phase_h_empirical_validation.md")
    return run(args, log)


if __name__ == "__main__":
    sys.exit(main())
