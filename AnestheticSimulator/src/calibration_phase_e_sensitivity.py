"""CP2 — Phase E sensitivity sweep on CLINICAL_EFFECTIVE_OCCUPANCY.

Tests whether the Stewart 2000 release-p reduction band (0.3-0.7) is
reproduced across a plausible range of CLINICAL_EFFECTIVE_OCCUPANCY values
or only at the single hand-tuned value 0.30.

Outputs: artifacts/calibration/phase_e_sensitivity.{csv,md}

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/calibration_phase_e_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase_e_markov_synapse import simulate_synapse, N_CA_COOPERATIVITY_WT

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "artifacts" / "kinetics" / "wave2_overlay.json"
OUT_CSV = ROOT / "artifacts" / "calibration" / "phase_e_sensitivity.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "phase_e_sensitivity.md"

OCC_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]


def main() -> int:
    overlay = json.load(open(OVERLAY))
    halothane = overlay["by_anesthetic"]["halothane"]
    raw_n_delta = halothane["UNC-64"]["parameters"]["n_Ca_delta"]["value"]

    # WT baseline
    wt = simulate_synapse(N_CA_COOPERATIVITY_WT, duration_ms=10000,
                          spontaneous_only=False, seed=42)
    wt_evoked = wt["evoked_release_p"]
    print(f"WT n={N_CA_COOPERATIVITY_WT}: evoked p = {wt_evoked:.3f}")
    print(f"Halothane raw n_Ca_delta = {raw_n_delta:.3f}")
    print()
    print(f"{'occ_factor':>12s} {'eff_n_delta':>11s} {'n_perturbed':>13s} {'evoked_p':>10s} {'fold_change':>13s} {'in 0.3-0.7':>12s}")

    rows = []
    for occ in OCC_VALUES:
        eff = raw_n_delta * occ
        n_pert = max(0.5, N_CA_COOPERATIVITY_WT + eff)
        result = simulate_synapse(n_pert, duration_ms=10000,
                                  spontaneous_only=False, seed=42)
        fold = result["evoked_release_p"] / wt_evoked if wt_evoked > 0 else float("nan")
        in_band = 0.3 <= fold <= 0.7
        rows.append({
            "occ_factor": occ,
            "raw_n_Ca_delta": raw_n_delta,
            "eff_n_Ca_delta": eff,
            "n_perturbed": n_pert,
            "evoked_release_p": result["evoked_release_p"],
            "fold_change_vs_WT": fold,
            "in_stewart_band": in_band,
        })
        print(f"  {occ:>10.2f} {eff:>11.3f} {n_pert:>13.3f} "
              f"{result['evoked_release_p']:>10.3f} {fold:>13.3f} {'YES' if in_band else 'no':>12s}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            row = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in r.items()}
            w.writerow(row)
    print(f"\nCSV: {OUT_CSV}")

    in_band_count = sum(1 for r in rows if r["in_stewart_band"])
    in_band_range = [r["occ_factor"] for r in rows if r["in_stewart_band"]]
    print(f"\nOccupancy values producing fold-change in Stewart 0.3-0.7 band: {in_band_count}/{len(rows)}")
    if in_band_range:
        print(f"  Range: [{min(in_band_range):.2f}, {max(in_band_range):.2f}]")

    # Verdict
    if in_band_count >= 4:
        verdict = "ROBUST — Stewart band reproduced across wide occupancy range; Phase E predictions defensible"
    elif in_band_count >= 2:
        verdict = "MODERATELY ROBUST — band reproduced across narrow but multi-point range"
    elif in_band_count == 1:
        verdict = "BRITTLE — only the hand-tuned 0.30 value produces in-band fold-change; PASS is post-hoc fitting"
    else:
        verdict = "FAILED — even hand-tuned 0.30 produces out-of-band; method is wrong"

    print(f"\nVerdict: {verdict}")

    with open(OUT_MD, "w") as f:
        f.write("# CP2 — Phase E CLINICAL_EFFECTIVE_OCCUPANCY sensitivity sweep\n\n")
        f.write("## Method\n\n"
                "Sweep CLINICAL_EFFECTIVE_OCCUPANCY across [0.10, 0.70] and observe halothane "
                "release-p fold-change vs WT. Stewart 2000 PMID 11095753 / van Swinderen 1999 PMID "
                "10051668 target band: 0.3-0.7.\n\n")
        f.write(f"WT baseline: n={N_CA_COOPERATIVITY_WT}, evoked p={wt_evoked:.3f}\n")
        f.write(f"Halothane raw n_Ca_delta from wave2_overlay.json (UNC-64 SNARE proxy): {raw_n_delta:.3f}\n\n")
        f.write("## Sweep results\n\n")
        f.write("| occ_factor | eff n_delta | n_perturbed | evoked_p | fold_change | in 0.3-0.7 band |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['occ_factor']:.2f} | {r['eff_n_Ca_delta']:.3f} | "
                    f"{r['n_perturbed']:.3f} | {r['evoked_release_p']:.3f} | "
                    f"{r['fold_change_vs_WT']:.3f} | {'✓' if r['in_stewart_band'] else '✗'} |\n")
        f.write(f"\n## Verdict: {verdict}\n\n"
                f"In-band count: {in_band_count}/{len(rows)} occupancy values\n")
        if in_band_range:
            f.write(f"In-band range: [{min(in_band_range):.2f}, {max(in_band_range):.2f}]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
