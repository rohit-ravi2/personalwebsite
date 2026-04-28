"""CP C.1 + C.2 — CeNGEN inventory + expression-to-conductance calibration.

Read CeNGEN panel data; map gene families to Wave 2 channel implementations;
calibrate scaling parameters using AVAL/AVAR/RIM cells where both biophysical
ground truth (Nicoletti) and expression data exist; report leave-one-out
validation.

CRITICAL METHODOLOGY DISCIPLINE: this is exploratory work. The calibration
is informative if it shows convergent scaling across multiple cells; it's
honest about uncertainty; it does NOT pretend to predict channels that
the CeNGEN panel doesn't cover (notably leak channels and IRK channels).

Output:
- cengen_coupling/cengen_channel_inventory.csv — per-(neuron, channel-gene)
  expression table
- cengen_coupling/expression_to_conductance_calibration.md — calibration
  methodology + parameter estimates + leave-one-out results
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cell_params import ALL_CELLS

PANEL_PATH = ROOT.parents[1] / "public" / "data" / "cengen-panel.json"
OUT = ROOT / "artifacts"
OUT_CENGEN = ROOT / "cengen_coupling"
CHK = ROOT / "checkpoints" / "path_c_cp1_c2.json"


# Gene → Wave 2 channel name mapping
# (CeNGEN gene → cell-builder channel name in Wave 2 production code)
GENE_TO_CHANNEL = {
    "egl-19": "egl19",
    "unc-2": "unc2",
    "cca-1": "cca1",
    "shl-1": "shl1",
    "kvs-1": "kvs1",       # not directly in Wave 2; mammalian Shaker analog
    "shk-1": "shk1",       # not in current Wave 2 cells but in channel library
    "exp-2": "exp2",       # not in current Wave 2 cells
    "unc-103": "unc103",
    "slo-1": "slo1iso",    # Wave 2 has slo1iso + slo1egl19
    "slo-2": "slo2",       # not in current Wave 2 cells
    "irk-2": "irk",        # IRK family; Wave 2 uses generic "irk"
    "kqt-1": "kqt1",
    "kqt-3": "kqt3",
    "egl-2": "egl2",       # Wave 2 RIM has egl2
    "nca-2": "nca",
    "unc-77": "nca",       # NCA-1 paralog; same channel mechanism
    "unc-80": "nca_aux",   # auxiliary subunit; not direct conductance
    "twk-18": "twk18",     # not in Wave 2 cells but in equation_validators infrastructure
}

# Channels not in CeNGEN panel (must use cell-builder defaults for these)
NOT_IN_CENGEN = {"leak", "irk"}  # leak is not gene-encoded; irk-1/3 not always covered


def load_panel():
    return json.load(open(PANEL_PATH))


def expression_for_neuron(panel, neuron_name: str) -> dict[str, float]:
    """Return {gene: TPM} for a given neuron; 0.0 for un-listed genes."""
    return panel["expression"].get(neuron_name, {})


def ground_truth_g_nS(cell_name: str) -> dict[str, float]:
    """Return per-channel conductance in nS for the named cell."""
    cell = ALL_CELLS[cell_name]
    out = {}
    if "g_nS" in cell:
        for ch, g in cell["g_nS"].items():
            if g > 0:
                out[ch] = g
    if "g_Scm2" in cell:
        for ch, g in cell["g_Scm2"].items():
            g_nS = g * cell["surf_cm2"] * 1e9
            if g_nS > 0:
                out[ch] = g_nS
    return out


def calibrate_alpha(panel, cells_for_calibration: list[str]) -> dict[str, dict]:
    """Per-channel linear scaling alpha: g_nS = alpha × TPM.

    Use multiple Wave 2 cells (AVAL, AVAR, RIM) to fit alpha per channel.
    Return median alpha + scatter per channel.
    """
    # For each channel that's in BOTH CeNGEN and at least one Wave 2 cell,
    # collect (TPM, g_nS) pairs across the calibration cells.
    pairs_per_channel: dict[str, list[tuple[str, float, float]]] = {}
    for cell_name in cells_for_calibration:
        # Cell name in CeNGEN: AVA → AVAL/AVAR; AIY → AIYL/AIYR; RIM → RIML/RIMR
        cengen_neurons = {
            "AVAL": "AVAL", "AVAR": "AVAR",
            "AIY": "AIYL", "RIM": "RIML",
        }
        cengen_neuron = cengen_neurons.get(cell_name)
        if not cengen_neuron:
            continue
        cell_expr = expression_for_neuron(panel, cengen_neuron)
        ground_g = ground_truth_g_nS(cell_name)
        # For each gene→channel mapping, if the channel is in the cell's ground truth
        # and the gene has a non-zero TPM, record the pair
        for gene, channel in GENE_TO_CHANNEL.items():
            tpm = cell_expr.get(gene, 0.0)
            g_nS = ground_g.get(channel, 0.0)
            if g_nS > 0 and tpm > 0:
                pairs_per_channel.setdefault(channel, []).append(
                    (cell_name, tpm, g_nS)
                )

    # Compute median alpha per channel; also report individual pairs
    calibration = {}
    for channel, pairs in pairs_per_channel.items():
        alphas = [g_nS / tpm for (_, tpm, g_nS) in pairs]
        calibration[channel] = {
            "n_cells": len(pairs),
            "pairs": [{"cell": c, "tpm": tpm, "g_nS": g} for (c, tpm, g) in pairs],
            "alpha_median": round(statistics.median(alphas), 4) if alphas else None,
            "alpha_mean": round(statistics.mean(alphas), 4) if alphas else None,
            "alpha_min": round(min(alphas), 4) if alphas else None,
            "alpha_max": round(max(alphas), 4) if alphas else None,
            "alpha_spread_ratio": round(max(alphas) / min(alphas), 2) if alphas else None,
        }
    return calibration


def loo_validation(panel, all_cells: list[str], calibration_func=calibrate_alpha) -> list[dict]:
    """Leave-one-out: train alpha on N-1 cells, predict the held-out cell's
    channels, compare to ground truth.
    """
    out = []
    for held_out in all_cells:
        train_cells = [c for c in all_cells if c != held_out]
        cal = calibration_func(panel, train_cells)
        # Predict held-out cell's channels
        cengen_neurons = {
            "AVAL": "AVAL", "AVAR": "AVAR", "AIY": "AIYL", "RIM": "RIML",
        }
        cengen_neuron = cengen_neurons.get(held_out)
        if not cengen_neuron:
            continue
        cell_expr = expression_for_neuron(panel, cengen_neuron)
        ground_g = ground_truth_g_nS(held_out)

        cell_predictions = []
        for gene, channel in GENE_TO_CHANNEL.items():
            tpm = cell_expr.get(gene, 0.0)
            g_actual = ground_g.get(channel, 0.0)
            if channel not in cal or tpm == 0:
                continue
            alpha = cal[channel]["alpha_median"]
            if alpha is None:
                continue
            g_predicted = alpha * tpm
            log_err = math.log10(g_predicted / g_actual) if g_actual > 0 else None
            cell_predictions.append({
                "gene": gene, "channel": channel, "tpm": tpm,
                "g_actual_nS": round(g_actual, 5),
                "g_predicted_nS": round(g_predicted, 5),
                "log10_err": round(log_err, 3) if log_err is not None else None,
            })
        out.append({
            "held_out": held_out,
            "predictions": cell_predictions,
            "calibration_used": {ch: cal[ch]["alpha_median"] for ch in cal},
        })
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_CENGEN.mkdir(parents=True, exist_ok=True)
    CHK.parent.mkdir(parents=True, exist_ok=True)

    panel = load_panel()

    # CP C.1 — Inventory CSV
    print("CP C.1 — CeNGEN channel inventory")
    inventory_rows = []
    target_neurons = [
        "AVAL", "AVAR", "AIYL", "AIYR", "RIML", "RIMR",   # Wave 2 ground truth
        "AVBL", "AVBR",                                     # un-validated forward command
        "PVCL", "PVCR",                                     # un-validated touch interneuron
        "ASHL", "ASHR",                                     # un-validated polymodal sensory (ASE substitute)
    ]
    for n in target_neurons:
        expr = panel["expression"].get(n, {})
        for gene, channel in GENE_TO_CHANNEL.items():
            tpm = expr.get(gene, 0.0)
            if tpm > 0:
                inventory_rows.append({
                    "neuron": n,
                    "gene": gene,
                    "wave2_channel": channel,
                    "tpm": tpm,
                    "biophysical_validation": "WAVE_2_GROUND_TRUTH" if n in (
                        "AVAL", "AVAR", "AIYL", "AIYR", "RIML", "RIMR"
                    ) else "EQUATION_DERIVED_PREDICTION",
                })
    inventory_path = OUT_CENGEN / "cengen_channel_inventory.csv"
    with open(inventory_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory_rows[0].keys()))
        w.writeheader()
        w.writerows(inventory_rows)
    print(f"  Inventory: {len(inventory_rows)} (neuron, channel) entries → {inventory_path}")

    # CP C.2 — Calibration
    print("\nCP C.2 — Calibration (AVAL, AVAR, AIY, RIM)")
    cal = calibrate_alpha(panel, ["AVAL", "AVAR", "AIY", "RIM"])
    print(f"  Channels with calibration data: {list(cal.keys())}")
    for ch, info in cal.items():
        print(f"    {ch}: n={info['n_cells']}, "
              f"α median={info['alpha_median']:.4f} nS/TPM, "
              f"spread {info['alpha_spread_ratio']}×")

    # Leave-one-out
    print("\n  Leave-one-out validation:")
    loo = loo_validation(panel, ["AVAL", "AVAR", "AIY", "RIM"])
    loo_log_errs = []
    for entry in loo:
        held = entry["held_out"]
        for pred in entry["predictions"]:
            if pred["log10_err"] is not None:
                loo_log_errs.append(abs(pred["log10_err"]))
        n_preds = len(entry["predictions"])
        if entry["predictions"]:
            mean_log_err = statistics.mean(abs(p["log10_err"]) for p in entry["predictions"]
                                            if p["log10_err"] is not None)
            print(f"    Hold out {held}: {n_preds} predictions, "
                  f"mean |log_err| = {mean_log_err:.3f}")

    md_path = OUT_CENGEN / "expression_to_conductance_calibration.md"
    with open(md_path, "w") as f:
        f.write("# CP C.1 + C.2 — CeNGEN-equation-coupling calibration\n\n")
        f.write("**Date:** 2026-04-28 (Wave P / Session 2 / Path C)\n\n")
        f.write("**Goal:** investigate whether canonical equations + CeNGEN gene "
                "expression can predict ion channel conductances for cells without "
                "published biophysical electrophysiology — the path past the "
                "literature cap.\n\n")
        f.write("**Critical methodology discipline:** the equation-derived predictions "
                "produced by this work block are FALSIFIABLE PREDICTIONS, not validated "
                "models. They are explicitly labeled as such throughout the artifacts.\n\n")

        f.write("## CP C.1 — Inventory\n\n")
        f.write(f"CeNGEN panel (Taylor et al. 2021): **{panel['_meta']['total_neurons']} neurons × "
                f"{panel['_meta']['total_panel_genes']} panel genes** (TPM values).\n\n")
        f.write(f"Channel-relevant genes mapped to Wave 2 cell-builder channel names: "
                f"**{len(GENE_TO_CHANNEL)} mappings**.\n\n")
        f.write(f"Channels NOT in CeNGEN panel (use cell-builder defaults instead): "
                f"`leak` (not gene-encoded; reflects passive membrane), some IRK subunits.\n\n")
        f.write(f"Inventory CSV: `cengen_coupling/cengen_channel_inventory.csv` "
                f"({len(inventory_rows)} rows).\n\n")

        f.write("## CP C.2 — Linear-scaling calibration\n\n"
                "Approach: g_nS = α × TPM, fit α per channel using Wave 2 cells "
                "(AVAL, AVAR, AIY, RIM) where both ground-truth conductance (Nicoletti) "
                "and CeNGEN expression exist. The simplest model; if α is reasonably "
                "convergent across cells (low spread ratio), linear scaling is "
                "informative. If spread is high (>10×), more sophisticated mapping "
                "(Hill function, per-channel-class) is required.\n\n")

        f.write("### Per-channel calibration\n\n")
        f.write("| channel | n cells | α median (nS/TPM) | α range | spread ratio |\n")
        f.write("|---|---|---|---|---|\n")
        for ch, info in cal.items():
            f.write(f"| {ch} | {info['n_cells']} | {info['alpha_median']} | "
                    f"[{info['alpha_min']}, {info['alpha_max']}] | "
                    f"{info['alpha_spread_ratio']}× |\n")
        f.write("\n")

        # Verdict on linear calibration quality
        spreads = [info["alpha_spread_ratio"] for info in cal.values()
                    if info["alpha_spread_ratio"] is not None and info["n_cells"] >= 2]
        median_spread = statistics.median(spreads) if spreads else float("inf")
        f.write(f"### Calibration verdict\n\n")
        f.write(f"Median α spread across channels with ≥2 cells: **{median_spread:.1f}×**.\n\n")
        if median_spread <= 5.0:
            f.write("**LINEAR SCALING DEFENSIBLE** — α convergent within 5× across cells.\n\n")
        elif median_spread <= 30.0:
            f.write("**LINEAR SCALING MARGINAL** — α scatter is large enough that "
                    "predictions for new cells should report uncertainty bounds, not "
                    "point estimates. Hill function or per-channel-class calibration "
                    "may improve fit.\n\n")
        else:
            f.write("**LINEAR SCALING INSUFFICIENT** — α scatter is too large for "
                    "predictive use. Either (a) TPM-to-conductance is genuinely non-linear "
                    "(Hill, threshold, saturation), (b) different cells use different "
                    "post-translational regulation, or (c) CeNGEN panel doesn't capture "
                    "the dominant ion channel subunits for these cells. Path C requires "
                    "more sophisticated mapping or wider gene panel.\n\n")

        f.write("### Leave-one-out validation\n\n")
        f.write("| held-out | n predictions | mean |log10_err| | per-channel breakdown |\n")
        f.write("|---|---|---|---|\n")
        for entry in loo:
            held = entry["held_out"]
            preds = entry["predictions"]
            if not preds:
                continue
            n = len(preds)
            errs = [abs(p["log10_err"]) for p in preds if p["log10_err"] is not None]
            mean_err = round(statistics.mean(errs), 3) if errs else "n/a"
            breakdown = ", ".join([
                f"{p['channel']}: log_err {p['log10_err']:+.2f}"
                for p in preds if p['log10_err'] is not None
            ])
            f.write(f"| {held} | {n} | {mean_err} | {breakdown} |\n")
        f.write("\n")

        loo_overall = round(statistics.mean(loo_log_errs), 3) if loo_log_errs else "n/a"
        f.write(f"**Overall LOO mean |log10_err|: {loo_overall}**\n\n")
        f.write("Interpretation: this is the predictive accuracy when applying the "
                "calibration to a held-out Wave 2 cell. If mean |log10_err| < 0.5 "
                "(within ~3×), the calibration generalizes; if > 1.0 (10×), the "
                "calibration overfits to the training cells.\n\n")

    state = {
        "checkpoint": "path_c_cp1_c2",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration": cal,
        "loo_overall_log_err": loo_overall,
        "median_spread_ratio": median_spread,
    }
    json.dump(state, open(CHK, "w"), indent=2, default=str)

    print(f"\n  Overall LOO mean |log10_err|: {loo_overall}")
    print(f"\n  MD: {md_path}")
    print(f"  Inventory CSV: {inventory_path}")
    print(f"  Checkpoint: {CHK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
