"""Stage 6 — Per-target and per-anesthetic rank correlation against clinical potency.

Uses existing 540-docking dataset; no new computation. Produces:

(A) Per-target Spearman rank correlation between predicted affinity (-ΔG = stronger
    binding) and a published clinical-potency proxy.
(B) Per-anesthetic Spearman rank correlation across all targets in the panel.
(C) Stratified: per-target rank for anesthetics, per-mechanism-class rank.

Clinical potency proxy: -log10(clinical_aqueous_EC50_uM). Lower EC50 = higher
potency = larger -log10. Anesthetics ranked etomidate (most potent on aqueous
basis: EC50 0.3 µM) > propofol (1) > isoflurane (290) ≈ sevoflurane (230) ≈
halothane (340) >> ketamine (5000 µM, weak on aqueous basis).

Important caveat: clinical potency reflects WHOLE-ANIMAL behavioral effect,
which is the integral of multi-target effects. A single target may not order
anesthetics in the same way as whole-animal potency. The rank-correlation
test is therefore SOFT — strong correlation supports the framing, weak
correlation is interpretable rather than failing.

Output: artifacts/calibration/stage6_rank_correlation.{csv,md}

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/calibration_stage6_rank_correlation.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "anesthetics" / "anesthetic_panel.csv"
BEST_OCC = ROOT / "artifacts" / "occupancy" / "best_pocket_per_target.csv"
TIER1 = ROOT / "targets" / "tier1_targets.csv"
OUT_CSV = ROOT / "artifacts" / "calibration" / "stage6_rank_correlation.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "stage6_rank_correlation.md"


def spearman(x: list[float], y: list[float]) -> tuple[float, int]:
    """Spearman ρ via rank-Pearson. Returns (ρ, n)."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan"), len(x)
    def ranks(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        # Average ranks for ties
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j+1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mean_rx = sum(rx) / len(rx)
    mean_ry = sum(ry) / len(ry)
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = math.sqrt(sum((a - mean_rx) ** 2 for a in rx))
    var_y = math.sqrt(sum((b - mean_ry) ** 2 for b in ry))
    if var_x == 0 or var_y == 0:
        return float("nan"), len(x)
    return cov / (var_x * var_y), len(x)


def load_panel() -> dict[str, dict]:
    out = {}
    with open(PANEL) as f:
        for r in csv.DictReader(f):
            name = r["name"].strip().lower()
            try:
                ec50 = float(r["clinical_aqueous_EC50_uM"])
            except (ValueError, KeyError):
                ec50 = float("nan")
            out[name] = {"ec50_uM": ec50, "log_potency": -math.log10(ec50) if ec50 > 0 else float("nan")}
    return out


def load_target_classes() -> dict[str, str]:
    out = {}
    if not TIER1.exists():
        return out
    with open(TIER1) as f:
        for r in csv.DictReader(f):
            out[r["gene_name"].strip()] = r.get("mechanism_class", "").strip()
    return out


def load_best_occ() -> list[dict]:
    out = []
    with open(BEST_OCC) as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def main() -> int:
    panel = load_panel()
    classes = load_target_classes()
    rows = load_best_occ()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Group by target
    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["gene"], []).append(r)

    # PART A — per-target rank correlation between predicted affinity and clinical potency
    per_target_rows = []
    for gene, group in sorted(by_target.items()):
        pairs = []
        for r in group:
            ane = r["anesthetic"].strip().lower()
            ec50 = panel.get(ane, {}).get("ec50_uM")
            if ec50 is None or math.isnan(ec50):
                continue
            try:
                aff = float(r["best_affinity_kcal_per_mol"])
            except ValueError:
                continue
            # Predicted "potency" proxy: -ΔG (more negative ΔG → larger -ΔG → "more potent")
            pred_potency = -aff
            log_clin_potency = -math.log10(ec50) if ec50 > 0 else float("nan")
            if math.isnan(log_clin_potency):
                continue
            pairs.append((ane, pred_potency, log_clin_potency))
        if len(pairs) < 3:
            continue
        rho, n = spearman([p[1] for p in pairs], [p[2] for p in pairs])
        # Order anesthetics by predicted potency (descending)
        pairs_sorted_by_pred = sorted(pairs, key=lambda t: -t[1])
        pairs_sorted_by_clin = sorted(pairs, key=lambda t: -t[2])
        per_target_rows.append({
            "gene": gene,
            "mechanism_class": classes.get(gene, ""),
            "n_anesthetics": n,
            "spearman_rho": rho,
            "predicted_order": ",".join(p[0] for p in pairs_sorted_by_pred),
            "clinical_potency_order": ",".join(p[0] for p in pairs_sorted_by_clin),
        })

    # PART B — per-anesthetic correlation across targets — i.e., do the same anesthetic
    # rank targets the same way as expected mechanism-class affinity?
    # This is harder to define without per-target experimental data; for now report
    # the spread (std of per-target predicted -ΔG per anesthetic).
    by_ane: dict[str, list[float]] = {}
    for r in rows:
        try:
            aff = float(r["best_affinity_kcal_per_mol"])
        except ValueError:
            continue
        by_ane.setdefault(r["anesthetic"].strip().lower(), []).append(-aff)

    # PART C — per-mechanism-class aggregation
    by_class: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        gene = r["gene"]
        cls = classes.get(gene, "unknown")
        try:
            aff = float(r["best_affinity_kcal_per_mol"])
        except ValueError:
            continue
        by_class.setdefault(cls, []).append((r["anesthetic"].strip().lower(), -aff))

    # Within each mechanism class: does the same rank-of-anesthetics pattern hold?
    class_rows = []
    for cls, entries in sorted(by_class.items()):
        # Group entries by anesthetic, average -ΔG across targets in that class
        by_ane_in_class: dict[str, list[float]] = {}
        for ane, p in entries:
            by_ane_in_class.setdefault(ane, []).append(p)
        avg_per_ane = {a: sum(v) / len(v) for a, v in by_ane_in_class.items()}
        if len(avg_per_ane) < 3:
            continue
        # Spearman: avg_per_ane vs clinical potency
        x = []
        y = []
        for a, avg in avg_per_ane.items():
            ec50 = panel.get(a, {}).get("ec50_uM")
            if ec50 is None or math.isnan(ec50):
                continue
            x.append(avg)
            y.append(-math.log10(ec50))
        if len(x) < 3:
            continue
        rho, n = spearman(x, y)
        sorted_anes = sorted(avg_per_ane.items(), key=lambda kv: -kv[1])
        class_rows.append({
            "mechanism_class": cls,
            "n_targets_in_class": len(set(g for g in by_target if classes.get(g, "") == cls)),
            "n_anesthetics": n,
            "spearman_rho_avg_potency": rho,
            "predicted_avg_potency_order": ",".join(a for a, _ in sorted_anes),
        })

    # Write CSVs
    fieldnames = ["gene", "mechanism_class", "n_anesthetics", "spearman_rho",
                  "predicted_order", "clinical_potency_order"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_target_rows:
            r["spearman_rho"] = f"{r['spearman_rho']:+.3f}"
            w.writerow(r)
    print(f"Per-target rank correlation: {OUT_CSV}")

    # Stdout summary
    print()
    print("=" * 60)
    print("STAGE 6 — Rank correlation (predicted vs clinical potency)")
    print("=" * 60)
    pos = sum(1 for r in per_target_rows if float(r["spearman_rho"]) > 0)
    neg = sum(1 for r in per_target_rows if float(r["spearman_rho"]) < 0)
    pos_strong = sum(1 for r in per_target_rows if float(r["spearman_rho"]) > 0.5)
    print(f"Per-target ρ summary across {len(per_target_rows)} targets:")
    print(f"  ρ > 0:    {pos}/{len(per_target_rows)}")
    print(f"  ρ > 0.5:  {pos_strong}/{len(per_target_rows)}")
    print(f"  ρ < 0:    {neg}/{len(per_target_rows)}")
    print()
    sorted_by_rho = sorted(per_target_rows, key=lambda r: -float(r["spearman_rho"]))
    print(f"Top 8 by ρ:")
    for r in sorted_by_rho[:8]:
        print(f"  {r['gene']:10s}  {r['mechanism_class']:25s}  ρ={r['spearman_rho']}  "
              f"predicted={r['predicted_order'][:40]}")
    print()
    print(f"Bottom 5 by ρ:")
    for r in sorted_by_rho[-5:]:
        print(f"  {r['gene']:10s}  {r['mechanism_class']:25s}  ρ={r['spearman_rho']}  "
              f"predicted={r['predicted_order'][:40]}")
    print()
    print("Per mechanism-class avg potency Spearman:")
    for r in class_rows:
        rho_str = f"{r['spearman_rho_avg_potency']:+.3f}" if not math.isnan(r['spearman_rho_avg_potency']) else "nan"
        print(f"  {r['mechanism_class']:25s}  ρ={rho_str}  "
              f"predicted_order={r['predicted_avg_potency_order'][:60]}")

    # Markdown summary
    with open(OUT_MD, "w") as f:
        f.write("# Stage 6 — Rank correlation analysis\n\n")
        f.write("## Method\n\n")
        f.write("For each target, compute Spearman ρ between predicted affinity (-ΔG) "
                "and clinical-potency proxy (-log10 of clinical aqueous EC50 µM). "
                "Strong positive ρ supports the multi-target framing; scrambled / negative "
                "ρ suggests pipeline doesn't track clinical potency at the per-target level.\n\n")
        f.write(f"Reference clinical EC50 (µM): " +
                ", ".join(f"{a}={p['ec50_uM']:.1f}" for a, p in sorted(panel.items())) + "\n\n")
        f.write(f"Implied clinical potency rank (highest→lowest, by aqueous EC50): " +
                ", ".join(a for a, p in sorted(panel.items(), key=lambda kv: kv[1]['ec50_uM'])) + "\n\n")
        f.write("## Per-target Spearman ρ\n\n")
        f.write("| gene | class | ρ | predicted order | clinical potency order |\n")
        f.write("|---|---|---|---|---|\n")
        for r in sorted_by_rho:
            f.write(f"| {r['gene']} | {r['mechanism_class']} | {r['spearman_rho']} | "
                    f"{r['predicted_order']} | {r['clinical_potency_order']} |\n")
        f.write("\n## Per mechanism-class average\n\n")
        f.write("| class | n_targets | n_anesthetics | ρ (avg potency) | predicted avg-potency order |\n")
        f.write("|---|---|---|---|---|\n")
        for r in class_rows:
            rho_str = f"{r['spearman_rho_avg_potency']:+.3f}" if not math.isnan(r['spearman_rho_avg_potency']) else "nan"
            f.write(f"| {r['mechanism_class']} | {r['n_targets_in_class']} | {r['n_anesthetics']} | "
                    f"{rho_str} | {r['predicted_avg_potency_order']} |\n")
        f.write("\n## Headline\n\n")
        f.write(f"- Per-target ρ > 0: {pos}/{len(per_target_rows)}\n")
        f.write(f"- Per-target ρ > 0.5: {pos_strong}/{len(per_target_rows)}\n")
        f.write(f"- Median ρ: {sorted([float(r['spearman_rho']) for r in per_target_rows])[len(per_target_rows)//2]:+.3f}\n")
        f.write("\n## Caveat\n\n")
        f.write("Clinical aqueous EC50 reflects whole-animal behavioral effect — the "
                "INTEGRAL of multi-target perturbation. A single target may not order "
                "anesthetics the same way as whole-animal potency. Strong rank correlation "
                "supports the framing. Weak correlation is interpretable (this target is "
                "not a primary potency driver) rather than disqualifying.\n")

    print(f"\nMarkdown: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
