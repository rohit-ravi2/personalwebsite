#!/usr/bin/env python3
"""P0 #2 — Build a compact CeNGEN gene-expression panel for the dashboard.

CeNGEN (Taylor et al. 2021) profiles 92 neuron *classes* by single-cell
RNA-seq. Our 300-neuron simulator uses individual left/right neurons
(ASHL, ASHR). This script:

  1) Parses the WS297 gene_association file to build
     WBGene00000001 → gene_name mappings (e.g. flp-1, unc-2, mec-4).
  2) Loads CeNGEN mean TPM per neuron class.
  3) Filters to a hand-curated panel of ~120 biologically load-bearing
     genes: ion channels, receptors, neuropeptides, NT markers,
     sensory transduction cascades.
  4) Broadcasts class-level TPM to per-neuron by treating L/R (and
     subscripted) sibling neurons as identical draws from the class.
  5) Writes public/data/cengen-panel.json for the dashboard polar
     bar chart in the locked-neuron popover.

The panel is organised into categories so the UI can render grouped
radial sections rather than 120 uniform bars.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CENGEN_CSV = Path(
    "/home/rohit/Desktop/C-Elegans/data/expression/cengen/derived/"
    "expression_neuron_mean.csv"
)
GENE_ASSOC = Path(
    "/home/rohit/Desktop/C-Elegans/data/wormbase_release_WS297/associations/"
    "c_elegans.PRJNA13758.WS297.gene_association.wb"
)
OUT_JSON = ROOT / "public" / "data" / "cengen-panel.json"
CONNECTOME_NPZ = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"


# Curated panel — grouped for the UI. Each group maps display_label →
# list of gene_name canonical forms (lowercase-hyphen).
PANEL: dict[str, list[str]] = {
    "NT marker": [
        "cho-1", "unc-17",       # ACh
        "eat-4",                 # Glu
        "unc-25", "unc-47",      # GABA
        "tph-1",                 # 5HT
        "cat-2",                 # DA
        "tbh-1",                 # OA
        "tdc-1",                 # TA
    ],
    "Voltage Ca": [
        "unc-2", "egl-19", "cca-1",    # Ca_v1 / Ca_v2 / T-type
    ],
    "Voltage K": [
        "shl-1", "shk-1", "kvs-1", "exp-2", "unc-103",
        "slo-1", "slo-2",
    ],
    "Voltage Na": [
        # C. elegans has no classical Na_v, but let's include what's annotated
        "nca-1", "nca-2", "unc-77", "unc-80",
    ],
    "AChR": [
        "acr-2", "acr-5", "acr-14", "acr-16", "unc-38", "unc-29",
        "unc-63", "lev-1", "lev-8",
    ],
    "GluR/GluCl": [
        "glr-1", "glr-2", "glr-3", "glr-4", "glr-5", "glr-6",
        "glc-1", "glc-2", "glc-3", "glc-4", "avr-14", "avr-15", "nmr-1",
    ],
    "GABAR": [
        "unc-49", "lgc-35", "lgc-36", "lgc-37", "lgc-38", "exp-1", "gab-1",
    ],
    "Peptide rx": [
        "npr-1", "npr-2", "npr-3", "npr-4", "npr-5", "npr-9",
        "npr-11", "npr-12", "npr-22",
        "dmsr-1", "dmsr-2", "dmsr-7",
        "frpr-3", "frpr-18", "frpr-19",
    ],
    "Monoamine rx": [
        "mod-1", "ser-1", "ser-4", "ser-5", "ser-7",   # 5HT
        "dop-1", "dop-2", "dop-3", "dop-4",            # DA
        "octr-1", "ser-3", "ser-6",                    # OA
        "tyra-2", "tyra-3", "ser-2",                   # TA
        "pdfr-1",                                      # PDF
    ],
    "Peptide gene": [
        "flp-1", "flp-2", "flp-6", "flp-11", "flp-14", "flp-18", "flp-21",
        "nlp-1", "nlp-3", "nlp-12", "nlp-18", "nlp-22", "nlp-49",
        "pdf-1", "pdf-2",
        "ins-1", "ins-6", "ins-11", "ins-18",
    ],
    "Sensory GCY": [
        "gcy-8", "gcy-14", "gcy-18", "gcy-22", "gcy-23", "gcy-35", "gcy-36",
    ],
    "CNG": [
        "tax-2", "tax-4",
    ],
    "TRP": [
        "osm-9", "ocr-2", "trp-1", "trp-2", "trpa-1",
    ],
    "DEG/ENaC": [
        "mec-4", "mec-10", "asic-1", "asic-2", "unc-105", "del-1",
    ],
    "Innexin": [
        "unc-7", "unc-9", "inx-1", "inx-3", "inx-4", "inx-13", "inx-19",
        "che-7",
    ],
    "Insulin/FOXO": [
        "daf-2", "daf-16", "age-1", "ist-1", "akt-1",
    ],
}


# L/R resolution: neuron → class (drop trailing L/R and/or numeric).
_CLASS_RE = re.compile(r"^([A-Z][A-Z0-9]*?)([LR]?)(\d*)$")


def neuron_to_class(name: str) -> str:
    """Map individual neuron name to its CeNGEN class label.

    Examples:
        ASHL → ASH, AWCON → AWC, VA12 → VA12, DA9 → DA9, ASE → ASE,
        AVAL → AVA, PVCR → PVC, RMDVL → RMDV (if in list, else RMD).
    """
    # Known special cases
    special = {
        "AWCON": "AWC", "AWCOFF": "AWC",
        "VA12": "VA12", "DA9": "DA9",
    }
    if name in special:
        return special[name]
    m = _CLASS_RE.match(name)
    if not m:
        return name
    base, lr, num = m.groups()
    # If trailing digit exists (e.g. VA1, VA2, ..., VA12), keep base+num
    # for classes that CeNGEN represents per-cell (DA9, VA12).
    if num and not lr:
        return name
    return base


def load_gene_map() -> dict[str, str]:
    """WBGene → canonical gene name (from gene_association GAF 2.2)."""
    out: dict[str, str] = {}
    with GENE_ASSOC.open() as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            wb = parts[1]
            name = parts[2]
            if wb.startswith("WBGene") and wb not in out:
                out[wb] = name
    return out


def main() -> None:
    print("loading gene name map…")
    gene_map = load_gene_map()  # WBGene00001444 → 'flp-1'
    name_to_wb = {v: k for k, v in gene_map.items()}

    print(f"loaded {len(gene_map)} genes.")

    # Which of our curated panel genes actually resolve to a WBGene?
    resolved: dict[str, str] = {}       # gene_name → WBGene
    missing: list[str] = []
    for cat, gene_names in PANEL.items():
        for g in gene_names:
            wb = name_to_wb.get(g)
            if wb is None:
                missing.append(g)
            else:
                resolved[g] = wb
    if missing:
        print(f"unresolved panel genes ({len(missing)}): "
              f"{', '.join(missing[:20])}{'…' if len(missing) > 20 else ''}")

    # Load CeNGEN matrix — rows = neuron class, cols = WBGene ids
    print("loading CeNGEN expression matrix…")
    with CENGEN_CSV.open() as f:
        header = next(csv.reader(f))
        wb_cols = header[1:]  # first col is neuron name
        wb_to_col = {wb: i for i, wb in enumerate(wb_cols)}
        # Build target column indices for resolved panel genes
        panel_cols: dict[str, int] = {}
        for gname, wb in resolved.items():
            if wb in wb_to_col:
                panel_cols[gname] = wb_to_col[wb]
        print(f"found {len(panel_cols)} of {len(resolved)} panel genes "
              f"in CeNGEN matrix.")

        # Stream rows, keeping only panel columns
        tpm_by_class: dict[str, dict[str, float]] = {}
        for row in csv.reader(f):
            if not row:
                continue
            neuron = row[0]
            vals = {gname: float(row[idx + 1])
                    for gname, idx in panel_cols.items()}
            tpm_by_class[neuron] = vals

    print(f"loaded {len(tpm_by_class)} CeNGEN neuron classes.")

    # Map to our 300-neuron brain via connectome.npz
    all_neurons: list[str] = []
    if CONNECTOME_NPZ.exists():
        cn = np.load(CONNECTOME_NPZ, allow_pickle=True)
        all_neurons = [str(s) for s in cn["names"]]
    else:
        print("connectome.npz not found — panel will include CeNGEN "
              "neuron classes only.")
        all_neurons = sorted(tpm_by_class.keys())

    # Per-neuron TPM — fall back to class-level when L/R mapping doesn't
    # resolve directly.
    per_neuron: dict[str, dict[str, float]] = {}
    unmapped: list[str] = []
    for n in all_neurons:
        klass = neuron_to_class(n)
        if klass in tpm_by_class:
            per_neuron[n] = tpm_by_class[klass]
        elif n in tpm_by_class:
            per_neuron[n] = tpm_by_class[n]
        else:
            unmapped.append(n)
    if unmapped:
        print(f"{len(unmapped)} neurons had no CeNGEN class match "
              f"(e.g. {', '.join(unmapped[:8])}); omitted from panel.")

    # Compute panel max per gene (for UI normalisation) so one
    # high-expressing neuron doesn't drown out dynamic range.
    gene_max: dict[str, float] = {}
    for gname in panel_cols:
        gene_max[gname] = max(
            (row.get(gname, 0.0) for row in per_neuron.values()),
            default=1.0,
        )
        if gene_max[gname] <= 0:
            gene_max[gname] = 1.0

    # Quantise TPMs to 2 decimals to keep the JSON compact
    payload = {
        "_meta": {
            "description": "CeNGEN (Taylor 2021) mean TPM for a curated "
                           "panel of C. elegans channels/receptors/peptides, "
                           "mapped to the 300-neuron brain.",
            "groups": list(PANEL.keys()),
            "genes_by_group": {k: [g for g in v if g in panel_cols]
                               for k, v in PANEL.items()},
            "gene_max_tpm": {g: round(v, 2) for g, v in gene_max.items()},
            "total_neurons": len(per_neuron),
            "total_panel_genes": len(panel_cols),
        },
        "expression": {
            n: {g: round(v, 2) for g, v in row.items() if v > 0.05}
            for n, row in per_neuron.items()
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT_JSON.stat().st_size / 1024
    print(f"wrote {OUT_JSON}: {kb:.1f} KB")


if __name__ == "__main__":
    main()
