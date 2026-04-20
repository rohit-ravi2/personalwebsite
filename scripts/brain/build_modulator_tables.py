#!/usr/bin/env python3
"""Phase 3d-1 — Build neuropeptide + monoamine modulator tables from
CeNGEN single-cell expression.

For each of 9 modulators (5 neuropeptides + 4 monoamines), extract
per-neuron releaser and receptor expression from the CeNGEN atlas and
emit a runtime-ready table that the v3 Brian2 modulation layer will
consume.

Outputs: artifacts/modulator_tables.npz with, for each modulator:
  releasers[modulator]         — (N_conn,) bool, neurons above
                                  synthesis-gene threshold
  releaser_weights[modulator]  — (N_conn,) float, synthesis expression
                                  (for continuous release modeling)
  target_weights[modulator]    — (N_conn,) float, signed sum of
                                  receptor expression × sign per receptor
  receptors[modulator]         — list[str], receptor gene symbols used
  releaser_gene[modulator]     — str, synthesis gene symbol used
  tau_s[modulator]             — float, concentration decay timescale (s)

Data sources:
  - CeNGEN per-neuron-class expression:
      data/expression/cengen/derived/expression_neuron_mean.csv
  - WormBase gene symbol resolution:
      data/wormbase_release_WS297/associations/…gene_association.wb
  - 300-neuron canonical set: scripts/brain/artifacts/connectome.npz
  - Receptor sign table: compiled below from:
      Brockie 2001, Maricq 1995 (iGluR),
      Ranganathan 2000, Tsalik 2003 (MOD-1, SER-*),
      Chase 2004, Suo 2003, Sanyal 2004 (dopamine receptors),
      Alkema 2005, Donnelly 2013 (tyramine),
      Roeder 2005 (octopamine),
      Turek 2016 (FLP-11 receptors),
      Bhattacharya 2014, Oranth 2018 (FLP-1, FLP-2),
      Janssen 2008 (PDF-1).

Threshold for calling a neuron a releaser: synthesis gene expression
≥ 5× median expression across all 91 neuron classes AND ≥ 5 TPM.
This is conservative — matches the CeNGEN "threshold 4" convention
(Taylor et al. 2021).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CELEGANS = Path("/home/rohit/Desktop/C-Elegans")
CENGEN_MEAN = CELEGANS / "data" / "expression" / "cengen" / "derived" / "expression_neuron_mean.csv"
GENE_ASSOC = CELEGANS / "data" / "wormbase_release_WS297" / "associations" / "c_elegans.PRJNA13758.WS297.gene_association.wb"
CONN_NPZ = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"
OUT = Path(__file__).resolve().parent / "artifacts" / "modulator_tables.npz"
OUT_META = OUT.with_suffix(".json")


# --------------------- Modulator definitions -------------------------
# Each entry: key modulator, then synthesis gene + receptor list with signs.
# Sign conventions for LIF modulation current:
#   +1 = receptor activation → net excitatory slow current
#   -1 = receptor activation → net inhibitory slow current
# The compiled signs reflect consensus from cited references (see
# module docstring).

PHARYNGEAL_NEURONS = {
    "M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5",
    "MCL", "MCR", "MI", "I1L", "I1R", "I2L", "I2R",
    "I3", "I4", "I5", "I6",
}

MODULATORS = {
    # --- Neuropeptides ---------------------------------------------
    "FLP-11": {
        "synthesis_gene": "flp-11",
        "receptors": {"npr-1": -1, "npr-22": -1, "dmsr-1": -1,
                       "dmsr-7": -1, "npr-11": -1},
        "tau_s": 20.0,
        "note": "Released by RIS → broad inhibition for sleep-like quiescence (Turek 2016).",
    },
    "FLP-1": {
        "synthesis_gene": "flp-1",
        "receptors": {"npr-4": -1, "npr-5": -1, "npr-11": -1},
        "tau_s": 15.0,
        "note": "AVK-derived; modulates forward/reverse balance (Bhattacharya 2014).",
    },
    "FLP-2": {
        "synthesis_gene": "flp-2",
        "receptors": {"npr-30": -1, "frpr-18": -1},
        "tau_s": 15.0,
        "note": "Exploratory/exploitatory transitions (Oranth 2018).",
    },
    "NLP-12": {
        "synthesis_gene": "nlp-12",
        "receptors": {"ckr-1": +1, "ckr-2": +1},
        "tau_s": 10.0,
        "note": "DVA-derived proprioceptive modulation of reversal frequency (Hu 2011).",
    },
    "PDF-1": {
        "synthesis_gene": "pdf-1",
        "receptors": {"pdfr-1": +1},
        "tau_s": 30.0,
        "note": "Arousal / exploratory drive (Janssen 2008).",
    },
    # --- Monoamines ------------------------------------------------
    "5HT": {
        "synthesis_gene": "tph-1",   # tryptophan hydroxylase
        "receptors": {"mod-1": -1,   # chloride channel, inhibitory
                       "ser-1": +1,  # Gq excitatory
                       "ser-4": -1,  # Gi inhibitory
                       "ser-5": +1,  # Gq excitatory
                       "ser-6": +1,  # Gs excitatory
                       "ser-7": +1}, # Gs excitatory
        "tau_s": 5.0,
        # v3.1 (Phase 3d-5): exclude pharyngeal neurons from NSM 5HT
        # targets. Pharyngeal 5HT is released by a separate source
        # (MC/I1/RIP) and acts on pharyngeal targets anatomically
        # isolated from NSM-derived head 5HT. Without this exclusion,
        # pharyngeal MOD-1 / SER-1 / SER-7 expression dominates the
        # target vector and masks the locomotion-relevant SER-4 signal
        # on AVB. Phase 3d-3 perturbation study flagged this (NSM
        # ablation went wrong direction).
        "target_exclude": PHARYNGEAL_NEURONS,
        "note": "NSM/HSN/ADF serotonin — dwelling, feeding state (Tsalik 2003). Pharyngeal excluded (v3.1).",
    },
    "DA": {
        "synthesis_gene": "cat-2",   # tyrosine hydroxylase
        "receptors": {"dop-1": +1,   # Gq excitatory
                       "dop-2": -1,  # Gi inhibitory
                       "dop-3": -1,  # Gi inhibitory
                       "dop-4": +1,  # Gq excitatory
                       "dop-5": -1,  # Gi inhibitory
                       "dop-6": +1}, # Gq excitatory
        "tau_s": 4.0,
        "note": "PDE/ADE/CEP dopamine — pirouette duration, roaming (Chase 2004).",
    },
    "TA": {
        "synthesis_gene": "tdc-1",   # tyrosine decarboxylase
        "receptors": {"ser-2": -1,   # Gi inhibitory
                       "lgc-55": -1, # chloride channel, inhibitory
                       "tyra-2": -1, # Gi inhibitory
                       "tyra-3": +1},# Gq excitatory
        "tau_s": 4.0,
        "note": "RIM tyramine — reversal gate, head-steering gain (Alkema 2005).",
    },
    "OA": {
        "synthesis_gene": "tbh-1",   # tyramine β-hydroxylase (OA-specific)
        "receptors": {"ser-3": +1,   # Gq excitatory
                       "ser-6": +1,  # Gs excitatory (shared with 5HT)
                       "octr-1": -1},# Gi inhibitory
        "tau_s": 4.0,
        "note": "RIC octopamine — stress arousal (Roeder 2005).",
    },
}


# --------------------- Helpers ---------------------------------------

def load_gene_symbol_map() -> dict[str, str]:
    """Return WBGene → symbol dict from gene association file."""
    mapping: dict[str, str] = {}
    with open(GENE_ASSOC) as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            wb, sym = parts[1], parts[2]
            if wb.startswith("WBGene") and sym:
                mapping[wb] = sym
    return mapping


def load_cengen_matrix() -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (neuron_class × gene matrix, symbol→WBGene map).
    Rows are neuron classes (ADA, ADE, ...), columns are WBGene IDs."""
    df = pd.read_csv(CENGEN_MEAN, index_col=0)
    df.index.name = "neuron_class"
    print(f"  CeNGEN matrix: {df.shape[0]} neuron classes × {df.shape[1]} genes")

    wb_to_sym = load_gene_symbol_map()
    sym_to_wb = {sym: wb for wb, sym in wb_to_sym.items()}
    return df, sym_to_wb


# Map CeNGEN neuron class names to our 300-neuron connectome names.
# CeNGEN uses classes (ADA) while connectome uses L/R pairs (ADAL/ADAR).
# Mapping: most classes expand to *L / *R; a few specials below.
SPECIAL_CLASS_MAP = {
    # CeNGEN uses "AVA" → connectome has AVAL, AVAR
    # Default: classname → [classname+L, classname+R]
    # But some classes are: AIN, AIZ, etc. Use same default rule.
    # Specials (unpaired or differently-named):
    "AVG": ["AVG"],   # unpaired
    "AVL": ["AVL"],
    "AVM": ["AVM"],
    "AQR": ["AQR"],
    "DVA": ["DVA"],
    "DVB": ["DVB"],
    "DVC": ["DVC"],
    "PDA": ["PDA"],
    "PVP": ["PVPL", "PVPR"],
    "PVQ": ["PVQL", "PVQR"],
    "PVR": ["PVR"],
    "PVT": ["PVT"],
    "PQR": ["PQR"],
    "RID": ["RID"],
    "RIH": ["RIH"],
    "RIS": ["RIS"],
    "ALA": ["ALA"],
    # Pharyngeal — check these manually
    "M1": ["M1"], "M2": ["M2L", "M2R"], "M3": ["M3L", "M3R"],
    "M4": ["M4"], "M5": ["M5"], "MC": ["MCL", "MCR"],
    "MI": ["MI"], "I1": ["I1L", "I1R"], "I2": ["I2L", "I2R"],
    "I3": ["I3"], "I4": ["I4"], "I5": ["I5"], "I6": ["I6"],
    "NSM": ["NSML", "NSMR"],
    # Ring motor
    "RIV": ["RIVL", "RIVR"], "RIS": ["RIS"],
}


def expand_neuron_class(cls: str, conn_set: set[str]) -> list[str]:
    """Expand a CeNGEN neuron class name to specific connectome neurons."""
    if cls in SPECIAL_CLASS_MAP:
        candidates = SPECIAL_CLASS_MAP[cls]
    else:
        # Default: L/R pair
        candidates = [cls + "L", cls + "R"]
    # Filter to those actually in the connectome
    return [n for n in candidates if n in conn_set]


def map_cengen_to_connectome(df: pd.DataFrame, conn_names: list[str]
                              ) -> pd.DataFrame:
    """Expand CeNGEN class-level rows into connectome-neuron-level rows.
    Each connectome neuron gets the expression of its parent class.
    Returns (N_conn × genes) DataFrame indexed by connectome name."""
    conn_set = set(conn_names)
    rows: list[tuple[str, np.ndarray]] = []
    mapped_classes: set[str] = set()
    for cls in df.index:
        members = expand_neuron_class(cls, conn_set)
        if not members:
            continue
        mapped_classes.add(cls)
        for n in members:
            rows.append((n, df.loc[cls].values))

    mapped_names = [r[0] for r in rows]
    mapped_data = np.stack([r[1] for r in rows]) if rows else np.zeros(
        (0, df.shape[1])
    )
    out = pd.DataFrame(mapped_data, index=mapped_names, columns=df.columns)

    # Report coverage
    missing = [n for n in conn_names if n not in out.index]
    extra_classes = set(df.index) - mapped_classes
    print(f"  Connectome neurons covered by CeNGEN: "
          f"{len(set(out.index) & conn_set)} / {len(conn_set)}")
    if missing[:8]:
        print(f"  Not in CeNGEN (first 8): {missing[:8]}")
    if extra_classes:
        print(f"  CeNGEN classes not mapped to connectome: "
              f"{sorted(extra_classes)[:8]}")
    return out


# --------------------- Main ------------------------------------------

def main() -> None:
    # Load connectome neuron set
    conn = np.load(CONN_NPZ, allow_pickle=True)
    conn_names: list[str] = [str(s) for s in conn["names"]]
    name_to_idx = {n: i for i, n in enumerate(conn_names)}
    N = len(conn_names)
    print(f"Target: {N} connectome neurons")

    # Load CeNGEN + gene symbol map
    print("\n[1] Loading CeNGEN expression + gene-symbol map…")
    df, sym_to_wb = load_cengen_matrix()
    print(f"  Gene symbols resolved: {len(sym_to_wb)}")

    # Map CeNGEN → connectome names
    print("\n[2] Mapping CeNGEN classes → connectome neurons…")
    expr_by_neuron = map_cengen_to_connectome(df, conn_names)

    # For each modulator, compute releaser + target weight vectors
    releasers: dict[str, np.ndarray] = {}
    releaser_weights: dict[str, np.ndarray] = {}
    target_weights: dict[str, np.ndarray] = {}
    receptors_used: dict[str, list[str]] = {}
    releaser_gene_used: dict[str, str] = {}
    unmatched_genes: list[tuple[str, str]] = []

    print("\n[3] Extracting per-modulator tables:")
    print(f"  {'modulator':<10} {'syn gene':<10} {'#releasers':>11} "
          f"{'#recep genes':>12} {'#targets':>9}")
    print("  " + "-" * 58)

    for mod_name, spec in MODULATORS.items():
        syn_sym = spec["synthesis_gene"]
        syn_wb = sym_to_wb.get(syn_sym)
        if syn_wb is None or syn_wb not in expr_by_neuron.columns:
            unmatched_genes.append((mod_name, f"synthesis:{syn_sym}"))
            # Still emit zero vectors so the layer can ignore this modulator
            releasers[mod_name] = np.zeros(N, dtype=bool)
            releaser_weights[mod_name] = np.zeros(N, dtype=np.float32)
            target_weights[mod_name] = np.zeros(N, dtype=np.float32)
            receptors_used[mod_name] = []
            releaser_gene_used[mod_name] = syn_sym
            print(f"  {mod_name:<10} {syn_sym:<10}  (synthesis gene not found)")
            continue

        # Build releaser mask from synthesis gene expression.
        # Adaptive threshold: CeNGEN expression scales vary dramatically
        # between gene classes (peptide precursors like flp-11 reach
        # ~1000 at RIS; synthesis enzymes like tph-1 cap around ~3 at
        # HSN). A per-gene relative threshold works for both:
        #   threshold = max(0.25 × max_expression, 0.5)
        # This keeps the top releasers (within 4× of the max expressor)
        # and requires a minimum absolute floor of 0.5 to filter noise.
        syn_expr = expr_by_neuron[syn_wb].reindex(conn_names, fill_value=0.0)
        syn_vec = syn_expr.values.astype(np.float32)
        max_expr = float(syn_vec.max())
        if max_expr <= 0:
            thr = np.inf
        else:
            thr = max(0.25 * max_expr, 0.5)
        mask = syn_vec >= thr
        releasers[mod_name] = mask
        releaser_weights[mod_name] = syn_vec * mask.astype(np.float32)
        releaser_gene_used[mod_name] = syn_sym

        # Build target weight vector: Σ_r expression(r) × sign(r)
        tgt = np.zeros(N, dtype=np.float32)
        resolved_receptors: list[str] = []
        for rec_sym, sgn in spec["receptors"].items():
            rec_wb = sym_to_wb.get(rec_sym)
            if rec_wb is None or rec_wb not in expr_by_neuron.columns:
                unmatched_genes.append((mod_name, f"receptor:{rec_sym}"))
                continue
            rec_expr = expr_by_neuron[rec_wb].reindex(conn_names, fill_value=0.0)
            tgt += sgn * rec_expr.values.astype(np.float32)
            resolved_receptors.append(rec_sym)

        # Apply per-modulator target exclusion mask if specified (v3.1:
        # 5HT excludes pharyngeal neurons since they're anatomically
        # isolated from NSM-derived central 5HT).
        exclude_set = spec.get("target_exclude", set())
        if exclude_set:
            excluded_count = 0
            for i, name in enumerate(conn_names):
                if name in exclude_set:
                    if tgt[i] != 0:
                        excluded_count += 1
                    tgt[i] = 0.0
            if excluded_count > 0:
                print(f"  ↳ {mod_name}: excluded {excluded_count} "
                      f"pharyngeal/anatomically-isolated targets")

        target_weights[mod_name] = tgt
        receptors_used[mod_name] = resolved_receptors

        n_rel = int(mask.sum())
        n_tgt = int(np.sum(np.abs(tgt) > 0.1))
        print(f"  {mod_name:<10} {syn_sym:<10} {n_rel:>11} "
              f"{len(resolved_receptors):>12} {n_tgt:>9}")

    # Biological-sanity checks: known releaser identities
    sanity_tests = [
        ("FLP-11", "RIS"),
        ("5HT", "NSML"),
        ("5HT", "NSMR"),
        ("DA", "PDEL"),
        ("DA", "PDER"),
        ("TA", "RIML"),
        ("TA", "RIMR"),
        ("NLP-12", "DVA"),
        ("PDF-1", "AVBL"),
    ]
    print("\n[4] Sanity checks (expected releasers):")
    for mod, neuron in sanity_tests:
        if neuron in name_to_idx:
            idx = name_to_idx[neuron]
            is_rel = releasers.get(mod, np.array([]))
            if len(is_rel) > 0 and is_rel[idx]:
                w = releaser_weights[mod][idx]
                print(f"  ✓ {mod:<8} releaser: {neuron} (w={w:.1f})")
            else:
                # Find top releasers for this modulator for comparison
                top_idx = np.argsort(-releaser_weights.get(
                    mod, np.zeros(N)))[:5]
                top_names = [conn_names[i] for i in top_idx
                             if releaser_weights[mod][i] > 0]
                print(f"  ✗ {mod:<8} {neuron} not releaser. "
                      f"Top releasers: {top_names[:5]}")
        else:
            print(f"  ? {mod:<8} {neuron} not in connectome")

    # Summary of top targets per modulator (top 8 most-modulated neurons)
    print("\n[5] Top 8 targets per modulator (by |target_weight|):")
    for mod in MODULATORS:
        tgt = target_weights[mod]
        if np.abs(tgt).max() == 0:
            continue
        top_idx = np.argsort(-np.abs(tgt))[:8]
        sorted_names = [(conn_names[i], float(tgt[i])) for i in top_idx
                        if abs(tgt[i]) > 0.1]
        print(f"  {mod:<8}: " +
              " ".join(f"{n}({w:+.1f})" for n, w in sorted_names))

    # Save
    out_arrs = {
        "neuron_order": np.array(conn_names, dtype=object),
        "modulators": np.array(list(MODULATORS.keys()), dtype=object),
    }
    for mod in MODULATORS:
        out_arrs[f"releasers_{mod}"] = releasers[mod]
        out_arrs[f"releaser_weights_{mod}"] = releaser_weights[mod]
        out_arrs[f"target_weights_{mod}"] = target_weights[mod]
        out_arrs[f"tau_{mod}"] = np.float32(MODULATORS[mod]["tau_s"])
    np.savez_compressed(OUT, **out_arrs)

    meta = {
        "modulators": {
            m: {
                "synthesis_gene": MODULATORS[m]["synthesis_gene"],
                "receptors": {r: int(s) for r, s in MODULATORS[m]["receptors"].items()},
                "tau_s": MODULATORS[m]["tau_s"],
                "note": MODULATORS[m]["note"],
                "receptors_resolved": receptors_used[m],
                "n_releasers": int(releasers[m].sum()),
                "n_targets_significant": int(np.sum(
                    np.abs(target_weights[m]) > 0.1)),
            }
            for m in MODULATORS
        },
        "unmatched_genes": unmatched_genes,
        "n_connectome_neurons": N,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {OUT_META} ({OUT_META.stat().st_size / 1024:.1f} KB)")
    if unmatched_genes:
        print(f"\nUnmatched genes: {unmatched_genes}")


if __name__ == "__main__":
    main()
