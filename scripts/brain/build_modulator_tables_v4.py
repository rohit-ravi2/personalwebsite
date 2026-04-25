#!/usr/bin/env python3
"""Phase 5 prep — T4-5 — extend modulator_tables with INS family.

Adds 6 insulin-like peptides to the existing 9-modulator set, producing
`artifacts/modulator_tables_v4.npz`. These signal primarily through the
DAF-2 receptor (the sole C. elegans insulin-like receptor) with tissue-
specific effects via downstream AGE-1 / PI3K / AKT / DAF-16.

Rationale: prior analysis in the project indicated INS-family peptides
are disproportionately represented in the connectome's peptidergic
signaling relative to the 9-modulator v3 set. Adding 6 targeted INS
peptides — each with a published ablation or overexpression phenotype
to validate against — is the first T4-5 expansion. Full peptidome
(~40 INS genes) is not included; this is the defensible first tier.

INS peptides added (each with a specific phenotype test in T4-5):

  INS-1    antagonist of DAF-2 (stress response modulation; Pierce 2001)
  INS-6    DAF-2 agonist (dauer formation, insulin-like cross-talk;
           Cornils 2011)
  INS-7    DAF-2 agonist (olfactory learning, chemotaxis adaptation;
           Chen 2013)
  INS-17   stress response, dauer (class A insulin cluster)
  INS-18   stress response, dauer, aging
  INS-22   DAF-16 negative regulator (RIS/quiescence coupling)
  DAF-28   canonical DAF-2 agonist (dauer, longevity; Li 2003)

Signs (+1 for DAF-2 agonist, -1 for antagonist):
  Agonists: INS-6, INS-7, INS-17, INS-18, DAF-28  → DAF-2 +1 →
    downstream AKT activation, which in our simplified membrane
    model corresponds to K+ leak modulation (excitability shift).
  Antagonists: INS-1, INS-22                       → DAF-2 -1

Time constants: 60-120 s. Slower than FLP/monoamine (10-30 s) because
insulin-mediated effects are primarily transcriptional. Our simplified
model captures the membrane-level excitability effect, which is faster
than the transcriptional response; 60s is a reasonable compromise.

Output: `modulator_tables_v4.npz` keyed like v3 tables but with 15
modulators (9 existing + 6 INS). When loaded by
`ClosedLoopEnv(modulator_tables_path=...)`, the modulation layer
handles the expanded set transparently.

Reference verification: each INS citation should be verified against
primary literature before T4-5 runs. Phase 0 flags this as user-
action-pending (same pattern as Gao-Hobert 2020 flag).

Usage:
  python build_modulator_tables_v4.py              # emit v4 tables
  python build_modulator_tables_v4.py --dry-run    # print summary only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"
V3_TABLES = ART / "modulator_tables.npz"
V4_TABLES = ART / "modulator_tables_v4.npz"
V4_META = V4_TABLES.with_suffix(".json")

# Per-peptide definitions
INS_PEPTIDES = {
    "INS-1": {
        "synthesis_gene": "ins-1",
        "receptors": {"daf-2": -1},
        "tau_s": 60.0,
        "phenotype_test": "INS-1 ablation → altered stress-response state "
                          "distribution under food deprivation (Pierce 2001)",
        "reference": "Pierce SB et al. (2001) Regulation of DAF-2 receptor "
                     "signaling by human insulin and ins-1, a member of the "
                     "unusually large and diverse C. elegans insulin gene "
                     "family. Genes Dev 15:672-686. DOI:10.1101/gad.867301",
    },
    "INS-6": {
        "synthesis_gene": "ins-6",
        "receptors": {"daf-2": +1},
        "tau_s": 90.0,
        "phenotype_test": "INS-6 ablation → accelerated dauer entry under "
                          "starvation (Cornils 2011)",
        "reference": "Cornils A, Gloeck M, Chen Z, Zhang Y, Alcedo J (2011) "
                     "Specific insulin-like peptides encode sensory information "
                     "to regulate distinct developmental processes. Development "
                     "138:1183-1193",
    },
    "INS-7": {
        "synthesis_gene": "ins-7",
        "receptors": {"daf-2": +1},
        "tau_s": 60.0,
        "phenotype_test": "INS-7 ablation → loss of olfactory associative "
                          "learning (Chen 2013, Tomioka 2006)",
        "reference": "Chen Z et al. (2013) Insulin signaling in intestine mediates "
                     "metabolism-driven aggressive behavior in C. elegans; "
                     "Tomioka M et al. (2006) The insulin/PI 3-kinase pathway "
                     "regulates salt chemotaxis learning in C. elegans. "
                     "Neuron 51:613-625",
    },
    "INS-17": {
        "synthesis_gene": "ins-17",
        "receptors": {"daf-2": +1},
        "tau_s": 90.0,
        "phenotype_test": "INS-17/18 double ablation → stress-response "
                          "phenotype (class A insulins)",
        "reference": "Pierce SB et al. (2001) Genes Dev; Kodama E et al. "
                     "(2006) Genes & Development",
    },
    "INS-18": {
        "synthesis_gene": "ins-18",
        "receptors": {"daf-2": +1},
        "tau_s": 90.0,
        "phenotype_test": "INS-18 ablation → dauer/aging phenotype. "
                          "Paired with INS-17.",
        "reference": "Pierce SB et al. (2001) Genes Dev",
    },
    "INS-22": {
        "synthesis_gene": "ins-22",
        "receptors": {"daf-2": -1},
        "tau_s": 60.0,
        "phenotype_test": "INS-22 ablation under osmotic shock → ΔQUI ≥ +0.15. "
                          "Opposite sign to RIS/FLP-11 phenotype; insulin-DAF-16 "
                          "pathway cross-talk with quiescence circuit.",
        "reference": "Reference verification pending — cited in project plan "
                     "as DAF-16 negative regulator; user to confirm primary "
                     "literature before T4-5 runs.",
    },
    "DAF-28": {
        "synthesis_gene": "daf-28",
        "receptors": {"daf-2": +1},
        "tau_s": 120.0,
        "phenotype_test": "DAF-28 ablation under food-deprivation → increased "
                          "dauer entry frequency (Li 2003)",
        "reference": "Li W, Kennedy SG, Ruvkun G (2003) daf-28 encodes a "
                     "C. elegans insulin superfamily member that is regulated "
                     "by environmental cues and acts in the DAF-2 signaling "
                     "pathway. Genes Dev 17:844-858",
    },
}


def _load_v3_tables():
    """Load existing v3 modulator tables; return as a mutable dict."""
    if not V3_TABLES.exists():
        print(f"ERROR v3 tables missing: {V3_TABLES}")
        print("Run build_modulator_tables.py first.")
        sys.exit(1)
    data = np.load(V3_TABLES, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _extend_with_ins(v3: dict, dry_run: bool = False):
    """Add INS peptides to the modulator tables. Uses DAF-2 as the sole
    receptor and synthesises releasers/targets from CeNGEN via the same
    pathway as build_modulator_tables — but since that pipeline requires
    the full CeNGEN data on disk, here we only *declare* the INS entries
    and leave the numerical releaser/receptor matrices as zero-initialized
    until `build_modulator_tables.py` is rerun with `INS_PEPTIDES` merged
    in. This is safer: explicit staging rather than silent row emission
    with questionable per-neuron weights."""
    existing_names = [str(n) for n in v3.get("modulators", np.array([]))]
    print(f"v3 tables have {len(existing_names)} modulators: "
          f"{existing_names}")

    # The INS scaffold is declarative: names + metadata. Numerical
    # releaser_weights / target_weights come from a CeNGEN pass that
    # has to be run with the full expression data (not reproduced
    # here). `build_modulator_tables.py` should be extended to include
    # the INS_PEPTIDES dict and rerun.
    ins_names = list(INS_PEPTIDES.keys())
    new_names = existing_names + ins_names

    summary = {
        "modulators": new_names,
        "new_entries": {
            n: {
                "synthesis_gene": INS_PEPTIDES[n]["synthesis_gene"],
                "receptor_genes": list(INS_PEPTIDES[n]["receptors"].keys()),
                "receptor_signs": list(INS_PEPTIDES[n]["receptors"].values()),
                "tau_s": INS_PEPTIDES[n]["tau_s"],
                "phenotype_test": INS_PEPTIDES[n]["phenotype_test"],
                "reference": INS_PEPTIDES[n]["reference"],
            }
            for n in ins_names
        },
        "phase5_action_required": (
            "To activate numerical releasers/targets: "
            "extend build_modulator_tables.py MODULATORS dict with the "
            "INS_PEPTIDES entries from build_modulator_tables_v4.py, "
            "then rerun. That emits the full (N_conn,) releaser and "
            "target matrices by running per-peptide through the CeNGEN "
            "expression pipeline."
        ),
    }
    print("\nDeclared INS peptides:")
    for n in ins_names:
        p = INS_PEPTIDES[n]
        sign = list(p["receptors"].values())[0]
        role = "AGONIST" if sign > 0 else "ANTAGONIST"
        print(f"  {n}  via {p['synthesis_gene']} → DAF-2 {role}  "
              f"(τ={p['tau_s']} s)")
        print(f"    Phenotype: {p['phenotype_test']}")

    if dry_run:
        return summary

    # Persist
    V4_META.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote metadata: {V4_META}")

    # Emit the v4 tables by copying v3 and appending placeholder rows
    out = {}
    for k, v in v3.items():
        out[k] = v
    # Update the modulators list
    out["modulators"] = np.array(new_names)

    # For each INS, emit zero-filled releaser/target arrays. Shapes:
    # releasers/target_weights are (len(modulators), N_conn). Pattern
    # in v3: one key per modulator name, e.g. `releasers_FLP-11`.
    # Detect N_conn from one of the existing entries.
    existing_rel_key = None
    for k in v3:
        if k.startswith("releasers_"):
            existing_rel_key = k
            break
    if existing_rel_key is None:
        print("Cannot infer N_conn from v3 tables; aborting placeholder emit.")
        sys.exit(1)
    n_conn = int(v3[existing_rel_key].shape[0])
    for n in ins_names:
        out[f"releasers_{n}"] = np.zeros(n_conn, dtype=bool)
        out[f"releaser_weights_{n}"] = np.zeros(n_conn, dtype=np.float32)
        out[f"target_weights_{n}"] = np.zeros(n_conn, dtype=np.float32)
        out[f"tau_s_{n}"] = np.float32(INS_PEPTIDES[n]["tau_s"])

    np.savez_compressed(V4_TABLES, **out)
    print(f"Wrote v4 tables: {V4_TABLES}  "
          f"({len(new_names)} modulators, "
          f"{V4_TABLES.stat().st_size / 1024:.1f} KB)")
    print("\nNOTE: INS releasers/targets are zero-initialised. To activate, "
          "extend build_modulator_tables.py with INS_PEPTIDES and rerun "
          "the full pipeline with CeNGEN data.")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    v3 = _load_v3_tables()
    _extend_with_ins(v3, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
