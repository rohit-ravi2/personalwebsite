# Phase A — Structural priors

**Phase letter:** A
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** none (entry phase).
**Successor:** Phase B (binding pose prediction) consumes Phase A's `.pdb` outputs.
**Compute:** Local RTX 4060 Ti ~12 GPU-h primary (ESMFold / Boltz-1 / OpenFold); free-tier Colab T4 ~10 hr overflow only. **External spend: $0.**
**Predictor stack:** ESMFold (MIT, Lin et al. 2023 *Science*) primary for monomers; Boltz-1 (MIT, Wohlwend et al. 2024) primary for pentamers / multimers; OpenFold (Apache 2.0, Ahdritz et al. 2024 *Nat Methods*) tertiary fallback; ColabFold free tier (T4) quaternary; AlphaFold-Multimer / RoseTTAFold-AllAtom remain available for cross-validation but are non-load-bearing.

---

## 1. Goal

Predict 3D atomic-resolution structures for all 25 Tier-1 anesthetic targets, validate them against published experimental homologs where available, and produce a curated set of `.pdb` files that can be fed into Phase B's docking pipeline.

The phase resolves a single load-bearing question: **do we have structurally credible models for the binding sites of all 25 Tier-1 targets?** If not, Phase B (docking) cannot run on the missing targets and Wave P scope shrinks.

---

## 2. Background

The Tier-1 panel mixes three structural classes:

1. **Pentameric Cys-loop receptors** (UNC-49, EXP-1, AVR-14, AVR-15, GLC-1, GLC-2, ACR-16, ACR-2, UNC-29, UNC-38, UNC-63, LEV-1) — homo- or heteropentamers requiring AlphaFold-Multimer or ColabFold's multimer mode. Mammalian / *C. elegans* homologs available:
   - GluCl (GLC-1/-2 family): Hibbs & Gouaux 2011 *Nature* PDB 3RHW (PMID 21572436).
   - GABA-A (UNC-49 homolog): mammalian α1β2γ2 PDB 6X3X (Laverty 2019, PMID 31182812; PMID lookup needed for full attribution).
   - nAChR (ACR/UNC-29/-38/-63/LEV-1 homolog): Torpedo nAChR PDB 7QL5 (Rahman 2022, PMID lookup needed).

2. **K2P channels** (TWK-18, TWK-7, TWK-29) — domain-swapped dimers. Mammalian homologs available:
   - TREK-1 PDB 6CQ6 (Lolicato 2017, PMID 28729657).
   - TRAAK PDB 4WFE (Brohawn 2014).
   - The *C. elegans* TWK family is sequence-divergent; AlphaFold prediction is the primary path.

3. **NCA channel complex** (NCA-1, NCA-2, UNC-79, UNC-80, NLF-1) — the worm homolog of NALCN. Mammalian NALCN cryoEM available:
   - NALCN PDB 7SX3 (Xie 2020, PMID 33020732).
   - UNC-79 / UNC-80 are auxiliary subunits with no high-confidence mammalian structural homolog at full length; they are predicted regions of intrinsic disorder and may fail AlphaFold confidence thresholds.

4. **SNARE machinery** (UNC-64, RIC-4, SNB-1, UNC-13, UNC-18, SNT-1) — coiled-coil four-helix bundle (UNC-64 / RIC-4 / SNB-1) plus regulatory partners (UNC-13, UNC-18) plus Ca-sensor (SNT-1). Homologs:
   - Neuronal SNARE bundle PDB 1SFC (Sutton 1998, PMID 9759724).
   - Munc13 C2A PDB 4XII; Munc18-syntaxin PDB 4JEU.
   - Synaptotagmin-1 C2A/C2B PDB 3F03.

5. **Mitochondrial Complex I subunits** (GAS-1 / NDUFS2 homolog, NUO-1 through NUO-6, MEV-1) — embedded in the membrane arm of Complex I. Mammalian Complex I cryoEM available:
   - Bovine Complex I PDB 5LDX (Zhu 2016, PMID 27548872).
   - Mouse PDB 6G2J (Agip 2018, PMID 30462775).

For each target, the Phase A workflow is:

1. **Pull pre-computed AlphaFold DB monomer structure** (UniProt-keyed) where available.
2. **Run AlphaFold-Multimer / ColabFold for oligomeric assemblies** (pentamers, dimers, the SNARE bundle).
3. **Cross-validate against the experimental homolog** — RMSD on aligned core, pLDDT > 70 at the binding pocket, FoldSeek similarity check.
4. **Run RoseTTAFold-AllAtom** (Krishna 2024, *Science*, PMID 38386700) on cases where AlphaFold confidence is low at the binding pocket — RFAA's all-atom output handles co-factors and lipid anchors more cleanly than AlphaFold for some K2P and Complex I cases.

---

## 3. Method

### 3.1 Tools and versions

- **ColabFold 1.5+** (Mirdita 2022, *Nature Methods*, PMID 35637307) — primary structure-prediction driver.
- **AlphaFold-Multimer** (Evans 2022, bioRxiv DOI `10.1101/2021.10.04.463034`) — for oligomeric assemblies. Used through ColabFold rather than the standalone DeepMind release because of compute / setup cost.
- **AlphaFold DB** (`alphafold.ebi.ac.uk`) — pre-computed monomer pulls keyed by UniProt accession. Faster than running AF locally for monomer entries.
- **RoseTTAFold-AllAtom** (Krishna 2024, *Science*, PMID 38386700) — backup all-atom predictor. Used on K2P and Complex I cases where AF confidence is low.
- **PyMOL 2.5+** or **ChimeraX 1.7+** — alignment, RMSD, visualization. PyMOL command-line is sufficient.
- **FoldSeek** (van Kempen 2023, *Nature Biotechnology*, PMID 37156916) — structural similarity search against PDB to validate predictions.
- **TM-align** (Zhang & Skolnick 2005) — for quantitative structural alignment.
- **DSSP** — for secondary-structure analysis.

### 3.2 Per-target workflow (concrete commands)

For each target in `targets/tier1_targets.csv`:

```bash
# Step 1: pull pre-computed AlphaFold DB monomer (if exists)
TARGET=UNC-49
UNIPROT=$(grep "^${TARGET}," targets/tier1_targets.csv | cut -d, -f8)  # uniprot_id column
wget -O artifacts/structures/${TARGET}_monomer_AFDB.pdb \
    https://alphafold.ebi.ac.uk/files/AF-${UNIPROT}-F1-model_v4.pdb

# Step 2: for pentamer/dimer/oligomer, run ColabFold multimer
# (input: 5 copies of UNC-49 sequence in one FASTA)
colabfold_batch \
    --num-models 5 \
    --num-recycle 3 \
    --use-gpu-relax \
    --rank pae \
    inputs/${TARGET}_pentamer.fasta \
    artifacts/structures/${TARGET}_multimer/

# Step 3: cross-validate against mammalian PDB homolog
PDB_HOMOLOG=$(grep "^${TARGET}," targets/tier1_targets.csv | cut -d, -f9)
pymol -cq -d "
    load artifacts/structures/${TARGET}_multimer/*rank_001*.pdb, predicted
    fetch ${PDB_HOMOLOG}, experimental, async=0
    align predicted, experimental
    print 'RMSD vs experimental homolog'
    quit
"

# Step 4: pLDDT extraction at binding pocket (residues from target_panel_rationale.md)
python src/phase_a_structures.py --extract-pocket-plddt \
    --target ${TARGET} \
    --pocket-residues "${POCKET_RES}" \
    --predicted artifacts/structures/${TARGET}_multimer/*rank_001*.pdb \
    --output artifacts/structures/${TARGET}_pocket_plddt.json

# Step 5: FoldSeek validation
foldseek easy-search \
    artifacts/structures/${TARGET}_multimer/*rank_001*.pdb \
    /path/to/foldseek_pdb_db \
    artifacts/structures/${TARGET}_foldseek.tsv \
    artifacts/foldseek_tmp/
```

### 3.3 Per-target pocket residue identification

For each Tier-1 target, the binding pocket is defined a priori based on structural homology to known anesthetic-binding sites:

| Target class | Pocket definition source | Residue identification method |
|---|---|---|
| GABA-A (UNC-49, EXP-1) | β+/α− interface, M2 helix | Olsen lab photolabel data; sequence alignment to mammalian α1β2γ2 |
| GluCl (AVR-14, AVR-15, GLC-1, GLC-2) | Pore lumen, M2 helix | Hibbs 2011 PDB 3RHW pocket residues |
| nAChR (ACR-16, ACR-2, UNC-29, UNC-38, UNC-63, LEV-1) | M2 helix, intersubunit interface | Forman/Miller mammalian nAChR studies |
| K2P (TWK-18, TWK-7, TWK-29) | Pore-lining S6, fenestration | Lolicato 2017 TREK-1 fenestration |
| NCA / NALCN (NCA-1, NCA-2) | Pore, S6 | Xie 2020 NALCN pore residues |
| UNC-79, UNC-80, NLF-1 | Functional domain (NCA-binding) | AlphaFold per-residue confidence + PAE matrix |
| SNARE (UNC-64, RIC-4, SNB-1) | Layer-zero hydrophobic core | Sutton 1998 layer-zero residues |
| UNC-13, UNC-18 | C2A / C2B Ca-binding loops; syntaxin-binding cleft | Munc13 C2A and Munc18 published binding sites |
| Complex I (GAS-1, NUO-*) | Q-binding tunnel; ND2 (NDUFS2 homolog) loop | Bovine Complex I PDB 5LDX Q-cavity residues |

The pocket residue numbers per target are tabulated in `targets/target_panel_rationale.md` (Phase A populates the actual residue list as it generates structures).

### 3.4 Validation harness

A reusable validation harness (in `src/phase_a_structures.py`) that takes a predicted PDB and a homolog PDB and returns:

- TM-score of full-length alignment.
- RMSD on aligned core (excluding flexible loops).
- pLDDT distribution: mean, median, fraction > 70 at pocket residues.
- PAE (predicted aligned error) at oligomeric interfaces (for multimer predictions).
- FoldSeek E-value of best PDB hit.

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| AlphaFold DB monomer pulls (25 targets) | wget | 0.5 | $0 |
| ESMFold local for any monomer not in AF DB or with stale entry | Local RTX 4060 Ti | ~3 | $0 |
| Boltz-1 pentamer / dimer / oligomer (12 cases) | Local RTX 4060 Ti (chunked attention) | ~25 | $0 |
| OpenFold backup runs (cases where Boltz-1 confidence is low) | Local RTX 4060 Ti | ~6 | $0 |
| ColabFold free-tier T4 overflow (pentameric edge cases that don't fit locally) | Free Colab T4 | ~10 | $0 |
| FoldSeek validation (25 targets) | local CPU | 2 | $0 |
| PyMOL alignment + pocket-pLDDT extraction | local CPU | 2 | $0 |
| **Total Phase A** | | **~12 GPU-h local + ~10 hr free Colab T4** | **$0** |

Free-tier Colab T4 has a ~12 hr/day session cap; ~10 hours total budget = ~1 day. Boltz-1 (Wohlwend et al. 2024) reports comparable accuracy to AlphaFold3 on small protein-protein complexes and is designed for consumer hardware; ESMFold (Lin et al. 2023 DOI 10.1126/science.ade2574) is ~10× faster than AF2 for monomers with single-sequence inference.

---

## 5. Preregistered success criteria (Gate A.1)

Phase A passes Gate A.1 if and only if all four criteria are met:

1. **A.1.1 — Coverage:** ≥ 22 of 25 Tier-1 targets have a predicted structure with the binding pocket modeled. The 3 allowed exceptions are explicitly documented (most likely candidates: UNC-79 long disorder, UNC-80 long disorder, NLF-1 short / no homolog).

2. **A.1.2 — Pocket confidence:** ≥ 22 of 25 targets have **pLDDT > 70 averaged over pocket residues**. Targets with pocket pLDDT < 70 are flagged for Phase B as "low-confidence pocket" — Phase B docking against them is uninformative.

3. **A.1.3 — Homolog cross-check:** For targets with a published experimental homolog (≥ 18 of 25), TM-score of full-length alignment ≥ 0.5, AND RMSD on aligned core ≤ 4 Å. Targets failing this are flagged for re-prediction with RoseTTAFold-AllAtom or excluded from Phase B.

4. **A.1.4 — Oligomeric interface confidence:** For pentamers and other oligomers, mean PAE at the inter-subunit interface ≤ 10 Å. Pentamers with inter-subunit PAE > 10 Å are flagged as "oligomer confidence low" and Phase B uses the monomer instead (less informative for inter-subunit anesthetic sites).

---

## 6. Halting rules

**Pause and surface to user:**

- All four predictors (ESMFold, Boltz-1, OpenFold, ColabFold T4) fail on the same pentamer → escalate; consider subunit-by-subunit pocket modeling or deferring that target to Tier 2.
- All GABA-A or all nAChR pentameric assemblies fail across the predictor stack → multi-target premise structurally compromised at the receptor class level; re-evaluate target panel.
- Any target's predicted structure differs from its mammalian homolog by RMSD > 8 Å on aligned core AND TM-score < 0.4 → flag as "structural disagreement, potential predictor failure"; do not silently include in Phase B.

**Document and continue:**

- Single target fails pLDDT > 70 at pocket → flag in `artifacts/structures/coverage_report.md`, exclude from Phase B, continue.
- Pre-computed AlphaFold DB has stale prediction (model_v3 or earlier) → re-run via ColabFold for that target, document the version difference.
- FoldSeek returns multiple high-similarity hits with conflicting fold assignments → document, prefer the canonical PDB hit, surface if ambiguity affects pocket assignment.

---

## 7. Output deliverables

All paths absolute, rooted at `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/`.

| File | Contents |
|---|---|
| `artifacts/structures/<TARGET>_monomer_AFDB.pdb` | 25 monomer pulls from AlphaFold DB |
| `artifacts/structures/<TARGET>_multimer/*rank_001*.pdb` | ~12 multimer predictions |
| `artifacts/structures/<TARGET>_pocket_plddt.json` | per-target pocket pLDDT distribution |
| `artifacts/structures/<TARGET>_foldseek.tsv` | FoldSeek hits |
| `artifacts/structures/<TARGET>_homolog_alignment.json` | TM-score + RMSD vs experimental homolog |
| `artifacts/structures/coverage_report.md` | Gate A.1 evaluation summary |
| `artifacts/structures/phase_a_completion.md` | Phase A end-of-block report |
| `artifacts/logs/phase_a_<DATE>.log` | execution log |

Phase A's primary downstream consumer is **Phase B** (`src/phase_b_dock.py`), which consumes the rank-1 PDB for each target.

---

## 8. Falsifiability checks

The phase's premise is: **"We can produce structurally credible models for the 25 Tier-1 targets sufficient for downstream docking."**

The premise is falsified if any of these surface:

1. **Coverage < 18 of 25 targets** with usable structures — too sparse for the multi-target framing to test.
2. **Pocket pLDDT < 70 on > 50% of targets** — most pockets are uncertain; Phase B docking is uninformative.
3. **Mammalian homolog cross-checks fail systematically** (TM-score < 0.5 on > 30% of cases) — AlphaFold is producing models that disagree with experimental data; downstream pharmacology is built on sand.

If any of these surface, **pause Phase A and re-evaluate target panel + tool choice** before entering Phase B. Surface immediately to user.

---

## 9. Integration points

**Inputs from earlier phases:** none (entry phase).

**Inputs from external resources:**

- AlphaFold DB monomer entries (UniProt-keyed pulls).
- Experimental PDB homologs (RCSB Protein Data Bank).
- Tier-1 panel CSV at `targets/tier1_targets.csv`.

**Outputs consumed by:**

- **Phase B** (`src/phase_b_dock.py`) — reads rank-1 PDB per target.
- **Phase D** (literature mining; selectively used to confirm pocket residues for kinetic-shift derivation).

---

## 10. Citation hygiene declaration

Every primary-source citation in this phase document carries a PMID or DOI:

- Hibbs & Gouaux 2011, *Nature*, PDB 3RHW — PMID 21572436. [VERIFIED via WebSearch in pre-flight]
- Mirdita 2022, ColabFold, *Nature Methods* — PMID 35637307. [VERIFIED]
- Krishna 2024, RoseTTAFold-AllAtom, *Science* — PMID 38386700. [VERIFIED]
- Lolicato 2017, TREK-1 fenestration, PDB 6CQ6 — PMID 28729657. [VERIFIED]
- Xie 2020, NALCN — PMID 33020732. [VERIFIED]
- Sutton 1998, neuronal SNARE complex, PDB 1SFC — PMID 9759724. [VERIFIED]
- Zhu 2016, bovine Complex I, PDB 5LDX — PMID 27548872. [VERIFIED]
- van Kempen 2023, FoldSeek, *Nature Biotechnology* — PMID 37156916. [VERIFIED]
- Evans 2022, AlphaFold-Multimer, bioRxiv — DOI `10.1101/2021.10.04.463034`. [VERIFIED]
- Laverty 2019, GABA-A α1β2γ2, PDB 6X3X — PMID 31182812. (PMID needs re-verification — note that 6X3X resolution date is 2020, the structure may be cited from a different paper.)
- Rahman 2022, Torpedo nAChR PDB 7QL5 — (PMID lookup needed). [BLOCKING for Phase A entry]
- Agip 2018, mouse Complex I PDB 6G2J — PMID 30462775. [VERIFIED]
- Zhang & Skolnick 2005, TM-align — DOI `10.1093/nar/gki524`.

**Pre-flight citation verification status (kickoff):** 9 of 13 primary citations explicitly verified by entry. Re-verification before Phase A executes will confirm the remaining 4 plus catch any entries that drift between kickoff and execution.

---

## 11. Risk register (Phase A)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pentameric prediction OOMs on 8 GB VRAM | Medium-High | Predictor ladder: ESMFold → Boltz-1 → OpenFold (all local with chunking) → ColabFold free tier T4 → subunit-by-subunit pocket modeling. AF-Multimer is non-load-bearing in this plan. |
| Free-tier Colab T4 quota exhausted before pentameric edge cases complete | Low (only ~10 hr cumulative needed) | Spread across 2-3 calendar days; if exhausted, escalate to subunit-by-subunit pocket modeling; **no escalation to Colab Pro / cloud burst — user has committed to $0 external spend** |
| ESMFold / Boltz-1 / OpenFold all fail on a target | Low-Medium | Use mammalian homolog directly (e.g., 6X3X for GABA-A pentamer); annotate as low-confidence; defer to Tier 2 |
| AlphaFold DB stale (model_v3 or earlier) for key targets | Low | Re-run via local ESMFold or Boltz-1 |
| UNC-79 / UNC-80 / NLF-1 are intrinsically disordered → all confidence metrics fail | High (these are known IDR-rich) | Predict the structured domains only, document the disordered regions, focus Phase B on the structured anesthetic-binding domain (if any) |
| Complex I full-assembly attempt (~45 subunits) → out of scope on local hardware | Certain | Canonical plan scopes to single-subunit-per-anesthetic-site (GAS-1 primary, NUO-1 through NUO-6 individually). Full assembly is DEFERRED. |
| FoldSeek database mismatch / version drift | Low | Use latest PDB-versioned database; document version |

---

## 12. Phase A execution plan (when this phase activates)

1. **Pre-flight citation verification.** Resolve all `(PMID lookup needed)` markers in §10 to verified PMIDs. Block Phase A entry on any unresolved.
2. **Tool installation.** ESMFold (`pip install fair-esm[esmfold]`), OpenFold (clone + pip install), Boltz-1 (`pip install boltz`), FoldSeek, PyMOL CLI. Optional: ColabFold notebook for free-tier T4 overflow.
3. **License bookkeeping.** Document MIT / Apache 2.0 license headers for the load-bearing predictors. AF-Multimer / RFAA, if used, are flagged as cross-validation only (academic, non-load-bearing).
4. **Monomer pulls (batch).** AlphaFold DB monomer pulls for all 25 targets in one wget batch.
5. **ESMFold smoke runs.** Run ESMFold locally on 2-3 *C. elegans* sequences to confirm the local pipeline works on the 4060 Ti.
6. **Pentamer / multimer Boltz-1 runs.** 12 multimer cases via local Boltz-1 with chunked attention; spill to OpenFold or free-tier Colab T4 only on OOM.
7. **Cross-validation harness.** Run TM-align + RMSD + pocket-pLDDT extraction across all 25 targets.
8. **Coverage report.** Compile `artifacts/structures/coverage_report.md` summarizing Gate A.1.
9. **End-of-phase report.** `artifacts/structures/phase_a_completion.md` with surfaced findings, blocked items, Phase B readiness assessment.

Phase A is executed as a single multi-week work block. Mid-flight surfacing applies; do not silently document a failed pocket pLDDT and continue — surface to user.
