# Phase B — Binding pose prediction

**Phase letter:** B
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phase A (structural priors). Phase B requires Gate A.1 to pass before entering.
**Successor:** Phase C (occupancy matrix) consumes Phase B's per-target binding affinity scores.
**Compute:** Local RTX 4060 Ti ~30 hours; optional free-tier Colab T4 DiffDock overflow ~8 hours. **External spend: $0.**
**Canonical method:** Vina + DiffDock + GNINA cascade. **FEP is DEFERRED / SPECULATIVE** — see §13.

---

## 1. Goal

For each Tier-1 target × each anesthetic combination, predict the binding pose and a binding affinity estimate (Vina score, GNINA scoring, DiffDock confidence). Produce a per-target × per-anesthetic binding matrix that Phase C converts into fractional occupancy at clinical concentrations.

The phase resolves: **does each Tier-1 target have a credible anesthetic binding mode at a structurally identifiable pocket?** If most targets show only diffuse, low-confidence binding, the docking-derived occupancy estimate is noise and Phase C cannot run cleanly.

---

## 2. Background

Anesthetic binding to ion channels is **diffuse, hydrophobic-pocket-preferring, and low-affinity**. Halothane Kd values for known targets are typically 100 µM to 1 mM range; isoflurane similar. This contrasts sharply with high-affinity drug-target docking (nM Kd) where Vina-style scoring excels. The relevant literature acknowledges:

- Vina is calibrated for high-affinity drug discovery; raw Vina scores do not map linearly to anesthetic Kd.
- DiffDock (Corso 2023, ICLR — DOI `10.48550/arXiv.2210.01776`) generates ensemble poses and is more useful for diffuse-binding cases.
- GNINA (McNutt 2021, *Journal of Cheminformatics* — PMID 34108026) uses a CNN-based rescoring on top of Vina poses and has been benchmarked on anesthetic-class molecules. GNINA's ML-rescored Vina has been shown to recover ranking accuracy within roughly 1 kcal/mol of FEP for ligand series in the regime relevant to multi-target occupancy ordering.
- Free Energy Perturbation (FEP) via AMBER or YANK provides absolute binding free energies but at ~100× the cost of docking. Wave P **does not use FEP on the canonical path** — multi-target framing requires *relative* per-target occupancy ranking, not absolute ΔG. FEP is documented as a DEFERRED / SPECULATIVE enhancement in §13.
- `fpocket` (Le Guilloux 2009, *BMC Bioinformatics* — PMID 19486540) provides unbiased cavity detection — used as a sanity check that the docking pocket actually exists as a cavity in the predicted structure.

Cross-validation against published photolabeling data is the strongest empirical anchor:

- Olsen lab GABA-A propofol photolabeling (Yip et al. 2013, *Nature Chemical Biology*; Jayakar 2014, *J Biol Chem*).
- Trudell isoflurane photolabeling on GABA-A.
- nAChR M2 helix photolabeling.

These define ~5 high-confidence pocket assignments on mammalian targets; cross-checking the predicted *C. elegans* docking pose against the homologous mammalian pocket residues is a Phase B success criterion.

---

## 3. Method

### 3.1 Cascade architecture (canonical, FEP-free)

Phase B uses a three-tool cascade per (target, anesthetic) pair:

1. **`fpocket`** to enumerate candidate cavities in the predicted structure. Output: ranked list of pockets by druggability score.
2. **AutoDock Vina 1.2** (Eberhardt 2021, *J Chem Inf Model* — PMID 34003684) for rigid-receptor docking with constrained pocket where a homolog-derived pocket exists. For pockets without homolog-derived constraints, Vina runs in unconstrained mode using fpocket-detected cavities.
3. **DiffDock** for ensemble pose generation. DiffDock is generative; it produces 10-40 candidate poses with confidence scores. Used to detect whether Vina has found a genuinely high-confidence pose or has converged to a local minimum.
4. **GNINA** to rescore both Vina and DiffDock poses with a 3D CNN. GNINA's scoring is more robust for low-affinity anesthetic-class molecules. **GNINA is the terminal step of the canonical cascade.** Its CNN-rescored Vina poses provide the per-target Kd estimates consumed by Phase C.

FEP is not part of the canonical cascade. See §13 for the deferred/speculative FEP appendix.

### 3.2 Per-(target, anesthetic) workflow

```bash
TARGET=UNC-49
ANESTHETIC=halothane
RECEPTOR=artifacts/structures/${TARGET}_multimer/rank_001.pdb
LIGAND=anesthetics/anesthetic_smiles/${ANESTHETIC}.sdf

# Step 1: fpocket cavity enumeration
fpocket -f ${RECEPTOR}
# Output: <RECEPTOR>_out/pockets/pocket*.pdb

# Step 2: identify the pocket of interest
# Either via homolog mapping (preferred) or top fpocket druggability score
POCKET_CENTER=$(python src/phase_b_dock.py --select-pocket \
    --receptor ${RECEPTOR} \
    --target ${TARGET} \
    --homolog-pocket targets/pocket_residues_homolog.csv)

# Step 3: prepare receptor + ligand
prepare_receptor4.py -r ${RECEPTOR} -o tmp/receptor.pdbqt
prepare_ligand4.py -l ${LIGAND} -o tmp/ligand.pdbqt

# Step 4: AutoDock Vina with constrained box around POCKET_CENTER
vina --receptor tmp/receptor.pdbqt --ligand tmp/ligand.pdbqt \
    --center_x ${POCKET_X} --center_y ${POCKET_Y} --center_z ${POCKET_Z} \
    --size_x 20 --size_y 20 --size_z 20 \
    --num_modes 9 --exhaustiveness 32 \
    --out artifacts/binding/${TARGET}_${ANESTHETIC}_vina.pdbqt \
    --log artifacts/binding/${TARGET}_${ANESTHETIC}_vina.log

# Step 5: DiffDock ensemble (Colab)
python diffdock_run.py \
    --receptor ${RECEPTOR} \
    --ligand ${LIGAND} \
    --num_samples 40 \
    --out artifacts/binding/${TARGET}_${ANESTHETIC}_diffdock/

# Step 6: GNINA rescoring on Vina + DiffDock pose ensemble
gnina --receptor ${RECEPTOR} \
    --ligand artifacts/binding/${TARGET}_${ANESTHETIC}_combined.sdf \
    --score_only \
    --cnn crossdock_default2018 \
    --out artifacts/binding/${TARGET}_${ANESTHETIC}_gnina.sdf \
    --log artifacts/binding/${TARGET}_${ANESTHETIC}_gnina.log
```

### 3.3 Cross-method agreement metric

For each (target, anesthetic) pair, define cross-method agreement as:

- **Pocket center agreement:** Vina top-1 pose centroid and DiffDock top-1 pose centroid within 5 Å of each other.
- **GNINA confirmation:** GNINA CNN score on the consensus pose ≥ 5.0 (calibrated against literature anesthetic dockings).
- **fpocket presence:** the consensus pose centroid lies within an fpocket-detected cavity.

A pair passes if all three agree. Pairs failing any are flagged as low-confidence.

### 3.4 Photolabel cross-validation

For the 5 mammalian targets with published photolabel data (GABA-A propofol Olsen; GABA-A isoflurane Trudell; GluCl propofol [if any]; nAChR halothane [if any]; K2P halothane [if any]):

- Identify the photolabeled residue on the mammalian target.
- Map to the homologous *C. elegans* residue via sequence alignment.
- Check whether the *C. elegans* docking pose contacts the homologous residue (within 5 Å of any ligand atom).

Pairs where the predicted *C. elegans* pose contacts the homologous photolabeled residue are flagged as "photolabel-confirmed." The pass criterion at Phase B is that ≥ 4 of 5 photolabel-available cases are photolabel-confirmed.

### 3.5 GNINA-derived Kd consumption by Phase C

After Vina/DiffDock/GNINA cascade across 25 targets × 6 anesthetics = 150 pairs, GNINA scores are converted to per-target Kd estimates via the McNutt 2021 calibration curve (CNN score → ΔG → Kd). Phase C consumes the full 150-pair GNINA-derived Kd matrix with an explicit factor-of-3 uncertainty band documented per pair.

Why this is sufficient for the multi-target framing: Gate C.1 asks whether **≥ 5 targets** exceed 10% occupancy at clinical EC50. That question is determined by *relative* per-target Kd ordering across the 150-pair matrix, not by absolute ΔG of any single pair. GNINA's CNN rescoring has been benchmarked to within ~1 kcal/mol of FEP for ligand series, which is well inside the noise floor of the partition-coefficient and Hill-coefficient assumptions in Phase C.

---

## 4. Compute budget (zero external spend)

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| fpocket on 25 receptors | local CPU | 1 | $0 |
| Vina docking 25 × 6 = 150 pairs at exh=32 | local RTX 4060 Ti GPU | 12 | $0 |
| DiffDock ensemble for 150 pairs (truncated receptors, mostly local; free-tier Colab T4 overflow) | local 4060 Ti + ~8 hr Colab T4 | 15 + 8 | $0 |
| GNINA rescoring 150 pairs | local RTX 4060 Ti | 3 | $0 |
| Photolabel cross-validation analysis | local CPU | 2 | $0 |
| **Total Phase B (canonical)** | | **~30 GPU-h local + ~8 hr free-tier Colab** | **$0** |

FEP top-10 confirmation is **dropped from the canonical budget**, saving ~50 GPU-hours of cloud time and the ~$200-400 burst cost. If FEP is later authorized (see §13), it adds ~30 sequential days of local FEP runs at ~50 ns × 11 windows × 10 pairs on the 4060 Ti, OR ~$200-400 cloud spend if executed remotely.

---

## 5. Preregistered success criteria (Gate B.1)

Phase B passes Gate B.1 if and only if all four criteria are met:

1. **B.1.1 — Coverage:** ≥ 22 of 25 Tier-1 targets have at least one anesthetic with a passing pose (cross-method agreement, GNINA score ≥ 5.0, fpocket-cavity-confirmed). Coverage below 22/25 limits Phase C.

2. **B.1.2 — Cross-method agreement rate:** Across all (target, anesthetic) pairs, ≥ 70% pass the cross-method agreement (Vina + DiffDock + GNINA). Pairs failing are flagged or excluded from Phase C.

3. **B.1.3 — Photolabel match:** Of the 5 (target, anesthetic) pairs with published photolabel data, ≥ 4 are photolabel-confirmed by the predicted pose. Pairs failing are flagged; if 0 of 5 confirm, the docking pipeline itself is suspect — pause Phase B.

4. **B.1.4 — GNINA cross-method-agreement on top 10:** For the 10 top-GNINA-score pairs, *all three* methods (Vina top-1 pose, DiffDock top-1 pose, GNINA-rescored consensus pose) must agree on pocket placement within 5 Å AND the GNINA CNN score must be ≥ 6.0 (vs. the 5.0 threshold used for the broader 150-pair matrix). This is the canonical replacement for the previous FEP-confirmation criterion. Failures here indicate the top GNINA hits may be local-minimum artifacts and Phase C should propagate factor-of-3 uncertainty bands on those pairs explicitly.

---

## 6. Halting rules

**Pause and surface to user:**

- Photolabel match rate < 2 of 5 → docking pipeline is producing structurally implausible poses; pause and re-evaluate tool choice (potentially escalate to deferred FEP path per §13).
- Vina/DiffDock disagreement rate > 60% (i.e., the two methods place poses > 5 Å apart on most pairs) → cascade is unreliable; pause.
- Top-10 GNINA cross-method-agreement (Gate B.1.4) fails on > 4 of 10 pairs → docking-derived Kd is not robust at the high-confidence end; pause and request user discussion of escalating to deferred FEP path per §13.

**Document and continue:**

- Single (target, anesthetic) pair fails cross-method agreement → flag in `artifacts/binding/coverage_report.md`, mark as low-confidence in Phase C input.
- A specific anesthetic (e.g., ketamine) fails most pairs → flag; ketamine is a control in any case.
- fpocket detects no cavity at the homolog-mapped pocket residues → use unconstrained Vina, document the inconsistency.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `artifacts/binding/<TARGET>_<ANESTHETIC>_vina.pdbqt` | Vina top-9 poses |
| `artifacts/binding/<TARGET>_<ANESTHETIC>_diffdock/` | DiffDock ensemble (40 poses) |
| `artifacts/binding/<TARGET>_<ANESTHETIC>_gnina.sdf` | GNINA-rescored poses |
| `artifacts/binding/<TARGET>_<ANESTHETIC>_consensus.json` | Consensus pose + metrics |
| `artifacts/binding/binding_matrix.csv` | 25 targets × 6 anesthetics × {Vina, GNINA, DiffDock_top1, Kd_estimate, uncertainty_band} |
| `artifacts/binding/photolabel_match.md` | Photolabel cross-validation report |
| `artifacts/binding/top10_gnina.csv` | Top-10 GNINA hits with cross-method-agreement metrics (replaces former `fep_top10.csv`) |
| `artifacts/binding/coverage_report.md` | Gate B.1 evaluation |
| `artifacts/binding/phase_b_completion.md` | end-of-block report |
| `artifacts/logs/phase_b_<DATE>.log` | execution log |

---

## 8. Falsifiability checks

The phase's premise: **"AutoDock Vina + DiffDock + GNINA + FEP-on-top cascade produces credible binding-affinity estimates for each (target, anesthetic) pair sufficient for occupancy estimation."**

Falsified if:

1. Photolabel match rate < 50% on the 5 anchor pairs — predicted poses systematically miss known anesthetic binding sites.
2. Vina/DiffDock disagreement on > 60% of pairs — methods are unreliable.
3. Top-10 GNINA cross-method-agreement (Gate B.1.4) fails on > 6 of 10 pairs — the high-confidence end of the docking matrix is not internally consistent and the GNINA-derived Kd ordering may not be reliable for Phase C.

Any of these surfaces means Phase B's docking-based path is not viable; alternatives (escalate to deferred FEP per §13, or fall back to mammalian-homolog docking only) need user discussion.

---

## 9. Integration points

**Inputs from earlier phases:**

- Phase A `artifacts/structures/<TARGET>_multimer/rank_001.pdb` — predicted structures.
- Phase A `artifacts/structures/<TARGET>_pocket_plddt.json` — pocket confidence.
- `targets/tier1_targets.csv` — homolog PDB references.
- `targets/pocket_residues_homolog.csv` — pocket residue mappings (Phase A populates).
- `anesthetics/anesthetic_smiles/*.sdf` — prepared ligands (Phase 0 prep).

**Outputs consumed by:**

- **Phase C** (`src/phase_c_occupancy.py`) — reads `binding_matrix.csv` for Kd values.
- **Phase D** (literature mining) — reads `consensus.json` for pocket residues used in MD setup.

---

## 10. Citation hygiene declaration

- Eberhardt 2021, AutoDock Vina 1.2 — PMID 34003684. [VERIFIED]
- McNutt 2021, GNINA — PMID 34108026. [VERIFIED]
- Corso 2023, DiffDock — DOI `10.48550/arXiv.2210.01776`. [VERIFIED via arXiv]
- Le Guilloux 2009, fpocket — PMID 19486540. [VERIFIED]
- Yip 2013, propofol GABA-A photolabel, *Nature Chem Bio* — (PMID lookup needed).
- Jayakar 2014, propofol GABA-A photolabel, *J Biol Chem* — (PMID lookup needed).
- Trudell isoflurane GABA-A photolabel — (specific paper + PMID lookup needed).

**Pre-flight verification status:** 4 of 7 verified. 3 photolabel-paper PMIDs are blocking items for Phase B entry.

---

## 11. Risk register (Phase B)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Vina scoring uncalibrated for anesthetics | High (known issue) | GNINA rescoring; document the calibration in `binding_matrix.csv`; factor-of-3 Kd uncertainty band propagated to Phase C |
| DiffDock fails on large pentameric receptors | Medium | Run on isolated pentamer-pocket subsystem; pre-truncate receptor to 30 Å around fpocket cavity; free-tier Colab T4 overflow if local 8 GB insufficient |
| GNINA CNN trained on different chemotypes than anesthetics | Medium | Use `crossdock_default2018` model; cross-check against benchmark anesthetic poses |
| GNINA-derived Kd ranking unreliable at the high-confidence end | Medium | Gate B.1.4 (cross-method agreement on top 10); escalate to deferred FEP path per §13 if gate fails |
| Photolabel cross-validation lookups produce 0 of 5 confirm | Low (anchor pairs are well-established) | Re-evaluate sequence alignment; check for *C. elegans*-specific receptor architecture differences |
| AlphaFold pocket flexibility not captured by rigid docking | Medium | Consider induced-fit docking for top hits; document as uncertainty band |
| `prepare_ligand4.py` fails on halothane (small, simple) | Low | Use OpenBabel as alternative ligand prep |

---

## 12. Phase B execution plan

1. Pre-flight citation verification (resolve 3 photolabel PMIDs).
2. Tool installation (Vina 1.2 local, DiffDock local + free-tier Colab T4 overflow, GNINA local, fpocket).
3. Pocket residue mapping (read `target_panel_rationale.md`; populate `targets/pocket_residues_homolog.csv` from sequence alignments).
4. Cascade runs across 25 × 6 = 150 pairs (Vina → DiffDock → GNINA, all local except DiffDock overflow).
5. Cross-method agreement analysis.
6. Photolabel cross-validation.
7. Top-10 GNINA cross-method-agreement evaluation (Gate B.1.4).
8. Compile `binding_matrix.csv` with all scores + Kd estimates + factor-of-3 uncertainty bands.
9. Gate B.1 evaluation; end-of-block report.

---

## 13. DEFERRED / SPECULATIVE — Free Energy Perturbation appendix

**Status:** DEFERRED. Not on the canonical Phase B path. Documented here so that if the user later requests absolute-affinity calibration on the top-10 GNINA hits, the methodology and cost picture are pre-staged.

### 13.1 Why FEP was dropped from the canonical path

Wave P's load-bearing Phase C question is whether **≥ 5 targets** exceed 10% occupancy at clinical EC50 (Gate C.1). That question is determined by the *relative* per-target Kd ordering across the 150-pair binding matrix, not by the absolute ΔG_bind of any single (target, anesthetic) pair.

GNINA's CNN-rescored Vina has been benchmarked to within ~1 kcal/mol of FEP for ligand series in the regime relevant here (McNutt 2021 PMID 34108026). At 1 kcal/mol uncertainty in ΔG, the Kd ratio across the 150-pair matrix is preserved well within the factor-of-3 uncertainty band Phase C already propagates from partition-coefficient and Hill-coefficient assumptions.

Adding FEP would buy: absolute ΔG_bind on the top-10 hits with calibration to ~factor-of-2 vs. experiment.

Adding FEP would cost: ~$200-400 cloud burst (Lambda Labs A100), OR ~30 sequential days of local FEP runs on the 4060 Ti. Neither is justified by the marginal gain in confidence on top-10 ordering.

### 13.2 If FEP is later authorized — methodology

For top-10 (target, anesthetic) hits by GNINA score:

- Receptor + ligand in POPC bilayer (CHARMM-GUI setup).
- AMBER ff14SB + GAFF2 ligand parameters with AM1-BCC partial charges.
- 11-window FEP (λ = 0, 0.1, ..., 1.0) with 5 ns equilibration + 10 ns production per window.
- Total per-pair: ~165 ns × ~50,000 atoms.
- Local 4060 Ti: ~3 days per pair × 10 pairs = ~30 days sequential. OR ~$25-50 per pair on Lambda Labs A100 burst (~$200-400 total for top 10).

FEP outputs absolute ΔG_bind, which converts to Kd via Kd = exp(ΔG / RT). Phase C would consume FEP-derived Kd for the top 10 hits and GNINA-derived Kd for the remaining 140.

### 13.3 If-then trigger

The deferred FEP path activates only on explicit user direction, OR if Gate B.1.4 (cross-method agreement on top 10) fails on > 4 of 10 pairs and the user authorizes the cost to disambiguate. Documented for the record only.
