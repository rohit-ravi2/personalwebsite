# Phase D — Per-target kinetic shift translation

**Phase letter:** D
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phase C (occupancy matrix). Phase D requires Gate C.1 to pass.
**Successor:** Phase G (network perturbation runs) consumes Phase D's per-target kinetic shifts.
**Compute:** local CPU minimal for literature-derived shifts; ~120 GPU-hours for 5-8 OpenMM MD runs on missing channels.

---

## 1. Goal

Convert the per-(target, anesthetic) occupancy matrix from Phase C into **per-target Hodgkin-Huxley / Markov rate-constant shifts** that can be applied as overlays on Wave 2's Brian2 channel implementations. Where mammalian or *C. elegans* electrophysiology data exists for the anesthetic effect on the target, translate directly. Where it does not (especially for *C. elegans*-specific subunits like TWK-18, NCA-1, AVR-14), run targeted ~100 ns OpenMM MD on the channel + anesthetic in POPC bilayer to estimate the kinetic shift de novo.

The phase's deliverable is a per-target "anesthetic kinetic-shift table" — the Wave-P-specific overlay that, applied to Wave 2's channels, produces the anesthetic-perturbed simulator.

---

## 2. Background

### 2.1 Mechanism class → kinetic shift class

| Mechanism class | Anesthetic effect (in mammals / where measured) | Kinetic shift form |
|---|---|---|
| GABA-A potentiation | Slower decay of IPSC; increased open probability | `τ_decay × (1 + k_pot × occupancy)`, `g_max × (1 + k_g × occupancy)` |
| GluCl potentiation | Reduced peak current; modulated open probability | `g_max × (1 - occupancy × b_factor)` |
| nAChR antagonism | Reduced open probability | `g_max × (1 - occupancy × c_factor)` |
| K2P potentiation (TWK-18-style) | Increased K leak; resting hyperpolarization | `g_max × (1 + k_K2P × occupancy)` |
| NCA / NALCN block | Reduced Na leak; reduced excitability | `g_max × (1 - occupancy × d_factor)` |
| SNARE machinery (UNC-13, UNC-18) | Reduced Ca cooperativity; reduced release p | `n_Ca → n_Ca − δ × occupancy` (van Swinderen 2004) |
| Mitochondrial Complex I | Reduced ATP production; metabolic stress | `Complex_I_rate × (1 - occupancy)` |

Where mammalian electrophysiology data exists (Cys-loop receptors, K2P, NCA), translate directly. Where *C. elegans* data is missing, MD provides a complementary estimate.

### 2.2 Concrete published anchor data

Where literature directly gives a quantitative kinetic shift:

- **van Swinderen 2004** (PMID lookup needed): halothane reduces Ca cooperativity n at the *C. elegans* NMJ from ~3.5 to ~2.0 at clinical concentration. Δn = -1.5 at saturation; for the Hill form, δ ≈ 1.5 × occupancy/saturating occupancy.
- **Crowder 1996** (PMID 8855256): halothane on muscle endogenous current — quantitative IV-curve shift.
- **GABA-A mammalian electrophysiology** (Hales & Lambert, Mihic 1997 *Nature*): isoflurane prolongs τ_decay by ~2-fold at clinical concentration.
- **K2P TWK-18 gain-of-function** (Sedensky 2001 PMID 11756669): halothane induces ~2-3× increase in resting K conductance via TWK-18.
- **Complex I gas-1** (Kayser 2001 — PMID lookup needed): GAS-1 mutant has ~30-50% reduced Complex I activity; halothane on WT Complex I produces an additional ~15-30% reduction at clinical concentration.

For each target × each anesthetic, Wave P's `kinetic_shifts.csv` carries a per-pair entry with:

- `shift_form` — one of `{gaba_potentiation, glucl_potentiation, nachr_antagonism, k2p_potentiation, nca_block, snare_cooperativity, complex_i_block, none}`.
- `shift_param_central` — the shift parameter at saturating occupancy (e.g., k_pot = 1.0 means τ_decay doubles at occupancy = 1).
- `shift_param_low`, `shift_param_high` — uncertainty bracket.
- `source` — one of `{literature_direct, literature_homolog, MD_derived, default}`.
- `source_paper_PMID` — primary source if literature-derived.

### 2.3 OpenMM MD on missing-data targets

Targets without direct published kinetic shifts:

1. **TWK-18 + halothane** — gain-of-function effect known qualitatively but not quantitatively.
2. **NCA-1 + halothane / isoflurane** — block effect known but not quantitatively.
3. **AVR-14 + halothane** — *C. elegans*-specific GluCl, no direct data.
4. **UNC-49 + halothane** — *C. elegans*-specific GABA-A.
5. **GAS-1 + halothane** — Complex I subunit, anesthetic effect on the worm-specific homolog.

For each of these, run OpenMM MD:

- System: predicted PDB (Phase A) + 200 POPC bilayer + 100 mM NaCl + 0.5-2 mM anesthetic + TIP3P water.
- Setup: CHARMM-GUI Membrane Builder; AMBER ff14SB protein; LIPID17 lipids; GAFF2 ligand parameters; AM1-BCC partial charges.
- Equilibration: 500 ps NPT.
- Production: 100 ns NPT at 310 K, 1 atm.
- Analysis: anesthetic occupancy in pocket; effect on channel pore radius; effect on selectivity-filter dynamics; effect on activation gate (M2 helix tilt for Cys-loop, S6 dynamics for K2P).

Per-system cost: ~50,000 atoms × 100 ns ≈ 30-50 ns/day on RTX 4060 Ti = ~3 days per system. Five systems = 15 days. Realistic budget month-by-month: 5-8 systems over 4-6 weeks.

The MD output is a **single kinetic-shift number** (e.g., "TWK-18 g_max increases by 2.4 ± 0.3 fold at saturating halothane"), extracted via a deliberate analysis pipeline that maps MD observables to HH-rate-constant changes. This mapping is the load-bearing methodological commitment of Phase D — the analysis pipeline must be preregistered.

### 2.4 MD-to-HH-shift mapping (preregistered)

For each MD-targeted (channel, anesthetic) pair:

- **Pocket occupancy from MD:** fraction of frames with anesthetic within 5 Å of the homolog-defined pocket residues.
- **Pore radius distribution change:** mean pore radius from HOLE analysis with vs without anesthetic.
- **Selectivity-filter RMSF change:** RMSF of selectivity-filter residues.
- **Activation-gate tilt change:** angle between M2 (Cys-loop) or S6 (K2P) and the membrane normal, with and without anesthetic.

Mapping rules:

- For **K2P**: pore-radius-mean increases by Δr → `g_max` multiplier = (Δr / r_0) × 2 (the factor 2 is empirical; documented per literature for similar K2P-anesthetic systems).
- For **NCA**: pore-radius-mean decreases by Δr → `g_max` multiplier = 1 − (|Δr| / r_0) × 2.
- For **Cys-loop receptors**: M2 tilt angle increases by Δθ → `τ_decay` multiplier = 1 + Δθ × 0.05 deg^-1 (calibrated against mammalian benchmarks).

These rules are the preregistered MD-to-HH translation. Modifications require an amendment block.

---

## 3. Method

### 3.1 Literature-direct shifts (no MD required)

For ~15 of 25 Tier-1 targets where mammalian or *C. elegans* electrophysiology data is available:

```python
# src/phase_d_kinetic_shifts.py --literature-only
target = "UNC-49"
anesthetic = "halothane"
# Look up shift in literature_shifts.csv; map to occupancy-scaled form
shift_form = "gaba_potentiation"
k_pot_at_saturation = 1.0  # τ_decay doubles at saturating occupancy
occupancy = occupancy_matrix[target, anesthetic, "1x"]
applied_shift = k_pot_at_saturation * occupancy
# Write to anesthetic_kinetic_shifts.npz
```

### 3.2 MD-derived shifts (5-8 targets)

For each MD-targeted system:

```bash
# Step 1: CHARMM-GUI Membrane Builder via web (one-time per system)
# Output: charmm-gui-<jobid>/
# Manual: download tar.gz, extract to artifacts/kinetics/<TARGET>_<ANESTHETIC>/

# Step 2: convert to OpenMM input files
python src/phase_d_kinetic_shifts.py --md-prep \
    --target TWK-18 --anesthetic halothane \
    --charmm-gui-dir artifacts/kinetics/TWK-18_halothane/ \
    --output artifacts/kinetics/TWK-18_halothane_openmm/

# Step 3: equilibration (500 ps NPT)
python -m openmm_runner \
    --system artifacts/kinetics/TWK-18_halothane_openmm/system.xml \
    --topology artifacts/kinetics/TWK-18_halothane_openmm/topology.pdb \
    --steps 250000 --dt 0.002 \
    --temperature 310 --pressure 1.0 \
    --output artifacts/kinetics/TWK-18_halothane_eq.dcd

# Step 4: production (100 ns NPT)
python -m openmm_runner \
    --system artifacts/kinetics/TWK-18_halothane_openmm/system.xml \
    --topology artifacts/kinetics/TWK-18_halothane_eq.pdb \
    --steps 50000000 --dt 0.002 \
    --temperature 310 --pressure 1.0 \
    --output artifacts/kinetics/TWK-18_halothane_prod.dcd \
    --report-interval 5000

# Step 5: analysis pipeline
python src/phase_d_kinetic_shifts.py --md-analyze \
    --target TWK-18 --anesthetic halothane \
    --trajectory artifacts/kinetics/TWK-18_halothane_prod.dcd \
    --output artifacts/kinetics/TWK-18_halothane_shift.json
```

### 3.3 Sanity-check against literature controls

Three MD systems serve as **calibration / sanity checks** with established literature comparison:

- TREK-1 + halothane (mammalian K2P): published Δr from MD literature ≈ +0.5 Å.
- GABA-A α1β2γ2 + isoflurane (mammalian Cys-loop): published M2 tilt change ≈ +3°.
- NALCN + halothane (mammalian NCA homolog): published pore-radius decrease ≈ -0.3 Å.

If Wave P's MD on these mammalian controls produces shifts within 2× of published values, the Wave P MD pipeline is calibrated and the *C. elegans*-specific MD outputs are trusted. If the controls disagree by > 2×, the MD pipeline is suspect and Wave P falls back to literature-only kinetic shifts (with reduced coverage).

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| Literature-direct shift lookups (15 targets) | local CPU | 4 | $0 |
| CHARMM-GUI Membrane Builder web setup (8 systems) | manual / web | 8 wall-clock (10 min compute each) | $0 |
| Equilibration 8 × 500 ps | local RTX 4060 Ti | 8 | $0 |
| Production MD 8 × 100 ns | local RTX 4060 Ti | 96 (12h × 8) | $0 |
| Analysis pipeline (8 trajectories) | local CPU | 4 | $0 |
| Calibration controls (3 mammalian) | local | already counted in 8 | $0 |
| **Total Phase D** | | **~120 GPU-hours + manual setup** | **$0** |

Phase D's MD compute is the largest local-GPU consumer in Wave P. The 4-6 week duration spans most of months 3-4 of the program.

---

## 5. Preregistered success criteria (Gate D.1)

Phase D passes Gate D.1 if and only if:

1. **D.1.1 — Coverage:** ≥ 80% of Tier-1 targets have a kinetic shift estimate (either literature-direct or MD-derived). 20 of 25 minimum.
2. **D.1.2 — MD calibration:** All 3 mammalian-control MD systems produce shifts within 2× of published values. Failure here means the MD pipeline is uncalibrated and downstream MD-derived shifts are unreliable.
3. **D.1.3 — Internal consistency:** For each mechanism class (GABA, GluCl, K2P, etc.), the magnitude and direction of kinetic shift is consistent within the class (no targets in the same class show opposite-sign shifts unless biology supports it).
4. **D.1.4 — Bracket sanity:** Shift parameter uncertainty bracket has finite width but not exceeding 5× central value. Shifts with > 5× bracket are flagged as low-confidence and Phase G uses central-only with a flag.

---

## 6. Halting rules

**Pause and surface:**

- All 3 mammalian-control MD systems disagree with published values by > 2× → MD pipeline is wrong; pause and revisit force field, simulation parameters, or analysis pipeline.
- A single MD system exceeds 200 ns (2× the budget) without converging → halt that system, document, fall back to literature-direct or no-shift.
- Cloud-burst-based enhanced sampling proposed mid-phase → halt and surface explicitly; canonical Phase D plan does not authorize external spend. Any cloud-burst escalation requires user reversal of the zero-spend commitment per `compute_budget.md` §4.

**Document and continue:**

- Single literature shift unavailable for a target → use MD; if MD fails, default `shift_form = none` and document.
- Literature shift has wide uncertainty bracket → use the published mean, document the range.
- MD trajectory has minor convergence concerns (< 10% drift in pocket occupancy over last 30 ns) → document, use the mean over the converged window.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `artifacts/kinetics/literature_shifts.csv` | Per-(target, anesthetic) literature-direct shift |
| `artifacts/kinetics/<TARGET>_<ANESTHETIC>_prod.dcd` | MD trajectory (NOT git-tracked) |
| `artifacts/kinetics/<TARGET>_<ANESTHETIC>_shift.json` | MD-derived per-pair shift |
| `artifacts/kinetics/anesthetic_kinetic_shifts.npz` | **Master output**: per-target kinetic shift table |
| `artifacts/kinetics/calibration_report.md` | Mammalian-control MD vs literature |
| `artifacts/kinetics/phase_d_completion.md` | end-of-block report |

The master `anesthetic_kinetic_shifts.npz` is structured:

```
- targets:          (25,)        target gene names
- anesthetics:      (6,)         anesthetic names
- shift_form:       (25, 6)      shift form per pair
- shift_central:    (25, 6)      central shift parameter at saturating occupancy
- shift_low:        (25, 6)      lower bracket
- shift_high:       (25, 6)      upper bracket
- source:           (25, 6)      'literature_direct' / 'literature_homolog' / 'MD_derived' / 'default' / 'none'
- source_PMID:      (25, 6)      primary source PMID if literature
- meta:             dict         Phase C source matrix; Phase A/B versions
```

---

## 8. Falsifiability checks

The phase's premise: **"Per-target kinetic shifts can be derived from a combination of published literature and targeted MD with sufficient confidence to drive Phase G network simulations."**

Falsified if:

1. Mammalian-control MD systems disagree with literature by > 2× consistently → MD pipeline is broken.
2. Literature search reveals no quantitative kinetic-shift data for any of the 5 mechanism classes → Wave P's translation framework is unfounded.
3. Shift uncertainty brackets are uniformly > 5× the central value → kinetic shifts are too uncertain to drive deterministic network simulation; Wave P pivots to a sensitivity-analysis framing rather than a predictive framing.

---

## 9. Integration points

**Inputs from earlier phases:**

- Phase C `artifacts/occupancy/occupancy_matrix.npz` — per-pair occupancy at clinical concentrations.
- Phase A `artifacts/structures/<TARGET>_multimer/rank_001.pdb` — structures for MD setup.
- Phase B `artifacts/binding/<TARGET>_<ANESTHETIC>_consensus.json` — pose for MD initial placement.

**Outputs consumed by:**

- **Phase E** (Markov synapses) — uses SNARE-machinery shifts directly.
- **Phase F** (metabolic layer) — uses Complex I shifts directly.
- **Phase G** (network runs) — applies all per-target shifts as overlays on Wave 2 channels.

---

## 10. Citation hygiene declaration

- van Swinderen 2004 — (PMID lookup needed). [BLOCKING]
- Crowder 1996 — PMID 8855256. [VERIFIED]
- Mihic 1997, *Nature*, GABA / glycine receptor anesthesia site — (PMID 9311784 likely; verify).
- Sedensky 2001, twk-18, *Am J Physiol Cell Physiol* — PMID 11756669. [VERIFIED]
- Kayser 2001, gas-1 Complex I — (PMID lookup needed). [BLOCKING]
- Hales & Lambert, GABA-A isoflurane τ_decay — (specific paper + PMID needed).
- Eastman 2017, OpenMM — PMID 28746567. [VERIFIED]
- Maier 2015, ff14SB — PMID 26574453. [VERIFIED]
- Klauda 2010, CHARMM lipids — (PMID 20496934 — this is for CHARMM36; LIPID17 has separate AMBER pub).
- Wang 2004, GAFF — DOI `10.1002/jcc.20035`. [VERIFIED]

**Pre-flight verification status:** 4 of 10 verified. 4 PMIDs are blocking items.

---

## 11. Risk register (Phase D)

| Risk | Likelihood | Mitigation |
|---|---|---|
| MD on RTX 4060 Ti runs slower than 30 ns/day | Medium | Enable mixed-precision; use Brian2-style throttle; consider Lambda Labs burst for 1-2 systems |
| AM1-BCC partial charges fail QM convergence on halothane | Low | Use RESP charges as fallback |
| POPC bilayer initial setup unstable | Medium | Use CHARMM-GUI's automated setup; document any manual fixes |
| Anesthetic diffuses out of pocket during equilibration | Medium | Place multiple anesthetics in / around pocket at start; use harmonic restraint during eq |
| Mammalian-control calibration fails | Medium | Documented fallback: literature-only path; reduced coverage |
| MD-derived shift disagrees with literature for a target where both exist | Medium | Use literature value, flag MD discrepancy; document |
| Pre-flight blocking on PMIDs delays phase entry | Low | Citation lookup is < 1 day work; surface to user as a tooling issue |

---

## 12. Phase D execution plan

1. Pre-flight citation verification (4 PMIDs).
2. Compile `literature_shifts.csv` for 15+ literature-direct (target, anesthetic) pairs.
3. Set up CHARMM-GUI Membrane Builder for 5 *C. elegans* systems + 3 mammalian controls.
4. Run mammalian-control MD first (calibration). Gate D.1.2 evaluated.
5. If D.1.2 passes, run *C. elegans*-specific MD systems.
6. Analysis pipeline across all 8 systems.
7. Compile `anesthetic_kinetic_shifts.npz`.
8. Gate D.1 evaluation; end-of-block report.
