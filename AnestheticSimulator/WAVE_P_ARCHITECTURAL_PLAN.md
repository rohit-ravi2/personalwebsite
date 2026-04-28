# Wave P — Architectural Plan

**Document type:** master planning document. Read second after `README.md`.
**Date:** 2026-04-27. **Author:** Wave P kickoff session.
**Status:** SCAFFOLDED. No phase has executed yet.
**Sibling tracks:** Wave 2 (channel translation, in flight); Notebook pipeline (structural connectomics, partially complete).

---

## 1. Executive summary

Wave P is a 6-month parallel research track that builds a **digital pharmacology platform** on top of the existing C. elegans biophysical simulator. The platform's job is to take a multi-target binding profile (anesthetic concentration, per-target affinity, per-target functional shift) and produce a falsifiable network-level prediction (loss of locomotion, EC50 curve, mutant phenotype) that can be matched against published *C. elegans* anesthetic data.

The motivating science: **general anesthesia is multi-target.** No single channel or receptor accounts for clinical immobilization. Halothane, isoflurane, sevoflurane, and propofol each bind dozens of channels (Cys-loop receptors, K2P channels, NCA / NALCN, voltage-gated K and Ca, NMDA, glycine), the SNARE machinery (UNC-13 / UNC-18 / syntaxin / SNAP-25 / synaptobrevin), and mitochondrial Complex I (the gas-1 hypothesis). The clinical effect emerges from the *parallel sum* of those small effects.

The strongest version of this premise is testable: at clinical EC50, multiple targets must show non-trivial occupancy (> 10%). If only one target does, the multi-target premise is **falsified**, and Wave P pivots. Gate C.1 (Phase C, occupancy matrix) is where this test runs.

If Gate C.1 passes, Wave P proceeds through kinetic-shift translation (Phase D), Markov synaptic-machinery overlay (Phase E), metabolic state coupling (Phase F), and full network-level perturbation runs (Phase G), terminating at empirical validation against eight anchor predictions from published wet-lab data (Phase H). Phases I and J are stretch goals (differentiable inverse design; network signature analysis).

**Success criterion for the program as a whole:** at least 4 of 8 anchor predictions match published wet-lab data within 2× tolerance, AND per-target lesion analysis fails to reproduce the full anesthetic effect (= multi-target framing supported).

**Falsification criterion for the program as a whole:** Gate C.1 fails; or at Gate H, the per-target lesion successfully reproduces the full effect from a single target, in which case the multi-target premise is wrong and the simulator is overengineered relative to a single-target model.

---

## 2. Project relationship to Wave 2 and the notebook pipeline

Wave P is the third concurrent track in the C. elegans simulator program. It interfaces with the other two via documented file-level handoffs and does not modify their code.

### 2.1 Wave 2 (channel translation) — Wave P consumes its outputs

Wave 2 is translating Nicoletti 2024's 22 NEURON channel `.mod` files into Brian2. As of 2026-04-26, **7 essential channels are translated and validated**: EGL-19, SHK-1, SHL-1, NCA, KQT-3, SLO-1 (isolated), SLO-1+EGL-19 (coupled). Max divergence < 1% vs NEURON reference.

AVA cell uses a 5-channel set: IRK + LEAK + EGL-19 + NCA + UNC-103. **IRK and UNC-103 are not yet translated** — Wave 2 has them as in-flight items. Wave P uses Nicoletti's NEURON reference for these two channels until Wave 2 ships them. See `integration/wave2_handoff.md` for the consumption protocol.

Wave 2 owns:

- `scripts/brain/wave2/channels/*.py` — per-channel Brian2 implementations.
- `scripts/brain/wave2/translation_patterns.md` — F1-F17 NMODL gotcha catalog (Wave P also uses this when translating receptors that Wave 2 hasn't yet touched).
- `scripts/brain/wave2/voltage_clamp_harness.py` and `plateau_harness.py` — validation harnesses that Wave P reuses for anesthetic-perturbed validation.

**Wave P does not modify any file under `scripts/brain/`.**

### 2.2 Notebook pipeline (structural connectomics) — Wave P consumes its outputs

The notebook pipeline at `New Notebooks/` runs 56 notebooks producing structural priors from Witvliet 2020 (222 neurons), CeNGEN expression, Bentley 2016 peptide-receptor mapping, Cook 2019, and Brittin contact matrix.

Wave P consumes these specific artifacts:

- `New Notebooks/data_derived/connectome_adult.npz` — 222-neuron synaptic adjacency.
- `New Notebooks/data_derived/expression_tpm.npz` — CeNGEN per-neuron transcript counts.
- `New Notebooks/data_derived/nb12_peptide_adjacency.npz` — peptide A_peptide matrix.
- `New Notebooks/data_derived/nb28_neuron_profile.csv` — 8 multiplex archetypes per neuron.
- `New Notebooks/data_derived/nb29_channel_rulebook.csv` — channel-density rules.
- `New Notebooks/data_derived/nb44_*` — per-neuron functional priors.

These priors fill the **290 non-Nicoletti neurons** for which no validated biophysics exists. CeNGEN expression scales channel-density estimates; nb28 archetypes predict anesthetic-sensitivity ranking by archetype (Wired_receiver and Multiplex_integrator have high SNARE load and should be the most SNARE-anesthesia-sensitive). See `integration/notebooks_handoff.md`.

**Wave P does not modify any notebook in `New Notebooks/`.**

### 2.3 Production simulator handoff — only after Phase H lands

The production simulator (`scripts/brain/`) integrates Wave P output via a single documented module that reads `artifacts/kinetics/anesthetic_kinetic_shifts.npz` and applies per-target multipliers to channel parameters at runtime. That handoff is described in `integration/production_simulator_handoff.md` but does NOT activate until Phase H validation gates pass. The production simulator runs unaffected by Wave P scaffolding.

---

## 3. Phase architecture

Wave P is structured as **10 phases (A through J)**, each preregistered in its own document under `preregistration/`. Phases A through H are sequential and load-bearing; Phases I and J are stretch goals.

### 3.1 Sequential pipeline (Phases A through H)

```
[A] Structural priors  →  [B] Binding poses  →  [C] Occupancy matrix
                                                       ↓
                                                  Gate C.1 (load-bearing)
                                                       ↓
[D] Kinetic shifts  →  [E] Markov synapses  →  [F] Metabolic layer
                                                       ↓
                                              [G] Network perturbation runs
                                                       ↓
                                              [H] Empirical validation
                                                       ↓
                                               (program success / fail)
```

### 3.2 Stretch phases (I, J)

```
                                              [H] Empirical validation
                                                  ↓               ↓
                            (if H passes ≥ 6/8)  ↓               ↓ (if H passes ≥ 4/8)
                                  ↓                                ↓
                       [I] JAX inverse design               [J] Network signatures
                                  ↓                                ↓
                       (occupancy estimate vs.            (Phi, Lyapunov, modularity
                         Phase C structural)               vs. mammalian EEG/fMRI)
```

### 3.3 Per-phase summary table

| Phase | Goal | Primary deliverable | Compute | Gate criterion |
|---|---|---|---|---|
| A | Predict 3D structures of all 25 Tier-1 targets | `artifacts/structures/*.pdb` | Local 4060 Ti ~12 GPU-h (ESMFold / Boltz-1 / OpenFold) + free-tier Colab T4 overflow ~10 hr | ≥ 22/25 with pLDDT > 70 at pocket |
| B | Predict anesthetic binding poses | `artifacts/binding/*.sdf` + scores | Local Vina/DiffDock/GNINA ~30h + ~8 hr free-tier Colab T4 overflow | ≥ 70% cross-method pocket agreement; GNINA top-10 cross-method agreement |
| C | Per-target occupancy at clinical conc | `artifacts/occupancy/occupancy_matrix.npz` | Local CPU minutes | **≥ 5 targets > 10% occupancy at 1× EC50** (else multi-target FALSIFIED) |
| D | Translate occupancy to kinetic shifts | `artifacts/kinetics/anesthetic_kinetic_shifts.npz` | Local 4060 Ti + 5-8 MD runs ~120h | ≥ 80% Tier-1 covered; MD-lit agreement 2× |
| E | Markov SNARE synaptic transmission | `artifacts/markov/*` Brian2 modules | Local CPU ~10h | mEPSC freq within 20%; Ca cooperativity n=3-5 |
| F | ATP[t] dynamics + K-ATP coupling | `artifacts/metabolic/*` modules | Local CPU minutes | gas-1 hypersensitivity within 50% |
| G | 2,400 perturbation runs | `artifacts/runs/*.npz` | Local Brian2 ~80h | WT EC50 within 2×; lesion fails to reproduce |
| H | Match 8 anchor predictions | `artifacts/validation/anchor_table.csv` | Local CPU minutes | ≥ 4/8 within 2× tolerance |
| I | Differentiable inverse problem | `artifacts/runs/inverse_occupancy.npz` | Local + Colab ~40h | Inverse occupancy within 3× of Phase C |
| J | Phi / Lyapunov / modularity | `artifacts/runs/signatures.npz` | Local CPU ~20h | Decreased Phi under anesthesia, p < 0.05 |

### 3.4 Per-phase preregistration documents

Each phase has its own document at `preregistration/phase_<letter>_<name>.md`. Each follows the same template:

1. **Goal** (1-2 paragraphs).
2. **Background** — relevant primary literature with verified PMIDs / DOIs.
3. **Method** — concrete tools, exact parameters, command-line examples. Not vague.
4. **Compute budget** — throughput estimates, local vs Colab vs cloud-burst breakdown.
5. **Preregistered success criteria** — numbered, falsifiable, threshold-based.
6. **Halting rules** — what triggers pause-and-surface vs document-and-continue.
7. **Output deliverables** — exact file paths.
8. **Falsifiability checks** — what would kill the phase's premise.
9. **Integration points** — what feeds in from earlier phases; what feeds out.
10. **Citation hygiene declaration** — all citations PMID/DOI verified or marked.
11. **Risk register specific to this phase**.

The preregistration documents are written **before** the phase executes. Modifications during execution require a documented amendment block at the bottom of the document; silent drift is treated as a methodology failure.

---

## 4. Target panel

### 4.1 Tier 1 (~25 targets, build first)

Tier 1 targets are selected on the criterion that they have **either** (a) a strong direct primary-source link to anesthetic mechanism in *C. elegans* (Sedensky 1992 → unc-79; Sedensky 2001 → twk-18; van Swinderen 1999 → unc-13; Morgan & Sedensky 1995 → gas-1) **or** (b) a high-confidence mammalian homolog with established anesthetic binding (GABA-A pentamers, nAChR pentamers, NALCN, K2P channels, mitochondrial Complex I).

| Class | Targets | Anchor mechanism |
|---|---|---|
| Cys-loop GABA receptors | UNC-49 (GABA-A), EXP-1 (cation-GABA) | β+/α− interface (mammalian Olsen lab photolabel data) |
| Cys-loop GluCl receptors | AVR-14, AVR-15, GLC-1, GLC-2 | Hibbs 2011 GluCl crystal structure (PDB 3RHW) |
| Cys-loop nAChR receptors | ACR-16, ACR-2, UNC-29, UNC-38, UNC-63, LEV-1 | Torpedo nAChR M2 helix (Forman/Miller cited) |
| K2P channels | TWK-18, TWK-7, TWK-29 | Sedensky 2001 PMID 11756669 (twk-18 halothane resistance gain-of-function) |
| NCA channel complex | NCA-1, NCA-2, UNC-79, UNC-80, NLF-1 | Sedensky 1992 PMID 1346264 (unc-79/unc-80 halothane resistance) |
| SNARE machinery | UNC-64, RIC-4, SNB-1, UNC-13, UNC-18, SNT-1 | van Swinderen 2004 (Ca cooperativity shift) |
| Complex I | GAS-1, NUO-1, NUO-2, NUO-3, NUO-4, NUO-5, NUO-6 + MEV-1 (Complex II control) | Morgan & Sedensky 1995 PMID 7549290 (gas-1 hypersensitivity); Kayser 2001 |

Total Tier 1 = 25 targets after merging the entries (mev-1 is a control rather than a positive target). Full panel with WormBase IDs, predicted oligomer state, AlphaFold-DB URLs, and mammalian PDB homologs at `targets/tier1_targets.csv`.

### 4.2 Tier 2 (~25 targets, after Tier 1 lands)

Tier 2 expands to glutamate receptors (NMR-1/2, GLR-1 through GLR-8), Ca channels (EGL-19 redux for completeness, UNC-2, CCA-1), additional voltage-gated K (SHK-1, SHL-1, KVS-1, SLO-1/2, IRK-1/2/3, KQT-1/3, EXP-2), peptide processing enzymes (EGL-3, EGL-21, KPC-1, NEP-1), and monoamine receptors. See `targets/tier2_targets.csv`.

Tier 2 only activates after Tier 1 ships through Phase H. There is no preregistered commitment to Tier 2 — it is a stretch panel that depends on Tier 1's outcome.

### 4.3 Anesthetic panel

| Anesthetic | Class | C. elegans EC50 | Anchor paper |
|---|---|---|---|
| Halothane | Volatile | ~3% atm (~340 µM aqueous) | Crowder 1996 PNAS PMID 8855256; Sedensky 1992 PMID 1346264 |
| Isoflurane | Volatile | ~5-7% atm | Crowder 1996; Morgan 1995 |
| Sevoflurane | Volatile | ~7% atm | Crowder lab |
| Propofol | IV | µM range | Boddington 2017 (PMID lookup needed) |
| Ketamine | NMDA antagonist (control) | mM range | (PMID lookup needed) |
| Etomidate | GABA-A specificity (control) | sub-µM | (PMID lookup needed) |

Full panel with SMILES, MW, log P, oil/water partition coefficient, and per-anesthetic anchor papers at `anesthetics/anesthetic_panel.csv`.

---

## 5. Key load-bearing methodological commitments

### 5.1 Membrane-partition adjustment is mandatory at Phase C

Halothane oil/water partition coefficient ≈ 250; isoflurane ≈ 90; propofol ≈ 1000+. These molecules concentrate near membranes 2-3 orders of magnitude above bulk aqueous concentration. Any occupancy estimate that uses bulk aqueous concentration without partition adjustment **systematically underestimates membrane-target occupancy by two to three orders of magnitude**.

Phase C's occupancy calculator uses the **membrane-adjusted concentration** for membrane-embedded targets (channels, receptors) and the **bulk aqueous concentration** for cytosolic targets (Complex I subunits, partial — Complex I is matrix-side, but most of its anesthetic-binding sites in NDUFS2/GAS-1 are in the membrane-embedded portion, so partition still applies).

This is documented as a per-target choice in `targets/tier1_targets.csv` and is a Phase C amendment if any target's compartment is reassigned.

### 5.2 Per-target lesion analysis is the load-bearing test of multi-target framing

At Phase G (network-level runs), each anesthetic effect is applied **alone** in a separate run series:

- Run series G.1: full multi-target effect (control).
- Run series G.2.1: only GABA-A potentiation, no other targets perturbed.
- Run series G.2.2: only NCA block, no other targets perturbed.
- Run series G.2.3: only K2P activation, no other targets perturbed.
- Run series G.2.4: only SNARE shift, no other targets perturbed.
- Run series G.2.5: only Complex I block, no other targets perturbed.
- (Per-class lesions, not per-target; per-target is Tier-2 expansion.)

The pass criterion for the multi-target framing is: **G.2.1 through G.2.5 individually fail to reproduce G.1's loss-of-locomotion threshold, but their combination (G.1) succeeds**. If any single G.2.x reproduces G.1 within 50%, the multi-target premise is empirically falsified at Phase G — the network behavior is dominated by one mechanism, and multi-target is overfit.

This is the second falsifiability checkpoint after Gate C.1.

### 5.3 Citation hygiene from day 1

Wave 2 lost approximately 3-4 weeks of work to the Mellem 2008 misattribution (the "20 mV / 600 ms in AVA" target turned out to be in RMD, with no specific numerical values quantified in the primary source). Wave 2 also caught Wang 2001 → SHK-1 misattribution (Wang 2001 is about SLO-1, not SHK-1) and a Liu 2018 / 2020 year drift.

Wave P enforces the following from day 1:

- Every cited paper carries PMID or DOI in the document where it appears.
- Where a PMID could not be verified at scaffolding time, the citation is marked `(PMID lookup needed)` rather than fabricated.
- Before any phase enters its execution work block, a citation-verification pre-flight checks every cited paper in that phase's preregistration document. Unverified citations block the phase.
- Quantitative biological targets (e.g., "halothane EC50 in *C. elegans* is ~3% atm") must trace to a specific figure or table in a specific primary source. "Approximate" or "near" framings without source are flagged.

The citation discipline is enforced by `validation/empirical_anchors.md`, which is the per-paper validation matrix. Every claim made in any preregistration document or CSV table must appear in this matrix or be flagged as unsourced.

### 5.4 No wet-lab work

Wave P is theoretical and computational only. Validation runs against published wet-lab data; the program does not generate any new wet-lab data and does not propose any wet-lab experiments. This constraint is imposed by the user's standing instructions and is non-negotiable.

### 5.5 Honest scope tags

Every claim in Wave P documents is labeled with one of:

- **SHIPPED** — implementation is complete, tested, and produces validated output.
- **SCAFFOLDED** — file structure and skeleton exist, but the implementation is a stub that prints "scaffolding pending" when run.
- **CALIBRATION-PENDING** — implementation is complete but parameters are placeholders awaiting empirical tuning.
- **DEFERRED** — explicitly out of scope for the current work block; tracked for future consideration.
- **SPECULATIVE** — discussed as a research direction but no commitment to building or validating.

The current state of every Wave P deliverable is SCAFFOLDED. Nothing has shipped. This is the kickoff package.

---

## 6. Compute budget rollup

The following is the planned compute envelope across the 6-month program. Per-phase numbers are in `infrastructure/compute_budget.md`; this table is the program-level rollup. **External spend: $0.**

| Resource | Total budget | Phase distribution |
|---|---|---|
| Local RTX 4060 Ti GPU-hours | ~700 | A:12 (ESMFold / Boltz-1 / OpenFold), B:30, C:0, D:120 (MD), E:10, F:0, G:80, H:0, I:40, J:20 |
| Local CPU-hours | ~200 | A:4, B:5, C:5, D:4, E:10, F:5, G:4, H:8, I:80, J:20 |
| Colab free-tier (T4) hours — overflow only | ~30 cumulative | A:10 (pentameric edge cases), B:8 (DiffDock receptors > 30 Å), I:12 (JAX overflow), others minimal |
| External spend (cloud burst, paid Colab, etc.) | **$0** | DEFERRED: FEP cloud burst on top-10 hits ($200-400) is documented but not authorized. See `preregistration/phase_b_binding_pose.md` §13. |

Compute is dominated by Phase D (~120 GPU-hours of OpenMM MD on missing-channel + anesthetic systems in POPC bilayer) and Phase G (~80 GPU-hours of Brian2 perturbation runs across 2,400 simulations). Phase A structure prediction is now local-first via ESMFold / Boltz-1 / OpenFold (MIT / Apache 2.0); free-tier Colab T4 is overflow only.

The realistic month-1 throughput is ~120 local GPU-hours + free-tier Colab overflow as needed, which fits Phase A + Phase B comfortably. Phase C is CPU-trivial. Phases D, E, F, G are months 2-5; Phase H is month 5-6; Phases I and J are deferred / stretch. **No external spend at any phase in the canonical plan.**

---

## 7. Risk register summary

Full risk register at `risk/risk_register.md`. Top risks summarized here:

1. **Pentameric structure prediction fails to fit in 8 GB VRAM** → fallback ladder: ESMFold (MIT) → Boltz-1 (MIT) → OpenFold (Apache 2.0) → ColabFold free tier T4 → subunit-by-subunit pocket modeling. [Phase A; risk register R14]
2. **Vina, DiffDock, GNINA disagree on most pockets** → flag uncertainty, photolabel cross-check, GNINA cross-method-agreement Gate B.1.4; escalate to deferred FEP path only on explicit user authorization. [Phase B]
3. **MD-derived kinetic shifts diverge from literature by > 2×** → 2× tolerance gate, revisit force field choice, fall back to literature-only translation. [Phase D]
4. **Multi-target framing falsified at Gate C.1** → pivot to single-target validation framework; Wave P scope shrinks; document the negative result as a publishable finding. [Phase C]
5. **WT EC50 wrong by > 5× at Phase G** → binding affinity miscalibration; revisit Phase B docking parameters and GNINA-to-Kd conversion; deferred FEP path available if user authorizes. [Phase G/H]
6. **gas-1 hypersensitivity does not reproduce** → metabolic layer wrong; revisit Phase F K-ATP coupling; or the gas-1 hypothesis itself is wrong (less likely). [Phase F/H]
7. **Complex I full-assembly intractable on local hardware** → canonical plan scopes to single-subunit-per-anesthetic-binding-site (GAS-1 primary; NUO-1 through NUO-6 individually). Full assembly is DEFERRED. [Phase A/F; risk register R21]
8. **JAX differentiable simulator does not converge** → defer Phase I; Phase H validates without it. Stretch goal anyway. [Phase I]
9. **Compute scope creep** → bound each phase by preregistered compute budget; Tier 2 only after Tier 1 ships; **no external spend without explicit user reversal of zero-cost commitment**. [program-level]
10. **Citation misattribution propagation** → mandatory pre-flight verification (Wave 2 lesson). Block phase entry on unverified PMIDs. [program-level]
11. **UNC-79 / UNC-80 NCA-complex auxiliary subunits structurally challenging** → focus on functional domain; structure predictors may fail on these; use Cook 2019 + structural homology arguments. [Phase A/B]

---

## 8. Timeline (6-month program)

Full month-by-month at `timeline/timeline.md`. Summary:

| Month | Phases active | Soft milestone |
|---|---|---|
| 1 | A (structures), tooling setup | All Tier-1 monomer structures predicted; pentamer pipeline working |
| 2 | B (docking), C (occupancy + Gate C.1) | Gate C.1 passed or program pivots |
| 3 | D (kinetic shifts; MD runs), E (Markov synapses) | Kinetic-shift table complete for ≥ 80% Tier-1; SNARE Markov module validated |
| 4 | F (metabolic), G (network runs start) | Metabolic layer validates gas-1; first 600 of 2,400 runs done |
| 5 | G (continued), H (validation) | All 2,400 runs done; anchor predictions tabulated |
| 6 | H (write-up), I + J optional | Wave P paper draft; stretch phases if H ≥ 6/8 anchors |

Wave 2 in parallel (consumed by Wave P): Wave 2 expected to ship IRK and UNC-103 by month 2; Tier-2 channel translations as needed by month 4-5.

---

## 9. Paper trajectory

Wave P targets a single paper at the end of the 6-month program:

**Title (working):** "Network-level pharmacology of *C. elegans* anesthesia: predicted multi-target binding profile reproduces immobilization EC50 and mutant phenotypes."

**Target venues** (in preference order): Cell Systems, Neuron, eLife, PLOS Computational Biology.

**Falsifiable claims**:

1. The predicted occupancy profile at clinical halothane EC50 shows N ≥ 5 targets with > 10% occupancy (Phase C).
2. The simulated WT halothane immobilization EC50 matches Crowder 1996 within 2× (Phase H).
3. Per-target lesion analysis fails to reproduce the WT effect — multi-target framing supported (Phase G).
4. The simulated *gas-1* hypersensitivity matches Morgan & Sedensky 1995 within 50% (Phase H).
5. The simulated *unc-79*, *twk-18*, and *unc-13* mutant phenotypes match published data within 50% (Phase H).

Outline at `papers/wave_p_paper_outline.md`.

---

## 10. Open questions that surface as the program runs

These are explicitly **not** preregistered with answers — they are surfacing-points where mid-flight discussion is expected:

- **Which mammalian PDB homolog is the right anchor for each *C. elegans* target?** Some targets (UNC-49 GABA-A) have clear mammalian homologs (6X3X). Others (GLC-1 GluCl) have a *C. elegans* crystal structure (3RHW Hibbs 2011) — use it directly. Others (TWK-18) have no high-resolution mammalian K2P at the relevant pocket. Per-target judgment, documented in `targets/target_panel_rationale.md`.

- **How aggressive should the membrane-partition adjustment be?** Halothane log P_oct ≈ 2.3 (oil/water ~250). For a transmembrane channel pocket, the local concentration is "membrane-side" not "aqueous bulk." Phase C uses K_p × [aqueous] for membrane-embedded pockets. If a target's pocket is on the extracellular face, K_p may not apply — flagged per-target.

- **Is the Hill coefficient n = 1 the right default?** Cys-loop receptors are pentameric and may have n > 1 at the binding site (cooperative binding across subunits). Wave P defaults to n = 1 for tractability and surfaces n > 1 only if per-target literature provides it. Documented as a methodological choice in Phase C.

- **Where does the simulator's behavioral readout live?** The production simulator's current FSM produces behavioral states (FORWARD, REVERSE, QUIESCENT) from command-neuron firing rates. Wave P's Phase G uses the same FSM but adds an "IMMOBILIZED" state defined as: command-neuron firing rate < threshold for > 10 sec across all motor-pool drivers. The threshold is calibrated against the WT control run (G.1.0).

These questions are tracked in `STATUS.md` and resolved as phases execute.

---

## 11. Cross-session methodology

Wave P inherits Wave 2's cross-session discipline:

- **Plan-first.** Each phase has a preregistration document. Modifications during execution require a documented amendment block.
- **Pre-flight pushback** before substantive moves. If a phase's plan has a load-bearing assumption (e.g., "halothane EC50 is 3% atm"), surface and verify it before executing.
- **Mid-flight surfacing.** If a phase produces a result that contradicts the preregistered expectation by > 2×, stop and surface — do not silently document and continue.
- **Stop-and-ask discipline.** If a phase encounters unexpected state (missing dependencies, license terms that differ from expectation, source-paper claim that doesn't match cited figure), ask rather than guess.
- **Cross-track adversarial review.** Wave 2 used three parallel sessions with adversarial review. Wave P starts as a single track but reserves the option to fork into adversarial review for high-stakes phases (especially Phase C and Phase H).

The methodology is non-negotiable. It is the reason Wave 2 caught its citation misattributions before they propagated.

---

## 12. Final commitments

Wave P kicks off under the following commitments:

1. **Multi-target framing** is the working hypothesis, falsifiable at Gate C.1 and Gate G (per-target lesion).
2. **Phase ordering A → H** is sequential and load-bearing. No Phase D work begins until Gate C.1 passes.
3. **Citation hygiene** is enforced from day 1. Unverified PMIDs block phase entry.
4. **No wet-lab work.** Validation is against published data only.
5. **Wave 2 and the notebook pipeline are not modified.** Wave P consumes their outputs through documented handoffs.
6. **Every claim is scope-tagged.** SHIPPED / SCAFFOLDED / CALIBRATION-PENDING / DEFERRED / SPECULATIVE.
7. **The kickoff package (this document set) does not ship implementation.** It establishes the scaffolding. Phase A executes in the first work block after this kickoff.

Wave P is a research program, not a deliverable. The 6-month outcome is a paper draft (or a documented negative result if Gate C.1 falsifies the premise) and a digital pharmacology platform that plugs into the production simulator.
