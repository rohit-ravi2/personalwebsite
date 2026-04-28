# Phase V Wave 1 — Research-Tool Feasibility Roadmap

**Companion to** `phase_v_w1_biophysical_audit_matrix.md`. Strategic synthesis of the audit matrix into prioritized work blocks, dependencies, application-specific milestones, and gating decisions.

---

## Executive summary

Three findings dominate the audit.

**1. The project's strategic distinguishing potential is integration above the channel layer, not biophysical depth on channels themselves.** OpenWorm c302 + Nicoletti 2019/2024 have already implemented Hodgkin-Huxley models for ~12 of the worm's voltage-gated channels with worm-specific kinetic parameters from Fawcett 2006, Johnstone 1997, and others. Re-implementing these from primary sources would be redundant. The realistic move is **import + integrate** their channel library, freeing bandwidth for the layers above (CeNGEN-coupled densities, peptide processing refinement, GABA_A allosteric framework if anesthesia is the application).

**2. Three application targets have very different feasibility profiles, and the project should commit to one as primary.** "Research-grade biophysical tool" isn't monolithic. (a) **General drug discovery / compound screening** is mostly aspirational — pharma-grade compound screening requires industrial-scale validation pipelines this project cannot match. (b) **Mechanistic insight into specific drug classes (anti-parasitics, anesthesia)** is research-artifact-feasible if the project commits to specific receptor pharmacology (UNC-49 GABA_A allosteric for anesthesia; AVR-14/15 GluCl for anthelmintic). (c) **Mechanistic exploration of specific worm phenomena (sleep, lethargus, foraging, learning)** is the most feasible and best-supported by data; this is where the project's CeNGEN integration + peptide processing refinement adds unique value. Recommendation: commit to (c) as primary application, (b) as opportunistic secondary if specific drug-class collaborator emerges, (a) as not-a-realistic-target.

**3. Three categories return "data-sparse, mostly aspirational" verdicts, which is informative but bounded.** Category E (epigenetics) — worm 5mC is minimal, primary modifications are 6mA + histones with sparse coupling to neural-dynamics applications. Category H (Hebbian plasticity) — worm uses graded transmission, no STDP in classical sense; habituation is the worm-native plasticity. Mitochondrial Ca + microdomains in Category I, neurosteroid + neuron-glia + metabolic coupling in Category J. These are 25-30% of the audit's mechanism inventory; the project should treat them as **not-near-term-targets** and not strategically position around them.

The realistic ceiling on what this project can become: a **mid-fidelity, well-validated, worm-grounded mechanistic simulator with unique strengths in CeNGEN integration + peptide processing + sensory cascades + behavioral closure**. Better than OpenWorm in specific dimensions (CeNGEN-grounded densities, modular peptide processing, integrated sensory transduction). Comparable in ion-channel depth (importable). Not competitive with industrial-scale drug-discovery platforms; not a biophysics research tool replacing wet-lab electrophysiology. **A worthy multi-year goal that produces papers along the way.**

---

## Per-category feasibility synthesis

### Category A (Ion channels) — production-grade-feasible, mostly via import

15 of 18 listed channels have rich-to-moderate worm-specific kinetic data AND existing implementations in Nicoletti 2019 / 2024 / c302 / OpenWorm muscle models. The strategic move is **integrate Nicoletti 2019's channel library into the simulator's compartmental-cell roster**. Estimated 1-2 weeks. Exceptions: HCN (worm role unclear, exploratory), TWK family (data sparse but **central for anesthesia work**), CLC chloride (lower utility). The TWK-40 work from Nichols 2017 is the high-leverage anesthesia-relevant gap not in c302 standard.

### Category B (Synaptic transmission machinery) — research-artifact-grade

Worm genetics rich (UNC-13, UNC-18, UNC-10/RIM, SNT-1, SNT-3, TOM-1, UNC-31/CAPS); quantitative kinetics mostly mammalian priors. Highest-leverage near-term: **receptor binding kinetics for major ligand-gated channels** (overlaps with Category C). Multi-pool vesicle dynamics, SNARE complex Markov schemes — research-artifact territory; useful for short-term plasticity research but not foundational for all applications.

### Category C (Receptor pharmacodynamics) — mixed; production-priority for specific applications

Receptor genetics rich; quantitative pharmacology mostly heterologous expression + mammalian priors. Highest research utility:

- **GABA_A (UNC-49) allosteric framework** is the central anesthesia-mechanism gap. Without it, anesthesia mechanism research isn't really achievable. Allosteric modulation modeling is high implementation complexity AND data-sparse for worm-specific quantitative claims. **Production-priority for anesthesia work; aspirational for quantitative claims without wet-lab validation.**
- **GluCl (AVR-14/15) kinetic upgrade** is anthelmintic-drug-discovery-relevant; data is rich for worm specifically. **Production-grade target.**
- **nAChR family (UNC-29/38/63 + others)** — anti-parasitic-drug-discovery-relevant; data rich. **Production-grade target.**
- **iGluR family (GLR-1/2/.../8, NMR-1/2)** — mammalian-priors-importable Markov schemes; research-artifact-grade.
- **GPCR-based modulator receptors** — current direct-current-modulation abstraction is sufficient for many applications; full GPCR-G-protein-cycle coupling is research-artifact-grade.

### Category D (Gene expression coupled to dynamics) — one production-priority entry

**Single high-leverage move: per-cell channel/receptor densities scaled by CeNGEN TPM.** The data is loaded already. Integration is a g_max multiplier per cell type. This is **nearly free relative to any other Category D mechanism** and would be a meaningful step toward biology-grounded simulator. Listed as T4-#8 in the original roadmap; should probably move up. Everything else in Category D (immediate-early gene transcription, mRNA localization, TF cascades to channel insertion, hours-days timescale dynamics) is research-artifact or aspirational due to worm-specific data sparsity and timescale mismatches with the current simulator's minutes-scale runs.

### Category E (Epigenetics) — mostly aspirational

C. elegans epigenetic landscape is unusual (minimal 5mC, primary modifications are 6mA via DAMT-1 plus histone marks). Genetic data is rich (Greer lab, Strome lab) but quantitative coupling to neural-dynamics applications is sparse. **Not a near-term implementation target.** May surface as a long-term research direction after other categories mature; suitable for collaboration with epigenetics-focused labs rather than primary work in this project.

### Category F (Peptide processing) — research-artifact-grade, refinement of existing modulator layer

Worm-specific data unusually rich (Husson 2007/2009 mass spec inventories, genetic dissection of BLI-4, EGL-3, KPC-1; UNC-31/CAPS DCV release machinery). Current modulator layer in the simulator is a coarse abstraction; refining to per-peptide kinetic dynamics is research-artifact-grade work that's importable from existing data. **Sub-priority for paper 2 (behavioral closure) but core for paper 3+ (mechanistic claims about modulation).** Estimated 3-4 weeks for refinement.

### Category G (Second messenger cascades) — sensory cascades already production-grade; rest research-artifact

Sensory transduction cascades (ASE/AWC/ASH/AFD/ALM) are **already implemented in the simulator** at production level (`sensory_transduction.py`). Extending the same approach to GPCR-modulated cascades (cAMP/PKA, IP3/Ca/PKC, cGMP/PKG more broadly) is research-artifact-grade work. Cross-talk between cascades is aspirational.

### Category H (Plasticity) — habituation realistic; Hebbian doesn't translate

"Plasticity" framing imports mammalian assumptions. Worm uses graded transmission; STDP doesn't apply. **Habituation/dishabituation in the tap-withdrawal circuit is the worm-native plasticity** with strong Rankin-lab literature. Modulator-induced gain changes are the second-most-grounded option. Hebbian LTP/STDP are not biologically apt for worm. Realistic implementation: habituation circuit + modulator-induced gain modulation at the network level. Lower priority for anesthesia/drug discovery; central for learning/memory research if that becomes an application focus.

### Category I (Ca signaling) — bulk pool sufficient near-term; ER dynamics research-artifact

Bulk cytosolic Ca pool (current h_kca patch with corrected α_Ca) is sufficient for plateau dynamics. ER Ca dynamics (ITR-1 IP3R, UNC-68 RyR, SCA-1 SERCA) are research-artifact-grade work, importable from mammalian priors with worm validation. Microdomains and mitochondrial Ca are aspirational. The α_Ca calibration question from yesterday's Wave 1 work falls cleanly in this category and is now resolved.

### Category J (Other) — gap junction modulation is the highest-leverage addition

Worm gap junctions (innexins UNC-7, UNC-9, INX family) are central to network dynamics; current implementation is fixed-strength. Voltage-dependent gap junction conductance modeling is research-artifact-grade and would refine network dynamics meaningfully. Volume transmission of monoamines, neurosteroid synthesis, neuron-glia, activity-dependent metabolism are all lower priority or aspirational.

---

## Cross-category dependencies

**Foundational chain (must be addressed before later additions are meaningful):**

1. **Channel library integration** (Category A, import from Nicoletti 2019) → produces compartmental-grade cellular dynamics across the 14+ plateau-relevant cells.
2. **Per-cell channel densities scaled by CeNGEN TPM** (Category D) → makes the channel library biology-grounded at per-neuron resolution.
3. **K_Ca + h-inactivation calibration** (Category A SLO-1 + I, the work-in-progress from Wave 1) → resolves the plateau termination mechanism. **Currently in progress via Session 3's compartmental cellular validation.**

After foundational chain:

4. **Receptor binding kinetics for major ligand-gated channels** (Category B + C) → upgrades synaptic transmission from instantaneous to kinetically accurate. Required for anesthesia mechanism work; useful for plasticity work.
5. **Peptide processing refinement** (Category F) → upgrades modulator layer from coarse abstraction to per-peptide kinetics with literature-grounded processing.
6. **GABA_A allosteric framework** (Category C, anesthesia-specific) → enables anesthesia mechanism work IF that becomes a primary application.

**Independent additions (don't block foundational chain):**

- Habituation/dishabituation circuit (Category H) — adds Rankin-lab learning model
- ER Ca dynamics (Category I) — refines plateau termination + intracellular signaling
- Gap junction voltage-dependence (Category J) — refines network dynamics
- GPCR cascade refinement (Category G) — refines modulator effects beyond direct current modulation

**Aspirational / data-gated:**

- Hours-days transcriptional dynamics (Category D)
- Hebbian plasticity (Category H)
- Epigenetic effects (Category E)
- Mitochondrial Ca, microdomains, neuron-glia, metabolic coupling

---

## Application-specific feasibility summary

### Anesthesia mechanism research

**Achievable subset:**
- Volatile anesthetic effects on TWK-40 (data exists, Nichols 2017)
- Volatile anesthetic effects on SLO-1 (slo-1 alcohol/anesthesia phenotypes well-characterized)
- Some anesthetic effects on EGL-19 / L-type Ca (mammalian priors)

**Substantial barrier:**
- GABA_A allosteric framework needed for propofol/etomidate. Worm has UNC-49 (GABA_A homolog) but quantitative allosteric modulation data is sparse. Would require either: (a) heterologous expression studies that don't exist in worm-specific literature, (b) extrapolation from mammalian GABA_A allosteric data, or (c) wet-lab work the project can't do.

**Realistic verdict: research-artifact-grade for some specific anesthetic mechanisms; not production-grade for general anesthesia mechanism claims.** The project could publish "TWK-40 + SLO-1 contributions to anesthetic-induced loss of arousal in C. elegans simulator" as a paper. It cannot become a tool that pharma uses to predict anesthetic potency.

### Anti-parasitic drug discovery

**Achievable:**
- Ivermectin / macrocyclic lactone effects on AVR-14/15 GluCl (worm-specific data rich)
- Levamisole / pyrantel effects on UNC-29/38/63 nAChR family (worm-specific data rich)
- Albendazole effects on β-tubulin (less neural, but precedent)

**Realistic verdict: research-artifact-grade with potential for production-grade if specific drug-class focus.** This is actually one of the more achievable application paths because the worm IS the model organism for anti-parasitics — the data ecosystem aligns. Could become a meaningful collaboration target.

### Plasticity / learning research

**Achievable:**
- Habituation/dishabituation in tap-withdrawal circuit (Rankin-lab data extensive)
- Modulator-induced gain changes (5HT, DA, OA, TA effects on synaptic gain)
- GLR-1 receptor trafficking (Burbea 2002, Grunwald 2004)

**Substantial barrier:**
- Hebbian-style LTP/STDP doesn't apply to worm graded transmission. Imports mammalian assumptions that fail.

**Realistic verdict: research-artifact-grade for worm-native plasticity (habituation + modulator-induced gain).** Different framing than mammalian-style plasticity work.

### Mechanistic exploration of specific worm phenomena

**Most achievable application class.** Sleep/lethargus (Turek 2016, RIS), foraging (food sensing → AVA modulation), satiety quiescence (DAF-7 → DAF-1 → RIS, see yesterday's RIS investigation), reversal command dynamics (today's Wave 1 work), learning circuits (habituation). Worm-specific data is rich; mechanisms are well-characterized at the genetic level; quantitative coupling to behavior is the project's distinguishing potential.

**Realistic verdict: research-artifact-grade with paths to production-grade for specific phenomena.** This is where the project's CeNGEN integration + peptide processing + sensory cascades + Mellem-grounded compartmental dynamics deliver unique value.

### General drug discovery / compound screening

**Mostly aspirational.** Pharma-grade compound screening requires industrial-scale validation pipelines, comprehensive receptor libraries with kinetic data, integration with high-throughput experimental data — none of which this project can build. Can be approached as "mechanism prediction for specific compound classes" (anti-parasitics in particular) but not general-purpose compound screening.

---

## Recommended near-term work blocks

**No timelines.** Each block is gated by completion of dependencies, not calendar.

### Block 1: Compartmental scaffold integration + Mellem cellular validation (currently in progress)

- Session 3's compartmental cellular validation (in flight) determines γ vs β' architectural commitment
- If γ passes Mellem cellular targets: integrate compartmental scaffold for the 14 plateau cells
- Voltage-scale fix (v_rest = -25 mV per cell) applied
- K_Ca + h calibration finalized (post-corrections from Wave 1)
- Per-cell tau_h calibration

**Gates:** Session 3 results tonight.

### Block 2: Channel library import from Nicoletti 2019

- Import HH-style implementations of ~12 voltage-gated channels (SHL-1, SHK-1, EGL-36, EGL-2, KQT-1/2/3, EXP-2, IRK-1/2/3, KVS-1, UNC-2, CCA-1, EGL-19)
- Validate cellular dynamics against Nicoletti 2019 published traces for AWCon, RMD, AVA, AIY, RIM, VA, VB, VD
- Calibrate per-cell channel densities

**Gates:** Block 1 architectural commitment.

### Block 3: CeNGEN-coupled per-cell channel densities

- Use existing CeNGEN expression data to scale g_max per channel per cell type
- Validate against Block 2's published-cell traces
- Document scaling methodology for cells without Nicoletti reference traces

**Gates:** Block 2 channel library available.

### Block 4: Receptor binding kinetics for major ligand-gated channels

- Implement Markov-state schemes for: UNC-49 (GABA_A), AVR-14/15 (GluCl), GLR-1/2/.../8 + NMR-1/2 (iGluR), nAChR family
- Calibrate from worm-specific dose-response where available, mammalian priors otherwise
- Validate against published synaptic responses (e.g., AVA depolarization to ASH activation per Lindsay 2011)

**Gates:** Block 1 architectural commitment. Independent of Block 2-3.

### Block 5: Peptide processing refinement

- Upgrade modulator layer from "release rate proportional to firing" to per-peptide processing pipeline
- Pre-pro-peptide → mature peptide via BLI-4/EGL-3/KPC-1
- DCV release via UNC-31/CAPS distinct from SV release
- Per-peptide diffusion + degradation + receptor-specific activation

**Gates:** independent of Blocks 1-4.

### Block 6 (anesthesia-specific, optional): GABA_A allosteric framework

- Markov-state scheme for UNC-49 with allosteric binding sites
- Volatile anesthetic effects on SLO-1, TWK-40, EGL-19 (data permitting)
- Validation against worm anesthesia phenotypes (loss of locomotion under volatiles)

**Gates:** Block 4 (receptor kinetics infrastructure). Activated only if anesthesia is the primary application focus.

---

## Recommended longer-term direction

**Multi-year arc, no timelines.** Sequencing matters more than calendar.

**Phase A — foundation:** Blocks 1-3 (compartmental + channel library + CeNGEN-coupled densities). Output: simulator with biology-grounded cellular dynamics.

**Phase B — refinement:** Blocks 4-5 (receptor kinetics + peptide processing). Output: simulator with biology-grounded synaptic + modulator dynamics.

**Phase C — application focus:** Block 6 if anesthesia, OR specific worm-phenomenon-focused blocks (sleep mechanism, learning circuit, foraging dynamics). Output: paper-publishable mechanistic claims for specific applications.

**Phase D — research artifact maturation:** documentation, validation against held-out experimental datasets, openness/usability for other labs to use, comparison with c302 and MetaWorm.

**Open question for synthesis:** does the project commit to anti-parasitic drug discovery as the primary application path? It's the most-feasible-given-data option per this audit. Worm-specific kinetic data exists for AVR-14/15 + nAChR family. Anti-parasitic drug discovery is a real $5+ billion/year industry. C. elegans is the canonical model organism for this. The project could become a useful tool for academic collaboration with anti-parasitic drug development groups — different application than the originally-imagined "general drug discovery" but more achievable.

---

## Open questions for synthesis discussion

1. **Application focus commitment.** "Research-grade biophysical tool" isn't monolithic. The project should commit to one primary application (recommended: mechanistic exploration of specific worm phenomena, with anti-parasitic mechanism research as opportunistic secondary). Alternative recommendations welcome.

2. **OpenWorm / MetaWorm collaboration vs independent development.** This audit found that c302 + Nicoletti 2019/2024 have done substantial work that's importable. Should this project formally collaborate, or fork their work, or develop independently? Different decisions imply different sequencing.

3. **Timescale ambition.** Categories D (gene expression) and H (plasticity) imply hours-to-days simulation timescales beyond the current minutes-scale runs. Does the project commit to extending temporal scope, or stay at minutes-scale and treat slow phenomena as boundary conditions?

4. **Wet-lab dependency.** Three categories return "data-sparse, requires external research progress to mature." If the project genuinely commits to research-tool development, should it also seek collaborator labs to generate worm-specific data for those gaps? Or accept the data ceiling and work within it?

5. **Paper sequencing under reframing.** Paper 1 (multi-modal connectomics) ships near-term independent of simulator. Paper 2 (behavioral simulator) needs Block 1 architectural commitment. Paper 3 (mechanistic claims) needs Blocks 1-5 minimum. Subsequent papers need application-specific focus. The reframing implies paper 4+ is the actual research-tool publication. Is that the right sequence?

6. **Validation against held-out data.** The audit identifies validation pathways per mechanism but doesn't specify which held-out datasets to validate against. Atanas 2023, Hallinen 2021, Yemini 2021 (NeuroPAL imaging), Kato 2015, Skora 2018 are candidates. Should validation methodology be standardized across blocks?

---

## Methodological note on this audit

This audit is **survey-level** — single-session, literature-informed, codebase-examined. Implementation complexity assessments are order-of-magnitude (low / medium / high) rather than exact estimates. Where literature reads conflicted with my prior expectations (Category A had more existing implementation than I expected, Category D had less coupling-data than the prompt suggested), I've documented the corrections.

**Confidence levels per category:** Categories A, B, F have high-confidence assessments (rich literature, clear codebase state). Categories C, D, G, I have moderate confidence (literature exists but quantitative data variable). Categories E, H, J have lower confidence on quantitative claims (data sparse + claims require extrapolation).

**Implementation estimates have ~30% upside risk:** any "2-week task" should be budgeted at 4-6 weeks for engineering surprises, integration issues, parameter calibration cycles, validation iteration. This is consistent with today's Wave 1 experience where Phase 0.5 calibration revealed parameter inconsistencies that changed the work scope.

**Audit's hidden assumption to flag:** I've assumed the simulator stays at single-cell + network-level resolution (no full molecular dynamics, no atomic-level structural modeling). For some applications (especially structure-based drug design), this is the wrong abstraction; tools like Rosetta, AlphaFold-based interactions, molecular dynamics with NAMD/GROMACS are needed. Those are out of scope per the prompt's "not atomic-level molecular dynamics" framing, but worth surfacing that the abstraction layer chosen here doesn't reach all "drug discovery" applications.

---

## Conclusion

The strategic reframing (research-grade biophysical tool) is **partially feasible given available data and project constraints**:

- **Achievable:** mid-fidelity simulator with biology-grounded ion channels (importable), CeNGEN-coupled densities (data ready), refined synaptic/modulator dynamics (research-artifact-grade), specific application focus (worm-native phenomena most-achievable, anti-parasitics second-most).
- **Aspirational:** general drug-discovery / compound screening, anesthesia mechanism research without GABA_A allosteric framework (requires data gap-filling), Hebbian plasticity (architectural mismatch), epigenetics (data + timescale gaps), neuron-glia interaction.

The project's distinguishing potential is **integration above the channel layer** — what c302 and Nicoletti haven't done — combined with the existing strengths of CeNGEN integration, sensory cascades, and modulator layer. That positioning supports paper 3 + paper 4 + an evolving research-tool artifact, but is not on a path to be a production research tool comparable to industrial pharma platforms.

Recommendation: commit to the partial feasibility, treat the strategic reframing as "multi-year mechanistic-exploration tool with application-specific paper milestones," and avoid setting expectations toward the production research tool that the data ecosystem doesn't yet support.

