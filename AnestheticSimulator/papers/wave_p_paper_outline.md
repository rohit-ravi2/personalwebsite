# Wave P paper outline

**Status:** SCAFFOLDED. Outline only. Paper drafting begins after Gate H.1 evaluation in month 6.

---

## Working title

**"Network-level pharmacology of *C. elegans* anesthesia: predicted multi-target binding profile reproduces immobilization EC50 and mutant phenotypes"**

(Title is provisional. If Gate C.1 falsifies the multi-target framing, the paper is reframed as a multi-target falsification study with title like: "Predicted multi-target anesthetic binding profile fails to reproduce *C. elegans* immobilization phenotype: implications for the multi-target hypothesis at network scale.")

---

## Target venues

In preference order:

1. **Cell Systems** — strong fit for systems biology + computational pharmacology.
2. **Neuron** — broader neuroscience audience; competitive.
3. **eLife** — open access, methodologically rigorous.
4. **PLOS Computational Biology** — strong methodological-focus venue.

**Backup:** Network Neuroscience, *Brain Communications*, or a focused workshop track at NeurIPS LMRL / ICLR GRL.

---

## Authorship plan

- Rohit Ravi — first author.
- (TBD if collaborators contribute substantively, e.g., wet-lab citation cross-checks).

---

## Falsifiable claims

The paper makes 5 falsifiable claims, each tied to a Wave P phase / gate:

1. **Predicted occupancy at clinical halothane EC50 shows ≥ 5 targets with > 10% occupancy** (Gate C.1, Phase C). Rests on AlphaFold structures + Vina/DiffDock/GNINA docking + Hill equation + membrane-partition adjustment.

2. **Simulated WT halothane immobilization EC50 matches Crowder 1996 within 2×** (Phase G, anchor 1). Rests on Wave 2 channels + Phase D kinetic shifts + Phase E Markov synapses + Phase F metabolic layer + connectome.

3. **Per-target lesion analysis fails to reproduce the WT effect — multi-target framing supported** (Gate G.1.5). Rests on the lesion sub-grid runs.

4. **Simulated *gas-1* hypersensitivity matches Morgan & Sedensky 1995 within 50%** (anchor 3). Rests on metabolic layer (Phase F) + Complex I occupancy (Phase C).

5. **Simulated *unc-79*, *twk-18*, *unc-13* mutant phenotypes match published data within 50%** (anchors 4, 6, 7).

If any of these is empirically falsified, the paper's claims contract or the paper reframes.

---

## Section outline

### Abstract (~150 words)

Summary of the multi-target framing, the simulator architecture, the Phase H validation result, and the claim set. Last sentence states the headline: "X of 8 anchor predictions match published wet-lab data within 2× tolerance."

### 1. Introduction (~1500 words)

- General anesthesia is a long-standing puzzle: clinically reproducible, mechanistically diffuse.
- Lipid hypothesis (Meyer-Overton) gave way to receptor-binding hypotheses.
- Mammalian work supports a multi-target picture: GABA-A potentiation, K2P potentiation, NCA / NALCN block, SNARE-machinery effects, Complex I inhibition. No single target accounts for clinical immobilization.
- *C. elegans* is a well-characterized model: Crowder 1996, Sedensky 1992, Morgan & Sedensky 1995, van Swinderen 1999/2004, Sedensky 2001, Boddington 2017. Quantitative EC50 + mutant phenotypes published.
- The multi-target hypothesis is theoretically attractive but not directly tested: existing studies isolate single targets; the *parallel sum* across targets has not been computed.
- This paper builds a digital pharmacology platform: structural priors → docking → occupancy at clinical concentrations → kinetic shifts → network simulation → predicted phenotypes vs published data.
- Falsifiable claims listed.
- Roadmap.

### 2. Methods (~3000 words)

Subsections 2.1 through 2.7 mirror the Wave P phases A through H:

#### 2.1 Target panel selection
- Tier-1 panel (25 targets): Cys-loop, K2P, NCA complex, SNARE machinery, Complex I.
- Selection criteria: direct *C. elegans* anesthetic-mechanism literature, or high-confidence mammalian homolog.

#### 2.2 Structural priors
- AlphaFold-Multimer / RoseTTAFold-AllAtom / AlphaFold DB pulls.
- Cross-validation against PDB experimental homologs.
- pLDDT and TM-score thresholds (Gate A.1).

#### 2.3 Binding pose prediction
- AutoDock Vina + DiffDock + GNINA cascade (canonical, FEP-free).
- fpocket cavity detection.
- Photolabel cross-validation.
- GNINA top-10 cross-method-agreement (Gate B.1.4).
- FEP is documented as DEFERRED / SPECULATIVE for absolute-affinity calibration (see Methods appendix).

#### 2.4 Occupancy at clinical concentrations
- Hill equation; n=1 default; sensitivity at n=2 for pentamers.
- Membrane partition adjustment for membrane-embedded targets.
- **Gate C.1 evaluation (load-bearing).**

#### 2.5 Kinetic shift translation
- Literature-direct shifts where available.
- OpenMM MD on missing-data targets (TWK-18, NCA-1, AVR-14, UNC-49, GAS-1).
- Mammalian-control MD calibration (Gate D.1.2).

#### 2.6 Network simulation
- Wave 2 channels (Brian2; from Nicoletti 2024 translations).
- Markov synapse module (Gillespie SSA; Ca-cooperative SNARE assembly).
- Metabolic layer (ATP[t] dynamics; K-ATP coupling).
- Connectome from Witvliet 2020 + Cook 2019.
- 2,400-run grid + 40-run lesion sub-grid.

#### 2.7 Empirical validation
- 8 anchor predictions; pass criterion ≥ 4/8 within tolerance.
- Per-anchor failure-mode mapping.

### 3. Results (~3000 words)

#### 3.1 Structural priors
- Coverage table; Gate A.1 result.

#### 3.2 Binding poses
- Photolabel match rate; cross-method agreement; GNINA top-10 cross-method-agreement.

#### 3.3 Occupancy matrix at clinical concentrations
- Heatmap visualization.
- Number of targets exceeding 10% across (anesthetic, dose) cells.
- **Gate C.1 verdict.**

#### 3.4 Kinetic shift table
- Per-target shift form, magnitude, source.
- Mammalian-control MD vs literature.

#### 3.5 Network simulation outputs
- WT halothane / isoflurane / propofol EC50.
- Mutant comparisons (gas-1, unc-79, unc-13, twk-18).
- **Per-target lesion analysis** (the load-bearing test).

#### 3.6 Anchor table
- 8 anchors: published value, simulator value, ratio, pass/fail.
- Headline: "X of 8 anchors pass."

#### 3.7 Failure-mode mapping
- For each failing anchor, the diagnosed upstream-phase issue.

### 4. Discussion (~2000 words)

#### 4.1 Multi-target framing supported / falsified
- Lesion test result.
- What this means for the anesthetic-mechanism literature.

#### 4.2 Comparison to single-target frameworks
- Why single-target frameworks fall short.
- What the multi-target framing predicts that single-target cannot.

#### 4.3 Limitations
- 25-target panel does not cover all anesthetic targets (Tier 2 deferred).
- Per-target Kd uncertainty bracket factor 3.
- *C. elegans*-specific targets (UNC-79, UNC-80) had limited structural confidence.
- Metabolic layer uses phenomenological K-ATP coupling pending Wave 2 K-ATP translation.

#### 4.4 Implications for human anesthesia
- Worm and human share most target classes.
- Quantitative differences in EC50 reflect Kp / pocket / kinetics.
- The framework generalizes; human-specific work is an extension.

#### 4.5 Future directions
- Tier 2 expansion.
- Phase I (inverse design) — empirical occupancy from calcium recordings.
- Phase J (network signatures) — cross-validation against mammalian EEG/fMRI.

### 5. Acknowledgments

Wave 2 collaboration; notebook pipeline source; Wave 2 channel translations (Nicoletti 2024 derivative).

### 6. References

Comprehensive reference list. Every citation has PMID/DOI verified per Wave P's citation hygiene declaration.

---

## Figures (planned)

- **Figure 1.** Conceptual overview: multi-target binding → kinetic shifts → network simulation → predicted phenotype vs published.
- **Figure 2.** Tier-1 target panel + structural-prior coverage.
- **Figure 3.** Occupancy heatmap at clinical concentrations (Gate C.1 visualization).
- **Figure 4.** Per-target kinetic shift table; MD vs literature controls.
- **Figure 5.** Network-level EC50 dose-response curves; WT vs mutants.
- **Figure 6.** Per-target lesion comparison (load-bearing).
- **Figure 7.** 8-anchor table; pass/fail matrix.
- **(Supplementary)** Per-anchor failure-mode mapping; per-target sensitivity analysis.

---

## Submission checklist (for month 6)

- [ ] All 8 anchor predictions evaluated.
- [ ] Lesion test result documented.
- [ ] Failure-mode mapping for any failed anchor.
- [ ] Citation hygiene: 100% PMID/DOI verified.
- [ ] Code archive: Wave P repo + Wave 2 pinned commit + notebook pipeline pinned commit.
- [ ] Data archive: Phase A-G NPZ outputs (subset) deposited in Zenodo.
- [ ] License notes: AlphaFold-Multimer + RoseTTAFold-AllAtom non-commercial-use noted in supplementary.
- [ ] Negative-result framing if Gate C.1 or Gate G.1.5 falsified.

---

## Backup paper trajectories

If Gate C.1 falsifies the multi-target framing, Wave P produces a different paper:

**"Multi-target anesthetic binding does not reproduce *C. elegans* immobilization: a network-level falsification."**

Target: PLOS Computational Biology, eLife, or NeurIPS LMRL workshop.

This is **still publishable** — a rigorous negative result is valuable in a field where the multi-target hypothesis is widely cited but rarely directly tested.

If Gate H produces 2-3/8 anchors (partial fail), the paper becomes a methods + partial-validation paper:

**"A digital pharmacology platform for *C. elegans* anesthesia: methods, scope, and partial validation against published phenotypes."**

The methods (Phases A-G) are the contribution; the validation is presented honestly with successes and failures both visible.
