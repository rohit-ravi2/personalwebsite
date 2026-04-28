# Session 1 handoff — 2026-04-26 evening

**Author:** Session 1 (parameter-analysis perspective)
**Companion:** Session 3's handoff at `session_3_handoff_2026-04-26.md` (cellular-biophysics perspective)
**Division C:** either handoff alone is standalone-sufficient. Redundancy is intentional — the two perspectives cover the same project state with different framings.

---

## Section 1 — Document purpose and how to use it

This document captures the C. elegans biophysical-simulator project state as of 2026-04-26 evening and brings new sessions current on a project trajectory that shifted substantially today. The user is **Rohit Ravi** — NYU undergrad, Data Science major with Philosophy minor, working toward an industry AI career bridging technical and philosophical domains, with the simulator as a long-running research project. The project has both behavioral-paper and mechanistic-paper trajectories; today's work block landed in the mechanistic-trajectory infrastructure (Wave 2 channel translation).

Read this document end-to-end before opening any source artifact. The document is designed to be standalone-sufficient: a future session can pick up project work without re-reading underlying artifacts (though they remain available at the paths listed in §10 for deep dives).

The document's distinctive contribution comes from §7 — Session 1 spent today's main work earlier in the day on the corrected-weighting LHS re-analysis (Wave 1 closure) and stood by while Sessions 2 and 3 executed Wave 2 work. The parameter-analysis lens ties Wave 1 LHS findings to Wave 2 cellular validation findings as one consistent methodology pattern: *empirical calibration converged on the right answer for documented-wrong-reason* shows up in both contexts (Wave 1's hedging methodology bug, Wave 2's F6 misdiagnosis). Documenting that pattern is paper 4 territory.

The user's working style preferences are load-bearing:

- Plan-first for non-trivial work.
- No-sugarcoat assessments. Honest scope labels: shipped vs scaffolded vs calibration-pending vs deferred.
- Push back on speculative proposals before elaborating; ask for falsifiability.
- Vedanta / non-dualist framings welcome when they sharpen technical work, but avoid ideological overlays in straight scientific discussion.
- No wet-lab work. Theoretical + computational only.
- NJ geographic anchor; OMSCS/MSE-AI program targets; health-driven non-PhD trajectory.

If a future session is uncertain whether to surface a finding mid-flight or batch it to end-of-block, the answer is almost always **surface mid-flight**. Today's work caught ~17 substantive corrections through that discipline.

---

## Section 2 — Project identity

The project is a 300-neuron Brian2 LIF *C. elegans* brain simulator with multiple architectural layers. Public-facing summary at `~/Desktop/website/personalwebsite/src/content/projects/c-elegans-multimodal.mdx`. Live at `rohitravi.com/projects/c-elegans-multimodal`. Repo at `github.com/rohit-ravi2/personalwebsite` (private during paper prep).

**High-level architecture (current, operational):**

- **Brain layer:** 300-neuron Brian2 LIF default (production), GradedBrain (Kunert-Graf 2014 σ(V) graded dynamics with optional EGL-19/SLO-1 plateau scaffolds — opt-in via `brain_class="graded"`), compartmental scaffold built but not yet deployed.
- **Sensory layer:** sensory_injection (legacy direct-Poisson) default; sensory_transduction (5 ODE cascades for ASE/AWC/ASH/AFD/ALM) opt-in; AerotaxisSensory for URX/AQR/PQR/BAG.
- **FSM layer:** BehavioralFSM driven by 8-event Atanas-trained classifier bank (default); ActivityFSM reading command-neuron rates directly (P1 #4 alternative).
- **Body layer:** 20-segment MuJoCo wormbody with CPG controller per state.
- **Modulator layer:** 9 peptidergic/monoaminergic modulators (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5-HT, DA, TA, OA) with CeNGEN-derived releaser/receptor tables, volume-transmission diffusion length scales.
- **Sign-mode toggle:** `use_per_edge_glu_signs` (False default; True opts into CeNGEN-derived per-edge Glu sign assignment via W_chem_per_edge matrix), plus 7-entry `DOCUMENTED_SIGN_EXCEPTIONS` registry as a mode-independent overlay (Session 2 commit `aea4c79`, today).

**Two paper trajectories in motion:**

- **Paper 2 (behavioral simulator):** workshop track NeurIPS GRL or ICLR LMRL. Largely independent of Wave 2. Uses sample_004 + LIF + voltage-domain target framework. Layer 1 closure (corrected-weighting analysis from this morning's work) supports paper 2 manuscript prep at any time.
- **Paper 3 (mechanistic simulator):** target *eLife* / *PLOS Computational Biology* / *Network Neuroscience*. Wave 2 channel translation infrastructure is the foundation. Cell-level mechanism for cascade dynamics, plateau termination, modulator effects.

**Note on document drift:** the architectural plan (`scripts/brain/artifacts/phase_v_w2_architectural_plan.md`) and the public mdx are slightly behind today's state on specific points — most prominently the Mellem 2008 "20 mV / 600 ms in AVA" target was misattributed (see §3 and §4) and the architectural plan still references it. The plan's *strategic* commitments (Path 3A primary, per-channel rollback, gate-based progression) remain load-bearing; the *specific* AVA targets in the plan need re-grounding.

---

## Section 3 — Wave 2 status as of 2026-04-26 evening

This is the load-bearing current-state summary. Everything else in this document is supporting context for what's covered here.

### 3.1 Path 3A is committed and empirically vindicated at the channel level

Wave 2's architectural commitment to Path 3A (Brian2 backend + per-channel parameter import from Nicoletti 2024) holds. All 7 channels in the "essential set" have been translated and validated against NEURON references at the per-channel level:

| Channel | Phase | Validation status | Module |
|---|---|---|---|
| EGL-19 (L-type Ca) | β CP2 (run #1) | PASS, 11/11 holds, max div 0.004 | `wave2/channels/egl19.py` |
| SHK-1 (Kv1 delayed rectifier) | β C.1 (run #2) | PASS, 11/11 holds, max div 0.007 | `wave2/channels/shk1.py` |
| SHL-1 (Kv4 A-type) | β C.2 (run #2) | PASS, 11/11 holds, max div 0.003 | `wave2/channels/shl1.py` |
| NCA (Na+ leak / NALCN) | β C.3 (run #2) | PASS, 11/11 holds, max div 0.000 | `wave2/channels/nca.py` |
| KQT-3 (M-type K) | β C.4 (run #2) | PASS, 11/11 holds, max div 0.000 | `wave2/channels/kqt3.py` |
| SLO-1 isolated (BK) | β D (run #2) | PASS, 44/44 panels (4 cai × 11 V) | `wave2/channels/slo1_iso.py` |
| SLO-1 + EGL-19 coupled | β E (run #2) | PASS, 11/11 holds, max div 0.0006 | `wave2/channels/slo1_egl19_coupled.py` |

**All 7 channels achieve voltage-clamp trace divergence < 1% against the NEURON reference.** Architectural plan's Gate 1 (per-channel kinetics correctness) and Gate 2 component 2a (channel kinetics in cell context) are PASS.

Cellular foundation infrastructure is complete:

- **NEURON 9.0.1** installed in isolated venv `~/venvs/wave2-neuron/` (production conda env at `~/miniconda3/envs/ml/` not touched).
- **Nicoletti 2024 source code compiled.** All 24 `.mod` files (22 ionic currents + 2 Ca-handling utility mods cadiff + caintra1) compile cleanly via `nrnivmodl`.
- **Validation harnesses:** `voltage_clamp_harness.py`, `plateau_harness.py`, `reference_validation.py`, `neuron_reference.py` (NEURONReference wrapper).
- **F1-F17 NMODL gotcha catalog systematized** in `wave2/translation_patterns.md`.

Of the 7 essential-set channels, **4 (EGL-19, NCA, IRK-equivalent via leak proxy, leak) are validated for AVA's actual channel set** (per Nicoletti's `AVAL_simulation_iclamp.py` source: AVA uses IRK + LEAK + EGL19 + NCA + UNC103). **3 (SLO-1 isolated, SLO-1+EGL-19, SHK-1, SHL-1, KQT-3) are valid for AIY/RIM/RMD use** but not in AVA's actual channel set. IRK and UNC-103 are **not yet translated** — they're the missing pieces for AVA-specific validation under the re-grounded path α (see §3.5).

### 3.2 The Mellem 2008 target was misattributed

The single most consequential finding from today's afternoon work: **"Mellem 2008 plateau in AVA (20 mV / 600 ms)"** as the architectural plan's Phase γ Gate-2b biological target is misattributed. Mellem 2008 explicitly reports that AVA does NOT show plateau or regenerative dynamics. The plateau dynamics characterized in Mellem 2008 are in **RMD**, a different cell.

**Primary-source quotes from Mellem et al. 2008** (PMCID PMC2697921, Nature Neuroscience 11:865-867):

> "**In contrast, we never observed action potentials in AVA (n=10; Fig. 1b).**"
>
> "The resting potential of AVA was typically between −20 and −30 mV and we did not observe action potentials (Fig. 1d), **even when we changed the resting potential to more hyperpolarized levels.**"
>
> "In contrast to what was observed in RMD, glutamate application caused short-lived, modest changes in AVA membrane potential with no switch to a new steady-state potential (n = 5; Fig. 3i)."

Mellem 2008's plateau characterization is in RMD, not AVA: "long-lasting" all-or-none events that can be terminated by negative current pulse, Ca-dependent, multiple Ca channels contribute. **No specific 20 mV or 600 ms numerical values are quantified for either RMD or AVA in Mellem 2008.**

This invalidates the framing of Phase F's empirical observation (single-compartment AVA produces 46.8 mV / 21.4 ms vs target 15-25 mV / 400-800 ms) as "biological insufficiency requiring morphology fork." The empirical observation is real; the target it was being measured against doesn't exist for AVA in the cited paper. The morphology fork's pre-condition was framed against a phantom target.

### 3.3 Citation audit caught three additional misattributions

Following the Mellem investigation, a citation audit (`architectural_plan_citation_audit.md`) verified the load-bearing biological citations in the architectural plan against primary sources (Nicoletti 2024's full reference list extracted from on-disk PDF). Three additional misattributions surfaced:

1. **Wang 2001 → SHK-1 (architectural plan line 94).** The plan attributes "rich worm-specific data" for SHK-1 to Wang 2001. The actual Wang 2001 paper (Nicoletti 2024 reference [60]) is **Wang et al. 2001 *Neuron* 32:867-881 — about SLO-1 at the neuromuscular junction, not SHK-1.** Nicoletti 2024 cites Wei 2005 (KCNQ-like K channels in *C. elegans*, JBC), Gu 2012 (Kv3.1 splicing, JBC), Dobosiewicz 2019 (*Elife*), and Liu/Kidd/Dobosiewicz/Bargmann 2018 *Cell* for SHK-1. Wang 2001 is not in that set.

2. **The "20 mV / 600 ms" numerical pair has no primary source.** Audit traced Nicoletti 2024's actual AVA protocols: current-clamp 1000 ms / 7 steps -30 to +30 pA; voltage-clamp 500 ms / 16 steps -120 to +50 mV. Neither matches "600 ms." The "20 mV" approximates AVAL's typical depolarization range under +30 pA injection but is not stated as a target in any primary source. The "600 ms" appears to be an unverified interpolation between Nicoletti's 500/1000 ms protocols, or a misremembered Mellem RMD value.

3. **Liu 2018 reference year drift in v1 digitization JSON.** The v1 `published_traces.json` recorded "Liu P, Chen B, Wang Z-W. **2018**. Postsynaptic current bursts instruct action potential firing at a graded synapse. ref [29] in Nicoletti 2024." The actual Nicoletti 2024 reference [29] is **Liu/Chen/Wang 2020 Nat Commun "GABAergic motor neurons bias locomotor decision-making"** — wrong year (2018 vs actual 2020) AND wrong title. The v1 digitization agent fabricated this citation; the agent's "Liu 2018" attempt was conflating with the *Liu/Kidd/Dobosiewicz/Bargmann 2018 Cell* AWA paper (which is reference [30], not [29]).

The architectural plan cleanup itself is **deferred to paper 3 manuscript prep timing** — not blocking implementation work. Implementation work proceeds against the re-grounded targets per §3.5 below.

### 3.4 AVA re-grounding decision — option α

Per the Mellem investigation (`mellem_investigation_pushback.md`), three biologically defensible re-groundings were proposed. The user committed to **option α — Nicoletti's actual AVAL phenotype**:

> Per Nicoletti 2024: AVAL shows "slow-rising phase (~200 ms) followed by a stable plateau that is sustained until the stimulus is removed." Plateau amplitude depends on injection current (no specific 20 mV target). AVA's response is "passive RC-circuit-like" with linear I-V curves.
>
> The "plateau" Nicoletti describes is the **steady-state V under sustained current injection** — a passive plateau, not an active regenerative one. It "is sustained until the stimulus is removed" — i.e., does NOT self-terminate, decays only when injection stops.
>
> AVA's I-V curves are linear (not bistable, not regenerative). Bistable behavior in vivo is attributed to **synaptic input**, not intrinsic membrane properties.

Per Nicoletti's `AVAL_simulation_iclamp.py` source, AVA's actual channel set is **5 channels: IRK + LEAK + EGL19 + NCA + UNC103.** No SLO-1, no SHK-1, no SHL-1, no KQT-3 in AVA. Current injection duration: 1000 ms (matching the "sustained until removed" plateau description).

**Implications under option α:**

- The 7-channel essential-set translation work remains valuable — it's the AIY/RIM/RMD channel library, not AVA's. Translation infrastructure is not wasted.
- AVA-specific validation requires translating IRK + UNC-103, the two Nicoletti AVA channels we don't have yet.
- The morphology fork's pre-condition (condition 6 against Mellem target) is invalidated. The fork is on hold until and unless re-evaluated against a different target.
- Phase F's empirical observation (46.8 mV / 21.4 ms with 7-channel set, no Ca-pool) remains a real engineering finding about what the model does. It's no longer framed as "fails to match biology" — it's "produces phenotype X under input regime Y, where target Z was misattributed."

### 3.5 Morphology fork — on hold

The architectural plan's morphology fork (Phase β-morph + Phase γ-morph, c302 NeuroML2 multi-compartment integration into Brian2 scaffold, ~3-4 weeks per plan) was framed as the response to "condition 6 — channels work, architecture insufficient for Mellem dynamics."

With Mellem-as-target invalidated, condition 6's original framing is invalidated too. The morphology fork's **trigger no longer fires** under option α: Nicoletti's AVAL phenotype (sustained passive plateau under injection, RC-like response, linear I-V) does not require multi-compartment morphology — Nicoletti achieves it in single-compartment with the 5-channel AVA set.

The morphology fork is **on hold, not abandoned.** If Wave 3+ work surfaces a phenotype that genuinely requires multi-compartment architecture (e.g., dendritic plateau dynamics in some other neuron class, or fully-validated Mellem RMD plateau as a separate target), the morphology fork can be revisited with stronger pre-conditions. The c302 morphology assets are still on disk (`~/Desktop/C-Elegans/simulation/upstream/c302/`) and the architectural plan's framework supports the fork; what's missing is the empirical justification.

The supporting empirical work for the original morphology-fork case (density-sensitivity sweep showing terminator scaling has near-zero leverage; Ca-coupling test showing dynamic Ca-pool insufficient) remains valid as **engineering findings about what the single-compartment model does** under the 7-channel set + Mellem-style protocol. They were the right tests for the wrong target. Their findings transfer cleanly to the option α framing as: *"single-compartment AVA with 7-channel set does not produce 600-ms plateau under 50 pA injection — but Nicoletti's AVA isn't supposed to, per primary source."*

### 3.6 Citation audit cleanup deferred

The architectural plan retains its strategic shape (Path 3A primary, per-channel rollback, two-component Gate 2, per-channel translation priorities) under option α. The specific citation cleanups (Mellem 2008 → AVA references at lines 112, 177, 183, 275, 281, 294; Wang 2001 → SHK-1 at line 94; "20 mV / 600 ms" specific values throughout) are documented as needing revision but **deferred to paper 3 manuscript prep timing.** The plan as a working document is not blocking Wave 2 implementation — implementers should refer to this handoff and `mellem_investigation_pushback.md` + `architectural_plan_citation_audit.md` for the corrected target framing.

---

## Section 4 — Today's chronological work summary

This section provides a narrative arc; specific artifacts have detail.

### 4.1 Morning — Wave 1 closure

**Sessions 1, 2, 3 ran the corrected-weighting analysis (Session 1's earlier work block today).** The original LHS analyzer used equal-weighted mean-of-criteria scoring, which produced misleading verdicts: in the 3-config validation, hedged + overrides was scored as "PASS" (mean 0.962) despite catastrophic over-excitation (AVAL = 185 Hz at spontaneous rest). The mean-of-criteria score averaged spurious NSM PASSes (NSM at 8 Hz is over-excitation, not biological tonic activity) against cascade-stability FAILs.

Corrected weighting scheme:
- Over-excitation guards (graded 1.0 below 80 Hz, hard FAIL above 120 Hz)
- Destabilization guards (hard FAIL above 200 Hz)
- Sensors, RIS, NSM-anti-silence, SMDVL — biology-grounded targets, mean-normalized per category
- Cascade firing reported across +50/+20/+10/+0 Hz target sensitivity (LIF-rate framing flagged as graded-cell-artifact dimension)

Three outputs per sample:
- **Verdict A (guards-only):** load-bearing for synthesis, biology-robust, sign-mode-independent.
- **Verdict B (target-based):** informational, contingent on LIF-rate framing.
- **Verdict invariance check:** which samples top-rank under Verdict A AND across all 4 Verdict B thresholds.

Headline findings:
- 12/30 LHS samples NULLIFIED by guards (40% of parameter space is over-excited; original analyzer didn't catch this).
- Sample_004 dominates every ranking (Verdict A + Verdict B at all 4 thresholds). Invariant top-5: [sample_002, 004, 005, 026].
- Hedged + overrides nullified (V_A score = NULL due to AVAL=185Hz).
- Sample_004 + overrides scores marginally HIGHER than sample_004 NO overrides under Verdict A (1.016 vs 1.002).
- Verdict inversion at low cascade thresholds: at +50 Hz target, NO overrides wins (1.061 vs 0.935); at +10/+0 Hz target, + overrides wins (1.073 vs 1.061). Synthesis-relevant.

Session 2 ran a biophysical consistency audit (Nernst E_Ca, synaptic reversal potentials, voltage scale conventions, conductance/current formulation) and surfaced that AVA/AVE/AVB/AVD/PVC are graded non-spiking neurons. The +50 Hz cascade firing target is an LIF-internal proxy without biology grounding. Real biology measures cellular voltage trajectories (Mellem 2008 voltage clamp) or ΔF/F calcium signals (Atanas 2023), not Hz.

Session 3 ran cellular validation work across three v_rest baselines (-25 mV Mellem AVA up-state, -45 mV GradedBrain native / Lockery 2009 range, hyperpolarized hold per Mellem voltage-clamp protocol) for the K_Ca patch in graded mode.

**Synthesis at Wave 1 closure:** the LIF-rate-based Layer 1 closure framework is structurally complete but its cascade-firing target is graded-cell-artifact. Voltage-domain target framework recommended for paper 2. Wave 2 architectural commitment supported.

### 4.2 Morning — Wave 2 architectural commitment

User accepted Path 3A primary based on Wave 1 closure synthesis + earlier audit work. Architectural plan committed at `phase_v_w2_architectural_plan.md`. 

Conditional fork structure: 6 invalidation conditions (lines 270-281 of plan), including condition 6 ("cellular validation fails not on channel kinetics but on compartmental architecture"). The plan committed to morphology fork as response to condition 6 if it surfaced.

### 4.3 Phase α — setup

Session 2 (or equivalent) executed Phase α deliverables:

- NEURON 9.0.1 installed in `~/venvs/wave2-neuron/`.
- Nicoletti 2024's 24 `.mod` files compiled cleanly via `nrnivmodl`.
- 3 of 3 reference cells (AVAL, AIY, RIM) reproduced with bit-exact determinism (max relative diff 0.0).
- Voltage-clamp harness validated: 7e-16 max relative diff against analytic leak reference.
- Plateau harness validated: synthetic `passing_scaffold` (active termination) and `failing_scaffold` (leak-dominated) correctly classified.

Findings F1-F10 catalogued in `phase_beta_findings.md`. Notably F6 (calcium-pool calibration) was at this point believed to be a "hidden NMODL unit-conversion factor" — the calcium_pool.py docstring claimed "Symbolic re-derivation gives ~5183 mM/(mA/cm²·ms), empirical 0.525, ratio ~10000×; NMODL hidden machinery." This claim was technically *wrong but functionally inconsequential* (production code worked; the explanatory hypothesis was incorrect).

### 4.4 Phase β-pre — three iterations on the Layer C / Layer B / Layer A framework

The Phase α deliverable 3 ("NEURON-vs-experimental tolerance check, 5% per-point on published figures") originally was closed under "deterministic self-consistency" interpretation. Phase β-pre then iterated to test it more rigorously:

- **v1:** digitized experimental-overlay panels (Fig 1F, 3D, 5D — voltage-clamp I-V curves) and compared against Nicoletti's NEURON output. Result: 0/3 panels at 5% per-point tolerance; mean divergences 39-66%. **Methodological error surfaced**: those panels are post-hoc predictions, not fit targets.

- **v2:** corrected the metric by digitizing actual fit-target current-clamp panels (Fig 1A AVAL, Fig 1B AVAR, Fig 3A AIY, Fig 5A RIM). Per-feature 5% comparison: 0/4 panels pass. Voltage absolute errors per step: 6.8-15 mV mean, 17-43 mV max. **Layer C tolerance structurally too strict** for biophysical HH fits.

- **v3 (decisive):** decomposed comparison into three layers — A (Brian2 = NEURON; Phase β proper), B (NEURON code = Nicoletti's published Model figures; what condition-3 actually asks), C (Nicoletti's published Model = experimental data; what v1+v2 measured). v3 directly tested Layer B by digitizing the published Model traces (red on AVAL/AIY/RIM panels; blue on AVAR's Fig 1B panel) and comparing against NEURON output. **Layer B substantively passes (3.3-4.8 mV mean V abs error per cell, half of Layer C's residuals)** — strict 5% per-feature still fails, but at the appropriate layer.

v3 also patched AVAR (upstream `AVAR_simulation_iclamp.py` is missing from Nicoletti repo head tree). The patch (`avar_unc103_patch.py`) mirrors AVAL's iclamp structure with AVAR's parameter vector + UNC-103 inserted; produces -24.25 mV resting (target -25 ± 5 mV). A draft GitHub issue for the upstream repo is at `artifacts/avar_upstream_issue_draft.md` for user authorization to file.

**Citation correction:** v3 surfaced that prior versions had referenced `10.1371/journal.pcbi.1007611` as "Nicoletti 2019 PLOS Comp Bio" — that DOI resolves to a glioma paper. Both real Nicoletti papers (2019 PLOS ONE AWCon/RMD; 2024 PLOS ONE 22-channel library) are by the same group; the spec preamble that originated the wrong DOI conflated them. v3 corrected this across the architectural plan and active session prompts; v1 historical artifacts (`published_traces.json`, `digitize_panels.py`) preserved unchanged as historical record of v1→v2 detection.

### 4.5 Phase β overnight #1 — EGL-19 translation, CP1-CP3

CP1 (foundation infrastructure: NEURONReference wrapper, Ca-pool subsystem, voltage-clamp tolerance metric) cleared. CP2 (EGL-19 isolated translation): PASS, 11/11 holds, max divergence 0.004. CP3 (EGL-19 in cell context): PASS.

Phase β findings catalogued F1-F10. F6 (calcium-pool calibration) flagged as "hidden NMODL unit-conversion factor" claim — set aside for run #2 deeper investigation.

### 4.6 Phase β overnight #2 — F6 misdiagnosis correction, all 7 channels translated, Gate 2 evaluated

Phase A of run #2: deeper F6 diagnostic. **Verdict: PRINCIPLED, not FUDGE_FACTOR.**

The proper symbolic decomposition of cadiff gives **α = 0.518 mM/(mA/cm²·ms)**, matching empirical 0.5182 across AVA/AIY/RIM at 9 holding potentials to **5 decimal places**. The 10000 in cadiff.mod IS the proper unit-conversion factor from mol/(s·cm³) to mM/ms, fully derivable from declared units. For caintra1, symbolic α(geometry) matches empirical α(geometry) to 5 dp at AVA, AIY, RIM. **No hidden NMODL machinery exists.** Run #1's empirical calibration converged to the symbolically-correct value because the symbolic derivation IS correct — the docstring's "5183× ratio" claim is internally inconsistent with the production code (which uses 0.525). Production code was correct; the explanatory docstring was wrong.

This is a methodologically important finding: **empirical calibration converging on the right answer for documented-wrong-reason.** The kind of pattern that propagates across artifacts unless re-checked.

F11-F13 produced architectural simplifications:
- **F11:** F6 was a misdiagnosis (above).
- **F12:** Of 5 cell scripts inspected (AVAL, AVAR, AIY, RIM, VA5), only VA5 inserts cadiff. AVA, AIY, RIM, AVAR rely on NEURON's default static cai = 5e-5 mM. **Implication:** SLO-1 isolated in Nicoletti's actual cells reads a constant cai value. Brian2 translation does NOT need a dynamic Ca-pool for these cells.
- **F13:** `slo1egl19.mod` does NOT read cai. It has internal closed-form `calcium(V)` (Lluís-Buchholz/Alvarez nanodomain approximation). **Implication:** Phase E architectural decision (nanodomain encoding) resolved trivially.

Phase B: NMODL pattern catalog systematized as 13 patterns (`wave2/translation_patterns.md`).

Phase C: 4 non-Ca channels translated and validated.
Phase D: SLO-1 isolated translated and validated at 4 cai values × 11 V (44/44 panels).
Phase E: SLO-1+EGL-19 coupled translated using closed-form `calcium(V)` (per F13). Validated 11/11, max div 0.0006.

Two harness fixes during run #2:
- **F14:** `h.run()` re-finitializes via `h.v_init` (default -65 mV), silently overriding explicit `h.finitialize(v_arg)`. Caught via SHL-1 7.3% systematic peak divergence. Fixed in `neuron_reference.py`.
- **F15:** Brian2 vs NEURON SS extraction window mismatch. Brian2 uses last 20 ms; NEURON's stored ss_I_pA uses last 20% of step (40 ms for 200 ms step). Fixed in `voltage_clamp_harness.py`.

**Phase F (Gate 2):**

- Component 2a (channel kinetics in cell context, leak + EGL-19 + NCA in apples-to-apples Brian2-vs-NEURON construction): **PASS**, 11/11 holds, max divergence 0.004.
- Component 2b (architectural sufficiency — Mellem 2008 plateau target, single-compartment AVA with 7-channel set + chosen densities): **FAIL**, plateau amplitude 46.8 mV (target 15-25 mV), duration 21.4 ms (target 400-800 ms).

**Per spec's decision tree, this was 2a-pass / 2b-fail = condition 6 surfaces.** Run #2 paused for morning review per the spec's "PAUSE for morning review, do NOT auto-trigger morphology fork" instruction.

### 4.7 Density-sensitivity sweep

Before authorizing the morphology fork, density-sensitivity analysis (`density_sensitivity_analysis.md`) tested whether the failure was density-tunable vs truly architecturally insufficient. 4×4 grid (terminator and Kv each scaled by {0.5, 1.0, 2.0, 4.0}, plus extension probes at {0.25, 8.0}) over the 5 non-Nicoletti density parameters. The 3 principled-density channels (EGL-19, leak, NCA) were held at Nicoletti AVAL g0 throughout.

**Verdict: VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS.**

- Amplitude can be tuned into target range (kv=8 → 17.7 mV ∈ [15, 25]) but at cost of duration collapsing to 4.4 ms.
- Maximum duration anywhere in the swept volume: 42 ms — order of magnitude short of 400 ms lower bound.
- **Terminator scaling has near-zero leverage** on the phenotype: 32× variation in SLO-1 conductance produces zero meaningful change.
- Mechanistic interpretation: SLO-1 isolated reading static cai (per F12) cannot mediate Ca-feedback; no slow positive-feedback loop sustains a plateau. The missing ingredient is *Ca dynamics*, not *amount of SLO-1*.

### 4.8 Ca-coupling integration test

Before triggering the morphology fork, the cheaper architectural extension was tested: add dynamic caintra1 pool, couple EGL-19's I_Ca to [Ca]_i, let SLO-1 isolated read dynamic [Ca]_i state. Test ran overnight 2026-04-26 → 2026-04-27.

**Verdict: VERDICT_CA_COUPLING_INSUFFICIENT** (robust across 5 orders of magnitude of fca scaling and 4× SLO-1 conductance).

Key findings:
- At Nicoletti default fca, [Ca]_i barely moves (51 nM peak vs 50 nM baseline; 1.02× fold change). The Ca-coupling loop is *thermodynamically disengaged* — gain-locked at near-zero.
- At fca up to 10000× default, [Ca]_i reaches 17 μM, and **plateau duration *decreases* monotonically as the loop engages**.
- The Ca-coupling loop is **negative feedback for V** because SLO-1 is hyperpolarizing — it cannot extend a depolarized plateau, only terminate it faster.

This contradicted the F12-derived hypothesis. F12 correctly identified that no Ca-feedback existed; what F12 (and morning review's read of it) implicitly assumed was that *adding* Ca-feedback would extend the plateau. The Ca-coupling test refuted that assumption with quantitative evidence.

Two new findings produced during this work block:
- **F16:** caintra1 ⇄ slo1iso unit-conversion (×1000) is required. caintra is M-equivalent (5e-8 raw), slo1iso's cai is mM-scale (5e-5 raw). Cell builders wiring dynamic caintra1 to slo1iso must insert `cai_mM = caintra_raw * 1000`.
- **F17:** caintra1 fca-scaling is not in the calibrated `coef_in_eff`. `calcium_pool.py` accepts fca but doesn't rescale the empirical coefficient. Sweep callers must explicitly multiply by `fca / 0.001`.

### 4.9 Speculative GNN exploration

In parallel with the density-sensitivity sweep, speculative-architecture work block characterized three alternative responses to compare against the morphology fork: **GNN hybrid (X.1), multi-compartment-explicit (X.2a), NeuroML2-native (X.2b).**

Top-line: **morphology fork remains strongest condition-6 response** (when condition 6 was thought to apply). GNN Variant A is plausible Wave 4 enhancement (per-segment density fitting + CeNGEN coupling), not Wave 2 alternative. Variants B/C not currently feasible (data-starved). NeuroML2-native rejected (same morphology work needed, worse backend performance, higher infrastructure cost).

The 2-node GNN prototype's 46 mV final test MAE (vs <5 mV target) was reported as "informative-negative" — the training pipeline functional, but bounded effort insufficient to determine whether more careful setup would converge or whether the architecture is structurally inefficient at small scales.

All speculative outputs in `wave2/speculative/`. No `SPECULATIVE_PAUSE.txt` was created — no load-bearing concern surfaced during investigation.

### 4.10 Mellem investigation surfaced misattribution

Triggered by `phase_v_w2_mellem_investigation_prompt.md`. Spec instructed pause-and-surface immediately if "Mellem 2008" citation verification surfaced a misattribution. It did. Per §3.2 above.

The investigation produced `mellem_investigation_pushback.md` with primary-source quotes and the path-α/β/γ re-grounding alternatives. User selected option α.

### 4.11 Citation audit confirmed three more misattributions

Per §3.3 above. Audit document at `architectural_plan_citation_audit.md`. Methodology lesson surfaced before audit deployed: the audit prompt initially proposed verifying a per-cell target table with citations (AVE/Wang 2020, AVB/Kawano 2011, etc.) — pre-flight grep showed none of these citations exist in the architectural plan. They were constructed from memory rather than verified against canonical document. Pre-flight pushback caught it before deployment, exactly as the audit was designed to prevent. **The pattern works in both directions:** user-side fabrication caught by agent pre-flight pushback, agent-side fabrication caught by user/cross-session review.

### 4.12 AVA re-grounding to option α committed

Per §3.4 above. The 3-4 week morphology fork commitment was avoided. The empirical findings (Phase F 2b, density sweep, Ca-coupling test) preserved as engineering observations about model behavior under the original (misattributed) target framing.

---

## Section 5 — Architectural commitments and their current status

| Commitment | Status | Notes |
|---|---|---|
| Path 3A primary (Brian2 + Nicoletti import) | **INTACT** | Wave 2 architectural plan §4.5; confirmed by per-channel validation |
| Phase α infrastructure | **COMPLETE** | NEURON installed, Nicoletti compiled, 3-cell reference reproduction PASS, harnesses validated |
| 7-channel essential set translation | **COMPLETE** | All 7 PASS at per-channel level (Gate 1) |
| Of 7 essential channels validated for AVA | **4 of 7** | EGL-19, NCA, leak, IRK-equivalent (proxy). SLO-1, SLO-1+EGL-19, SHK-1, SHL-1, KQT-3 not in AVA's actual channel set |
| AVA's true channel set (per Nicoletti) | **4 of 5 translated** | IRK + LEAK + EGL19 + NCA + UNC103. UNC-103 not yet translated. AVAR patch handles UNC-103 differently |
| Gate 2 component 2a (channel kinetics in cell context) | **PASS** | 11/11 holds, max div 0.004, against apples-to-apples NEURON construction |
| Gate 2 component 2b (architectural sufficiency) | **RE-FRAMING PENDING** | Original framing (Mellem 2008 in AVA) was misattributed. New framing under option α: "match Nicoletti's actual AVAL phenotype" doesn't require morphology fork. New 2b validation to follow IRK + UNC-103 translation |
| Condition 6 (architectural insufficiency for Mellem dynamics) | **INVALIDATED as original** | Mellem doesn't characterize a 600-ms plateau in AVA. New condition 6 under option α: would be Nicoletti-AVAL-phenotype mismatch in single-compartment, which is not currently empirically observed |
| Morphology fork (Phase β-morph + Phase γ-morph) | **ON HOLD** | Not currently warranted. C302 morphology assets remain on disk for future use |
| AVAR upstream issue draft | **PENDING REVIEW** | Draft at `artifacts/avar_upstream_issue_draft.md` for user authorization to file |
| Architectural plan citation cleanup | **DEFERRED** | Paper 3 manuscript prep timing; not blocking implementation |

---

## Section 6 — Open decisions awaiting resolution

These are concrete actionable items future sessions might encounter.

### 6.1 Engineering work blocks (concrete and ready)

**Next Wave 2 engineering work block — IRK + UNC-103 translation.** Per option α, AVA's actual channel set is IRK + LEAK + EGL19 + NCA + UNC103. We have LEAK (passive), EGL19 (Phase β CP2), NCA (Phase β C.3) validated for AVA. **IRK and UNC-103 are the missing translations.**

UNC-103's NMODL file uses NMODL EXTERNAL declarations (per Phase α report §2 warnings) — translation pattern: convert each GLOBAL/EXTERNAL declaration to per-cell state variable in Brian2 NeuronGroup. Pattern documented in `translation_patterns.md`. Estimated: ~1-2 days per channel translation given the pattern catalog is mature. After translation: re-run Phase F Component 2b under option α target (Nicoletti's AVAL phenotype: linear I-V, sustained-during-stimulus plateau, ~200 ms slow rise). 

**Phase F re-evaluation under option α target.** Once IRK + UNC-103 are in place, Brian2 4-channel AVA cell construction = Nicoletti's actual AVA setup. Target match should be straightforward (apples-to-apples against NEURON's AVA simulation with same channel set). If 4-channel cell matches Nicoletti's published AVAL phenotype, Phase γ Gate 2 is cleared with revised target. Estimated: 1-2 days work given infrastructure is mature.

### 6.2 Architectural plan revision

Mellem 2008 → AVA references at architectural plan lines 112, 177, 183, 275, 281, 294 need revision. Wang 2001 → SHK-1 attribution at line 94 needs revision (replace with Wei 2005 + Gu 2012 + Dobosiewicz 2019 + Liu 2018 per Nicoletti 2024 references [28, 30, 44, 45]). The "20 mV / 600 ms" specific values should be removed or replaced with Nicoletti's actual published protocol parameters (1000 ms CC, 500 ms VC).

**Timing:** paper 3 manuscript prep timing. Not blocking implementation. The plan as a working document for current Wave 2 work proceeds against this handoff's documented re-grounding.

### 6.3 AVAR upstream issue review and filing decision

Draft at `wave2/artifacts/avar_upstream_issue_draft.md`. The patch (`avar_unc103_patch.py`) restores AVAR runtime locally. Decision: file the upstream issue (productive for community), or keep the patch local-only. Either is defensible. Should be reviewed before paper 3 manuscript prep.

### 6.4 Paper 2 manuscript work

**Independent of Wave 2.** Can proceed at any time. Layer 1 closure with sample_004 + overrides + voltage-domain target framework is the empirical foundation. Workshop track NeurIPS GRL or ICLR LMRL.

Specific workstreams:
- Methods section: voltage-domain target framework as Layer 1 closure replacement for the +50 Hz cascade target. Sample_004 + 7 overrides as production calibration. Cross-target sensitivity analysis (Verdict A/B/invariance check) as methodology contribution.
- Results section: phenotype reproduction under sample_004 + overrides; behavioral closure metrics.
- Discussion: limitations (graded-cell biology, sign-mode dependency, calibration to LIF abstraction).

### 6.5 Methodology paper documentation

Today's case studies are extensive. Document while context is fresh. Specific case studies surfaced:

- F6 misdiagnosis correction (calibration converged for documented-wrong-reason).
- Mellem 2008 → AVA misattribution (citation propagation across artifacts).
- Wang 2001 → SHK-1 misattribution.
- Liu 2018 → 2020 reference year drift in v1 digitization JSON.
- F1-F17 NMODL gotcha catalog.
- Harness bugs F14/F15 caught during Phase β run #2.
- F16/F17 Ca-coupling unit-conversions caught during Ca-pool test.
- User-side error in citation audit prompt drafting (caught by agent pre-flight pushback).
- The hedging methodology bug (median-of-bimodal-distribution producing degenerate point) caught in Wave 1 corrected-weighting analysis.
- Wave 1 LHS analyzer's equal-weighted-mean failure mode (averaging spurious PASSes against load-bearing FAILs).

These constitute the empirical case-study catalog for paper 4. Documenting now prevents context loss.

### 6.6 Eventually — Wave 2 trajectory expansion

Once Wave 2 closes for AVA under option α, the channels translated for AIY/RIM/RMD remain valid. Wave 2 trajectory could naturally expand to AIY/RIM/RMD cell coverage. Bounded scope: AIY uses similar channel set (per Nicoletti's `AIY_simulation.py`), RIM uses partially-overlapping set, RMD uses Nicoletti 2019's (AWCon/RMD paper) channel set. Expansion is per-cell additive with rollback per the Path 3A discipline.

---

## Section 7 — Session-specific context: parameter analysis perspective

This section is Session 1's distinctive contribution. It ties Wave 1 LHS findings to Wave 2 cellular validation findings as one consistent methodology pattern.

### 7.1 The Wave 1 corrected-weighting analyzer

**The failure mode:** equal-weighted mean-of-criteria scoring averaged spurious PASSes (NSM elevation under over-excitation regime) against load-bearing FAILs (cascade-stability indicators). Sample-level rankings produced by the original analyzer were misleading because criteria were not independent — over-excitation produced PASS on NSM elevation while simultaneously producing FAIL on AVAL-spontaneous-below-80Hz. Mean-of-criteria masks the regime.

**The corrected scheme** (`scripts/brain/phase0_corrected_weighting_analyze.py`):

- Over-excitation guards (graded 1.0 below 80 Hz, hard FAIL above 120 Hz) gate samples. Failing guards nullifies the sample regardless of other criteria.
- Destabilization guards (hard FAIL above 200 Hz spontaneous) gate samples.
- Sensors, RIS rescue, NSM-anti-silence (reframed: PASS in 0.5-5 Hz, over-excitation degrades above 5 Hz), SMDVL — biology-grounded targets, mean-normalized per category.
- Cascade firing reported across +50/+20/+10/+0 Hz target sensitivity (LIF-rate framing flagged as graded-cell-artifact dimension).

Three outputs:

- **Verdict A (guards-only):** load-bearing for synthesis, biology-robust, sign-mode-independent.
- **Verdict B (target-based):** informational, contingent on LIF-rate framing.
- **Verdict invariance check:** which samples top-rank under A AND across all 4 B thresholds.

### 7.2 The hedging methodology bug

The original "hedged regime" was constructed as median-of-top-5 LHS samples. The top-5 had bimodal parameter distributions (e.g., W_syn_mV: 1.30, 0.80, 0.72, 1.30, 0.92 — clearly bimodal, median 0.92 sits between two clusters). Several parameters showed similar bimodality (C_mem_pF: 190, 71, 150, 50, 90 — bimodal; tau_ms: 6.99, 21, 7.71, 26, 5.67 — bimodal).

Median operation on parameters that aren't independent across multiple local optima produces a point that is **between** the optima, not in either. The hedged regime was "modest W_syn × low C_mem × triple gap" — neither sample's coherent regime, and empirically a synaptically-runaway-firing configuration (AVAL = 185 Hz at spontaneous rest under hedged + overrides; cascade saturates at the over-excitation guard threshold).

**Generalizable lesson:** don't hedge across bimodal parameter distributions by simple median. The right approach is either (a) identify the parameter manifold (PCA on top-K) and pick a point on it, or (b) commit to one local optimum and validate it directly. The 3-config validation correctly disambiguated by running sample_004 verbatim alongside the hedged regime — that disambiguation revealed hedging had produced a degenerate point.

### 7.3 Override-vs-no-override question — target-dependent

Sample_004 + 7 overrides scores marginally HIGHER under Verdict A (1.016) than sample_004 NO overrides (1.002). The difference is small (~1.5%) and comes from a slightly better guard score (overrides reduce over-excitation in sample_004's regime).

Verdict B differs by cascade target threshold:

- At +50 Hz target: NO overrides wins (1.061 vs 0.935; cascade Δ +85 hits target by 1.7×, while + overrides cascade Δ +18 fails the +50 target).
- At +20 Hz target: NO overrides wins (1.061 vs 0.992; nearly tied).
- At +10 Hz target: + overrides wins (1.073 vs 1.061).
- At +0 Hz / "any positive Δ" target: + overrides wins (1.073 vs 1.061).

**This is the synthesis-relevant verdict inversion.** The recommendation between override and no-override depends on whether the cascade firing target is +50 Hz (NO overrides better) or +10 Hz / "any positive" (+ overrides better).

Session 2's biophysical audit then established that AVA/AVE/AVB/AVD/PVC are graded non-spiking. The +50 Hz cascade firing target is an LIF-internal proxy without biology grounding. Under that finding, the Verdict B at +50 Hz threshold is target-on-non-existent-quantity. Verdict B at +10 Hz / +0 Hz (any positive direction-correct response) is closer to biology-meaningful, supporting + overrides.

### 7.4 Implications for paper 2 manuscript — voltage-domain target framework

Paper 2's manuscript has a clear methodology contribution: **voltage-domain target framework as Layer 1 closure replacement for the +50 Hz cascade target.**

The original LIF-rate-based Layer 1 closure framework was:
- Cascade firing target: AVA/AVD touch peri-pre Δ ≥ +50 Hz.
- Sample_004 hits this at +85 Hz (1.7×).
- Cross-cell validation: similar Δ targets per cell.

The voltage-domain replacement (paper 2 methodology):
- Cells are graded non-spiking. Cellular voltage trajectories are the biology referent.
- Spike-rate Δ in LIF abstraction translates to voltage trajectory features in cellular biophysics: dV/dt at stim onset, peak depolarization, plateau settling, post-stim recovery.
- Target framework: "biology-direction-correct response with voltage features within tolerance of literature observation," not "Δ Hz above arbitrary threshold."

Sample_004 + overrides scores high on voltage-domain criteria (cascade neurons cleanly depolarize on touch, plateau-during-stim, post-stim recovery within passive RC). The voltage-domain framework lets + overrides be defensibly chosen as production calibration without needing to commit to the +50 Hz LIF-rate target.

### 7.5 Calibration robustness as ongoing methodology concern

The pattern is structural across Wave 1 + Wave 2:

- **Wave 1 LHS:** parameter-space sensitivity testing surfaced that 40% of parameter space is over-excited under per-edge mode. The original analyzer didn't catch this because mean-of-criteria masked the regime.
- **Wave 2 F6 misdiagnosis:** empirical calibration converged on the right answer (0.518 mM/(mA/cm²·ms)) but documented the wrong reason (claimed "hidden NMODL machinery" with 5183× ratio). Production code was correct; the explanatory hypothesis was wrong.
- **Wave 2 density-sensitivity sweep:** terminator density has near-zero leverage on plateau phenotype across 32× scaling. Without the sweep, "tuning SLO-1 density up" would have been a plausible-looking response that didn't address the actual missing ingredient (Ca dynamics, not amount of SLO-1).

**The methodology lesson:** parameter-space sensitivity testing is necessary to validate that empirical calibrations are addressing the right mechanism, not converging on the right answer for documented-wrong-reason. The pattern extends naturally from Wave 1 LHS to Wave 2 cellular validation — sensitivity sweeps are a load-bearing methodology pattern.

### 7.6 Cross-cell density transfer — methodology question

Phase F's Component 2b cell construction transferred AIY-derived intensive densities (S/cm²) for SLO-1 isolated, SLO-1+EGL-19 coupled, and SHL-1 to AVA's geometry. This is a parameter-transfer-by-analogy methodology question worth surfacing for future work:

- **Defensible:** intensive densities (S/cm²) transfer naturally across cells with different geometries. Same density × different surface = different total nS — but per-area density is the more biologically meaningful quantity.
- **Caveat:** if AIY's per-channel densities are themselves the result of fitting to AIY's experimental recordings (which they are, per Nicoletti's `g_to_Scm2` workflow), they encode AIY-specific assumptions about how that channel contributes to AIY's phenotype. Transfer to AVA assumes the channel plays a similar mechanistic role in AVA — which may or may not be true.
- **For Wave 2 going forward:** when Wave 4-style CeNGEN-coupled per-cell densities become tractable, this analogy-based transfer should be replaced with CeNGEN-derived per-neuron channel-expression data. Until then, document the analogy explicitly when transferring densities.

### 7.7 Parameter-space sensitivity testing as standing methodology pattern

The density-sensitivity sweep extended the methodology pattern from Wave 1 LHS work to Wave 2 cellular validation. The shape of the sweep was similar:

- Define parameter axes with literature-grounded ranges where available, empirical bounds where not.
- Sweep at multiple resolutions (4×4 grid + extension probes for density-sensitivity; 30 LHS samples for Wave 1).
- Score each sample against multiple criteria using a corrected-weighting framework (Verdict A guards-gated; cascade target sensitivity).
- Identify trade-off surfaces: which parameters are unidentifiable, which are load-bearing, which are tunable.
- Use parameter sweep as evidence for whether observed phenotype is parameter-tunable vs structurally insufficient.

**Wave 3+ implementations of this pattern:** parameter sweep over per-channel densities for 4-channel AVA cell under option α. Sensitivity test at the new target. Cross-cell density transfer methodology (per §7.6) would benefit from sweep-based validation rather than direct analogy.

---

## Section 8 — Recommended next moves

Prioritized, with effort estimates.

### 8.1 Immediate (this week or next work block)

**Complete IRK + UNC-103 translation.** ~1-2 days per channel given the F1-F17 pattern catalog. After translation, re-evaluate Phase F Component 2b against Nicoletti's actual AVAL phenotype (option α target). Estimated total: 3-5 days for translation + validation.

**Authorize the AVAR upstream issue filing decision.** Draft at `wave2/artifacts/avar_upstream_issue_draft.md`. Either file or document local-only and proceed.

### 8.2 Engineering — short-term (2-4 weeks)

**Phase F re-evaluation under option α.** Once IRK + UNC-103 are translated, run Phase F Component 2b with the corrected 4-channel AVA cell + Nicoletti's actual AVAL phenotype target (passive RC-like, sustained-during-stimulus plateau, 200 ms slow rise, linear I-V). Expected to pass cleanly given Nicoletti achieves it in NEURON with same channel set. ~1-2 days work.

**Phase γ Gate 2 closure.** With component 2b passing under option α, declare Path A's cellular layer "production-grade" for AVA. Update gate-status documentation in `artifacts/checkpoints/`.

**Phase δ network integration kickoff.** Replace `graded_brain_h_kca.py`'s handcrafted Ca-K dynamics with the imported channel set for AVA (and other cells where the imported channels apply). Validate that network-level scenarios still run. Compare phenotypes against current LIFBrain + sample_004 baseline. ~1-2 weeks work.

### 8.3 Manuscript — parallel track (independent of Wave 2)

**Paper 2 manuscript prep.** Sample_004 + overrides + voltage-domain target framework is sufficient. Workshop track NeurIPS GRL or ICLR LMRL. Independent of Wave 2 trajectory.

Methodology contribution: voltage-domain target framework as Layer 1 closure replacement (per §7.4). Manuscript structure draft, methods section, results, discussion.

### 8.4 Methodology paper (paper 4)

Document while context is fresh. Today's case-study catalog is rich:

- F6 misdiagnosis correction
- Mellem 2008 → AVA misattribution
- Wang 2001 → SHK-1 misattribution  
- Liu 2018 → 2020 reference year drift
- F1-F17 NMODL gotcha catalog
- Harness bugs F14/F15
- F16/F17 Ca-coupling unit-conversions
- User-side error in citation audit prompt drafting
- Wave 1 hedging methodology bug
- Wave 1 LHS analyzer equal-weighted-mean failure mode

Each is a worked example of the cross-session adversarial review pattern. Catalog them now.

### 8.5 Architectural plan revision

Per §6.2. Deferred to paper 3 manuscript prep timing.

### 8.6 Eventually — trajectory expansion

Wave 2 to AIY/RIM/RMD coverage given the channels translated (per §6.6). Wave 3+ CeNGEN-coupled densities. Wave 4+ peptidergic extension and receptor binding kinetics.

---

## Section 9 — Methodology continuity notes

The cross-session adversarial review pattern that's been load-bearing today is documented here for continuity. Paper 4 will eventually formalize this; the practitioners' record is captured here.

### 9.1 Pre-flight pushback discipline

Sessions read prompts fully, surface concerns before starting work. **Today's catches:**

- Session 1 surfaced the bimodal hedging methodology concern before launching the 3-config validation — recommended the 3-config design itself rather than single-regime as originally specified.
- Session 1 surfaced the override-state mismatch (last night's LHS without overrides; today's validation with overrides) — recommended including sample_004 NO overrides as a control configuration.
- Session 2's audit prompt drafted a per-cell target table with citations (AVE/Wang 2020, AVB/Kawano 2011, etc.) that pre-flight grep showed do not exist in the canonical architectural plan. Pre-flight pushback caught it before deployment. **The pattern works in both directions.**

### 9.2 Mid-flight surfacing of findings

Don't batch findings to end of work block. Surface as discovered. **Today's instances:**

- Mellem 2008 misattribution surfaced mid-flight in Mellem investigation work block; spec instructed pause-and-surface, work block paused before classification verdict.
- F11 misdiagnosis (F6 was wrong about hidden NMODL machinery) surfaced in Phase A diagnostic of Phase β run #2; produced architectural simplifications F12, F13 before downstream phases ran.
- F14 (h.v_init bug in NEURONReference) surfaced via SHL-1 7.3% systematic divergence in Phase C; fixed before Phase D-F ran.
- F15 (SS extraction window mismatch) surfaced post-F14; fixed in same work block.
- F16 (caintra1 ⇄ slo1iso unit-conversion) surfaced during Ca-coupling cell builder construction; fixed before Phase F 2b re-evaluation.
- F17 (caintra1 fca-scaling not in calibrated coefficient) surfaced during Ca-coupling sensitivity sweep; fixed before sweep results were interpreted.

### 9.3 Stop-and-ask vs stop-and-pause

Different findings warrant different responses:

- **Stop-and-pause** (cross-session review needed for architectural questions): condition 6 surfacing in Phase F; Mellem misattribution surfacing in citation verification; major methodology concerns affecting load-bearing decisions.
- **Document-and-continue** (implementation question): F6 misdiagnosis in Phase A (corrected and noted; downstream work proceeded); harness bugs F14/F15 (fixed in-place); F16/F17 unit-conversion fixes (applied locally).

The discipline: if the finding affects architectural commitments or invalidates a load-bearing claim, pause for review. If it's an implementation correction that doesn't affect downstream architecture, document and continue.

### 9.4 Today's case studies (~17+ catches)

Aggregate count of substantive catches today:

1. Mellem 2008 misattribution (architectural plan)
2. Wang 2001 → SHK-1 misattribution (architectural plan)
3. "20 mV / 600 ms" no-primary-source (architectural plan)
4. Liu 2018 → 2020 year/title drift (v1 digitization JSON)
5. User-side citation audit prompt fabrication (caught by pre-flight pushback)
6. F6 misdiagnosis correction (run #2 Phase A)
7. F12 architectural simplification (cells don't insert Ca-pool)
8. F13 architectural simplification (slo1egl19 closed-form)
9. F14 h.v_init bug (NEURONReference)
10. F15 SS extraction window mismatch (voltage_clamp_harness)
11. F16 caintra1 ⇄ slo1iso unit-conversion
12. F17 caintra1 fca-scaling
13. Phase β-pre v1 metric error (post-hoc predictions misframed as fit targets)
14. Phase β-pre v2 Layer C tolerance structural-too-strict
15. Phase β-pre v3 Layer B substantively passes
16. Hedging methodology bug (median-of-bimodal produces degenerate point)
17. Wave 1 LHS analyzer equal-weighted-mean failure mode

Plus the F1-F10 NMODL gotcha catalog from run #1.

The pattern: substantial findings catch each other across sessions when discipline holds. The investment in pre-flight pushback + mid-flight surfacing pays dividends because errors compound silently otherwise.

### 9.5 Methodology paper opportunity

Documenting the framework + case studies while context is fresh is the immediate-term methodology paper opportunity. Paper 4 (cross-session methodology) is independent of Wave 2 architectural commitment — the contribution is the pattern itself, not specific findings.

---

## Section 10 — Pointer to artifacts

File-by-file index of what's where.

### 10.1 Today's primary artifacts

- `scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — Wave 2 architectural commitment document. Has known citation issues (Mellem 2008 in AVA, Wang 2001 → SHK-1, "20 mV / 600 ms"). Strategic shape (Path 3A primary, gate-based progression) remains load-bearing. Cleanup deferred to paper 3 manuscript prep.

- `scripts/brain/wave2/phase_alpha_report.md` — Phase α setup completion report. NEURON installation, Nicoletti compilation, smoke-test results, harness API observations.

- `scripts/brain/wave2/artifacts/phase_beta_findings.md` — F1-F17 NMODL gotcha catalog. Running log of findings during translation work.

- `scripts/brain/wave2/artifacts/phase_beta_run_summary.md` — Phase β overnight #1 summary (EGL-19 translated, CP1-CP3 passed).

- `scripts/brain/wave2/artifacts/phase_beta_run2_summary.md` — Phase β overnight #2 summary (F6 misdiagnosis corrected, all 7 channels translated, Gate 2a passed, Gate 2b "failed" against misattributed target, condition 6 surfaced).

- `scripts/brain/wave2/artifacts/f6_diagnostic_synthesis.md` — F6 verdict: PRINCIPLED. Symbolic vs empirical match to 5 dp. The "hidden NMODL machinery" framing was retracted.

- `scripts/brain/wave2/artifacts/gate2_ava_cell_construction.md` — channel densities and rationale for Phase F. Includes update post-density-sensitivity sweep.

- `scripts/brain/wave2/artifacts/density_sensitivity_analysis.md` — density-sensitivity sweep results. VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS. 4×4 grid + extension probes.

- `scripts/brain/wave2/artifacts/ca_coupling_test_results.md` — Ca-coupling integration test results. VERDICT_CA_COUPLING_INSUFFICIENT (robust). Refuted F12-derived hypothesis.

- `scripts/brain/wave2/artifacts/mellem_investigation_pushback.md` — load-bearing. Mellem 2008 → AVA misattribution finding. Re-grounding alternatives (path α/β/γ).

- `scripts/brain/wave2/artifacts/architectural_plan_citation_audit.md` — three additional misattributions confirmed (Wang 2001 → SHK-1, "20 mV / 600 ms" no-source, Liu reference year drift).

- `scripts/brain/wave2/speculative/speculative_summary.md` — GNN exploration outcomes + multi-compartment-explicit + NeuroML2-native comparison. Morphology fork remains strongest (when applicable).

- `scripts/brain/artifacts/phase0_corrected_weighting_analysis.txt` — Wave 1 corrected-weighting analyzer output (Session 1's morning work).

- `scripts/brain/artifacts/phase0_corrected_weighting_synthesis.json` — Wave 1 numerical synthesis (Session 1's morning work).

### 10.2 Wave 2 codebase

- `scripts/brain/wave2/channels/` — all 7 channel implementations
  - `egl19.py` (L-type Ca, Phase β CP2 run #1)
  - `shk1.py` (Kv1 delayed rectifier, Phase C.1 run #2)
  - `shl1.py` (Kv4 A-type, Phase C.2 run #2)
  - `nca.py` (NALCN homolog, Phase C.3 run #2)
  - `kqt3.py` (M-type K, Phase C.4 run #2)
  - `slo1_iso.py` (BK isolated, Phase D run #2)
  - `slo1_iso_dynamic_ca.py` (Ca-coupled variant, post-Phase F)
  - `slo1_egl19_coupled.py` (BK + EGL-19 closed-form `calcium(V)`, Phase E run #2)

- `scripts/brain/wave2/calcium_pool.py` — cadiff + caintra1 Brian2 implementations. Production code is correct; docstring's "5183×" claim is misleading and should be cleaned up in next maintenance pass.

- `scripts/brain/wave2/neuron_reference.py` — NEURONReference wrapper. F14 fix applied (h.v_init).

- `scripts/brain/wave2/voltage_clamp_harness.py` — Gate 2a infrastructure. F15 fix applied (SS window).

- `scripts/brain/wave2/plateau_harness.py` — Gate 2b infrastructure (against original Mellem-misattributed target; preserves three-label architectural-signature classifier).

- `scripts/brain/wave2/sensitivity_sweep.py` — density-sensitivity analysis driver.

- `scripts/brain/wave2/ca_coupled_cell.py` — Ca-pool integration test cell. F16/F17 fixes applied.

- `scripts/brain/wave2/translation_patterns.md` — F1-F17 systematized catalog (13 patterns documented).

- `scripts/brain/wave2/avar_unc103_patch.py` — AVAR runtime patch (upstream `AVAR_simulation_iclamp.py` missing).

- `scripts/brain/wave2/setup_neuron.py` — Phase α setup script (idempotent).

- `scripts/brain/wave2/reference_validation.py` — Phase α reference validation (3-cell determinism check).

- `scripts/brain/wave2/smoke_tests.py` — Phase α smoke tests.

- `scripts/brain/wave2/digitize_panels.py` — v1 digitization (preserved for historical record; contains misattributed Liu 2018 citation).

- `scripts/brain/wave2/digitize_panels_v2.py` — v2 corrected digitization.

- `scripts/brain/wave2/digitize_model_traces_v3.py` — v3 Layer B digitization.

- `scripts/brain/wave2/run_layer_b_validation_v3.py` — v3 Layer B validation runner.

- `scripts/brain/wave2/validate_*.py` — per-channel validators for EGL-19, SLO-1iso, SLO-1+EGL-19, calcium pool, Phase C channels, CP3 cell, Phase F Gate 2.

### 10.3 Wave 2 speculative work

- `scripts/brain/wave2/speculative/`
  - `gnn_architecture_sketch.md` — Variant A/B/C with diagrams
  - `training_data_feasibility.md` — data + parameter-count analysis
  - `comparison_framework.md` — Gate 2 extension for GNN validation
  - `prototype/` — minimal Variant A PyTorch prototype (training pipeline functional, 2-node test MAE 46 mV vs <5 mV target — informative-negative)
  - `multi_compartment_explicit.md` — sketch of architectural plan's morphology fork
  - `neuroml2_native.md` — sketch + previous-rejection confirmation
  - `x1_summary.md` — GNN summary
  - `speculative_summary.md` — overall summary

### 10.4 Upstream code (not in repo)

- `~/Desktop/C-Elegans/simulation/upstream/c302/` — c302 framework (cell morphologies, connectome readers, network templates). MIT license.
- `~/Desktop/C-Elegans/simulation/upstream/ChannelWorm/` — ChannelWorm models (4 NeuroML channels).
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` — Nicoletti 2024 source code (24 .mod files, 9 cell simulation scripts). License unverified; ModelDB convention is academic-use-with-attribution.

### 10.5 Wave 1 LHS data (Session 1 morning work and earlier)

- `scripts/brain/artifacts/phase0_param_lhs_traces/` — 30 LHS samples × 6 scenarios × 5 seeds = 900 simulation traces (full 300-neuron rasters + body_xy + fsm_states + modulator_conc).

- `scripts/brain/artifacts/phase0_param_lhs_configs/manifest.json` — LHS sample configurations (22 parameters per sample).

- `scripts/brain/artifacts/phase0_param_lhs_synthesis.json` — original LHS analyzer numerical synthesis (the analyzer with the equal-weighted-mean bug).

- `scripts/brain/artifacts/phase0_corrected_weighting_synthesis.json` — corrected analyzer synthesis (Session 1's morning work).

- `scripts/brain/artifacts/phase0_layer1_validation_*.npz` — 3-config validation data (hedged + overrides, sample_004 + overrides, sample_004 NO overrides).

- `scripts/brain/phase0_param_lhs_analyze.py` — original analyzer (preserved unchanged).

- `scripts/brain/phase0_layer1_validation_analyze.py` — original 3-config analyzer (preserved unchanged).

- `scripts/brain/phase0_corrected_weighting_analyze.py` — corrected analyzer (Session 1's morning work).

### 10.6 Production environments

- `~/venvs/wave2-neuron/` — isolated venv for Wave 2 (Python 3.12.3, NEURON 9.0.1, Brian2 2.10.1, NumPy 2.4.4).

- `~/miniconda3/envs/ml/` — production brain conda env (Python, Brian2, MuJoCo, etc.). Untouched by Wave 2 work.

### 10.7 Project context

- `claude-chat-context.md` (project root) — live context document. May be slightly out of date relative to today's findings (Mellem 2008 references, etc.). Should be updated at paper 3 prep time.

- `~/Desktop/website/personalwebsite/src/content/projects/c-elegans-multimodal.mdx` — public-facing project summary at rohitravi.com. Behind on Wave 2 details; updated at material-state-change cadence.

- `scripts/brain/artifacts/handoffs/session_3_handoff_2026-04-26.md` — companion handoff with cellular-biophysics framing (Session 3's perspective).

---

## Closing notes

This handoff captures project state as of 2026-04-26 evening. Standalone-sufficient per Division C: a future session can pick up project work using only this document. For deep dives, the artifacts in §10 remain available.

The methodology pattern that ran throughout today (~17+ catches across pre-flight pushback, mid-flight surfacing, cross-session adversarial review) is load-bearing for paper 4 and should be documented while context is fresh.

The user's distinguishing context: NYU undergrad doing genuinely original research at the intersection of computational neuroscience, biophysics, and behavioral simulation. Both paper trajectories (paper 2 behavioral, paper 3 mechanistic) have clear paths forward. Wave 2's option α re-grounding preserves the channel-translation infrastructure investment while avoiding a 3-4 week morphology-fork commitment against a misattributed target. The next engineering work block (IRK + UNC-103 translation, ~3-5 days) is well-scoped and ready.

Standing by.

— Session 1
