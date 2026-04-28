# Session 3 handoff — 2026-04-26 evening

**Author:** Session 3 (cellular-biophysics perspective)
**Companion:** Session 1's handoff at `session_1_handoff_2026-04-26.md` (parameter-analysis perspective)
**Division C:** either handoff alone is standalone-sufficient. Redundancy is intentional — the two perspectives cover the same project state with different framings.

---

## Section 1 — Document purpose and how to use it

This document captures the C. elegans biophysical-simulator project state as of 2026-04-26 evening and brings new sessions current on a project trajectory that shifted substantially today. The user is **Rohit Ravi** — NYU undergrad, Data Science major with Philosophy minor, working toward an industry AI career bridging technical and philosophical domains, with the simulator as a long-running research project. The project has both behavioral-paper and mechanistic-paper trajectories; today's work block landed in the mechanistic-trajectory infrastructure (Wave 2 channel translation).

Read this document end-to-end before opening any source artifact. The document is designed to be standalone-sufficient: a future session can pick up project work without re-reading underlying artifacts (though they remain available at the paths listed in §10 for deep dives).

The document's distinctive contribution comes from §7 — Session 3 spent the morning on Wave 1 cellular validation work (the τ_d wall finding, structural insufficiency conclusion for the single-compartment K_Ca+h sandbox), then was idle while Sessions 1 and 2 executed Wave 2 work. The cellular-biophysics lens connects Wave 1 cellular findings to today's Wave 2 architectural results: **Wave 1's "single-compartment can't sustain biology-grounded plateau" finding directly motivated the architectural plan's condition 6, which today's Phase F empirically activated — but the biological target condition 6 was being measured against turned out to be misattributed. The directional prediction was right; the specific quantitative target was wrong.** That intersection is the cellular-biophysics framing's distinctive contribution to synthesis.

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

Of the 7 essential-set channels, **4 (EGL-19, NCA, IRK-equivalent via leak proxy, leak) are validated for AVA's actual channel set** (per Nicoletti's `AVAL_simulation_iclamp.py` source: AVA uses IRK + LEAK + EGL19 + NCA + UNC103). **3 (SLO-1 isolated, SLO-1+EGL-19, SHK-1, SHL-1, KQT-3) are valid for AIY/RIM/RMD use** but not in AVA's actual channel set. IRK and UNC-103 are **not yet translated** — they're the missing pieces for AVA-specific validation under the re-grounded path α (see §3.4).

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

**Cellular-biophysics implication** (Section 7 elaborates): Wave 1's compartmental + K_Ca + h cellular validation work was *also* being measured against the same misattributed target. Wave 1 found "single-compartment graded fails Mellem; compartmental scaffold also fails Mellem at v_rest = −25 mV with 50 pA injection due to dendritic τ_d wall." That finding is correct as an *engineering observation about model behavior* but its *biological framing* ("Mellem says AVA should plateau for 600 ms") was wrong. The cellular validation methodology was sound; the per-cell phenotype target needed verified primary sources.

### 3.3 Citation audit caught three additional misattributions

Following the Mellem investigation, a citation audit (`architectural_plan_citation_audit.md`) verified the load-bearing biological citations in the architectural plan against primary sources (Nicoletti 2024's full reference list extracted from on-disk PDF). Three additional misattributions surfaced:

1. **Wang 2001 → SHK-1 (architectural plan line 94).** The plan attributes "rich worm-specific data" for SHK-1 to Wang 2001. The actual Wang 2001 paper (Nicoletti 2024 reference [60]) is **Wang et al. 2001 *Neuron* 32:867-881 — about SLO-1 at the neuromuscular junction, not SHK-1.** Nicoletti 2024 cites Wei 2005 (KCNQ-like K channels in *C. elegans*, JBC), Gu 2012 (Kv3.1 splicing, JBC), Dobosiewicz 2019 (*Elife*), and Liu/Kidd/Dobosiewicz/Bargmann 2018 *Cell* for SHK-1. Wang 2001 is not in that set.

2. **The "20 mV / 600 ms" numerical pair has no primary source.** Audit traced Nicoletti 2024's actual AVA protocols: current-clamp 1000 ms / 7 steps -30 to +30 pA; voltage-clamp 500 ms / 16 steps -120 to +50 mV. Neither matches "600 ms." The "20 mV" approximates AVAL's typical depolarization range under +30 pA injection but is not stated as a target in any primary source. The "600 ms" appears to be an unverified interpolation between Nicoletti's 500/1000 ms protocols, or a misremembered Mellem RMD value.

3. **Liu 2018 reference year drift in v1 digitization JSON.** The v1 `published_traces.json` recorded "Liu P, Chen B, Wang Z-W. **2018**. Postsynaptic current bursts instruct action potential firing at a graded synapse. ref [29] in Nicoletti 2024." The actual Nicoletti 2024 reference [29] is **Liu/Chen/Wang 2020 Nat Commun "GABAergic motor neurons bias locomotor decision-making"** — wrong year (2018 vs actual 2020) AND wrong title. The v1 digitization agent fabricated this citation; the agent's "Liu 2018" attempt was conflating with the *Liu/Kidd/Dobosiewicz/Bargmann 2018 Cell* AWA paper (which is reference [30], not [29]).

The architectural plan cleanup itself is **deferred to paper 3 manuscript prep timing** — not blocking implementation work. Implementation work proceeds against the re-grounded targets per §3.4 below.

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

The Wave 1 cellular validation work that produced the τ_d wall finding (Session 3's morning work; see §7) sits in the same category: empirically real engineering observation about what the compartmental scaffold does, framed against a target that turned out to be misattributed. The compartmental scaffold itself remains shipped scaffolding (compartmental_neurons.py + compartmental_neurons_kca.py sandbox) for any Wave 3+ work block that surfaces a phenotype warranting compartmental dynamics.

### 3.6 Citation audit cleanup deferred

The architectural plan retains its strategic shape (Path 3A primary, per-channel rollback, two-component Gate 2, per-channel translation priorities) under option α. The specific citation cleanups (Mellem 2008 → AVA references at lines 112, 177, 183, 275, 281, 294; Wang 2001 → SHK-1 at line 94; "20 mV / 600 ms" specific values throughout) are documented as needing revision but **deferred to paper 3 manuscript prep timing.** The plan as a working document is not blocking Wave 2 implementation — implementers should refer to this handoff and `mellem_investigation_pushback.md` + `architectural_plan_citation_audit.md` for the corrected target framing.

---

## Section 4 — Today's chronological work summary

This section provides a narrative arc; specific artifacts have detail.

### 4.1 Morning — Wave 1 closure (Session 1, 2, 3 in parallel)

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

Session 3 ran cellular validation work on the compartmental scaffold + K_Ca + h sandbox (`compartmental_neurons_kca.py` built today). Tested at v_rest = −25 mV (Mellem AVA up-state framing) and v_rest = −65 mV (scaffold's original mammalian-template baseline) with injection-current sweep. Found:

- At v_rest = −25 with 50 pA injection: K_Ca dominance prevents plateau ignition across all g_KCa values [0.25, 0.5, 1.0, 2.0, 5.0] nS. Plateau response capped at +5 mV vs +20 mV target.
- At v_rest = −65 with 200 pA injection: Mellem amplitude (+22.6 mV) achievable, but duration (30 ms) 20× too short.
- Mechanistic diagnosis: dendritic τ_d = 20 ms is too short. Leak conductance dominates termination (~115 pA outward at peak vs 26 pA inward I_Ca). K_Ca and h contribute negligibly at the regime that produces correct amplitude.
- Verdict: γ-architectural-extension. Compartmentalization addresses K_Ca dominance issue (single-compartment had K_Ca dominate; compartmental's leak isolation works) but the underlying constraint is τ_m / R_input across all architectures — Goodman 1998 implies worm dendritic τ should be 200-500 ms; scaffold uses 20 ms.

**Synthesis at Wave 1 closure:** the LIF-rate-based Layer 1 closure framework is structurally complete but its cascade-firing target is graded-cell-artifact. Voltage-domain target framework recommended for paper 2. Wave 2 architectural commitment supported. Session 3's cellular finding directly motivated condition 6 in the Wave 2 architectural plan — see §7.

### 4.2 Morning — Wave 2 architectural commitment

User accepted Path 3A primary based on Wave 1 closure synthesis + earlier audit work. Architectural plan committed at `phase_v_w2_architectural_plan.md`.

Conditional fork structure: 6 invalidation conditions (lines 270-281 of plan), including condition 6 ("cellular validation fails not on channel kinetics but on compartmental architecture"). The plan committed to morphology fork as response to condition 6 if it surfaced. **Condition 6's framing was directly motivated by Session 3's Wave 1 cellular validation findings** — see §7.4.

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

This is a methodologically important finding: **empirical calibration converging on the right answer for documented-wrong-reason.** The kind of pattern that propagates across artifacts unless re-checked. (Cellular biophysics analog: the same pattern showed up in Wave 1's compartmental scaffold parameters — α_Ca = 0.05 dimensionless/(pA·ms) was off by 100× from the literature-grounded value, but the empirical [Ca] dynamics didn't saturate as my analytical prediction suggested because the cell never reached the depolarization regime where it would have. Empirical-vs-analytical divergences have multiple possible explanations; the audit discipline matters.)

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

Cellular-biophysics observation worth highlighting: the **two-component Gate 2 design (2a kinetics correctness + 2b architectural sufficiency) was specifically motivated by Wave 1 cellular validation findings.** The decoupling matters because per-channel passes don't entail cell-level passes — Wave 1 already demonstrated this pattern (compartmental scaffold's per-cell parameters all passed individually under static conditions, but the cell-level dynamics under realistic injection-current protocols failed Mellem). Without the 2a/2b decoupling, Phase F would have produced a single ambiguous "Gate 2 FAIL" verdict that conflated channel kinetics with architectural sufficiency.

### 4.7 Density-sensitivity sweep

Before authorizing the morphology fork, density-sensitivity analysis (`density_sensitivity_analysis.md`) tested whether the failure was density-tunable vs truly architecturally insufficient. 4×4 grid (terminator and Kv each scaled by {0.5, 1.0, 2.0, 4.0}, plus extension probes at {0.25, 8.0}) over the 5 non-Nicoletti density parameters. The 3 principled-density channels (EGL-19, leak, NCA) were held at Nicoletti AVAL g0 throughout.

**Verdict: VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS.**

- Amplitude can be tuned into target range (kv=8 → 17.7 mV ∈ [15, 25]) but at cost of duration collapsing to 4.4 ms.
- Maximum duration anywhere in the swept volume: 42 ms — order of magnitude short of 400 ms lower bound.
- **Terminator scaling has near-zero leverage** on the phenotype: 32× variation in SLO-1 conductance produces zero meaningful change.
- Mechanistic interpretation: SLO-1 isolated reading static cai (per F12) cannot mediate Ca-feedback; no slow positive-feedback loop sustains a plateau. The missing ingredient is *Ca dynamics*, not *amount of SLO-1*.

This is the **direct empirical analog of Session 3's Wave 1 finding** — Session 3 found that K_Ca scaling (g_KCa sweep) had limited leverage at v_rest = −25 in compartmental mode (cell hyperpolarized below v_rest and plateau didn't ignite). Session 2's density-sensitivity sweep found the same shape (terminator scaling near-zero leverage) at v_rest = −65 in single-compartment mode. The mechanism is different (K_Ca dominance vs Ca-feedback gain-locked at zero) but the *shape* of the finding is the same: K-channel terminator parameters have limited leverage on the plateau phenotype because the limiting mechanism is elsewhere in the architecture.

### 4.8 Ca-coupling integration test

Before triggering the morphology fork, the cheaper architectural extension was tested: add dynamic caintra1 pool, couple EGL-19's I_Ca to [Ca]_i, let SLO-1 isolated read dynamic [Ca]_i state. Test ran overnight 2026-04-26 → 2026-04-27.

**Verdict: VERDICT_CA_COUPLING_INSUFFICIENT** (robust across 5 orders of magnitude of fca scaling and 4× SLO-1 conductance).

Key findings:
- At Nicoletti default fca, [Ca]_i barely moves (51 nM peak vs 50 nM baseline; 1.02× fold change). The Ca-coupling loop is *thermodynamically disengaged* — gain-locked at near-zero.
- At fca up to 10000× default, [Ca]_i reaches 17 μM, and **plateau duration *decreases* monotonically as the loop engages**.
- The Ca-coupling loop is **negative feedback for V** because SLO-1 is hyperpolarizing — it cannot extend a depolarized plateau, only terminate it faster.

This contradicted the F12-derived hypothesis. F12 correctly identified that no Ca-feedback existed; what F12 (and morning review's read of it) implicitly assumed was that *adding* Ca-feedback would extend the plateau. The Ca-coupling test refuted that assumption with quantitative evidence.

**Cellular-biophysics observation** (elaborated in §7.5): SLO-1 in single-compartment with bulk Ca-pool is structurally hyperpolarizing as a feedback element. The Ca-coupling loop is negative feedback for V. To extend a depolarized plateau, the architecture needs either (a) a depolarizing Ca-coupled mechanism (e.g., CICR via IP3R/RyR releasing Ca that activates depolarizing channels), or (b) compartmental isolation that sequesters the K_Ca termination from the plateau-sustaining current sources, or (c) cellular phenotypes that don't actually require sustained-bout dynamics in the first place (which option α points to).

Two new findings produced during this work block:
- **F16:** caintra1 ⇄ slo1iso unit-conversion (×1000) is required. caintra is M-equivalent (5e-8 raw), slo1iso's cai is mM-scale (5e-5 raw). Cell builders wiring dynamic caintra1 to slo1iso must insert `cai_mM = caintra_raw * 1000`.
- **F17:** caintra1 fca-scaling is not in the calibrated `coef_in_eff`. `calcium_pool.py` accepts fca but doesn't rescale the empirical coefficient. Sweep callers must explicitly multiply by `fca / 0.001`.

### 4.9 Speculative GNN exploration

In parallel with the density-sensitivity sweep, speculative-architecture work block characterized three alternative responses to compare against the morphology fork: **GNN hybrid (X.1), multi-compartment-explicit (X.2a), NeuroML2-native (X.2b).**

Top-line: **morphology fork remains strongest condition-6 response** (when condition 6 was thought to apply). GNN Variant A is plausible Wave 4 enhancement (per-segment density fitting + CeNGEN coupling), not Wave 2 alternative. Variants B/C not currently feasible (data-starved). NeuroML2-native rejected (same morphology work needed, worse backend performance, higher infrastructure cost).

The 2-node GNN prototype's 46 mV final test MAE (vs <5 mV target) was reported as "informative-negative" — the training pipeline functional, but bounded effort insufficient to determine whether more careful setup would converge or whether the architecture is structurally inefficient at small scales.

Session 3 was positioned to comment if GNN exploration produced positive results that would have warranted cellular-biophysics scrutiny (does the per-segment density learning produce biology-defensible density distributions, vs. fitting to artifact). The "informative-negative" outcome means no scrutiny was needed at this work block; the GNN trajectory remains a Wave 4+ option.

All speculative outputs in `wave2/speculative/`. No `SPECULATIVE_PAUSE.txt` was created — no load-bearing concern surfaced during investigation.

### 4.10 Mellem investigation surfaced misattribution

Triggered by `phase_v_w2_mellem_investigation_prompt.md`. Spec instructed pause-and-surface immediately if "Mellem 2008" citation verification surfaced a misattribution. It did. Per §3.2 above.

The investigation produced `mellem_investigation_pushback.md` with primary-source quotes and the path-α/β/γ re-grounding alternatives. User selected option α.

### 4.11 Citation audit confirmed three more misattributions

Per §3.3 above. Audit document at `architectural_plan_citation_audit.md`. Methodology lesson surfaced before audit deployed: the audit prompt initially proposed verifying a per-cell target table with citations (AVE/Wang 2020, AVB/Kawano 2011, etc.) — pre-flight grep showed none of these citations exist in the architectural plan. They were constructed from memory rather than verified against canonical document. Pre-flight pushback caught it before deployment, exactly as the audit was designed to prevent. **The pattern works in both directions:** user-side fabrication caught by agent pre-flight pushback, agent-side fabrication caught by user/cross-session review.

### 4.12 AVA re-grounding to option α committed

Per §3.4 above. The 3-4 week morphology fork commitment was avoided. The empirical findings (Phase F 2b, density sweep, Ca-coupling test) preserved as engineering observations about model behavior under the original (misattributed) target framing.

Session 3's Wave 1 cellular validation findings (compartmental + K_Ca + h sandbox at v_rest = −25) similarly preserved as engineering observations. The compartmental scaffold (`compartmental_neurons.py`) and K_Ca patch sandbox (`compartmental_neurons_kca.py`) remain shipped scaffolding. They become potentially-relevant if Wave 3+ work surfaces a phenotype that warrants compartmental dynamics; for Wave 2 under option α, they're not on the critical path.

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
| Compartmental scaffold (Wave 1 work) | **SHIPPED, UNDEPLOYED** | `compartmental_neurons.py` + `compartmental_neurons_kca.py` sandbox built, never integrated into LIFBrain. Available for Wave 3+ work if needed |
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
- Session 3's α_Ca scale catch (sign error + scale error in K_Ca patch parameters before launch).
- Session 3's d[Ca]/dt sign convention catch (same; would have produced "K_Ca patch fails" verdict for wrong reason).
- Session 3's analytical h_ss derivation showing tau_h-only sweep would not have produced passes (averted ~1 day of unproductive sweep).

These constitute the empirical case-study catalog for paper 4. Documenting now prevents context loss.

### 6.6 Eventually — Wave 2 trajectory expansion

Once Wave 2 closes for AVA under option α, the channels translated for AIY/RIM/RMD remain valid. Wave 2 trajectory could naturally expand to AIY/RIM/RMD cell coverage. Bounded scope: AIY uses similar channel set (per Nicoletti's `AIY_simulation.py`), RIM uses partially-overlapping set, RMD uses Nicoletti 2019's (AWCon/RMD paper) channel set. Expansion is per-cell additive with rollback per the Path 3A discipline.

**Cellular-biophysics watchpoint:** AIY, RIM, RMD cellular validation will trigger the same per-cell phenotype-target verification need that Mellem 2008 → AVA surfaced. Lesson: verify each cell's phenotype target against primary sources before structuring validation around it. The pattern documented in `mellem_investigation_pushback.md` should generalize.

---

## Section 7 — Session-specific context: cellular biophysics perspective

This section is Session 3's distinctive contribution. It connects Wave 1 cellular validation work to today's Wave 2 architectural findings, with emphasis on the *intersection*: Wave 1's "single-compartment cellular insufficiency" finding directly motivated condition 6 in the architectural plan; today's Phase F empirically activated condition 6; the citation audit then revealed condition 6's biological framing was wrong. The directional cellular-biophysics prediction was correct; the specific Mellem target it was being measured against was misattributed.

### 7.1 The Wave 1 compartmental + K_Ca + h sandbox

**The setup:** Session 3 built `compartmental_neurons_kca.py` as a sandboxed extension of the existing compartmental scaffold (`compartmental_neurons.py`), adding three variants:

- **base** — scaffold equations as-is, no h, no K_Ca
- **h_only** — adds h-inactivation as direct multiplicative gate `I_ca = g_ca × m_inf × h × (E_Ca − v_d)`
- **h_kca** — adds h-inactivation + intracellular [Ca] pool dynamics + Ca-activated K+ current on dendrite

Pre-flight analytical work derived h_ss = 0.3 / (0.3 + m_inf) from the scaffold's existing h equation form. At v_rest = −25 mV with v_ca_half = −30 mV, m_inf ≈ 0.70 → h_ss ≈ 0.30. **h-only termination is structurally insufficient** — it leaves ~70% of I_Ca permanently uninactivated, so the cell cannot relax to baseline after plateau ignition. This was documented as an analytical prediction; the K_Ca patch was the proposed fix.

Pre-flight catches (documented in §6.5 above):
- d[Ca]/dt sign convention (the prompt initially specified `−α_Ca × I_Ca`, which would be backwards under the scaffold's "I_Ca > 0 = inward = depolarizing" convention; corrected to `+α_Ca × I_Ca`)
- α_Ca scale issue (initial value 0.05 dimensionless/(pA·ms) would produce non-physiological [Ca] saturation of K_Ca; corrected to 0.0005)

Both catches were structural-correctness issues that would have produced "K_Ca patch fails" verdicts for wrong mechanistic reasons.

### 7.2 The empirical findings — compartmental + K_Ca + h cannot pass Mellem at any tested regime

**Phase 1.5 sensitivity sweep on AVAL** at v_rest = −25 mV (Mellem AVA "up-state" framing per pre-Mellem-investigation prompt) with 50 pA × 100 ms injection across g_KCa ∈ [0.25, 0.5, 1.0, 2.0, 5.0] nS:

| g_KCa (nS) | v_d settle | plateau_amp | plateau_dur | failure mode |
|---:|---:|---:|---:|---|
| 0 (h_only) | −10.2 | +1.9 | 0 | h-only insufficient (Phase 0 prediction) |
| 0.25 | −15.4 | +1.9 | 0 | K_Ca pulling v_d below v_rest |
| 0.5 | −20.3 | +2.2 | 0 | continuing |
| 1.0 | −31.3 | +5.0 | 0 | v_d now below v_rest |
| 2.0 | −47.4 | +5.1 | 4 | strong K_Ca dominance |
| 5.0 | −48.8 | +2.2 | 0 | full K_Ca dominance |

Failure mode: at v_rest = −25 with v_ca_half = −30, m_inf at rest is ≈ 0.7 (cell sits at activation midpoint). K_Ca compensates by pulling v_d below v_rest. Plateau ignition fails because the cell can't be pushed back up to v_ca_half from the K_Ca-dominated equilibrium below v_rest. **Same K_Ca dominance pattern Session 2 observed in single-compartment graded mode.**

**Phase 1.5b at v_rest = −65 mV (scaffold's original mammalian template):** identical results across all variants. m_inf at −65 ≈ 0.003; 50 pA insufficient to reach plateau threshold. All variants produce +4.5 mV plateau, no termination dynamics.

**Phase 1.5c at v_rest = −65 mV with 500 pA (Session 2's scaled injection):** plateau ignites strongly (+48-75 mV amplitude). Compartmentalization DOES support plateau dynamics at this regime. But amplitudes are 2-4× Mellem target; durations are 6-70 ms (10-100× too short).

**Phase 1.5d injection sweep at v_rest = −65, g_KCa = 2 nS, h_kca:** found the regime where amplitude hits Mellem target.

| inject (pA) | plateau_amp | plateau_dur |
|---:|---:|---:|
| 25 | +2.2 | 0 |
| 50 | +4.5 | 0 |
| 100 | +9.3 | 21 |
| 150 | +14.9 | 31 |
| **200** | **+22.6** ✓ | **30** ✗ |
| 300 | +40.6 | 10 |
| 500 | +53.3 | 18 |

**Mellem amplitude target (+20 mV ±5) IS achievable at 200 pA injection. Mellem duration target (600 ms ±200) is NOT achievable at any tested injection.** The amplitude/duration trade-off doesn't intersect the Mellem operating point.

### 7.3 The τ_d wall — the load-bearing diagnostic finding

At v_rest = −65 with 200 pA injection (the regime that produces correct amplitude):
- h_min = 0.94 (h barely inactivates)
- f_Ca_max = 0.16 (K_Ca barely activates)
- **Yet plateau terminates in 30 ms**

K_Ca and h contribute negligibly to termination at this regime. **The dendritic leak is the dominant termination force**: g_leak_d = C_mem / τ_d = 100 pF / 20 ms = 5 nS pulling v_d toward v_rest.

After plateau peaks at v_d ≈ −42 mV, leak current = 5 nS × (−65 − (−42)) = **−115 pA outward**, far exceeding the I_Ca = 26 pA inward at that voltage. **Plateau terminates because membrane leak dominates the post-injection equilibrium**, not because of K_Ca or h-inactivation.

For Mellem-grounded 600 ms plateau (under the misattributed Mellem framing), dendritic τ_d would need to be ~200-500 ms (10-25× longer than current scaffold's 20 ms). This is consistent with the broader R_input / τ_m gap I documented in earlier project audit work: LIFBrain has 100 MΩ vs Goodman 1998's ~5 GΩ for worm neurons — 50× too low across the architecture.

**The scaffold's τ_d = 20 ms is mammalian-cortical-template; biological worm dendrite τ would be 200-500+ ms based on Goodman 1998 R_input.** This is the τ_d wall.

### 7.4 Connection to today's findings — condition 6 was empirically real

The architectural plan's condition 6 ("cellular validation fails not on channel kinetics but on compartmental architecture") was added to the plan **specifically because Session 3's Wave 1 cellular validation predicted this failure mode would surface**. The Wave 1 finding (single-compartment / single-architecture-leak-isolation cannot reproduce sustained-bout dynamics matching Mellem-style 600 ms targets) directly motivated the conditional fork structure.

Today's Phase F empirically activated condition 6 via Session 2's Wave 2 work:
- Per-channel kinetics: PASS (Gate 2 component 2a)
- Cell-level architectural sufficiency: FAIL (Gate 2 component 2b — 46.8 mV / 21.4 ms vs target 15-25 mV / 400-800 ms)
- Density sweep refused parameter rescue: VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS
- Ca-coupling refused architectural rescue: VERDICT_CA_COUPLING_INSUFFICIENT

**Wave 1's prediction was directionally correct.** Phase F's empirical signature (channels pass per-channel, cell fails) matches exactly the pattern Wave 1 surfaced in the compartmental + K_Ca + h sandbox. The diagnostic methodology (decouple per-channel correctness from cell-level dynamics, classify by which fails) transferred cleanly from Wave 1 to Wave 2.

### 7.5 The deeper finding — biology framing was wrong

Then the citation audit surfaced that the Mellem-2008-in-AVA target was misattributed. **AVA doesn't biologically have a 600 ms self-sustaining plateau per Mellem 2008 primary source.** The plateau dynamics characterized in Mellem 2008 are in RMD, a different cell. Mellem 2008 explicitly reports:

> "we never observed action potentials in AVA (n=10)"
>
> "even when we changed the resting potential to more hyperpolarized levels"

So Wave 1's directional prediction ("single-compartment cannot sustain biology-grounded plateau") was correct as a *generic statement about single-compartment limitations*, but the *specific quantitative target* it was being measured against (Mellem 2008 in AVA) was a phantom. Under option α (Nicoletti's actual AVAL phenotype: passive RC-like, sustained-during-stimulus plateau, ~200 ms slow rise, linear I-V), single-compartment is *expected* to suffice — Nicoletti achieves it in single-compartment NEURON with the 5-channel AVA set.

**Methodologically:** the cellular validation methodology was sound (per-cell phenotype targets, decoupled per-channel and cell-level metrics, parameter sweeps to rule out density tunability, mechanism analysis to identify limiting factors). What was missing was **citation hygiene on the per-cell phenotype targets themselves.** The lesson generalizes: when a cell-level validation surfaces a "fails biology" verdict, verify the biology citation against primary source before drawing architectural conclusions.

This is a paper-4 case study: cellular validation methodology works as designed; the failure mode caught was real architecture-vs-target mismatch; but the *target* was a propagated misattribution that survived through architectural plan drafting because nobody re-grounded it against primary source until the Mellem investigation specifically asked the question.

### 7.6 The mechanistic finding — SLO-1 in single-compartment is hyperpolarizing feedback

Today's Ca-coupling integration test (Session 2's overnight work, `ca_coupling_test_results.md`) produced a mechanistic finding that connects directly to Wave 1's observations:

**SLO-1 in single-compartment with bulk Ca-pool is structurally hyperpolarizing as a feedback element.** The Ca-coupling loop (EGL-19 inward Ca → [Ca]_i rises → SLO-1 K_Ca activates → outward K current) is **negative feedback for V**. It can only terminate plateau, never extend it.

This explains Wave 1's empirical observations:
- At v_rest = −25 with K_Ca patch active: K_Ca dominance prevents plateau ignition (cell sits at K_Ca-balanced equilibrium below v_rest, can't be pushed back up).
- At v_rest = −65 with K_Ca patch active: K_Ca contributes to termination but the dominant termination force is leak (τ_d wall). K_Ca + h are subdominant.

In both regimes, K_Ca cannot extend a plateau. To match a sustained-bout phenotype like the misattributed Mellem 600 ms target, the architecture would need either:

- **Compartmentalization with sequestered K_Ca** — K_Ca on dendrite, plateau machinery on dendrite, soma isolated from K_Ca termination so it can sustain output. Wave 1's compartmental sandbox attempted this (K_Ca on dendrite); the τ_d wall finding shows it doesn't work in the current scaffold parameters because dendritic τ_d is too short for sustained dynamics regardless of K_Ca placement.
- **A depolarizing Ca-coupled mechanism** — e.g., CICR via IP3R/RyR releasing Ca that activates depolarizing channels. Not in current architecture.
- **Cellular phenotypes that don't require sustained-bout dynamics** — option α's recognition that AVA doesn't biologically have such dynamics. The "fix" is the biology framing, not the architecture.

Per option α, the cellular phenotype target for AVA is **Nicoletti's passive-RC-like sustained-during-stimulus response**, which doesn't require any of the above. Single-compartment with the 5-channel AVA set should produce it cleanly.

### 7.7 Architectural decision-making at cellular level — the 2a/2b decoupling

Wave 2's two-component Gate 2 design (2a = channel kinetics in cell context, 2b = architectural sufficiency for biology target) was specifically motivated by Wave 1 cellular validation findings. The decoupling matters for diagnostic clarity:

- **2a-pass / 2b-pass:** architecture sufficient, channels correct → ship.
- **2a-fail / 2b-fail:** channel kinetics broken (probably) → re-translate channels.
- **2a-pass / 2b-fail:** channel kinetics correct, architecture insufficient (condition 6) → architectural extension.
- **2a-fail / 2b-pass:** anomalous; would suggest pseudo-pass on biology target via wrong mechanism. Worth investigating.

Without the decoupling, Phase F would have produced a single ambiguous "Gate 2 FAIL" verdict that conflated channel kinetics with architectural sufficiency. The two-component design enabled clean classification of today's Phase F result as 2a-pass / 2b-fail = condition 6 candidate. Then the density sweep + Ca-coupling test ruled out parameter and minor-architectural-extension fixes, confirming condition 6.

The Wave 1 cellular validation methodology produced this design. Wave 1's findings (compartmental scaffold's per-channel parameters all reasonable, but cell-level dynamics under Mellem-style protocols failed) demonstrated the pattern: per-channel passes don't entail cell-level passes. That motivated the 2a/2b separation in Wave 2's plan.

### 7.8 Cellular biophysics methodology — cellular validation detects what channel-level cannot

**The methodology pattern:** per-channel validation can confirm channel kinetics are correct in isolation (Gate 1, Phase β proper). Cellular validation can confirm or refute architectural sufficiency in cell context (Gate 2, Phase F). **The two layers are independent.** Channel-level passes don't entail cell-level passes; cell-level fails don't necessarily mean channel-level fails.

This is the diagnostic signature that condition 6 was designed around. Today's empirical activation:
- All 7 channels pass per-channel validation (Gate 1).
- Cell-level fails Mellem target (Gate 2 component 2b).
- Density sweep shows it's not parameter-tunable (architecturally insufficient, not under-calibrated).
- Ca-coupling test shows it's not minor-extension-fixable (dynamic Ca-pool doesn't extend plateau).
- Conclusion: pre-Mellem-investigation, this was condition 6 = morphology fork warranted.
- Post-Mellem-investigation: the target was misattributed. The architectural insufficiency conclusion holds *as an engineering observation*, but the *biological framing* shifted.

Reusable methodological pattern for Wave 2+ cellular validation:
1. Run per-channel validation (Gate 1).
2. Run cell-level validation against per-cell phenotype target (Gate 2).
3. If cell-level fails: run density sensitivity to rule out parameter tunability.
4. If density sensitivity refuses rescue: run mechanism-extension test (e.g., add dynamic Ca-pool, add CICR, add compartmentalization) to rule out minor-extension fixes.
5. **Verify per-cell phenotype target against primary source** before concluding architectural insufficiency.
6. If all of above hold and architecture is genuinely insufficient: condition-6-style architectural fork.

Step 5 is the lesson from Mellem misattribution. Without it, the methodology produces well-supported wrong-target conclusions.

### 7.9 Connection to morphology integration question

c302's 607 cell morphologies are locally cloned at `~/Desktop/C-Elegans/simulation/upstream/c302/` and ready for integration. Whether morphology fork triggers depends on cellular phenotype targets:

- **Under option α** (Nicoletti's actual AVAL passive RC-like response): morphology fork doesn't trigger. Single-compartment is sufficient per Nicoletti's NEURON simulations.
- **Under different cellular targets** (e.g., RMD plateau dynamics per Mellem 2008 — which is the cell Mellem actually characterizes — or AWA spike mechanism per Liu 2018, or future targets): morphology integration may become relevant.

This is downstream Wave 2 / Wave 3 territory. The morphology assets are preserved; the architectural plan's framework supports the fork; what's missing is the empirical justification for any specific cell. The pattern is: when a Wave 2+ cell validation surfaces 2a-pass / 2b-fail under a biology-verified target that single-compartment provably can't reach, morphology fork becomes the candidate.

The Wave 1 compartmental scaffold (`compartmental_neurons.py`) and K_Ca patch sandbox (`compartmental_neurons_kca.py`) are NOT the same as the morphology fork — they're a different architectural extension (2-compartment soma+dendrite with axial coupling, vs. c302 multi-segment morphology). Both are extensions beyond single-compartment but at different scales. The compartmental scaffold may be useful for the simpler "leak isolation between two compartments" question; the morphology fork is for the spatial-detail question (per-segment channel densities, spatial coupling, dendritic Ca microdomains).

### 7.10 The compartmental scaffold remains shipped scaffolding

`compartmental_neurons.py` (15-cell roster, 2-compartment soma+dendrite with axial coupling, L-type Ca on dendrite for has_plateau cells) and `compartmental_neurons_kca.py` (sandboxed extension with h-inactivation + K_Ca patch + bulk Ca pool) remain on disk as built infrastructure. They are not currently deployed (Wave 2 didn't activate them; the architectural plan's morphology fork is on hold).

If a future cellular phenotype target surfaces that requires compartmental dynamics — e.g., a cell where Wave 2's per-cell phenotype validation produces 2a-pass / 2b-fail under a verified primary-source target — the compartmental scaffold is one architectural candidate to test before committing to the more expensive morphology fork. Per the Wave 1 finding, the scaffold's existing parameters (especially τ_d = 20 ms mammalian template) would need recalibration before deployment; the τ_d wall finding identifies the load-bearing parameter to address.

Production scaffold's known issues for future deployment work:
- v_rest = −65 mV mammalian template (scaffold default; Wave 1 sandbox overrode to −25 mV but did not propagate to production)
- τ_d = 20 ms mammalian-cortical timescale (way short of Goodman 1998 worm dendrite)
- v_ca_half = −30 mV (scaffold default; under v_rest = −25 produces tonic activation regime that K_Ca compensates for)
- h-equation form `dh/dt = (1-h)/τ_h - (m_inf × h)/(τ_h × 0.3)` with hardcoded 0.3 ratio (non-standard departure from textbook HH inactivation; produces h_ss = 0.3/(0.3+m_inf) plateau-asymptotic floor)

These are cellular-biophysics calibration items that would need to be addressed before any future compartmental deployment. None are currently on the critical path under option α.

### 7.11 Summary — what cellular biophysics contributes to synthesis

1. **Wave 1's directional prediction was right.** Single-compartment cannot reproduce sustained-bout dynamics matching Mellem-style 600 ms targets. The architectural plan's condition 6 was the right conditional fork to add.

2. **The specific target was wrong.** Mellem 2008 doesn't characterize a 600 ms plateau in AVA. The cellular validation work was being measured against a phantom target.

3. **Methodological pattern for paper 4:** verify per-cell phenotype targets against primary sources before drawing architectural conclusions from cellular validation. The pattern that surfaced today (validation methodology was sound; target was misattributed) generalizes.

4. **2a/2b decoupling in Gate 2 was load-bearing.** Without it, Phase F would have produced an ambiguous "Gate 2 FAIL" rather than a clean "channels work, architecture insufficient" diagnostic.

5. **Mechanistic finding:** SLO-1 in single-compartment with bulk Ca-pool is hyperpolarizing feedback for V. It cannot extend plateaus, only terminate them. This is structurally tied to the Wave 1 K_Ca dominance / τ_d wall observations.

6. **Compartmental scaffold remains shipped.** Available for future cellular targets that warrant it. Not on critical path under option α.

7. **The next cellular validation work** (IRK + UNC-103 translation, Phase F re-evaluation under option α) is bounded engineering. Per Nicoletti's AVAL phenotype, single-compartment with 5-channel AVA set should pass cleanly. The infrastructure built during Wave 2 (validation harnesses, NMODL pattern catalog, NEURONReference wrapper) handles it.

---

## Section 8 — Recommended next moves

Prioritized, with effort estimates and cellular-biophysics-emphasis annotations.

### 8.1 Immediate (this week or next work block)

**Complete IRK + UNC-103 translation.** ~1-2 days per channel given the F1-F17 pattern catalog. After translation, re-evaluate Phase F Component 2b against Nicoletti's actual AVAL phenotype (option α target). Estimated total: 3-5 days for translation + validation.

**Cellular-biophysics watchpoint:** UNC-103 specifically had GLOBAL→per-cell state issues per F2 in the NMODL gotcha catalog (per Phase α report §2 warnings). The translation pattern is documented but the per-channel implementation may surface additional patterns. If UNC-103 translation produces unexpected behavior in cellular context (vs per-channel passing in isolation), surface mid-flight per the cross-session methodology pattern.

**Authorize the AVAR upstream issue filing decision.** Draft at `wave2/artifacts/avar_upstream_issue_draft.md`. Either file or document local-only and proceed.

### 8.2 Engineering — short-term (2-4 weeks)

**Phase F re-evaluation under option α.** Once IRK + UNC-103 are translated, run Phase F Component 2b with the corrected 4-channel AVA cell + Nicoletti's actual AVAL phenotype target (passive RC-like, sustained-during-stimulus plateau, 200 ms slow rise, linear I-V). Expected to pass cleanly given Nicoletti achieves it in NEURON with same channel set. ~1-2 days work.

**Cellular-biophysics expectation:** under option α, the 4-channel AVA cell should produce a passive-RC-like plateau that matches Nicoletti's published AVAL trace within Layer A tolerance (Brian2-vs-NEURON deterministic match). The slow-rising 200 ms phase is set by the cell's effective τ_m × few; the sustained-during-stimulus plateau is set by the I_Ca/leak balance under sustained injection; the linear I-V is set by the channel-set's combined I-V relationship. None of these requires plateau-termination machinery (K_Ca, h-inactivation, Ca-pool dynamics). The 7-channel essential set's translation work is overkill for AVA but valid for AIY/RIM/RMD where those channels matter.

**Phase γ Gate 2 closure.** With component 2b passing under option α, declare Path A's cellular layer "production-grade" for AVA. Update gate-status documentation in `artifacts/checkpoints/`.

**Phase δ network integration kickoff.** Replace `graded_brain_h_kca.py`'s handcrafted Ca-K dynamics with the imported channel set for AVA (and other cells where the imported channels apply). Validate that network-level scenarios still run. Compare phenotypes against current LIFBrain + sample_004 baseline. ~1-2 weeks work.

### 8.3 Manuscript — parallel track (independent of Wave 2)

**Paper 2 manuscript prep.** Sample_004 + overrides + voltage-domain target framework is sufficient. Workshop track NeurIPS GRL or ICLR LMRL. Independent of Wave 2 trajectory.

Methodology contribution: voltage-domain target framework as Layer 1 closure replacement (per Session 1's §7.4). Manuscript structure draft, methods section, results, discussion.

**Cellular-biophysics contribution to paper 2:** voltage-domain targets are cellular biophysics targets. Sample_004's spike rate Δ targets translate to voltage trajectory features (dV/dt at stim onset, peak depolarization, plateau settling, post-stim recovery). The per-cell phenotype reproduction is the empirical foundation. Paper 2 doesn't need Wave 2's mechanistic channel kinetics, but the *framework* of "biology-direction-correct voltage features within tolerance of literature observation" is the cellular-biophysics contribution to paper 2 methodology.

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
- Session 3's α_Ca scale catch
- Session 3's d[Ca]/dt sign convention catch
- Session 3's analytical h_ss derivation averting unproductive sweep

Each is a worked example of the cross-session adversarial review pattern. Catalog them now.

**Cellular-biophysics meta-pattern for paper 4:** the 2a/2b decoupling, the requirement to verify per-cell phenotype targets against primary sources, and the methodology pattern (per-channel validation + cell-level validation + density sensitivity + mechanism extension test + biology citation verification) constitute a reusable framework for biophysical cellular validation at scale. The framework is independent of the specific simulator architecture; it generalizes to any compartmental modeling project where channel-level correctness is necessary but not sufficient for cell-level phenotype matching.

### 8.5 Architectural plan revision

Per §6.2. Deferred to paper 3 manuscript prep timing.

Cellular-biophysics annotation for the revision: the architectural plan's specific AVA targets (Mellem-style 20 mV / 600 ms references at multiple lines) should be replaced with Nicoletti's actual AVAL protocol parameters (1000 ms CC, sustained-during-stimulus passive plateau, ~200 ms slow rise, linear I-V). The plan's *strategic* commitments (Path 3A primary, gate-based progression, two-component Gate 2) remain valid; the *specific* cellular targets need re-grounding per option α.

### 8.6 Eventually — trajectory expansion

Wave 2 to AIY/RIM/RMD coverage given the channels translated (per §6.6). Wave 3+ CeNGEN-coupled densities. Wave 4+ peptidergic extension and receptor binding kinetics.

**Cellular-biophysics watchpoints for trajectory expansion:**

- **AIY/RIM cellular targets** need primary-source verification before structuring validation. The Mellem misattribution lesson directly applies: don't propagate cited targets without verifying primary source.
- **RMD is the actual Mellem 2008 plateau cell.** Wave 3+ RMD cellular validation will be the legitimate test of "compartmental architectural sufficiency for biology-grounded plateau dynamics" since RMD does have characterized plateau behavior. The Wave 1 compartmental scaffold and K_Ca patch sandbox become potentially relevant for that work.
- **Per-cell density transfer methodology** — when Phase F's component 2b cell construction transferred AIY-derived intensive densities (S/cm²) to AVA, that was a parameter-transfer-by-analogy choice. Future cell validations should either (a) use Nicoletti's per-cell densities directly when available, (b) document the analogy explicitly when transferring, or (c) replace with CeNGEN-derived per-neuron channel-expression data when Wave 4 makes that tractable.

---

## Section 9 — Methodology continuity notes

The cross-session adversarial review pattern that's been load-bearing today is documented here for continuity. Paper 4 will eventually formalize this; the practitioners' record is captured here.

### 9.1 Pre-flight pushback discipline

Sessions read prompts fully, surface concerns before starting work. **Today's catches:**

- Session 1 surfaced the bimodal hedging methodology concern before launching the 3-config validation — recommended the 3-config design itself rather than single-regime as originally specified.
- Session 1 surfaced the override-state mismatch (last night's LHS without overrides; today's validation with overrides) — recommended including sample_004 NO overrides as a control configuration.
- Session 2's audit prompt drafted a per-cell target table with citations (AVE/Wang 2020, AVB/Kawano 2011, etc.) that pre-flight grep showed do not exist in the canonical architectural plan. Pre-flight pushback caught it before deployment. **The pattern works in both directions.**
- Session 3 surfaced the d[Ca]/dt sign convention error in the Wave 1 K_Ca patch prompt — `−α_Ca × I_Ca` would be backwards; corrected to `+α_Ca × I_Ca`. Saved both Session 2 and Session 3 from implementing broken Ca pool dynamics.
- Session 3 surfaced the α_Ca scale concern in the same prompt — initial value 0.05 dimensionless/(pA·ms) would produce non-physiological [Ca] saturation; corrected to 0.0005.
- Session 3 surfaced the analytical derivation that tau_h-only sweep would not produce passes (h_ss = 0.3/(0.3+m_inf) is structural, not parameter-tunable). Averted ~1 day of unproductive sweep work.

### 9.2 Mid-flight surfacing of findings

Don't batch findings to end of work block. Surface as discovered. **Today's instances:**

- Mellem 2008 misattribution surfaced mid-flight in Mellem investigation work block; spec instructed pause-and-surface, work block paused before classification verdict.
- F11 misdiagnosis (F6 was wrong about hidden NMODL machinery) surfaced in Phase A diagnostic of Phase β run #2; produced architectural simplifications F12, F13 before downstream phases ran.
- F14 (h.v_init bug in NEURONReference) surfaced via SHL-1 7.3% systematic divergence in Phase C; fixed before Phase D-F ran.
- F15 (SS extraction window mismatch) surfaced post-F14; fixed in same work block.
- F16 (caintra1 ⇄ slo1iso unit-conversion) surfaced during Ca-coupling cell builder construction; fixed before Phase F 2b re-evaluation.
- F17 (caintra1 fca-scaling not in calibrated coefficient) surfaced during Ca-coupling sensitivity sweep; fixed before sweep results were interpreted.
- Session 3's "Trigger 2 met" in Wave 1 cellular validation Phase 1.5 — surfaced the τ_d wall finding before completing the full 6-cell × 4-variant Phase 2 matrix. AVA result was dispositive; full sweep would have produced same verdict on more cells.

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
18. Session 3 d[Ca]/dt sign convention catch
19. Session 3 α_Ca scale catch
20. Session 3 analytical h_ss derivation (averts unproductive sweep)

Plus the F1-F10 NMODL gotcha catalog from run #1.

The pattern: substantial findings catch each other across sessions when discipline holds. The investment in pre-flight pushback + mid-flight surfacing pays dividends because errors compound silently otherwise.

### 9.5 Methodology paper opportunity

Documenting the framework + case studies while context is fresh is the immediate-term methodology paper opportunity. Paper 4 (cross-session methodology) is independent of Wave 2 architectural commitment — the contribution is the pattern itself, not specific findings.

The cellular-biophysics-specific methodological pattern (cellular validation can detect architectural insufficiency that channel-level cannot; per-cell phenotype targets need primary-source verification; 2a/2b decoupling is load-bearing for diagnostic clarity) is a sub-section of paper 4 and connects to the broader cross-session adversarial review framework.

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

### 10.2 Session 3 Wave 1 cellular validation artifacts

- `scripts/brain/compartmental_neurons.py` — production compartmental scaffold (15-cell roster, 2-compartment soma+dendrite, L-type Ca on dendrite for has_plateau cells, hardcoded 0.3 ratio in h equation form). Built earlier in project; not deployed in production. Status: shipped scaffolding.

- `scripts/brain/compartmental_neurons_kca.py` — Session 3's Wave 1 sandbox extension. h-inactivation + bulk Ca-pool + K_Ca on dendrite. v_rest correction to −25 mV (overrides scaffold's −65 mV mammalian template). Three variants: base, h_only, h_kca. Imports CompartmentalParams + COMPARTMENTAL_ROSTER from compartmental_neurons.py (single source of truth for per-cell parameters). Status: shipped sandbox; never integrated.

- `scripts/brain/graded_brain_h_kca.py` — Session 2's parallel Wave 1 sandbox in single-compartment graded mode. Three variants: base, h_only, h_kca. Same parameter conventions as compartmental sandbox for cross-architecture comparability.

### 10.3 Wave 2 codebase

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

### 10.4 Wave 2 speculative work

- `scripts/brain/wave2/speculative/`
  - `gnn_architecture_sketch.md` — Variant A/B/C with diagrams
  - `training_data_feasibility.md` — data + parameter-count analysis
  - `comparison_framework.md` — Gate 2 extension for GNN validation
  - `prototype/` — minimal Variant A PyTorch prototype (training pipeline functional, 2-node test MAE 46 mV vs <5 mV target — informative-negative)
  - `multi_compartment_explicit.md` — sketch of architectural plan's morphology fork
  - `neuroml2_native.md` — sketch + previous-rejection confirmation
  - `x1_summary.md` — GNN summary
  - `speculative_summary.md` — overall summary

### 10.5 Upstream code (not in repo)

- `~/Desktop/C-Elegans/simulation/upstream/c302/` — c302 framework (cell morphologies, connectome readers, network templates). MIT license. Available for morphology fork if Wave 3+ surfaces cellular target requiring it.
- `~/Desktop/C-Elegans/simulation/upstream/ChannelWorm/` — ChannelWorm models (4 NeuroML channels).
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` — Nicoletti 2024 source code (24 .mod files, 9 cell simulation scripts). License unverified; ModelDB convention is academic-use-with-attribution.

### 10.6 Wave 1 LHS data (Session 1 morning work and earlier)

- `scripts/brain/artifacts/phase0_param_lhs_traces/` — 30 LHS samples × 6 scenarios × 5 seeds = 900 simulation traces (full 300-neuron rasters + body_xy + fsm_states + modulator_conc).

- `scripts/brain/artifacts/phase0_param_lhs_configs/manifest.json` — LHS sample configurations (22 parameters per sample).

- `scripts/brain/artifacts/phase0_param_lhs_synthesis.json` — original LHS analyzer numerical synthesis (the analyzer with the equal-weighted-mean bug).

- `scripts/brain/artifacts/phase0_corrected_weighting_synthesis.json` — corrected analyzer synthesis (Session 1's morning work).

- `scripts/brain/artifacts/phase0_layer1_validation_*.npz` — 3-config validation data (hedged + overrides, sample_004 + overrides, sample_004 NO overrides).

- `scripts/brain/phase0_param_lhs_analyze.py` — original analyzer (preserved unchanged).

- `scripts/brain/phase0_layer1_validation_analyze.py` — original 3-config analyzer (preserved unchanged).

- `scripts/brain/phase0_corrected_weighting_analyze.py` — corrected analyzer (Session 1's morning work).

### 10.7 Production environments

- `~/venvs/wave2-neuron/` — isolated venv for Wave 2 (Python 3.12.3, NEURON 9.0.1, Brian2 2.10.1, NumPy 2.4.4).

- `~/miniconda3/envs/ml/` — production brain conda env (Python, Brian2, MuJoCo, etc.). Untouched by Wave 2 work.

### 10.8 Project context

- `docs/claude-chat-context.md` (project root) — live context document. May be slightly out of date relative to today's findings (Mellem 2008 references, etc.). Should be updated at paper 3 prep time.

- `~/Desktop/website/personalwebsite/src/content/projects/c-elegans-multimodal.mdx` — public-facing project summary at rohitravi.com. Behind on Wave 2 details; updated at material-state-change cadence.

- `scripts/brain/artifacts/handoffs/session_1_handoff_2026-04-26.md` — companion handoff with parameter-analysis framing (Session 1's perspective).

---

## Closing notes

This handoff captures project state as of 2026-04-26 evening. Standalone-sufficient per Division C: a future session can pick up project work using only this document. For deep dives, the artifacts in §10 remain available.

The methodology pattern that ran throughout today (~17+ catches across pre-flight pushback, mid-flight surfacing, cross-session adversarial review) is load-bearing for paper 4 and should be documented while context is fresh.

The user's distinguishing context: NYU undergrad doing genuinely original research at the intersection of computational neuroscience, biophysics, and behavioral simulation. Both paper trajectories (paper 2 behavioral, paper 3 mechanistic) have clear paths forward. Wave 2's option α re-grounding preserves the channel-translation infrastructure investment while avoiding a 3-4 week morphology-fork commitment against a misattributed target. The next engineering work block (IRK + UNC-103 translation, ~3-5 days) is well-scoped and ready.

The cellular-biophysics framing's distinctive contribution: Wave 1's directional cellular-validation prediction was correct (single-compartment cannot reproduce sustained-bout dynamics matching Mellem-style targets), but the specific quantitative target was misattributed, so the architectural-extension conclusion that flowed from it (morphology fork warranted) was correct against the wrong target. The cellular validation methodology itself remains sound and reusable; the lesson is the citation hygiene step (verify per-cell phenotype targets against primary sources) needs to be a first-class part of the methodology pattern. The 2a/2b decoupling in Gate 2 design — directly motivated by Wave 1 cellular findings — was load-bearing for diagnostic clarity in today's Phase F evaluation. The compartmental scaffold and K_Ca patch sandbox remain shipped scaffolding for any Wave 3+ work block that surfaces a cellular phenotype warranting compartmental dynamics; under option α, they're not on the critical path.

Standing by.

— Session 3
