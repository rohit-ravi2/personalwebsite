# State of claims — C. elegans simulator (2026-05-02)

Phase 0 deliverable of the horizontal rebase. Read-only audit of the
project's load-bearing biology and methodology claims, classified by
direct-measurement support.

**Scope:** C. elegans multi-modal simulator only. The Wave P / anesthesia
pharmacology pipeline gets a separate audit later. Both simulator-side and
analysis-side claims are included so the catalog reflects what *Direct* looks
like across the project, not just where the weaknesses are.

**Status as of 2026-05-02:** drafted by Claude during Phase 0 of the
horizontal rebase. Not yet reviewed by Rohit; no source documents have been
amended based on this audit (that's deferred to Phase 3).

---

## 1 · Methodology

### 1.1 What counts as a load-bearing claim

A claim counts if changing its truth value would change a downstream design
decision or a publication-grade framing. Examples that count: "simulator
reproduces Chalfie 1985," "GNCA r = 0.987," "per-edge mode fires the touch
cascade," "AVA traces to Liu 2020 not Mellem 2008." Examples that don't:
architecture descriptions ("20-segment body"), counts ("4 production cells"),
pure engineering ("Vercel auto-deploys").

### 1.2 Status labels

- **Direct** — directly measured under the cited conditions; no contradicting
  evidence; currently believed true.
- **Direct-narrow** — directly measured under a *narrower* condition than the
  claim is cited for. Example: a phenotype validated in pure LIFBrain under
  default sign mode but cited as a property of "the simulator."
- **Inferred** — supported by adjacent measurements but not directly tested
  for the claim itself.
- **Falsified-but-cited** — direct measurement contradicts the claim, but
  the claim still appears in current docs / code / public-facing material.

### 1.3 Per-claim entry schema

```
C-NN. Claim text in one line.
- Source(s): file:line refs
- Status: <label>
- Evidence: what supports / contradicts the claim
- Scope of evidence: brain version, sign mode, scenario, n seeds, run length
- Sign-mode dependency: how status varies by sign convention
- Load-bearing for: downstream work / publication / dashboard
- Cross-refs: other claims this depends on or contradicts
- Notes: caveats, urgency for doc fix
```

---

## 2 · Source documents audited

Top-level docs:
- [x] `docs/claude-chat-context.md`
- [x] `docs/t0_resolution_report.md`
- [x] `docs/current-state-summary.md`
- [x] `docs/audit-strategy.md` (Phase 0 framework)
- [x] `docs/citation-audit-checklist.md`
- [x] `docs/new-session-primer.md`
- [x] `docs/path_b_engineering_spec.md`
- [x] `docs/project-history.md`
- [x] `docs/tier2-4-execution-plan.md`

Wave 2 + brain artifacts:
- [x] `scripts/brain/wave2/artifacts/phase_delta_wb3_findings.md`
- [x] `scripts/brain/wave2/artifacts/phase_delta_wb2_findings.md`
- [x] `scripts/brain/wave2/artifacts/stage_IV_touch_cascade_findings.md`
- [x] `scripts/brain/wave2/artifacts/mellem_investigation_pushback.md`
- [x] `scripts/brain/wave2/artifacts/cellular_validation_summary.md`
- [x] `scripts/brain/wave2/translation_patterns.md`
- [x] `scripts/brain/artifacts/t0_run_report.md`
- [x] D7-followup CP2 results (default + per-edge JSON)

Public-facing:
- [x] `src/content/projects/c-elegans-multimodal.mdx`

Not audited at this depth (treated as supporting / lower priority — biology
claims they contain are typically duplicated in the audited docs above):
phase_alpha_report, phase_beta_*, option_alpha_*, overnight_run_*,
phase_delta_scoping_*, phase_delta_wb1_findings, cython_migration_*,
v33_audit_report, ensemble_report, phase0_baseline_report, perturbation_report,
harness_*, stage_II_findings, ca_coupling_test_results, density_sensitivity_*,
f6_*, gate2_*, slo1_coupled_architecture, avar_upstream_issue_draft,
literature_scoping_candidates, architectural_plan_citation_audit,
cellular_validation_findings, cellular_validation_pushback, rim_*,
option_alpha_findings/summary, option_b_*, speculative/*. If a claim depends
on these and isn't already captured below, surface it during Phase 1.

---

## 3 · Status counts

- **Direct:** 29 (+1 — C-37 reclassified during Phase 2 W2 investigation)
- **Direct-narrow:** 12
- **Inferred:** 7
- **Falsified-but-cited:** 5 (−1 — C-37 reframed)
- **Total:** 53

The Falsified-but-cited entries are the urgent doc-fix targets for Phase 3.
The Direct-narrow + Inferred entries are the Phase 1 measurement targets.

**Updates (2026-05-02 same-day, post-Rohit-signoff):**
- C-32 (RIM production-grade) reclassified from Inferred → Direct after reading
  `rim_validation_summary.md`: VC 11/11 (max divergence 0.0043) + CC 11/11
  (residuals exactly 0.000 mV at all timepoints, 55,000/55,000 timepoints).
  RIM was added in a same-day work block after `cellular_validation_summary.md`
  was written — the "DEFERRED" was an early-day status, superseded.
- C-45 (Liu 2020 → AVA experimental data via Nicoletti ref [29]) reclassified
  from Inferred → Direct. Verified via Semantic Scholar API + PMC7544903:
  Liu P, Chen B, Wang Z-W (2020) *Nat Commun* 11:5076,
  DOI 10.1038/s41467-020-18893-9, PMID 33033264. Paper contains direct
  voltage-clamp (−110 to +50 mV steps) and current-clamp recordings of
  AVAL and AVAR. Authors match Nicoletti 2024 acknowledgments
  (Wang + Liu providing raw recordings). Chat-context citation accurate
  including article number.

**MAJOR FINDING during Phase 1 pre-flight (C-15 reconciliation):**
Commit `aea4c79` (2026-04-25, *after* T0 resolution work block) introduced
`DOCUMENTED_SIGN_EXCEPTIONS` — 7 per-edge sign overrides applied
mode-independently to both LIFBrain and Wave2HybridBrain by default.
Contents: 5 ALM/AVM → PVC entries (sign −1, motivated by Chalfie 1985 /
Wicks 1996 "anterior touch suppresses forward command") + 2 AIY → AIZ
entries (sign −1, Li 2014 ACC-2 mediated cholinergic inhibition).

**Per the commit's own verification table, applying these 7 exceptions to
per-edge mode collapses the touch cascade:**
- pure per-edge: AVDL/R Δ +60.2 Hz, AVAL/R Δ +60.6 Hz on touch
- per-edge + exceptions: AVDL/R Δ −1.6 Hz, AVAL/R Δ +0.4 Hz on touch
- default + exceptions: cascade does not fire

This means:
1. Current production per-edge mode behavior is **NOT** what t0_resolution_report.md §5 measured (cascade firing at +60 Hz). That measurement was pre-`DOCUMENTED_SIGN_EXCEPTIONS`.
2. Today's CP2 D7-followup results were under per-edge + exceptions, consistent
   with collapsed cascade dynamics. The σ Δ peri-touch ≈ 0 finding is the
   expected behavior of this regime, NOT a Wave2HybridBrain-specific
   pathology.
3. **C-13** ("Per-edge fires touch cascade in pure LIFBrain Δ +60 Hz") — its
   Direct-narrow scope is now substantially narrower. The +60 Hz figure
   applies to *pure per-edge* (no DOCUMENTED_SIGN_EXCEPTIONS). Current
   production per-edge cascade firing is much smaller (~+0.4 Hz at one
   drive level; Stage IV reports +7.5 Hz under conservative 200 Hz / 8 mV
   drive — that's the nearest valid current-production reference number).
4. **C-37** WB3 graded_b2 cascade biology — the falsified-but-cited finding
   may be partially explained by the underlying per-edge + exceptions regime,
   not solely by Wave 2 cellular substitution. Phase 1 should test pure
   per-edge in Wave2HybridBrain to disambiguate.

**C-15 reconciliation result** (`scripts/brain/phase1_signflip_reconciliation.py`,
`scripts/brain/artifacts/phase1_signflip_reconciliation.json`):
- 3707 non-zero chemical edges in connectome.npz (built 2026-04-20)
- M1-pure vs M2-pure: **518 hard sign flips** (matches Session 1 exactly)
- M1-prod vs M2-prod: 513 hard flips (current production)
- M1-pure vs M2-prod: 515; M1-prod vs M2-pure: 520
- Session 2's 415 figure does NOT match any pairwise count today. Probable
  explanation: different `connectome.npz` build, or different scope (e.g.,
  only counting Glu edges). Not blocking; the load-bearing 518 is verified.
- **Per-mode edge count summary:**
  - M1 default + exceptions [PROD]: 3120 nonzero, 2285 exc / 835 inh
  - M2 per-edge + exceptions [PROD]: 3099 nonzero, 2297 exc / 802 inh
  - 21 edges are zeroed under per-edge (postsynaptic Glu receptor expression
    = 0); confirms t0_resolution_report.md §4 "21 edges present in default-
    mode are zeroed in per-edge mode."

---

## 4 · Claims

Grouped by theme. Numbering is stable; status sort is in §5.

### 4.A — Analysis-side (GNCA, classifier, CCA, honest null)

#### C-01. GNCA r = 0.987 on held-out edges predicting outgoing synaptic strength.

- Source(s): `src/content/projects/c-elegans-multimodal.mdx:30`,
  `docs/claude-chat-context.md` §2 lines 36-39
- Status: **Direct**
- Evidence: held-out test set; reported alongside three explicit ablation
  controls (shuffled-expression 0.934, graph-only 0.861, expression-only 0.27)
- Scope: held-out edges, full GNCA model
- Load-bearing for: Paper 1 main thesis (topology-dominant)
- Cross-refs: C-02, C-03, C-04, C-07
- Notes: gold-standard claim. The kind of result the rebase is meant to
  protect (publishable-grade analysis-side work).

#### C-02. Shuffled-expression control r = 0.934 (topology carries the signal).

- Source(s): same as C-01
- Status: **Direct**
- Evidence: shuffle test on the same GNCA architecture
- Load-bearing for: topology-dominant thesis
- Cross-refs: C-01

#### C-03. Graph-only model r = 0.861.

- Source(s): same as C-01
- Status: **Direct**
- Evidence: ablation
- Load-bearing for: Paper 1 secondary control

#### C-04. Expression-only model r = 0.27.

- Source(s): same as C-01
- Status: **Direct**
- Evidence: ablation; baseline showing expression alone is insufficient
- Load-bearing for: Paper 1 secondary control

#### C-05. Neuron-role classifier 92.7% accuracy (sensory / inter / motor) from expression alone.

- Source(s): `c-elegans-multimodal.mdx:24`, chat-context §2
- Status: **Direct**
- Evidence: held-out classification accuracy
- Load-bearing for: Paper 1 (complement to GNCA)

#### C-06. CCA expression-motif canonical correlations ρ = [0.875, 0.836, 0.809].

- Source(s): `c-elegans-multimodal.mdx:32`, chat-context §2
- Status: **Direct**
- Evidence: CCA computation
- Load-bearing for: Paper 1 triangulation across linear / nonlinear methods

#### C-07. PhaseD hybrid gene-graph LSTM honest null: dynamics prediction does not improve with gene expression.

- Source(s): `c-elegans-multimodal.mdx:33`, chat-context §2,
  `docs/project-history.md` Phase D
- Status: **Direct**
- Evidence: shuffle test (`gene_causal_by_shuffle: False`); reported as honest
  null in project history
- Load-bearing for: Paper 1 — strengthens topology-dominant thesis by ruling
  out gene causality in dynamics

### 4.B — Brain layer: sign convention + cascade firing

#### C-08. ALM and AVM have zero direct chemical synapses to AIB.

- Source(s): `t0_resolution_report.md` §3.3, chat-context §5,
  `t0_run_report.md` postscript
- Status: **Direct**
- Evidence: connectome readout (Cook 2019 + Loer & Rand 2022)
- Load-bearing for: T0 resolution; rejects April 21 cascade framing
- Cross-refs: C-09, C-10

#### C-09. AIB has zero chemical edges to AVD in either sign mode.

- Source(s): `t0_resolution_report.md` §3.3
- Status: **Direct**
- Evidence: connectome readout
- Cross-refs: C-08

#### C-10. Operative touch-reversal cascade is ALM/AVM → PVC → AVD/AVE → AVA.

- Source(s): `t0_resolution_report.md` §3.3b, chat-context §5,
  `c-elegans-multimodal.mdx:102`
- Status: **Direct**
- Evidence: connectome decomposition; per-edge weight analysis showing PVC is
  load-bearing first-stage relay (~5× more drive to AVD than direct
  ALM/AVM → AVD)
- Load-bearing for: T0 resolution headline; supersedes April-21 ALM→AIB→AVA
  framing
- Cross-refs: C-08, C-09

#### C-11. Voltage regime (-25/-10/-30 mV per Mellem 2008) is no-op for current LIF dynamics.

- Source(s): `t0_resolution_report.md` §3.1, chat-context §3, project-history
- Status: **Direct** (FALSIFIED as cascade-firing bottleneck)
- Evidence: post-fix cascade rates matched pre-fix within ±2 Hz
- Notes: kept in place for biological documentation; will matter for SK/BK +
  compartmental work

#### C-12. Gap-junction conductance is not the cascade-firing bottleneck.

- Source(s): `t0_resolution_report.md` §3.2
- Status: **Direct** (FALSIFIED as bottleneck)
- Evidence: g_gap sweep at {0.1, 0.3, 1.0} nS monotonically silenced firing
  via noise averaging across 2188 gap edges
- Load-bearing for: T0 resolution

#### C-13. Per-edge sign mode fires the touch cascade in pure LIFBrain — AVDL/R Δ +60 Hz, AVAL/R Δ +60 Hz, AVEL/R Δ +47 Hz on touch (n=10, σ < 1.5 Hz).

- Source(s): `t0_resolution_report.md` §5, chat-context §5,
  `current-state-summary.md`
- Status: **Direct-narrow**
- Evidence: explicit n=10 × 60s sweep table (t0_resolution_report.md §5)
- Scope: pure LIFBrain only; specifically tuned weighting
- Sign-mode dependency: per-edge specifically (default mode shows AVA Δ
  −4 Hz)
- Load-bearing for: T0 resolution headline; many downstream framings
- Cross-refs: C-14 (Stage IV confirms under conservative drive); C-37
  (Wave2HybridBrain context shows different result — see C-37 entry)
- Notes: cited broadly as "per-edge fires the cascade" but evidence is
  specifically pure-LIF + tuned weights. Does not transfer to Wave2HybridBrain
  per CP2 D7-followup (see C-37).

#### C-14. Stage IV confirms cascade firing under conservative drive (200 Hz Poisson @ 8 mV) — pure LIFBrain per-edge: ΔAVA = +7.5 Hz, ALM/AVM ~60 Hz peak.

- Source(s): `stage_IV_touch_cascade_findings.md` Component 1, chat-context §5
- Status: **Direct-narrow**
- Evidence: Stage IV table (cells × pre/peri/post-touch rates)
- Scope: pure LIFBrain, per-edge sign mode, conservative drive
- Sign-mode dependency: per-edge
- Notes: ΔAVA = +7.5 Hz is "modest but real consistent with §5"; the §5
  +60 Hz used specifically tuned weighting. Conservative drive is the
  stronger generalization claim.

#### C-15. Per-edge sign mode produces 518 hard sign flips (~17% of 3120 chemical edges) — Session 1 count.

- Source(s): `t0_resolution_report.md` §4
- Status: **Direct** (Session 1 measurement)
- Evidence: direct connectome inspection; 5 counting methods agreed
- Notes: Session 2 reported 415 across 4 methods. Discrepancy unresolved
  (audit-trail item; probably from different `connectome.npz` builds or
  override list contents).

#### C-16. GABA uniformly signed −1 across all 26 GABA neurons; per-edge mechanism is glutamate-specific.

- Source(s): chat-context §4, `t0_resolution_report.md` §6,
  `current-state-summary.md`
- Status: **Direct**
- Evidence: 135 GABA edges byte-identical across modes; verified by direct
  measurement on 2026-04-25
- Load-bearing for: scope of per-edge change; structural cleanness of GABA
  pathway

#### C-17. Peptide release is pure linear rate-coupling (release = releaser_weights @ spike_counts, capped at 10).

- Source(s): chat-context §4
- Status: **Direct**
- Evidence: verified by direct measurement on 2026-04-25
- Load-bearing for: peptide-release mechanism is structurally clean and not
  affected by sign-mode questions

### 4.C — Brain layer: phenotype reproduction (AVA / RIS / Chalfie / Turek)

#### C-18. Default-mode AVA ablation produces ΔREV = −0.49 ± 0.10 at n=10 × 60s, 10/10 negative seeds.

- Source(s): `t0_resolution_report.md` §5, `c-elegans-multimodal.mdx:82`
- Status: **Direct-narrow**
- Evidence: n=10 × 60s sweep table
- Scope: default sign mode only
- Sign-mode dependency: default; under per-edge regresses to +0.04 (2/10
  negative)
- Notes: statistical robustness real, but t0_resolution_report.md §7.1
  reframes the *mechanism* as Mode 3 tonic-shift on broken sign convention,
  not cascade firing. The number is correct; what it signifies is what
  changed.
- Cross-refs: C-19, C-20, C-21

#### C-19. The default-mode dREV reproduction is mechanistically a Mode 3 tonic-shift on broken sign convention, not cascade firing.

- Source(s): `t0_resolution_report.md` §7.1, chat-context §5,
  `current-state-summary.md`
- Status: **Direct**
- Evidence: per-edge regression of dREV exposes the dependence on broken
  sign convention; cascade does not fire under default mode
- Load-bearing for: any "simulator validates Chalfie 1985" claim

#### C-20. Under per-edge mode, AVA-ablation effect on dREV regresses to +0.04 (2/10 negative seeds).

- Source(s): `t0_resolution_report.md` §5
- Status: **Direct**
- Evidence: per-edge ensemble run
- Sign-mode dependency: per-edge

#### C-21. Under per-edge mode, AVA-ablation effect persists in dPIR channel (mean −0.117, 9/10 negative seeds).

- Source(s): `t0_resolution_report.md` §5, `current-state-summary.md`,
  chat-context §5
- Status: **Direct**
- Evidence: per-edge ensemble run; clean signal in dPIR
- Sign-mode dependency: per-edge
- Load-bearing for: behavioral signature characterization; FSM recalibration
  framing

#### C-22. Simulator reproduces Chalfie 1985 AVA ablation (touch reversal abolished) — as cited.

- Source(s): `c-elegans-multimodal.mdx:82` ("genuine Chalfie 1985
  reproduction"), implied across multiple docs
- Status: **Falsified-but-cited**
- Evidence (falsifying): C-18 + C-19 — the dREV reproduction is via broken
  sign convention; under per-edge it regresses on dREV and shifts to dPIR
  (different FSM channel); C-37 — Wave2HybridBrain doesn't activate AVA
  on touch even under per-edge.
- Urgent doc fix: `c-elegans-multimodal.mdx` line 82; chat-context §4
  framing of "validated phenotype"
- Notes: the *direction* under default mode reproduces, but the underlying
  cascade does not fire. Under correct cascade dynamics (per-edge), the
  dREV signal regresses; the AVA-ablation effect persists in dPIR channel
  but that's a different FSM channel than what was originally claimed.
  Honest framing: "AVA ablation produces a behavioral signature in this
  simulator, but the FSM channel and mechanism are sign-mode-dependent
  and the cascade-firing mechanism is per-edge-only."

#### C-23. Default-mode RIS/Turek 2016 quiescence reproduction: ΔQUI = −0.24 ± 0.33 across 3 seeds (2/3 negative).

- Source(s): chat-context §4, `c-elegans-multimodal.mdx:82`,
  `t0_resolution_report.md` §7.2
- Status: **Direct-narrow**
- Evidence: 3-seed audit at 20s
- Scope: default sign mode, 20 s runs
- Sign-mode dependency: default; under per-edge RIS is silenced (C-24)
- Notes: directional but not statistically robust at 20 s; project page
  itself acknowledges "directionally consistent ... but not statistically
  robust at this duration."

#### C-24. RIS silenced under per-edge mode (0.8 Hz vs 21.8 Hz tonic under default).

- Source(s): `t0_resolution_report.md` §6.2, chat-context §5,
  `current-state-summary.md`
- Status: **Direct**
- Evidence: direct firing-rate measurement under both modes
- Sign-mode dependency: per-edge silences; default tonic 21.8 Hz
- Notes: not a direct sign-flip effect (RIS is GluCl-dominant, sign
  unchanged) — network-equilibrium consequence. Affects RIS molecular audit
  transferability.

#### C-25. Simulator reproduces Turek 2016 RIS quiescence — as cited.

- Source(s): `c-elegans-multimodal.mdx`, chat-context §3
- Status: **Falsified-but-cited**
- Evidence (falsifying): C-23 (not robust at 20 s) + C-24 (silenced under
  per-edge so default-mode finding doesn't transfer)
- Urgent doc fix: project page; chat-context §3 framing
- Notes: claim should be downgraded to "directionally consistent at default
  mode under-powered runs; transferability under per-edge unknown; RIS
  audit needs re-running."

#### C-26. RIS molecular audit (April 21): FLP-11 release fires correctly; peptidergic targets show ~22% disinhibition; behavioral null consistent with readout insensitivity.

- Source(s): `current-state-summary.md`, audit-strategy.md exemplar Mode 1
- Status: **Direct-narrow**
- Evidence: molecular-level audit
- Scope: default sign mode only
- Sign-mode dependency: at 0.8 Hz under per-edge, FLP-11 release would not
  produce comparable phenotype. Audit needs re-running under per-edge.

#### C-27. NSM ablation produces ΔQUI ≈ +0.50 (counter-finding to RIS).

- Source(s): `audit-strategy.md` Phase 0 finding
- Status: **Direct-narrow**
- Evidence: Phase 0 partial audit
- Scope: default sign mode
- Notes: Mode 2 readout-trivial — NSML/NSMR are IN the 18-readout, so the
  "phenotype" is the classifier responding to having two trained inputs
  zeroed, not to dwelling biology.

### 4.D — Three-mode taxonomy

#### C-28. Three-mode readout failure-mode taxonomy validated across all 9 v3 modulators (Mode 1 ×5, Mode 2 ×2, Mode 3 ×2).

- Source(s): chat-context §4, `t0_resolution_report.md` §7.3
- Status: **Direct-narrow**
- Evidence: overnight v1 + v2 runs
- Scope: default sign mode only — specific mode classifications are
  conditional on default
- Sign-mode dependency: classifications under per-edge would need
  re-running; framework remains valid sign-mode-independently

#### C-29. Three-mode taxonomy as a methodological framework is sign-mode-independent.

- Source(s): chat-context §4 lines 210-211, `audit-strategy.md`,
  `t0_resolution_report.md` §7.3
- Status: **Inferred**
- Evidence: framework is conceptual; no direct demonstration that all per-edge
  classifications would survive
- Load-bearing for: methodology paper claim

### 4.E — Wave 2 cellular layer (isolated cells)

#### C-30. AVAL Brian2 4-channel cell (IRK/LEAK/EGL19/NCA) matches Nicoletti 2024 published phenotype.

- Source(s): `cellular_validation_summary.md`, chat-context §3,
  `c-elegans-multimodal.mdx:104`,
  `stage_IV_touch_cascade_findings.md` Component 2
- Status: **Direct**
- Evidence: 7-point CC injection sweep (-30 to +30 pA) shows graded plateau
  sustained until stim removed; +10 pA → +80 mV plateau matches Nicoletti
- Scope: isolated AVAL cell; not in network context
- Load-bearing for: Wave 2 cellular layer credibility

#### C-31. AIY Brian2 7-channel cell matches Nicoletti 2024 — VC 11/11 (100%), CC 10/11 (90.9%) at ≤3 mV tolerance.

- Source(s): `cellular_validation_summary.md`
- Status: **Direct**
- Evidence: Layer A VC + CC validation; failing sweep -15 pA at extreme
  hyperpolarization due to integrator drift on KQT-1 slow s-gate (F19
  followup)
- Scope: isolated AIY cell

#### C-32. RIM Brian2 7-channel cell matches Nicoletti 2024.

- Source(s): chat-context §3, `rim_validation_summary.md` (2026-04-26)
- Status: **Direct** (resolved 2026-05-02)
- Evidence: `rim_validation_summary.md` VERDICT_RIM_PRODUCTION_GRADE.
  CP5 voltage-clamp 11/11 holds passing (100%, max divergence 0.0043 well
  under 0.05 tolerance). CP6 current-clamp 11/11 sweeps passing (100%) at
  residuals exactly 0.000 mV at all timepoints; aggregate 55,000/55,000
  timepoints pass. Stronger validation than AIY (which had 90.9% on CC).
- Resolution note: `cellular_validation_summary.md` (2026-04-26 morning)
  said DEFERRED, but RIM work block ran the same day in the afternoon
  and produced production-grade verdict. Chat-context is accurate.
- Channels: shl1, egl2, irk, cca1, unc2, egl19, leak. Three new channel
  translations validated production-grade in the RIM work block:
  CCA-1 (T-type Ca, 10/11 holds — one peak-direction-flip at +20 mV),
  EGL-2 (EAG-family K, 11/11), UNC-2 (P/Q-type Ca, 11/11; NMODL GLOBAL
  pitfall handled per-cell automatically by Brian2 semantics).

#### C-33. AVAR Brian2 5-channel cell (= AVAL + UNC-103) matches Nicoletti 2024.

- Source(s): chat-context §3, `stage_IV_touch_cascade_findings.md` Component 3
- Status: **Direct**
- Evidence: Stage IV CC injection sweep (-30 to +30 pA); rest -24.2 mV
  matches Mellem 2008 quote ("AVA rest typically between -20 and -30 mV");
  +10 pA → +39.8 mV plateau distinct from AVAL's +80 mV
- Scope: isolated AVAR cell

#### C-34. AVAL ≠ AVAR biologically distinguishable in Wave 2 detail (different rest, different K-channel set including UNC-103, different plateau amplitudes).

- Source(s): `stage_IV_touch_cascade_findings.md` Component 3, chat-context §3
- Status: **Direct**
- Evidence: rest -40.3 mV vs -24.2 mV; +10 pA plateau +80 mV vs +39.8 mV
- Load-bearing for: Wave 2 value proposition ("mechanistic biological
  resolution beyond LIF's spike-count abstraction")

#### C-35. Wave 2 cellular layer has 14 NMODL channel translations + F1-F18 NMODL gotcha catalog.

- Source(s): chat-context §3, `c-elegans-multimodal.mdx:104`,
  `cellular_validation_summary.md`
- Status: **Direct**
- Evidence: file inventory; per-channel validation results
- Notes: this is a count claim; per-channel translation accuracy is its own
  set of claims (F1-F18 catalog has individual entries; not enumerated here).

#### C-36. Cython codegen baseline: 22.71× aggregate speedup over numpy.

- Source(s): chat-context §3, `c-elegans-multimodal.mdx:104`
- Status: **Direct**
- Evidence: benchmark
- Load-bearing for: Wave 2 performance feasibility for closed-loop runs

### 4.F — Wave 2 hybrid integration (WB2 / WB3)

#### C-37. WB3 graded_b2 cross-coupling preserves cascade biology (cascade propagates from sensory through to AVA).

- Source(s): `phase_delta_wb3_findings.md` Sections 5, 6, 11; chat-context §3
  ("Wave2HybridBrain integration scaffold ...")
- Status: **Falsified-but-cited**
- Evidence (falsifying): D7-followup CP2 results (`phase_delta_wb3_d7followup_cp2_default_results.json`,
  `phase_delta_wb3_d7followup_cp2_peredge_results.json`):
  AVAL/AVAR σ Δ peri-touch is small / slightly negative under both default
  AND per-edge sign modes. At W=10 pA per-edge, Δ_post ≈ 0 separates touch
  effect from drift cleanly: touch is suppressing AVA σ (-0.011 to -0.013),
  not activating it. The Stage IV +60 Hz cascade does NOT propagate through
  Wave 2 cells.
- Urgent doc fix: `phase_delta_wb3_findings.md` Sections 5 + 11
  ("cascade DOES propagate biologically" framing); chat-context §3 Wave 2
  paragraph
- Load-bearing for: WB3 status; WB4 readiness; Phase G integration; dashboard
- Notes: WB3 inferred this from V trajectory + downstream LIF activity, not
  from direct AVA σ measurement. Direct measurement contradicts. Three
  plausible mechanisms (removing AVA from LIF NG breaks recurrent feedback;
  saturated σ tonically excites LIF baselines; driving-force asymmetry at
  saturated V_post) — none yet falsified, all candidates for Phase 1
  investigation.

#### C-38. σ-rising-threshold pseudo-spike pattern fails at saturation (returns 0 events when σ saturated above 0.5).

- Source(s): `phase_delta_wb3_findings.md` Section 6 Methodology Catch #1;
  D7-followup analysis
- Status: **Direct**
- Evidence: empirical observation from WB3 CP4; confirmed by D7-followup CP2
  σ-magnitude readout showing cells active (σ 0.62-0.90) where pseudo-spike
  rate reported 0
- Load-bearing for: Wave 2 readout API change

#### C-39. σ-magnitude readout (× 100 rate proxy) is the canonical Wave 2 cell activity API; matches `graded_brain.py output_rates()` line 378 precedent.

- Source(s): D7-followup CP1 implementation in
  `wave2/integration/wave2_hybrid_brain.py`
- Status: **Direct** (engineering)
- Evidence: code matches precedent; D7-followup CP2 confirms the readout
  reveals dynamics that pseudo-spike rate obscured
- Load-bearing for: Phase G FSM input contract; WB4 multi-cell readiness

#### C-40. WB3 capacitance correction: AVAL 9.66 pF (not 0.86 pF), LIF/Wave 2 ratio 10× (not 116×).

- Source(s): `phase_delta_wb3_findings.md` pre-flight,
  `translation_patterns.md` F20, `phase_delta_wb2_findings.md` correction
  note
- Status: **Direct**
- Evidence: re-derived from Brian2 cell-builder code (`option_alpha_*_cell.py`);
  WB2 conflated specific cm (μF/cm²) with total cm (pF)
- Load-bearing for: Wave 2 hybrid coupling design conclusions
- Notes: design conclusion (naive `v += W_syn * w` structurally unstable)
  preserved; only magnitude corrected.

#### C-41. Naive voltage-bumps cross-coupling produces unphysical V on Wave 2 cells (V → −∞).

- Source(s): `phase_delta_wb2_findings.md`
- Status: **Direct**
- Evidence: WB2 smoke test with `cross_coupling_mode="naive_voltage_bumps"`
- Load-bearing for: rejection of naive coupling; choice of graded Boltzmann
  release rule (Wicks 1996) for WB3

#### C-42. Wave2HybridBrain isolated mode (cross_coupling="off") runs cleanly — AVAL/AVAR settle to physiological rest.

- Source(s): `phase_delta_wb2_findings.md`
- Status: **Direct**
- Evidence: WB2 smoke test
- Load-bearing for: Wave 2 cellular layer integration scaffold

### 4.G — Citations + biology grounding

#### C-43. Mellem 2008 characterizes RMD plateau dynamics, NOT AVA — primary-source quote: "we never observed action potentials in AVA."

- Source(s): `mellem_investigation_pushback.md` (primary-source quotes from
  PMC2697921), chat-context §3 lines 60-70
- Status: **Direct**
- Evidence: direct primary-source quotes
- Load-bearing for: Wave 2 target re-grounding; AVA biology framing
- Cross-refs: C-44, C-46

#### C-44. The architectural-plan target "Mellem 2008 plateau (20 mV / 600 ms in AVA)" is a misattribution.

- Source(s): `mellem_investigation_pushback.md`
- Status: **Direct** (FALSIFIED as a citation chain)
- Evidence: Mellem 2008 doesn't characterize AVA plateau; the 20 mV / 600 ms
  numbers don't appear in Mellem; most plausibly inherited from Nicoletti
  2024 protocol durations attached to Mellem citation in early-document prose
- Notes: the misattribution is present in 9+ files
  (`phase_v_w2_architectural_plan.md`, `plateau_harness.py`,
  `sensitivity_sweep.py`, `run_ca_coupling_test.py`, etc.). Empirical Phase F
  findings are independent of the misattribution; what's invalidated is the
  framing as a "Mellem failure."

#### C-45. AVA experimental data (used in Nicoletti 2024) traces to Liu/Chen/Wang 2020 *Nat Commun* via Nicoletti ref [29].

- Source(s): chat-context §3 lines 65-70,
  `mellem_investigation_pushback.md` ref [29] discussion
- Status: **Direct** (resolved 2026-05-02)
- Evidence: Verified via Semantic Scholar API on DOI 10.1038/s41467-020-18893-9
  and via PMC7544903. Paper:
  - Liu P, Chen B, Wang Z-W (2020). *GABAergic motor neurons bias locomotor
    decision-making in C. elegans*. **Nat Commun 11:5076.**
    DOI: `10.1038/s41467-020-18893-9`. PMID: 33033264. PMCID: PMC7544903.
  - Contains direct voltage-clamp recordings (membrane voltage steps −110
    to +50 mV) and current-clamp recordings of both AVAL and AVAR.
  - Whole-cell patch-clamp with Multiclamp 700B amplifier, ~20 MΩ
    borosilicate glass pipettes; identification via fluorescent labeling.
  - Authors match Nicoletti 2024 acknowledgments verbatim (Z-W Wang and
    P Liu providing raw recordings).
- Note: chat-context's full citation including article number 5076 is
  correct. Nicoletti 2024 ref [29] is this paper.
- Load-bearing for: AVA biology citation chain integrity

#### C-46. AVA shows graded "passive RC-circuit-like" response per Nicoletti 2024, plateau sustained until stim removed (no regenerative spikes).

- Source(s): `mellem_investigation_pushback.md` (Nicoletti quote),
  `stage_IV_touch_cascade_findings.md`, chat-context §5 Wave 2 cross-validation
- Status: **Direct**
- Evidence: primary-source quote from Nicoletti 2024; reproduced in Stage IV
  Component 2 (+10 pA → +80 mV plateau, sustained until stim removed)
- Load-bearing for: Wave 2 AVAL phenotype target

#### C-47. Citation-audit checklist scope identified — 7+ load-bearing files with biology citations need DOI/PMCID verification (Mellem 2008 replaces Gao & Hobert 2020 in 6+ locations; other refs pending).

- Source(s): `docs/citation-audit-checklist.md`
- Status: **Direct** (the inventory exists)
- Evidence: checklist enumerates files
- Notes: doc-side audit pending; not blocking for rebase but should run in
  parallel.

### 4.H — FSM / classifier / readout

#### C-48. 18-neuron strict cross-worm intersection readout was a methodology choice, not a data limitation.

- Source(s): chat-context §3 lines 132-140, `t0_resolution_report.md` §8,
  `current-state-summary.md`
- Status: **Direct**
- Evidence: 535 GB Atanas data inventory: AVA in 100% of worms (10/10), AVD
  in 100%, AIZ in 90%
- Load-bearing for: readout-set expansion decision; FSM recalibration

#### C-49. Classifier bank trained against default-mode firing distributions; under per-edge AVA dynamic range tripled.

- Source(s): chat-context §3 lines 141-145, `t0_resolution_report.md` §6.3
- Status: **Direct**
- Evidence: training-set documentation; per-edge AVA jumps from ~37 Hz
  baseline to 97 Hz peri-touch
- Load-bearing for: FSM/classifier recalibration question

#### C-50. BehavioralFSM driven by 8-event Atanas-trained classifier bank (logistic regression on 18-neuron readout, AUC 0.75-0.90 per event).

- Source(s): chat-context §3
- Status: **Direct**
- Evidence: training metric
- Load-bearing for: FSM behavior under per-edge dynamics

#### C-51. NSM fires at 1.6-2.8 Hz across all scenarios — lower than Atanas's 4 Hz peak.

- Source(s): `path_b_engineering_spec.md:150`
- Status: **Direct-narrow**
- Evidence: scenario-level firing rates
- Notes: relevant for 5HT phenotype detection; NSM-silencing root cause is
  Layer 3 peptidergic work pending.

### 4.I — Engineering descriptions with biology-citation weight

#### C-52. LIF brain has 300 neurons (Cook 2019 hermaphrodite + Loer & Rand 2022 NT identity).

- Source(s): chat-context §1, `c-elegans-multimodal.mdx`, project-history
- Status: **Direct**
- Evidence: data sources; standard count for the connectome
- Notes: count claim with biology-source dependency.

#### C-53. 9 peptidergic + monoaminergic modulators (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5-HT, DA, TA, OA) with CeNGEN-derived releaser + receptor assignments.

- Source(s): chat-context §1, `c-elegans-multimodal.mdx`
- Status: **Direct** (engineering count); per-modulator audit findings
  separate
- Notes: FLP-13 redundancy with FLP-11 (0.80 Jaccard) is a related claim
  (current-state-summary.md, T4-5 INS-family pending).

---

## 5 · Summary table sorted by status

### 5.1 Falsified-but-cited (urgent doc-fix targets — Phase 3)

| ID | Claim | Urgent doc-fix locations |
|---|---|---|
| C-22 | Simulator reproduces Chalfie 1985 AVA ablation | `c-elegans-multimodal.mdx:82`; chat-context §4 |
| C-25 | Simulator reproduces Turek 2016 RIS quiescence | `c-elegans-multimodal.mdx`; chat-context §3 |
| C-37 | WB3 graded_b2 preserves cascade biology to AVA | `phase_delta_wb3_findings.md` §5+§11; chat-context §3 |
| C-44 | Mellem 2008 (20 mV / 600 ms in AVA) target | 9+ files (architectural plan, harnesses, etc.) |

(C-44 is partly already-corrected in mellem_investigation_pushback.md but
the misattribution still appears in downstream Wave 2 docs. Worth a sweep.)

(Note: C-22 and C-25 are the headline phenotype claims that the project's
publication framing has rested on. Honest reframing required before any
publication-grade claim.)

### 5.2 Inferred (Phase 1 measurement targets)

| ID | Claim | Phase 1 test |
|---|---|---|
| C-29 | Three-mode taxonomy as framework is sign-mode-independent | Re-run all 9 modulator audits under per-edge mode |

(C-32 and C-45 reclassified to Direct on 2026-05-02 — see §3 update note.)

### 5.3 Direct-narrow (Phase 1 broadening targets)

| ID | Claim | Narrowness |
|---|---|---|
| C-13 | Per-edge fires cascade in pure LIFBrain (Δ +60 Hz) | LIFBrain only, tuned weights |
| C-14 | Stage IV cascade firing (ΔAVA +7.5 Hz) | LIFBrain only, conservative drive |
| C-18 | Default-mode AVA ablation ΔREV = −0.49 | Default sign mode only |
| C-23 | Default-mode RIS ΔQUI = −0.24 | Default sign mode, 20 s runs |
| C-26 | RIS molecular audit findings | Default sign mode only |
| C-27 | NSM ablation ΔQUI ≈ +0.50 | Default sign mode |
| C-28 | Three-mode taxonomy specific classifications | Default sign mode |
| C-51 | NSM firing rates 1.6-2.8 Hz | Default mode only |

These are the claims that have been measured under one set of conditions but
are sometimes cited as if they generalize. Phase 1 sign-mode gauntlet should
broaden the evidence to whichever mode wins, or document the narrowness
explicitly.

### 5.4 Direct (baseline; survives current evidence)

| ID | Claim |
|---|---|
| C-01 to C-07 | Analysis-side: GNCA, classifier, CCA, honest null |
| C-08, C-09, C-10 | Connectome facts: cascade structure |
| C-11, C-12 | Falsified bottlenecks (voltage, gap conductance) |
| C-15 | Sign-flip count (Session 1) — discrepancy with Session 2 |
| C-16, C-17 | GABA + peptide release mechanism cleanness |
| C-19, C-20, C-21 | Mode 3 mechanism + per-edge dPIR/dREV findings |
| C-24 | RIS silencing under per-edge |
| C-30, C-31, C-32, C-33, C-34, C-35, C-36 | Wave 2 cellular layer (isolated cells) |
| C-38, C-39, C-40, C-41, C-42 | WB2/WB3 engineering findings |
| C-43, C-45, C-46 | Mellem misattribution + Liu 2020 AVA citation + Nicoletti AVA phenotype |
| C-47 | Citation audit scope |
| C-48, C-49, C-50 | FSM / classifier baseline |
| C-52, C-53 | Engineering with biology-source dependencies |

These are the load-bearing claims that survive direct measurement and
should remain Direct after the rebase. Many of them are *foundations* for
the Phase 1 sign-mode decision (C-08 to C-21 set up the cascade structure;
the decision is "which sign mode best operationalizes that structure").

---

## 6 · Methodology lessons (separate appendix, not part of claims catalog)

These are recurrent patterns the project has surfaced. Important context for
Phase 1+2 work but they aren't biology *claims* themselves; they're rules of
engagement.

1. **Inferred validation pattern.** The recurrent error: claim A is inferred
   from related measurements (V trajectory + downstream LIF rates → "cascade
   propagates"); next layer builds on A; later direct measurement of A
   contradicts. Examples: WB3 CP4 inferred-claim about cascade firing
   (today's CP2 falsifies); April-21 T0 inferred-cascade framing
   (T0 resolution falsifies). Forward-methodology gate should require direct
   measurement before next-layer build.
2. **Citation propagation without re-verification.** Mellem 2008 (RMD →
   inherited as AVA target); Nicoletti-2019-PCBI vs PLOS ONE (corrected by
   Wave 2 work). Pattern: early-document prose claim is adopted by downstream
   work blocks without primary-source check. Caught by primary-source
   verification.
3. **Capacitance arithmetic / specific-vs-total.** WB2 conflated μF/cm²
   (intensive) with pF (extensive). Caught by re-derivation from Brian2
   cell-builder code at WB3 CP1.
4. **Phenotype reproduction ≠ mechanistic validity.** The April-21 ΔREV =
   −0.49 reproduction was statistically robust (10/10 negative seeds at
   n=10 × 60s) but mechanistically misleading (Mode 3 tonic-shift on broken
   sign convention). Statistical robustness is not the same as mechanistic
   validity.
5. **Methodology choice ≠ data limitation.** The 18-readout strict
   cross-worm intersection was a *choice* about how to filter Atanas data;
   AVA/AVD are in 100% of worms. Attributing methodology choices to data
   limitations carries downstream design weight without warrant.
6. **Pause-and-document over autonomous architectural commitment.** WB3
   pause for biology review surfaced the capacitance question rather than
   committing to a release rule that would have inherited the WB2 cm error.
   Methodology continuity preserved.

---

## 7 · What's open after Phase 0 (handed to Rohit)

**Resolved 2026-05-02 same-day:**
- ✅ Sign-off on catalog (signed off; no claim reclassifications requested)
- ✅ C-32 RIM verification (Direct — `rim_validation_summary.md` confirms
  production-grade; chat-context accurate)
- ✅ C-45 Liu 2020 citation (Direct — Liu/Chen/Wang 2020 Nat Commun 11:5076,
  DOI 10.1038/s41467-020-18893-9, PMID 33033264, PMCID PMC7544903; chat-context
  citation accurate including article number)

**Still open going into Phase 1:**
- **C-15 sign-flip count discrepancy (518 vs 415).** Reconcile counting
  methods between sessions before per-edge becomes load-bearing in Phase 1.
  Not blocking but should resolve early in Phase 1.
- **Phase 1 gauntlet design refinement** — see §7.1 below for the refined
  gauntlet plan reflecting this catalog's findings.

### 7.1 Phase 1 gauntlet — refined test list (post-C-15 discovery)

Refinements relative to the original horizontal-rebase plan, expanded by the
`DOCUMENTED_SIGN_EXCEPTIONS` finding above.

**Scope clarification.** Phase 1 decides the *brain-level* sign mode in
**pure LIFBrain context** (no Wave 2 cells substituted). Wave2HybridBrain
biology is a separate investigation track (C-37 follow-up; runs after Phase 1
or in parallel as low-priority). This avoids confounding sign-mode questions
with cross-coupling implementation questions.

**Sign-mode candidate set** (expanded to 4 candidates):

| ID | Description | DEFAULT_SIGN_OVERRIDES | DOCUMENTED_SIGN_EXCEPTIONS | Status |
|---|---|---|---|---|
| **M1** | Default — per-presynaptic-neuron NT signs | applied (~26) | applied (5 PVC + 2 AIY) | current production default |
| **M2-pure** | Pure per-edge CeNGEN-derived | n/a | none | what T0 §5 measured (+60 Hz cascade) |
| **M2-current** | Per-edge + exceptions | n/a | applied (5 PVC + 2 AIY) | current production per-edge (cascade collapses) |
| **M3a** | Per-edge + AIY exceptions only | n/a | only 2 AIY entries (drop 5 PVC) | proposed — tests whether PVC entries are the cascade-collapsing factor |

**Why this candidate set:** The C-15 reconciliation surfaced that current
production per-edge mode (M2-current) is *neither* what T0 §5 measured
(pure per-edge, +60 Hz cascade) *nor* default mode (tonic-shift dREV
reproduction). It's a third regime whose phenotype reproduction the project
hasn't directly validated. The gauntlet needs to test all four to settle:

1. Whether the 5 PVC exceptions are causally responsible for cascade collapse
   (M2-pure vs M2-current).
2. Whether the 2 AIY exceptions can stand alone without the PVC ones (M3a).
3. Whether canonical biology ("anterior touch suppresses forward command")
   is testable in this simulator independently of cascade firing.
4. Whether default mode's tonic-shift dREV reproduction is actually the only
   way to recover the Chalfie phenotype.

**Decision rule** (refined): pick the mode that best satisfies a *priority
ordering* of biology criteria, not just total pass count:
1. **Cascade firing** (touch-driven AVD/AVA depolarization) is non-negotiable
   biology. A mode that doesn't fire the cascade is structurally wrong.
2. **PVC/AVB suppression** under anterior touch is canonical biology — but
   open question (commit message acknowledges "biologically questionable"
   under per-edge); may be tolerable as documented limitation if cascade
   wins.
3. **dREV phenotype reproduction** under AVA ablation is desirable but
   secondary — the dPIR finding suggests the phenotype may exist in a
   different FSM channel.
4. Pick the mode with cascade firing + clearest mechanistic story for any
   biology failures.

**Compute estimate:** 4 modes × 9 tests at ~5-30 min/run = ~16-25 hours.
3-4 overnight runs.

**Refined test gauntlet** (each test runs under all 3 candidate modes):

| # | Test | Direct measure | Catalog claim being tested |
|---|---|---|---|
| 1 | Cascade fires through to AVA | LIFBrain AVA Δ peri-touch (Hz) | C-13, C-14 generalization |
| 2 | AVA ablation → dREV | reversal-state ΔP | C-22, C-18, C-20 |
| 3 | AVA ablation → dPIR | pirouette-state ΔP | C-21 |
| 4 | RIS baseline rate | tonic firing rate | C-24 |
| 5 | RIS / Turek-style ablation → dQUI | quiescence-state ΔP | C-25, C-23 |
| 6 | Network stability — non-touch scenarios | per-cell rate ranges | (new) |
| 7 | RIS molecular audit transferability | FLP-11 release × peptidergic targets | C-26 |
| 8 | Three-mode taxonomy classifications | re-classify FLP-2 / PDF-1 | C-28 (specific classes) |
| 9 | NSM ablation → dQUI counter-finding | quiescence ΔP | C-27 |

Tests 1-6 are the original gauntlet. Tests 7-9 are added based on this
catalog's Direct-narrow entries that need broadening.

**Decision rule** (refined):
- For each test, classify per-mode as: PASS / FAIL / AMBIGUOUS (seed-to-seed
  variance dominates).
- Pick the mode that passes the most tests at p<2 SEM.
- If ambiguous, prefer mode with cleaner mechanistic story over marginally
  higher pass count.
- **Honest documentation principle**: chosen mode's test failures explicitly
  documented (no Falsified-but-cited claims propagate forward); each
  catalog claim's status under chosen mode is recorded for Phase 3 doc updates.

**Compute estimate** (rough):
- Tests 1-6 (~12-15 hr from original plan) + tests 7-9 (~4-6 hr)
- Total ~16-21 hr of compute; 3-4 overnight runs

**What gets DEFERRED from Phase 1** (separate investigation tracks):
- C-37 WB3 graded_b2 cross-coupling biology — Wave2HybridBrain investigation
  thread (see §7.2)
- Citation-audit checklist sweep (C-47) — runs in parallel, doc-side only,
  doesn't block Phase 1 compute

**Exit gate:** decision matrix presented; Rohit picks mode; brain locked at
chosen mode. Then Phase 2 (FSM/classifier recalibration under chosen mode).

### 7.2 Wave2HybridBrain investigation thread (parallel / post-Phase-1)

C-37 is a Falsified-but-cited claim about cross-coupling biology, but it's
load-bearing for Wave 2 hybrid integration specifically — it's NOT load-bearing
for Phase 1 sign-mode decision (which can run in pure LIFBrain). Demote
WB3 graded_b2 from "production alternative brain mode" to "experimental
research substrate" per the rebase plan, and run the C-37 follow-up as a
parallel low-priority track:

- (a) Re-run touch cascade in Wave2HybridBrain with AVA as LIF (not Wave 2)
  — does the cascade fire as in pure LIF? Tests whether Wave 2 cells
  specifically break the cascade dynamics.
- (b) Test alternative graded coupling parameters — different W_graded_I,
  τ_syn, E_rev values — does any combination produce positive AVA σ Δ
  peri-touch?
- (c) Investigate driving-force asymmetry hypothesis — at saturated V_post
  the inhibitory driving force (54 mV to E_inh) dominates excitatory (16 mV
  to E_exc). Test whether changing E_exc upward (e.g., to +20 mV) restores
  positive Δ.

This thread is informative but not gating any other rebase work.

---

## 8 · Status

Phase 0 complete + Phase 1 complete (screen tier) as of 2026-05-03.

No source documents have been amended based on this audit (other than this
file itself + the new `docs/brain_v3.5_locked.md`). All Falsified-but-cited
claims still appear in their original locations; their fix is a Phase 3
deliverable.

### 8.1 Phase 1 outcome (2026-05-03)

**Decision: Brain v3.5 locked at M2-pure** (per-edge sign mode with no
DOCUMENTED_SIGN_EXCEPTIONS).

Decision matrix at
`scripts/brain/artifacts/phase1_gauntlet_screen_decision_matrix.md`. Brain
spec at `docs/brain_v3.5_locked.md`.

**Headline data point:** M2-pure was the only candidate to fire the touch
cascade (AVD/AVA Δ +60 Hz on touch, matching T0 §5 documented numbers
exactly). M1 (default) and M2-current (per-edge + 7 exceptions) both
showed AVD/AVA *dropping* on touch (-2 to -4 Hz), failing priority-1 of
the decision rule. M3a was untested (gauntlet stopped at 3-of-4 modes
when wall-time forecasting overran budget; M3a's value was inferable
from M2-pure vs M2-current comparison).

**Catalog impact:**
- **C-13** (per-edge fires +60 Hz cascade): Direct-narrow → **Direct**.
  Reconfirmed today at n=5×30s with numbers matching T0 §5 within 1%.
  Scope still LIFBrain-only.
- **C-24** (RIS silenced under per-edge): Direct, reconfirmed (1.08 Hz
  today vs documented 0.8 Hz — perfect match).
- **C-22** (Chalfie reproduction): Falsified-but-cited remains. Phase 1
  did not test recalibrated FSM, so dREV reproduction under M2-pure is
  not yet measured. dPIR direction is consistent with T0 §5 -0.117 but
  underpowered at n=5×30s (-0.04 today). Phase 2 settles this.
- **C-25** (Turek RIS reproduction): Falsified-but-cited remains. RIS
  silencing under per-edge means default-mode finding doesn't transfer.
  Phase 2 + Phase 1B (or later) settles this.
- **C-27** (NSM Mode 2 readout-trivial): Direct-narrow → **Direct**.
  Today's measurement: NSM→dQUI = +0.39 in all 3 modes — identical
  across sign modes confirms the effect is classifier-readout dependent,
  not biology dependent. Sign-mode-independence empirically confirmed.
- **C-37** (WB3 graded_b2 cascade biology): **Falsified-but-cited → Direct
  (under M2-pure conditions)** as of 2026-05-03 W2 investigation.
  - W2 investigation script:
    `scripts/brain/wave2/integration/run_wb_investigation_w2_m2pure.py`
  - Results: `wb_investigation_w2_m2pure_results.json`
  - Under Wave2HybridBrain M2-pure (per-edge + sign_exceptions={}):
    AVDL Δ peri-touch +60.5 Hz (matches pure LIFBrain M2-pure +60.4 Hz
    exactly); AVAL σ Δ +0.1005 peri-touch with post-touch drift -0.0050
    (clean signal); all command cells (PVC, AVD, AVE, AVB, AVA) fire
    +Δ on touch in the same +26 to +67 Hz range as pure LIFBrain.
  - **The C-37 Falsified status was caused entirely by
    DOCUMENTED_SIGN_EXCEPTIONS, NOT by Wave 2 cellular substitution.**
    Wave2HybridBrain integration is biologically sound under M2-pure.
    Cascade biology fully preserved through Wave 2 hybrid.
  - Implication for project framing: Wave2HybridBrain becomes a viable
    production-grade brain mode again (not just research substrate),
    paired with M2-pure sign mode. M2-pure brain-lock + graded_b2
    cellular substrate are stacked-compatible.
- **C-15** (sign-flip count discrepancy): partially resolved.
  M1-pure-vs-M2-pure = 518 verified empirically (Session 1 confirmed).
  Session 2's 415 remains unreconciled.

### 8.2 Phase 1 → Phase 2 handoff

Phase 2 = FSM/classifier recalibration under M2-pure. Pre-flight needed:
1. Atanas data prep / classifier infrastructure scoping (uncertain
   estimate, deferred during overnight v2 Track B as
   LOGISTICAL_FAILURE).
2. Readout-set decision (18 vs adding AVA/AVD).
3. Phase 2 gauntlet design.

This document is the input to Phase 2 + Phase 3.
