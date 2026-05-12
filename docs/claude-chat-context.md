# C. elegans multi-modal simulator — Claude.ai Projects context pack

**Purpose of this file:** Upload to a Claude.ai "C. elegans simulator"
Project as knowledge. Every chat in that project will inherit this
context, so Rohit can discuss ideas without re-explaining the setup.

**Maintenance:** Live doc. When the simulator's state changes
materially (new phenotype validated, brain recalibrated, new paper
angle), update this file and re-upload.

Current as of **2026-04-25**, post-T0 resolution work block.
(Previous version: 2026-04-21, post-T0 validation run — superseded
by today's diagnostic findings; see §5 for what changed.)

---

## 1 · One-paragraph identity

A connectome-constrained, modulation-aware, embodied *C. elegans*
digital twin. 300-neuron Brian2 LIF brain derived from the Cook 2019
hermaphrodite wiring + Loer & Rand 2022 neurotransmitter table,
driving a 20-segment MuJoCo body through a 5-state behavioural FSM.
Nine peptidergic/monoaminergic modulators (FLP-11, FLP-1, FLP-2,
NLP-12, PDF-1, 5-HT, DA, TA, OA) with CeNGEN-derived releaser +
receptor assignments and literature-grounded diffusion lengths.
Published at https://www.rohitravi.com/projects/c-elegans-multimodal.
Code at https://github.com/rohit-ravi2/personalwebsite (private).

## 2 · Current thesis

*Local connectomic topology in C. elegans is largely self-
determining, while gene expression acts as a modulatory refinement
layer rather than a primary architect of wiring.*

Evidence supporting this from the analyses side:
- **GNCA** (Graph Neural Cellular Automata) predicting synaptic
  strength: r = 0.987 on held-out edges. Shuffled-expression
  control: r = 0.934 (topology carries the signal). Graph-only:
  0.861. Expression-only: 0.27.
- **Neuron-role classifier** 92.7% accuracy (sensory/inter/motor)
  from expression alone.
- **CCA** between expression space and motif space: ρ = [0.875,
  0.836, 0.809].
- **Honest null**: PhaseD hybrid gene-graph LSTM showed
  `gene_causal_by_shuffle: False` — dynamics prediction doesn't
  improve with gene expression on this dataset.

## 3 · Architecture — current live state

Four layers, each with a 'legacy' and an 'upgraded' path:

### Brain
- **Default**: `LIFBrain` (Brian2, 300 neurons, chemical + gap
  synapses, W_syn=0.8 mV tuned for balanced regime).
- **Alternative**: `GradedBrain` (T1a Kunert-Graf 2014, σ(V)
  continuous release + L-type Ca plateau on 14 command neurons).
- **Scaffold for v4 (Wave 1, sandbox)**: 15-neuron compartmental pool
  (soma+dendrite + L-type Ca + slow h-inactivation) in
  `compartmental_neurons.py`. Compiles in Brian2 but NOT production-
  wired into ClosedLoopEnv (per Phase δ scoping primary-source check).
  Earlier framing of "calibrated against Mellem 2008 voltage-clamp
  data" was based on the Mellem-2008-as-AVA-anchor framing that
  Wave 2's Mellem investigation later identified as misattributed —
  Mellem 2008 (Mellem/Brockie/Madsen/Maricq, *Nat Neurosci* 11:865-867)
  characterizes RMD plateau dynamics, NOT AVA. AVA experimental
  data (rest, IV curves, current-clamp) traces to Liu/Chen/Wang 2020
  *Nat Commun* 11:5076 (DOI 10.1038/s41467-020-18893-9, ref [29] in
  Nicoletti 2024 PDF; raw recordings shared informally with
  Nicoletti per her acknowledgments). Wave 2 cellular layer (below)
  supersedes the v4 sandbox scaffold for AVA-class cells.
- **Wave 2 cellular layer (production-grade)**: 4 Brian2 cell
  builders matching Nicoletti 2024 published phenotypes to
  5-decimal-place agreement on full VC + CC validation panels.
  AVAL (4 channels: IRK/LEAK/EGL19/NCA), AIY (7 channels including
  SLO1+EGL19, KQT1, SHL1; eca=127.59 mV per F18 multi-USEION-ca
  asymmetric ion_style override), RIM (7 channels including CCA1,
  EGL2, UNC2; eca=60 mV symmetric), AVAR (5 channels = AVAL +
  UNC103). 14 NMODL channel translations + F1-F18 gotcha catalog +
  cython codegen baseline (22.71× aggregate speedup over numpy).
- **Wave2HybridBrain integration — production-grade under M2-pure**
  (C-37 resolved per rebase Phase 2). The original CP2 D7-followup
  inferred that Wave 2 cellular substitution itself broke the
  recurrent-feedback dynamics that produce §5's +60 Hz cascade.
  The W2 investigation (`scripts/brain/wave2/integration/run_wb_investigation_w2_m2pure.py`)
  refuted that inference: under M2-pure (per-edge +
  `sign_exceptions={}`), Wave2HybridBrain fires the cascade
  identically to pure LIFBrain M2-pure (AVDL Δ +60.5 Hz vs pure
  LIF +60.4 Hz; AVAL σ Δ +0.1005 peri-touch with ~zero post-touch
  drift). The CP2 finding (AVA σ slightly negative under per-edge +
  DOCUMENTED_SIGN_EXCEPTIONS) was a sign-mode question, not a
  cellular-substitution question — the 7-entry exceptions registry
  flips ALM/AVM→PVC edges and collapses the cascade upstream of
  Wave 2 cells, just as it does in pure LIF under the same
  exceptions. Wave2HybridBrain + M2-pure adds biophysical
  resolution beyond LIF without breaking cascade dynamics. WB3
  findings amendment in `scripts/brain/wave2/artifacts/phase_delta_wb3_findings.md`
  postscript records the corrected interpretation; original CP2
  measurement preserved as historical record.
- **Sign-convention mode** (constructor flag
  `use_per_edge_glu_signs`): two glutamate sign-assignment modes
  exist in the codebase.
  - **Default** (`False`, current production): per-presynaptic-
    neuron NT sign (Glu = −1) with ~26 hand-picked overrides for
    sensory + interneuron Glu sources where iGluR is known to
    dominate (DEFAULT_SIGN_OVERRIDES in `lif_brain.py`).
  - **Per-edge** (`True`): postsynaptic-receptor-derived signs
    from CeNGEN iGluR/GluCl expression ratios. The connectome
    artifact contains a precomputed `W_chem_per_edge` matrix
    (CeNGEN-derived). Switching to this mode flips ~518 chemical
    edges (14% of total) for Glu sources targeting iGluR-dominant
    neurons.
  - **Current default is per-neuron mode pending PVC/AVB handling
    resolution.** See §5 for what's open.
- **Voltage regime** (in place since 2026-04-25): LIF parameters
  patched to v_rest=−25 mV, v_thr=−10 mV, v_reset=−30 mV per
  Mellem 2008. Preserves the 15 mV rest-to-threshold gap so LIF
  dynamics are coordinate-translated unchanged. Patch is a no-op
  for current LIF dynamics; kept for biological documentation
  and for future SK/BK + compartmental work that depends on
  absolute voltages.

### Sensory
- **Default**: `sensory_injection.py` — direct Poisson spike trains
  into target neurons per preset ("touch_anterior" → ALM/AVM @ 180 Hz).
- **Upgraded (P1 #8)**: `sensory_transduction.py` — five ODE
  cascades (ASE GCY-22/cGMP/TAX-4, AWC ODR-10/Gα_i/cGMP-drop
  OFF-cell, ASH OSM-9/OCR-2+TRPA-1, AFD GCY-8/18/23, ALM MEC-4/10).
  Each cascade has τ_rise, τ_decay, and adaptation constants with
  literature citations. Selected via `sensory_mode="transduction"`.
- **P0 #3**: `AerotaxisSensory` adds URX/AQR/PQR (high-O2) + BAG
  (O2-off + CO2-on) via Gray 2004 / Zimmer 2009. Drives an
  `aerotaxis` scenario with a 7% → 21% O2 linear gradient.

### FSM (behavioural state)
- **Default**: `BehavioralFSM` driven by 8-event Atanas-trained
  classifier bank (logistic regression on 18-neuron strict
  cross-worm intersection readout, AUC 0.75-0.90 per event).
- **Upgraded (P1 #4)**: `ActivityFSM` reads command-neuron firing
  rates directly (AVA/AVE for reversal, AVB/PVC for forward,
  SMDV/RIV for omega, RIS for quiescence, NSM for feeding dwell)
  and triggers on z-scored deviation from a 20 s EMA baseline,
  with 2 s warmup window. Selected via `fsm_mode="activity"`.
- **Readout-set capability** (data-derived, not yet exercised):
  the strict 18-neuron cross-worm intersection that defines the
  current readout was a methodology choice from the Atanas data
  pre-processing, not a data limitation. AVA is in 100% of Atanas
  worms (10/10), AVD in 100%, AIZ in 90%; the strict intersection
  filter excluded canonical command interneurons that ARE
  identifiable in the data. Readout expansion to include AVA/AVD
  is now clearly possible from existing data. Deferred until per-
  edge / FSM recalibration questions resolve.
- **Calibration regime caveat**: the 18-readout classifier bank
  was trained against firing patterns from default-mode
  (per-neuron sign) dynamics. Under per-edge mode, AVA's dynamic
  range tripled and the AVA-ablation effect shifts FSM channels
  (dREV → dPIR; see §5). Recalibration is an open question.

### Body
- **Default**: `wormbody.xml` — 20 segments, 19 hinge actuators,
  CPG-driven per-state.
- **Scaffold for v4**: `wormbody_v2.xml` — same skeleton + 80
  quadrant (DL/DR/VL/VR) actuators + sites, plus
  `motor_innervation.json` (540 sparse neuron→muscle weights from
  White 1986 + Cook 2019 + Pereira 2015 rules). Quadrant actuators
  are position-typed on hinge joints (MuJoCo semantics bug — they
  should be `<muscle>` on `<tendon><spatial>`); muscle-driver
  code that reads motor rates and writes to the 80 actuators
  doesn't exist yet.

### Dashboard / UI
Built in React + Astro + Tailwind. Lives at
`src/components/react/CelegansDashboard.tsx` (~3.7k LOC). Renders:
- 20-seg body with state-coloured glow + trail + D/V compass
- 3D 300-neuron brain (rotatable, NT-filter chips, search)
- Raster view (clickable, NT-colored)
- 2D arena (chemotaxis food patch OR aerotaxis O2 gradient)
- 9-modulator concentration strip (heatmap + line overlay)
- FSM timeline with event-fire carets + stim labels
- Event-probability line plot with time ticks
- CeNGEN gene-expression polar chart on locked-neuron popover
- Synthetic Ca trace + ego-network + firing-rate history on
  locked neuron
- Live stats + circuit badges + activity-FSM z-role badges
- URL-hash sharable links, CSV/PNG export, ?-help overlay

## 4 · What's validated (honest)

**AVA / Chalfie 1985 reproduction — Falsified-but-cited per horizontal
rebase (2026-05-08).** The originally-cited reproduction ("AVA ablation
abolishes reversal under touch, ΔREV = −0.49 ± 0.10 at n=10 × 60s, 10/10
negative seeds, default sign mode") was real as a statistical result but
ran via a Mode 3 tonic-shift mechanism on the broken default sign convention,
not via circuit cascade dynamics. Phase 1 of the rebase locked the brain
at M2-pure (per-edge, sign_exceptions={}) — the only sign mode firing the
+60 Hz touch cascade — and Phase 2 retrained the full readout stack
(A2-balanced 21-cell classifier with leave-one-worm-out CV, M2-pure
calibration, recalibrated FSM thresholds). Under the recalibrated stack at
default tier (n=10×60s), AVA → dREV = +0.229 ± 0.137 (2/10 negative) under
M2-pure — wrong direction, no Chalfie reproduction.

The "dPIR channel-shift" hypothesis (T0 §5 originally reported dPIR =
−0.117, 9/10 negative seeds at n=10×60s under legacy stack) is **refuted**
under the recalibrated stack: dPIR = −0.005 ± 0.005 (1/10 negative) at the
same statistical power. The legacy dPIR finding was a stack-dependent
artifact, not a real channel-shifted phenotype. The Chalfie 1985 phenotype
is currently NOT reproduced by this simulator under correct cascade
dynamics. See `docs/brain_v3.5_locked.md` and the catalog at
`docs/state_of_claims_2026-05-02.md` (C-21, C-22).

**Alternative finding under M2-pure (Direct, awaiting literature precedent
verification):** AVA-ablation produces a robust dFWD signal at −0.302 ±
0.102 (7/10 negative seeds, Cohen's d ≈ 0.93) at n=10×60s. Forward-locomotion
suppression rather than reversal abolition. Biologically interpretable via
AVA-AVB gap-junction coupling (Wang 2020) but not the textbook Chalfie 1985
phenotype. Headline-result decision deferred pending literature precedent
verification.

**RIS / Turek 2016 quiescence pathway — Falsified-but-cited per horizontal
rebase.** Original default-mode finding ΔQUI = −0.24 ± 0.33 across 3 seeds
(2/3 negative) was directionally consistent but underpowered. Under M2-pure
RIS is silenced at 1 Hz (vs 21 Hz default), and Phase 2.5 default-tier
gauntlet measured RIS → dQUI = −0.007 ± 0.026 (4/10 negative) — null at
full power. The recalibrated activity_fsm thresholds further confirm
quiescent-state detection is structurally compromised under M2-pure
(RIS at 1 Hz baseline / 2.5 Hz stim peak gives z_stim = 1.10, below the
2.5 threshold). The April 21 RIS molecular audit (FLP-11 release fires
correctly, peptidergic targets show ~22% disinhibition) was conducted under
default mode and does not transfer to M2-pure.

**Three-mode readout failure-mode taxonomy** (validated across 9
v3 modulators in overnight v1 + v2 runs): Mode 1 readout-blind ×5
(FLP-11, FLP-1, NLP-12, OA, TA), Mode 2 readout-trivial ×2 (5HT,
DA), Mode 3 readout-cascade ×2 (FLP-2, PDF-1). The taxonomy as
a methodological framework is sign-mode-independent. The
specific Mode classifications were established under default
sign convention and may shift under per-edge.

**GABA + peptide release mechanism (structurally clean).** GABA
uniformly signed −1 across all 26 GABA-releasing neurons via the
per-presynaptic-neuron sign field; per-edge mechanism is
glutamate-specific by design (CeNGEN iGluR/GluCl ratios are
glutamate-receptor-specific). 135 GABA edges are byte-identical
across both sign modes. Peptide release is pure linear rate-
coupling (release = releaser_weights @ spike_counts, capped at
10). Both verified by direct measurement on 2026-04-25.

Pipeline works end-to-end: stim → brain → classifier → FSM → body →
MuJoCo → state distribution → dashboard JSON.

## 5 · T0 resolution (2026-04-25, supersedes April 21 framing)

**Status:** the T0 cascade-failure question is resolved at the
architectural level. The original April 21 framing — that the
v3 LIF brain failed to propagate touch through ALM → AIB → AVA
and that synaptic weight calibration was the prerequisite fix —
is now obsolete. Both the cascade description and the fix
category were wrong.

Full diagnostic record: `docs/t0_resolution_report.md`.
Original April 21 framing preserved as historical record at
`scripts/brain/artifacts/t0_run_report.md` (with dated
postscript referencing this resolution).

### What was wrong with the April 21 framing

1. **Wrong cascade.** AIB has zero chemical edges to AVD in
   this connectome (Cook 2019 + Loer & Rand 2022). ALM/AVM also
   have zero direct chemical edges to AIB. The "ALM → AIB → AVA"
   pathway never existed in the simulator's wiring. AIB is in
   the chemotaxis pirouette circuit, not the touch reversal
   circuit.

2. **Wrong fix category.** The cascade failure was not a
   synaptic weight calibration problem. It was a sign-assignment
   problem. The simulator's default per-presynaptic-neuron sign
   convention treated glutamate edges to iGluR-dominant cells
   as inhibitory.

### What the operative cascade actually is

> **ALM/AVM → PVC → AVD/AVE → AVA**

PVC is the load-bearing first-stage relay. ALM/AVM glutamate
inputs to PVC sign as inhibitory under default convention (PVC
drops on touch, removing its dominant cholinergic excitation of
AVD — that's the original "command interneurons decreased" finding
from April 21). Under per-edge sign convention (PVC iGluR-
dominant per CeNGEN, ratio 9.6×), the same edges sign excitatory
and PVC fires UP on touch. PVC then drives AVD/AVE
cholinergically (~5× more drive to AVD than direct ALM/AVM
glutamate to AVD); AVD/AVE drive AVA cholinergically; recurrent
positive-feedback within the command pair amplifies all command
neurons to ~97 Hz coherently.

### What today's diagnostic block established

Three falsifications and one architectural fix:

- **Voltage regime is not the bottleneck** (FALSIFIED).
  Patching v_rest from −65 to −25 mV per Mellem 2008 was a
  coordinate translation; LIF dynamics unchanged. Voltage fix
  kept in place for biological documentation.
- **Gap conductance is not the bottleneck** (FALSIFIED).
  Increasing g_gap monotonically silenced the network because
  gap junctions average noise across coupled cells.
- **AIB-cascade premise was wrong** (FALSIFIED at the connectome
  level). Read the wiring before continuing to calibrate it.
- **Per-edge sign mode resolves the cascade firing**
  (CONFIRMED). Under `use_per_edge_glu_signs=True`, AVDL/R fire
  Δ +60 Hz on touch, AVAL/R Δ +60 Hz, AVEL/R Δ +47 Hz, n=10
  with seed-to-seed variance under 1.5 Hz.

### What the resolution implies for the original phenotype claim

The simulator did reproduce the *direction* of Chalfie 1985
under default mode (AVA ablation → reduced reversal). It did so
through a Mode 3 tonic-shift mechanism on the broken sign
convention's elevated AVA baseline, not through circuit cascade
firing. Under per-edge mode (correct cascade), the AVA-ablation
effect on dREV regresses to null.

The originally-reported dPIR channel-shift escape hatch (T0 §5
text said "persists on dPIR, mean −0.117, 9/10 negative seeds")
was **refuted by the horizontal rebase Phase 2.5 default tier**:
under recalibrated stack at n=10×60s, dPIR = −0.005 ± 0.005
(1/10 negative). That finding was a legacy-stack artifact. See
the rebase deliverables at `docs/state_of_claims_2026-05-02.md`
(C-21 reclassified Direct → Falsified-but-cited) and
`docs/brain_v3.5_locked.md`. The Chalfie 1985 phenotype is
currently NOT reproduced by this simulator under correct cascade
dynamics.

The reproduction was technically valid as a statistical result
under default mode. It was mechanistically misleading as a
biology claim. Statistical robustness (10/10 seeds at n=10) is
not the same as mechanistic validity.

**Alternative finding under M2-pure (Direct, awaiting literature
verification):** AVA → dFWD = −0.302 ± 0.102 (7/10 negative,
Cohen's d ≈ 0.93) at n=10×60s. Forward-locomotion suppression
rather than reversal abolition. Biologically interpretable via
AVA-AVB gap-junction coupling but not the textbook Chalfie 1985
phenotype.

### What's open after the resolution

1. **PVC/AVB over-activation under per-edge mode.** PVC fires
   Δ +60-70 Hz and AVB fires Δ +51-57 Hz on touch. Canonical
   biology has anterior touch suppressing forward locomotion —
   biologically questionable. Two interpretations remain open:
   (A) CeNGEN expression-vs-function mismatch (PVC has iGluR
   receptors but the ALM synapse onto PVC may be functionally
   GluCl-mediated), or (B) canonical biology more nuanced than
   textbook (PVC excitation on anterior touch may be defensible).
   Neither is yet falsified.
2. **FSM/classifier recalibration question.** Refined by dPIR
   finding: the question is not just "retrain to recover dREV"
   but characterize what behavioral signature AVA-ablation
   produces under correct cascade dynamics, and which FSM
   channels best reflect the circuit-level Chalfie phenotype.
3. **RIS silencing under per-edge.** RIS goes from 21.8 Hz tonic
   (default) to 0.8 Hz (per-edge) — a network-equilibrium
   effect, not a direct sign flip. RIS molecular audit findings
   from earlier in the project don't transfer to per-edge mode
   without re-running.

### What the T0 resolution does NOT settle

- Whether per-edge becomes production default or stays opt-in
  (depends on resolving PVC/AVB and FSM recalibration).
- Plateau dynamics (Wave 2 cellular layer addresses this for
  AVAL/AVAR/AIY/RIM specifically — see §3 Wave 2 entry. AVAL/AVAR
  Wave 2 phenotype is RC-passive graded plateau matching Nicoletti
  2024, not regenerative; this is the actual published biology and
  differs from the previously-assumed "20 mV / 600 ms plateau in
  AVA" framing carried over from the Mellem-as-AVA-anchor
  misattribution).
- Membrane time constant τ (10 ms in simulator vs 50-200 ms in
  worm computational models).

### Stage IV cross-validation (overnight 2026-04-27/28)

Independent test under conservative drive (200 Hz Poisson on
ALML/ALMR/AVM @ 8 mV) confirms §5 cascade-firing claim:
ALM/AVM activate ~60 Hz peak; AIB/PVC/AVD/AVA all show
ΔHz > 0; ΔAVA = +7.5 Hz (modest but real, consistent with
§5 above which used specifically tuned weighting for the +60 Hz
result). The cascade fires under per-edge mode regardless of
stim parameters within reasonable range.

The §5 quoted "broken cascade" framing — which appeared in
some derived documents — is a v3-LIF-activity-mode-FSM-specific
phenotype, not a LIF-general failure. Per-edge LIF + per-neuron
LIF differ on whether the cascade fires; Stage IV confirms the
per-edge resolution holds.

Wave 2 cellular layer's contribution is orthogonal: AVAL and
AVAR are biologically distinguishable under Wave 2 detail
(AVAL rest -40 mV / +80 mV plateau at +10 pA; AVAR rest -24 mV
/ +40 mV plateau at +10 pA — different cells, different leak,
different K-channel set including UNC-103) where they are
identical-except-for-indices in LIF. Wave 2's value
proposition is "mechanistic biological resolution beyond LIF's
spike-count abstraction," not "fix to a broken cascade."

## 6 · Roadmap (prioritised — substantial reordering 2026-04-25)

The 2026-04-21 roadmap put synaptic weight calibration as the
load-bearing first step (Tier 2 #1) on the assumption that the
ALM → AIB → AVA cascade was an undercalibrated wire. T0
resolution (§5) established that the cascade was a sign-
convention problem, not a weight problem, and the operative
cascade isn't even through AIB. The roadmap below reflects the
post-resolution priorities.

### Block 1 — Resolve T0 follow-on questions (~3-6 weeks)

These are the open questions surfaced by per-edge mode validation.
They sit between the T0 resolution and any production decision
about per-edge as default.

1. **PVC/AVB handling under per-edge mode.** Investigate whether
   PVC's iGluR-dominant CeNGEN expression accurately predicts
   functional dominance at the ALM/AVM synapses, or whether
   per-edge needs targeted overrides. Literature dive +
   possibly per-edge override sweep. Two interpretations open
   (CeNGEN expression-vs-function mismatch, or canonical biology
   more nuanced) — neither yet falsified.
2. **FSM/classifier recalibration under per-edge dynamics.**
   Three sub-questions: (a) does AVA-ablation under correct
   cascade produce the Chalfie phenotype through dPIR? (b)
   would recalibrated thresholds re-route the signal to dREV?
   (c) is the existing 18-readout architecture fundamentally
   incompatible with per-edge dynamics? Bank retraining is the
   technical prerequisite (deferred during overnight v2 Track B
   as LOGISTICAL_FAILURE).
3. **Network-stability scan under per-edge mode** for non-touch
   scenarios (osmotic_shock, food, chemotaxis, aerotaxis,
   spontaneous). Per-edge changes ~14% of chemical edges; touch
   scenario validates one regime, others may differ.
4. **RIS silencing investigation.** Why does RIS go from 21.8 Hz
   to 0.8 Hz under per-edge? Network-equilibrium consequence of
   broader sign changes, not a direct sign flip on RIS itself.
   Affects RIS molecular audit transferability.
5. **Per-edge re-runs of audited phenotypes.** RIS molecular
   audit, three-mode taxonomy, Mode 3 modulator results all
   conducted under default mode. Need re-running to determine
   which findings transfer.

### Block 2 — Production decision + scaffold completion (~1-2 mo)

6. **Production sign-mode decision.** Per-edge as default, opt-
   in, or hybrid (curated per-edge override list). Depends on
   Block 1 #1 and #2 outcomes.
7. **Wave 2 cellular layer integration** (substantively complete
   2026-04-26-28): the Wave 1 v4 sandbox `compartmental_neurons.py`
   was superseded by Wave 2's NEURON→Brian2 channel-translation
   pipeline. Production-grade now: AVAL, AIY, RIM, AVAR (4 cells,
   5-decimal-place Nicoletti 2024 match). Network integration
   (Wave2HybridBrain class exists) PAUSED at WB3 — capacitance
   mismatch + release-event rule are open biology questions
   (`scripts/brain/wave2/artifacts/phase_delta_wb2_findings.md`,
   3 questions for review). Earlier framing of "calibrate plateau
   dynamics against Mellem 2008" carried the misattribution that
   Wave 2's Mellem investigation later corrected — Mellem 2008
   characterizes RMD plateau, not AVA. The corrected target for
   AVA is Nicoletti 2024's published phenotype (RC-passive graded
   plateau, NOT regenerative; 4-channel set; references Liu/Chen/
   Wang 2020 as upstream source). Independent of T0 resolution;
   plateau dynamics are a separate active-conductance question
   that per-edge sign mode does not address.
8. **Muscle driver**: new `muscle_driver.py` reads motor-neuron
   rates, applies innervation matrix, writes to v2 actuators.
   Replace position actuators with real `<muscle>` on
   `<tendon><spatial>`.
9. **Sensory transduction calibration** against published ΔF/F
   traces (Chalasani 2007 AWC, Suzuki 2008 ASE, Hilliard 2005
   ASH). Independent of T0.

### Block 3 — Validation + publication-grade claims (~2-3 mo)

10. **Ensemble audit with corrected brain** under chosen sign
    mode and recalibrated FSM. Does AVA-ablation reproduce
    Chalfie cleanly through some FSM channel? Does RIS/Turek
    clear 2·SEM under per-edge?
11. **Aerotaxis phenotype** validation. Does the sim navigate
    toward preferred O2 (12%)?
12. **Parameter uncertainty quantification**: 200-point Latin-
    hypercube sample over dominant parameters, propagate to
    phenotype statements.

### Block 4 — Unique-in-field architectural work (~6 mo)

13. **CeNGEN-conductance coupling**: scale per-neuron ion-
    channel conductance by CeNGEN TPM. Closes the connectomics-
    transcriptomics loop architecturally.
14. **Readout-set expansion** if Block 1 #2 indicates the
    18-neuron readout is a bottleneck. AVA/AVD are in 100% of
    Atanas worms — the strict cross-worm intersection that
    produced the 18-set was a methodology choice, not a data
    limitation.
15. **WebGPU-compiled brain** for live in-browser sim (10 kHz
    on a 4060 Ti is plausible).
16. **Pheromone / multi-worm** environment.

### Items now lower priority (originally Tier 2 #1)

- **Synaptic weight calibration as ActivityFSM unblock** — the
  original framing is obsolete. Cascade fires at +60 Hz under
  per-edge without weight tuning. Specific weights (notably
  ALM/AVM → PVC) may still need fine-tuning if PVC over-
  activation under per-edge is judged a real bug; that's a
  narrower question covered by Block 1 #1.

## 7 · Tech stack + constraints

**Sim backend:** Brian2 2.9 + MuJoCo (Python) + numpy. Conda env at
`~/miniconda3/envs/ml/bin/python`.

**Frontend:** Astro 4.16 + React 18 + Tailwind. Bundle pre-renders
JSON scenarios (no live Brian2 in browser yet).

**Deploy:** Vercel auto-deploy on push to main.

**Local compute:** RTX 4060 Ti, 8 GB VRAM. Real-time ratio 2.6× for
LIF brain + MuJoCo body (30 s sim = 80 s wall).

**Storage:** `/home/rohit/Desktop/website/personalwebsite/` is the
Astro repo. Brain code under `scripts/brain/`, generated JSONs
under `public/data/`, artifacts under `scripts/brain/artifacts/`.

**Hard constraints:** No PhD (industry route, health-driven).
No wet-lab work. NJ geographic anchor. Theoretical +
computational only.

## 8 · Current publication plan

**Paper 1** — multi-modal analysis: *eLife* / *PLOS Computational
Biology* / *Network Neuroscience*. Topology-dominant /
gene-modulatory framing. Anchors: GNCA r=0.987, NT-classifier
92.7%, CCA ρ≈0.87, honest null on gene-causal LSTM. Draft in
progress.

**Paper 2** — methods paper: NeurIPS GRL / ICLR LMRL workshop
track. GNCA architecture for connectome-constrained synaptic
prediction. Could include the T0 falsification-test methodology
(§5) as a secondary contribution.

**Potential paper 3 (if Block 2 + Block 3 of §6 roadmap land):**
the first *C. elegans* simulator with validated ActivityFSM +
transduction cascades + compartmental dynamics + CeNGEN-
conductance coupling. Single-author accessible.

## 9 · Who I am (for context in chats)

NYU undergrad, Data Science major with Philosophy minor. Industry-
track, not PhD. Working toward AI roles that bridge technical and
philosophical domains. Strong linear algebra / calculus /
probability foundations. Intellectual interests span
neuroscience, consciousness studies, quantum computing, and
Vedantic non-dualism.

**Working style preferences** (important):
- Plan first for non-trivial work, execute second.
- Rigor over brevity. Full-credit-quality reasoning when explaining.
- Push back on speculative proposals before elaborating; ask for
  falsifiability.
- No wet-lab bio work ever. Only theoretical + computational.
- Vedanta / non-dualist framings welcome when they sharpen
  technical work; avoid ideological overlays.
- Direct, no-sugarcoat assessments. Honest scope labels (shipped
  vs. scaffolded vs. calibration-pending).

## 10 · How to use this project in chats

**Questions that should cite this doc:**
- "Where are we on the simulator?"
- "Is X phenotype reproduced?"
- "What would a v3.5 brain look like?"
- "Where does CeNGEN data plug in?"
- "What's in the publication plan?"

**Questions that should defer to rohitravi.com/projects/c-elegans-multimodal:**
- "What does the dashboard show?"
- "Can I see a live example of X?"

**Questions that should ask for current state:**
- "What changed since §5 was written?" (this file drifts)
- Anything about commit hashes, specific file contents, or
  exact code paths.

## 11 · Suggested Claude.ai Project setup

**Project name:** `C. elegans multi-modal simulator`

**Custom instructions (paste into the Project's instructions field):**

> You are helping Rohit Ravi think through an in-silico *C. elegans*
> simulator project combining connectomics, transcriptomics, and
> embodied simulation. Knowledge files describe the current state,
> architecture, validated phenotypes, open issues, and roadmap.
>
> Working style:
> - Be direct and non-sugarcoating. Label shipped vs. scaffolded
>   vs. calibration-pending work honestly.
> - Push back on speculative claims — ask for falsifiability before
>   elaborating.
> - Prefer rigor over brevity when explaining technical concepts.
> - No wet-lab suggestions; theoretical + computational only.
> - Vedanta / non-dualist framings are welcome when they sharpen
>   technical work, but avoid ideological overlays in straight
>   scientific discussion.
>
> When asked about the simulator state, cite §5 (T0 resolution)
> as the load-bearing recent finding for the LIF brain: two
> glutamate sign-assignment modes (per-neuron NT default,
> per-edge CeNGEN-derived). Default mode reproduces the Chalfie
> 1985 phenotype on dREV via a tonic-shift mechanism but does
> not fire the touch cascade. Per-edge mode fires the cascade
> correctly (AVD/AVA Δ +60 Hz on touch) but breaks the dREV
> phenotype reproduction (AVA-ablation effect shifts to dPIR
> channel). PVC/AVB over-activation under per-edge is an open
> question; FSM/classifier recalibration under per-edge dynamics
> is an open question. Per-edge is currently opt-in pending
> resolution. Full record at `docs/t0_resolution_report.md`.
>
> Also cite §3 Wave 2 cellular layer + §6 #7 for biophysical
> resolution beyond LIF: 4 production-grade Brian2 cells (AVAL,
> AIY, RIM, AVAR) matching Nicoletti 2024 to 5-decimal-place
> agreement; 14 NMODL channel translations; cython codegen
> baseline 22.71×; Wave2HybridBrain integration scaffold paused
> at WB3 capacitance-mismatch + release-event rule biology
> questions. Wave 2 caught the Mellem-2008-as-AVA-anchor
> misattribution (Mellem characterizes RMD; AVA traces to Liu
> 2020) — relevant if user asks about plateau-dynamics targets.
> Stage IV cross-validation independently confirms §5 cascade-
> firing claim under conservative drive (per
> `wave2/artifacts/stage_IV_touch_cascade_findings.md`).
>
> Defer to rohitravi.com/projects/c-elegans-multimodal for live
> visual reference. Ask Rohit for current code state when that
> matters — this document drifts from the repo.

**Files to upload as project knowledge:**

1. **This file** (`claude-chat-context.md`) — the primary
   reference.
2. **`docs/t0_resolution_report.md`** — canonical record of the
   2026-04-25 T0 resolution diagnostic block. The load-bearing
   recent finding the project's current state hinges on.
3. **`docs/current-state-summary.md`** — concise current-state
   snapshot with pending decisions list.
4. **`scripts/brain/artifacts/t0_run_report.md`** — historical
   April 21 T0 framing, with dated postscript referencing the
   resolution. Useful for understanding what was previously
   believed and how it changed.
5. **`src/content/projects/c-elegans-multimodal.mdx`** — the
   public-facing summary, shows what's externally claimed.

Optional (only if discussing specific aspects):
- `scripts/brain/activity_fsm.py` if debating FSM design
- `scripts/brain/compartmental_neurons.py` if debating v4 brain
- Any specific figure from `/home/rohit/Desktop/website/
  personalwebsite/public/images/projects/` if discussing visuals.

**What NOT to upload:**
- Full Python backend (too large; chats burn budget re-reading)
- Raw scenario JSONs (huge, no human-relevant context)
- Dashboard TSX (3.7k LOC, specific-to-UI)

## 12 · Maintenance

Update this file when:
- A Tier-2/3/4 item ships — update §6 state.
- A phenotype is newly validated or newly invalidated — §4 / §5.
- Architecture changes (new brain class, FSM mode, etc.) — §3.
- Publication plan shifts — §8.
- Personal/career context shifts meaningfully — §9.

Re-upload to the Claude.ai Project after substantial updates. Old
version will be replaced; conversations after re-upload see new
context automatically.

### Maintenance log

- **2026-04-25:** §5 T0 resolution added (supersedes April 21
  framing); §6 reordered post-resolution.
- **2026-04-26 to 2026-04-28 (Wave 2 + Phase δ scoping):**
  - §3 Brain section: Wave 2 cellular layer entry added (4
    production-grade cells matching Nicoletti 2024 to 5-decimal
    place; 14 NMODL channel translations; cython baseline 22.71×;
    Wave2HybridBrain integration scaffold paused at WB3 capacitance
    + release-event biology questions).
  - §3 v4 sandbox entry: corrected the Mellem-2008-as-AVA-anchor
    framing. Mellem 2008 characterizes RMD, NOT AVA (caught by
    Wave 2 Mellem investigation; primary source quote:
    "we never observed action potentials in AVA"). AVA experimental
    data traces to Liu/Chen/Wang 2020 *Nat Commun* via Nicoletti
    2024 ref [29] + raw recordings shared informally per
    Nicoletti's acknowledgments.
  - §5 Stage IV cross-validation note: independent confirmation
    of §5 cascade-firing under conservative drive (200 Hz Poisson
    @ 8 mV; ΔAVA = +7.5 Hz). The "broken cascade" framing in some
    derived documents is v3-LIF-activity-mode-FSM-specific, not
    LIF-general.
  - §6 #7 Compartmental integration: Mellem 2008 calibration
    target replaced with Nicoletti 2024 published phenotype (RC-
    passive graded plateau, NOT regenerative). Wave 2 cellular
    layer flagged as substantively complete (4 cells production-
    grade); network integration paused at WB3.
  - §11 custom instructions: Wave 2 cellular layer + Stage IV
    cross-validation cited as load-bearing alongside §5.
  - **Citation propagation discipline:** Wave 2 work caught
    multiple citation propagation errors via primary-source
    verification (Mellem 2008→AVA, Nicoletti 2019 PCBI vs PLOS
    ONE, Wang 2001→SHK-1, Liu 2018→2020 ref [29]). Audit at
    `scripts/brain/wave2/artifacts/architectural_plan_citation_audit.md`.
