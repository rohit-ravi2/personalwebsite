# Audit strategy — three layers, matched to claims

*Methodological document produced at Phase 0 close-out (2026-04-21).*
*Supersedes the implicit "behavioral phenotype audit = success metric"
assumption used in v3.0 through v3.3 audits.*

**2026-04-25 update:** the three-layer framework and the three readout
failure modes (Mode 1/2/3) below remain valid — they are sign-mode-
independent methodological constructs. The specific AVA/touch entries
(Mode 3 conditional, "post-T4-3" framing) are superseded; T4-3 weight
calibration is no longer the operative fix. The cascade firing was
resolved at the architectural level by per-edge sign convention
(`use_per_edge_glu_signs=True`), and the operative cascade in this
connectome is ALM/AVM → PVC → AVD/AVE → AVA, not the long-assumed
ALM→AIB→AVA. See `docs/t0_resolution_report.md` for the canonical
record. Per-ablation routing table updated below.

## The problem Phase 0 revealed

The project had implicitly assumed behavioral phenotype reproduction
(measured through the classifier-bank → FSM → state-distribution
pipeline) was the gold-standard validation metric. Phase 0 produced
three findings that invalidate that assumption:

1. **T0 finding (pre-Phase-0):** the AVA/Chalfie phenotype reproduction
   runs through the classifier's distributed correlation pattern, not
   through a biologically-correct command-neuron cascade. Passing the
   behavioral test doesn't mean the underlying cascade works.

2. **RIS molecular audit (Phase 0):** the RIS/Turek quiescence
   mechanism IS operating at molecular and cellular levels — FLP-11
   releases on osmotic stress, FLP-11 inhibits its 152 peptidergic
   targets, and the pathway dissociates cleanly via FLP-11 knockout.
   But the classifier-FSM readout uses an 18-neuron set that minimally
   overlaps with FLP-11's primary targets. The behavioral test shows
   ΔQUI ≈ 0 despite the mechanism functioning correctly.

3. **NSM counter-finding (Phase 0, partial audit):** ablating NSM
   produces ΔQUI ≈ +0.50 — a strong, consistent signal. But NSML
   and NSMR are IN the 18-neuron readout. The "phenotype" is the
   classifier responding to having two of its 18 trained inputs
   zeroed, not to any serotonergic dwelling biology.

These three findings sort into three readout failure modes. The
audit strategy below distinguishes them.

## Three readout failure modes

### Mode 1 — Readout-blind

Ablated neuron's effect operates through targets outside the 18-neuron
classifier readout. Behavioral test produces false null.

- **Exemplar:** RIS (GABA targets partially overlap readout, FLP-11
  targets almost don't).
- **Likely other instances:** any modulator-releaser whose targets are
  biologically diverse. FLP-1 (AVK), FLP-2 (RID), NLP-12 (DVA),
  PDF-1 (AVB), tyramine (RIM), octopamine (RIC). Most of the 9 v3
  modulators, in fact.

### Mode 2 — Readout-trivial

Ablated neuron IS in the 18-neuron readout. Behavioral test produces
strong signal, but it's the classifier responding to trivially-zeroed
inputs, not to biological mechanism propagating through cascade
dynamics.

- **Exemplar:** NSM (NSML/NSMR in readout; ablating them zeros their
  classifier contribution directly).
- **Likely other instances:** AIBL, ASEL, AUAL, AVEL, AVER, CEPDL,
  I3, IL2DL, M3L/R, NSML/R, OLQDL/R/VL, RMER, SMDVL, URXL — any
  ablation of these 18 produces readout-trivial signals regardless
  of biology.

### Mode 3 — Readout-real (partial / hypothetical)

Ablated neuron isn't in the readout, but its effect propagates to the
readout via real synaptic cascade. Behavioral test may reflect actual
biology.

- **Exemplar candidate:** AVA (not in readout; AVE/AVER are in readout
  and receive RIS inhibition + downstream motor-neuron drive). T0
  established that under default sign convention the cascade doesn't
  actually fire on touch — the apparent phenotype runs through
  classifier pattern via Mode 3 tonic-shift. **2026-04-25 update:**
  per-edge sign mode resolves the cascade firing at the architectural
  level (ALM/AVM → PVC → AVD/AVE → AVA fires at +60 Hz on touch);
  AVA-ablation effect persists in dPIR channel under per-edge (mean
  −0.117, 9/10 negative seeds) but the FSM/classifier was
  calibrated to read it through dREV under default-mode dynamics.
  AVA is now genuinely Mode 3 (cascade-real) under per-edge sign
  mode, conditional on resolving the FSM/classifier recalibration
  question. See `docs/t0_resolution_report.md`.
- **Likely other instances:** under per-edge mode the operative
  cascade flips for additional command neurons (AVD, AVE, AVB, PVC
  all fire UP on touch; PVC and AVB excitation is biologically
  questionable — open question). Sensory-driven behaviors (ASI,
  ASH, ASK) need re-evaluation under per-edge before classification.

## Three audit layers

### Layer A — Molecular pathway audit

**What it measures:** whether the underlying biological machinery
operates. For modulator ablations: does the peptide/amine concentration
track correctly? Do target neurons' firing rates respond? Do the
pathways dissociate cleanly under selective knockouts?

**Implementation:** `phase0_ris_pathway_audit.py` template. Supports
selective knockout modes (full ablation, synaptic-only knockout via
`syn_inh.w[mask] = 0`, release-only knockout via
`modulation.releaser_weights[m, n] = 0`). Saves full telemetry
(modulator concentration buffer + full 300-neuron raster). Follow-up
analyzer dissects into 4 sub-layers (molecular / cellular-GABA /
cellular-peptidergic / behavioral).

**Cost:** ~45 min for 5 seeds × 4 conditions at 60s (n=20 runs).

**Applies to:** all 9 v3 modulators (FLP-11, FLP-1, FLP-2, NLP-12,
PDF-1, 5HT, DA, TA, OA). All 6 INS peptides when T4-5 activates.

### Layer B — Behavioral phenotype audit (restricted)

**What it measures:** whether a specific phenotype reproduces at audit
rigor. Only useful for ablations whose effect genuinely propagates to
the behavioral readout.

**Implementation:** `phase0_audit.py --mode phenotype --ablations ...`
filtered to qualifying ablations. Default tier is n=10 × 60s.

**Cost:** ~14 min per ablation per seed (2 runs × 60s × 3.06×
wall ratio × inter-process contention). At n=10 × 1 ablation: ~2.5 hrs.

**Applies only to:**
- **Unambiguously:** nothing in the current 18-neuron readout regime
  under default sign convention produces clean behavioral tests.
  Every ablation falls into Mode 1, 2, or conditional-Mode-3.
- **Conditionally (under per-edge sign mode, post-2026-04-25):**
  AVA/touch. The operative cascade ALM/AVM → PVC → AVD/AVE → AVA
  fires under per-edge mode; AVA ablation produces a measurable
  behavioral effect, but on dPIR rather than dREV under the
  current FSM/classifier calibration (which was trained against
  default-mode firing distributions). Whether this counts as a
  clean behavioral test depends on resolving the FSM recalibration
  question. See `docs/t0_resolution_report.md` §6.3.
- **Not useful under default sign mode:** RIS, NSM, RIM, AVB, PDE,
  FLP-1, FLP-2, NLP-12, PDF-1 (all Mode 1 or Mode 2). Re-running
  these under per-edge mode is an open follow-on item; some may
  shift category once the cascade dynamics are correct.

### Layer C — Trajectory correlation (readout-agnostic)

**What it measures:** do simulator full-network dynamics match Atanas
calcium imaging at event-aligned time windows? Operates on the full
300-neuron raster, not the 18-neuron readout subset. Immune to the
readout-blind / readout-trivial failure modes.

**Implementation:** `phase6_trajectory_correlate.py`. Per-neuron ×
per-event ρ distribution vs Atanas ΔF/F. The capstone falsification.

**Cost:** trivial compute — pure offline analysis on scenario audit
outputs. Scales linearly with neuron count and event instances.

**Applies to:** global simulator fidelity. Not ablation-specific.

## Per-ablation routing table

| ablation | readout status | expected behavior | correct audit layer |
|---|---|---|---|
| RIS | Mode 1 (readout-blind) | false null behaviorally | **Molecular** |
| NSM | Mode 2 (readout-trivial) | strong signal, no biology | **Molecular** (for real test; behavioral is misleading) |
| RIM | Mode 1 (RIM not in readout; targets diverse) | likely null | **Molecular** |
| PDE | Mode 1 | likely null | **Molecular** |
| AVB | Mode 1 (forward partners PVC/RIB not in readout) | likely null | **Molecular** or **Trajectory** |
| AVA | Mode 3 — cascade real under per-edge mode (2026-04-25) | dPIR signature preserved (−0.117, 9/10 neg seeds) under per-edge; dREV channel calibrated against default-mode dynamics | **Behavioral** (under per-edge, conditional on FSM recalibration) + **Trajectory** |
| FLP-11 target-gene KO | — | molecular pathway dissection | **Molecular** only |
| Sensory cell ablations (ASH, ASE, AWC…) | Mode 3 conditional | depends on downstream cascade | **Molecular** (cascade verification) + **Behavioral** if cascade works |

## Revised Phase 0 close-out

Originally planned: 6-ablation behavioral audit at n=10 × 60s
(~12 hrs compute). **Revised after readout architecture analysis:**

- **Phenotype audit reduced to AVA/touch only** (~2 hrs compute).
  Tests whether the T0-identified pattern-mediation phenotype survives
  at audit rigor. Informative regardless of pass/fail.
- **Molecular audit runs for RIS** (~45 min compute). Primary RIS
  test — confirms or refutes the readout-insensitivity hypothesis
  with error bars.
- **Scenario audit continues** — provides baseline data for T4-6
  trajectory correlation and firing-rate statistics across 6 scenarios.
- **NSM/RIM/AVB/PDE behavioral rows skipped** — readout architecture
  makes them uninformative regardless of result. ~9 hrs compute saved.

Partial data from the killed audit (RIS 10 seeds + NSM 7 seeds)
preserved at `artifacts/phase0_phenotype_partial_preservd.csv`.

## Connection to the 4-layer falsification framework

The project's paper contribution is shifting from "we reproduce
Chalfie 1985 and Turek 2016" (now known to be either pattern-mediated
or readout-invisible) toward a 4-layer falsification framework:

1. **Classifier readout correctness** (T0): do command cascades
   actually drive phenotypes, or does the readout's trained pattern
   mediate the signal?
2. **Subthreshold dynamics correctness** (Phase 0 plateau diagnostic):
   do membrane voltages match voltage-clamp data?
3. **Body kinematics correctness** (T4-1): does simulator curvature
   match Tierpsy centerlines?
4. **Trajectory correlation correctness** (T4-6): do simulated
   full-network dynamics match Atanas ΔF/F event-aligned?

Each layer catches failure modes invisible to the others. The
readout-blind / readout-trivial / readout-real taxonomy sits at Layer
1, refining it. Specifically: Layer 1 isn't just "do command cascades
drive phenotypes?" It's "does the readout architecture itself let us
observe whether command cascades drive phenotypes, and if not, what
three failure modes explain why?"

The molecular audit as a sub-layer of Layer 1 provides the answer for
modulatory mechanisms: test them at the peptide-concentration /
target-firing-rate level instead of behavioral state distribution,
because the readout architecture can't resolve them behaviorally.

## Going forward — Tier 2-4 audit discipline

For every Tier 2-4 phase that produces a claim, the audit plan should
explicitly answer:

- Which layer(s) does this claim live in? (A, B, C, or combinations)
- For behavioral claims (Layer B): is the ablated neuron in Mode 3?
  (If Mode 1 or 2, use Layer A instead.)
- For modulatory claims: Layer A is mandatory.
- For trajectory claims: Layer C.

Do NOT run Layer B by default. Layer B is expensive and often
uninformative. Layer A is cheaper, more informative for modulator
mechanisms, and directly testable with selective knockouts.

## Artifact inventory

- **Molecular audit tools:** `phase0_ris_pathway_audit.py`,
  `phase0_ris_pathway_analyze.py`. Template for extending to any
  modulator (adapt `apply_gaba_ko` / `apply_flp11_ko` per modulator).
- **Behavioral audit tools:** `phase0_audit.py` (with `--ablations`
  filter and `--tier` / `--mode` flags).
- **Trajectory audit tools:** `phase6_trajectory_correlate.py`.
- **Scenario baseline tools:** `phase0_audit.py --mode scenario`
  (produces per-seed full-raster NPZs consumed by Layer C analysis).

## Changelog

- 2026-04-25: AVA-specific entries updated for T0 resolution.
  The three-layer framework and three readout failure modes are
  unchanged (sign-mode-independent methodological constructs).
  AVA/touch entries that referenced "post-T4-3" weight calibration
  as the operative fix are superseded — the cascade was resolved
  at the architectural level by per-edge sign convention. Operative
  cascade corrected from ALM→AIB→AVA to ALM/AVM → PVC → AVD/AVE
  → AVA (PVC is the load-bearing first-stage relay). AVA is now
  Mode 3 cascade-real under per-edge sign mode, conditional on
  FSM/classifier recalibration. See `docs/t0_resolution_report.md`.
- 2026-04-21: document created at Phase 0 close-out based on RIS
  molecular audit + partial NSM phenotype audit findings. Audit
  strategy formally revised away from "behavioral phenotype = success."
