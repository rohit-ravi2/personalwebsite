# T0 Resolution Report — 2026-04-25

Canonical record of the diagnostic work block that resolved the T0
cascade-failure question. Supersedes the April 21 T0 framing in
`scripts/brain/artifacts/t0_run_report.md` and §5 of
`docs/claude-chat-context.md`.

This file describes what was tested, what was measured, and what each
result means for the simulator. It does not extend or speculate beyond
the data.

## 1. Starting state (April 21 framing)

The April 21 T0 report (`scripts/brain/artifacts/t0_run_report.md`)
documented two findings that anchored the project's understanding of
the simulator at that point:

1. The v3 LIF brain's AVA/Chalfie phenotype reproduction
   (ΔREV = −0.57 ± 0.37, n=3) was running through the classifier's
   distributed correlation pattern, not through circuit dynamics.
2. The blocker was framed as a synaptic weight calibration problem —
   tune `W_syn` so that ALM → AIB → AVA produces a real AVA burst
   on touch. Tier 4-3 was the planned fix.

The proposed mechanism for the failure was that the LIF brain's
synaptic weights were undercalibrated for the touch cascade, and that
calibration of ALM→AIB→AVA edges would be the load-bearing fix.

Both parts of that framing are now obsolete. Today's diagnostic block
(2026-04-25) established what the actual mechanism is, and the
proposed fix path was wrong about which cascade was operative and
about what kind of fix was needed.

## 2. Diagnostic methodology principles applied

The session followed four principles that shaped the work:

- **Single-change discipline.** Each test isolated one variable so
  effects could be attributed unambiguously. No simultaneous
  calibration tweaks. Cascade-firing was the mechanistic stopping
  criterion, not phenotype passing.
- **Falsifiability before elaboration.** Each hypothesis was
  pre-specified with measurable outcomes before the test ran. Two
  hypotheses (voltage regime, gap conductance) were falsified
  cleanly because the pre-spec was honest.
- **Direct measurement settles disputes.** When two sessions reached
  opposite conclusions about the simulator's runtime sign-mode
  default, the resolution was a 5-minute direct measurement of the
  signed weight in memory, not more code reading.
- **Mechanism is not the same as phenotype.** Phenotype reproduction
  is necessary but not sufficient evidence that the simulator does
  the right circuit-level thing. The April 21 phenotype "PASS" was
  technically valid as a statistical result but masked a wrong
  mechanism.

## 3. Sequence of falsifications

### 3.1. Voltage regime (FALSIFIED)

**Test:** Patch `lif_brain.py` LIF parameters from
(v_rest=−65, v_thr=−50, v_reset=−70) → (v_rest=−25, v_thr=−10,
v_reset=−30). Mellem 2008 (PMID 18587393) reports AVA rest at
−20 to −30 mV; previous values were a mammalian-cortical template
~40 mV off. The patch preserves the 15 mV rest-to-threshold gap
and 5 mV reset-below-rest, so it is a coordinate translation.

**Result:** No-op for LIF dynamics. Post-fix cascade rates at
default g_gap matched historical pre-fix rates within ±2 Hz across
every cascade neuron. Phenotype unchanged. AVA still drops on
touch.

**Interpretation:** LIF dynamics under noise + synaptic kicks
depend only on relative voltages, not absolute. The patch is
biologically defensible (correct numbers for future
HH/compartmental work that depends on absolute voltages, e.g. Ca
channel activation thresholds) but is invisible to LIF. Kept in
place; flagged as a no-op for current dynamics.

### 3.2. Gap-junction conductance (FALSIFIED)

**Test:** Sweep `g_gap` ∈ {0.1, 0.3, 1.0} nS at default LIF
configuration, n=10 × 60s AVA/touch each.

**Result:** Increasing gap conductance monotonically harmed both
cascade firing and phenotype reproduction:

| g_gap | network mean rate | AVA Δ on touch | ΔREV (phenotype) | neg seeds |
|---|---|---|---|---|
| 0.1 nS (default) | ~45 Hz | −4.0 Hz | −0.49 ± 0.10 | 10/10 |
| 0.3 nS | ~24 Hz | −5.5 Hz | −0.12 ± 0.05 | 7/10 |
| 1.0 nS | ~5 Hz | −0.7 Hz | +0.00 ± 0.00 | 0/10 |

**Interpretation:** Gap junctions average out per-neuron noise
across coupled cells. With 2188 gap edges in the brain, increasing
conductance synchronizes the network and reduces effective
per-neuron noise — fewer threshold crossings, lower firing rates.
At g_gap=1.0 the whole network drops to ~5 Hz baseline, AVA
ablation has nothing to remove, phenotype zeros out.

### 3.3. The AIB-cascade premise (FALSIFIED at the connectome level)

**Test:** Read the connectome to verify ALM→AIB chemical and gap
edges before continuing to calibrate them.

**Result:** ALM and AVM have **zero direct chemical synapses to
AIB** in this connectome. Neither is in AIB's input set —
chemically or via gap junction. AIB's largest inputs are AIZ
(Glu, signed inhibitory), AIA (ACh excitatory), and a handful of
sensory neurons (ADL, ASE, ASG, ASH, AWC, ASK). **AIB has zero
chemical edges to AVD in either sign mode.**

**Interpretation:** The "ALM → AIB → AVA cascade" framing in the
April 21 T0 report was not an undercalibrated wire — there was no
wire to calibrate. AIB is not in the operative touch-reversal
cascade in this connectome at all; it is in the chemotaxis
pirouette circuit (driven by AWC/AIA/AIZ).

### 3.3b. The operative touch-reversal cascade (corrected description)

After diagnostic decomposition under per-edge mode, the operative
cascade in this connectome (Cook 2019 hermaphrodite + Loer & Rand
2022 NT identity) is:

> **ALM/AVM → PVC → AVD/AVE → AVA**

PVC is the load-bearing first-stage relay. Mechanism by stage:

- **ALM/AVM → PVC** (5 glutamatergic edges total: ALML→PVCL,
  ALML→PVCR, ALMR→PVCR, AVM→PVCL, AVM→PVCR; raw weights 11, 4,
  10, 10, 17). Under default sign convention, all signed
  inhibitory; under per-edge (PVC iGluR-dominant, ratio 9.6×),
  all signed excitatory. This is the load-bearing flip.
- **PVC → AVD/AVE** (cholinergic, ACh+, sign-mode-invariant).
  PVC delivers ~5× more drive to AVD than direct ALM/AVM
  glutamate inputs to AVD. PVC pair contributes weight ~23 to
  AVDL and ~17 to AVDR; direct ALM/AVM contribute ~1 and ~4
  respectively.
- **AVD/AVE → AVA** (cholinergic, ACh+, sign-mode-invariant).
  Strong: AVDL→AVAL = 37, AVDR→AVAR = 52.
- **Recurrent positive-feedback within the command pair**
  (AVDL→AVDR, AVAR→AVDR, AVEL→AVDL etc.) amplifies once AVD/AVE
  fire, pulling all command neurons to ~97 Hz coherently under
  per-edge mode.

The April 21 T0 report and §5 of the previous context pack both
described the cascade as "ALM → AIB → AVA." That description was
wrong about which neurons were operative. Replace throughout.

Direct ALM/AVM → AVD chemical edges exist but are weak (raw 1, 3
to AVDL/AVDR). Direct ALM/AVM → AVA chemical edges are zero. AVD
is reachable from touch sensors only via PVC chemically; via gap
junctions ALM/AVM couple weakly to AVD (AVM↔AVDL gap weight 8,
ALMR↔AVDR weight 2 — small enough that under any tested g_gap
they don't carry the cascade alone).

### 3.4. Per-edge vs per-neuron sign convention (the actual mechanism)

After the cascade premise was corrected, AVD became the operative
target. Direct decomposition of AVD's drive at rest showed:

- AVD's largest "inhibitors" under default sign convention (LUA,
  FLP, PQR, PHB) do not fire more on touch. They fire less or
  stay flat. AVD does not gain inhibition on touch.
- AVD drops on touch because PVC drops, removing PVC's dominant
  cholinergic excitation of AVD. PVC drops because ALM
  glutamatergically inhibits PVC under default signs.
- For AVDR specifically, ALML/AVM directly chemically synapse on
  AVDR with raw weights 3 and 1. Under default signs (Glu = −1)
  these signal as inhibition; on touch they contribute −258 / −88
  to AVDR's drive change.

The connectome already contains a precomputed alternative signed
weight matrix `W_chem_per_edge` derived from CeNGEN iGluR/GluCl
expression ratios per postsynaptic neuron. It is loaded only when
`use_per_edge_glu_signs=True` is passed to LIFBrain (default
False). This was deferred per the file comment "default off
pending v3.3 recal."

The two sessions working on this question reached opposite
conclusions about whether the simulator's runtime default was
already per-edge or whether it was per-neuron. The dispute was
settled by direct measurement (§4 below).

## 4. The settling moment — direct runtime measurement

Two `ClosedLoopEnv` instances were constructed (one with
`use_per_edge_glu_signs=False`, one with `=True`). The actual
signed weight in memory for the FLP→AVDL edge and several others
was read directly from the Brian2 `syn_exc.w` / `syn_inh.w`
arrays.

| edge | DEFAULT mode (in memory) | PER-EDGE mode (in memory) |
|---|---|---|
| FLPL → AVDL | INH w = −30.0 | EXC w = +30.0 |
| FLPL → AVDR | INH w = −34.0 | EXC w = +34.0 |
| PQR → AVDL | INH w = −15.0 | EXC w = +15.0 |
| LUAL → AVDL | INH w = −27.0 | EXC w = +27.0 |
| ALML → AVDR | INH w = −3.0 | EXC w = +3.0 |
| AVM → PVCL | INH w = −10.0 | EXC w = +10.0 |
| PHBL → AVDL | INH w = −6.0 | EXC w = +6.0 |
| ASHL → AVDL (control: per-neuron override) | EXC w = +7.0 | EXC w = +7.0 |
| AIBL → AVAL (control) | EXC w = +5.0 | EXC w = +5.0 |
| AVDL → AVAL (control: ACh, no Glu) | EXC w = +37.0 | EXC w = +37.0 |

**Settled:** The simulator's default behavior is per-NEURON
presynaptic NT-sign with ~26 hand-picked overrides. The per-edge
matrix is precomputed in `connectome.npz` but unused unless the
constructor flag is set.

**Bulk:** 518 hard sign flips (exc↔inh) between the two modes
across 3120 unique non-zero chemical edges (~17%). 21 edges
present in default-mode are zeroed in per-edge mode (postsynaptic
Glu receptor expression = 0). Total net excitation changes from
+8006 to +8297 (~4% increase) — modest at the network level
because flips go in both directions.

## 5. Per-edge sweep results (n=10 × 60s, AVA/touch)

Single change: `use_per_edge_glu_signs=True` at default g_gap=0.1.
All other parameters identical to post-volt baseline. Comparison
against post-volt baseline at default g_gap.

### Cascade firing (control runs)

| neuron | post-volt baseline (default signs) | per-edge mode | change |
|---|---|---|---|
| ALML (sensory) | 2.4 → 89.2 (Δ +87) | 1.2 → 89.2 (Δ +88) | same — sensory drives correctly |
| AVM | 0.7 → 87.8 (Δ +87) | 0.6 → 87.6 (Δ +87) | same |
| AIBL | 8.3 → 10.4 (Δ +2.1) | 11.2 → 13.1 (Δ +2.0) | flat (no ALM→AIB pathway under either) |
| AIBR | 14.8 → 14.0 (Δ −0.8) | 12.7 → 16.1 (Δ +3.4) | weakly UP under per-edge |
| **AVDL** | 42.2 → 34.4 (Δ −7.8) | 36.5 → 96.8 (Δ **+60.2**) | flips DOWN → UP |
| **AVDR** | 41.6 → 33.5 (Δ −8.2) | 36.5 → 96.7 (Δ **+60.2**) | flips DOWN → UP |
| **AVAL** | 45.2 → 40.6 (Δ −4.6) | 36.8 → 97.0 (Δ **+60.3**) | flips DOWN → UP |
| **AVAR** | 46.0 → 42.5 (Δ −3.4) | 37.1 → 98.0 (Δ **+61.0**) | flips DOWN → UP |
| AVEL | 29.3 → 23.6 (Δ −5.7) | 33.8 → 81.2 (Δ +47.4) | flips DOWN → UP |
| AVER | 33.9 → 28.9 (Δ −5.0) | 33.3 → 81.0 (Δ +47.7) | flips DOWN → UP |
| AVBL (forward cmd) | 45.6 → 37.0 (Δ −8.7) | 30.3 → 81.5 (Δ **+51.2**) | flips DOWN → UP — see §6 |
| AVBR (forward cmd) | 45.8 → 38.7 (Δ −7.1) | 34.8 → 91.3 (Δ **+56.6**) | flips DOWN → UP — see §6 |
| PVCL (forward cmd) | 44.6 → 37.9 (Δ −6.7) | 36.5 → 96.9 (Δ **+60.4**) | flips DOWN → UP — see §6 |
| PVCR (forward cmd) | 44.5 → 36.5 (Δ −8.0) | 36.5 → 106.3 (Δ **+69.8**) | flips DOWN → UP — see §6 |
| RIML | 32.1 → 31.4 (Δ −0.7) | 19.6 → 28.0 (Δ +8.4) | weakly UP |
| RIMR | 32.2 → 33.4 (Δ +1.2) | 24.2 → 36.4 (Δ +12.1) | weakly UP |
| RIS | 21.8 → 20.7 (Δ −1.1) | **0.8 → 1.4** (Δ +0.6) | tonically silenced — see §6 |

Per-seed variance under 1.5 Hz across all cascade neurons (n=10).

### Phenotype (AVA-ablation effect on touch)

| condition | ΔREV ± SEM | neg seeds | ΔQUI ± SEM | ΔPIR ± SEM | neg PIR seeds | verdict |
|---|---|---|---|---|---|---|
| post-volt baseline | −0.49 ± 0.10 | 10/10 | +0.52 ± 0.10 | (not the load-bearing channel) | — | PASS-on-dREV via tonic-shift |
| g_gap=0.3 | −0.12 ± 0.05 | 7/10 | +0.17 ± 0.08 | — | — | DIRECTIONAL |
| g_gap=1.0 | +0.00 ± 0.00 | 0/10 | +0.00 ± 0.00 | — | — | NULL (network silenced) |
| **per-edge (default g_gap)** | **+0.04 ± 0.02** | **2/10** | **+0.06 ± 0.03** | **−0.117 ± 0.031** | **9/10** | **dREV null; dPIR shows preserved AVA-ablation effect** |

**dPIR finding (2026-04-25):** Under per-edge mode, dREV regresses
to null (+0.04, 2/10 negative) but **dPIR retains a clean
AVA-ablation effect** (mean −0.117, 9/10 negative seeds). AVA
ablation under correct cascade dynamics still produces a behavioral
signature — it appears in a different FSM/classifier output channel
than under default mode. The Chalfie 1985 phenotype's behavioral
expression in this simulator may be channel-dependent in a way the
April 21 single-channel framing did not anticipate.

## 6. Side effects and open questions

### 6.1. PVC/AVB over-activation (open question, not settled)

Under per-edge mode, PVC fires up Δ +60-70 Hz and AVB fires up Δ
+51-57 Hz on touch. Canonical biology has anterior touch
suppressing forward locomotion through AVB inhibition; PVC's
response is less unambiguous in literature but excitation on
anterior touch is biologically questionable.

The mechanism is traceable: PVC's CeNGEN ratio is 9.6× iGluR-
dominant, so ALM/AVM glutamate inputs to PVC sign as excitatory
under per-edge mode. AVB's direct ALM input is correctly
inhibitory (AVB is GluCl-dominant, ratio 0.63), but PVC's
excitation propagates to AVB through PVC→AVB cholinergic edges.

Two interpretations are both consistent with the data; neither is
yet falsified:

- **Interpretation A (CeNGEN tells receptor presence, not
  functional dominance):** PVC has iGluR receptors in its
  expression profile, but the ALM synapse onto PVC is functionally
  GluCl-mediated (or some other inhibitory mechanism). The
  per-edge convention's assumption that receptor presence equals
  functional dominance is wrong for at least PVC.
- **Interpretation B (canonical biology is more nuanced than the
  textbook story):** The Faumont 2011 / Chalfie 1985 chassis story
  may oversimplify; per-edge dynamics may match newer literature
  about parallel chassis circuits where forward command isn't
  strictly suppressed during reversal.

This is a class-of-bug question, not just a PVC question.
Resolution requires per-edge functional measurements
(electrophysiology data per synapse), not just receptor
expression.

### 6.2. RIS silencing under per-edge mode

RIS goes from 21.8 Hz tonic baseline (default mode) to 0.8 Hz
(per-edge mode). RIS is GluCl-dominant (sign = −1, unchanged
across modes), so this is not a direct sign-flip effect — it is a
network-equilibrium consequence of broader sign changes upstream.

This has implications for any RIS/sleep-related work and for the
RIS molecular audit findings from earlier in the project. The
April 21 RIS molecular audit (FLP-11 release fires correctly,
peptidergic targets show ~22% disinhibition, behavioral null
consistent with readout insensitivity) was conducted under default
mode. RIS at 0.8 Hz under per-edge would not produce comparable
FLP-11 release; the audit needs re-running before any conclusions
about RIS mechanism transfer to per-edge mode.

### 6.3. FSM / classifier calibration debt (refined per dPIR finding)

The 18-readout classifier bank was trained on default-mode firing
distributions. Under per-edge mode, AVA jumps from ~37 Hz baseline
to 97 Hz peri-touch — a tripled dynamic range, and a different
correlation structure across the readout neurons. The classifier
does not decode per-edge-mode dynamics into the same FSM channel
as under default mode.

The dPIR finding (§5) refines this question. Under default mode,
AVA ablation produced its phenotype effect on dREV. Under per-edge
mode, the same ablation still produces an effect (clean dPIR
signature, 9/10 negative seeds) — but in a different FSM channel.
The recalibration question is therefore not "retrain to recover
dREV." It is broader:

> Characterize what behavioral signature AVA-ablation produces
> under correct cascade dynamics, and determine which FSM/
> classifier channels best reflect the circuit-level Chalfie
> phenotype.

Three sub-questions for the next work block:

1. Does AVA-ablation in the corrected simulator produce the
   biological phenotype Chalfie 1985 described, just measured
   through different FSM channels (dPIR is plausibly that
   measurement)?
2. Or does the recalibration produce a recovered dREV at
   different threshold settings within the existing FSM logic?
3. Or is the per-edge-mode dynamics regime fundamentally
   incompatible with the existing 18-readout classifier
   architecture, requiring a wider redesign?

These are not equivalent questions and the answer determines what
"recalibration" means scope-wise. The classifier bank retraining
that was deferred during overnight v2 Track B (LOGISTICAL_FAILURE)
is the technical prerequisite, but the conceptual decision needs
to come first.

## 7. What this means for the project's prior validated phenotypes

### 7.1. AVA / Chalfie 1985 reproduction (re-described)

The April 21 finding ("AVA ablation abolishes touch-driven
reversal, ΔREV = −0.57 ± 0.37") is now re-described as:

> **AVA ablation reproduces the direction of the Chalfie 1985
> phenotype on dREV under default sign convention via a Mode 3
> tonic-shift mechanism. The cascade itself does not fire on
> touch under default convention. Per-edge sign convention makes
> the cascade fire correctly; AVA-ablation's behavioral effect
> persists (clean dPIR signature, mean −0.117, 9/10 negative
> seeds) but no longer appears on dREV (mean +0.04, 2/10
> negative seeds) because the FSM/classifier was calibrated to
> default-mode dREV-channel response. The biological phenotype is
> not necessarily lost under per-edge — it shifts FSM channels.**

The phenotype reproduction on dREV was technically valid as a
statistical result. It was mechanistically misleading as a biology
claim. Any downstream work that relied on "AVA/Chalfie passes" as
evidence that the simulator captured the touch-reversal cascade
should be re-examined; what passed was the readout's sensitivity
to AVA's tonic baseline through the dREV channel, not the
circuit-level firing of the cascade.

The audit-quality improvement (10/10 seeds under post-volt
baseline at n=10 × 60s) was real. It was a more reliable
measurement of a wrong-mechanism reproduction. Statistical
robustness is not the same as mechanistic validity.

The persistence of an AVA-ablation effect on dPIR under per-edge
mode is informative: the simulator's circuit-level response to
AVA loss is preserved, but the FSM/classifier was calibrated to
read it through a channel (dREV) that changes meaning under
per-edge dynamics. Whether dPIR is the "right" measurement of the
Chalfie phenotype in the corrected simulator, or whether a
recalibrated classifier would re-route the signal to dREV, or
whether the FSM architecture itself needs redesign, is open.

### 7.2. RIS / Turek 2016 quiescence (status updated)

The April 21 RIS finding (ΔQUI = −0.24 ± 0.33 across 3 seeds,
2/3 negative; molecular audit showing FLP-11 release fires
correctly) was conducted under default sign convention. RIS is
silenced under per-edge mode (0.8 Hz vs 21.8 Hz tonic), so the
default-mode finding does not transfer. The RIS phenotype and
molecular audit need re-running under per-edge mode before any
claim about RIS mechanism in this simulator.

### 7.3. Three-mode taxonomy (overnight v1 + v2)

The three-mode readout failure-mode taxonomy validated across all
9 v3 modulators (Mode 1 ×5: FLP-11, FLP-1, NLP-12, OA, TA;
Mode 2 ×2: 5HT, DA; Mode 3 ×2: FLP-2, PDF-1) was established
under default sign convention. The taxonomy itself is a
methodological framework about how readout architecture interacts
with circuit-level effects, and that framework remains valid
regardless of sign mode. The specific Mode classifications per
modulator are conditional on default mode and would need
re-classification under per-edge mode.

### 7.4. Other audited phenotypes

The historical v3.0 / v3.1 / v3.2 / v3.3 perturbation audits, the
overnight runs (D1 modulator audit, A1+B4 peptide expression
audit, RIS molecular audit, NSM audit), and any phenotype claim
predating today were all conducted under default sign convention.
Each needs its mechanism re-examined under per-edge mode to
determine whether the result was genuine circuit reproduction or
sign-convention artifact.

This does not invalidate the audits as data — they are valid
measurements of what the simulator does in default mode. It
re-frames what they tell us about the underlying biology.

## 8. What remains open

**Architectural decisions:**
- Whether per-edge becomes the production default or stays opt-in.
- How to handle PVC/AVB under per-edge mode (curated per-edge
  override list, hybrid sign convention, or accept the over-
  activation as a known limitation; see §6.1 — two interpretations
  remain open and neither is yet falsified).
- FSM/classifier recalibration against per-edge firing
  distributions; specifically, how to map AVA-ablation effect
  across FSM channels (the dPIR finding suggests channel-shifting
  rather than signal loss; see §6.3).

**Open suspects after today's work** (updates the open-suspects
list from earlier diagnostic blocks):
- **PVC functional-vs-receptor-expression mismatch.** CeNGEN
  receptor expression may diverge from functional dominance at
  specific synapses (Finding 9, two interpretations open).
- **FSM/classifier calibration regime debt.** 18-readout
  classifier trained against firing patterns from default-mode
  dynamics; under per-edge dynamics the AVA-ablation effect
  shifts FSM channels (dREV → dPIR). Recalibration question is
  broader than just recovering dREV (Finding 8).
- **Chemical weight scaling** — possibly less urgent now that
  cascade fires at +60 Hz under per-edge, but specific weights
  (e.g. ALM/AVM → PVC) may need fine-tuning if PVC over-activation
  is judged to be a real bug.
- **Membrane time constant τ** — 10 ms in simulator vs 50-200 ms
  in worm computational models; implied input resistance 30-50×
  off from Goodman 1998's measurements. Independent of T0
  resolution.

**Specific investigations:**
- Network-stability scan under per-edge mode for non-touch
  scenarios (osmotic_shock, food, chemotaxis, aerotaxis,
  spontaneous).
- RIS silencing investigation: why does RIS go quiet at network
  equilibrium under per-edge mode?
- Re-running RIS molecular audit, three-mode taxonomy, and any
  Mode 3 modulator results under per-edge mode.
- Behavioral signature characterization under per-edge mode:
  does AVA-ablation produce the Chalfie phenotype through dPIR
  (preserved under per-edge), or does the recalibration produce
  recovered dREV at different threshold settings?

**Items lower priority now that cascade fires:**
- Tier 4-3 synaptic weight calibration (originally framed as the
  T0 fix; the cascade now fires at Δ +60 Hz without weight tuning,
  so the framing is obsolete; weight calibration may still be
  useful for fine-tuning but is not load-bearing).
- Tier 4-2 plateau dynamics calibration (separate question;
  per-edge sign mode does not address plateau dynamics, which
  are an active-conductance phenomenon requiring SK/BK channels
  and compartmental modeling).

**Items still on the previous critical path:**
- Per-edge sign-flip count discrepancy: Session 1 reports 518
  hard sign flips (5 counting methods agree); Session 2 reports
  415 (4 methods). Discrepancy unresolved, probably from different
  `connectome.npz` builds or override list contents. Not blocking
  but worth reconciling.
- Voltage fix is in place but is a no-op for current LIF
  dynamics. Stays for biological documentation; will matter when
  SK/BK and compartmental work starts.

**Audit-trail items surfaced today:**
- 535 GB `data/external/` inventory revealed AVA is in 100% of
  Atanas worms (10/10), AVD in 100%, AIZ in 90%. The strict
  cross-worm intersection filter that produced the 18-readout
  set excluded canonical command interneurons that ARE present
  in the data. Readout expansion is now clearly possible from
  existing data; the 18-readout set was a methodology choice,
  not a data limitation.

## 9. File references

Code changes (committed in `b87e03e`):
- `scripts/brain/lif_brain.py` — voltage regime patch + sign-mode
  documentation in module header.
- `scripts/brain/closed_loop_env.py` — `g_gap_ns` and
  `use_per_edge_glu_signs` pass-through to LIFBrain constructor.
- `scripts/brain/phase0_audit.py` — `--g-gap-ns` and
  `--use-per-edge-glu` CLI flags.

Analysis scripts:
- `scripts/brain/phase0_postvolt_compare.py` — per-condition
  cascade + phenotype side-by-side comparison harness.
- `scripts/brain/phase0_avd_drive_decomp.py` — per-source
  resting-drive decomposition for AVD.

Sweep artifacts (n=10 × 60s, AVA/touch):
- `scripts/brain/artifacts/phase0_postvolt_phenotype_default.csv`
  — post-volt baseline, g_gap=0.1.
- `scripts/brain/artifacts/phase0_postvolt_gap03_phenotype_default.csv`
  — gap sweep, g_gap=0.3.
- `scripts/brain/artifacts/phase0_postvolt_gap10_phenotype_default.csv`
  — gap sweep, g_gap=1.0.
- `scripts/brain/artifacts/phase0_postvolt_peredge_phenotype_default.csv`
  — per-edge sign mode at default g_gap.

Companion documents:
- `scripts/brain/artifacts/t0_run_report.md` — original T0 report
  (April 21) with dated postscript referencing this file.
- `docs/current-state-summary.md` — updated state summary post-
  this work block.
- `docs/claude-chat-context.md` — §3, §4, §5, §6 updated to
  reflect per-edge findings.

---

## Postscript — horizontal rebase 2026-05-08

After this April 25 resolution work, a horizontal rebase ran Phase 0
(state-of-claims catalog) → Phase 1 (sign-mode decision gauntlet across
4 candidates: M1 default, M2-pure, M2-current per-edge + DOCUMENTED_SIGN_EXCEPTIONS,
M3a per-edge + AIY-only exceptions) → Phase 2 (fresh CV-trained classifier
under A2-balanced 21-cell readout + M2-pure calibration + recalibrated FSM
thresholds) → Phase 2.5 gauntlet validation. Two findings from that rebase
update this report:

**1. The DOCUMENTED_SIGN_EXCEPTIONS registry (commit aea4c79, added 2026-04-25
after this report) collapses the per-edge cascade.** Per the Phase 1
gauntlet: under per-edge + 7 DOCUMENTED_SIGN_EXCEPTIONS (the production
default at time of writing), AVDL Δ peri-touch = −2.50 Hz instead of the
+60.2 Hz this report measured under pure per-edge (no exceptions). The
5 ALM/AVM → PVC entries in the registry sign-flip the cascade-initiating
edges, suppressing the cascade upstream of AVA. Phase 1 of the rebase
locked the brain at M2-pure (per-edge + `sign_exceptions={}`) as the only
sign mode firing the cascade.

**2. The §5 dPIR channel-shift finding (mean −0.117, 9/10 negative seeds at
n=10×60s) is REFUTED under the recalibrated stack.** Phase 2.5 default tier
re-ran the same n=10×60s phenotype protocol under M2-pure with the new
classifier (`classifier_bank_v2_a2balanced.npz`) + new calibration
(`calibration_m2pure.npz`) + new FSM thresholds. AVA → dPIR = −0.005 ± 0.005
(1/10 negative) — essentially zero. The original dPIR finding was a
legacy-stack artifact (legacy 18-readout classifier + legacy calibration +
legacy FSM thresholds), not a real channel-shifted phenotype. Catalog claim
C-21 reclassified from Direct → Falsified-but-cited.

**The §5 "behavioral effect persists in dPIR channel" claim should be
read as: that effect was an artifact of the legacy readout stack, not a
real biological signal.** Under correct cascade dynamics + recalibrated
readout, the Chalfie 1985 phenotype is not reproduced on dREV or dPIR.

**Alternative phenotype finding (Direct, awaiting literature precedent):**
under M2-pure with recalibrated stack, AVA → dFWD = −0.302 ± 0.102 (7/10
negative, Cohen's d ≈ 0.93). Forward-locomotion suppression rather than
reversal abolition. Biologically interpretable via AVA-AVB gap-junction
coupling (Wang/Liu/Chen 2020, Nat Commun 11:5076) but not the textbook
Chalfie 1985 phenotype.

**Companion rebase documents:**
- `docs/state_of_claims_2026-05-02.md` — full state-of-claims catalog
- `docs/brain_v3.5_locked.md` — M2-pure brain spec (Phase 1 lock)
- `docs/phase2_preflight.md` — Phase 2 architecture sign-off
- `scripts/brain/artifacts/phase1_gauntlet_screen_decision_matrix.md` — Phase 1 results
- `scripts/brain/artifacts/phase2_gauntlet_default_decision_matrix.md` — Phase 2.5 default tier
- `scripts/brain/wave2/artifacts/phase_delta_wb3_findings.md` — WB3 findings (with D7-followup postscript + C-37 resolution)
