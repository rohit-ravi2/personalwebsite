# Mellem 2008 investigation — pre-classification pushback

**Date:** 2026-04-26
**Triggering work block:** `phase_v_w2_mellem_investigation_prompt.md`
**Status:** PAUSED before classification verdict. Citation verification step
surfaced a load-bearing misattribution; the spec instructed an immediate
stop-and-surface in this scenario.

> **Spec instruction (verbatim):** "If 'Mellem 2008' citation verification
> surfaces a misattribution: STOP and surface immediately. This would
> invalidate the condition-6 target itself."

The classification (A/B/C/D) is not produced because the prerequisite
question — what biological phenomenon Wave 2 is targeting — needs
cross-session resolution before classification is meaningful.

---

## Headline finding

**The architectural plan's Phase γ Gate-2b target — "Mellem 2008 plateau
(15-25 mV / 400-800 ms, target 20 mV / 600 ms) in AVAL" — is a misattributed
target.** Mellem 2008 explicitly reports that AVA does NOT show plateau or
regenerative dynamics. The plateau dynamics characterized in Mellem 2008
are in **RMD**, a different cell.

The condition-6 architectural-insufficiency conclusion that the Brian2
single-compartment model "fails to reproduce Mellem 2008 plateau in AVA"
is a category error. Per the actual Mellem 2008 paper, single-compartment
AVA showing only graded responses is **consistent with**, not in conflict
with, the experimental record.

This does not invalidate the empirical findings of Phase F (the plateau
amplitude is 46.8 mV with leak-dominated termination at 21.4 ms). Those
phenotypes are real model behaviors. What's invalidated is the framing of
those phenotypes as a *failure to match Mellem 2008*. Mellem 2008 has no
600 ms / 20 mV plateau in AVA to match against.

---

## Verified citations and primary-source quotes

### Mellem 2008 — verified citation, but misattributed for AVA

**Mellem JE, Brockie PJ, Madsen DM, Maricq AV (2008). Action potentials
contribute to neuronal signaling in C. elegans.** *Nature Neuroscience*
11(8):865-867. DOI: `10.1038/nn.2131`. PMID: 18587393. PMCID: PMC2697921.
Published online June 29, 2008. University of Utah, Department of Biology.

**Citation is real and the authors / lab are correct.** The misattribution
is in *which neuron* the plateau dynamics belong to.

**Primary-source quotes** (PMCID: PMC2697921):

> "In 14 out of 16 RMD neurons, we found that the voltage response to
> depolarizing current ramps was linear from approximately −80 mV to −60
> mV, but the voltage response then became regenerative, leading to a
> solitary action potential" (Fig. 1a).

> "We found bistable potentials associated with 54 of 98 RMD action
> potentials" (Fig. 1c-e).

> "**In contrast, we never observed action potentials in AVA (n=10; Fig.
> 1b).**"

> "The resting potential of AVA was typically between −20 and −30 mV and
> we did not observe action potentials (Fig. 1d), **even when we changed
> the resting potential to more hyperpolarized levels.**"

> "In contrast to what was observed in RMD, glutamate application caused
> short-lived, modest changes in AVA membrane potential with no switch
> to a new steady-state potential (n = 5; Fig. 3i)."

**Plateau mechanism (in RMD, not AVA):**

> "Bath application of the Na+-channel blocker TTX (2 µM), failed to block
> action potentials" and when "both external Na+ and Ca2+ were replaced by
> NMDG+...depolarizing current steps no longer elicited action potentials
> (Fig. 2b). These results indicate that **Ca2+ has a critical role in
> action potential generation in RMD.**"

KO/mutant comparisons (RMD): unc-2(e55) reduces amplitude / slows onset
but plateaus persist; egl-19(n2368), cca-1(ad1650), and nca-1(gk9);nca-2(gk5)
double mutant all show preserved plateaus. Mellem concludes "multiple classes
of Ca2+ channels contribute to action potentials, including perhaps yet to
be identified channels."

**No specific plateau amplitude or duration quantified for RMD, either.**
Only qualitative descriptors: "long-lived," voltage relaxes to "approximately
−10 mV" from rest of "approximately −73 mV" (Fig. 1c). Stated as
"long-lasting" rather than with a specific ms value.

### Lockery & Goodman 2009 commentary

**Lockery SR, Goodman MB. The quest for action potentials in C. elegans
neurons hits a plateau.** *Nature Neuroscience* 12(4):377-378 (2009).
DOI: `10.1038/nn0409-377`. PMCID: PMC3951993.

Reframes Mellem 2008's RMD events as plateau potentials:

> "The events recorded by Mellem et al. fit the criteria of plateau
> potentials perfectly in that they are long-lasting, all-or-none events
> that can be terminated by a negative current pulse."

> "RMD plateau potentials depend on a current carried mainly by Na+ and
> Ca2+ ions, which is consistent with the biophysical mechanisms of
> plateau potentials in other organisms."

**No reference to AVA in this review.** No discussion of CICR, IP3R, RyR,
persistent Na current, or NMDA in C. elegans. Field consensus per this
review: plateau mechanism is Na+/Ca2+ but specific channels not
characterized.

### Liu 2018 — most likely identification, but for AWA, not AVA

**Liu Q, Kidd PB, Dobosiewicz M, Bargmann CI (2018). C. elegans AWA
Olfactory Neurons Fire Calcium-Mediated All-or-None Action Potentials.**
*Cell* 175(1):57-70. DOI: `10.1016/j.cell.2018.08.018`. Published online
September 13, 2018. Bargmann lab, Rockefeller University.

**This paper is about AWA, not AVA.** AWA spike mechanism: EGL-19 voltage-
gated CaV1 calcium channels (initiation), SHK-1 Shaker-type potassium
channels (termination), SLO-2 (calcium/chloride-activated K). Confirmed by:

> "AWA fires calcium spikes, initiated by EGL-19 voltage-gated CaV1 calcium
> channels and terminated by SHK-1 Shaker-type potassium channels."

> "shk-1 mutants have prolonged action potentials, presumably associated
> with prolonged EGL-19 channel opening."

If F12's "Liu 2018" is this Liu/Kidd/Dobosiewicz/Bargmann 2018 paper, then
**the rationale for porting EGL-19+SHK-1+SLO-1 from AWA biology into the
Brian2 AVA model is built on biology that Liu 2018 explicitly characterized
in AWA, not AVA**.

There is also Liu, Chen, Wang (2013) *Nat Commun* 4:1911 "Postsynaptic
current bursts instruct action potential firing at a graded synapse" —
but that one studies neuromuscular junction (body-wall muscle as
postsynaptic), not AVA. Note: this is 2013 (with 2014 corrigendum), not
2018, so it cannot be "Liu 2018."

A separate candidate is Liu, Chen, Mailler, Wang (2017) *Nat Commun* 8:14818
on antidromic-rectifying gap junctions — used patch-clamp on AVA but is
about gap-junction electrical coupling, not intrinsic AVA plateau dynamics.

### Nicoletti 2024 — actual experimental basis for AVA modeling

**Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G, Filippi S
(2024). Biophysical modeling of the whole-cell dynamics of C. elegans
motor and interneurons families.** *PLOS ONE* 19(3):e0298105.
DOI: `10.1371/journal.pone.0298105`. PMCID: PMC10980225.

**Q. Liu** (the AWA spiking author, Liu/Kidd/Dobosiewicz/Bargmann) is a
co-author of Nicoletti 2024. This explains the lab connection — the
Bargmann-trained worm electrophysiologist Q. Liu is providing the
experimental recordings that anchor Nicoletti's models.

The Liu et al. reference [29] in Nicoletti 2024 (cited as the AVA
electrophysiology source) was not extractable from the truncated PDF
content I could fetch. **Acquiring the actual reference [29] is a
prerequisite for clean Wave 2 grounding.** Recommended source: download
the Nicoletti 2024 PDF locally and read the reference list directly.

**What Nicoletti 2024 actually says about AVA is critical:**

> "The current-clamp responses of both neurons [AVAL, AVAR] are
> characterized by a slow-rising phase (~200 ms) followed by a stable
> plateau that is sustained until the stimulus is removed."

> "responses to hyperpolarizing and depolarizing stimuli, resembling those
> of a passive RC-circuit."

> "Overall, the I-V curves of AVAL and AVAR display linear behavior. Taken
> together with computational studies, this might suggest that the
> spontaneous bimodal distribution of the AVA voltage observed
> experimentally is more likely related to a bistable synaptic input than
> to the physiological properties of the neurons."

**Key takeaways from Nicoletti 2024 on AVA:**

1. AVA's response is *graded / passive RC-circuit-like*, not regenerative.
2. The "plateau" Nicoletti describes is the **steady-state V under sustained
   current injection** — a passive plateau, not an active regenerative one.
   It "is sustained until the stimulus is removed" — i.e., it does NOT
   self-terminate, it just decays when injection stops.
3. AVA's I-V curves are linear (not bistable / not regenerative).
4. Bistable behavior in vivo is attributed to **synaptic input**, not
   intrinsic membrane properties.

**Nicoletti's AVA channel set (from `AVAL_simulation_iclamp.py` source)**:

```python
soma.insert('irk')
soma.insert('leak')
soma.insert('egl19')
soma.insert('nca')
```

**Only 4 channels: IRK, LEAK, EGL19, NCA.** No SLO-1, no SHK-1, no SHL-1,
no KQT-3. Current injection: 1000 ms (matching the "sustained until removed"
plateau description). The "600 ms" figure in the architectural plan is
**not** a Nicoletti AVA plateau target — Nicoletti uses 1000 ms steps and
extracts steady-state V from a 40-ms window during the plateau.

---

## Citation chain audit

The architectural plan's `phase_v_w2_architectural_plan.md` references
"Mellem 2008 plateau (20 mV / 600 ms)" in seven places:

```
Line 112: 2b reproduces Mellem 2008 plateau (20 mV / 600 ms) and SLO-1-dominated termination
Line 177: AVAL under current-clamp injection reproduces Mellem 2008 voltage-clamp plateau dynamics
Line 178: Plateau amplitude 15-25 mV (target 20)
Line 179: Plateau duration 400-800 ms (target 600)
Line 183: leak τ ≈ 10 ms vs τ_d ≈ 20 ms vs Mellem 600 ms target
Line 242: imported and validated against Mellem 2008 + Nicoletti 2024 traces
Line 275: Nicoletti's models don't reproduce Mellem 2008 cellular targets
Line 281: dynamics still wrong / Mellem 2008 600 ms target
Line 294: current-clamp plateau dynamics against Mellem 2008
```

**No `phase_v_w2_*.md` document cites a specific figure or table in
Mellem 2008 with the 20 mV / 600 ms numerical pair.** The values appear
only as architectural-plan declarations.

**`digitize_panels.py:474-477` (run #1 phase β-pre work) records:**

> "Mellem 2008 fallback was not pursued because Nicoletti 2024 alone provides
> three high-quality experimental-overlay panels covering the cells of
> interest."

So the Mellem 2008 plateau trace was *never* digitized for the project;
the only digitized panels are RIM and AIY voltage-clamp I-V curves from
Nicoletti 2024. The "Mellem 2008 published plateau trace ... 1
representative trace (digitized at `published_traces.json`)" claim in
`speculative/training_data_feasibility.md:30` is internally inconsistent
with `digitize_panels.py` — that JSON contains Nicoletti panels, not Mellem.

**Most plausible reconstruction of how the misattribution arose:**

1. Wave 1 cellular-validation work surfaced a leak τ vs τ_d mismatch in
   single-compartment scaffold, framed as "plateau too short."
2. The architectural plan adopted "Mellem 2008 plateau" as a target name
   informed by general C. elegans plateau literature (which is anchored
   in RMD via Mellem 2008, then loosely extended to "command interneurons"
   via reviews like Lockery & Goodman 2009 and Schultheis 2011).
3. The numerical values "20 mV / 600 ms" appear to derive from Nicoletti
   2024's voltage-clamp protocol duration (600 ms steps in the 2019 paper)
   and AVAL's depolarization range under current injection (~-25 mV rest
   to ~-5 mV peak ≈ 20 mV) — these *Nicoletti 2024* numbers were attached
   to the *Mellem 2008* citation in the plan's prose.
4. Subsequent Phase α / β / β-pre / γ work blocks adopted the target
   without re-checking the source.

This is the same class of citation propagation as the Nicoletti-2019-PCBI
misattribution that v3 corrected in 2026-04-26 (`digitize_panels.py:470-474`).
The pattern is: a citation appears in an early-document prose, downstream
work adopts the target without re-verification, then verification surfaces
the error mid-flight.

---

## What this implies for Wave 2

**Phase F (Gate 2b) failure is the same empirical observation regardless
of citation:** Brian2 single-compartment AVA produces 46.8 mV / 21.4 ms
under 50 pA × 100 ms injection. That phenotype is real.

**What changes is the interpretation of that phenotype:**

| Claim | Status given Mellem misattribution |
|---|---|
| "Brian2 model fails to reproduce Mellem 2008 in AVA" | **Invalid** — Mellem doesn't characterize a plateau in AVA |
| "Brian2 model produces 46.8 mV / 21.4 ms vs target 15-25 mV / 400-800 ms" | Real numbers, but the target source needs re-grounding |
| "Single-compartment AVA architecture is insufficient for Mellem dynamics" | Re-frames as: **"Single-compartment AVA architecture is insufficient for [whatever the actual target is]"** — and that target needs to be specified |
| Density-sensitivity sweep verdict (`VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS`) | Empirical findings stand; "duration fails" is relative to a target that may not be biologically real for AVA |
| Ca-coupling test verdict (`VERDICT_CA_COUPLING_INSUFFICIENT`) | Empirical finding stands; SLO-1 negative-feedback mechanism is real and the conclusion that bulk Ca-pool can't extend plateau in this architecture is robust |
| "Morphology fork triggered by condition 6" | **Premise needs re-examination** — if Mellem isn't the target, what is, and does that target need morphology to reach? |

**The empirical mechanistic findings from Phase F + density sweep + Ca-coupling
test are not invalidated.** What they actually established is:

- Brian2 single-compartment AVA with the 7-channel essential set produces
  amplitude ~25-50 mV depending on density (tunable) and duration ~5-30 ms
  (fundamentally short, not tunable).
- SLO-1 in this architecture is a hyperpolarizing terminator (Ca-coupling
  is negative feedback for V).
- Adding bulk dynamic Ca-pool does not extend duration; it can only shorten
  it further.

These are clean engineering findings about *what the model does*. The
problem is the comparison-target (Mellem 2008) was misattributed.

---

## What the actual target should be — open questions

Three biologically defensible targets are candidates, and the project needs
to commit to one or document why none of them apply:

**Target 1 — Match Nicoletti 2024's AVAL phenotype directly**

Per Nicoletti 2024: AVAL shows "slow-rising phase (~200 ms) followed by a
stable plateau that is sustained until the stimulus is removed." Plateau
amplitude depends on injection current (no specific 20 mV target). AVA's
response is "passive RC-circuit-like" with linear I-V curves.

If this is the target:
- The 4-channel set (IRK, LEAK, EGL19, NCA) is what should be in Brian2's
  AVA model — *not* the 7-channel essential set.
- The "200 ms slow rise" is the relevant time scale, not 600 ms plateau.
- The plateau "sustained until stimulus removed" — i.e., does NOT
  self-terminate — means SLO-1 termination is not part of AVA's biology
  per Nicoletti.
- Brian2's current 21 ms duration with sustained injection is a real
  problem, but the diagnosis is different: the cell is *terminating
  during the stim*, not *failing to extend a Mellem plateau*.

This is the **lowest-risk target** and matches what the actual experimental
recordings the published model is fit to characterize.

**Target 2 — Match RMD plateau (the actual Mellem 2008 result)**

If the project wants to demonstrate plateau dynamics from a
single-compartment model, RMD is the experimentally characterized cell.
RMD plateau: "long-lasting" (no specific ms), V relaxes to ~-10 mV from
rest at -73 mV (~63 mV depolarization, much larger than the 20 mV claimed
for AVA), Ca-dependent (TTX-insensitive, Ca2+-removal abolishes), KO data
suggests multiple Ca channels contribute.

Nicoletti 2024 has an RMD model from the 2019 paper (Nicoletti, Loppini,
Chiodo, Folli, Ruocco, Filippi 2019, PLOS ONE). If RMD plateau is the
target:
- Switch the validation cell from AVAL to RMD.
- Use Nicoletti 2019's RMD channel set, not AVA's.
- The 20 mV / 600 ms numbers are still not the right targets — Mellem 2008
  doesn't quantify RMD plateau amplitude/duration with those values.

**Target 3 — Match a different paper that does characterize AVA plateau**

Possible candidates:

- **Gao et al. 2015** (*Nat Commun* 6:6323) "The NCA sodium leak channel
  is required for persistent motor circuit activity that sustains
  locomotion." Reports AVA "post-stimulation discharge increase" in 3 of 4
  recordings under whole-cell voltage clamp with ChR2 stimulation. This
  may be the actual experimental basis for the "AVA persistent activity"
  framing if anything is.

- **Liu et al. (Liu Q in Nicoletti 2024 ref [29])** — the actual experimental
  recordings used for Nicoletti's AVA model fitting. Need the full reference.

- **Lindsay, Lockery et al. patch-clamp on AVA** — characterizes AVA's
  response to optogenetic input from ASH but doesn't characterize intrinsic
  plateau dynamics.

If the target is "AVA persistent activity" per Gao 2015 or similar, the
biology and target numbers need to be re-derived from those primary
sources. The architectural plan's "20 mV / 600 ms" pair is unlikely to
be sourceable to any specific primary paper without further verification.

---

## What I did NOT do (and why)

The work block specified producing a classification verdict (A: SPATIAL_DOMINANT,
B: MECHANISTIC_DOMINANT, C: COMBINATION_REQUIRED, D: UNCLEAR). **I did not
produce that verdict** because the underlying question — "what mechanism
produces Mellem's 600 ms plateau in AVA?" — is not answerable: Mellem's
2008 paper documents that there *is no* 600 ms plateau in AVA in their
recordings.

The classification framework is well-designed for a real biological target.
It just needs a real target to be applied to.

If forced to apply the framework to *RMD* (Mellem's actual plateau cell):
classification would lean **B (MECHANISTIC_INGREDIENT_DOMINANT)** — Mellem's
ion-substitution data shows Ca2+-dependence as load-bearing, and KO data
shows multiple Ca channels contribute (suggesting persistent Ca current
and possibly CICR). But this would require also abandoning the AVA cell
target and the 7-channel-essential-set commitment, which is a large
architectural pivot that warrants cross-session deliberation, not
autonomous classification.

---

## Recommended morning-review questions

1. **Was the "Mellem 2008 plateau in AVA" target deliberately chosen or
   inherited?** If deliberately, what was the source? If inherited, when
   does the citation chain originate?

2. **What is the actual biological target for Wave 2's Phase γ Gate-2b?**
   Three candidates above (Nicoletti AVAL phenotype / Mellem RMD / Gao
   AVA persistence) plus possibly a fourth not yet identified. Each
   implies different validation criteria, different cell, possibly
   different channel set.

3. **Should the 7-channel essential set be revisited?** The current set
   (EGL-19, SLO-1iso, SLO-1+EGL-19, SHK-1, SHL-1, NCA, KQT-3) has the
   provenance of Nicoletti's broader cell library plus AWA spike-mechanism
   biology imported from Liu 2018. If the target is Nicoletti's AVAL,
   the channel set should drop to 4 (IRK, LEAK, EGL19, NCA) — the
   Brian2 work already done on SLO-1, SHK-1, SHL-1, KQT-3 is preserved
   for other cells (AIY, RIM, RMD, etc.) but not used in AVA.

4. **Should the morphology fork be deferred until target is re-grounded?**
   The architectural plan's morphology fork is a 3-4 week commitment
   triggered by 2b failure. If 2b is failing against an unverified target,
   the trigger should be re-examined before the commitment.

5. **What is the actual Phase F success criterion for the AVA single-
   compartment Brian2 model?** Several options:
   - Match Nicoletti's NEURON AVAL trace within X% (apples-to-apples
     with same channel set, since Nicoletti's NEURON model IS the
     reference).
   - Match Nicoletti's published AVAL voltage-clamp I-V curves (already
     done for Phase F 2a).
   - Match a specific experimental trace from Liu Q (Nicoletti's ref [29]).

6. **Does paper 3 manuscript still need to "validate against Mellem 2008"?**
   If Mellem 2008 isn't characterizing AVA, the manuscript framing should
   be honest about this. Paper 3 can still cite Mellem 2008 in the
   introduction (as the foundational C. elegans plateau paper, RMD
   biology) but should not claim its model is validated against Mellem
   2008's AVA recordings — there are none.

7. **Does this affect the broader Wave 2 architectural commitment?**
   Path A (import Nicoletti) is unchanged in its premise. The question
   is whether Phase γ's gate criteria need re-specification.

---

## Strategic implication

This is a **methodology success** for the cross-session adversarial review
pattern: the pre-Wave-2 architectural plan locked in a target ("Mellem 2008
20 mV / 600 ms in AVA") that subsequent work blocks adopted without
re-verification. Phase F's empirical findings then confronted reality with
that target and produced "condition 6 surfaces" — which the spec correctly
escalated to morning review rather than autonomous architectural commitment.
The literature investigation work block then verified the target itself.

Without this pause-and-verify step, Wave 2 could have committed 3-4 weeks
to a morphology fork chasing a target that doesn't biologically exist for
AVA. Catching this here saves that misallocation and preserves the
empirical findings (which are clean engineering observations about the
model's behavior, regardless of comparison target).

This also generalizes: any other "target" claimed against primary literature
in the architectural plan should be verified before next architectural
commitments. The Nicoletti-2019-PCBI misattribution (already corrected in
v3) and now the Mellem-2008-AVA misattribution are two instances of the
same propagation error pattern.

---

## Files of relevance

**Where the misattribution lives:**
- `brain/artifacts/phase_v_w2_architectural_plan.md:112,177-179,242,275,281,294`
- `brain/wave2/plateau_harness.py:58,348,610`
- `brain/wave2/sensitivity_sweep.py:2,28,100,434`
- `brain/wave2/run_ca_coupling_test.py:7,13,64,77,118,285,515,527,560,562,564,573,586`
- `brain/wave2/validate_cp3_egl19_cell.py:8,24`
- `brain/wave2/phase_alpha_report.md:142,248,250`
- `brain/wave2/speculative/multi_compartment_explicit.md:48,89,91,93,127,128,161,174`
- `brain/wave2/speculative/training_data_feasibility.md:26,30,33,45,110,144,146,151,152,162`
- `brain/wave2/speculative/gnn_architecture_sketch.md:5,14`
- `brain/wave2/speculative/speculative_summary.md:13,29,42`

**Empirical findings that are independent of the misattribution:**
- `brain/wave2/artifacts/phase_beta_run2_summary.md` — Phase F 2a/2b results
- `brain/wave2/artifacts/gate2_ava_cell_construction.md` — cell construction
- `brain/wave2/artifacts/density_sensitivity_analysis.md` — density sweep
- `brain/wave2/artifacts/ca_coupling_test_results.md` — Ca-coupling test
- `brain/wave2/artifacts/phase_beta_findings.md` — F1-F17 catalog

**Primary sources accessed:**
- Mellem 2008 PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2697921/
- Lockery & Goodman 2009 PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3951993/
- Nicoletti 2024 PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10980225/
- Liu (Q) Kidd Dobosiewicz Bargmann 2018 Cell:
  https://www.cell.com/cell/fulltext/S0092-8674(18)31034-1
- Gao et al. 2015 Nat Commun (search-result citation):
  https://www.nature.com/articles/ncomms7323

**Local source files inspected:**
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAL_simulation_iclamp.py`
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/README.md`

---

*Standing by for cross-session review of citation chain + target re-grounding
decision before classification verdict can be meaningfully produced.*
