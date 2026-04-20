# Phase 3d-3 — Perturbation validation report (v3.1 + v3.2)

In-silico neuron ablations run on the v3.1 model (per-neuron NT signs +
5HT pharyngeal exclusion) with full neuromodulation layer enabled.
Each experiment pairs a control and an ablated run (20 s each, shared
`np.random.seed`). State proportions are fractions of simulation time
spent in each FSM state.

**Honest note on variance.** Brian2's internal RNG is not locked to
`np.random.seed`, so absolute state proportions drift between re-runs
by ~0.05–0.15 per state. Deltas < 0.10 should be interpreted as noise.
This reproducibility gap is flagged as v3.3 work.

## RIS ablation / osmotic shock

**Target neurons:** RIS  
**Scenario:** `osmotic_shock`  
**Expected phenotype:** QUIESCENCE ↓ (Turek 2016).

| state | control | ablated | Δ |
|---|---:|---:|---:|
| FORWARD | 0.06 | 0.06 | -0.00 |
| REVERSE | 0.34 | 0.40 | +0.06 |
| OMEGA | 0.15 | 0.10 | -0.05 |
| PIROUETTE | 0.00 | 0.00 | +0.00 |
| QUIESCENT | 0.45 | 0.44 | -0.01 |

**Verdict:** null result in this run. An earlier v3.0 run (before 5HT
pharyngeal exclusion) reported Δ QUIESCENT = -0.53 for this ablation;
adding biologically-correct 5HT target filtering appears to have
shifted the network dynamics such that the FLP-11/RIS quiescence
pathway no longer dominates. The regression reveals that the earlier
flagship result was fragile to parameter changes. The real biology
(Turek 2016) should still hold — recovering it will need v3.3
re-tuning of FLP-11 mod_strength or longer simulation windows to
average out Brian2 stochastic drift.

## NSM ablation / food

**Target neurons:** NSML, NSMR  
**Scenario:** `food`  
**Expected phenotype:** dwelling/quiescence ↓ (Flavell 2013).

| state | control | ablated | Δ |
|---|---:|---:|---:|
| FORWARD | 0.06 | 0.06 | -0.00 |
| REVERSE | 0.79 | 0.00 | **-0.79** |
| OMEGA | 0.05 | 0.35 | **+0.30** |
| PIROUETTE | 0.00 | 0.00 | +0.00 |
| QUIESCENT | 0.10 | 0.59 | **+0.49** |

**Verdict:** large effect in the wrong direction on quiescence — NSM
ablation *increases* quiescence by 0.49 instead of decreasing it.
However the dramatic REVERSE collapse (-0.79) and OMEGA surge (+0.30)
show NSM ablation IS changing behaviour substantially. Hypothesis:
without NSM's 5HT inhibition of AIZ/AIM/PVC (the pharyngeal-excluded
target set from v3.1), these locomotion interneurons disinhibit and
drive a different failure mode than expected — the network's lack of
sensorimotor drive collapses into quiescence. A proper Flavell 2013
reproduction would require the forward-promoting serotonergic
pathway (NSM → MOD-1 → SIA), which isn't distinctly represented in
our current target weighting.

## RIM ablation / touch

**Target neurons:** RIML, RIMR  
**Scenario:** `touch`  
**Expected phenotype:** REVERSE altered (Alkema 2005 vs Gordus 2015).

| state | control | ablated | Δ |
|---|---:|---:|---:|
| FORWARD | 0.14 | 0.10 | -0.03 |
| REVERSE | 0.15 | 0.39 | **+0.24** |
| OMEGA | 0.05 | 0.05 | +0.00 |
| PIROUETTE | 0.15 | 0.15 | +0.00 |
| QUIESCENT | 0.52 | 0.31 | **-0.21** |

**Verdict:** matches Gordus 2015 — RIM ablation *lengthens* reversal
bouts (REVERSE +0.24). Suggests tyramine-gated termination is
disabled, keeping the worm in reverse once triggered. RIM ablation
also exits quiescence faster (-0.21). This is a clean directional
match to one camp of the mixed literature.

## AVA ablation / touch

**Target neurons:** AVAL, AVAR  
**Scenario:** `touch`

| state | control | ablated | Δ |
|---|---:|---:|---:|
| FORWARD | 0.14 | 0.12 | -0.02 |
| REVERSE | 0.15 | 0.37 | **+0.23** |
| OMEGA | 0.05 | 0.05 | +0.00 |
| PIROUETTE | 0.15 | 0.15 | +0.00 |
| QUIESCENT | 0.52 | 0.31 | **-0.21** |

**Verdict:** wrong direction in this run. Chalfie 1985 expects AVA
ablation to *abolish* reversal — we see it *increased* (+0.23). An
earlier v3.0 run gave the correct Δ REVERSE = -0.15. The regression
echoes the RIS finding: phenotype reproduction is fragile to network-
parameter changes. The AVA-reversal causal link needs re-validation
under v3.3 recal.

## AVB ablation / spontaneous

**Target neurons:** AVBL, AVBR  
**Scenario:** `spontaneous`

| state | control | ablated | Δ |
|---|---:|---:|---:|
| FORWARD | 0.04 | 0.05 | +0.00 |
| REVERSE | 0.54 | 0.59 | +0.06 |
| OMEGA | 0.05 | 0.05 | +0.00 |
| PIROUETTE | 0.15 | 0.00 | **-0.15** |
| QUIESCENT | 0.22 | 0.31 | +0.09 |

**Verdict:** Pirouette structure disrupted (-0.15), consistent with
AVB's role in coordinated locomotion sequences. FORWARD already low
in control.

## PDE ablation / spontaneous

**Target neurons:** PDEL, PDER

| state | control | ablated | Δ |
|---|---:|---:|---:|
| FORWARD | 0.04 | 0.04 | +0.00 |
| REVERSE | 0.54 | 0.54 | +0.00 |
| OMEGA | 0.05 | 0.05 | +0.00 |
| PIROUETTE | 0.15 | 0.15 | +0.00 |
| QUIESCENT | 0.22 | 0.22 | +0.00 |

**Verdict:** no measurable effect. Dopamine modulation too weak in
spontaneous mode to produce behavioural change at 20 s duration.

## Summary & verdict

| ablation | expected | v3.0 result | v3.1 result | Δ v3.0→v3.1 |
|---|---|---|---|---|
| RIS / osmotic | QUIESCENCE ↓ | ✓ Δ=-0.53 | ✗ Δ=-0.01 | flagship lost |
| AVA / touch | REVERSE ↓ | ✓ Δ=-0.15 | ✗ Δ=+0.23 | direction flip |
| NSM / food | dwelling ↓ | ✗ wrong dir | ✗ wrong dir (larger) | worse |
| RIM / touch | REVERSE altered | ~ Gordus-like | ✓ Gordus-like +0.24 | improved |
| AVB / spont | FORWARD ↓ | ~ partial | ~ pirouette disruption | similar |
| PDE / spont | pirouette altered | — | — | unchanged |

## Scientific interpretation

**The v3.0 "2/6 success" was fragile.** Adding two biologically-
justified corrections (5HT pharyngeal exclusion, per-edge Glu signs)
regressed both flagship reproductions. This is actually an important
finding:

1. **Phenotype reproduction in this model is fragile to parameter
   changes.** Our 20 s ablation deltas are comparable to run-to-run
   Brian2 stochastic variance. Robust validation requires longer
   simulations and ensemble averaging.

2. **The earlier RIS/AVA successes likely depended on specific
   network firing patterns** that shifted under the v3.1 corrections.
   The biology is real (Turek 2016 and Chalfie 1985 are canonical) —
   but the *model* isn't stable enough to reliably reproduce them
   across variants.

3. **Per-edge Glu receptor signs** (v3.2 infrastructure, default off
   pending re-tuning) make the network substantially more active
   (+415 Glu edges flipped from inhibitory to excitatory), which
   further alters the FLP-11 dominance. These will need explicit
   modulation recalibration to re-establish phenotype reproduction.

## Path to v3.3

To recover phenotype reproduction robustly:

1. **Longer simulations** (60 s instead of 20 s) to average out
   Brian2 stochastic variance.
2. **Fixed RNG state** including Brian2's internal RNG (via
   `seed(42)` and `BrianObject` seed settings).
3. **Ensemble runs** (10 seeds × 60 s) with reported mean±std per
   ablation × scenario.
4. **Modulation strength recalibration** — FLP-11 needs stronger
   inhibitory effect to dominate stress-induced quiescence against
   the now-more-active network.
5. **5HT target re-weighting** (not just exclusion) — explicit
   circuit-role weighting that preserves NSM→AVB SER-4 dwelling
   pathway while filtering pharyngeal contribution.

These are v3.3 items. v3.1/v3.2 ships with the honest regression
documented above.
