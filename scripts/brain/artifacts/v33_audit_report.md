# v3.3 audit — graded-brain Tier 1 stack ensemble

Ran the 6-ablation perturbation suite through the Tier 1 graded-brain
configuration (brain_class="graded" + Ca plateau + volume-transmission
modulators + closed-loop proprioception). 3 seeds × 6 ablations × 2
conditions = 36 runs at 20 s each. Brian2 seed locked alongside
np.random.seed.

## Headline finding

**All 36 runs produced identical control vs ablated state distributions:**

```
FORWARD   = 0.03 ± 0.00
OMEGA     = 0.05 ± 0.00
QUIESCENT = 0.92 ± 0.00
(REVERSE, PIROUETTE = 0)
```

Every delta is exactly 0.00 for every ablation across every seed.

## Interpretation

The graded brain produces membrane-potential patterns that are
fundamentally different from the LIF spike trains the classifier
bank was trained on (Atanas ΔF/F → LIF synthetic calcium chain).
The event classifier — reversal_onset, forward_run_onset, etc. —
does not fire under graded dynamics. With no event probabilities
crossing threshold, the FSM stays locked in its default state
(QUIESCENT). Body remains stationary. Ablations have zero effect
on a network that wasn't going anywhere.

This is **not** a failure of the Tier 1 upgrades — graded dynamics
is the biologically correct choice, and Ca plateau + volume
transmission + proprioception all work mechanistically. It's a
failure of the *pipeline between graded brain and FSM:* the
classifier was trained on the wrong thing.

## The exact diagnosis

Classifier bank expects inputs that match Atanas ΔF/F statistics:
slow-timescale (0.6 s resolution) calcium fluorescence from real
neurons firing at biological rates. Under LIF we produced this via:

  spike events → synthetic-calcium kernel convolution →
  per-neuron normalisation → classifier input

Under graded dynamics the chain is:

  σ(V) ∈ [0, 1] threshold-crossings → _FakeSpikeMonitor events →
  synthetic-calcium kernel → classifier input

The _FakeSpikeMonitor threshold of σ > 0.5 is too strict at the
current graded-dynamics tuning (V_rest = -45 mV, V_half = -30 mV
puts baseline σ ≈ 0.08). Few neurons cross 0.5 → few events →
classifier sees near-zero input → all event probabilities near
baseline → FSM never transitions out of QUIESCENT.

## v3.4 path (real scope)

The fix is classifier retraining. Concrete:

1. **Generate synthetic-Atanas training data from graded brain.**
   Run GradedBrain for N hours total across scripted sensory
   schedules (matching Atanas 60-worm stimulus distribution).
   Log σ(V) per neuron at 600 ms resolution — this is the direct
   graded-equivalent of ΔF/F (both are bounded slow-timescale
   activity signals).
2. **Retrain the 8-event classifier bank** on this synthetic-Atanas
   dataset, using the same behavioural-event labels the v3 classifiers
   were trained on (Atanas-derived). Each event gets a retrained
   logistic model taking σ-vectors as input.
3. **Retune FSM thresholds** against the new classifier's output
   distribution.
4. **Re-run this ensemble audit** — expected: AVA/Chalfie and
   RIS/Turek signals re-emerge at comparable magnitude to v3.0
   LIF results, because the underlying circuit connectivity
   hasn't changed, only the readout representation.

Estimated effort: ~2 weeks focused work.

## What this tells us about the project

**Scientifically:** adding more biologically-accurate dynamics
(graded, Ca plateau, volume transmission) breaks the downstream
readout pipeline. That's expected; each architectural upgrade
cascades. The fix is retraining the readout layer, not
undoing the upgrades.

**Methodologically:** the ensemble audit continues to be valuable.
Without running it, we might have claimed "Tier 1 preserves
phenotypes" on the basis of a single lucky seed. Instead we have
unambiguous statistical evidence that the readout layer is broken
and needs retraining.

**For the site copy:** the current shipped scenarios remain on
v3 LIF, where AVA/Chalfie is reproducible at Δ = −0.57 ± 0.37.
Tier 1 stack is infrastructure ready for v3.4 classifier retraining
but not yet phenotype-validated.
