# Phase G — halothane dose-response on minimal LIF demo

**Substrate:** 50-neuron Brian2 LIF demo (40 E + 10 I), recurrent E↔I.

**Perturbation:** AnestheticPerturbation (Phase G v1) consumes wave2_overlay_v2.json. Hyperpolarizing currents from complex_i_block + k2p_potentiation + snare_cooperativity + nachr_antagonism additive.

## Dose-response

| dose × EC50 | firing rate (Hz) | n_spikes | complex_i max | K2P max | GABA max | SNARE max | nAChR max | GluCl max | hyperpol (pA) |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 51.00 | 5100 | 0.333 | 0.262 | 0.184 | 0.073 | 0.384 | 0.333 | -58.5 |
| 0.003 | 39.00 | 3900 | 0.599 | 0.516 | 0.403 | 0.190 | 0.652 | 0.599 | -110.6 |
| 0.010 | 24.00 | 2400 | 0.833 | 0.780 | 0.693 | 0.439 | 0.862 | 0.833 | -167.0 |
| 0.030 | 0.00 | 0 | 0.937 | 0.914 | 0.871 | 0.701 | 0.949 | 0.937 | -201.5 |
| 0.100 | 0.00 | 0 | 0.980 | 0.973 | 0.958 | 0.887 | 0.984 | 0.980 | -220.0 |
| 0.300 | 0.00 | 0 | 0.993 | 0.991 | 0.985 | 0.959 | 0.995 | 0.993 | -226.5 |
| 1.000 | 0.00 | 0 | 0.998 | 0.997 | 0.996 | 0.987 | 0.998 | 0.998 | -228.9 |
| 3.000 | 0.00 | 0 | 0.999 | 0.999 | 0.999 | 0.996 | 0.999 | 0.999 | -229.6 |

**Baseline firing rate (lowest dose):** 51.00 Hz

**Demo-network 50%-suppression dose:** ≈ 0.010× clinical EC50 (100× tighter than the Crowder 1996 PMID 8873562 behavioral EC50 anchor at 1× clinical).

## Validation against literature — honest reading

Crowder 1996 reports halothane behavioral EC50 in *C. elegans* at ~3% atm (~280 µM aqueous, = 1× clinical EC50 by Phase D definition). The Phase G demo network's 50%-firing-rate suppression dose at ~0.01× clinical is **100× tighter** than Crowder's behavioral EC50.

**This gap is informative, not a failure.** Two contributing factors:

1. **Binding-side saturation:** wave2_overlay_v2.json has CP7-corrected occupancies that approach 1.0 at clinical EC50 across all 30 Tier-1 targets (8 mechanism classes). At 1× clinical EC50 the binding pipeline reports essentially-full target engagement; the dose-response shape is therefore compressed at the high end. Behavioral EC50 in real *C. elegans* is determined by COUPLING — how target engagement maps onto downstream physiology — not by additional binding to under-saturated targets.

2. **Demo-network coupling sensitivity:** the minimal 50-neuron LIF network is more sensitive to current perturbations than real *C. elegans* (no muscle buffer, no graded-potential redundancy, no neuropeptide modulation). Real behavioral immobilization sits at the intersection of (binding × coupling × behavioral threshold). The demo captures binding × coupling but the threshold is not calibrated.

**Implication for Phase G:** the dose-response curve SHAPE is correct (monotonic suppression of firing rate with increasing engagement). The behavioral EC50 value will require either (a) calibration against LIFBrain with command-interneuron readout to muscle, OR (b) reformulating Phase G to consume Phase F's behavioral threshold layer (which itself is parameter-locked per CP1, so this is not a quick fix).

**Honest verdict:** Phase G demo network produces a *binding-coupled* dose-response curve. Mapping it onto Crowder's behavioral EC50 requires a behavioral threshold calibration that is out of overnight scope.

## Caveats

- Demo network is NOT LIFBrain (Wave 2 production substrate). LIFBrain integration is the next step; deferred to bounded follow-up.
- Phase G v1 uses simplified hand-curated channel expression. CeNGEN-derived per-cell expression (v2) will sharpen target localization.
- Hyperpolarizing currents calibrated to 50 pA per Complex I unit + 30 pA per K2P unit (round numbers); not literature-derived. CP B follow-up: calibrate against measured K-ATP single-channel conductance.
- Dose-response uses additive I_ext rather than connectome W_chem modifications. In LIFBrain, the same AnestheticPerturbation class hooks into W_chem directly via apply_to_brain().
