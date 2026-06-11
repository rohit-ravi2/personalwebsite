# V7 Closeout Addendum — §9.4 open items resolved

**Date:** 2026-06-10
**Scope:** closes the two compute-bound open items from `v7_final_summary.md` §9.4
(mouse bootstrap CIs; LHS for fly + mouse). No model parameters, α, or
perturbation tables were touched — both items are post-hoc statistical
summaries of the **frozen V6 simulation output**. The preregistration hash
`533b624a…` is unchanged and no V7 verdict is reopened.

These are reported here in the same honest-direction discipline as the main
closeout: results are stated where the data landed, not steered toward a
prediction. Both items were explicitly **non-preregistered** (the prereg
scoped mouse CIs as deferred and LHS as worm-only), so neither changes a
V7 gate verdict; they fill in the cross-organism symmetry the page was
missing.

---

## §A — Mouse bootstrap 95% CIs (M5 extension)

**Open item (§9.4):** "Mouse bootstrap CI is not currently computed. The V5 M5
bootstrap artifact contains worm and fly CIs only … the page reports worm + fly
CIs at one location and mouse fold-errors as point predictions."

### A.1 Method and provenance

The original M5 generating script was never committed — only its output JSON
(`artifacts/v5_controls/M5_bootstrap_CIs.json`, commit `b519551`) survived. It
was reconstructed here as `src/state_validation/v7_m5_bootstrap_ci.py`, which
fills that provenance gap as a side effect.

Method, matching the committed M5 `method` string exactly: **95% bootstrap CI,
1000 resamples, on per-dose seed-mean qf, refit EC50 each iteration.** For each
bootstrap iteration the per-seed quiescent-fraction values at each dose are
resampled with replacement (independently per dose), averaged to a seed-mean
curve, and the EC50 is re-extracted with the same log-linear threshold-crossing
fit (`hill_fit_ec50`, threshold 0.5) used everywhere in the pipeline.
`rng_seed = 20260503`, `n_boot = 1000` — identical to the worm/fly run.

### A.2 Reconstruction validation

Before trusting the mouse numbers, the script reproduced the already-published
worm V3 + fly V4 WT CIs from their raw ensemble CSVs:

| condition | committed M5 | reconstruction |
|---|---|---|
| worm halothane WT | 316.9 [296.5, 334.1] | **316.9 [296.5, 334.1]** |
| worm isoflurane WT | 290.8 [276.1, 309.6] | 291.0 [275.8, 309.0] |
| fly halothane WT | 361.2 [343.0, 372.8] | **361.2 [343.0, 372.8]** |
| fly isoflurane WT | 322.9 [314.6, 327.9] | 322.7 [314.8, 328.3] |

Halothane matches to the decimal in both organisms; the ≤0.3 µM isoflurane
differences are RNG-ordering noise (the original interleaved more conditions in
one stream). Methodology confirmed faithful.

### A.3 Mouse V6 result

Added to `M5_bootstrap_CIs.json` as the `mouse_V6` block:

| condition | bootstrap median | 95% CI | published | in CI? |
|---|---|---|---|---|
| mouse halothane WT | 296.9 µM | [289.9, 307.4] | 350 | ✗ |
| mouse isoflurane WT | 273.2 µM | [268.8, 277.6] | 290 | ✗ |

The bootstrap medians sit on top of the existing V6 point predictions
(297.2 / 273.2 µM) — correctly centered, with tight CIs.

**Reading.** Mouse extends the same pattern the M5 commit first flagged for
worm/fly: the model is **precise** (tight CIs) but the published values fall
**just outside** the CIs — predicted low by ~15% (halothane) and ~6%
(isoflurane). This is consistent with the §3.4 / §8.1 finding that mouse
predictive accuracy is magnitude-driven on the generic random graph; it adds
no new claim, it makes the three-organism CI table symmetric. The honest tally
of published-EC50-inside-CI is now **1 / 6** across all WT volatile anchors
(worm isoflurane held-out is the lone hit — the strongest result by this
stricter criterion, as the original commit noted).

---

## §B — LHS sensitivity extended to fly + mouse (M3 extension)

**Open item (§9.4):** "M3 LHS ran worm only per the pre-registration's compute
budget. Extending to fly + mouse would tighten the sensitivity claim
cross-organism but is not required by the pre-registration."

### B.1 Method

Identical LHS protocol to the preregistered worm M3c: 100 joint Latin-Hypercube
samples (`scipy.stats.qmc`, seed 20260502), each perturbing **all** halothane-
active class parameters (EC50 + max_effect) simultaneously within ±50%; 8 doses
× 3 seeds × 30 s at frozen α; 95% CI taken as [2.5, 97.5] percentiles of the
predicted halothane EC50 over the 100 samples.

Driver: `src/state_validation/v7_m3_lhs_extend.py`. It uses one **persistent**
worker pool across all samples (tasks tagged by sample index) rather than the
per-sample pool respawn in `run_lhs`; the LHS factor generation and EC50 fit are
byte-identical, so the numbers match the intended protocol. Verification: the
driver reproduces the preregistered worm CI **[178.4, 447.6], median 311.6 µM**
exactly. 100/100 samples crossed threshold in every organism.

This is reported as an **exploratory extension** in a separate artifact
(`artifacts/v7_sensitivity/v7_sensitivity_lhs_allorganisms.json`); the
preregistered worm-only `v7_sensitivity_verdict.json` (M3c) is left untouched.

### B.2 Result

| organism | LHS median | 95% CI (µM) | CI width | median fold | ⊂ [200, 600]? |
|---|---|---|---|---|---|
| worm (prereg) | 311.6 | [178.4, 447.6] | 269 | 1.09 | no (low tail 178) |
| fly | 328.0 | [273.5, 422.6] | 149 | 1.04 | **yes** |
| mouse | 283.0 | [229.7, 367.6] | 138 | 1.24 | **yes** |

### B.3 Reading

1. **All three medians are well-anchored** (fold 1.04–1.24 vs published). Joint
   ±50% literature-scale uncertainty on every parameter does not de-center any
   organism's EC50 prediction.
2. **The worm M3c lower-tail deviation is worm-specific, not architecture-wide.**
   The closeout (§4.2) reported worm's 95% CI extending below the preregistered
   [200, 600] band (low tail 178 µM) and read it as tail width, not central
   mis-calibration. The extension confirms that reading: fly **[273.5, 422.6]**
   and mouse **[229.7, 367.6]** both sit entirely inside [200, 600]. The
   architecture is not generically brittle under joint parameter uncertainty;
   the wider worm tail is a worm-substrate property.
3. **Worm has the widest CI (269 µM) despite being the calibration anchor.** Fly
   (149) and mouse (138) are roughly half as wide. A plausible mechanistic
   reading: worm's behavioral readout runs over a far smaller command-interneuron
   set (~16 neurons) than fly (~658) or mouse (300), so the seed-mean qf curve is
   less self-averaging and the threshold crossing is more sensitive to joint
   perturbation. This is a hypothesis the data is consistent with, not a tested
   claim.

**Net.** The exploratory cross-organism LHS does not change any V7 verdict, but
it removes the one place a reviewer could over-read worm's M3c deviation as
evidence of architecture-wide parameter fragility: the two non-anchor organisms
are tighter and fully in-band.

---

## Artifact index (this addendum)

| artifact | path |
|---|---|
| Bootstrap CI script (reconstructed + mouse) | `src/state_validation/v7_m5_bootstrap_ci.py` |
| Updated M5 CIs (now incl. mouse_V6) | `artifacts/v5_controls/M5_bootstrap_CIs.json` |
| LHS extension driver | `src/state_validation/v7_m3_lhs_extend.py` |
| Fly LHS raw | `artifacts/v7_sensitivity/v7_sensitivity_lhs_fly.csv` |
| Mouse LHS raw | `artifacts/v7_sensitivity/v7_sensitivity_lhs_mouse.csv` |
| Combined 3-organism LHS summary | `artifacts/v7_sensitivity/v7_sensitivity_lhs_allorganisms.json` |
| Worm LHS (prereg, unchanged) | `artifacts/v7_sensitivity/v7_sensitivity_lhs.csv` |

Both §9.4 compute items are now closed. The remaining §9.4 item — the
interactive `AnesthesiaPipeline.tsx` panel update to surface Sub-Q2 redundancy
and Sub-Q1 percentile histograms — is a front-end task, not a compute gap, and
is left for a page-rewrite session.
