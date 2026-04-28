# Density-sensitivity analysis — Phase F Component 2b
## Verdict: VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS (Condition 6 confirmed via duration)

**Date:** 2026-04-26
**Trigger:** Phase β run #2 — Component 2b produced 46.8 mV / 21.4 ms vs Mellem
2008 targets 15-25 mV / 400-800 ms. Before authorizing the morphology fork, sweep
the 5 non-Nicoletti-AVA channel densities to distinguish density-tunable from
truly architecture-insufficient.

**Driver:** `wave2/sensitivity_sweep.py`
**Raw data:** `wave2/artifacts/density_sensitivity_results.json`

---

## Methodology

**Sweep axes** (over factors applied multiplicatively to the Phase F baseline):

- **Axis 1 — terminator block:** scale `g_slo1iso` and `g_slo1egl19` together
  by factor in `{0.5, 1.0, 2.0, 4.0}` for the main grid, plus `{0.25, 8.0}`
  extension at `kv=4.0`.
- **Axis 2 — voltage-gated K block:** scale `g_shk1`, `g_shl1`, `g_kqt3`
  together by factor in `{0.5, 1.0, 2.0, 4.0}` for the main grid, plus
  `{0.25, 8.0}` extension at `term=1.0`.

**Held fixed at Nicoletti's published AVAL values throughout (principled):**

- `g_leak = 1.336e-5` S/cm² (Nicoletti AVAL g0)
- `g_egl19 = 9.288e-6` S/cm² (Nicoletti AVAL g0)
- `g_nca = 0.0` (Nicoletti AVAL g0)
- `e_leak = -39 mV`, `eca = +60 mV`, `ek = -80 mV`
- `v_init = -60 mV`, `cai_static = 5e-5 mM`

**Mellem 2008 protocol** (identical to Phase F 2b for direct comparability):

- 200 ms settle at `I = 0`
- 100 ms × 50 pA injection
- 1500 ms post-stim recovery
- Brian2 RK4, `dt = 0.025 ms`

**Pass criteria:**

- amplitude in `[15, 25]` mV
- duration in `[400, 800]` ms
- both required for `arch_pass`

**Diagnostic:** release-tau ratio against pure-leak τ_m = 64.3 ms. Signature
classified `active_termination` if ratio < 0.6, `leak_dominated` if 0.6 ≤
ratio < 1.4, `no_termination` if ratio ≥ 1.4.

**Sanity:** the `(term=1.0, kv=1.0)` baseline reproduced Phase F's published
result to floating point: `46.85 mV / 21.4 ms`.

---

## Results — main 4×4 grid

Format: `amp_mV / dur_ms / signature` per cell. Bold = amplitude in `[15, 25]`
target. No cell hits duration target.

| term \ kv | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|
| **0.5** | 52.91 / 29.5 / no_term | 47.90 / 21.3 / no_term | 39.21 / 15.4 / no_term | 26.40 / 9.0 / no_term |
| **1.0** | 50.13 / 29.8 / no_term | 46.85 / 21.4 / no_term | 39.01 / 15.4 / no_term | 26.39 / 9.0 / no_term |
| **2.0** | 46.73 / 30.1 / no_term | 45.23 / 21.6 / no_term | 38.64 / 15.4 / no_term | 26.38 / 9.0 / no_term |
| **4.0** | 42.98 / 30.0 / no_term | 43.04 / 21.8 / no_term | 37.99 / 15.4 / no_term | 26.34 / 9.0 / no_term |

**Range over the grid:**

- amplitude: 26.34 – 52.91 mV  (target 15-25 mV: never reached in main grid)
- duration: 9.0 – 30.1 ms  (target 400-800 ms: max is **30 ms**, more than 13×
  short of the lower bound)

## Results — extension probes

Wider factor range to confirm the trend extrapolates monotonically.

| Probe | term | kv | amp (mV) | dur (ms) | signature |
|---|---|---|---|---|---|
| Push amp down further | 1.0 | **8.0** | **17.71** | **4.4** | active_termination |
| Push amp up | 1.0 | 0.25 | 50.44 | 42.4 | no_termination |
| Push terminator very low | 0.25 | 4.0 | 26.41 | 9.0 | no_termination |
| Push terminator very high | 8.0 | 4.0 | 26.28 | 9.0 | no_termination |

**Key observations:**

1. At `kv=8.0` the amplitude finally enters target range (17.7 mV ∈ [15, 25]),
   but **duration drops to 4.4 ms** — fundamentally incompatible with the
   400-800 ms target.
2. At `kv=0.25` (lowest Kv tried) duration peaks at **42 ms** — still 10× short
   of target, and amplitude is way too high (50 mV).
3. **Terminator scaling is essentially irrelevant.** Comparing `term=0.25` vs
   `term=8.0` at fixed `kv=4.0`: amplitude differs by 0.13 mV, duration is
   identical (9.0 ms). A 32× variation in SLO-1 conductance produces zero
   meaningful change in the plateau phenotype.

---

## Verdict assignment

Per the spec's classification:

- ✗ `VERDICT_DENSITY_TUNABLE` — rejected. No combination achieves amplitude AND
  duration in target ranges simultaneously.
- ✓ **`VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS`** — confirmed. The `kv=8.0,
  term=1.0` combination achieves amplitude pass (17.7 mV ∈ [15, 25]) but
  duration of 4.4 ms — far below the 200 ms threshold the spec uses for the
  duration-fails sub-clause, let alone the 400 ms lower target. Across the
  entire sweep (main grid + extension probes), the **maximum duration ever
  achieved is 42.4 mV (`term=1.0, kv=0.25`)**, with amplitude 50.4 mV. The
  maximum duration among `amp_pass` cells is **4.4 ms**.
- ✗ `VERDICT_DURATION_TUNABLE_AMPLITUDE_FAILS` — rejected. No combination
  achieves duration in `[400, 800]` ms regardless of amplitude.
- ✗ `VERDICT_NEITHER_TUNABLE` — partially overlapping but the spec's amplitude-
  tunable-duration-fails clause is more specific given the kv=8 amplitude pass.

---

## Mechanistic interpretation

The data tell a clean story. At fixed 50 pA injection, after the cell reaches
quasi-steady-state during the stim:

- **Amplitude** is set by the steady-state balance of the depolarizing current
  (50 pA injection + EGL-19 Ca current at the depolarized v) against the active
  K conductance pool. Increasing Kv conductance pulls the steady-state V down
  monotonically. This is straightforward IV-curve algebra.
- **Duration** is set by how long the cell sustains depolarization above the
  +5 mV threshold *after* stim release. With the stim removed, the cell's
  recovery is governed by the time constant of its hold:
    - At low Kv, V holds high during the stim because EGL-19's slow Ca current
      and accumulated SLO-1 activation almost balance leak — but this hold
      collapses on stim release because the active K currents are too weak to
      feed back without injection. Recovery is governed by the cell's effective
      RC, which is dominated by passive properties at this membrane condition.
    - At high Kv, V never reached high enough during stim to engage SLO-1
      meaningfully; recovery on release is fast.

In neither regime does the architecture support a sustained, self-maintaining
plateau. The spec's three diagnostic notes from `gate2_ava_cell_construction.md`
hold up empirically:

> Mellem 2008 plateau is a 600 ms graded depolarization phenomenon that relies
> on Ca-induced Ca-release and subsequent SLO-1 activation. Our cell has no
> dynamic Ca pool, so SLO-1's Ca-feedback can't operate.

Specifically:

- **SLO-1 isolated reads bulk `cai_static = 5e-5 mM`** (per finding F12); it
  delivers a *constant* K conductance whose only V-dependence is through its
  own gating, not through Ca-feedback. Scaling its `gbar` only reweights this
  steady contribution — it cannot create dynamics.
- **SLO-1+EGL-19 coupled** uses a deterministic V-dependent `calcium(V)`
  formula (per finding F13), so it too contributes a deterministic, V-driven
  K conductance with no positive feedback loop.
- **No EGL-19 → cai → SLO-1 feedback loop exists** in the current single-
  compartment essential set. The architecture lacks the dynamic Ca pool that
  would let the depolarizing Ca current and the K terminator interact on the
  hundreds-of-ms timescale Mellem 2008 reports.

The terminator-axis having near-zero leverage on the phenotype is the load-
bearing finding here: it directly demonstrates that **adding more SLO-1
conductance does not extend the plateau**, because the missing ingredient is
not "amount of SLO-1" but "Ca dynamics that drive SLO-1 over hundreds of ms".

The `no_termination` signature throughout the central grid (release τ ≈ 180-630
ms vs leak τ_m = 64 ms; ratio 2.8-9.8) means after stim release the cell
*keeps* depolarizing/holding for longer than passive leak would predict.
This is the active-current hold on V: EGL-19 stays partially open at depolarized
v and contributes inward Ca current that fights leak. This is the opposite of
Mellem 2008's profile (active termination).

---

## Recommendation to morning review

**Condition 6 is empirically confirmed by this sweep**, with high confidence.

- The 5 non-Nicoletti density parameters together cannot produce a Mellem-style
  plateau in the current single-compartment AVA architecture, even with a 32×
  range on terminators and 32× on Kv (combined 1024× variation in the search
  space).
- The duration ceiling across this entire density volume is ~42 ms — an order
  of magnitude short of even the lowest plausible biological plateau.
- The amplitude tradeoff vs duration is monotonic and tight: the only way to
  pass amplitude (kv=8) is via Kv that crushes duration to <5 ms.
- The mechanism of the failure is identifiable and structural: missing dynamic
  Ca pool → SLO-1 isolated cannot mediate Ca-feedback → no slow positive-feedback
  loop sustains a plateau.

**Therefore the morphology fork's pre-condition is met.** This sweep does not
*select* a fork direction; that remains the user's decision. But it cleanly
rules out "wrong densities masquerading as architectural insufficiency" within
the bounded search the spec authorized.

Possible architectural responses (NOT autonomously chosen, in priority order
that the data support):

1. **Add a dynamic Ca-pool (caintra1) so SLO-1 isolated has Ca-feedback.** This
   is the *cheapest* architectural extension that addresses the identified
   mechanism. Per F12 it diverges from Nicoletti's published AVA insertion
   set, but Mellem 2008 measures biological reality, not Nicoletti's reduced
   model. Could be tested against this same sweep harness.
2. **Multi-compartment morphology (the spec's "morphology fork").** ~3-4 weeks
   per architectural plan. Justified by this sweep, but option (1) is a cheaper
   first probe.
3. **Re-investigate Mellem 2008's exact protocol.** Different temperature, drug
   conditions, or cell preparation may make our 50 pA / 100 ms protocol not
   directly comparable. Worth a literature pass before committing to (1) or
   (2).
4. Density optimization via Nicoletti's `g_to_Scm2` workflow — likely
   insufficient given this sweep's coverage, but a more rigorous L-BFGS sweep
   over the 5-D space (rather than 2 grouped axes) would close the loop.

**The principled-density channels (EGL-19, NCA, leak) were not varied,
respecting the spec.** Should the user wish to question Nicoletti's AVAL
densities directly, that is a separate work block with stronger justification
required.

---

## Files produced

```
wave2/
├── sensitivity_sweep.py                                 [driver script]
└── artifacts/
    ├── density_sensitivity_analysis.md                 [this file]
    └── density_sensitivity_results.json                [raw sweep + extension data]
```

The driver supports re-running with different factor lists by editing
`term_factors` and `kv_factors` near the top of `main()`.

---

*End of density-sensitivity analysis. Standing by for morning review of
verdict + architectural-response selection.*
