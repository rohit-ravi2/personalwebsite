# Wave 2 RIM cell construction (CP4)

**Date:** 2026-04-26
**Module:** `wave2/option_alpha_rim_cell.py`
**Status:** PASS (smoke test)

---

## Channel set (verified primary-source order)

7 channels per `RIM_simulation_iclamp.py` lines 31-38 (insertion order):

```
[shl1, egl2, irk, cca1, unc2, egl19, leak]
```

3 of these are USEION ca: cca1, unc2, egl19.
3 are USEION k: shl1, egl2, irk.
1 is leak (no ion mechanism, just non-specific current).

## Density choices (S/cm² convention, no gScm2 rescale)

Per `RIM_simulation.py` line 25 ("conductances in S/cm^2"), the published g
vector is **already in S/cm²** and is passed directly to
`RIM_simulation_iclamp` without `gScm2()` rescaling.

| Channel | g (S/cm²) | Source position in g |
|---|---|---|
| shl1  | 9.049e-04 | g[0] |
| egl2  | 1.412e-04 | g[1] |
| irk   | 3.273e-04 | g[2] |
| cca1  | 8.452e-04 | g[3] |
| unc2  | 9.677e-05 | g[4] |
| egl19 | 3.201e-04 | g[5] |
| leak  | 9.677e-05 | g[6] |

`eleak = -50 mV` (g[7]), `cm = 1.5 μF/cm²` (g[8]).

## F18-aware eca handling (explicit, value verified)

**RIM uses eca = 60 mV.** This is the empirically-verified runtime value of
`seg.eca` after `h.run()` in NEURON for RIM's section.

This value differs from AIY's eca = 127.59 mV. The reason is the **F18
refinement** discovered during CP4 pre-flight:

- **F18 trigger correction:** the override is NOT triggered by "multiple
  USEION ca" (which is what yesterday's AIY F18 finding initially predicted).
  The actual trigger is **asymmetric USEION declarations across channels** —
  specifically, when at least one channel declares `USEION ca READ eca`
  WITHOUT `WRITE ica` (a "READ-only" Ca reader, e.g. slo1egl19).
- AIY: egl19 (READ eca, WRITE ica) + slo1egl19 (READ eca only). Asymmetric →
  ion_style promoted to Nernst → seg.eca becomes 127.59 mV.
- RIM: cca1, unc2, egl19 — all three have IDENTICAL declarations
  (READ eca, WRITE ica). Symmetric → ion_style preserved → seg.eca stays 60.

**Empirical confirmation:**
- AIY runtime: `seg.eca = 127.5895 mV` (overridden via Nernst)
- RIM runtime: `seg.eca = 60.0000 mV` (preserved, NOT overridden)
- ion_style codes: AIY = 49 (0b110001), RIM = 8 (0b1000)

## UNC-2 GLOBAL handling decision (per CP3)

`unc2.mod` line 19: `GLOBAL minf, hinf, mtau, htau, munc2, hunc2`.

**Decision: no special Brian2 handling required.** All six variables are
either (a) derived assignments computed each DERIVATIVE step from the
cell's own `v` (so single-cell use sees correct values regardless of
GLOBAL declaration) or (b) diagnostic copies of `m, h` written in
BREAKPOINT (informational only). The actual integrated STATE
`{m, h}` is RANGE-by-default per NMODL convention and is per-instance.

In Brian2, every NeuronGroup variable is per-cell. We translated the six
GLOBAL-declared variables as per-cell `: 1` declarations in `UNC2_EQS`.
For our single-NeuronGroup (1 cell) Layer A validation, this matches
NEURON exactly. Future multi-cell-per-Brian2-NeuronGroup deployment will
get correct per-cell semantics for free. No NMODL GLOBAL pitfall surfaces
in our translation.

Documented in `wave2/channels/unc2.py` module docstring.

## Geometry and IC

- `surf_cm2 = 103.34e-8` cm² (neuromorpho RIML)
- `cm_uFcm2 = 1.5`
- `Ra = 100`
- `v_init = -60 mV` (matches `h.finitialize(-60)`)
- All channel gates SS-initialized at -60 mV via `<chan>_init_states`.
- `cai = 5e-5 mM` (NEURON default; no Ca pool)

## Integration choices

- Brian2 `rk4` (matches AIY/AVAL precedent for multi-Ca-channel cells).
- `defaultclock.dt = 0.025 ms` (Brian2 high-resolution).
- For Layer A current-clamp comparison vs NEURON, RIM_simulation_iclamp
  uses `h.dt = 0.04 ms`; we use `dt = 0.04 ms` on both sides for parity.

## Comparison to AIY's construction

| Aspect | AIY | RIM |
|---|---|---|
| Channels | 7 (egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1) | 7 (shl1, egl2, irk, cca1, unc2, egl19, leak) |
| USEION ca | 2 (egl19, slo1egl19) | 3 (cca1, unc2, egl19) |
| USEION ca asymmetry | YES (slo1egl19 reads eca w/o writing ica) | NO (all three READ eca + WRITE ica) |
| Runtime eca | 127.59 mV (Nernst overridden) | 60 mV (preserved) |
| g convention | nS at cell level → gScm2 rescaled | S/cm² already (no rescale) |
| eleak | -89.57 mV | -50 mV |
| cm | 1.6 μF/cm² | 1.5 μF/cm² |
| surf | 65.89e-8 cm² | 103.34e-8 cm² |
| Ca pool | none | none |
| GLOBAL state | none in K channels; F2 pattern in caintra1 (not used in AIY) | unc2 has GLOBAL pitfall (handled per per-cell semantics in Brian2) |

## Smoke test result

100 ms passive (no clamp, no inject), v_init = -60 mV:
- Final V: -43.68 mV (cell settles depolarized vs. rest because RIM has
  significant tonic CCA-1 ica from -60 mV gating + eleak = -50 mV that
  isn't strongly hyperpolarizing).
- Per-channel currents at t=100 ms (mA/cm²):
  - ica_cca1 = -2.226e-03 (strong inward; CCA-1 partial activation)
  - ica_unc2 = -3.750e-05 (small inward; UNC-2 partial activation)
  - ica_egl19 = -1.676e-04 (moderate inward)
  - ik_shl1 = +8.254e-04 (outward A-type K)
  - ik_egl2 = +4.000e-04 (outward EAG K)
  - ik_irk = +5.930e-04 (outward inward-rectifier — note SS at depol is small)
  - i_leak = +6.113e-04 (outward at v > eleak)
- Total i_total range -3.026e-03 to +2.319e-04 mA/cm². Cell builds and
  integrates without runtime errors.
