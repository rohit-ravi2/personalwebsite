# Phase 0 — W0.4b — T2-#4 sensory cascade baseline

Current-state shape characterisation of the 5 transduction cascades 
in `sensory_transduction.py`. Runs each cascade standalone with its 
canonical stimulus protocol (Thiele 2009 / Chalasani 2007 / Hilliard 
2005 / Clark 2006 / O'Hagan 2005 analogs) and records the rate trace.

Used as the **pre-calibration reference**: T2-#4 will refit parameters 
against digitised ΔF/F from the reference figures (pending `docs/
references/` data). Frechet distance between these baseline traces and 
the calibrated versions is the exit metric.

## Trace characteristics

| cascade | peak (Hz) | peak t (s) | rise τ (s) | decay τ (s) | final (Hz) |
|---|---|---|---|---|---|
| ASE | 22.86 | 2.0 | 0.0 | 0.7 | 0.0 |
| AWC | 140.0 | 6.0 | 3.0 | 2.6 | 0.0 |
| ASH | 204.24 | 3.15 | 0.05 | 1.3 | 0.0 |
| AFD | 130.0 | 11.0 | 6.75 | — | 130.0 |
| ALM | 155.35 | 2.05 | 0.0 | 0.15 | 0.0 |

## Canonical stimulus protocols

- **ASE**: 0 → 1 salt step at t=2s, sustained.
- **AWC**: odorant pulse 3-6s (expect firing on offset at t=6s).
- **ASH**: aversive pulse 2-4s.
- **AFD**: 20°C baseline, ramp to 25°C during 5-10s.
- **ALM**: 100ms touch impulse at t=2s.

## T2-#4 exit threshold (ratified)

- Each cascade's simulated rate trace within ≤10% Frechet 
  distance of the digitised published ΔF/F (after z-score 
  normalisation to account for amplitude-unit mismatch).
- No regression on touch / osmotic / salt / food scenarios 
  with `sensory_mode=transduction` in the ensemble audit.

## References

- Thiele T, Faumont S, Lockery SR (2009) J Neurosci — ASE salt.
- Chalasani SH et al. (2007) Nature — AWC OFF-cell.
- Hilliard MA et al. (2005) EMBO — ASH polymodal.
- Clark DA et al. (2006) J Neurosci — AFD thermal.
- O'Hagan R et al. (2005) Nat Neurosci — ALM MEC-4/10.