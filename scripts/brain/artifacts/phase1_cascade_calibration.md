# Phase 1 — T2-#4 sensory cascade calibration

Grid-search fit of each cascade's time constants against the 
digitised published ΔF/F reference. Frechet distance computed on 
peak-normalised traces (amplitude-unit-invariant).

## Per-cascade results

| cascade | status | pre-fit Frechet | post-fit Frechet | improvement |
|---|---|---|---|---|
| ASE | pending_reference | — | — | — |
| AWC | pending_reference | — | — | — |
| ASH | pending_reference | — | — | — |
| AFD | pending_reference | — | — | — |
| ALM | pending_reference | — | — | — |

## Pending references

- **ASE**: references/thiele_2009_ase/trace.npz not found
- **AWC**: references/chalasani_2007_awc/trace.npz not found
- **ASH**: references/hilliard_2005_ash/trace.npz not found
- **AFD**: references/clark_2006_afd/trace.npz not found
- **ALM**: references/ohagan_2005_alm/trace.npz not found

## Exit threshold

- Each fitted cascade: post-fit Frechet ≤ 0.10 (10% of peak-normalised range)
- Improvement over defaults: ≥ 0.05 per cascade
- No regression on touch / osmotic / salt / food scenarios when `sensory_mode=transduction` is enabled (ensemble audit).