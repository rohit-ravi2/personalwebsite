# Current state summary

*Last updated: 2026-04-21*

Narrative layer on top of `scripts/brain/artifacts/phase0_baseline_report.md`. Updated at phase boundaries.

## Simulator execution profile

Brian2 2.9.0 on CPU (numpy codegen target). Measured wall/simulated ratio on the shipped v3 LIF brain: **3.06×**. Full phenotype audit at n=10 × 60s: ~6.1 hours wall.

## Phase roadmap status

- **Phase 0** — in progress. baseline measurement + audit infra
- **T2-#4 sensory cascade calibration** — pending. baseline captured; digitisation pending for Frechet eval
- **T4-2 compartmental plateau calibration** — pending. baseline captured; Gao & Hobert 2020 digitisation pending
- **T4-1 motor coupling** — pending. CPG baseline captured for curvature-ρ comparison
- **T4-3 synaptic calibration (T0 fix)** — pending. baseline captured; currently AVA doesn't fire on touch
- **T4-4 CeNGEN-conductance coupling** — pending. end of sequence; architectural overlay
- **T4-5 INS-family peptide expansion** — pending. 6-peptide selection confirmed
- **T4-6 trajectory correlation** — pending. baseline ρ distribution captured; capstone

## Ratified thresholds

- **T4-3 synaptic calibration** — AVAL peri ≥ 20 Hz AND Δ ≥ +15 Hz on ≥ 8/10 seeds
- **T4-5 RIS/Turek phenotype** — ΔQUI ≤ −0.30 with 95% CI excluding zero
- **T4-2 AVA plateau (Gao & Hobert 2020)** — plateau duration ∈ [480, 720] ms (20% of 600 ms target); amplitude ∈ [18, 22] mV above rest
- **T2-#4 sensory cascade calibration** — Each cascade's rate trace ≤ 10% Frechet distance to digitised published ΔF/F (Thiele 2009, Chalasani 2007, Hilliard 2005, Clark 2006, O'Hagan 2005)
- **T4-1 motor coupling (curvature ρ)** — median ρ vs Tierpsy pool ≥ max(0.6, CPG_baseline + 0.15)
- **T4-6 trajectory correlation** — median ρ increases ≥ +0.10 relative to baseline; tail named in paper

## References

- Primary measurement: `scripts/brain/artifacts/phase0_baseline_report.md`
- Per-subsystem: `phase0_plateau_baseline.md`, `phase0_cascade_baseline.md`, `phase0_swap_jitter.md`
- Digitised reference traces: `scripts/brain/references/`
