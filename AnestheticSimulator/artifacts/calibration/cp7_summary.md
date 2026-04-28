# CP7 — Allosteric correction + chemical class stratification

## Correction applied

- f_allo = **2.51×** (CP5 median-based)
- Direction: divide pipeline-predicted Kd by f_allo
- Rationale: T1 strict subset signed median log_err = +0.399 (positive bias = pipeline overestimates Kd; consistent with PAM allosteric coupling theory η ~ 0.4)

## Per-chemical-class metrics (T1 strict subset only)

| chem_class | n | pre signed_mean | post signed_mean | pre mean |log_err| | post mean |log_err| | post % within 10× |
|---|---|---|---|---|---|---|
| ALKANE_HALOGENATED | 4 | +0.628 | +0.229 | 0.628 | 0.428 | 100% |
| ETHER_HALOGENATED | 6 | +0.287 | -0.112 | 0.360 | 0.153 | 100% |
| IV_ARYLCYCLOHEXYLAMINE | 1 | +0.974 | +0.575 | 0.974 | 0.575 | 100% |
| IV_IMIDAZOLE | 3 | +0.685 | +0.286 | 0.977 | 0.844 | 100% |
| IV_PHENOL | 3 | +0.568 | +0.169 | 0.707 | 0.661 | 67% |

## Class-specific bias interpretation

After universal f_allo correction, residual signed_mean per chemical class:

- **ALKANE_HALOGENATED**: signed_mean = +0.229 → positive residual (+0.23) — needs class-specific tightening
- **ETHER_HALOGENATED**: signed_mean = -0.112 → near-zero — universal correction sufficient
- **IV_ARYLCYCLOHEXYLAMINE**: signed_mean = +0.575 → positive residual (+0.57) — needs class-specific tightening
- **IV_IMIDAZOLE**: signed_mean = +0.286 → positive residual (+0.29) — needs class-specific tightening
- **IV_PHENOL**: signed_mean = +0.169 → near-zero — universal correction sufficient

## Halogenated non-immobilizer baseline

Hexafluoroethane is a halogenated alkane non-immobilizer per Eger 2001 — used as a negative-control test of whether Wave P discriminates the non-immobilizer class from clinical alkanes by binding profile.

At 1000 µM aqueous (clinical-range halogenated alkane concentration), post-CP5-correction:

- hexafluoroethane engages ≥10%: **30/30** common targets
- cis-DCE engages ≥10% (anesthetic positive control): **22/30** common targets

**Inverted discrimination:** hexafluoroethane engages 8 MORE targets than cis-DCE — pipeline biased toward bulk lipophilicity over shape/conformational specificity.

## Outputs

- `cp7_corrected.csv` — per-row pre/post-correction comparison (n=24)
- `cp7_class_stratified.csv` — per-chemical-class metrics
- `wave2_overlay_v2.json` — corrected occupancies for downstream Phase E/F/G use
