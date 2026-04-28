# CP C.1 + C.2 — CeNGEN-equation-coupling calibration

**Date:** 2026-04-28 (Wave P / Session 2 / Path C)

**Goal:** investigate whether canonical equations + CeNGEN gene expression can predict ion channel conductances for cells without published biophysical electrophysiology — the path past the literature cap.

**Critical methodology discipline:** the equation-derived predictions produced by this work block are FALSIFIABLE PREDICTIONS, not validated models. They are explicitly labeled as such throughout the artifacts.

## CP C.1 — Inventory

CeNGEN panel (Taylor et al. 2021): **159 neurons × 133 panel genes** (TPM values).

Channel-relevant genes mapped to Wave 2 cell-builder channel names: **18 mappings**.

Channels NOT in CeNGEN panel (use cell-builder defaults instead): `leak` (not gene-encoded; reflects passive membrane), some IRK subunits.

Inventory CSV: `cengen_coupling/cengen_channel_inventory.csv` (76 rows).

## CP C.2 — Linear-scaling calibration

Approach: g_nS = α × TPM, fit α per channel using Wave 2 cells (AVAL, AVAR, AIY, RIM) where both ground-truth conductance (Nicoletti) and CeNGEN expression exist. The simplest model; if α is reasonably convergent across cells (low spread ratio), linear scaling is informative. If spread is high (>10×), more sophisticated mapping (Hill function, per-channel-class) is required.

### Per-channel calibration

| channel | n cells | α median (nS/TPM) | α range | spread ratio |
|---|---|---|---|---|
| egl19 | 3 | 1.072 | [0.0769, 1.7398] | 22.62× |
| nca | 3 | 0.3289 | [0.12, 0.4485] | 3.74× |
| unc2 | 1 | 1.4286 | [1.4286, 1.4286] | 1.0× |
| shl1 | 1 | 12.0669 | [12.0669, 12.0669] | 1.0× |

### Calibration verdict

Median α spread across channels with ≥2 cells: **13.2×**.

**LINEAR SCALING MARGINAL** — α scatter is large enough that predictions for new cells should report uncertainty bounds, not point estimates. Hill function or per-channel-class calibration may improve fit.

### Leave-one-out validation

| held-out | n predictions | mean |log10_err| | per-channel breakdown |
|---|---|---|---|
| AVAL | 4 | 0.481 | egl19: log_err -0.48 |
| AVAR | 4 | 0.361 | egl19: log_err -0.07, nca: log_err -0.57, nca: log_err -0.44 |
| AIY | 1 | 0.51 | nca: log_err +0.51 |
| RIM | 3 | 1.262 | egl19: log_err +1.26 |

**Overall LOO mean |log10_err|: 0.556**

Interpretation: this is the predictive accuracy when applying the calibration to a held-out Wave 2 cell. If mean |log10_err| < 0.5 (within ~3×), the calibration generalizes; if > 1.0 (10×), the calibration overfits to the training cells.

