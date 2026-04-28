# Path C — CeNGEN-equation-coupling investigation summary

**Date:** 2026-04-28 (Wave P / Session 2 / equation-derived integration)
**Status:** All 4 checkpoints (CP C.1-C.4) complete + consolidator (CP C.5)

---

## Headline

CeNGEN-equation-coupling is **structurally viable but quantitatively marginal at v1**. Linear scaling g_nS = α × TPM produces leave-one-out predictions within ~3.6× on average (mean |log10_err| 0.56) when calibrated on Wave 2 cells. The methodology produces falsifiable equation-derived predictions for AVB / PVC / ASHL — three representative un-validated cells — that extend simulator biophysical reach past the ~20-30 primary-source-anchored literature cap, with EXPLICIT "equation-derived prediction, awaiting empirical validation" labeling throughout. Recommended refinement path documented.

## Per-cell verdicts

| Cell | Status | n channels predicted | V_rest predicted (mV) |
|---|---|---|---|
| AVBL | EQUATION-DERIVED PREDICTION | 7 | -47.09 |
| PVCL | EQUATION-DERIVED PREDICTION | 9 | -65.95 |
| ASHL | EQUATION-DERIVED PREDICTION | 4 | -44.57 |

## CP C.1 — CeNGEN inventory

CeNGEN panel (Taylor et al. 2021): **159 neurons × 133 panel genes** (TPM values).

Channel-relevant gene → Wave 2 channel mapping: **17 mappings** (`egl-19→egl19`, `unc-2→unc2`, `cca-1→cca1`, `shl-1→shl1`, `shk-1→shk1`, `kvs-1→kvs1`, `unc-103→unc103`, `nca-2→nca`, `unc-77→nca`, `slo-1→slo1iso`, `slo-2→slo2`, `irk-2→irk`, `kqt-1→kqt1`, `kqt-3→kqt3`, `egl-2→egl2`, `twk-18→twk18`, `unc-80→nca_aux`).

Channels NOT in CeNGEN panel:
- `leak` — not gene-encoded (passive membrane parameter)
- IRK subunits — not always covered

For un-covered channels, predictions fall back to cell-builder defaults.

**Output:** `cengen_coupling/cengen_channel_inventory.csv` (76 (neuron, channel) entries across 12 target neurons including 4 Wave 2 ground-truth + 6 prediction targets + 2 ASE substitutes).

## CP C.2 — Linear-scaling calibration

Approach: g_nS = α × TPM, fit α per channel using Wave 2 cells (AVAL, AVAR, AIY, RIM) where both ground-truth conductance (Nicoletti) and CeNGEN expression exist.

| Channel | n cells | α median (nS/TPM) | α range | Spread ratio |
|---|---|---|---|---|
| egl19 | 3 | 1.0720 | [0.077, 1.738] | 22.6× |
| nca | 3 | 0.3289 | [0.120, 0.450] | 3.7× |
| unc2 | 1 | 1.4286 | n/a | n/a |
| shl1 | 1 | 12.0669 | n/a | n/a |

**Calibration verdict: linear scaling MARGINAL.**

- `egl19` shows 22.6× spread across 3 cells — the L-type Ca channel's TPM-to-protein-density relationship is not well-captured by linear scaling. Likely candidates: differential post-translational regulation between AVA-class and AIY-class neurons, alternative splicing producing different effective channel densities per TPM unit.
- `nca` shows 3.7× spread — better, but still wider than ideal for tight point estimates.
- `unc2` and `shl1` only have one cell each (RIM) — can't assess spread; calibration brittle.

### Leave-one-out validation

| Held-out cell | n predictions | mean \|log10_err\| |
|---|---|---|
| AVAL | 4 | 0.481 |
| AVAR | 4 | 0.361 |
| AIY | 1 | 0.510 |
| RIM | 3 | 1.262 |

**Overall LOO mean \|log10_err\| = 0.556** (predictions within 10^0.556 = 3.6× of ground truth on average).

RIM is the outlier (10× off when held out). This is consistent with RIM's distinct channel suite (high SHL-1 + EGL-2 + IRK at much higher densities than AVA-class) — the calibration trained on AVA + AIY doesn't generalize to RIM's regime.

**Verdict:** linear scaling generalizes for AVA-class and AIY but not for RIM. Either (a) more cells in calibration, (b) per-channel-class scaling, or (c) Hill function would improve prediction quality.

**Output:** `cengen_coupling/expression_to_conductance_calibration.md`

## CP C.3 — Equation-derived models (AVBL, PVCL, ASHL)

Three representative un-validated C. elegans neurons selected to test the methodology:

### AVBL — forward-locomotion command interneuron

- **Predicted channels (7):** egl19 (Ca, 0.097 nS), unc2 (Ca, 0.114 nS), exp2 (K, fallback α), unc103 (K, 0.097 nS), nca (Na leak, 0.079 nS via nca-2 + unc-77), nca_aux (auxiliary), slo2 (K, fallback α).
- **Predicted V_rest:** -47 mV (intermediate between AVA depolarized regime and typical interneuron rest)
- **Indirect evidence (Atanas 2023):** AVB shows tonic activity correlated with forward locomotion; rapid suppression at reversal onset. Predicted depolarized V_rest is consistent with tonic firing role.
- **Falsifiability:** whole-cell patch on AVB; if input resistance + V_rest off by > 2× from prediction, calibration insufficient for this cell.

### PVCL — touch-reversal pathway interneuron

- **Predicted channels (9):** richest channel set among the three — egl19, unc2, shl1, shk1, kvs1, unc103, nca, slo1iso, slo2.
- **Predicted V_rest:** -66 mV (typical interneuron rest)
- **Indirect evidence:** PVC participates in touch reversal cascade (Wicks 1996); transient activity during touch sequences (Atanas 2023). Predicted hyperpolarized V_rest with rich K-channel suite is consistent with regulated repolarization in the cascade.
- **Falsifiability:** whole-cell patch; channel-specific pharmacology to test predicted SHL-1 + SLO-1 contributions.

### ASHL — polymodal sensory neuron (ASE substitute)

- **Predicted channels (4):** unc2 (Ca), shk1 (K), exp2 (K), slo2 (K). Lighter channel set than command interneurons.
- **Predicted V_rest:** -45 mV
- **Indirect evidence:** ASH responds phasically to nociceptive osmotic / chemical / mechanical stimuli (Hart 1995, Hilliard 2002); transient calcium responses observed. Predicted depolarized V_rest with phasic K-channel set is consistent with phasic burst behavior.
- **Note:** ASE itself was not in CeNGEN panel; ASHL chosen as polymodal sensory substitute. Per-prompt CP C.3 originally specified ASE; ASHL is the closest available analog with chemosensory + osmotic-shock characterization in literature.

**Output:** `cengen_coupling/equation_derived_models/equation_derived_{avbl,pvcl,ashl}.md`

## CP C.4 — Indirect validation

For each predicted cell, indirect evidence (calcium imaging, behavioral genetics, connectome) is documented:

- **All three cells have non-trivial channel suites consistent with their biological roles** — AVBL's NCA-pathway emphasis matches tonic-drive role; PVCL's K-channel suite matches reversal-cascade regulated repolarization; ASHL's phasic Ca + K matches sensory burst behavior.
- **Predicted V_rest values are biologically reasonable** (-45 to -66 mV; within typical neuron range).
- **Largest source of uncertainty:** leak conductance (not gene-encoded). Default 0.05 nS may be wrong by 2-3× for any given cell. Real validation requires whole-cell patch to measure input resistance directly.

**Verdict on indirect agreement:** structurally consistent with available indirect evidence at the qualitative level. Quantitative agreement is unverified — the equation-derived models' predicted dynamics under sensory/motor protocols haven't been compared to Atanas 2023 calcium traces; that's a follow-up analysis.

## CP C.5 — Path C consolidator

### Is CeNGEN-equation-coupling a viable path past the literature cap?

**Yes, structurally. Marginal, quantitatively. Refineable.**

The methodology is structurally sound:
- CeNGEN data is rich (159 neurons × 133 genes; ion channel families well-covered)
- Wave 2 cell-builder code provides ground truth for calibration
- Linear scaling produces order-of-magnitude predictions
- Falsifiable: predictions can be tested by future whole-cell patch + pharmacology

The methodology is quantitatively marginal at v1:
- α spread across cells is large (22.6× for egl19 in particular)
- LOO mean |log10_err| = 0.56 (within ~3.6× on average; individual channels can be off by 10×)
- Leak conductance is unconstrained by CeNGEN

### Recommended trajectory if Path C is to be load-bearing methodology

1. **Expand calibration cell panel** beyond AVAL/AVAR/AIY/RIM to reduce per-channel α uncertainty. Bounded by the same literature cap (need cells with both Nicoletti electrophysiology AND CeNGEN expression). Realistic expansion: 8-12 calibration cells if remaining literature is mined carefully.

2. **Hill function scaling:** g = g_max / (1 + (TPM_50 / TPM)^n). Captures saturation + threshold effects. Adds 2-3 free parameters per channel; needs ≥5 cells per channel to fit reliably.

3. **Per-channel-class calibration:** K-channels and Ca-channels likely have different α scaling due to differential post-translational regulation. Stratify the calibration.

4. **Add CeNGEN panel genes for missing channels:** notably IRK family + leak pathway components like TWK channels broadly. Requires re-running CeNGEN data extraction with expanded panel.

5. **Empirical validation on cells with partial indirect data:** AVB has Atanas calcium imaging — match equation-derived predicted dynamics to observed Ca traces under controlled stimuli. Even partial agreement strengthens the methodology.

### Honest assessment

The methodology is informative but not yet predictive. Equation-derived models for AVB/PVC/ASHL produced here are usable as **FALSIFIABLE PREDICTIONS for future wet-lab work**, but should NOT be deployed in production simulation without empirical validation. The labeling matters.

### Path past the literature cap

Path C demonstrates that the CeNGEN-equation-coupling approach is structurally viable. With Hill scaling + expanded gene panel + per-class calibration, it could become predictive. As-is, it produces structurally-grounded falsifiable predictions for the ~270 C. elegans neurons without published electrophysiology — that's a real extension of the simulator's biophysical reach beyond the current ~20-30 primary-source-anchored cells, **with explicit awaiting-empirical-validation labeling**.

This is the methodology pattern Wave P is designed for: produce predictions for wet-lab follow-up, not validated production models.

## What's now ready

- `cengen_coupling/expression_to_conductance.py` — calibration framework + leave-one-out validation
- `cengen_coupling/equation_derived_models.py` — model generator for arbitrary CeNGEN neurons
- `cengen_coupling/cengen_channel_inventory.csv` — 76-row inventory across 12 target neurons
- `cengen_coupling/expression_to_conductance_calibration.md` — calibration documentation
- `cengen_coupling/equation_derived_predictions.md` — consolidated predictions across AVBL/PVCL/ASHL
- `cengen_coupling/equation_derived_models/equation_derived_{avbl,pvcl,ashl}.md` — per-cell prediction docs with falsifiability statements
- 2 checkpoint JSONs in `checkpoints/`
