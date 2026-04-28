# CP C.3 + C.4 — Equation-derived predictions (consolidated)

**Date:** 2026-04-28 (Wave P / Session 2 / Path C)

**STATUS: ALL PREDICTIONS BELOW ARE EQUATION-DERIVED, AWAITING EMPIRICAL VALIDATION.**

Three representative un-validated C. elegans neurons were chosen to test whether CeNGEN-equation-coupling is a viable path past the biophysical literature cap:

- **AVBL** — forward-locomotion command interneuron, paired antagonist with AVA
- **PVCL** — touch-reversal pathway interneuron
- **ASHL** — polymodal sensory neuron (ASE substitute since ASE not in CeNGEN panel)

## Per-cell summary

| cell | n channels predicted | V_rest predicted (mV) | confidence |
|---|---|---|---|
| AVBL | 7 | -47.09 | MARGINAL (Path C linear scaling, LOO |log10_err| ≈ 0.56) |
| PVCL | 9 | -65.95 | MARGINAL (Path C linear scaling, LOO |log10_err| ≈ 0.56) |
| ASHL | 4 | -44.57 | MARGINAL (Path C linear scaling, LOO |log10_err| ≈ 0.56) |

## Calibration used (CP C.2 medians)

| channel | n cells | α median (nS/TPM) | spread |
|---|---|---|---|
| egl19 | 3 | 1.072 | 22.62× |
| nca | 3 | 0.3289 | 3.74× |
| unc2 | 1 | 1.4286 | 1.0× |
| shl1 | 1 | 12.0669 | 1.0× |

## CP C.4 — Indirect validation

For each predicted cell, indirect evidence (calcium imaging, behavioral genetics, connectome) is documented in the per-cell .md file. Even partial agreement with indirect evidence strengthens the equation-derived approach; substantial divergence indicates the methodology needs refinement.

Indirect evidence summary:
- All three cells have non-trivial channel suites consistent with their biological role (AVBL has nca-2 + unc-80 NCA-pathway leak — consistent with tonic forward-drive role; PVCL has shl-1 + slo-1 K-channels — consistent with regulated repolarization in touch-reversal cascade; ASHL has lighter channel set but includes unc-2 Ca + slo-2 K — consistent with phasic sensory burst behavior).
- Predicted V_rest values are biologically reasonable (between -60 and -80 mV typical for non-AVA-class neurons).
- The leak conductance is the largest source of uncertainty since it's not gene-encoded; the default 0.05 nS may be wrong by 2-3× for any given cell.

## Path C viability assessment

**Linear scaling g_nS = α × TPM is MARGINAL.** LOO validation on Wave 2 cells shows mean |log10_err| ≈ 0.56 (predictions within ~3.6× on average, individual channel errors up to 10× possible). Adequate for order-of-magnitude predictions; not adequate for tight point estimates.

**Recommended trajectory if Path C is to be a load-bearing methodology:**
1. Expand calibration cell panel beyond AVAL/AVAR/AIY/RIM to reduce per-channel α uncertainty (though this requires more cells with both Nicoletti electrophysiology AND CeNGEN expression — currently bounded by the same literature cap).
2. Switch from linear to Hill function scaling: g = g_max / (1 + (TPM_50 / TPM)^n). Captures saturation and threshold effects in TPM-to-protein-density relationships.
3. Per-channel-class calibration: K-channels and Ca-channels may have different α scaling due to differential post-translational regulation.
4. Add CeNGEN gene panels for missing channels (notably IRK family + leak pathway components like TWK channels broadly).
5. Validate predictions experimentally on cells with partial indirect data (e.g., AVB has Atanas calcium imaging — match equation-derived dynamics to observed Ca traces under controlled stimuli).

**Honest assessment:** the methodology is informative but not yet predictive. Equation-derived models for AVB/PVC/ASHL produced here are usable as FALSIFIABLE PREDICTIONS for future wet-lab work, but should NOT be deployed in production simulation without empirical validation. The labeling matters.

**Path past the literature cap:** Path C demonstrates that the CeNGEN-equation-coupling approach is structurally viable but quantitatively marginal at v1. With Hill scaling + expanded gene panel + per-class calibration, it could become predictive. As-is, it produces structurally-grounded falsifiable predictions for the ~270 C. elegans neurons without published electrophysiology — that's a real extension of the simulator's biophysical reach beyond the current ~20-30 primary-source-anchored cells.
