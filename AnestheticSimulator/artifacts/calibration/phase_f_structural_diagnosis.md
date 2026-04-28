# CP1 — Phase F structural collapse diagnosis

**Date:** 2026-04-27
**Status:** confirmed parameter-locked; Option A (downgrade to PASS_PARAMETER_TUNED with consistent-with-biology framing) selected.

---

## 1. Sensitivity sweep replication and extension

Joint sweep of Phase F predicted gas-1/WT hypersensitivity ratio across {block_factor × GAS1_COMPLEX_I_FACTOR}.

| gas1\block | 0.05 | 0.10 | 0.30 | 0.50 | 0.706 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|
| 0.30 | 13.39 | 13.33 | 12.88 | 14.30 | 13.56 | 14.03 | (inf) |
| 0.40 | **2.45** | **2.50** | **2.45** | **2.47** | **2.49** | **2.48** | (inf) |
| 0.50 | 1.66 | 1.67 | 1.66 | 1.66 | 1.66 | 1.66 | (inf) |
| 0.60 | 1.36 | 1.36 | 1.37 | 1.36 | 1.36 | 1.36 | (inf) |
| 0.70 | 1.21 | 1.21 | 1.21 | 1.21 | 1.21 | 1.21 | (inf) |

Across **19× variation** in block_factor (0.05 to 0.95), the ratio at GAS1=0.4 varies by **0.05** (range 2.45–2.50). The ratio is determined essentially entirely by GAS1_COMPLEX_I_FACTOR.

## 2. Analytical confirmation

Phase F dose-finding logic (`predicted_anesthetic_dose_for_immobilization`):

```
complex_i_rate(genotype, dose) = base_complex_i_rate × (1 - dose × (1 - block_factor))
```

where:
- `base_complex_i_rate` = 1.0 (WT) or `GAS1_COMPLEX_I_FACTOR` = 0.4 (gas-1)

ATP steady-state is linear in `complex_i_rate`:

```
ATP_ss(complex_i_rate) = (complex_i_rate × K_COMPLEX_I_WT + K_COMPLEX_II) / K_BASE_CONSUMPTION
```

The dose-finding searches for the dose `d` such that ATP_ss falls below the K-ATP-opening threshold producing |V_shift| ≥ V_SHIFT_IMMOBILIZATION.

**Algebraic structure (ignoring K_COMPLEX_II non-linearity):**

For WT: `1 - d_WT × (1 - bf) = ATP_crit / K_COMPLEX_I_WT`
For gas-1: `GAS1 × (1 - d_g1 × (1 - bf)) = ATP_crit / K_COMPLEX_I_WT`

Hence:
```
d_WT × (1 - bf) = 1 - ATP_crit/K_I
d_g1 × (1 - bf) = 1 - ATP_crit/(GAS1 × K_I)
```

The ratio:
```
d_WT / d_g1 = (1 - ATP_crit/K_I) / (1 - ATP_crit/(GAS1 × K_I))
```

**The (1 - bf) term cancels entirely in the ratio.** The block_factor (which is the only anesthetic-specific input from wave2_overlay.json) has no influence on the predicted hypersensitivity ratio.

**This is structural to the model formulation**, not a parameter-tuning bug. The ratio is determined by:
- GAS1_COMPLEX_I_FACTOR (sets the relative ATP availability between genotypes)
- ATP_crit (set by the K-ATP coupling parameters K_ATP_HALF, G_K_ATP_MAX, V_SHIFT_IMMOBILIZATION)
- K_COMPLEX_II constant (introduces ~10% non-linearity, insufficient to reach anesthetic differentiation)

The K_COMPLEX_II baseline contributes only ~10% non-linearity (0.3 of 1.3 total production). Across the input range tested, this introduces variation of ~0.05 in the predicted ratio — visible in the 2.45-2.50 spread but far below biologically-meaningful anesthetic differentiation.

## 3. Morgan & Sedensky 1995 primary source check

**Paper: Morgan PG, Sedensky MM. *Anesthesiology* 81(4):888-898 (1994). PMID 7943840** — "Mutations conferring new patterns of sensitivity to volatile anesthetics in C. elegans."

Key finding: gas-1 mutants are hypersensitive to volatile anesthetics. The reported ratios across volatiles:
- Halothane: gas-1 EC50 ~50% of WT (so hypersensitivity ratio ~2×)
- Isoflurane: gas-1 ~30-40% of WT (hypersensitivity ratio 2.5-3×)
- Other halogenated agents: similar magnitude

**The volatiles produce SIMILAR but not identical hypersensitivity ratios across the panel.** Morgan's data shows differential ratios within the 2-3× band, not a single universal ratio.

Key biological observation: **gas-1 conferred broad hypersensitivity across volatile anesthetics, but the magnitude is sensitive to chemical structure** — fluoroethers vs alkanes vs ethers showed slightly different fold-shifts. This is consistent with gas-1 affecting a downstream pathway (Complex I → ATP → membrane potential) that all volatiles converge on, with anesthetic-specific small variations.

## 4. Decision: Option A — Downgrade to PASS_PARAMETER_TUNED with consistent-with-biology framing

**Reasoning:**

1. **Phase F is structurally parameter-locked.** Block_factor cancels in the ratio mathematically. The model cannot differentiate anesthetics, regardless of parameter tuning.

2. **Morgan reports similar (but not identical) hypersensitivity ratios across volatiles.** The 2-3× target band reflects the broad consistency of the gas-1 phenotype, not anesthetic-specific differentiation. So Phase F's predicted ratio (which is constant across anesthetics) is *consistent with* Morgan's observation that all volatiles are similarly affected, but does *not* independently capture the small inter-anesthetic differences Morgan reports.

3. **The predicted 2.48× value at GAS1_COMPLEX_I_FACTOR=0.4 falls in Morgan's 2-3× band, but this was achieved by tuning** GAS1_COMPLEX_I_FACTOR specifically to land in the band. Other plausible values in Kayser's 30-50% range produce different ratios:
   - GAS1=0.3 → ratio 13× (way outside band)
   - GAS1=0.5 → ratio 1.66× (just below band)
   - GAS1=0.4 → ratio 2.48× (in band)

4. **Honest verdict for Phase H anchor #1:**
   > Phase F predicts a gas-1/WT hypersensitivity ratio of 2.48× when GAS1_COMPLEX_I_FACTOR is set to 0.4 (within the Kayser 2001 30-50% Complex I activity reduction range, choosing the lower end). The ratio is structurally insensitive to the anesthetic-specific Complex I block factor from wave2_overlay.json — the (1-block_factor) term cancels mathematically in the ratio. The prediction is therefore **consistent with the broad Morgan & Sedensky 1995 phenotype** (gas-1 hypersensitivity 2-3× across volatiles) when the gas-1 parameter is tuned to the band, but **does not constitute an independent test** of Wave P's anesthetic-specific binding pipeline. The pipeline's anesthetic-specific information is preserved in the wave2_overlay's Complex I rate factors but is not reflected in Phase F's gas-1 hypersensitivity output.

**Verdict downgraded:** PASS → **PASS_PARAMETER_TUNED** with confidence MEDIUM.

Note that the gas-1 hypersensitivity claim is still biologically supported — the Wave P binding pipeline correctly identifies Complex I (NDUFS2/GAS-1) as a halothane target with strong calibration (predicted Kd 357 µM vs experimental IC50 400 µM, log_err -0.05). That structural finding is robust. What's parameter-tuned is the *behavioral threshold* layer that converts ATP changes into immobilization. The structural prediction (anesthetic engages Complex I with µM affinity) and the genetic prediction (gas-1 mutants are hypersensitive to anesthetics whose immobilization mechanism passes through Complex I) are both supported by the biology — just not via Phase F's specific behavioral threshold model.

## 5. Required next steps (out of CP1 scope)

- **Phase F reformulation (Option C)** would require introducing anesthetic-specific dependencies beyond block_factor, such as direct K-ATP modulation by anesthetic, or membrane partition K_p amplification at K-ATP channel pockets. This is reformulation, deferred for separate work block.
- **Network-level validation (Phase G)** would test whether the Wave 2 Brian2 simulator with wave2_overlay perturbation produces gas-1 hypersensitivity. If yes, Phase G provides the independent test that Phase F cannot. The structural binding pipeline + network propagation is the pathway that would close the loop.

## 6. Conclusion

Phase F's gas-1 hypersensitivity prediction stays in the Phase H validation table with the verdict **PASS_PARAMETER_TUNED** and confidence **MEDIUM**. The framing shifts from "Wave P predicts Morgan's 2-3× ratio" to "Wave P's prediction is consistent with Morgan's 2-3× ratio when the gas-1 Complex I residual rate parameter is set to the lower end of Kayser's empirical range." This is the honest claim.

The anesthetic-specific clustering at 2.48-2.49× across all 5 anesthetics is documented as a structural feature of the model (block_factor cancels in the ratio) rather than a biological prediction.

The structural binding prediction for Complex I (calibrated against NDUFS2/halothane log_err -0.05) remains independent of Phase F's behavioral threshold layer and is robust on its own merits.
