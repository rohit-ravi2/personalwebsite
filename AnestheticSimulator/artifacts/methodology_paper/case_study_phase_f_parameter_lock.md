# Case study 1 — Phase F structural parameter-lock

**Project:** AnestheticSimulator / Wave P pharmacology pipeline
**Date diagnosed:** 2026-04-27 (CP1 of rigor-tightening pass)
**Methodology pattern:** sensitivity-sweep before validation claim

---

## Finding

Wave P's Phase F (metabolic ATP layer + gas-1 hypersensitivity prediction) appeared to PASS the Morgan & Sedensky 1994 (PMID 7943840) anchor: the predicted gas-1/WT halothane hypersensitivity ratio of 2.48× falls within the published 2-3× band. This was originally counted as a "5/6 PASS" anchor in the WAVE_P_PHASE_H_VALIDATION table.

A sensitivity sweep across the model's two key inputs revealed that the predicted ratio is structurally invariant to the anesthetic-specific input. Specifically, varying `block_factor` (the per-anesthetic Complex I rate at 1× EC50, the only anesthetic-specific input from `wave2_overlay.json`) across **19× variation** (0.05 to 0.95) at fixed `GAS1_COMPLEX_I_FACTOR=0.4` produced a ratio of 2.45-2.50 — variation of **0.05** total. The ratio is determined essentially entirely by `GAS1_COMPLEX_I_FACTOR`.

| `gas1\block` | 0.05 | 0.10 | 0.30 | 0.50 | 0.706 | 0.85 |
|---|---|---|---|---|---|---|
| 0.30 | 13.39 | 13.33 | 12.88 | 14.30 | 13.56 | 14.03 |
| 0.40 | **2.45** | **2.50** | **2.45** | **2.47** | **2.49** | **2.48** |
| 0.50 | 1.66 | 1.67 | 1.66 | 1.66 | 1.66 | 1.66 |
| 0.70 | 1.21 | 1.21 | 1.21 | 1.21 | 1.21 | 1.21 |

(Joint sweep: `block_factor` columns × `GAS1_COMPLEX_I_FACTOR` rows. The row chosen for the validation claim, 0.40, is bolded.)

## Analytical confirmation

The dose-finding logic searches for the dose `d` at which ATP_steady_state crosses the K-ATP-opening threshold producing |V_shift| ≥ V_SHIFT_IMMOBILIZATION. Algebraically, ignoring the K_COMPLEX_II non-linearity:

```
For WT: 1 - d_WT × (1 - bf) = ATP_crit / K_I
For gas-1: GAS1 × (1 - d_g1 × (1 - bf)) = ATP_crit / K_I
```

Solving:

```
d_WT × (1 - bf) = 1 - ATP_crit/K_I
d_g1 × (1 - bf) = 1 - ATP_crit/(GAS1 × K_I)

d_WT / d_g1 = (1 - ATP_crit/K_I) / (1 - ATP_crit/(GAS1 × K_I))
```

**The (1-bf) term cancels entirely in the ratio.** The block_factor — which is the only anesthetic-specific input from the binding pipeline — has no influence on the predicted hypersensitivity ratio.

The 2.48× value at GAS1=0.4 was achieved by tuning `GAS1_COMPLEX_I_FACTOR` to the lower end of Kayser 2001 PMID 11278828's empirically-reported 30-50% Complex I activity reduction range. Within that range, the predicted ratio varies dramatically (2.5× → 1.7× → 1.4× as GAS1 goes 0.4 → 0.5 → 0.6). The choice of 0.4 specifically is what produced the in-band 2.48×.

## How the issue was caught

Standard validation workflow at the time would have produced:
- Phase F runs against `wave2_overlay.json` per anesthetic → produces 5 ratios
- Morgan band = 2-3× → 5/6 anesthetics in band → PASS_5/6
- Headline claim: "Wave P predicts Morgan's gas-1 hypersensitivity for 5/6 volatile anesthetics."

The sensitivity sweep added a single methodological step: **before claiming the validation, vary the model's inputs across plausible ranges and observe the spread of outputs.** If the spread of outputs is small relative to the spread of inputs, the model is parameter-locked and the apparent validation is post-hoc fitting.

The sweep was prompted by a pre-flight pushback — an explicit "before launching CP1-CP8 work block, verify that Phase F sensitivity has been tested." This is a recurring methodology pattern in the project: pre-flight pushback that includes a single sentence "test whether the prediction is sensitive to its inputs" reliably surfaces parameter-locking.

## Methodology lesson

**Surface 1 (Wave P-specific):** Phase F's gas-1 hypersensitivity prediction is consistent with Morgan's 2-3× band when GAS1_COMPLEX_I_FACTOR is tuned to 0.4, but is NOT an independent test of the binding pipeline's anesthetic-specificity. The verdict was downgraded from `PASS_5/6` to `PASS_PARAMETER_TUNED` with `confidence MEDIUM`.

**Surface 2 (general):** "model is consistent with biology" ≠ "model independently predicts biology." A model can be consistent with an empirical finding because:

(a) **Genuine prediction:** the model's machinery produces the empirical value when fed plausible inputs without parameter tuning to the value.
(b) **Post-hoc tuning:** one or more tunable parameters were chosen to land the prediction in the empirical band.
(c) **Structural lock:** the model's output is determined by parameters chosen to land in the band, with the inputs (the parts that connect the model to upstream science) cancelling out in the prediction.

Distinguishing (a), (b), and (c) requires sensitivity analysis. (a) shows wide output spread when inputs vary plausibly; (b) shows narrow spread but inputs still matter; (c) shows narrow spread AND inputs cancel out analytically.

**Surface 3 (broader):** computational pipelines that connect upstream binding/structural predictions to downstream behavioral predictions through a fixed coupling layer are at risk of structural lock. The coupling layer's parameters must produce the right behavioral phenotype, but its functional form may make the upstream prediction irrelevant. This is invisible without sensitivity analysis.

## Generalization

This pattern shows up in:

- **Computational drug discovery pipelines** where binding affinity → cellular IC50 → tissue EC50 → behavioral readout. Each translation step introduces a coupling layer that may be tuned to known behavioral data, masking the upstream binding prediction.
- **Model-based reinforcement learning evaluations** where the dynamics model is calibrated to produce realistic episode returns; the apparent quality of the policy may be invariant to the dynamics model's accuracy.
- **Biophysical neuron simulators** with parameters tuned to reproduce known firing patterns. New experimental conditions may be insensitive to the parameter choices that landed the baseline.

The protective methodology is universal: **before claiming validation, sensitivity-sweep across the inputs that are supposed to drive the prediction.** If the prediction is invariant or near-invariant to those inputs, the validation rests on tuning, not on the upstream science.

## Wave P-specific implications

1. The original "5/6 PASS" headline was downgraded to `PASS_PARAMETER_TUNED` in the rigor-tightened verdict (CP8).
2. Phase F reformulation (separate work block) is required to make WT_dose absolute and gas-1_dose relative to a fixed behavioral threshold so that block_factor doesn't cancel. This would allow per-anesthetic Phase F predictions.
3. The binding-side prediction (NDUFS2/halothane Vina-Kd 357 µM vs Hanley 2002 IC50 400 µM, log_err 0.001) remains independent of Phase F's behavioral threshold layer and is robust on its own merits.
4. Phase H validation table now stratifies binding-side claims (`VERIFIED`) from behavioral-mapping claims (`STRUCTURALLY_GROUNDED_AWAITING_WETLAB`) — separates structural prediction quality from coupling-layer robustness.

## Methodological deliverable

A 50-line Python script (`src/preflight_phase_f_saturation.py`) was added to the repository as the canonical "is this prediction parameter-locked?" check. Running this script before claiming Phase F validation reproduces the sensitivity table and surfaces the analytical (1-bf) cancellation immediately.

## Reference figures

- `artifacts/calibration/phase_f_structural_diagnosis.md` — full diagnosis with sweep tables and analytical derivation
- `src/preflight_phase_f_saturation.py` — reproducible sensitivity-test script
- `src/calibration_phase_e_sensitivity.py` — analogous test for Phase E (which passed cleanly, vs Phase F's parameter-lock)
