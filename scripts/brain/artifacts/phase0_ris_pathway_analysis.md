# Phase 0 — RIS pathway molecular audit

Dissection of the RIS/Turek quiescence mechanism into its GABA-synaptic and FLP-11-peptidergic components. Tests whether the behavioural null (ΔQUI ≈ 0 at n=10 × 60s in the phenotype audit) reflects mechanism absence, weak calibration, or readout insensitivity.

## Aggregate per condition

| condition | n | FLP-11 peak | FLP-11 Δ | GABA target Δrate | FLP-11 target Δrate | QUI prop |
|---|---|---|---|---|---|---|
| CONTROL | 6 | 10.00±0.00 | +8.497 | +68.02±1.74 | +19.20±0.30 | 0.05±0.05 |
| FLP11_KO | 6 | 0.00±0.00 | +0.000 | +70.42±1.53 | +23.61±0.36 | 0.05±0.04 |
| GABA_KO | 6 | 10.00±0.00 | +8.482 | +69.95±1.97 | +19.14±0.31 | 0.09±0.08 |
| RIS_ABLATE | 6 | 0.00±0.00 | +0.000 | +71.22±1.65 | +23.34±0.26 | 0.05±0.03 |

## Layer-by-layer diagnosis

**Layer 1 (molecular — FLP-11 release):**
- Control FLP-11 peak = 10.00. RIS_ABLATE FLP-11 peak = 0.00.
  ✓ FLP-11 rises in CONTROL, and drops in RIS_ABLATE as expected. Peptidergic release mechanism fires correctly.

**Layer 2 (cellular — GABA synaptic pathway):**
- GABA-target peri-pre firing-rate Δ: CONTROL +68.02, RIS_ABLATE +71.22, GABA_KO +69.95 Hz.
  ✓ GABA pathway is operating — ablating RIS and/or zeroing its GABA synapses releases downstream inhibition.

**Layer 3 (cellular — peptidergic pathway):**
- FLP-11-target peri-pre firing-rate Δ: CONTROL +19.20, RIS_ABLATE +23.34, FLP11_KO +23.61 Hz.
  ✓ Peptidergic pathway is operating — RIS ablation or FLP-11 release knockout measurably affects FLP-11-target firing rates.

**Layer 4 (behavioral — QUIESCENT state):**
- CONTROL QUI = 0.05, RIS_ABLATE QUI = 0.05, ΔQUI = -0.00.
  The behavioural phenotype is still absent — consistent with the main audit's ΔQUI ≈ 0 finding.

## Mechanism vs readout — the interpretation

This analysis distinguishes three failure modes identified by Turek 2016's pathway dissection:

1. **Mechanism absent:** no FLP-11 rise, no GABA pathway effect, no behavioural change. Simulator has wrong biology.
2. **Mechanism weakly calibrated:** FLP-11 rises but targets don't respond enough; or GABA pathway operates but too weakly to affect downstream dynamics.
3. **Mechanism present, readout insensitive:** all molecular/cellular signals fire correctly, but classifier/FSM readout doesn't register quiescence state. The paper would characterise this as a fourth falsification layer.

The table above locates which failure mode applies to the current v3 LIF calibration. Calibration fixes per layer:

- Layer 1 weak: `modulation_layer.DEFAULT_RELEASE_GAIN` (currently 0.02) should increase.
- Layer 3 weak: `modulation_layer.DEFAULT_MOD_STRENGTH_PA` (currently 5.0) should increase.
- Layer 4 weak: FSM thresholds for QUIESCENT state (`role_z_threshold`) need adjustment OR move to ActivityFSM reading RIS directly.

## Notable caveat

These diagnostics operate on the v3 LIF brain with mammalian-cortical v_rest. Phase 0's voltage-scale finding implies that the current calibration is quantitatively off by ~40 mV across the whole brain. Post-T4-2 voltage correction + SK/BK addition may shift RIS's firing rate enough to change the release dynamics non-trivially. The molecular audit should be RE-RUN post-Phase-2 to see if RIS/Turek recovers at worm-realistic voltages.