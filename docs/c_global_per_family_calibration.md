# Per-cell-family C_global calibration — v2 Deliverable 4 (Group C)

**Status:** v2 Group C deliverable. Per-cell-family C_global calibration
against measured V_rest targets per §3.0 methodology + §8.11
measurement-vs-fit audit.

**Date:** 2026-05-12

**Reference:** `docs/channel_parameter_derivation_methodology.md` §3.0
(v2 calibration framework) + `docs/v_rest_targets.md` (measurement
anchors) + `docs/layer1_design_decisions.md` §8.11 (measurement-vs-fit).

---

## 1 · Calibration protocol

Per methodology §3.0 v2:

```
For each cell family (AVA, AIY, RIM):
  1. Fix γ (Phase 2 + v2 refinements), TPM (Phase 3), E_translation=1.0
  2. Sweep C_global value across orders of magnitude
  3. For each C_global: build Path 2 cell, run 3s rest sim, measure V_rest
  4. Find C_global producing V_rest within published range
  5. Verify rest stability
  6. Document calibration path + result
```

Order-of-magnitude scan: 1e3, 1e4, 1e5, 1e6, 1e7 (initial). Refined low-
range sweep added for RIM (1e1 to 3e3) after initial sweep failed.

Script: `scripts/brain/wave2/calibrate_path2_v2.py`. Results JSON:
`scripts/brain/wave2/artifacts/path2_v2_calibration_sweep.json`.

---

## 2 · Calibration results

### 2.1 AVA-class — calibrated successfully ✓

Anchor cell: AVAL. Target V_rest range: [−50, −15] mV; central −32 mV.

| C_global | V_rest (AVAL) | in range? |
|---:|---:|---|
| 1e3 | −51.4 mV | NO (just below range) |
| **1e4** | **−47.7 mV** | **YES ← selected** |
| 1e5 | +10.7 mV | NO (depolarized out) |
| 1e6 | +26.6 mV | NO |
| 1e7 | +29.7 mV | NO |

**C_global_AVA = 1.0 × 10⁴ channels/(cm²·TPM)** produces V_rest = −47.7 mV
at AVAL. This is within published range; closer to range hyperpolarized
edge than to the −32 mV central guidance.

**Verification:** Run AVAR with this C_global. Expected: V_rest somewhat
more depolarized than AVAL (AVAR's published rest is −24 mV vs AVAL's
−40). Will validate in Deliverable 5 Tier B.

### 2.2 AIY-class — calibrated successfully ✓

Anchor cell: AIY. Target V_rest range: [−95, −55] mV; central −75 mV.

| C_global | V_rest (AIY) | in range? |
|---:|---:|---|
| 1e3 | −88.9 mV | YES |
| **1e4** | **−85.4 mV** | **YES ← selected (closer to central)** |
| 1e5 | +26.3 mV | NO |
| 1e6 | +28.1 mV | NO |
| 1e7 | +29.8 mV | NO |

**C_global_AIY = 1.0 × 10⁴ channels/(cm²·TPM)** produces V_rest = −85.4 mV
at AIY. Within published range; on hyperpolarized side of central −75 mV.

**Convenient emergence:** Same C_global value works for both AVA and AIY
families. This suggests the per-family-C_global hypothesis is partly
right (cell families do need same C_global treatment) but may be more
homogeneous than expected. The cell-class V_rest differences arise from
DIFFERENT γ × TPM × surface area patterns per channel, not from
different C_global scaling.

### 2.3 RIM-class — CALIBRATION FAILED ✗ (substantive finding)

Anchor cell: RIM. Target V_rest range: [−65, −40] mV; central −52 mV.

Initial order-of-magnitude sweep:

| C_global | V_rest (RIM) | in range? |
|---:|---:|---|
| 1e3 | −11.3 mV | NO (way too depolarized) |
| 1e4 | −3.0 mV | NO |
| 1e5 | +14.3 mV | NO |
| 1e6 | +14.6 mV | NO |
| 1e7 | +14.6 mV | NO |

Refined low-range sweep:

| C_global | V_rest (RIM) | in range? |
|---:|---:|---|
| 1e1 (= 10) | −12.3 mV | NO |
| 3e1 | −12.3 mV | NO |
| 1e2 | −12.3 mV | NO |
| 3e2 | −12.1 mV | NO |
| 1e3 | −11.3 mV | NO |
| 3e3 | −9.4 mV | NO |

**Even at C_global = 10 (channels contribute essentially 0), RIM plateaus
at V_rest = −12.3 mV.** Target [−65, −40] is unreachable by adjusting
C_global alone.

**Diagnosis:** RIM's V_rest at near-zero channel conductance is set by
the substrate's pump + leak system from §7.2 v2, not by channel
parameterization. The §7.2 v2 calibration anchored pumps against AVAL
conditions; RIM's leak split (K/Na fractions derived from GHK-fit to
e_leak = −50 mV) interacts with pump electrogenicity differently for
RIM than for AVA/AIY, producing a substrate-level V_rest of −12 mV
independent of channel currents.

**Consistent with §7.2 v2 finding:** RIM was an outlier in pump-leak
balance under linear-TPM-density assumption. The §7.2 v2 work documented
this and deferred to v2 channel work (which we're doing now). v2 channel
work shows the pump-leak substrate issue is real and not addressable by
channel parameterization alone.

**v3 candidate refinements (for future work):**
1. RIM-specific leak split — re-derive K/Na fractions from RIM's
   physiological ion concentrations, not from GHK on Nicoletti's
   e_leak = −50 (which may itself be a non-unique fit)
2. RIM pump capacity adjustment — current §7.2 v2 anchors all cells
   to AVAL pump density; RIM may need different scaling
3. Investigate RIM-specific ion gradients — RIM may have different
   resting [K]_in or [Na]_in than the mammalian-default mid-cell value
4. Kinetic parameter audit (Layer 1.5 v3 or Layer 2) — RIM's leak
   reversal e_leak = −50 from Nicoletti is itself a fit-derived value
   subject to measurement-vs-fit audit (§8.11)

**v2 deployment decision:** Accept RIM Tier B failure for v2; document
as substantive finding. AVA + AIY calibration successful provides
methodology demonstration. RIM-specific refinement deferred to v3 or
Layer 2 work block.

---

## 3 · Calibrated values

```python
C_GLOBAL_PER_FAMILY = {
    "AVA": 1.0e4,   # calibrated against AVAL V_rest = -47.7 mV (in target range)
    "AIY": 1.0e4,   # calibrated against AIY V_rest = -85.4 mV (in target range)
    "RIM": 1.0e4,   # CALIBRATION FAILED; documented finding
}
```

Stored in `scripts/brain/wave2/channels/derived_channel_parameters.py`.

**Note:** All three families share C_global = 1.0e4 by coincidence of the
order-of-magnitude scan. This is NOT because per-family C_global is
unnecessary — RIM's failure shows the substrate-level issue is family-
specific. It's that AVA and AIY happen to satisfy V_rest with the same
order-of-magnitude C_global.

---

## 4 · Biophysical plausibility (per §3.4 methodology)

C_global = 1.0e4 channels/(cm²·TPM):
- Max density at max TPM (203.9 in AVA UNC-2): 2.04e6 channels/cm²
  (< 1e7 saturation threshold ✓)
- Max total channels per cell (max TPM × max surface):
  203.9 × 1.124e-5 × 1.0e4 = 22.9 channels (> 1 minimum ✓)
- Compared to v1 C_global = 1.73e4: ~1.7× smaller; gives slightly
  fewer channels per cell, more consistent with C. elegans small-
  neuron biology

C_global = 1.0e4 is biophysically plausible.

---

## 5 · Calibration methodology contribution

This calibration demonstrates the **machine-code-up principle (§2.9)
in action**:

- Full Layer 1 intracellular machinery preserved (ion dynamics + dynamic
  Nernst + pumps + Ca buffering)
- C_global calibrated against MEASURED V_rest (Liu 2020 + Nicoletti
  underlying voltage data), NOT against Nicoletti's derived gbar fits
- Same global constant works for AVA + AIY families (single-parameter
  calibration successful for 2/3 families)
- RIM failure surfaces a substrate-level issue (pump-leak balance from
  §7.2 v2) that's INDEPENDENT of channel parameterization — exposed
  cleanly because the calibration sweep can isolate substrate-level
  V_rest from channel-level V_rest contribution

**Methodologically:** v2 demonstrates that biology-derived parameters
with V_rest-measurement calibration can succeed (AVA + AIY) without
fitting to per-cell Nicoletti gbar values. Where it fails (RIM),
failure surfaces a different substrate-level issue rather than just
parameter values being wrong.

---

## 6 · Acceptance criteria status

- [x] C_global computed per family with explicit calibration path
- [x] Reference (V_rest measurement) used as anchor, not derived gbar fits
- [x] Biophysical sanity checks pass
- [x] AVA + AIY families calibrated successfully
- [ ] RIM family CALIBRATION FAILED → substantive finding for v3
- [x] Documentation complete

**Group C SHIPPED with 2/3 families calibrated + 1 substantive finding
for v3.** Group D (4-tier validation) proceeds with current C_global
values; RIM's expected Tier B failure documented.

---

## 7 · Files of record

- This document: `docs/c_global_per_family_calibration.md`
- Calibration script: `scripts/brain/wave2/calibrate_path2_v2.py`
- Sweep results JSON: `scripts/brain/wave2/artifacts/path2_v2_calibration_sweep.json`
- Derived module (updated): `scripts/brain/wave2/channels/derived_channel_parameters.py`
- v_rest_targets reference: `docs/v_rest_targets.md`
- Methodology reference: `docs/channel_parameter_derivation_methodology.md` §3.0
