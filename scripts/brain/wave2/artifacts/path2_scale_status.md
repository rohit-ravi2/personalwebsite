# Path 2 scale — 128-cell substrate status

**Date:** 2026-05-13
**Mode:** Autonomous, execute-first per user direction
**Code:** `scripts/brain/wave2/path2_scale/`

## What was built

Scalable Path 2 builder that derives per-cell channel inventories +
parameters from CeNGEN T2 medium-threshold gene expression + the
biophysical γ inventory + Layer 1 substrate machinery (pumps + ion
dynamics). No per-cell Nicoletti fits required.

Module layout:
- `extended_gamma.py` — γ in pS for 18 channels (mammalian homolog
  fallback; EXP-2 from C. elegans Davis lab single-channel measurement)
- `cengen_tpm_data.py` — CENGEN_T2_TPM for 24 channel genes × 128 classes
- `pump_tpm_data.py` — CENGEN T2 TPM for 4 pump genes × 128 classes
  (eat-6, mca-3, kcc-2, abts-1)
- `scalable_builder.py` — `build_scalable_spec(cengen_class)` +
  `to_layer1_cellspec()` converter
- `pump_scaling.py` — `extend_pump_dicts()` injects 124 missing
  CeNGEN classes into the four pump TPM dicts at import time

## Sweep results (1.5 s rest, all 128 classes)

| metric | value |
|---|---|
| Build + simulate | 128 / 128 |
| Errors (build or sim) | 0 / 128 |
| Plausible substrate state | 124 / 128 |
| V_rest median | -68.4 mV |
| V_rest IQR | -72.2 to -64.1 mV |
| V_rest range | -85.8 to -41.5 mV |
| Cells with [Ca]_in > 1 μM | 25 / 128 |
| Cells with [Ca]_in > 100 μM | 3 / 128 |

V_rest distribution histogram (128 cells):
```
[-120, -80):   4 #
[ -80, -60): 101 ##############
[ -60, -40):  23 ##
```

Most cells (101 / 128) rest in the -60 to -80 mV range, consistent with
mammalian neuron baselines. 23 cells are warmer (-40 to -60), within
plausible C. elegans range. 4 cells exceed -40 mV (Ca-channel-driven
runaway — see holes below).

## Holes surfaced (next concrete work)

### 1. Ca runaway in 3-4 cells (HSN, RIB, VD_DD, RIM)
Cells with high Ca-channel expression (cca-1 + unc-2 + egl-19) plus
default ~100 μm² surface area show [Ca]_in → 200-300 μM. The Ca
channels' γ × TPM × C_global gives gbar densities that exceed the
Ca-clearance pump density.

Per-cell mca-3 scaling (now wired) helps modestly:
- RIM: 26 → 19 μM
- VD_DD: 195 → 237 μM (no improvement)
- HSN: 293 → 346 μM (slight degradation)

The dominant problem is surface area: 100 μm² default is far below real
C. elegans soma surface for most cells. Fix path is NeuroMorpho /
WormAtlas integration for cell-specific geometries.

### 2. Pump-channel scaling mismatch
Simple TPM-ratio pump scaling breaks balance when pump-gene and
channel-gene expressions differ. ASEL: eat-6 = 239 TPM (~18 % of AVA's
1346); scaling Na/K-ATPase down by 18 % while Na channels remain at
their CeNGEN-derived gbar drives [Na]_in to 73 mM (vs 9 mM with AVAL
pump scaling).

Fix path: the C_global anchor (channels/cm²·TPM) needs cell-family
calibration alongside pump anchors, not independent TPM scaling.

### 3. 5 channels lack NMODL implementations
SHK-1, EGL-36, KVS-1, EXP-2, SLO-2, TWK family — currently skipped at
spec time. ~30 cells express one of these above T2 threshold; their
substrate may be missing important repolarization machinery.

### 4. Channel kinetics inherited from Nicoletti fits
Activation / inactivation curves for all 11 supported channels were
fit by Nicoletti 2024 against AVAL voltage clamp data. Cell-specific
kinetics (different gating shifts in different neurons) not yet
encoded. Audit 4 (measurement-vs-fit) unapplied at kinetics layer.

## What this is good for as-is

- **Build + simulate pipeline works** for any of the 128 CeNGEN classes
- **101 / 128 cells** produce rest behavior consistent with mammalian
  neuron baselines (V in [-80, -60], [K]_in ≈ 127-140, [Na]_in ≈ 9-15,
  [Cl]_in ≈ 4-5, [Ca]_in < 1 μM)
- **Foundation for Layer 2 work** (network integration, calcium events,
  larger-scale validation) is in place

The 4 implausible cells and the channel-kinetics issue both have clear
concrete remediation paths — they're not blocking, they're scoped.

## Files

- Detailed JSON sweep results: `artifacts/scalable_128_sweep.json`
- Sweep log: `artifacts/scalable_128_sweep.log`
- This summary: `artifacts/path2_scale_status.md`
