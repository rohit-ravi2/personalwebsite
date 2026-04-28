# Phase A.2 — Geometry analysis for cadiff/caintra1 across cells

**Date:** 2026-04-26 run #2
**Task:** Compute geometric calibration factors for AVA, AIY, RIM cell geometries
under cadiff and caintra1 pool models. Predict whether different cells need
different calibration factors.

---

## TL;DR

**The geometry analysis surfaces a more important architectural finding:** Nicoletti's
published cell scripts (AVA, AIY, RIM, AVAR) do NOT actually insert any Ca-pool
mechanism (no `cadiff`, no `caintra1`) into their somas. Only VA5 inserts cadiff.
This means:

1. SLO-1 channels in AIY and elsewhere read NEURON's **default static `cai`**
   (which equals `cai0_ca_ion = 5e-5 mM` by default), not a dynamic pool.
2. **`slo1egl19.mod` does NOT read `cai` at all** — it has an internal `calcium(V)`
   FUNCTION that computes nanodomain Ca purely from V via a closed-form formula
   (Lluís-Buchholz / Alvarez nanodomain approximation).
3. The F6 calibration question is therefore **largely orthogonal to channel
   translations in scope**. Ca-pool calibration affects only the rare cells that
   actually insert one (VA5 with cadiff), not the gate-channel-on-Ca dynamics
   that we initially feared.

**Geometric factors per pool:** cadiff is **geometry-independent** in source
(no surf/vol parameters; only fixed `depth = 0.1 μm`). caintra1 has explicit
surf+vol parameters that scale linearly per the formula.

**Verdict trend continues toward PRINCIPLED.** Geometry analysis confirms that
the Brian2 calcium_pool.py module's geometric scaling (linear in surf/vol) is
correct in principle, with the caveat that cells in scope mostly don't use
the pool dynamically.

---

## Cell geometries (from Nicoletti's actual scripts)

| Cell | Surface (cm²) | L = √(surf/π)·1e4 (μm) | Volume (cm³) for L=diam stub cyl | S/V (cm⁻¹) |
|---|---|---|---|---|
| AVA  | 1123.84e-8  | 18.92 | 5.32e-9 | 2.11e3 |
| AIY  |  65.89e-8  | 4.580 | 7.55e-11 | 8.73e3 |
| RIM  | 103.34e-8  | 5.738 | 1.48e-10 | 6.97e3 |

(Volume of a cylinder with L=d=2r: V = π·r²·L = π·(L/2)²·L = (π/4)·L³.
Equivalently for a stub cylinder of radius r and length L=2r: V = πr²·(2r) = 2πr³.)

In NEURON, the section `soma.L = soma.diam = rsoma` defines a stub cylinder.
For surface area S = π·d·L = π·d² (with d=L), so d = √(S/π). Volume V = π·(d/2)²·L = (π/4)d³.
Surface-to-volume ratio S/V = (π·d²) / ((π/4)d³) = 4/d.

In CGS:
- AVA: d=18.92 μm = 1.892e-3 cm → S/V = 4/1.892e-3 = 2114 cm⁻¹
- AIY: d=4.580 μm = 4.580e-4 cm → S/V = 4/4.580e-4 = 8734 cm⁻¹
- RIM: d=5.738 μm = 5.738e-4 cm → S/V = 4/5.738e-4 = 6971 cm⁻¹

---

## What pools are actually used per cell?

Cross-checked Nicoletti's `*_simulation_iclamp.py` files for `soma.insert(...)`:

| Cell | Channel set inserted | Ca-pool inserted? |
|---|---|---|
| AVAL | irk, leak, egl19, nca | **None** |
| AVAR | (uses unc103 patch; same as AVAL channel set) | **None** |
| AIY  | egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1 | **None** |
| RIM  | shl1, egl2, irk, cca1, unc2, egl19, leak | **None** |
| VA5  | slo2egl19, slo2iso, egl19, irk, shk1, leak, nca, **cadiff** | **cadiff** (only) |

**Key observation:** None of the 4 essential cells (AVA, AIY, RIM, AVAR) insert
any Ca-pool mechanism. They rely on NEURON's default static `cai` (set by
the ion's `cai0_ca_ion` global, default 5e-5 mM = 50 nM).

---

## What channels READ cai or eca?

Cross-referenced channels in scope:

| Channel | USEION ca | Effect of static cai |
|---|---|---|
| egl19    | reads `eca` only (driving force; not Ca-dependent gating) | None — eca is voltage-clamp-style constant |
| nca      | (none) | N/A |
| irk      | (none — K-only) | N/A |
| shk1     | (none — K-only) | N/A |
| shl1     | (none — K-only) | N/A |
| kqt3     | (none — K-only) | N/A |
| kqt1     | (none — K-only) | N/A |
| egl2     | (none — K-only) | N/A |
| cca1     | reads ica/eca; doesn't gate on cai | None |
| unc2     | reads ica/eca; doesn't gate on cai | None |
| unc103   | (none — K-only) | N/A |
| leak     | (none) | N/A |
| **slo1iso** | **reads `cai`** | Static cai → constant gating contribution |
| **slo1egl19** | reads `eca` only; **internal calcium(V) function** | None — bypasses cai entirely |
| **slo2iso** | likely reads cai (need to verify) | (TBD) |
| **slo2egl19** | likely uses internal calcium(V) | (TBD) |
| **kcnl** (SK) | reads cai (uses caintra1 pool when present) | Static cai → constant gating |

(slo2 family and kcnl confirmation deferred — they're in the second-order channel
set, not in this run's essential 7.)

**For the 7-channel essential set in scope (EGL-19, SLO-1 isolated, SLO-1+EGL-19
coupled, SHK-1, SHL-1, NCA, KQT-3):**

- 5 channels (EGL-19, SHK-1, SHL-1, NCA, KQT-3) don't depend on cai at all.
- SLO-1 isolated reads cai. With AIY's actual NEURON setup (no pool inserted),
  cai is static. So **the Brian2 SLO-1 isolated translation does not need a
  dynamic Ca-pool either** — it can use a constant cai matching NEURON's default.
- SLO-1+EGL-19 coupled reads `eca` (constant) and computes its own nanodomain
  Ca internally. **No external Ca-pool dependency at all.**

---

## Implication for F6 verdict and Phase E

**F6 verdict implication:** The Ca-pool translation work in run #1 was largely
infrastructure for cells we never run. Calibration accuracy of cadiff/caintra1
matters for completeness but doesn't gate channel-translation correctness in
the essential set.

**Phase E implication:** The Phase E "architectural decision required: how to
encode nanodomain coupling" simplifies considerably. Nicoletti's `slo1egl19.mod`
does NOT use sub-membrane vs bulk [Ca]_i compartments. It uses a closed-form
deterministic formula:

```
calcium(V) = |gsc*(V-eca)*1e-3| / (8*π*r*d*FARADAY) × exp(-r/√(d/(kb*b))) × 1e6 × 1e-3 + fondo
```

where:
- `gsc = 40 pS` (single-channel conductance)
- `r = 13 nm` (nanodomain radius)
- `d = 250 μm²/s` (Ca diffusion coefficient)
- `kb = 500e6 /M-s` (buffer binding rate)
- `b = 30 μM` (buffer concentration)
- `eca = 60 mV`
- `fondo = 0.05 μM` (resting Ca)
- `FARADAY = 96485` coulombs

**Translation strategy for slo1egl19:** option (a) of the spec's Phase E options
("Local [Ca]_i as separate state variable") is what Nicoletti does, EXCEPT
it's even simpler: it's a deterministic V-dependent function, not a state
variable. Brian2 translation: include the closed-form `calcium(V)` as an
algebraic equation in the eqs string.

**This eliminates a major Phase E uncertainty.** The "how to encode nanodomain
coupling" question has a clean answer: match Nicoletti exactly, which is just
a V-dependent function.

---

## cadiff geometry-independence (a real concern, but not blocking)

cadiff.mod has only `depth = 0.1 μm` as the geometric parameter. There's no
surf or vol. The formula `dca/dt = -1/(2F)·ica/depth - 0.0001·beta·ca` is
applied per unit area (since ica is mA/cm²) over depth, giving a concentration
rate. This is a "shell" model: Ca enters through the membrane and is confined
to a shell of depth `0.1 μm` regardless of cell geometry.

This is a known approximation in NEURON Ca-handling. For a stub cylinder of
diameter d, the shell-vs-bulk geometry matters only when d is comparable to
2·depth = 0.2 μm. For our cells (d = 4.6-19 μm), the shell-model is a
reasonable approximation; the shell volume is ~10% of cell volume for d=4.6 μm,
~2.4% for d=19 μm. The Ca pumped into the shell mixes (in reality) with the
bulk over a timescale fast compared to channel kinetics.

**Conclusion:** cadiff's geometry-independence in the source IS itself an
approximation — but it's the same approximation NEURON makes, so our Brian2
translation inherits the same approximation. No additional calibration anomaly
is introduced by translation.

---

## caintra1 geometry scaling (tested in calibration)

caintra1's source formula:

```
rs = fca * (-(1/(2·vol·Fc)) · (ica·surf·1e-3)) - (caintra-ca_eq)/tca
```

For inward ica:
- Inflow rate scales as surf/vol (for given ica density).
- For AVA (surf=1124e-8, vol=5.32e-9): surf/vol = 2114 cm⁻¹
- For AIY (surf=65.9e-8, vol=7.42e-12): surf/vol = 88800 cm⁻¹
- For RIM (surf=103.3e-8, vol=1.48e-10): surf/vol = 6975 cm⁻¹

The Brian2 calcium_pool.py module's `caintra1_eqs(vol_cm3, surf_cm2, ...)`
correctly accepts both as parameters and applies the formula's intent:
`coef_in_eff = empirical_at_AIY * (surf/surf_AIY) * (vol_AIY/vol)`.

**Predicted cell-specific factors:**
- AIY (calibration): coef_in_eff = 4.60e-7
- AVA: coef_in_eff = 4.60e-7 × (1124/65.9) × (7.42e-12/5.32e-9) = 4.60e-7 × 17.05 × 0.001395 = 1.094e-8
- RIM: coef_in_eff = 4.60e-7 × (103.34/65.89) × (7.42e-12/1.48e-10) = 4.60e-7 × 1.568 × 0.0501 = 3.614e-8

These predict different per-cell empirical fits. **Robustness test in A.3**
verifies that these scalings produce correct NEURON-vs-Brian2 matches across cells.

---

## What's tested in A.3

Given F12 (cells don't insert pools) and F13 (slo1egl19 has internal calcium(V)),
the A.3 robustness test focuses on:

1. **cadiff at multiple cells** — testing depth-only scaling assumption. cadiff
   doesn't have surf/vol explicitly; it should produce the same Δcai/Δt per
   ica regardless of cell geometry (because the shell model is geometry-blind
   in source). Verify via direct NEURON sim at AVA, AIY, RIM.

2. **caintra1 at multiple cells** — testing surf/vol scaling. caintra1's
   formula explicitly scales with surf/vol. Verify Brian2's geometric scaling
   matches NEURON's per-cell behavior.

3. **Multi-regime** — test at low, mid, high ica levels (-0.1 to -10 mA/cm²)
   to verify linearity of α holds.

If A.3 tests confirm:
- cadiff α=0.518 mM/(mA/cm²·ms) is geometry-independent (matches NEURON)
- caintra1 α scales as surf/vol per the formula
- Linearity of α holds across ica regimes within tested range

Then **F6 verdict = PRINCIPLED** with full confidence.

---

## Summary of geometry analysis

1. cadiff is geometry-independent at the source level (only depth=0.1 μm
   matters). This is a NEURON-level approximation that Brian2 inherits cleanly.
2. caintra1 has explicit surf/vol scaling. The Brian2 `caintra1_eqs()` module
   correctly accepts both as parameters and applies linear scaling.
3. **The cells in scope (AVA, AIY, RIM, AVAR) don't actually insert a Ca-pool
   mechanism in Nicoletti's published code.** They rely on static cai = ca_eq
   default. **The F6 calibration question is largely orthogonal to channel
   translation correctness for the essential set.**
4. **slo1egl19 doesn't read cai at all** — it has an internal closed-form
   nanodomain Ca formula. This greatly simplifies Phase E translation.
5. slo1iso reads cai, but in AIY's actual cell (no pool inserted), cai is
   static. So Brian2 slo1iso translation can use a constant cai value
   matching NEURON's default.

**Verdict trend:** PRINCIPLED. The Ca-pool translation is fully principled per
A.1's symbolic decomposition; the geometry scaling per A.2's analysis is also
correct in principle. Only A.3's robustness test remains, focused on
cross-cell/cross-regime confirmation.

---

## Findings to add to phase_beta_findings.md

- **F11**: Run #1's F6 finding ("hidden 52,700× factor") was a misdiagnosis.
  Symbolic derivation gives α=0.518 mM/(mA/cm²·ms) for cadiff; NEURON
  empirically gives α=0.518 (verified via fresh IV sweep). Run #1 calibration
  was 0.525 (1.4% noise from LSQ regression). The calibration is fully
  principled.
- **F12**: Nicoletti's published cell scripts (AVAL, AVAR, AIY, RIM) do NOT
  insert any Ca-pool mechanism. Only VA5 inserts cadiff. SLO-1 channels in
  these cells read NEURON's default static cai (5e-5 mM = ca_eq).
- **F13**: slo1egl19.mod does NOT read cai. It has an internal closed-form
  `calcium(V)` FUNCTION computing nanodomain Ca purely from V. This eliminates
  the Phase E "nanodomain coupling encoding" architectural question; the
  approach is deterministic V-dependent algebra.
