# Layer 1 §7.3 — Per-cell integration findings (2026-05-12)

**Status:** Infrastructure SHIPPED. Acceptance criteria UNMET pending §7.3.5
channel refit. Substantive structural finding documented.

**Reference docs:**
- `docs/layer1_design_decisions.md` §8 — inherited parameter audit methodology
- `docs/substrate_redesign_roadmap.md` §7.3.5 — Layer 1.5 entry

---

## 1 · Deliverables shipped

```
scripts/brain/wave2/layer1_cells.py           Layer 1 integrated cell builders
                                              (AVAL, AVAR, RIM, AIY)
scripts/brain/wave2/validate_layer1_cells.py  5s rest validation script
```

Per-cell builder composes:
- Ion dynamics + dynamic Nernst (Layer 1 §7.1)
- Pumps: Hill Na/K + Payne KCC-2 + approximate ABTS-1 + threshold-MM Ca
  clearance (Layer 1 §7.2 v2)
- Nicoletti channel set per cell (Wave 2 channel modules), with channel
  reversal potentials BRIDGED to dynamic Nernst via subexpression
- LEAK split into K + Na components by GHK-derived permeability fractions
  (preserves Nicoletti's e_leak)
- Membrane V equation summing all electrogenic currents

Per-cell channel sets:
- **AVAL**: EGL-19 (Ca), IRK (K), NCA (g=0, no-op), LEAK (split)
- **AVAR**: AVAL set + UNC-103 (K), with NCA g ≠ 0
- **RIM**: SHL-1, EGL-2, IRK, CCA-1, UNC-2, EGL-19, LEAK (6 channels)
- **AIY** (v1, simplified): EGL-19, KQT-1, SHL-1, NCA, LEAK (5 channels;
  SLO-1 family deferred — SLO-1 + EGL-19 coupled and SLO-1 isolated forms
  require additional state + Ca coupling and are out of §7.3 v1 scope)

Pump parameters TPM-scaled from AVAL anchor (§7.2 v2 calibrated values).

---

## 2 · Validation results (5s rest, default ion concentrations)

```
cell    V_rest    [K]_in       [Na]_in       [Cl]_in     [Ca]_in       verdict
AVAL    -53.2 mV  115.4 mM     15.6 mM       4.55 mM     2.3 μM        FINDING
                  Δ -17.6%     Δ +55.5%      Δ -8.9%     Δ +4544%
AVAR    -36.8 mV   56.3 mM    100.1 mM       5.76 mM     55 μM         FINDING
                  Δ -59.8%     Δ +901%       Δ +15.2%    Δ +109000%
RIM     +13.8 mV    2.4 mM     61.2 mM       9.5 mM      666 μM        CATASTROPHIC
                  Δ -98.3%     Δ +512%       Δ +89.6%    Δ +1.3M%
AIY     +29.1 mV    1.3 mM    153.8 mM       5.6 mM      200 μM        CATASTROPHIC
                  Δ -99.1%     Δ +1438%      Δ +11.9%    Δ +400000%
```

V_rest for AVAL/AVAR landed within published range (Mellem 2008 + Stage IV
Wave 2). All four cells fail ion stability acceptance.

Severity correlates with channel density + Ca channel count:
- AVAL: 1 Ca channel, moderate density → moderate drift
- AVAR: 1 Ca channel, AVAL set + UNC-103 NCA non-zero → worse drift
- RIM: 3 Ca channels (EGL-19 + CCA-1 + UNC-2) at high density → catastrophic
- AIY: 1 Ca channel + small cell volume → catastrophic per-volume rates

---

## 3 · Root cause diagnosis

### 3.1 Nicoletti's implicit E_Ca = 60 mV assumption

Nicoletti 2024 simulations use **fixed E_Ca = 60 mV** as a model parameter
throughout. Under physiology with `[Ca]_out = 2 mM`:

```
E_Ca = (RT/2F) · ln([Ca]_out / [Ca]_in)
60   = 12.63 · ln(2 / [Ca]_in)
ln(2/[Ca]_in) = 4.75
[Ca]_in = 2 / 115.6 = 17.3 μM
```

So Nicoletti's E_Ca = 60 mV implies **[Ca]_in ≈ 17 μM** — 340× higher than
the mammalian-default 50 nM authorized in §6.5.

This is invisible in Nicoletti's published model because their simulations
don't track [Ca]_in explicitly — the reversal potential is hardcoded as
a fitting parameter, not derived from physical state.

### 3.2 Driving force discrepancy

Under Nicoletti's fit assumptions (E_Ca = 60):
- At V_rest ≈ −39 mV (AVAL): V − E_Ca = −99 mV

Under Layer 1's physiological substrate (E_Ca = 134 mV at [Ca]_in = 50 nM):
- At V_rest ≈ −53 mV: V − E_Ca = −187 mV

The Layer 1 driving force is **89% larger** in magnitude. Same gbar
produces nearly 2× the Ca current at rest.

### 3.3 Cascade to runaway

1. Excess Ca influx through EGL-19 (and CCA-1/UNC-2 where present)
2. Lumped Ca clearance (mca-3 TPM-scaled) saturates at I_max but can't
   match the excess influx
3. [Ca]_in accumulates into μM-mM range
4. As Ca rises, E_Ca decreases (Nernst), but slowly (κ_B = 100 buffering)
5. Ca channel activation depolarizes V (Ca is depolarizing)
6. Depolarization activates more Ca channels (positive feedback)
7. K channels (IRK, etc.) only partially compensate; their gbar was also
   tuned to balance Nicoletti's fixed-E_Ca regime
8. Pump capacity (Na/K-ATPase) cannot maintain K + Na gradients under
   depolarized V; K leaks out faster than pump can restore
9. RIM/AIY cross the failure threshold; V drifts to Ca/Na reversal range

The cascade terminates at a new "equilibrium" where saturated pump matches
excess channel influx, but at non-physiological [Ca]_in and depolarized V.

### 3.4 Quantitative steady-state check (AVAL)

At AVAL's final state ([Ca]_in = 2.3 μM, V = −53 mV):
- E_Ca = 12.63 · ln(2/2.3e-3) = 12.63 · 6.77 = 85.5 mV
- EGL-19 driving force: V − E_Ca = −138 mV
- EGL-19 ica ≈ g · m · h · ΔV ≈ 9.29e-6 · 0.0015 · 0.5 · (−138) = −0.97 μA/cm²
- Ca clearance at delta = 2.27 μM: ca_clear_I_max · delta/(K_half + delta)
  = 2.02e-6 · 2.27e-3 / (5e-4 + 2.27e-3) = 2.02e-6 · 0.82 = 1.65 μA/cm²

Pump > channel influx, so [Ca]_in is approaching equilibrium from above
(still decreasing slightly at t=5s; would reach steady state at lower
[Ca]_in given longer simulation). The substrate trajectory confirms the
inherited-parameter mismatch.

---

## 4 · Acceptance criteria status

```
                                AVAL  AVAR  RIM   AIY
±2% [K]_in stability:           FAIL  FAIL  FAIL  FAIL
V_rest in published range:      PASS  PASS  FAIL  FAIL
[Cl]_in in [3, 7] mM:           PASS  PASS  FAIL  PASS
[Ca]_in near 50 nM target:      FAIL  FAIL  FAIL  FAIL
No runaway concentration drift: FAIL  FAIL  FAIL  FAIL
```

**Overall: §7.3 acceptance criteria UNMET on all four cells.**

---

## 5 · Why this is a substantive finding, not a regression

Per Rohit's 2026-05-12 §7.3 direction:
> Don't autonomously retune pump parameters during §7.3 — if real channels
> don't resolve drift, surface as substantive finding rather than retuning

The finding surfaces a transferable methodological insight:

**Inherited parameter fits encode implicit ion-state assumptions invisible
until the substrate makes state explicit.** When [Ca]_in becomes a state
variable, Nicoletti's fit reveals its assumed [Ca]_in. This is true of any
inherited fit against a fixed-reversal model: the fit is internally
consistent against its own assumption but not biophysically transferable.

The methodology contribution: **"parameter audit before integration"** as
a standing step in the substrate redesign roadmap. See design doc §8.5 +
roadmap cross-cutting tracks for forward-looking flags on other parameter
sets at risk (Wicks 1996 graded release, Nicoletti Ca pool, peptide
release constants).

This is the substrate-redesign methodology paying off: by making implicit
state explicit, we surface previously-invisible inconsistencies in inherited
fits. The right resolution is §7.3.5 audit + refit (the deferred work block),
not §7.3 retuning.

---

## 6 · Resolution path

§7.3.5 (Layer 1.5) — Channel-Substrate Consistency Audit and Refit.

See `docs/substrate_redesign_roadmap.md` §7.3.5 entry for scope, acceptance
criteria, pre-flight scoping questions, and estimate (3-5 work blocks).

§7.3.5 BLOCKS §7.4 (Phase F restructure depends on correct Ca dynamics
feeding ATP consumption). The blocking dependency is documented.

Pre-flight scoping required before §7.3.5 deployment:
- Refit objective function (which features to match: peak I, steady-state
  I, kinetics, weighted combination)
- Refit method (manual visual match vs digitized least-squares)
- K-channel refit decision (E_K shift is smaller but non-zero)
- Validation tolerance specification

---

## 7 · Files of record

- This document: `scripts/brain/wave2/artifacts/layer1_7_3_findings.md`
- Layer 1 design + methodology: `docs/layer1_design_decisions.md` (§8 added)
- Roadmap: `docs/substrate_redesign_roadmap.md` (§7.3.5 added)
- Cell builders: `scripts/brain/wave2/layer1_cells.py`
- Validation script: `scripts/brain/wave2/validate_layer1_cells.py`
- Pump infrastructure (still valid, unaffected): `scripts/brain/wave2/pumps/`
- Ion dynamics foundation (still valid, unaffected):
  `scripts/brain/wave2/ion_dynamics.py`
