# Wave 2 option α-1 CP3 — AVA cell construction

**Date:** 2026-04-26 Session 3 (option α-1 resumed scope)
**File:** `wave2/option_alpha_ava_cell.py`
**Target:** Nicoletti's actual AVAL phenotype (true 4-channel set)

---

## Channel set: `[IRK + LEAK + EGL19 + NCA]`

**Source:** `nicoletti_2024/AVAL_simulation_iclamp.py` lines 29-32:
```python
soma.insert('irk')
soma.insert('leak')
soma.insert('egl19')
soma.insert('nca')
```

This is the canonical, published 4-channel AVAL. The original Wave 2 option α
prompt's "5-channel set including UNC-103" was an orchestrator-side error
(see `option_alpha_pushback.md` and `option_alpha_findings.md`). UNC-103
belongs to AVAR, not AVAL. The 4-channel set is the biological referent for
"Nicoletti's actual AVAL phenotype" the option α directive instructs us to
ground in.

---

## Parameter vector (Nicoletti AVAL canonical)

**Source:** `AVAL_simulations.py` line 26:
```python
g0 = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]
#     [egl19,    leak,     irk, nca, eleak, cm]   units: nS for first 4, mV for eleak, μF/cm² for cm
```

**Surface area:** `surf = 1123.84e-8 cm²` (from neuromorpho AVAL).

**Conversion to S/cm² (Brian2 intensive convention):**
```
g_Scm2 = g_nS * 1e-9 / surf
```
Yields:

| Channel | g (nS) | g (S/cm²)    |
|---------|--------|--------------|
| egl19   | 0.1044 | 9.288e-6     |
| leak    | 0.1502 | 1.336e-5     |
| irk     | 0.1    | 8.898e-6     |
| nca     | 0      | 0            |

`e_leak = -39 mV`, `cm = 0.859551 μF/cm²`, `eca = 60 mV`, `ek = -80 mV`.

---

## Architectural decisions

### NCA inclusion with gbar=0 (no-op numerically)

**Decision:** include NCA in the cell equations with gbar=0.

**Rationale:** Nicoletti inserts NCA in `AVAL_simulation_iclamp.py`
(line 32) but assigns `g_nca = 0` in her parameter vector. Numerically this
is a no-op — NCA contributes zero current. But for apples-to-apples fidelity
with Nicoletti's NEURON cell, we replicate her insertion list. Also keeps
the cell structure consistent if future work non-zeros NCA's gbar (e.g.,
sensitivity analyses, AVAR transition).

**Alternative considered:** omit NCA entirely. Numerically equivalent but
diverges from Nicoletti's published cell at the structural level. Rejected
to preserve apples-to-apples comparison fidelity.

### No Ca pool (no cadiff, no caintra1)

**Decision:** do NOT insert any Ca-pool mechanism.

**Rationale:** Nicoletti's `AVAL_simulation_iclamp.py` does NOT insert
cadiff or caintra1. EGL-19 produces ica that's not consumed by any pool.
NEURON's default `cai = 5e-5 mM` (P13 in translation_patterns) is what
EGL-19's reading would see (though EGL-19's gating is voltage-only — no
cai dependence in Nicoletti's parameterization, verified per egl19.py
DERIVATIVE block).

**Alternative considered:** add caintra1 for completeness. Rejected: Nicoletti
deliberately omitted it from AVAL (suggesting her AVA model doesn't track
internal Ca dynamics), so adding it would diverge from her published model.

### No SLO-1, SHK-1, SHL-1, KQT-3, UNC-103

**Decision:** do NOT include any of these channels.

**Rationale:**
- **SLO-1, SHK-1, SHL-1, KQT-3:** these are AIY/RIM/etc. channels, not in
  Nicoletti's AVAL. Including them produces a synthetic, non-Nicoletti AVA.
  This is what Phase F's 2b cell did (7-channel essential set) — useful for
  testing Mellem 2008 plateau dynamics but explicitly NOT Nicoletti's AVAL.
- **UNC-103:** in AVAR, not AVAL. Including it produces neither AVAL
  (different from 5-channel) nor AVAR (different geometry/parameters).
  CP1 translates UNC-103 for future AVAR work, but it's NOT in CP3's cell.

### Geometry from AVAL_simulations.py

**Decision:** use Nicoletti's exact AVAL surface area.

```
surf = 1123.84e-8 cm²    # neuromorpho AVAL stub-cylinder approximation
```

**Implementation:** stub-cylinder with `L = sqrt(surf/π) × 1e4 = 18.91 μm`
(from her wrapper). Brian2 doesn't distinguish L vs diam at the
single-compartment level; only surface matters.

### v_init = -60 mV

**Decision:** initialize at -60 mV per Nicoletti's standard.

**Source:** `AVAL_simulation_iclamp.py` line 75: `h.finitialize(-60)`.

This is hyperpolarized relative to the cell's natural rest potential under
the leak-dominated regime (-39 mV e_leak suggests rest near there at 0
injection). The cell will drift toward rest during settle phase.

---

## Differences from existing Phase F 2a cell

`validate_phase_f_gate2.py` line 56 (`build_brian2_ava_2a`) constructs a
3-channel `[leak + EGL-19 + NCA]` cell — explicitly missing IRK. That cell
was the correct apples-to-apples comparison given Phase F's deliberately
restricted scope (testing channel kinetics in a minimal cell context).

**This cell adds IRK** to give the true 4-channel published parameterization,
matching Nicoletti's `AVAL_simulation_iclamp.py` literally.

---

## Differences from existing Phase F 2b cell

`validate_phase_f_gate2.py` line 188 (`build_brian2_ava_2b`) constructs a
7-channel `[leak + EGL19 + NCA + SLO1iso + SLO1egl19 + SHK1 + SHL1 + KQT3]`
cell — adding 5 non-Nicoletti channels for Mellem 2008 plateau testing.
That cell has no biological referent in Nicoletti's actual AVAL; it was
designed to test architectural sufficiency of an "essential set" hypothesis
that ultimately failed (Condition 6 surfaced).

**This cell explicitly excludes** SLO-1/SHK-1/SHL-1/KQT-3 to ground in
Nicoletti's actual published parameterization.

---

## Smoke test results

`python wave2/option_alpha_ava_cell.py` self-test:
```
Initial V: -60.00 mV
Final V (after 100 ms): -44.54 mV
V range: [-60.00, -44.54] mV
i_total_mAcm2 range: [-2.536e-04, -6.074e-05] mA/cm²
  ica_egl19 (final): -4.260e-06 mA/cm² (inward Ca current at -45 mV)
  ik_irk    (final):  1.749e-05 mA/cm² (outward K current — consistent with hyperpolarized hold)
  i_leak    (final): -7.397e-05 mA/cm² (inward leak driving toward e_leak=-39)
```

Cell builds, integrates, and shows physiologically reasonable currents at
intermediate hold. Settling toward ~-44 mV reflects the balance of leak
(driving toward -39 mV) against IRK (driving toward -80 mV at activated
state) plus small EGL-19 Ca contribution.

---

## Use in CP4

CP4 will:
- (component 2a) compare voltage-clamp Brian2 4-channel vs NEURON 4-channel
  via `NEURONReference("AVAL")` (which already constructs Nicoletti's
  canonical AVAL via her wrapper).
- (component 2b) compare current-clamp using the **1000 ms protocol**
  (NOT 100 ms Mellem-legacy) with 7 current steps -30 to +30 pA per
  Nicoletti's actual `AVAL_simulation_iclamp.py` lines 53-55, 69, and
  `AVAL_simulations.py` line 31.

NEURON reference path is direct upstream invocation (cleanest, and IS the
published reference) rather than NEURONReference custom-mode synthetic
construction.
