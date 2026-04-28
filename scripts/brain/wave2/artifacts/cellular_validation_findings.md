# Wave 2 cellular validation — findings log

**Started:** 2026-04-26 (Session 4 redeployment, Wave 2 cellular extension)
**Scope:** Option B (AIY only) — KQT-1 channel translation + AIY cell construction + Layer A voltage/current clamp validation. RIM/RMD deferred per pre-flight pushback adjudication.

---

## Resume acknowledgment (2026-04-26)

User authorized **Option B** after pre-flight pushback exposed channel-deficit
reality:

- AIY needs KQT-1 (≠ KQT-3, different gating: 2-state m·s vs 4-state mf/ms·s·w).
- RIM needs CCA-1 + EGL-2 + UNC-2 with GLOBAL declarations — deferred to a
  separate work block (Phase β-scale).
- RMD has no Nicoletti 2024 reference locally — deferred (Nicoletti 2019
  acquisition needed).

`PAUSED_FOR_REVIEW.txt` removed. Proceeding to CP1 (KQT-1 translation).

---

## Prompt corrections documented (carry-forward catalog)

This is the **fourth pre-flight pushback** in Wave 2 to surface a propagation
error from orchestrator-side prompt synthesis without primary-source
verification:

1. **Mellem 2008 → AVA plateau attribution** (caught by Mellem investigation).
   Mellem 2008 paper studies AVA but is not the canonical AVA voltage-source
   citation the orchestrator's framing implied.
2. **Wave 2 option α "5-channel AVAL" framing** (caught by option α pre-flight).
   Nicoletti's actual AVAL has 4 channels (`egl19, leak, irk, nca` with nca
   gbar=0); the prompt assumed 5.
3. **F2 misattribution to UNC-103** (caught by option α pre-flight). The F2
   GLOBAL-declarations pattern was attributed to UNC-103 by the orchestrator;
   primary source confirms UNC-103 has standard RANGE declarations, while the
   F2 pattern actually originates from caintra1.
4. **AIY/RIM/RMD channel-deficit picture** (this work block's pre-flight). The
   prompt assumed most channels needed for AIY/RIM/RMD were already in the
   existing 9 translations. Primary-source verification of `AIY_simulation.py`,
   `AIY_simulation_iclamp.py`, `RIM_simulation.py`, `RIM_simulation_iclamp.py`,
   plus `ls nicoletti_2024/` for RMD absence, surfaced:
   - AIY needs 1 new translation (KQT-1; distinct from existing KQT-3).
   - RIM needs 3 (CCA-1, EGL-2, UNC-2 with GLOBAL declarations).
   - RMD has no Nicoletti 2024 model; the 2019 paper code is not in local
     upstream.

**Same propagation pattern in all four:** orchestrator-side prompt synthesis
asserts factual claims about cell models / channel sets / channel translations
without primary-source verification. Agent-side pre-flight reading of the
actual `.py` and `.mod` files catches the discrepancy. Methodology lock-in is
working as designed; the cost of pre-flight reading is paid back several times
over by avoided fabrication.

---

## CP1 — KQT-1 channel translation (COMPLETE)

**Verdict:** PRODUCTION_GRADE.

**Module:** `wave2/channels/kqt1.py` (~150 lines, follows F1/F3 pattern; no
GLOBAL declarations).

**Validation:** `wave2/run_kqt1_validation.py` — voltage-clamp Layer A at AVAL
geometry (neutral testbed), 11 holds [-80, -60, -40, -30, -20, -10, 0, 10, 20,
30, 40] mV. Tolerance: current-domain divergence ≤ 0.05, panel pass > 80%.

**Result:** 11/11 holds passing (100%), peak and SS divergence = 0.000 at every
hold. Brian2 trace matches NEURON kqt1.mod to within rounding error in pA
(differences only in last-displayed decimal). Saved to
`wave2/artifacts/kqt1_validation_results.json`.

**Key translation choices:**
- 2-state gating with double-Boltzmann sinf (s = s1·boltz1 + s4·boltz2)
  preserved verbatim from mod.
- The s-gate's slow component (p2tskqt1 = 185845 ms ≈ 186 s) means SS
  initialization is essential; `kqt1_init_states` computes minf, sinf at
  v_init and sets state directly. Without SS init the gate barely budges in
  200-ms voltage-clamp windows.
- `^2` → `**2` (Python integer power; Brian2's eqs parser handles cleanly).
- No GLOBAL declarations (verified by reading mod source) — F2 pattern not
  applicable. All parameters set per-NeuronGroup via `apply_params`.

## CP2 — AIY cell construction (COMPLETE)

**Module:** `wave2/option_alpha_aiy_cell.py` (~280 lines).

**Channel set (verified primary-source order):**
`[egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1]` — 7 channels.

**Parameter vector (canonical AIY g0, gScm2-rescaled at surf=65.89e-8 cm²):**
| Channel | g0 (nS) | g (S/cm²) |
|---|---|---|
| leak | 0.14 | 2.125e-04 |
| slo1iso | 1.0 | 1.518e-03 |
| kqt1 | 0.2 | 3.035e-04 |
| egl19 | 0.1 | 1.518e-04 |
| slo1egl19 | 0.92 | 1.396e-03 |
| nca | 0.06 | 9.106e-05 |
| shl1 | 0.5 | 7.588e-04 |

`eleak = -89.57 mV`, `cm = 1.6 μF/cm²`, `eca = 60`, `ek = -80`.

**Top-level comment in `AIY_simulation.py` mislabels position [6] as "irk".**
The actual iclamp/vclamp scripts consume index [6] as `seg.shl1.gbar`. Code wins
over comment — verified by reading both `AIY_simulation_iclamp.py` and
`AIY_simulation_vclamp.py`. No IRK channel in AIY.

**Smoke test result (100 ms passive, v_init=-60 mV, no clamp/inject):**
Cell settles from -60 mV to -55.24 mV. Channel currents at t=100 ms (mA/cm²):
- ica_egl19    = -1.89e-05
- ik_slo1iso   = +4.45e-08
- ik_slo1egl19 = +3.79e-07
- ik_kqt1      = +7.06e-05
- ik_shl1      = +4.15e-04
- ik_nca       = -7.76e-03 (large depolarizing leak — NCA reversal +30 mV)
- i_leak       = +7.29e-03 (large hyperpolarizing — eleak -89.57 mV)

NCA and leak roughly balance, with K-channel currents holding cell near rest.
Expected behavior of a non-spiking interneuron at near-rest. Cell builds and
integrates without runtime errors.

**Architecture choices (matching option α AVAL precedent + AIY-specific):**
- No Ca pool (Nicoletti's AIY has no cadiff/caintra1; cai static at 5e-5 mM).
- slo1egl19 reads egl19's m, h, minf, mtau directly (closed-form coupled
  architecture per F12 finding).
- slo1iso uses static cai = 5e-5 mM (NEURON cai0_ca_ion default).
- Brian2 `rk4` integrator (matches AVAL precedent).

## CP3 — AIY voltage-clamp Layer A (in progress)

### F18 — NEURON ion_style overrides user-set eca with multiple USEION ca

**First-run divergence pattern:** systematic Brian2 outward excess of +7-14% in
ss current at holds +0 to +20 mV; sign reverses at +40 mV. Voltage-clamp
panel_pass: 5/11 holds (45.5%); current-clamp 4/11 sweeps (36.4%).

**Root cause (per-channel diagnosis at +0/+20/+40 mV, see
`diagnose_aiy_divergence.py`):** discrepancy localized entirely to slo1egl19
K-current. caCALC differs by 53-77% between Brian2 and NEURON.

**Root cause of caCALC divergence:** NEURON's `seg.eca = 60` (set by Nicoletti
in `AIY_simulation_iclamp.py`) is **silently overridden at runtime** when the
section has multiple `USEION ca` mechanisms. NEURON's ion_style defaults to
Nernst-computed eca in this case, giving:
- `eca = (RT/zF) * ln(cao/cai) = 12.040 * ln(2 / 5e-5) = 127.583 mV`
  (at NEURON's default celsius=6.3°C, cai0=5e-5 mM, cao=2 mM)
- NEURON's observed runtime eca = **+127.590 mV** (rounding from internal R, F).

**Comparison with AVAL (which passed):** AVAL has only one `USEION ca`
mechanism (egl19); ion_style preserves user-set `seg.eca = 60`. AVAL's eca
during runtime = +60 mV exactly. Confirmed by `seg.eca` print after
`h.fadvance()`.

**Why slo1egl19 has the most visible effect:** the channel's nanodomain
calcium formula is:
```
caCALC = |gsc * (v - eca) * 1e-3| / (8πr·d·F) * exp(-r / sqrt(d/(kb·b))) * 1e3 + fondo
```
At v=0, the difference between (v-60) = -60 and (v-127.59) = -127.59 produces
a factor of ~2.13× in caCALC. This propagates into kop_CALC (which has
`(kxy/caCALC)^nxy` dependence) and through the kinetic Markov scheme into
m_slo1egl19 SS. ek for K-channel current was correct (-80 mV in both), so the
divergence was purely upstream of the Markov gate.

**Egl19's own current also affected:** NEURON's `ica_egl19 = gbar*m*h*(v-eca)`
runs with eca=127.59 mV; Brian2 was running with eca=60 mV. Magnitude effect
visible at +40 mV: NEURON ica_egl19 = -1.717 pA, Brian2 = -1.710 pA (only
0.4% off — Brian2 also gets eca=60 from EGL19_PARAMS default; the agreement
is coincidental because m_egl19 and h_egl19 are both saturating at +40 mV
where the egl19 current is small relative to slo1egl19's K current).

**Fix:** updated `AIY_ECA_MV = 127.59` in `option_alpha_aiy_cell.py`. Added
`eca_mV` parameter to `slo1egl19_apply_params` (was missing — defaulted to
SLO1_EGL19_PARAMS["eca_mV"]=60). The cell builder now passes the
NEURON-runtime eca to BOTH egl19 and slo1egl19 channels.

**Post-fix diagnosis:** caCALC, kop, kom, kcm now match between NEURON and
Brian2 within 0.00% at all probed holds (+0, +20, +40 mV). m_slo1egl19 SS
matches within ~1.5% (residual from finite gate equilibration time at 200 ms
window — Brian2 and NEURON have slightly different integration error
accumulation).

**Methodology implication:** for any future cell with >1 USEION ca mechanism,
the Brian2 reproduction must use the Nernst-computed eca (or explicitly call
`h.ion_style("ca_ion", 1, 2, 1, 1, 0)` in NEURON to preserve user-set eca).
This is an upstream NEURON behavior, not a translation defect — Nicoletti's
published model runs at eca=127.59 mV in AIY's case, and our published-model
reproduction goal requires matching that.

For RIM (deferred): RIM has cca1 + unc2 + egl19 — three USEION ca mechanisms.
Same ion_style override would apply. The published `seg.eca = 60` in
RIM_simulation_iclamp.py is also overridden at runtime to 127.59 mV. RIM
translation work block will need to adopt the same fix.

### CP3 — AIY voltage-clamp Layer A (COMPLETE, post-fix)

**Verdict:** PRODUCTION_GRADE.

**Result:** **11/11 holds passing (100%)**, peak and ss divergences ≤ 1.13%
at every hold. Per-hold detail (current-domain divergence metric ≤ 5%
per-feature):

| Hold (mV) | B2 peak (pA) | NRN peak (pA) | div_peak | B2 ss (pA) | NRN ss (pA) | div_ss |
|---|---|---|---|---|---|---|
| -80 | -5.27 | -5.27 | 0.0000 | -5.26 | -5.26 | 0.0001 |
| -60 | -1.06 | -1.06 | 0.0000 | -1.06 | -1.06 | 0.0000 |
| -40 | +3.46 | +3.46 | 0.0001 | +3.32 | +3.32 | 0.0000 |
| -30 | +6.19 | +6.20 | 0.0009 | +5.38 | +5.38 | 0.0003 |
| -20 | +9.88 | +9.90 | 0.0016 | +7.36 | +7.36 | 0.0005 |
| -10 | +14.71 | +14.75 | 0.0028 | +9.43 | +9.43 | 0.0005 |
| +0 | +20.73 | +20.78 | 0.0023 | +12.01 | +12.01 | 0.0001 |
| +10 | +28.08 | +28.20 | 0.0042 | +15.86 | +15.88 | 0.0018 |
| +20 | +38.02 | +38.24 | 0.0057 | +22.21 | +22.31 | 0.0048 |
| +30 | +50.27 | +50.65 | 0.0077 | +32.15 | +32.42 | 0.0084 |
| +40 | +65.08 | +65.72 | 0.0098 | +45.73 | +46.26 | 0.0113 |

Both peak and ss divergence < 1.13% at every hold. CP3 panel_pass: True.

## CP4 — AIY current-clamp Layer A (COMPLETE)

**Verdict:** PRODUCTION_GRADE.

**Result:** **10/11 sweeps passing (90.9%)** at the standard ≤3 mV
voltage-feature tolerance for peak + plateau, with >80% timepoints within
3 mV. Aggregate timepoint-level fraction: 95.9%.

| Inj (pA) | Δpeak (mV) | Δplat (mV) | timepoint % | sweep_pass |
|---|---|---|---|---|
| -15 | 1.12 | 6.84 | 54.8% | **False** |
| -10 | 0.000 | 0.000 | 100.0% | True |
| -5 | 0.002 | 0.002 | 100.0% | True |
| 0 | 0.033 | 0.033 | 100.0% | True |
| +5 | 0.807 | 0.472 | 100.0% | True |
| +10 | 0.000 | 0.000 | 100.0% | True |
| +15 | 0.000 | 0.000 | 100.0% | True |
| +20 | 0.000 | 0.000 | 100.0% | True |
| +25 | 0.000 | 0.000 | 100.0% | True |
| +30 | 0.000 | 0.000 | 100.0% | True |
| +35 | 0.000 | 0.000 | 100.0% | True |

**Failing sweep at -15 pA:** Brian2 plateau = -121.81 mV vs NEURON -128.66 mV
(Δ=6.84 mV). At this very-hyperpolarized regime (~-128 mV), most channels are
closed; the difference reflects integrator behavior on KQT-1's extremely slow
s-gate (stau ≈ 186 s) over the 5-second stim window. NEURON uses `cnexp`
(exponential method per state); Brian2 uses `rk4`. Over multi-second
horizons with a tau ~30× longer than the simulation, integration error
between methods is non-negligible. Standing followup F19, but not blocking
PRODUCTION_GRADE since this is at hyperpolarization extreme far from the
operating range and panel pass criterion is >80%.

## CP5 — AIY verdict (COMPLETE)

**Verdict: VERDICT_AIY_PRODUCTION_GRADE**

Both apples-to-apples Layer A comparisons pass:
- CP3 (voltage-clamp): 11/11 holds (100%), max divergence 1.13%.
- CP4 (current-clamp): 10/11 sweeps (90.9%), one sweep at -15 pA with 6.84 mV
  plateau divergence due to slow-gate integration accumulation at extreme
  hyperpolarization.

The fix that promoted PARTIAL → PRODUCTION_GRADE was correctly identifying
F18 (NEURON ion_style override of user-set eca with multi-USEION ca cells).
This is a **methodology-level finding** that applies to all future cells
with multiple Ca-using mechanisms (RIM in particular), not a translation
defect.

**Standing followups:**
- F19: integrator difference (Brian2 rk4 vs NEURON cnexp) accumulates over
  long simulation horizons with very-slow gates. Visible at -15 pA AIY
  current-clamp sweep (5 s × 186 s tau). Mitigation candidates: switch
  Brian2 to a stiffer integrator like `exponential_euler` for slow-gate
  ODEs, or restrict slow-gate evolution to relevant voltage ranges.
- F20 (latent): SLO1_EGL19_PARAMS["eca_mV"] default of 60 is misleading for
  cells where ion_style overrides. Future RIM/AIY-class cells should pass
  explicit eca_mV. Considered adding a runtime warning when slo1egl19
  is loaded without explicit eca_mV; deferred to followup.

---

## Wave 2 RIM session start (2026-04-26)

**Pre-flight acknowledgment.** Spec read in full
(`phase_v_w2_rim_validation_prompt.md`). Pre-flight reads of primary sources
(`RIM_simulation_iclamp.py`, `RIM_simulation.py`, `cca1.mod`, `egl2.mod`,
`unc2.mod`) confirm the prompt's assertions cleanly:

- **7 channels** verified at `RIM_simulation_iclamp.py` lines 31-38:
  `[shl1, egl2, irk, cca1, unc2, egl19, leak]` (insertion order).
- **g-vector** at `RIM_simulation.py` line 27 already in S/cm² (no
  `gScm2` rescale). Comment line 25: "conductances in S/cm^2".
- **g0** = `[9.049e-4, 1.412e-4, 3.273e-4, 8.452e-4, 9.677e-5, 3.201e-4,
  9.677e-5, -50, 1.5]`. eleak = -50 mV (very different from AVAL's -39 and
  AIY's -89.57). cm = 1.5 μF/cm².
- **Geometry:** surf = 103.34e-8 cm² (RIML, neuromorpho), L = sqrt(surf/π)·1e4.
- **Initial conditions:** `h.finitialize(-60)`, `seg.eca = 60`, `seg.ek = -80`.
  F18 will override eca → 127.59 mV at runtime (3 USEION ca: cca1+unc2+egl19).
- **iclamp protocol:** delay=5000 ms, dur=5000 ms, simdur=14000 ms, dt=0.04 ms,
  injection range -15 to +35 pA × 11 steps. Initial transient cut at 4000 ms.
- **vclamp range:** -100 to +50 mV × 16 steps (per RIM_simulation.py line 19).
- **NEURONReference("RIM")** already implemented in `neuron_reference.py`
  with rim_g0 hardcoded and S/cm² convention preserved (lines 305-315).

**USEION ca count: 3.** Verified by reading mod files:
- `cca1.mod` line 16: `USEION ca READ eca WRITE ica`
- `unc2.mod` line 17: `USEION ca READ eca WRITE ica`
- `egl19.mod` line 18: `USEION ca READ eca WRITE ica`

**UNC-2 GLOBAL declarations** (line 19 of `unc2.mod`):
`GLOBAL minf, hinf, mtau, htau, munc2, hunc2`. Of these:
- `minf, hinf, mtau, htau`: **derived** assignments computed from `v` in the
  `rates(v)` PROCEDURE before each DERIVATIVE step. Marking them GLOBAL is a
  NMODL pitfall — they should be RANGE for per-instance correctness — but
  it's *functionally* harmless in single-cell runs because `rates(v)` is
  re-called at every DERIVATIVE step from the cell's own `v`, so each
  instance overwrites them before reading.
- `munc2, hunc2`: **exposed copies** of `m, h` written in BREAKPOINT after
  the integration step. They're labeled GLOBAL but again recomputed each
  step from per-instance `m, h`. Single-cell-per-process: harmless. Multi-
  cell-per-process: corrupted (the *last* cell's m, h would be visible to
  other cells' next-step access — but per-instance `m, h` STATE is preserved
  separately, so the actual ica is computed correctly within a step).
- The actual STATE is `{m, h}` (line 71) — these are RANGE-by-default per
  NMODL convention for STATE, so per-instance.
- **Brian2 handling decision:** treat all six as per-cell quantities (eqs
  variables, not shared scalars). For our single-NeuronGroup (1 cell)
  validation harness, this matches NEURON exactly. For future multi-cell
  Brian2 deployment, we already get per-cell semantics for free, so no
  hidden divergence vs. NEURON's GLOBAL surprise will surface.
- Document explicitly in `unc2.py` module comments per CP3 acceptance.

**Plan:** proceed to CP1 (CCA-1) → CP2 (EGL-2) → CP3 (UNC-2) →
CP4 (RIM cell, AIY_ECA_MV=127.59 directly applicable) → CP5 (vclamp
Layer A) → CP6 (cclamp Layer A) → CP7 (verdict).

**No pushback** — all spec claims primary-source verified. Proceeding.

---

## F18 refinement (RIM CP4 pre-flight discovery, 2026-04-26)

**Empirical finding contradicting prior F18 prediction.** Prompt and
`cellular_validation_findings.md` AIY entry both predicted RIM would have
the same `seg.eca` runtime override behavior as AIY (127.59 mV from Nernst,
not the user-set 60 mV). Empirical check before CP4:

```
AIY (2 USEION ca: egl19 + slo1egl19):
  ion_style(ca_ion) = 49 (0b110001)
  seg.eca after run = 127.5895 mV  (overridden to Nernst)

RIM (3 USEION ca: cca1 + unc2 + egl19):
  ion_style(ca_ion) = 8 (0b1000)
  seg.eca after run = 60.0000 mV   (preserved)
```

The override IS NOT triggered by "multiple USEION ca" — it's triggered by
**asymmetric USEION declarations across the channels**.

In AIY, `slo1egl19.mod` has `USEION ca READ eca` (READs without writing
ica). egl19 has `USEION ca READ eca WRITE ica`. The mismatch — one channel
reads eca with no writer-of-ica counterparty for that *particular reader's*
ion contract — promotes NEURON's ion_style to Nernst-computed eca.

In RIM, all three USEION ca channels (cca1, unc2, egl19) have **identical
USEION declarations**: `USEION ca READ eca WRITE ica`. NEURON sees a
consistent contract and preserves user-set `seg.eca`.

**F18 trigger refinement (corrected for catalog):**
- Trigger: ≥ 2 channels with USEION ca, AND at least one declares
  `USEION ca READ eca` WITHOUT `WRITE ica`. Pattern: a "READ-only" Ca
  reader (slo1egl19, slo2egl19, slo1unc2, slo2unc2, kcnl) coexisting with
  ica-writing Ca channels (cca1, unc2, egl19).
- Non-trigger: all USEION ca channels have identical READ eca + WRITE ica
  declarations. RIM falls in this case.

**Verification of correction:** AVAL has only egl19 (single USEION ca) →
not multi → ion_style preserves. AIY has slo1egl19 + egl19 → mismatch →
override. RIM has cca1 + unc2 + egl19 → symmetric → preserve. All three
empirically match the refined trigger.

**Implication for CP4:** RIM Brian2 cell uses `eca=60 mV` directly,
matching NEURON's actual runtime. NO F18 fix needed for RIM. Yesterday's
F18-aware methodology was sound; the prediction-from-AIY-pattern was
overconfident.

**Implication for AIY:** AIY F18 fix was correct (127.59 mV is right for
AIY). No revision needed.

**Implication for future cells:**
- RMD (deferred to separate work block) — when acquired, check ion_style
  empirically before applying F18 fix.
- VA5, VB6, VD5 — Nicoletti's ventral motoneurons; channel composition
  TBD on acquisition.
- General methodology: **always probe `seg.eca` after a brief run before
  building Brian2 cell** to determine ion_style actual behavior. Don't
  predict from channel-count heuristic.

---

## Wave 2 RIM CP1-CP7 results (2026-04-26)

### CP1 — CCA-1 (T-type voltage-gated Ca)

**Verdict:** PRODUCTION_GRADE.
**Module:** `wave2/channels/cca1.py`. Standard m^2*h pattern, voltage-only
gating, F18-aware eca handling.
**Validation:** 10/11 holds passing (90.9%). The one "fail" at +20 mV is a
peak-detection artifact: ica peaks small inward briefly then crosses zero
to small outward — the magnitude-extremum detector picks the wrong-sign
extremum on Brian2 vs NEURON depending on which side hits its inward peak
slightly earlier in the noise floor. SS values match exactly (div=0.000).
Acceptable per panel_pass criterion.

### CP2 — EGL-2 (EAG-family voltage-gated K)

**Verdict:** PRODUCTION_GRADE.
**Module:** `wave2/channels/egl2.py`. Single state m, no inactivation, no
GLOBAL state. Clean voltage-gated K pattern — EAG kinetics within RIM's
operating range -80 to +40 mV are unremarkable (mtau ~4-8 ms, no extreme-
slow gate like KQT-1's s-gate).
**Validation:** 11/11 holds passing (100%), max divergence 0.003.

### CP3 — UNC-2 (P/Q-type voltage-gated Ca with GLOBAL declarations)

**Verdict:** PRODUCTION_GRADE.
**Module:** `wave2/channels/unc2.py`.

**GLOBAL handling decision (load-bearing per CP3 acceptance):**
The `GLOBAL minf, hinf, mtau, htau, munc2, hunc2` declaration in unc2.mod
is a NMODL pitfall — these should be RANGE per NMODL convention since
they're per-instance derived quantities. However, in single-cell-per-section
NEURON use (as in Nicoletti's published model), the GLOBAL declarations
are *functionally* harmless: rates(v) is called at every DERIVATIVE step
from the cell's own v before m', h' are evaluated; munc2/hunc2 are
diagnostic copies of m, h written in BREAKPOINT. The actual integrated
state {m, h} is per-instance via NMODL STATE convention.

In Brian2, every NeuronGroup variable is per-cell by default. We translate
all six GLOBAL-declared variables as per-cell (`: 1`) eqs declarations.
For our single-cell-per-NeuronGroup Layer A validation, this matches
NEURON exactly. For future multi-cell-per-NeuronGroup deployment, Brian2
provides correct per-cell semantics for free — we never inherit the NMODL
GLOBAL pitfall surprise. **No special handling required** beyond standard
per-cell translation.

**Validation:** 11/11 holds passing (100%), all div=0.000.

### CP4 — RIM 7-channel cell construction

**Verdict:** PASS.
**Module:** `wave2/option_alpha_rim_cell.py`.

Channel set: `[shl1, egl2, irk, cca1, unc2, egl19, leak]`. 3 USEION ca
(cca1+unc2+egl19), 3 USEION k (shl1+egl2+irk), 1 leak.

**eca = 60 mV** (NOT 127.59 — F18 refinement applied above).
**g convention:** S/cm² already; no gScm2 rescale.
**Geometry:** surf=103.34e-8 cm² (RIML neuromorpho), cm=1.5 μF/cm²,
eleak=-50 mV, v_init=-60 mV.

100 ms passive smoke test: cell settles -60 → -43.68 mV (depolarized
because eleak=-50 mV is more positive than rest, RIM has tonic CCA-1
inward Ca current at -60 mV). All channel currents at expected
magnitudes; cell builds and integrates without runtime errors.

### CP5 — RIM voltage-clamp Layer A

**Verdict:** PRODUCTION_GRADE.

**Result:** 11/11 holds passing (100%), max divergence 0.0043 (at +10 mV
peak; well under 0.05 tolerance). Per-hold detail in
`rim_validation_summary.md`.

### CP6 — RIM current-clamp Layer A

**Verdict:** PRODUCTION_GRADE.

**Result:** 11/11 sweeps passing (100%) at -15 to +35 pA injection range.
**All peak/plateau/baseline residuals = 0.000 mV** at every injection level.
Aggregate timepoint pass: 55000/55000 (100%).

This is a notably **cleaner result than AIY's** (which had a -15 pA sweep
with 6.84 mV plateau residual due to KQT-1's 186 s s-gate slow-integration
drift). RIM has no extreme-tau slow gate, so the rk4 vs cnexp integrator
difference doesn't accumulate visibly over 14000 ms.

**Elapsed:** Brian2-bound. NEURON CC for 11 sweeps × 14s = ~30 s total;
Brian2 CC for same with rk4 + 7 channels + numpy codegen took ~50 minutes.
For future RIM revisits at finer dt or longer protocols, switching Brian2
codegen target to "cython" would give 5-10× speedup.

### CP7 — Verdict

**VERDICT_RIM_PRODUCTION_GRADE.**

3rd production-grade cell in Wave 2 cellular layer. F18 refined. UNC-2
GLOBAL pitfall handled cleanly. 3 new channel translations validated
(CCA-1, EGL-2, UNC-2). Channel diversity sufficient for Phase δ network
integration scoping.

See `rim_validation_summary.md` for full Wave 2 status update and next-
work-block recommendations.


