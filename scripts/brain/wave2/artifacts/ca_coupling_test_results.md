# Ca-coupling integration test — Phase F 2b on Ca-coupled AVA cell

## Verdict: **VERDICT_CA_COUPLING_INSUFFICIENT** (robust)

**Date:** 2026-04-26 → 2026-04-27 (overnight)
**Trigger:** density-sensitivity sweep (`density_sensitivity_analysis.md`)
confirmed `VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS` for Phase F's static-cai
cell. The load-bearing finding from that sweep was that terminator (SLO-1)
scaling has near-zero leverage on phenotype, because SLO-1 isolated reads a
*static* `cai = 5e-5 mM` (per F12) and SLO-1+EGL-19 coupled uses a
deterministic `calcium(V)` (per F13). No Ca-feedback loop exists in the
single-compartment essential set.

This work block tested the cheaper architectural extension *before* triggering
the morphology fork: add the dynamic `caintra1` pool, couple EGL-19's I_Ca to
[Ca]_i, and let SLO-1 isolated read the *dynamic* [Ca]_i state.

**The hypothesis is empirically refuted.** The Ca-coupling loop, even when
fully engaged at unphysiologically large `fca`, *shortens* plateau duration
rather than extending it — because SLO-1 isolated is a hyperpolarizing
terminator, and the loop is therefore negative feedback for V. The morphology
fork is now triggered with a stronger evidentiary base.

---

## Architecture decisions (locked in before runs)

1. **Pool: caintra1, not cadiff.** Nicoletti pairs caintra1 with slo1iso
   conceptually (when she pairs them at all); cadiff is the Yale Purkinje
   adaptation used by VA5. Geometry scaled to AVA: vol=129.6e-12 cm³,
   surf=1123.84e-8 cm². The empirical effective coefficient
   `caintra1_coef_in_eff` was rescaled from AIY calibration linearly with
   surf/vol per the formula structure (mirrors `calcium_pool.py`'s logic).

2. **SLO-1 isolated → dynamic [Ca]_i** via a new module
   `channels/slo1_iso_dynamic_ca.py`. Eqs identical to the static variant
   except `cai_mM : 1` is *not* declared as a parameter — it is supplied as
   a state variable by the pool subsystem. The original `slo1_iso.py` is
   preserved so prior validations and the published Phase F 2b run continue
   to behave identically.

3. **SLO-1+EGL-19 coupled keeps closed-form `calcium(V)` — Option A.**
   Per the work-block spec, Option A (faithful to Nicoletti) was chosen over
   Option B (replace `calcium(V)` with bulk dynamic [Ca]_i). Rationale:
   (a) Nicoletti's coupled variant encodes *nanodomain* Ca, not bulk Ca;
   (b) replacing it would change two things at once and confound the test;
   (c) the F12 mechanism diagnosis specifically named SLO-1 *isolated*'s
   missing Ca-feedback as the candidate ingredient. Option A surgically
   tests just that mechanism.

4. **Conductances unchanged from Phase F 2b baseline.** `g_egl19`, `g_leak`,
   `g_nca` at Nicoletti AVAL g0; `g_slo1iso`, `g_slo1egl19`, `g_shl1`,
   `g_shk1`, `g_kqt3` at Phase F 2b baseline. Any phenotype change is
   therefore attributable to the Ca-coupling change.

5. **Ca-pool parameters at Nicoletti `caintra1.mod` defaults.** fca=0.001,
   tca=50 ms, ca_eq=0.05e-6 M (NEURON numeric 5e-8 in raw NMODL units).

### Architecture decision documented during execution

6. **Unit-conversion bridge between caintra1 and slo1iso (NEW finding).**
   `caintra1.mod` declares STATE `caintra` with parameter `ca_eq=0.05e-6 (M)`
   stored as raw 5e-8. `slo1iso.mod` declares `cai (mM)` and reads it as 5e-5
   mM at NEURON's default. The two cell-state variables differ by a 1000×
   numerical factor: caintra is M-scale (5e-8 raw = 50 nM = 5e-5 mM), while
   slo1iso's cai is mM-scale (5e-5 raw = 50 nM = 5e-5 mM). To wire them
   together correctly in Brian2, the cell builder defines:

       cai_mM = caintra_raw * 1000.0

   This conversion is **the load-bearing fix that made the integration
   physically meaningful.** A first-pass build that used `cai_mM` directly
   from the pool (without conversion) gave SLO-1 isolated a [Ca]_i of
   0.05 nM (vs the channel's effective Ca-affinity range ~50 μM), keeping
   the channel essentially closed regardless of dynamics. After fixing the
   conversion, the channel's m gate at rest matches the static-cai 50-nM
   case exactly, and the loop is properly enabled.

7. **fca-scaling of the pool's effective coefficient (NEW finding).**
   `calcium_pool.caintra1_eqs()` computes `coef_in_eff` from an empirical
   AIY-geometry calibration that absorbs `fca = 0.001`. It does NOT rescale
   when callers pass a different `fca`. Per the NMODL formula, rate_inward is
   linear in fca, so the cell builder applies an explicit scaling
   (`coef_in_eff *= fca/0.001`) to keep `calcium_pool.py` untouched while
   honoring the `fca` axis in the secondary sweep. Without this fix, the
   sweep over fca produces byte-identical phenotypes — masking the real
   sensitivity. (Documented and applied in `ca_coupled_cell.py`.)

---

## Primary run — single coupled cell at Nicoletti defaults

**Protocol** (identical to Phase F 2b for direct comparability):

- 200 ms settle at I=0
- 100 ms × 50 pA injection
- 1500 ms post-stim recovery
- Brian2 RK4, dt=0.025 ms

**Phenotype:**

| Metric | Ca-coupled (this run) | Phase F 2b (static cai) | Mellem target |
|---|---|---|---|
| Plateau amplitude (mV) | **46.85** | 46.85 | [15, 25] |
| Plateau duration (ms) | **21.4** | 21.4 | [400, 800] |
| τ_release (ms) | 240.6 | — | — |
| Release τ ratio | 3.74 | 3.74 | <0.6 (active term.) |
| Architectural signature | no_termination | no_termination | active_termination |
| amp_pass | ✗ | ✗ | — |
| dur_pass | ✗ | ✗ | — |
| **arch_pass** | **✗** | **✗** | — |

**Ca-pool diagnostics:**

| Metric | Value |
|---|---|
| [Ca]_i baseline (mM) | 5.00e-5  (= 50 nM) |
| [Ca]_i peak during stim (mM) | 5.11e-5  (= 51 nM) |
| [Ca]_i fold change peak/base | **1.02×** |
| [Ca]_i decay τ (ms) | 50.87 |

The pool decays with the prescribed `tca = 50 ms`, confirming the differential
equation integrates correctly. But `[Ca]_i barely rises` (1.02× over baseline)
because EGL-19's I_Ca is small in absolute terms: peak `ica_egl19_mAcm2 ≈
-9.3e-5 mA/cm²`, total Ca current ~1 pA at AVA's 1124e-8 cm² surface. With
fca=0.001, the influx coefficient is too small to overcome efflux at this
ica magnitude.

**At Nicoletti's default parameters, the Ca-coupling loop is *thermodynamically
disengaged*** — [Ca]_i variation is negligible relative to SLO-1 isolated's
Ca affinity (kxy=55.7 μM, kyx=34.3 μM). The loop is structurally correct but
gain-locked at near-zero.

---

## Secondary sweep — Ca-coupling sensitivity probe

12-cell grid spanning physiological-near regime (`grid_A`: fca×{1,10},
tca×{1,5}, slo1×{1,4}, 8 cells) and loop-engagement regime (`grid_B`:
fca×{100,1000,10000,100000}, tca×5, slo1×1, 4 cells).

| fca (rel. to default) | tca (ms) | slo1 factor | [Ca]_i peak | amp (mV) | dur (ms) | signature |
|---|---|---|---|---|---|---|
| 1× | 50 | 1.0 | 51 nM | 46.85 | 21.4 | no_term |
| 1× | 50 | 4.0 | 51 nM | 46.84 | 21.4 | no_term |
| 10× | 50 | 1.0 | 61 nM | 46.85 | 21.4 | no_term |
| 10× | 50 | 4.0 | 61 nM | 46.84 | 21.4 | no_term |
| 1× | 250 | 1.0 | 52 nM | 46.85 | 21.4 | no_term |
| 1× | 250 | 4.0 | 52 nM | 46.84 | 21.4 | no_term |
| 10× | 250 | 1.0 | 70 nM | 46.85 | 21.4 | no_term |
| 10× | 250 | 4.0 | 70 nM | 46.84 | 21.4 | no_term |
| 100× | 250 | 1.0 | 248 nM | 46.83 | 21.4 | no_term |
| 1000× | 250 | 1.0 | 2.0 μM | 46.59 | 21.3 | no_term |
| 10000× | 250 | 1.0 | 17 μM | 43.66 | 19.5 | no_term |
| 100000× | 250 | 1.0 | 96 μM | 36.80 | 15.4 | **active_termination** |

**Key observations:**

1. **In the physiologically plausible regime** (fca up to 10× default,
   slo1 up to 4×), plateau duration is invariant at 21.4 ms — the
   Ca-coupling loop has no measurable effect on V dynamics.

2. **At fca=1000× default, [Ca]_i reaches 2 μM** (entering SLO-1's
   activation range), but amplitude only drops 0.26 mV and duration is
   essentially unchanged (21.3 ms).

3. **At fca=10000× default, [Ca]_i reaches 17 μM**, amplitude drops to
   43.66 mV, duration drops to 19.5 ms — duration is moving in the WRONG
   direction relative to Mellem.

4. **At fca=100000× (a complete Ca flood, [Ca]_i = 96 μM, > Nicoletti's
   own kxy of 55.7 μM)**, the architectural signature flips to
   `active_termination` (release τ ratio < 0.6) and duration drops to
   **15.4 ms** — actively *shorter* than the static-cai baseline.

**The monotone trend is the load-bearing finding:** as the Ca-coupling
loop engages, plateau duration *decreases* monotonically. The Ca-coupling
loop is **negative feedback** for V (because SLO-1 is hyperpolarizing
when activated). It cannot extend a depolarized plateau — it can only
terminate it faster.

This contradicts the F12-derived hypothesis. F12 correctly identified that
no Ca-feedback existed; what F12 (and the morning review's read of it)
implicitly assumed was that *adding* Ca-feedback would extend the plateau.
This sweep refutes that assumption with quantitative evidence: across 5
orders of magnitude of fca scaling, the loop is either (a) so weak it has
no effect, or (b) strong enough to engage, in which case it shortens the
plateau, never extends it.

---

## Mechanistic conclusion

The Mellem 2008 600 ms plateau is *not* sustainable in this single-compartment
architecture by Ca-coupling alone. The mechanism Mellem 2008 reports must rely
on something the current channel set + bulk Ca-pool architecture lacks:

- **CICR (Ca-induced Ca release) from internal stores** — adds a *positive*
  Ca-feedback loop (depolarizing Ca-current modulated by store-released Ca).
  Not in Nicoletti's channel set.
- **NMDA-like persistent inward currents** modulated by [Ca]_i — would give
  the depolarizing-current-with-Ca-positive-feedback needed. Not in Nicoletti's
  channel set.
- **Multi-compartment morphology** — could spatially separate a fast
  spike-initiating zone from a slowly inactivating dendritic Ca compartment.
  This is the canonical "morphology fork" path.
- **Dendritic NMDA-Ca plateaus or Ca-channel persistence** — biologically
  established mechanisms for sustained dendritic plateaus in mammalian
  neurons; analogues exist in some C. elegans neuron classes.

None of (a-c) are accessible without architectural escalation. (a) and (b)
require new channel/mechanism translations beyond Nicoletti's set. (c) is the
morphology fork the spec authorized as a fallback.

---

## Verdict and recommendation to morning review

**Primary verdict: `VERDICT_CA_COUPLING_INSUFFICIENT`** at Nicoletti defaults.

**Sweep verdict: same — robustly across 5 orders of magnitude of fca and 4×
SLO-1 conductance.** No combination tested produces duration > 21.4 ms; the
trend with engaged Ca-coupling is *decreasing* duration.

**Recommendation:** the morphology fork is now triggered with stronger
mechanistic justification than after the density-sensitivity sweep alone:

- The density-sensitivity sweep ruled out density-tunability within the
  static-cai architecture.
- This Ca-coupling test rules out the cheapest architectural extension —
  adding the bulk dynamic Ca-pool that F12 named as missing.
- Together they establish that **single-compartment + Nicoletti's channel
  set + bulk Ca-pool, in any combination, cannot reproduce Mellem 2008's
  plateau dynamics**.

The mechanistic story is now clean: Mellem's biology requires either CICR,
multi-compartment morphology, or a channel set with positive Ca-feedback —
none of which are present in the current essential set.

**This is not a request to immediately commit to morphology integration.**
The user's three architectural-response options from `density_sensitivity_
analysis.md` remain open, with the ordering now refined:

1. **~~Add a dynamic Ca-pool~~ — DONE in this work block, refuted.**
2. **Multi-compartment morphology** — the spec's morphology fork. Justified
   by both this and the density sweep. Still ~3-4 weeks of work.
3. **Re-investigate Mellem 2008's protocol** — temperature, drugs, cell prep
   may differ from our 50 pA / 100 ms simulation. Would close the loop on
   "is this even the right comparison".
4. **Add new channel/mechanism translations** — CICR, persistent Na-Ca,
   etc. Not in Nicoletti's published set; would require independent
   validation.

Option 3 is the cheapest probe before committing to (2) or (4). Option 4
would be a substantial detour from the Wave 2 plan.

---

## Architectural insufficiency: tighter framing

Beyond the morphology question, this work block produced **two new general
findings** about the wave2 codebase that should propagate:

- **F16: caintra1 ⇄ slo1iso unit-conversion (×1000) is required.** The Brian2
  state from `calcium_pool.caintra1_eqs()` is in raw NMODL units (M-equivalent,
  5e-8 at rest). Channels reading `cai_mM` expect mM units (5e-5 at rest).
  Cell builders that wire dynamic caintra1 to slo1iso (or any cai-reading
  channel) MUST insert `cai_mM = caintra_raw * 1000`. Documented and
  implemented in `ca_coupled_cell.py`.

- **F17: caintra1 fca-scaling is not in the calibrated `coef_in_eff`.**
  `calcium_pool.caintra1_eqs()` accepts `fca` but doesn't rescale the
  empirical coefficient. Sweep callers must explicitly multiply `coef_in_eff`
  by `fca / 0.001` to honor the `fca` axis. Documented in
  `ca_coupled_cell.py` (could fold into `calcium_pool.py` with a follow-up
  but kept local here to respect scope).

Both are surfaced in `phase_beta_findings.md` candidate territory.

---

## Files produced

```
wave2/
├── ca_coupled_cell.py                          [cell builder w/ caintra1 + dyn-Ca SLO-1]
├── run_ca_coupling_test.py                     [driver: primary + secondary sweep]
├── channels/
│   └── slo1_iso_dynamic_ca.py                  [Ca-dynamic SLO-1 isolated variant]
└── artifacts/
    ├── ca_coupling_test_results.md             [this file]
    └── ca_coupling_test_results.json           [raw + downsampled traj + sweep rows]
```

All existing files (`channels/slo1_iso.py`, `calcium_pool.py`,
`validate_phase_f_gate2.py`, etc.) untouched. Phase F's published 46.8 mV /
21.4 ms reproduces byte-identically when the Ca-coupling cell is built — the
dynamic-pool insertion is non-destructive on the v-trajectory at default fca.

---

*End of Ca-coupling integration test report. Standing by for morning review.*
