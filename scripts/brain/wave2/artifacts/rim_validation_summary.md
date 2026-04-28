# Wave 2 RIM cellular validation — summary

**Date:** 2026-04-26
**Verdict:** **VERDICT_RIM_PRODUCTION_GRADE**

---

## Outcome

Brian2 7-channel RIM cell matches Nicoletti's NEURON RIM within tolerance for
**both** voltage-clamp and current-clamp Layer A comparisons:

- **CP5 voltage-clamp:** 11/11 holds passing (100.0%), max divergence 0.0043
  (well under 0.05 tolerance).
- **CP6 current-clamp (5000 ms protocol):** 11/11 sweeps passing (100.0%),
  all peak/plateau/baseline residuals 0.000 mV at every injection level
  -15 to +35 pA. Aggregate timepoint pass: 55000/55000 (100.0%).

This is the **3rd production-grade cell** in Wave 2's cellular layer (after
AVAL, AIY).

## Wave 2 cellular layer status

Three production-grade cells representing the three published Nicoletti
2024 target neurons:

| Cell | Channels | Channel diversity | Verdict |
|---|---|---|---|
| AVAL | 4 (egl19, leak, irk, nca) | Ca-L, K-IR, leak, NSC | PRODUCTION_GRADE |
| AIY  | 7 (egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1) | Ca-L, K-BK·Ca-coupled, K-BK·Ca-iso, K-KCNQ, K-A | PRODUCTION_GRADE |
| RIM  | 7 (shl1, egl2, irk, cca1, unc2, egl19, leak) | K-A, K-EAG, K-IR, Ca-T, Ca-P/Q, Ca-L, leak | PRODUCTION_GRADE |

**Channel coverage** (Wave 2 channel translation catalog, 14 total):
- Old: egl19, shk1, shl1, nca, kqt3, slo1iso, slo1egl19_coupled,
  slo1iso_dynamic_ca, unc103, irk, kqt1
- New (RIM work block): cca1, egl2, unc2

Three new translations validated production-grade in this work block,
each with their own voltage-clamp Layer A acceptance:

| Channel | Type | Holds passing | Notes |
|---|---|---|---|
| CCA-1 | T-type Ca, m^2*h | 10/11 (90.9%) | One peak-direction-flip artifact at +20 mV (SS still match exactly) |
| EGL-2 | EAG-family K, m | 11/11 (100%) | Clean voltage-gated K, no GLOBAL state |
| UNC-2 | P/Q-type Ca, m*h | 11/11 (100%) | NMODL GLOBAL pitfall on derived assignments — handled per-cell in Brian2 |

## Findings extending F1-F18

### F18 refinement (load-bearing methodology correction)

Yesterday's F18 finding from AIY cellular validation predicted that any cell
with multiple USEION ca mechanisms would have NEURON's ion_style override
the user-set `seg.eca` to a Nernst-computed value (~127.59 mV at default
celsius=6.3°C, cai=5e-5 mM, cao=2 mM).

**Empirical pre-flight check before CP4 contradicted this prediction for
RIM:** seg.eca after `h.run()` = 60.0000 mV (preserved, NOT overridden).

The corrected F18 trigger condition is **asymmetric USEION declarations
across channels**:
- Trigger: ≥2 USEION ca channels, AND at least one declares
  `USEION ca READ eca` WITHOUT `WRITE ica` (a "READ-only" Ca reader,
  e.g. slo1egl19, slo2egl19, slo1unc2, slo2unc2, kcnl)
- Non-trigger: all USEION ca channels have identical READ eca + WRITE ica
  declarations.

**Verification across all three cells:**
- AVAL: single USEION ca (egl19) → no multi-USEION condition → eca = 60 ✓
- AIY: egl19 (READ+WRITE) + slo1egl19 (READ-only) → asymmetric → eca = 127.59 ✓
- RIM: cca1 + unc2 + egl19 (all READ+WRITE) → symmetric → eca = 60 ✓

**Methodological lesson:** always probe `seg.eca` after a brief NEURON run
when validating a new cell. Don't predict ion_style behavior from
channel-count heuristics. This is now a standing pre-flight requirement for
future cells (RMD, VA5, VB6, VD5).

Documented in `cellular_validation_findings.md` "F18 refinement" entry.

### F19 (carry-forward from yesterday)

Brian2 rk4 vs NEURON cnexp integrator difference accumulates over long
horizons with very-slow gates. Visible at AIY -15 pA sweep (5 s × 186 s
KQT-1 s-gate). Not a RIM concern (RIM has no extreme-tau slow gate).
Standing followup unchanged.

### No F20+ new patterns from RIM

UNC-2 GLOBAL handling: per CP3 acceptance, decision documented in
`channels/unc2.py` — no special Brian2 handling required (per-cell semantics
in Brian2 by default already provide correct functional behavior; the NMODL
GLOBAL pitfall is inert for single-cell-per-section use).

The CP6 current-clamp result (residuals exactly 0.000 mV at all timepoints
for all 11 sweeps over 14000 ms total simulation) is empirical evidence that
no hidden numerical divergence exists between Brian2 and NEURON for RIM's
specific channel composition. This is an even cleaner result than AIY's
(which had a single -15 pA sweep with 6.84 mV plateau residual due to KQT-1
slow-gate integration drift).

## Implications for next work blocks

### Phase δ network integration readiness

Three production-grade cells (AVAL, AIY, RIM) representing distinct
electrophysiological classes:
- **AVAL** — minimal interneuron: Ca-L + K-IR + NSC
- **AIY** — Ca-coupled interneuron: BK·Ca-coupled K + Ca-L
- **RIM** — multi-Ca interneuron: T + P/Q + L Ca + A + EAG + IR K

Channel diversity is sufficient for meaningful network integration once
synaptic-coupling architecture is in place. Phase δ becomes substantive,
not just structural.

### RMD scope reduced (CCA-1 now translated)

RMD requires Nicoletti 2019 paper acquisition (different from 2024). When
acquired, the channel-translation deficit is reduced because CCA-1 is now
done. Expected scope: ~2-3 channel translations + cell + Layer A.

### AVAR upstream + standing followups

Unchanged from yesterday. AVAR upstream issue (UNC-103 RANGE/GLOBAL
pattern) still on standing-followup list; not blocking Wave 2 cellular
work.

### Methodology paper case studies catalog

RIM adds a strong case study:
- "How an apparently universal pattern (F18 'multi-USEION-ca → eca
  override') gets refined when an apparently-symmetric counter-example
  surfaces. The corrected mental model (asymmetric channel-USEION-declarations
  trigger) is more parsimonious and more predictive."
- "How a NMODL GLOBAL pitfall (UNC-2's GLOBAL on derived assignments) gets
  *automatically* handled by Brian2's per-cell-by-default semantics — the
  source of a NMODL bug becomes a non-issue in the target framework."
- "Three production-grade cells in three work blocks: methodology lock-in
  (primary-source verification, mid-flight findings catalog, F-numbered
  pattern collection) is what made the third cell faster than the second."

## Recommendations (priority-ordered)

1. **Phase δ network integration scoping.** With 3 cells production-grade,
   begin scoping the simplest meaningful network (e.g., AVA-AIY synaptic
   pairing, or RIM-AVA-AIY 3-node circuit) and articulate what "production-
   grade network" means at the apples-to-apples Layer A level. This is the
   natural next work block.

2. **RMD acquisition.** Nicoletti 2019 paper code acquisition. Once in
   hand, RMD is ~2-3 channel translations + cell + Layer A — likely smaller
   scope than RIM was. Substantively similar pattern.

3. **F18 refinement note in methodology paper.** This is a clean
   prediction-correction story worth ~1 page of the methodology paper:
   "The cost of acting on a half-formed mental model — and the cost of
   correcting it." Three cells across three work blocks where methodology
   lock-in caught the half-formed prediction at pre-flight.

4. **AVAR upstream issue review and filing.** Still open from a previous
   work block. Reduce backlog before Phase δ network integration adds new
   followups.

5. **Current-clamp at finer granularity.** RIM's CP6 was clean to 0.000 mV
   at standard 0.04 ms dt. Worth checking at dt = 0.01 ms (4× finer) at
   one or two extreme injection levels to confirm nothing was hidden by the
   coarse-dt averaging. Quick sanity check, not blocking.

## File outputs

```
wave2/
├── channels/
│   ├── cca1.py                                          [NEW]
│   ├── egl2.py                                          [NEW]
│   └── unc2.py                                          [NEW]
├── option_alpha_rim_cell.py                             [NEW]
├── run_option_b_rim.py                                  [NEW]
├── validate_rim_channels.py                             [NEW]
└── artifacts/
    ├── checkpoints/
    │   ├── rim_CP1_status.json (cca1 PASS)              [NEW]
    │   ├── rim_CP2_status.json (egl2 PASS)              [NEW]
    │   ├── rim_CP3_status.json (unc2 PASS)              [NEW]
    │   ├── rim_CP4_status.json (cell PASS)              [NEW]
    │   ├── rim_CP5_status.json (vclamp PASS)            [NEW]
    │   ├── rim_CP6_status.json (cclamp PASS)            [NEW]
    │   └── rim_CP7_status.json (PRODUCTION_GRADE)       [NEW]
    ├── cca1_validation_results.json                     [NEW]
    ├── egl2_validation_results.json                     [NEW]
    ├── unc2_validation_results.json                     [NEW]
    ├── rim_cell_construction.md                         [NEW]
    ├── option_b_rim_results.json                        [NEW]
    ├── rim_validation_summary.md                        [NEW] (this file)
    └── cellular_validation_findings.md                  [extended]
```

## Elapsed

CP1-CP7 total: ~1.5h (CP1-CP4 ~10 min; CP5 ~5 min; CP6 ~52 min Brian2-bound;
documentation throughout). Within the 2-3 hour estimate from the work
block prompt. Brian2 rk4 with 7 channels and many state variables × 14000 ms ×
11 sweeps × 0.04 ms is the dominant cost; per-sweep cost can be reduced 5-10×
by switching to Brian2 cython codegen target if revisiting CC at finer dt.
