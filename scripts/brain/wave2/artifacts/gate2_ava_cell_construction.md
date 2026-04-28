# Gate 2 AVA cell construction — channel densities and rationale

**Date:** 2026-04-26 run #2 invocation 1
**For:** Phase F components 2a and 2b

---

## Component 2a — apples-to-apples [leak + EGL-19 + NCA]

**Cell construction:** Brian2 [leak + EGL-19 + NCA] @ AVAL geometry vs NEURON
[leak + EGL-19 + NCA] @ AVAL geometry. Both built from `custom_spec` so neither
has IRK or UNC-103 — the comparison is apples-to-apples.

**Densities (matching Nicoletti's AVAL g0 vector):**
- surf = 1123.84e-8 cm²
- cm = 0.859551 μF/cm²
- g_leak = 0.150164e-9 / surf = 1.336e-5 S/cm²
- e_leak = -39 mV
- g_egl19 = 0.104385e-9 / surf = 9.288e-6 S/cm²
- g_nca = 0 (Nicoletti's AVAL value)
- eca = 60 mV
- ek = -80 mV

**Rationale:** the spec's Phase F preferred path is "construct a NEURON AVA reference
with only NCA + EGL-19 + leak (mirror Brian2's subset). This requires modifying
Nicoletti's AVA simulation script to suppress IRK/UNC-103 — local patch in wave2/."
We achieve this via NEURONReference's `custom` mode without modifying upstream.

**Result:** PASS, 11/11 holds, max div 0.004.

---

## Component 2b — full 7-channel essential set + Mellem targets

**Cell construction:** Brian2 [leak + egl19 + slo1iso + slo1egl19 + shk1 + shl1 +
nca + kqt3] @ AVAL geometry. No NEURON reference (Mellem 2008 plateau targets).

**Densities chosen:**

| Channel | Density (S/cm²) | Source |
|---|---|---|
| leak | 1.336e-5 | AVAL g0 |
| egl19 | 9.288e-6 | AVAL g0 |
| nca | 0.0 | AVAL g0 (zero in Nicoletti's AVAL) |
| slo1iso | 1.518e-3 | AIY g0 (1.0 nS / 65.89e-8 cm² = 1.518e-3 S/cm²) |
| slo1egl19 | 1.396e-3 | AIY g0 (0.92 nS / 65.89e-8 cm²) |
| shl1 | 7.589e-4 | AIY g0 (0.5 nS / 65.89e-8 cm²) |
| shk1 | 1e-4 | conservative default (not in AIY's set; VA5 has shk1 but exact gbar not extracted) |
| kqt3 | 1e-4 | conservative default (AIY uses kqt1 not kqt3) |

**Rationale:** the spec instructs "use Nicoletti's published AVA channel densities
for channels she provides (NCA, EGL-19). For channels not in Nicoletti's AVA
(SLO-1, SHK-1, SHL-1, KQT-3), use reasonable defaults from Nicoletti's other
cells (e.g., AIY's SLO-1+EGL-19 density, RIM's SHK-1 density) scaled by AVA
capacitance."

**Note on densities being intensive:** Nicoletti's per-cell g vectors are in nS
(extensive). The S/cm² conversion via `g_nS / surf_cm²` gives the intensive
S/cm² density. Per Brian2 convention, Brian2 cells use S/cm² directly. So the
"AIY S/cm² density" applied to AVA's surface gives a different total nS but the
same S/cm² density. This is the natural transfer of intensive parameters across
cell sizes.

**Initial conditions:**
- v_init = -60 mV (hyperpolarized; let cell settle naturally)
- cai_mM_static = 5e-5 (NEURON default; matches Nicoletti's AIY where slo1iso is used)

**Mellem 2008 protocol:**
- 200 ms settle at I=0
- 100 ms × 50 pA injection
- 1500 ms post-stim recovery

**Result:**
- v_rest after settle = -62.74 mV (vs Mellem's -25 mV — but our cell has different
  channel set than Mellem's biological AVA, so v_rest naturally differs)
- Peak V during stim = -15.90 mV (i.e., depolarization of 46.8 mV from baseline)
- Plateau amplitude = 46.8 mV (FAIL — target 15-25 mV: too large)
- Plateau duration = 21.4 ms (FAIL — target 400-800 ms: too short)

**Verdict: 2b FAIL.** The cell does NOT reproduce Mellem 2008 plateau signature.

---

## Diagnosis: why does 2b fail?

The failure is **NOT** due to channel translation defects. Components 2a and the
per-channel validations (Phases C, D, E) all PASS.

The failure is structural:

1. **Plateau amplitude too large (46 mV vs 15-25 mV target):**
   The depolarization driven by 50 pA injection ÷ cell capacitance × pulse duration
   is too large for AVA's small surface area. With AVAL's g_leak ≈ 13.4 μS/cm² ×
   surf 1124e-8 cm² = 0.150 nS, the input resistance is ~6.7 GΩ. Injection of
   50 pA produces dV/dt = 50 pA / (cm·surf) = 50e-12 / (0.86e-6 × 1124e-8 × 1e6 / 1)
   over ~100 ms reaches several tens of mV. Without sufficient counter-balancing
   K current, the cell over-depolarizes.

2. **Plateau duration too short (21 ms vs 400-800 ms target):**
   After stim ends, the cell's V repolarizes too fast. SLO-1 isolated reads bulk
   cai (static at 5e-5 mM in our setup), so it provides constant K conductance
   that can't dynamically respond to depolarization. SLO-1+EGL-19 provides
   nanodomain-driven activation, but the time constant of repolarization is
   determined by leak τ = cm/g_leak ≈ 0.86 / 1.34e-5 ≈ 64 ms — and our observed
   21 ms is even faster, suggesting active K current is dominating.

3. **Mellem 2008 plateau is a 600 ms graded depolarization phenomenon** that
   relies on Ca-induced Ca-release and subsequent SLO-1 activation. Our cell
   has no dynamic Ca pool, so SLO-1's Ca-feedback can't operate. Without that
   feedback loop, the plateau termination is leak-dominated rather than
   active-K-dominated.

---

## Per spec's decision tree

> **2a-pass / 2b-fail:** **Condition 6 surfaces.** Channels work, architecture
> insufficient. Per architectural plan: **PAUSE for morning review, do NOT
> auto-trigger morphology fork.**

This is the load-bearing decision the cross-session adversarial review pattern
is designed for. **Pausing here for morning review is the correct outcome.**

The diagnosis suggests:
- Single-compartment AVA cell with current channel set cannot produce Mellem
  2008 plateau dynamics.
- Reasons: (1) v_rest mismatch, (2) lack of dynamic Ca-pool feedback for SLO-1,
  (3) channel density choices may not be optimal for AVA-specific dynamics.
- Architectural directions to consider:
  a. Add dynamic Ca-pool (caintra1 or cadiff) so SLO-1 isolated has Ca-feedback
  b. Tune channel densities via parameter optimization (similar to Nicoletti's
     g_to_Scm2 + iclamp matching workflow)
  c. Add multi-compartment morphology (the spec's "morphology fork" — but per
     decision tree, do NOT auto-trigger this)
  d. Investigate whether Mellem 2008's experimental setup matches our simulated
     setup (different drug conditions, different cells)

These are inputs to morning review, NOT autonomous architectural commitments.

---

## Update 2026-04-26 (post density-sensitivity sweep)

A 4×4 density-sensitivity sweep + 4 extension probes (see
`density_sensitivity_analysis.md`) was run to distinguish "wrong densities
masquerading as architecture-insufficient" from "true architectural
insufficiency". The 5 non-Nicoletti density parameters were varied across
combined 1024× range (32× terminator × 32× Kv).

**Verdict: VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS.** Amplitude can be tuned
into target range via Kv (kv=8 → 17.7 mV) but only at the cost of duration
collapsing to 4.4 ms. Maximum duration anywhere in the swept volume is 42 ms —
an order of magnitude short of the 400-800 ms target.

**Condition 6 is empirically confirmed.** The mechanism is identified:
SLO-1 isolated reading static `cai = 5e-5 mM` (per F12) cannot mediate
Ca-feedback, so no slow positive-feedback loop sustains a plateau. Terminator
density has near-zero leverage on the phenotype, directly demonstrating that
the missing ingredient is *Ca dynamics*, not *amount of SLO-1*.

The densities documented in the table above remain the cleanest defaults for
single-compartment AVA construction within Nicoletti's framework. Revising
them is unwarranted given the sweep result.
