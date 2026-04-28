# Wave 2 option α-1 — mid-flight findings

**Date:** 2026-04-26 (Session 3 resumed; α-1 authorized)
**Status:** COMPLETE — all CPs PASS, final verdict PRODUCTION_GRADE.
See `option_alpha_summary.md` for the outcome doc.

---

## Resume acknowledgment

User authorized **option α-1** (corrected scope) after pre-flight pushback.
Prior session paused at pre-flight when primary-source verification surfaced
two errors in the original α prompt:

1. **AVAL has 4 channels, not 5.** Original prompt's
   `[IRK, LEAK, EGL19, NCA, UNC103]` was an orchestrator-side error —
   conflated AVAR's channel list with AVAL's. Verified directly from
   `nicoletti_2024/AVAL_simulation_iclamp.py` lines 29-32.
2. **F2 misattribution.** Original prompt's "F2 GLOBAL→per-cell pattern for
   UNC-103" was a separate error — F2 is about caintra1's Ca trajectory
   (per `phase_beta_findings.md` and `translation_patterns.md` P2);
   UNC-103's NMODL has no GLOBAL state. UNC-103 is a clean voltage-gated
   K translation following the SHK-1/SHL-1/KQT-3 pattern.

Both caught by primary-source pre-flight verification before code commit —
same propagation pattern as Mellem 2008 / Nicoletti 2019 PCBI / Wang 2001 /
speculative per-cell table errors documented in
`architectural_plan_citation_audit.md`.

Full pushback record: `option_alpha_pushback.md`.

---

## CP1 — UNC-103 translation: PASS

11/11 holds, divergence < 0.001 across all holds. Translation followed
the SHK-1/SHL-1/KQT-3 pattern verbatim (no GLOBAL state, voltage-only,
m·h gates with PRODUCT-form tau).

**Bug found+fixed during validation:** `NEURONReference._default_currents_to_record()`
custom-cell branch was missing `'unc103'` in the K-channel list. NEURON
inserted UNC-103 correctly but the recorder didn't include its `ik` in
the I_total summation, so initial test showed NEURON returning only
`i_leak` (~3 pA) while Brian2 showed ~thousands of pA. One-line fix in
`neuron_reference.py`.

Output: `wave2/channels/unc103.py`, `wave2/run_option_alpha_cp1.py`,
`wave2/artifacts/checkpoints/option_alpha_CP1_status.json`.

---

## CP2 — IRK translation: PASS

11/11 holds, max divergence 0.008. Single-gate (m only) inwardly-rectifying
K with U-shaped tau. Translation clean — no special handling needed.

Output: `wave2/channels/irk.py`, `wave2/run_option_alpha_cp2.py`,
`wave2/artifacts/checkpoints/option_alpha_CP2_status.json`.

---

## CP3 — Brian2 AVA cell with TRUE 4-channel set: PASS (smoke test)

Cell built from `[egl19 + leak + irk + nca]` with NCA gbar=0 per Nicoletti.
Geometry/cm/eleak from `AVAL_simulations.py` line 26. Smoke test runs
without error; cell drifts to physiologically reasonable resting V over
100 ms.

Output: `wave2/option_alpha_ava_cell.py`,
`wave2/artifacts/option_alpha_cell_construction.md`,
`wave2/artifacts/checkpoints/option_alpha_CP3_status.json`.

---

## CP4 — Phase F re-evaluation: PRODUCTION_GRADE

**Component 2a (voltage-clamp):** 11/11 holds pass, max divergence 0.0035
against NEURONReference("AVAL") (Nicoletti's canonical wrapper).

**Component 2b (current-clamp 1000 ms protocol):** 7/7 sweeps pass with
~5-decimal-place V trajectory agreement. Direct upstream invocation of
`AVAL_simulation_iclamp.py`'s section construction with all 7 published
current levels (-30 to +30 pA in 10 pA steps).

The agreement quality (typical residual ~0.001 mV across 200+ mV V swings)
is the cleanest apples-to-apples Brian2-vs-NEURON match achieved in any
Wave 2 phase to date.

Output: `wave2/run_option_alpha_cp4.py`,
`wave2/artifacts/option_alpha_phase_f_results.json`,
`wave2/artifacts/checkpoints/option_alpha_CP4_status.json`.

---

## CP5 — Outcome summary: COMPLETE

See `option_alpha_summary.md` for the full outcome document with verdict
**PRODUCTION_GRADE**, key observations, limitations, and file map.

Headline: Brian2 4-channel AVA reproduces Nicoletti's canonical NEURON
AVAL to 5 decimals of V trajectory. No translation defects, no
architectural gaps. Original Wave 2 option α prompt's 5-channel framing
was an orchestrator-side error caught by pre-flight verification; the
corrected α-1 scope produced clean PRODUCTION_GRADE results.
