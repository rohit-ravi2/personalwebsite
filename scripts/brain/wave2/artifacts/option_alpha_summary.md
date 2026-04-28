# Wave 2 option α-1 — outcome summary

**Date:** 2026-04-26 (Session 3 redeployment, α-1 scope authorized post-pushback)
**Final verdict:** **PRODUCTION_GRADE**

---

## Headline result

**Brian2 4-channel AVA cell `[IRK + LEAK + EGL19 + NCA]` reproduces
Nicoletti's NEURON AVAL phenotype to ~5 decimal places of voltage
trajectory across the canonical 1000 ms current-clamp protocol at all
7 published current levels (-30 to +30 pA in 10 pA steps).** This is the
cleanest apples-to-apples Brian2-vs-NEURON agreement achieved in any
Wave 2 phase to date.

Voltage-clamp Layer A (component 2a) also passes cleanly: 11/11 holds
with maximum divergence 0.0035 against the same canonical NEURON cell.

---

## Sub-checkpoint results

| CP  | Deliverable                                  | Status                | Notes                                     |
|-----|----------------------------------------------|-----------------------|-------------------------------------------|
| CP1 | `wave2/channels/unc103.py`                   | PASS (11/11 holds)    | Bug found+fixed in NEURONReference        |
| CP2 | `wave2/channels/irk.py`                      | PASS (11/11 holds)    | Clean translation                         |
| CP3 | `wave2/option_alpha_ava_cell.py`             | PASS (smoke test)     | True 4-channel Nicoletti AVAL             |
| CP4 | `wave2/run_option_alpha_cp4.py`              | PRODUCTION_GRADE      | 2a + 2b both pass to 5-decimal precision  |
| CP5 | This document                                | (this document)       |                                           |

---

## What changed from the original Wave 2 option α prompt

The original prompt instructed building a **5-channel AVA cell** with
`[IRK + LEAK + EGL19 + NCA + UNC103]`. Pre-flight verification surfaced
two errors in that framing:

### Error 1: AVAL is 4 channels, not 5

**Source:** `nicoletti_2024/AVAL_simulation_iclamp.py` lines 29-32:
```python
soma.insert('irk')
soma.insert('leak')
soma.insert('egl19')
soma.insert('nca')
```

No `soma.insert('unc103')`. UNC-103 is in **AVAR** (5-channel set), not
AVAL. The original prompt conflated AVAR's channel list with AVAL's.

### Error 2: F2 framing for UNC-103 was misattribution

The original prompt claimed UNC-103 needed special "GLOBAL→per-cell"
handling per F2 in the F1-F17 catalog. But F2 (per
`phase_beta_findings.md` and `translation_patterns.md` P2) is about
**caintra1's Ca trajectory**, not UNC-103. UNC-103's NMODL has no GLOBAL
declarations — STATE block is standard `m h` per-cell (RANGE-default).

UNC-103 follows the same translation pattern as SHK-1, SHL-1, KQT-3 —
clean voltage-gated K, no special handling needed.

### Resolution: option α-1

User authorized **α-1** after reviewing the pushback:

- **CP1:** translate UNC-103 anyway (useful for future AVAR work, clean
  voltage-gated K regardless). Output: `wave2/channels/unc103.py`. Not
  used in CP3/CP4.
- **CP2:** translate IRK as planned. Output: `wave2/channels/irk.py`.
- **CP3:** build Brian2 AVA cell with TRUE 4-channel set
  `[IRK + LEAK + EGL19 + NCA]`, NCA gbar=0 per Nicoletti. Output:
  `wave2/option_alpha_ava_cell.py`.
- **CP4:** Phase F re-evaluation with 1000 ms protocol matching
  Nicoletti's actual AVAL_simulation_iclamp.py protocol. Apples-to-apples
  via direct upstream invocation. Output:
  `wave2/run_option_alpha_cp4.py` + `option_alpha_phase_f_results.json`.

This restored alignment between the prompt's prose framing ("re-ground
to Nicoletti's actual AVAL phenotype") and the actual biological referent
(4-channel AVAL).

---

## Bug surfaced and fixed

**Location:** `wave2/neuron_reference.py`,
`NEURONReference._default_currents_to_record()` custom-cell branch
(around line 759).

**Bug:** UNC-103 was missing from the K-channel list used to determine
which currents to record. When a custom-mode cell included UNC-103, the
mechanism was inserted and gbar set correctly, but its `ik` contribution
was silently dropped from the I_total summation in voltage-clamp output.

**Symptom:** UNC-103 voltage-clamp validation initially showed NEURON
producing only ~3 pA at depolarized holds (pure leak only) while Brian2
showed thousands of pA. Diagnostic confirmed the mechanism was correctly
inserted; the recording list was the gap.

**Fix:** added `'unc103'` to the K-channel tuple in
`_default_currents_to_record()`. Verified post-fix: 11/11 holds with
divergence < 0.001.

This bug had been latent since UNC-103 was first listed in the `k_using`
set (insertion was correct, recording was incomplete). It surfaced
because CP1's UNC-103 validation was the first systematic test of UNC-103
in custom-mode NEURONReference. Earlier work used UNC-103 only via
`avar_unc103_patch.py` for full AVAR construction (a different code path).

---

## Methodological observations

### Pre-flight verification caught two propagation errors

Both errors in the original prompt were **citation-style propagation
errors** of the same kind documented in
`architectural_plan_citation_audit.md` (Mellem 2008 / Nicoletti 2019 PCBI
/ Wang 2001 / speculative per-cell table). Same pattern:

- Orchestrator-side prompt synthesis introduces a factual error
  (e.g., "AVAL has channel X" when it's actually AVAR).
- Agent-side primary-source verification catches it before code commit.
- Without pre-flight verification, the error would propagate into 4
  subsequent checkpoints (CP1 documentation, CP3 cell construction,
  CP4 NEURON reference construction, CP5 outcome interpretation).

The pushback document (`option_alpha_pushback.md`) is a load-bearing
artifact: it externalizes the verification step, makes the propagation
chain explicit, and triggers a course correction before code commits.
This is the same adversarial pattern that caught the Mellem 2008
investigation in a prior session.

### NEURONReference as canonical NEURON path

Component 2a uses `NEURONReference("AVAL")` — Nicoletti's actual
construction via her wrapper. This is preferred over custom-mode for
canonical comparisons because:

1. It exercises Nicoletti's `gScm2()` rescaling pipeline (the parameter
   conversion is part of her published model and should be tested).
2. It uses her exact section ordering, parameter assignment loop, and
   eca/ek setting conventions.
3. Future canonical-AVAL work can re-use it without redundant custom_spec
   construction.

Component 2b uses **direct upstream invocation** (re-implementing the
Section construction inside our runner with verbatim copies of
Nicoletti's code lines) because:

1. We need to sweep our chosen current levels with NEURONReference's
   per-section single-instance discipline.
2. Direct copy ensures the upstream construction is bit-identical to
   what Nicoletti's published wrapper produces.

Both paths produce the same NEURON behavior; they're equivalent ways to
invoke the same canonical model.

### Voltage trajectory agreement to 5 decimals

The Brian2 vs NEURON V trajectories agree to ~0.001 mV across the 1000 ms
current-clamp protocol. Sources of remaining residual:

- Brian2 uses `rk4` integration; NEURON uses CVODE-style implicit methods
  by default (configured via `cnexp` per-mod for state variables).
- Brian2 simulates V on a uniform grid; NEURON's adaptive timestep can
  differ slightly.
- Floating-point round-off across ~100k integration steps.

That residual stays at the ~0.001 mV level even when V swings 200 mV
(from +120 at +30 pA to -175 at -30 pA) is strong evidence that all
channel kinetics are translated correctly.

---

## Limitations / non-results

### Non-physiological V extremes

At ±30 pA injection the cell reaches +120 mV / -175 mV. These are not
physiological — real C. elegans neurons stay roughly bounded by E_K and
E_Na. Nicoletti's 4-channel AVAL parameterization lacks the additional
machinery to constrain V (no Na+ channels, no active pumps, no large K+
conductances at extremes). NEURON shows the same extremes — this is a
feature of the model, not a translation defect.

This means the option α-1 cell **cannot** be used as-is for biological
phenomena that require V to stay in plausible biological range. For that,
we'd need either:
- Additional channels (Na+, more diverse K+) → Phase F 2b cell territory
- Restricted current ranges (e.g., -10 to +10 pA) where V stays bounded
- Multi-compartment morphology with active dendrites

Within Nicoletti's published parameterization, this is the cell.

### Mellem 2008 plateau dynamics still unreproduced

The 4-channel cell does NOT reproduce Mellem 2008's 600 ms graded
depolarization plateau. That phenotype requires SLO-1 + Ca-induced Ca
release dynamics that aren't in Nicoletti's 4-channel AVAL. Phase F's
2b cell (7-channel essential set) was the previous attempt to reproduce
Mellem; it failed (Condition 6 surfaced — see
`gate2_ava_cell_construction.md`).

The α-1 cell is **explicitly NOT trying to reproduce Mellem**. It's
matching Nicoletti's own AVAL phenotype, which is a different (simpler,
non-plateau) electrical behavior. The Phase F 2b architectural-
sufficiency question remains open for separate investigation.

---

## Files produced

- `wave2/channels/unc103.py` — UNC-103 voltage-gated K (CP1)
- `wave2/channels/irk.py` — IRK inwardly-rectifying K (CP2)
- `wave2/option_alpha_ava_cell.py` — Brian2 4-channel AVA cell (CP3)
- `wave2/run_option_alpha_cp1.py` — UNC-103 validation runner
- `wave2/run_option_alpha_cp2.py` — IRK validation runner
- `wave2/run_option_alpha_cp4.py` — Phase F re-evaluation runner
- `wave2/diagnose_unc103.py` — diagnostic that surfaced NEURONReference bug
- `wave2/artifacts/option_alpha_pushback.md` — pre-flight pushback document
- `wave2/artifacts/option_alpha_findings.md` — mid-flight findings
- `wave2/artifacts/option_alpha_cell_construction.md` — CP3 architectural choices
- `wave2/artifacts/option_alpha_summary.md` — this document
- `wave2/artifacts/option_alpha_phase_f_results.json` — full CP4 results
- `wave2/artifacts/unc103_validation_results.json` — CP1 detail
- `wave2/artifacts/irk_validation_results.json` — CP2 detail
- `wave2/artifacts/checkpoints/option_alpha_CP{1,2,3,4}_status.json` — checkpoint status

## Files modified

- `wave2/neuron_reference.py` — added `'unc103'` to custom-cell K-current
  recording list (one-line bug fix; line ~759).

---

## Verdict

**PRODUCTION_GRADE.**

Brian2 translations of UNC-103 and IRK are validated. The Brian2 4-channel
AVA cell built from Nicoletti's published AVAL parameter vector reproduces
the canonical NEURON AVAL phenotype to ~5 decimal places of V trajectory
under both voltage-clamp Layer A (11/11 holds) and current-clamp Layer A
(7/7 sweeps × 5000 timepoints, 100% pass). No translation defects, no
architectural gaps, no DEEPER_FINDING required.

The corrected α-1 scope (after pre-flight pushback) produced cleaner
results than the original 5-channel framing would have allowed — the
5-channel synthetic AVAL would have had no biological referent to compare
against, while the 4-channel AVAL has Nicoletti's exact canonical cell as
its reference.
