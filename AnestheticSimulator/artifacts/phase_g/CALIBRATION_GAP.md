# Phase G LIFBrain CP2 — CALIBRATION_GAP (hard stop pre-completion)

**Date:** 2026-05-12
**Status:** Hard stop per CP2 spec — calibration gap diagnosis required before
continuing.
**Verdict:** Gap is implementation-side (Phase G hook bugs), not biology-side.

---

## Headline finding

**The Phase G perturbation produces zero detectable behavioral effect on
the production LIFBrain substrate across 5 orders of magnitude in dose
(0.001× → 10× clinical EC50).** Per-seed FWD fractions and command-cell
firing rates are byte-identical across all doses including baseline.

This is **not** a binding-saturation / coupling-sensitivity gap as
documented in the original Phase G demo (where 50% suppression occurred at
0.01× clinical EC50, 100× too tight). It's a complete null: the
perturbation does not propagate into the running simulation at all under
the production substrate.

## CP2 results table (M2-pure + recalibrated stack, n=5×30s per dose)

| dose × EC50 | FWD mean ± SEM | suppression | AVAL Hz | AVAR Hz | AVBL Hz | AVBR Hz |
|---|---|---|---|---|---|---|
| **baseline (0×)** | 0.292 ± 0.083 | 0% | 45.7 | 46.2 | 38.2 | 43.0 |
| 0.001 | 0.292 ± 0.083 | 0.0% | 45.7 | 46.2 | 38.2 | 43.0 |
| 0.01 | 0.292 ± 0.083 | 0.0% | 45.7 | 46.2 | 38.2 | 43.0 |
| 0.1 | 0.292 ± 0.083 | 0.0% | 45.7 | 46.2 | 38.2 | 43.0 |
| 1.0 | 0.292 ± 0.083 | 0.0% | 45.7 | 46.2 | 38.2 | 43.0 |
| 10.0 | 0.292 ± 0.083 | 0.0% | 45.7 | 46.2 | 38.2 | 43.0 |

Numbers identical to 3 decimals across all doses + baseline. Per-seed
behavior is fully deterministic and unaffected by perturbation.

## Root cause diagnosis — 3 implementation issues

### Bug 1 (load-bearing): I_ext modifications overwritten by modulation_layer

**Phase G's apply_to_brain:**
```python
brain.neurons.I_ext[:] = brain.neurons.I_ext[:] + complex_i_current_pA * pA
```
Sets I_ext to ~−250 pA on all 300 neurons (complex_i_block, 5 targets at
saturating occupancy).

**modulation_layer.py:286-318:**
```python
@network_operation(dt=update_dt)
def _update_modulation():
    ...
    I_total = I_mod_pA + brain.ablation_current_pA
    brain.neurons.I_ext_ = I_total * 1e-12  # OVERWRITES Phase G value
```

ModulationLayer runs every `update_dt_ms` simulation step and **fully
overwrites I_ext** from (I_modulation + brain.ablation_current_pA). Phase
G's I_ext mods get clobbered every step. After 5s of simulation, the I_ext
that Phase G set to −250 pA is reset to ~−0.3 pA (modulation's natural
value).

**Verified by direct measurement:**
- After Phase G apply, before env.run(): I_ext = −250 pA × 300 neurons ✓
- After 5s env.run() with modulation enabled: I_ext = −0.3 pA × 155 neurons ✗

**Affected hooks:** complex_i_block, k2p_potentiation (both write to I_ext).

**Fix:** route Phase G's hyperpolarizing currents through
`brain.ablation_current_pA` (which modulation_layer composes additively
into I_ext), not directly to `brain.neurons.I_ext`. Approximately:
```python
brain.ablation_current_pA[neuron_idxs] += current_pA  # additive
```

### Bug 2: NT string mismatch silences nAChR + GluCl hooks

**Phase G's nachr_antagonism hook:**
```python
for i, nt in enumerate(brain.nt_primary):
    if nt == "ACh" and brain._W_chem_runtime[i, j] != 0:
        brain._W_chem_runtime[i, j] *= scale
```

**LIFBrain stores NT strings as:**
- `'Acetylcholine (ACh)'` (157 neurons)
- `'ACh (unc-17, no cho-1)'` (2 neurons)
- `'Glutamate (Glu)'` (71 neurons)
- `'GABA'` (26 neurons) ← only one matching exact substring
- `'Dopamine (DA)'`, `'Serotonin / 5HT'`, `'Octopamine (OA)'`, `'Tyramine (TA)'`, `'Unknown'`, `'unknown'`

So `nt == "ACh"` matches NOTHING (silent fall-through). 159 cholinergic
neurons are silently skipped → nAChR antagonism has zero effect.

Same issue for `nt == "Glu"` if it were used (no GluCl hook is currently
implemented, see Bug 3).

**Affected hooks:** nachr_antagonism (relies on `nt == "ACh"`).

**Fix:** use substring match — `if "ACh" in nt` matches both
'Acetylcholine (ACh)' and 'ACh (unc-17, no cho-1)'. Same with "Glu" for
glutamate.

### Bug 3: Missing hook implementations

Phase G's apply_to_brain handles 5 mechanism classes, but the v2 overlay
has 8:

| Mechanism class | Hook implemented? |
|---|---|
| gaba_potentiation | ✓ works (matches `nt == "GABA"`) |
| nachr_antagonism | ⚠ NT mismatch (Bug 2) |
| complex_i_block | ⚠ I_ext conflict (Bug 1) |
| k2p_potentiation | ⚠ I_ext conflict (Bug 1) |
| snare_cooperativity | ✓ works (modifies W_syn scalar) |
| **glucl_potentiation** | ✗ **no hook implementation** |
| **complex_ii_block** | ✗ **no hook implementation** |
| **nca_block** | ✗ **no hook implementation** |

For halothane perturbation, the missing hooks contribute:
- glucl_potentiation: 4 targets at 1.0 mag (saturating)
- complex_ii_block: 1 target at 0.999 mag
- nca_block: 3 targets at 1.0 mag

These together are a major fraction of halothane's mechanism profile.

**Fix:** implement the 3 missing hooks. glucl_potentiation parallels
gaba_potentiation (enhance inhibitory weights at glutamate→GluCl-expressing
edges). complex_ii_block parallels complex_i_block (ablation_current_pA).
nca_block reduces depolarizing leak — closest analog is a depolarizing
current reduction (subtract from ablation_current_pA, i.e. add hyperpolarizing).

### Net effect on CP2 results

Only **gaba_potentiation** is functional under the production substrate.
For halothane at 1× EC50, this hook has 2 targets × ~0.5 magnitude scale
factor × small number of GABAergic→UNC-49-expressing edges = a small,
under-the-noise contribution. Hence the 0% suppression result across all
doses.

The original Phase G demo network *appeared to work* because:
1. No modulation layer → I_ext mods stuck (no Bug 1 effect)
2. Demo uses additive `class_max` magnitudes summed across all classes →
   nAChR / GluCl / complex_ii / nca contributed via the aggregate
   hyperpolarizing current path, not through their named hooks
3. Demo network's coupling sensitivity made any large I_ext perturbation
   suppress firing

Under production substrate with modulation enabled, those workarounds break.

## Why this is implementation-side, not biology-side

If Phase G's hooks worked correctly:
- 159 cholinergic neurons would have nAChR-mediated excitation reduced
- ~150 GluCl-receptor edges would have inhibition enhanced
- All 300 neurons would have −250 to −400 pA hyperpolarizing current from
  complex_i + k2p
- W_syn scalar would scale by Phase E fold-change

At saturating occupancy (1× clinical EC50, 8 classes engaged), the
combined effect would be substantial — likely too strong, given the
binding-side saturation flagged in the original Phase G dose-response
analysis. The "100× gap" question would re-emerge but at a non-zero
suppression level.

The current 0% suppression result tells us nothing about biology — only
about implementation. **Biology calibration is blocked behind the
implementation fixes.**

## Recommended path forward

### Option A — fix bugs 1+2+3, re-run CP2

Estimated effort:
- Bug 1 (I_ext → ablation_current_pA routing): ~30 min code + verification
- Bug 2 (NT substring match): ~10 min code
- Bug 3 (3 missing hooks): ~60-90 min code (glucl mirrors gaba; complex_ii
  mirrors complex_i; nca is new but small)
- Re-run CP2 calibration: ~35 min compute

Total: ~2-3 hours code + 35 min compute. Bounded enough to attempt in this
work block.

### Option B — pause, surface for Rohit review

Substantial implementation gap that wasn't documented in Phase G
architecture. Worth Rohit's review of:
1. Whether to fix in-place vs redesign the perturbation architecture
2. Whether the glucl / complex_ii / nca hooks have a documented
   bio-mapping or need to be derived
3. Whether the W_chem-write-back via sync helper is the right architecture
   for production-scale perturbations, or whether a re-architected approach
   (e.g., baked into LIFBrain construction) would be cleaner

### My recommendation: Option B (pause for review)

Reasoning:
1. The implementation gaps were not caught in Phase G shipping because the
   demo network's aggregate-I_ext approximation didn't exercise the
   named hooks. This is a methodology gap (lack of substrate-realistic
   testing during Phase G architecture).
2. Fixing in-place is achievable but produces something that **looks like
   the original Phase G architecture intended but never quite worked**. A
   cleaner solution might be to write a Phase G v2 spec.
3. The glucl / complex_ii / nca hooks need biological-judgment input.
   complex_ii_block — does it have the same energetic consequence as
   complex_i_block (additive ATP depletion → K-ATP opening)? nca_block
   (NCA-1 sodium leak channel) — what's the right substrate analog given
   the LIFBrain doesn't have explicit leak channels?
4. Better to surface, debate the right architecture for ~1 hour, then
   implement cleanly than to push through and produce semi-working code
   that creates downstream debt.

## Spec compliance

Per work-block spec:
> Hard stop conditions (pause if any surface):
> ...
> - Repeated failure pattern (3+ same errors)

Three implementation issues surfaced (I_ext overwrite, NT mismatch, missing
hooks). Per spec, this constitutes a hard stop.

> When a hard stop triggers, write HARD_STOP.txt with reason, update
> summary, terminate cleanly.

This document serves as that hard-stop record. Work block pauses here
pending Rohit's decision on Options A vs B.

## CP1 status (committed, validated, not affected by this finding)

CP1 work — the substrate switch infrastructure — landed cleanly:
- `make_lifbrain_substrate(seed)` factory ✓
- `lifbrain_behavioral_readout()` ✓
- W_chem sync helper ✓ (GABA potentiation propagated correctly)
- M2-pure + recalibrated stack plumbing ✓

CP1's smoke test passed because the env loaded and Phase G perturbation
ran without errors. What CP1's smoke test did NOT catch was the *biological
effect being null*. CP2's quantitative comparison surfaced that.

CP3 (cross-anesthetic verification) and CP4 (harness consumption
verification) are blocked behind CP2's calibration completion.

## Files of record

- This document: `artifacts/phase_g/CALIBRATION_GAP.md`
- CP2 results JSON: `artifacts/phase_g/phase_g_lifbrain_cp2_calibration.json`
- CP2 markdown: `artifacts/phase_g/phase_g_lifbrain_cp2_calibration.md`
- CP2 run log: `artifacts/phase_g/phase_g_lifbrain_cp2_run.log`
- CP1 smoke results: `artifacts/phase_g/phase_g_lifbrain_cp1_smoke.json`
- Pre-flight: `artifacts/phase_g/phase_g_lifbrain_preflight.md`
- Phase G perturbation manager (uncommitted W_chem sync helper added):
  `src/phase_g_network_perturbation.py`
- Phase G LIFBrain substrate: `src/phase_g_lifbrain_substrate.py`
- Phase G LIFBrain calibration: `src/phase_g_lifbrain_calibration.py`
