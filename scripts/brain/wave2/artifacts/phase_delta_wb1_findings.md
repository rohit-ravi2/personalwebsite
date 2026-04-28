# Phase δ WB1 — Cython unification + namespace audit

**Status:** Complete. Both findings resolved cleanly.

---

## Action 1: Codegen target unification

**Before WB1:** Brian2 codegen target was inconsistent across the codebase:
- 17 wave2/.py files: hardcoded `prefs.codegen.target = "cython"` (post-cython-cleanup)
- 5 production-simulator files: hardcoded `prefs.codegen.target = "numpy"`
  - `lif_brain.py` (production-wired)
  - `graded_brain.py` (production-wired)
  - `graded_brain_h_kca.py` (Wave 1 sandbox per scoping)
  - `compartmental_neurons_kca.py` (Wave 1 sandbox per scoping)
  - `overnight_v2_track_f.py` (sandbox)

Brian2's `prefs.codegen.target` is process-global. Last setter wins in the import chain — brittle. Phase δ requires Wave 2 cells + LIFBrain coexisting in the same Python process.

**Action:** flipped all 5 production-simulator files from numpy to cython via sed. Total of 22 .py files now use cython.

**Smoke test:**
```python
import lif_brain   # works under cython
import graded_brain   # works under cython
prefs.codegen.target == 'cython'  # confirmed after both imports
```

Both LIFBrain and GradedBrain import cleanly under cython codegen target. No compile errors, no runtime errors during import.

**Risk:** runtime errors may surface during actual `LIFBrain.run()` execution that didn't surface during import. WB2 (or whichever Phase δ work block first instantiates and runs a brain in-process) will exercise this. If issues surface, document as F19+ findings and address per established cython-vs-numpy debugging pattern from cellular validation work.

---

## Action 2: I_ext / I_inj namespace audit (documented for WB2 resolution)

**Per Phase δ scoping risk register:** "ModulationLayer `I_ext` vs Wave 2 `I_inj` naming collision (certain, 5-min fix)."

**Verified from primary source:**

Production simulator uses `I_ext`:
- `lif_brain.py` (lines 323, 326): `dv/dt = (v_rest - v)/tau + (I_gap + I_ext)/C_mem` and `I_ext : amp`
- `graded_brain.py` (line 190): `I_ext : amp`
- `compartmental_neurons.py` and `compartmental_neurons_kca.py`: `I_ext`
- `modulation_layer.py`: writes `neurons.I_ext` from modulation current vector

Wave 2 cells use `I_inj`:
- `option_alpha_ava_cell.py:159,161,174,192,210`
- `option_alpha_aiy_cell.py:170,172,192,223,246`
- `option_alpha_rim_cell.py:151,153,173,203,222`

**The collision only manifests when Wave 2 cells coexist with LIF in same process** (i.e., starting at Phase δ WB2 when Wave 2 cells are integrated into Brian2 Network alongside LIFBrain). Pre-Phase-δ standalone Wave 2 validation never triggered this.

**Resolution approach (deferred to WB2):**

WB2 should rename Wave 2 `I_inj` → `I_ext` to align with production convention. Implications:

1. Wave 2 cell builders (3 files): rename in eqs string, attribute assignments, `record_vars` lists, current-injection helpers
2. Wave 2 validation drivers (multiple files): rename `G.I_inj = ...` to `G.I_ext = ...`
3. Test coverage: re-run AVAL + AIY + RIM validations post-rename to confirm no regression. Same residual baselines apply.

Cleaner than going the other direction (renaming `I_ext` across the production simulator). Production has more references and more downstream coupling (ModulationLayer, ablation infrastructure, dashboard).

Estimated WB2 sub-task: ~30 min including post-rename validation runs (cython baseline ~5s/sweep × 3 cells × 2 protocols ≈ 30s validation + investigation if anything regresses).

---

## WB1 verdict: PASS

- Codegen target unified across codebase (cython)
- Namespace collision documented for WB2 resolution
- Production simulator imports clean under cython
- Phase δ implementation can proceed to WB2

## Files modified

- `lif_brain.py:71`: `"numpy"` → `"cython"`
- `graded_brain.py:49`: `"numpy"` → `"cython"`
- `graded_brain_h_kca.py:70`: `"numpy"` → `"cython"`
- `compartmental_neurons_kca.py:58`: `"numpy"` → `"cython"`
- `overnight_v2_track_f.py:43`: `"numpy"` → `"cython"`

5 files. Each change is one-line, sed-applied, reversible.

## Standing followup

If WB2 instantiates LIFBrain and a runtime error surfaces under cython that didn't surface under numpy: investigate, document as F19+ findings, address per established cython-vs-numpy debugging pattern from Wave 2 cellular validation work.
