# Wave 2 cellular validation — pre-flight pushback

**Date:** 2026-04-26 (Session 4 redeployment, Wave 2 cellular extension)
**Author:** Wave 2 cellular validation pre-flight
**Status:** PAUSED_FOR_REVIEW — channel-translation envelope and RMD reference availability differ from prompt's framing.

---

## Summary

The prompt assumes most channels needed for AIY/RIM/RMD are already in
our 9 existing translations, with at most "one missing channel like
CCA-1" gating RMD's scope. **Primary-source verification of Nicoletti's
actual cell scripts shows the channel-deficit picture is materially
different.** AIY needs 1 new translation; RIM needs 3 new translations;
RMD has no Nicoletti 2024 reference at all (Mellem investigation's
"Nicoletti 2019 had RMD model" claim verified only in the abstract — the
2019 PLOS ONE paper exists but its source code is **not** in our local
upstream).

This pushback documents what primary-source verification surfaces, then
proposes three scope-discipline options for user adjudication.

---

## Primary-source verification: AIY

**Source:** `nicoletti_2024/AIY_simulation_iclamp.py` lines 28-38 (insertion
order in `soma.insert()` calls) and `nicoletti_2024/AIY_simulation.py`
line 25 (parameter g0 with comment).

**Insertion order (canonical):**
```python
soma.insert('egl19')
soma.insert('slo1egl19')
soma.insert('nca')
soma.insert('leak')
soma.insert('slo1iso')
soma.insert('kqt1')
soma.insert('shl1')
```

**Parameter mapping (gAIY_scaled indices, from iclamp):**
- `[0]` → leak.gbar
- `[1]` → slo1iso.gbar
- `[2]` → kqt1.gbar
- `[3]` → egl19.gbar
- `[4]` → slo1egl19.gbar
- `[5]` → nca.gbar
- `[6]` → shl1.gbar
- `[7]` → leak.e (eleak)
- `[8]` → cm

**Published g0 (AIY_simulation.py line 25):**
```python
# top-level comment: "leak, slo1iso, kqt1, egl19, slo1egl19, nca, irk, eleak, cm"
g0 = [0.14, 1, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
```

**Inconsistency in upstream:** The top-level comment in `AIY_simulation.py`
labels position [6] as "irk" but the iclamp script consumes it as
`shl1.gbar`. Trust the iclamp script (the actual executed code path).
The comment is wrong.

**Geometry:** `surf=65.89e-8 cm²`, `vol=7.42e-12`, `Ra=100`, `cm` from g0.

**Initial conditions:** `h.finitialize(-60)`.

**Current-clamp protocol:**
- `stim.delay = 1000 ms`, `stim.dur = 5000 ms`
- `simdur = 11000 ms`
- 11 levels: `linspace(s1=-15 pA, s2=35 pA, ns=11)` per AIY_simulation.py
- `h.dt = 0.4 ms` (note: much coarser than AVAL's 0.025 ms)

**Verified channel set:** `[egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1]`.
**7 channels.**

### AIY translation status against existing 9 channels

Existing: `egl19, irk, kqt3, nca, shk1, shl1, slo1_egl19_coupled, slo1_iso, slo1_iso_dynamic_ca, unc103`.

| AIY channel | Available? | Notes |
|---|---|---|
| egl19 | YES | `channels/egl19.py` |
| slo1egl19 | YES | `channels/slo1_egl19_coupled.py` |
| nca | YES | `channels/nca.py` |
| leak | YES (built-in) | inline in cell construction |
| slo1iso | YES | `channels/slo1_iso.py` (and `slo1_iso_dynamic_ca.py`) |
| **kqt1** | **NO** | KQT-3 exists; KQT-1 is a different channel |
| shl1 | YES | `channels/shl1.py` |

**KQT-1 vs KQT-3 are NOT interchangeable.** Both are KCNQ-family K
channels but the gating equations are clearly different (verified by
diff of mod files):

- KQT-1: `ik = gbar * m * s * (v - ek)` — single m·s gating, simpler.
- KQT-3: `ik = gbar * (0.3*mf + 0.7*ms) * s * w * (v - ek)` — fast/slow
  m components plus extra w state, more complex.

Different parameter values, different gating-variable count (KQT-1 has
2 states; KQT-3 has 4 states). Substituting KQT-3 for KQT-1 would be a
fabrication.

**KQT-1 translation feasibility:** kqt1.mod is 104 lines, no GLOBAL
declarations, standard m·s gating with m·∞ and s·∞ steady-states plus
mtau and stau time constants. **Easier than KQT-3**, follows the same
F1-F17 pattern as SHK-1/SHL-1/UNC-103. Estimated translation effort:
~30 min following the established channel-translation pattern.

**Decision required:** is KQT-1 translation in-scope for this work block?

---

## Primary-source verification: RIM

**Source:** `nicoletti_2024/RIM_simulation_iclamp.py` lines 31-37.

**Insertion order:**
```python
soma.insert('shl1')
soma.insert('egl2')
soma.insert('irk')
soma.insert('cca1')
soma.insert('unc2')
soma.insert('egl19')
soma.insert('leak')
```

**Parameter mapping (gRIM_scaled indices):**
- `[0]` → shl1.gbar
- `[1]` → egl2.gbar
- `[2]` → irk.gbar
- `[3]` → cca1.gbar
- `[4]` → unc2.gbar
- `[5]` → egl19.gbar
- `[6]` → leak.gbar
- `[7]` → leak.e
- `[8]` → cm

**Published g vector (RIM_simulation.py line 27, units S/cm² already):**
```python
g = [0.0009048750067326097,    # shl1
     0.0001411644285181245,    # egl2
     0.0003272854640954744,    # irk
     0.0008451919806776876,    # cca1
     9.676795045480941e-05,    # unc2
     0.00032005818627638106,   # egl19
     9.676795045480941e-05,    # leak
     -50,                      # eleak
     1.5]                      # cm
```

**Note:** RIM's g vector is already in S/cm² (no `gScm2()` rescaling
applied), unlike AVAL/AIY. RIM_simulation.py does NOT call gScm2().
Verify in implementation.

**Geometry:** `surf=103.34e-8 cm²`, `Ra=100`.

**Current-clamp protocol:**
- `stim.delay = 5000 ms`, `stim.dur = 5000 ms` (later than AIY/AVAL — 5s
  pre-stim)
- `simdur = 14000 ms` (14 second runs! Long; will need timeout monitoring.)
- 11 levels: `linspace(-15 pA, 35 pA, 11)`
- `h.dt = 0.04 ms` (10x finer than AIY)
- Output cuts initial transient at t=4000 ms; reports time relative to
  4000 ms.

**Verified channel set:** `[shl1, egl2, irk, cca1, unc2, egl19, leak]`.
**7 channels.**

### RIM translation status against existing 9 channels

| RIM channel | Available? | Notes |
|---|---|---|
| shl1 | YES | `channels/shl1.py` |
| **egl2** | **NO** | EAG-family K channel |
| irk | YES | `channels/irk.py` |
| **cca1** | **NO** | T-type Ca |
| **unc2** | **NO** | P/Q-type Ca, has GLOBAL declarations (F2-class) |
| egl19 | YES | `channels/egl19.py` |
| leak | YES (built-in) | |

**Three new channel translations required for RIM.**

### RIM channel translation feasibility

- **CCA-1 (cca1.mod, 105 lines):** standard m²·h Ca gating, no GLOBAL.
  `ica = gbar * m * m * h * (v - eca)`. Clean translation. Estimated
  effort: ~30-45 min.
- **EGL-2 (egl2.mod, 85 lines):** voltage-gated K, presumed standard
  pattern (haven't read in full). ~30 min if standard.
- **UNC-2 (unc2.mod, 132 lines):** P/Q-type Ca with **GLOBAL minf, hinf,
  mtau, htau, munc2, hunc2** declarations. This is F2-class (same pattern
  as caintra1's GLOBAL state). The state itself is `STATE { m h }` per
  cell, but global ASSIGNED variables are exposed. Translation requires
  reading the existing F2 pattern handling (see `slo1_egl19_coupled.py`
  and caintra1 work). Estimated effort: ~45-60 min if F2 pattern is
  well-established; longer if subtleties surface.

**Total RIM translation work:** ~2-2.5 hours minimum. **Beyond AVAL
template's "no new translations" envelope. Comparable to a full Wave 2
phase (Phase β was multi-hour translation effort across 6-9 channels).**

---

## Primary-source verification: RMD

**Critical finding:** Nicoletti 2024's local upstream
(`nicoletti_2024/`) has NO RMD simulation script.

**Files present in upstream:**
```
AIY_simulation*.py
AVAL_simulation*.py
AVAR_simulation*.py
RIM_simulation*.py
VA5_simulation*.py
VB6_simulation*.py
VD5_simulation*.py
```

No RMD. The Mellem investigation document
(`mellem_investigation_pushback.md`) noted:

> "Nicoletti 2024 has an RMD model from the 2019 paper (Nicoletti,
> Loppini, Chiodo, Folli, Ruocco, Filippi 2019, PLOS ONE)."

Verified locally:
- 2019 PLOS ONE paper exists (DOI 10.1371/journal.pone.0218738) — see
  `cca1.mod` and `kqt3.mod` headers, which cite "Nicoletti et al. PloS
  One 2019".
- The 2019 paper code itself is **not** in `simulation/upstream/`. Only
  some of its mod files ended up in the 2024 paper's `nicoletti_2024/`
  directory (cca1, kqt3, unc2, slo1iso, slo1egl19, slo2iso, slo2egl19,
  slo2unc2, kvs1, egl36, egl2 are 2019-paper provenance).
- `simulation/cells/RMD*.cell.nml` and `simulation/upstream/c302/.../RMD*.cell.nml`
  exist, but those are **NeuroML c302 cells**, not Nicoletti's
  single-cell HH models. Different framework, different model.

**RMD reference acquisition would require:**
1. Locating the 2019 paper's published code repository (probably a
   different GitHub repo than 2024's).
2. Cloning it locally (substantial — c302 alone is 1+ GB).
3. Inspecting their RMD simulation script for channel set, parameters,
   protocol.
4. Possibly translating any 2019-only channels not in our existing set
   (e.g., kvs1, egl36 are present in `nicoletti_2024/` mod files but
   may have 2019-specific RMD parameterizations not in 2024 cells).

**This is Scope C** by definition: substantial reference acquisition is
required before any RMD translation can begin. RMD does not fit this
work block's envelope under the prompt's own scope rules.

---

## Scope reassessment

The prompt's framing was:

> "Most likely realistic outcome: AIY PRODUCTION_GRADE + RIM
> PRODUCTION_GRADE + RMD DEFERRED (Scope C)."

Primary-source verification shows the realistic outcome is more
constrained:

- **AIY:** PRODUCTION_GRADE possible if KQT-1 translation is
  authorized (~30 min translation + standard validation = ~1.5-2 hours
  total).
- **RIM:** PRODUCTION_GRADE requires 3 new channel translations (CCA-1,
  EGL-2, UNC-2). Total RIM work block ~3-4 hours just for translations
  + Layer A validation + cell construction. **This exceeds the
  established AVAL template's envelope substantially.** It is more
  comparable to Phase β (multi-channel translation phase) than to a
  cell-validation work block.
- **RMD:** Scope C — Nicoletti 2024 has no RMD; 2019 paper code not
  local; reference acquisition required.

---

## Three options for user adjudication

### Option A — Strict envelope (no new channel translations)

- **AIY:** Cannot proceed. KQT-1 missing. Document scope state, defer.
- **RIM:** Cannot proceed. 3 channels missing. Document scope state, defer.
- **RMD:** Already Scope C. Defer.
- **Outcome:** Wave 2 cellular validation extends from AVAL alone to
  AVAL alone. No new cells production-grade this work block.
- **Use of session:** brief — write scope evaluations and stop.

### Option B — AIY only (one channel translation tractable)

- **AIY:** Translate KQT-1 (clean voltage-gated K, follows established
  pattern) + run AIY validation. Likely PRODUCTION_GRADE outcome.
- **RIM:** Defer (3 channels too much for envelope).
- **RMD:** Scope C, defer.
- **Outcome:** AVAL + AIY production-grade. RIM/RMD documented for
  future work blocks.
- **Use of session:** moderate — one channel translation + one cell
  validation.

### Option C — AIY + RIM (full Wave 2 cellular extension to interneurons)

- **AIY:** Translate KQT-1, validate AIY.
- **RIM:** Translate CCA-1, EGL-2, UNC-2. Validate RIM.
- **RMD:** Scope C, defer.
- **Outcome:** AVAL + AIY + RIM production-grade. RMD documented.
  Substantial Wave 2 progress (3 cells instead of 1).
- **Use of session:** large — 4 channel translations + 2 cell
  validations. Risk of UNC-2's GLOBAL handling surfacing F2-class
  subtleties that delay completion.
- **Risk:** UNC-2's F2-class GLOBAL pattern is not novel (caintra1
  established the pattern), but per-channel calibration may surface
  unexpected issues. Should plan for sub-checkpoint pause if UNC-2
  doesn't translate cleanly.

### My recommendation

**Option B (AIY only)**, with a follow-up work block scoped explicitly
for RIM's 3-channel translation phase. Reasoning:

1. AIY is a clean fit for the AVAL template — single missing channel,
   clean pattern, standard validation. High-confidence outcome.
2. RIM's 3 translations push the work block from "cellular validation"
   into "channel translation phase + cellular validation," which is the
   shape of Phase β, not Phase F. The two phase shapes have different
   discipline patterns and should be separately scoped.
3. RMD's Scope C status is clean and unambiguous — no time spent on
   ambiguous boundary cases.
4. The "substantively successful" outcome the prompt mentions becomes
   AVAL + AIY = 2 cells production-grade, which is real Wave 2 progress
   without scope creep.

Option C is also defensible if the user wants to attempt a larger
work block; flagged risks are real but tractable.

Option A is defensible if the user wants strict envelope discipline and
prefers separate channel-translation phases over combined work blocks.

---

## Verification methodology

This pre-flight surfaced the channel-deficit picture by reading three
sources directly:

1. `AIY_simulation_iclamp.py` for AIY's actual `soma.insert()` calls.
2. `RIM_simulation_iclamp.py` for RIM's actual `soma.insert()` calls.
3. `ls nicoletti_2024/` to confirm RMD's absence.

Plus cross-checks against:
- Existing `wave2/channels/` directory listing (10 modules).
- `wave2/artifacts/option_alpha_summary.md` (AVAL precedent).
- `wave2/artifacts/mellem_investigation_pushback.md` (RMD context).
- `mod` file diffs to verify KQT-1 ≠ KQT-3.

This is the fourth pre-flight pushback to surface a propagation error
in a Wave 2 cellular prompt:

1. Mellem 2008 → AVA plateau attribution (caught by mellem investigation).
2. Wave 2 option α "5-channel AVAL" framing (caught by option α pre-flight).
3. F2 misattribution to UNC-103 (caught by option α pre-flight).
4. **This pushback** — channel-deficit picture for AIY/RIM/RMD.

The pattern is consistent: orchestrator-side prompt synthesis introduces
factual claims about channel sets / cell models / channel translations
without primary-source verification. Agent-side pre-flight reading of
the actual scripts catches it.

---

## Standing by

`PAUSED_FOR_REVIEW.txt` written. No code or new artifacts created beyond
this pushback document. Awaiting user adjudication of Option A / B / C
before any implementation work proceeds.
