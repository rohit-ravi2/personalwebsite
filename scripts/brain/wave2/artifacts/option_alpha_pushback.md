# Wave 2 option α — pre-flight pushback

**Date:** 2026-04-26 (Session 3 redeployment)
**Triggering work block:** `phase_v_w2_option_alpha_prompt.md`
**Status:** PAUSED before CP1 begins. Pre-flight verification surfaced load-bearing
inconsistency between the prompt's stated channel set and Nicoletti's actual
AVAL code (and the prior session's `mellem_investigation_pushback.md`).

---

## Headline finding

**The prompt instructs constructing a Brian2 AVA cell with the 5-channel set
"IRK + LEAK + EGL19 + NCA + UNC103." This is wrong for AVAL. UNC-103 is in
AVAR, not AVAL.** The actual Nicoletti AVAL is a **4-channel** cell:
**IRK, LEAK, EGL19, NCA** (with NCA gbar=0 in published parameters, so
effectively 3 active conductances).

The prompt's framing also attributes the "GLOBAL→per-cell pattern" to F2 in
the F1-F17 catalog. That's not what F2 says. F2 is about caintra1's absolute
trajectory matching NEURON. UNC-103's NMODL (read directly) declares no
GLOBAL state — its STATE block is standard per-cell `m h` with RANGE-default
semantics. The "F2 GLOBAL→per-cell" framing in the prompt appears to be a
prompt-author misremembering and does not correspond to a real translation
gotcha that needs special handling for UNC-103.

If I proceed as written, CP3's AVA cell would be inconsistent with the
target Nicoletti AVAL phenotype it's supposed to match in CP4. The
"apples-to-apples NEURON 5-channel reference" called for in CP4 would be
synthetic — Nicoletti's NEURON AVAL has 4 channels, not 5.

This warrants cross-session resolution before I commit code.

---

## Primary-source verification

### Nicoletti's actual AVAL — 4 channels, not 5

**File:** `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAL_simulation_iclamp.py`
**Lines 29-32:**
```python
soma.insert('irk')
soma.insert('leak')
soma.insert('egl19')
soma.insert('nca')
```

**Line 38-44 (per-segment gbar assignment):**
```python
seg.egl19.gbar=gAVA_scaled[0]
seg.leak.gbar=gAVA_scaled[1]
seg.irk.gbar=gAVA_scaled[2]
seg.nca.gbar=gAVA_scaled[3]
seg.leak.e=gAVA_scaled[4]
```

No `soma.insert('unc103')`. No `seg.unc103.gbar` assignment. The vclamp
variant (`AVAL_simulation_vclamp.py` lines 30-34) is identical. Both AVAL
sims insert exactly 4 channels.

**Per AVAL_simulations.py line 25-26 (the wrapper that runs both):**
```python
# coductances: egl19, leak, irk, nca, eleak, cm
g0=[0.104385,0.150164,0.1,0,-39,0.859551]
```

So Nicoletti's published AVAL parameter vector has `g_nca = 0`. The
biological AVA in her model is effectively 3 active channels: IRK, LEAK,
EGL19. NCA is in the channel set but turned off.

### Nicoletti's actual AVAR — 5 channels (this is where UNC-103 lives)

**File:** `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAR_simulation_vclamp.py`
**Lines 38-42:**
```python
soma.insert('irk')
soma.insert('leak')
soma.insert('egl19')
soma.insert('nca')
soma.insert('unc103')
```

**AVAR_simulation.py line 27-28:**
```python
# CONDUCTANCES: EGL19, LEAK, IRK, NCA, UNC103, ELEAK, CM
g0=[0.0643372,0.225225,0.042079,0.0493356,0.0481669,-37,0.751761]
```

UNC-103 is part of AVAR's 5-channel set, not AVAL's.

### The prior pushback document agrees

`wave2/artifacts/mellem_investigation_pushback.md` lines 190-203 state explicitly:

> "Nicoletti's AVA channel set (from `AVAL_simulation_iclamp.py` source):
>
> ```python
> soma.insert('irk')
> soma.insert('leak')
> soma.insert('egl19')
> soma.insert('nca')
> ```
>
> **Only 4 channels: IRK, LEAK, EGL19, NCA.** No SLO-1, no SHK-1, no SHL-1,
> no KQT-3."

The prompt for option α is internally inconsistent with the very pushback
document that established option α. The prompt's channel list adds UNC-103
to the 4-channel set; the pushback says 4 channels.

### The avar_unc103_patch.py confirms UNC-103 is AVAR-specific

`wave2/avar_unc103_patch.py` (already in the codebase) is an upstream-bug
workaround: AVAR_simulation.py imports `AVAR_simulation_iclamp` from a file
that's not present in the upstream tree. The patch reconstructs that file
and includes `soma.insert("unc103")` because the *AVAR comment line*
explicitly lists UNC-103. The patch lives in our directory specifically
because UNC-103 belongs to AVAR. There's no analogous insertion in any
AVAL code path.

---

## F2 framing in the prompt — also incorrect

**Prompt claim:** "F2 lesson: UNC-103 has GLOBAL state in NMODL that needs
per-cell conversion in Brian2."

**Actual F1-F17 catalog (from `phase_beta_findings.md` and
`translation_patterns.md`):**

- F1: cadiff and caintra1 do NOT write NEURON's `cai` ion variable
- F2: caintra1's Ca trajectory has questionable absolute values
- F3: cadiff produces dramatic Ca transients (1e-1 mM range)
- F4: Decision — use cai_mM internally to match NEURON
- F5: caintra1's `cai` doesn't update — channels reading cai must use mechanism state
- F6: NMODL hidden unit-conversion factor (later revised in F11)
- F7-F10: calibration / VC protocol / unit handling
- F11-F13: Ca-pool diagnosis (run #2 Phase A revision of F6)
- F14: h.run() re-finitializes via h.v_init (default -65 mV)
- F15: Brian2 vs NEURON SS extraction window mismatch
- F16, F17: caintra1↔slo1iso unit conversion + fca scaling (per the prompt's
  own preamble, but not in the catalog under those numbers — the catalog
  stops at F15 in `phase_beta_findings.md`)

**There is no UNC-103-related F-finding in the catalog.** The "GLOBAL→per-cell
state" concern in F2 is actually about caintra1's `GLOBAL calcium` ASSIGNED
variable, which couples the caintra1 mechanism to channels that read it.
This is a wholly different concern than per-cell vs global state of a
voltage-gated K channel.

**UNC-103's actual NMODL (`unc103.mod` in the Nicoletti tree):**
```
NEURON {
    SUFFIX unc103
    USEION k READ ek WRITE ik
    RANGE gbar,g,curr
}

PARAMETER {
   v (mV)
   ek (mV)
   celsius (degC)
   gbar=2.9 (S/cm2)
   ...
}

STATE {
    m h
}
```

No `GLOBAL` declarations. STATE is `m h` (per-cell — RANGE is the implicit
default for STATE in NMODL when not declared GLOBAL). UNC-103 follows the
exact same translation pattern as SHK-1, SHL-1, KQT-3 (all already done in
Phase β). There is no special GLOBAL→per-cell handling needed.

---

## What this implies

If I proceed as the prompt is written:

1. **CP1 UNC-103 translation:** Mostly OK — the channel translation itself
   would be a straightforward voltage-gated K channel translation following
   the established pattern. The "F2 GLOBAL→per-cell" comment in the module
   would be misleading documentation but the resulting Brian2 module would
   be correct. UNC-103 would be needed for Phase F's AVAR work eventually
   (avar_unc103_patch already exists for the NEURON side, but Brian2 has
   no UNC-103 channel module yet). So translating it is justifiable on its
   own.

2. **CP2 IRK translation:** Correct and necessary. IRK is in AVAL's 4-channel
   set and Brian2 doesn't have it yet.

3. **CP3 AVA cell construction with 5-channel set:** **Wrong.** Building a
   Brian2 AVA cell with UNC-103 inserted does not match Nicoletti's AVAL.
   It would produce a synthetic cell with no biological referent — neither
   AVAL (4 channels) nor AVAR (5 channels with different geometry, different
   surface area, different parameter vector).

4. **CP4 Phase F re-evaluation:** **Doubly wrong.** Apples-to-apples NEURON
   reference with 5 channels would either:
   - (a) Use AVAL geometry + 5 channels (UNC-103 added) — not Nicoletti's
     AVAL, no biological referent, the "Nicoletti's actual AVAL phenotype"
     comparison is bogus.
   - (b) Use AVAR's parameter vector — but then the Brian2 cell needs to
     match AVAR's geometry (1121.79e-8 cm² vs 1123.84e-8 cm²) and the
     comparison is "Brian2 AVAR vs NEURON AVAR" which is a different
     project than the option α framing of "match Nicoletti's AVAL phenotype."

---

## Three resolution options

### Option α-1: Drop UNC-103 from CP3, build true 4-channel AVAL

**CP3:** Construct Brian2 AVAL with `[IRK, LEAK, EGL19, NCA]` matching
Nicoletti's published AVAL parameter vector (g_nca=0, g_egl19=0.104385 nS,
g_leak=0.150164 nS, g_irk=0.1 nS, surf=1123.84e-8 cm², cm=0.859551 μF/cm²,
e_leak=-39 mV).

**CP4:** Apples-to-apples Brian2 4-channel AVAL vs NEURON 4-channel AVAL.
NEURONReference already supports this via `custom_spec` (per Phase F's
2a — see `gate2_ava_cell_construction.md` line 10-13). Or we can directly
invoke Nicoletti's `AVAL_simulation_iclamp` for the 1000ms current-clamp
protocol since that's the published reference path.

**CP1 UNC-103 still done** — it's needed for AVAR work later, and translating
it now is cheap given the IRK translation is happening anyway. But its
output is not used in CP3 or CP4.

**Pros:**
- Matches Nicoletti's AVAL exactly. Apples-to-apples is meaningful.
- "Nicoletti's actual AVAL phenotype" target is biologically grounded.
- Consistent with the prior session's pushback.

**Cons:**
- Departs from the prompt's "5-channel" framing. Requires acknowledgment that
  the prompt was internally inconsistent.

### Option α-2: Switch target cell to AVAR

If the prompt's "5-channel" framing was deliberate (UNC-103 is in there for
a reason), the target cell should be AVAR, not AVAL. Build Brian2 AVAR with
`[IRK, LEAK, EGL19, NCA, UNC103]` matching AVAR's parameter vector + geometry.

**CP4:** Apples-to-apples Brian2 AVAR vs NEURON AVAR (via the existing
avar_unc103_patch). Phase F's 2a already passed for AVAL — this would be a
new validation track for AVAR.

**Pros:**
- Preserves the "5-channel set" framing.
- Exercises the avar_unc103_patch infrastructure.

**Cons:**
- Major scope shift: switching from AVAL to AVAR target. Not what the prompt's
  prose says ("re-ground to Nicoletti's actual AVAL phenotype").
- The pushback's "Nicoletti's actual AVAL phenotype" (slow-rising, sustained
  plateau, linear I-V) is described per AVAL recordings. AVAR is similar per
  Nicoletti 2024 but the experimental anchor is AVAL.
- Requires re-doing all the Brian2 cell construction infrastructure for AVAR
  (different surf, different parameter vector).

### Option α-3: Build both — AVAL 4-channel + AVAR 5-channel

Most ambitious. Build CP3a as Brian2 AVAL (4-channel) and CP3b as Brian2 AVAR
(5-channel). CP4 evaluates both against their respective NEURON references.

**Pros:**
- Comprehensive. Both AVA cells get the full Phase F treatment.
- UNC-103 translation gets exercised in a real cell context.

**Cons:**
- Roughly 2× the work. Likely exceeds session capacity if all 5 CPs are to
  complete cleanly.
- The prompt explicitly says "If you finish with capacity remaining, stop
  and surface for user discussion rather than expanding scope" — building
  both feels like scope expansion.

---

## Recommendation

**Option α-1** seems most consistent with both the prompt's prose framing
("re-ground to Nicoletti's actual AVAL phenotype") and the prior session's
`mellem_investigation_pushback.md` finding that AVAL has 4 channels.

The "5-channel" specification in the prompt's CP3 + CP4 is most likely a
prompt-author error: the author was probably reading AVAR's channel list
when writing the prompt, or conflating AVAR's channel list (5 incl. UNC-103)
with AVAL's. Either way, the actual *biological* target — Nicoletti's
"slow-rising phase, sustained plateau, linear I-V" AVA phenotype — is
characterized for AVAL with the 4-channel set.

Translating UNC-103 in CP1 is still valuable: it completes Brian2's Nicoletti
channel coverage for AVA neurons (both AVAL and AVAR), and it's a clean
voltage-gated K translation following the established Phase β pattern. The
F2-GLOBAL framing in the prompt should just be ignored — the actual
translation has no GLOBAL state issue.

**If user concurs with α-1:**
- CP1 UNC-103: translate as usual, voltage-gated K, no special GLOBAL handling
- CP2 IRK: translate as planned
- CP3 AVA cell: 4-channel `[IRK, LEAK, EGL19, NCA]` matching Nicoletti's AVAL
  (g_nca=0 per published vector; surf and cm from AVAL_simulation_iclamp.py)
- CP4: apples-to-apples Brian2 4-channel vs NEURON 4-channel via existing
  `custom_spec` infrastructure or direct upstream invocation
- CP5: outcome summary

**If user prefers α-2 or α-3:** different CP3/CP4 plan.

---

## Other concerns surfaced during pre-flight

### Concern: NCA in Nicoletti's AVAL has gbar=0

Per `AVAL_simulations.py` line 26: `g0=[0.104385, 0.150164, 0.1, 0, -39, 0.859551]`.
The 4th entry is g_nca = 0. Nicoletti inserts NCA into AVAL but assigns it
zero conductance. So functionally AVAL is 3 active channels (IRK, LEAK,
EGL19) plus an inserted-but-silent NCA.

**Implication for Brian2 translation:**
- Inserting NCA with gbar=0 is harmless (no current contribution).
- Decision: should Brian2 AVAL include NCA for fidelity (matches Nicoletti's
  insertion list) or omit it (functionally equivalent, simpler eqs)?
- Recommend: include NCA with gbar=0 for apples-to-apples fidelity. This
  matches what Nicoletti's NEURON cell does, even though it's a no-op
  numerically.

### Concern: NEURONReference custom mode for 4-channel may already exist

Per `gate2_ava_cell_construction.md`, the existing infrastructure built for
Phase F 2a uses NEURONReference with `custom_spec` to construct a NEURON
AVA cell with `[leak + EGL-19 + NCA]` (no IRK, no UNC-103) — a 3-channel
subset of AVAL. To extend this to "all 4 of Nicoletti's published AVAL
channels including IRK," NEURONReference would need to support either:

- IRK insertion in custom mode (likely already supported — check
  `neuron_reference.py` for `irk` in the channel-name dispatch)
- OR direct invocation of upstream `AVA_simulation_iclamp` (already does
  exactly the published 4-channel construction)

The simpler path is direct upstream invocation, since that IS the published
reference. Same approach used by `avar_unc103_patch.py` for AVAR. Decision:
should CP4 invoke upstream directly (cleanest, no extra infrastructure),
or extend NEURONReference (more flexible for future variants)?

Recommend: invoke upstream directly for the canonical 4-channel reference.
NEURONReference custom mode remains for ablation studies (subsets of
channels). This avoids the prompt's concern that "NEURONReference may
require extension for custom 5-channel AVA" — there's no 5-channel AVA
construction needed; AVAL is 4 channels and the published script already
implements it.

### Concern: 1000 ms protocol and 600 ms protocol numerical artifacts

Per the prompt: "Phase F's '100 ms × 50 pA' protocol was Mellem-legacy;
correct protocol per Nicoletti is **1000 ms** current-clamp duration."

`AVAL_simulation_iclamp.py` line 53-55 confirms:
```python
stim.delay=1023
stim.amp=10  # placeholder, overwritten
stim.dur=1000
```

Stim onset at 1023 ms, duration 1000 ms, simdur 2500 ms — i.e., post-stim
recovery for 477 ms. Current range scanned at line 69:
`numpy.linspace(start=s1, stop=s2, num=ns)` where `AVAL_simulations.py`
line 31 calls with s1=-0.03, s2=0.03, ns=7 → 7 levels from -30 pA to +30 pA.

Brian2 plateau_harness.py probably needs adjustment to match this protocol
(1000 ms instead of 100 ms duration). Will verify during CP4 setup.

### Concern: g_to_Scm2 conversion

Nicoletti's parameter vectors are in nS but channels expect S/cm². The
conversion is `g[i] * 1e-9 / surf` for the first N indices (where N is the
"index" arg to `gScm2`, e.g., 3 for AVAL meaning indices 0..3 are converted
and 4..5 pass through). Already documented in avar_unc103_patch.py and used
in Phase F 2a infrastructure. Just need to be careful about it.

---

## Status

**PAUSED.** Awaiting cross-session decision on α-1 / α-2 / α-3.

`wave2/artifacts/PAUSED_FOR_REVIEW.txt` is being updated to point at this
new pushback (the existing marker pointed at the prior Mellem investigation,
which has been resolved by the option α directive — but option α as prompted
has its own internal inconsistency that needs resolution).

If user reads this and concurs with α-1, work resumes from CP1 with the
4-channel target for CP3.

If user concurs with α-1 but wants me to also translate UNC-103 in CP1
even though it's not used in CP3/CP4 (still valuable for future AVAR work):
this is the recommended path and matches the prompt's CP1 directive on
its own merits.

If the prompt was deliberate about the 5-channel set (e.g., user wants to
explore whether adding UNC-103 to AVAL makes a substantive difference in
Phase F outcomes — a synthetic-AVA experiment): user should explicitly
authorize this scope, since it departs from "match Nicoletti's actual AVAL
phenotype" toward "explore a non-Nicoletti hybrid AVA cell."

---

## Files referenced in this pushback

- `wave2/artifacts/mellem_investigation_pushback.md` (prior pushback that
  established 4-channel AVAL)
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAL_simulation_iclamp.py`
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAL_simulation_vclamp.py`
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAL_simulations.py`
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAR_simulation.py`
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAR_simulation_vclamp.py`
- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/unc103.mod`
- `wave2/avar_unc103_patch.py`
- `wave2/translation_patterns.md` (F1-F13 catalog)
- `wave2/artifacts/phase_beta_findings.md` (F1-F15 with detail)
- `wave2/artifacts/gate2_ava_cell_construction.md` (Phase F 2a infrastructure)
