# Stage I — Literature scoping for Wave 2 expansion

**Date:** 2026-04-27/28 (overnight run)
**Spec reference:** `phase_v_w2_overnight_full_pipeline_prompt.md` Stage I
**Prerequisites:** orchestrator already located Mellem pushback doc, upstream
Nicoletti 2024 mod-file directory verified.

---

## Headline finding

**Stage I produces 4 strong candidates with all primary-source biology already
in-repo (Nicoletti 2024 upstream mod files).** No external paywalled literature
needed for Stage II expansion.

The four cells are: **AVAR, VA5, VD5, VB6**. Together with existing
AVAL/AIY/RIM, this brings Wave 2 cellular layer to 7 production-grade cells.

Touch cascade members (ALM, AVM, AIB, AVB) are documented separately because
they are NOT in Nicoletti 2024's corpus. Stage I.3 surfaces this as a known
constraint that affects Stage IV cascade-validation strategy but does NOT
block Stage II.

---

## CP I.1 — Candidate enumeration

### Source 1: Nicoletti 2024 (already in repo)

The upstream directory `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
contains complete NMODL `.mod` files and Python simulation drivers for **seven
cells**:

| Cell | Type | Status in Wave 2 |
|---|---|---|
| AVAL | command interneuron (reversal) | PRODUCTION-GRADE (existing) |
| AVAR | command interneuron (reversal) | **STAGE II CANDIDATE 1** |
| AIY (AIYL/AIYR) | sensory integration interneuron | PRODUCTION-GRADE (existing) |
| RIM (RIML/RIMR) | reversal-modulating interneuron | PRODUCTION-GRADE (existing) |
| VA5 | A-type motor neuron (backward) | **STAGE II CANDIDATE 2** |
| VD5 | D-type GABAergic motor neuron | **STAGE II CANDIDATE 3** |
| VB6 | B-type motor neuron (forward) | **STAGE II CANDIDATE 4** |

**Primary source:** Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G,
Filippi S (2024). "Biophysical modeling of the whole-cell dynamics of C. elegans
motor and interneurons families." *PLOS ONE* 19(3):e0298105.
DOI: `10.1371/journal.pone.0298105`. PMCID: PMC10980225.

**Verification basis:** the Q. Liu (Bargmann-trained electrophysiologist,
co-author of Nicoletti 2024) provided the patch-clamp recordings that anchor
the models. AVAL & AIY & RIM models from this paper are already
production-grade in Wave 2; same primary-source provenance applies to AVAR,
VA5, VB6, VD5.

### Per-cell channel inventory (verified via direct read of vclamp scripts)

| Cell | Channels (NEURON insert order) | Channels existing in Wave 2 | Novel channels |
|---|---|---|---|
| AVAR | irk, leak, egl19, nca, unc103 | irk, leak (inline), egl19, nca, unc103 (5/5) | **0** |
| VA5 | slo2egl19, slo2iso, egl19, irk, shk1, leak, nca, cadiff | egl19, irk, shk1, leak (inline), nca (5/8) | **3** (slo2egl19, slo2iso, cadiff) |
| VD5 | slo2egl19, slo2iso, egl19, cca1, irk, shk1, leak, nca, cadiff | egl19, cca1, irk, shk1, leak (inline), nca (6/9) | **3** (slo2egl19, slo2iso, cadiff) — same as VA5 |
| VB6 | slo2egl19, slo1egl19, slo2unc2, slo1unc2, slo2iso, slo1iso, egl19, unc2, cca1, irk, shk1, leak, nca, cadiff | slo1egl19 (slo1_egl19_coupled), slo1iso, egl19, unc2, cca1, irk, shk1, leak, nca (9/14) | **5** (slo2egl19, slo2iso, slo2unc2, slo1unc2, cadiff) |

**Key insight:** VA5 and VD5 share the same novel-channel set. Translating
slo2egl19 + slo2iso + cadiff once unlocks BOTH cells. VB6 needs only 2
additional translations (slo2unc2, slo1unc2) on top of those, both of which
likely follow patterns already established in slo1_egl19_coupled.py (a
calcium-coupled coupled K-channel translation idiom).

### Source 2: Nicoletti 2019 (RMD)

**Primary source:** Nicoletti M, Loppini A, Chiodo L, Folli V, Ruocco G,
Filippi S (2019). PLOS ONE. RMD biophysical model.

**Status:** mod files NOT in current upstream directory (only 2024 cells
present). RMD remains a F19 standing followup per cellular_validation_findings.

**Mellem 2008 ground truth:** Mellem JE, Brockie PJ, Madsen DM, Maricq AV
(2008). *Nat Neurosci* 11:865-867. PMC2697921. **Characterizes RMD plateau,
NOT AVA.** Direct quote: "we never observed action potentials in AVA (n=10)."
RMD shows regenerative plateau, ~63 mV depolarization peak from rest of -73 mV,
Ca-dependent (TTX-insensitive, NMDG+ replacement abolishes), KO-tested.

**Stage II RMD candidacy:** **DEFERRED.** Acquiring Nicoletti 2019 mod files
or re-deriving RMD parameters from Mellem 2008 KO data is an investigation
beyond the overnight envelope. Not in Stage II target list. Documented for
future Wave 3 work.

### Source 3: Touch cascade members (ALM, AVM, AIB, AVB)

These cells are NOT in Nicoletti 2024's corpus. Web-search literature scan
results (CP I.3) below.

---

## CP I.2 — Mellem 2008 reexamination (READ from existing pushback doc)

Per orchestrator pre-flight resolution #2, Mellem 2008 was already reexamined
by today's earlier work block. Findings at
`wave2/artifacts/mellem_investigation_pushback.md`. Direct ground truth:

- **Mellem 2008 = RMD**, not AVA. AVA recordings (n=10) showed graded responses
  only with NO action potentials.
- RMD plateau is Ca-dependent, "long-lasting" (no specific ms quantified),
  voltage relaxes to ~-10 mV from rest of ~-73 mV.
- RMD requires Nicoletti 2019 mod files (not available locally) for
  production-grade Wave 2 implementation — **DEFERRED to Wave 3**.

**Implication for Stage II target list:** RMD is not feasible this overnight.
Documented and proceed.

**Implication for Stage IV §5 falsification baseline:** Mellem 2008 cannot
serve as the AVA-plateau target. The §5 baseline needs to be re-grounded to
Nicoletti 2024's AVAL phenotype (graded passive RC-like response, "slow rise
~200 ms followed by stable plateau sustained until stimulus removed"). The
existing Wave 2 AVAL model already matches this; Stage IV's "test whether
expanded brain reproduces touch cascade where pure LIF cannot" is the relevant
test, not "match Mellem 2008 plateau in AVA."

---

## CP I.3 — Touch cascade prioritization (ALM/AVM/AIB/AVB scoping)

### ALM (anterior touch receptor — left/right)

**Primary VC/CC sources:**
- O'Hagan R, Chalfie M, Goodman MB (2005). "The MEC-4 DEG/ENaC channel of
  Caenorhabditis elegans touch receptor neurons transduces mechanical signals."
  *Nat Neurosci* 8:43-50. — First mechanoreceptor current recordings from PLM,
  later extended to ALM.
- Eastwood AL, Sanzeni A, Petzold BC, Park S, Vergassola M, Pruitt BL, Goodman
  MB (2015). "Tissue mechanics govern the rapidly adapting and symmetrical
  response to touch." *PNAS* 112(50):E6955-E6963.
- **Katta S, Sanzeni A, Das A, Vergassola M, Goodman MB (2019).** "Progressive
  recruitment of distal MEC-4 channels determines touch response strength in
  C. elegans." PMC6785734. — ALM-specific protocol, EPC-10 amplifier,
  -60 mV holding, 2.9 kHz filter, 10 kHz digitization, junction-potential
  corrected -14 mV.

**Channel inventory characterized:**
- MEC-4/MEC-10 DEG/ENaC mechanoreceptor channel (transduction)
- No comprehensive K+ channel sweep characterizing intrinsic membrane
  properties at Nicoletti's level of completeness. ALM electrophysiology is
  about the MRC (mechanoreceptor current), not the broader channel set.

**Verdict:** ALM has VC/CC primary literature, but **the recording focus is
mechanoreceptor transduction, not whole-cell biophysics**. Building a Wave 2
ALM cell would require either (a) using a generic LIF-like passive cell with
MEC-4 transduction layered on top — minimal added value over current LIF — or
(b) deriving an HH model from scratch using Goodman lab data, which is a
research project beyond the overnight envelope. **DEFERRED**.

### AVM (anterior ventral mechanoreceptor)

Same situation as ALM. AVM is one of the six TRNs (ALMR/L, PLMR/L, AVM, PVM)
characterized in the same Goodman lab works above. No comprehensive HH-style
biophysical model exists in the literature. **DEFERRED**.

### AIB (interneuron, AIBL/AIBR)

Web search returned NO direct VC/CC primary source for AIB specifically.
Only related work: AIY (different cell, characterized by Faumont/Lockery
2006 PMID 16554520) and circuit-level synaptic-current recordings via
post-synaptic patching (Mellem 2002).

**Verdict:** No published whole-cell biophysical characterization of AIB
sufficient for Wave 2 production-grade modeling. **DEFERRED**.

### AVB (forward command interneuron)

Web search returned:
- Shen et al. 2024 *Science Advances* — AVA-AVB tonic/phasic interaction;
  electrophysiology focused on relative activity/inhibition, not channel
  kinetics.
- Hierarchical inhibition circuits 2025 — in situ electrophysiology on
  PVP→AVB and DVC→AVA, again circuit-level not channel-level.
- Mellem 2002 — early AVB recordings via stimulating sensory neurons and
  recording postsynaptic currents.

**Verdict:** No comprehensive VC/CC channel-by-channel characterization of
AVB matching Nicoletti's protocols. **DEFERRED**.

### Touch cascade summary

ALM/AVM/AIB/AVB cellular Wave 2 modeling is **outside this overnight's scope**.
Nicoletti 2024's seven cells are the available corpus. Stage IV touch cascade
validation must therefore use:
- AVAL (Wave 2, anchor) + LIF-mode for ALM/AVM/AIB/AVB.
- This is acceptable per the spec — Stage IV's central test is whether the
  expanded brain (which includes Wave 2 AVAL with its EGL-19/NCA/IRK/LEAK
  channel set) propagates touch input to AVAL more biologically than pure LIF.
  ALM/AVM remain LIF in the LIF scaffold; the cascade hits AVAL via the
  connectome wiring.

---

## CP I.4 — Ranked candidate list

### Stage II target list (top 4)

| Rank | Cell | Primary source | Cycle time est. | Novel channels | Strategic value |
|---|---|---|---|---|---|
| 1 | **AVAR** | Nicoletti 2024 (mod files in repo) | 30-60 min | 0 | Bilateral pair to AVAL; immediate gain in connectome integration; smallest risk per minute |
| 2 | **VA5** | Nicoletti 2024 (mod files in repo) | 60-90 min | 3 (slo2egl19, slo2iso, cadiff) | A-type motor neuron (backward locomotion); first motor cell at production-grade; unlocks VD5 for free |
| 3 | **VD5** | Nicoletti 2024 (mod files in repo) | 30-45 min (after VA5) | 0 incremental | D-type GABAergic motor neuron; shares VA5's novel translations; cheap follow-up |
| 4 | **VB6** | Nicoletti 2024 (mod files in repo) | 60-90 min | 2 incremental (slo2unc2, slo1unc2) | B-type motor neuron (forward locomotion); second motor cell; touch cascade B-type connection |

**Total Stage II envelope estimate: 3-5 hours wall clock** (1-3 hour
single-invocation windows; multi-invocation continuation expected).

### Deferred candidates

| Cell | Reason for deferral | Future-wave route |
|---|---|---|
| RMD | Nicoletti 2019 mod files not in upstream; Mellem 2008 KO data alone insufficient for HH parameters | Acquire Nicoletti 2019 source code → Wave 3 |
| ALM, AVM | Goodman lab characterizes MRC transduction, not whole-cell biophysics | Generic LIF + MEC-4 transduction cascade (already in `sensory_transduction.py`) covers it |
| AIB | No primary VC/CC source | Wait for future characterization paper |
| AVB | Circuit-level only, no channel kinetics | Wait for future characterization paper |

### Methodology pre-flight cleared

- Each candidate has primary source already in repo (Nicoletti 2024 upstream)
- All NMODL files locally inspectable; no paywalled fetches needed
- Channel pattern catalog (translation_patterns.md F1-F18) covers most
  expected gotchas; cadiff pattern partially handled by `wave2/calcium_pool.py`
- No risk of Mellem-2008-style citation misattribution since channel inventory
  is read directly from the .mod files, not from text prose

---

## Stage I acceptance check

- [x] ≥4 strong candidates beyond AVAL/AIY/RIM: **YES (4: AVAR, VA5, VD5, VB6)**
- [x] Each candidate has primary source verified: **YES** (Nicoletti 2024
      upstream mod files directly inspected)
- [x] Touch cascade members prioritized if data exists: **N/A** (data does NOT
      exist for ALM/AVM/AIB/AVB at Nicoletti's level; documented as deferred
      with primary-source rationale)
- [x] Per-cell cycle-time estimates documented: **YES** (above table)

**Stage I PASSES.** No hard stops triggered.

---

## State after Stage I

- This document: `wave2/artifacts/literature_scoping_candidates.md`
- Status JSON: `wave2/artifacts/checkpoints/stage_I_status.json` (next file)
- Running findings: `wave2/artifacts/overnight_run_findings.md` (active)

**Proceed to Stage II.** Order: AVAR → VA5 → VD5 → VB6. AVAR's near-zero
novel-channel cost makes it the obvious first target.
