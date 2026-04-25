# New-session primer — paste this into the first turn

Copy everything between the `---` lines into your first message. It
is designed to prime a fresh Claude session with full context on
the C. elegans simulator project and pick up exactly where the
2026-04-25 session left off (Phase 0 complete; T0 cascade-failure
question resolved at the architectural level by per-edge sign
convention; PVC/AVB handling and FSM/classifier recalibration
under per-edge are the two open follow-on questions).

**Updated 2026-04-25 (post-T0 resolution).** Earlier version
(2026-04-25 morning) framed T0 as still-open with voltage fix as
the load-bearing finding; that framing is superseded — see
`docs/t0_resolution_report.md` for the full diagnostic record.

---

I'm resuming work on my C. elegans multi-modal simulator project.
Before doing anything, please read these documents in order so you
have full project context:

**In-repo docs (read first — they reflect the post-T0-resolution
state as of 2026-04-25):**

  1. `docs/t0_resolution_report.md` — **READ FIRST.** Canonical
     record of the 2026-04-25 T0 diagnostic block. The simulator's
     current state hinges on the findings here.
  2. `docs/current-state-summary.md` — concise current-state
     snapshot with pending decisions list.
  3. `docs/claude-chat-context.md` — full architecture + validation
     state. §3, §4, §5, §6 reflect post-T0-resolution framing.
  4. `scripts/brain/artifacts/t0_run_report.md` — historical April
     21 T0 framing with dated postscript referencing the
     resolution. Useful for understanding what was previously
     believed and how it changed.
  5. `docs/citation-audit-checklist.md` — citation hygiene work
     items.
  6. `docs/audit-strategy.md` — three-layer audit framework + three
     readout failure modes (framework still valid; AVA-specific
     entries annotated for the per-edge resolution).
  7. `docs/tier2-4-execution-plan.md` — Phase 3 has a dated
     supersession note; rest of execution plan still stands.

**Auto-memory files (supplementary context; some entries predate
the T0 resolution — defer to the in-repo docs above when they
disagree):**

  - `project_celegans_session_resume` — orientation snapshot.
  - `project_celegans_phase0_complete` — Phase 0 + overnight
    runs (three-mode taxonomy across 9 modulators).
  - `ref_celegans_audit_strategy` — pointer to docs/audit-strategy.md.
  - `ref_celegans_citation_corrections` — Gao-Hobert→Mellem 2008,
    Rogers 2003→Cohen 2009 etc.
  - `project_celegans_classifier_gap` — original T0 finding
    (superseded; the classifier-mediated framing was correct
    in direction but the mechanism was a sign-convention bug,
    not "the classifier learned distributed pattern").
  - `project_celegans_tier2_plan` — pre-T0-resolution plan
    (reorganized into Block 1/2/3/4 in claude-chat-context.md §6).
  - `ref_celegans_known_bugs` — SSR fail, defensive indexing,
    Brian2 units, ActivityFSM gotcha.
  - `feedback_audit_before_advance` — seed-locked n≥3 audit rule.

Project facts to know before we talk:

1. **T0 cascade-failure question is RESOLVED at the architectural
   level.** The simulator's failure to propagate touch through to
   AVA was caused by the default per-presynaptic-neuron Glu = −1
   sign convention treating glutamate→iGluR-dominant edges as
   inhibitory. Per-edge sign convention (constructor flag
   `use_per_edge_glu_signs=True`, already in the codebase but off
   by default) makes the cascade fire: AVDL/R Δ +60 Hz on touch,
   AVAL/R Δ +60 Hz, AVEL/R Δ +47 Hz, n=10 with seed-to-seed
   variance under 1.5 Hz. Full record: `docs/t0_resolution_report.md`.

2. **The operative cascade is ALM/AVM → PVC → AVD/AVE → AVA**,
   not the long-assumed ALM → AIB → AVA. AIB has zero chemical
   edges to AVD in this connectome and is not in the touch-
   reversal pathway. PVC is the load-bearing first-stage relay
   (ALM/AVM glutamate → PVC, flips excitatory under per-edge
   because PVC is iGluR-dominant); PVC then drives AVD/AVE
   cholinergically (~5× more drive to AVD than direct ALM/AVM
   glutamate). Recurrent positive-feedback in the command pair
   amplifies all command neurons to ~97 Hz coherently.

3. **Phase 0 is COMPLETE as of 2026-04-22.** Three-mode taxonomy
   validated across all 9 v3 modulators (under default sign
   convention; Mode classifications would need re-running under
   per-edge):
   - Mode 1 (readout-blind): 5 — FLP-11, FLP-1, NLP-12, TA, OA
   - Mode 2 (readout-trivial): 2 — 5HT (NSM in readout), DA (CEPDL in readout)
   - Mode 3 (readout-cascade-via-tonic-shift): 2 — FLP-2 (AIA+RID), PDF-1 (AVB)
   The taxonomy as a methodological framework is sign-mode-
   independent; the per-modulator classifications above are
   conditional on default mode.

4. **The April 21 phenotype reproduction was a sign-convention
   artifact on the dREV channel.** AVA-ablation under default
   produced ΔREV = −0.49 (10/10 negative seeds at n=10 × 60s).
   Under per-edge mode (with cascade firing), ΔREV regresses to
   +0.04 (2/10 negative) — but **dPIR retains a clean AVA-
   ablation effect** (mean −0.117, 9/10 negative). The Chalfie
   1985 phenotype reproduction in this simulator is currently
   channel-dependent in a way the original audit did not
   anticipate. FSM/classifier recalibration under per-edge is an
   open question.

5. **Voltage fix is in place but is a no-op for LIF dynamics.**
   Patched `v_rest=−25, v_thr=−10, v_reset=−30` per Mellem 2008
   (PMID 18587393). Preserves the 15 mV rest-to-threshold gap so
   LIF dynamics are coordinate-translated unchanged. Kept in
   place for biological documentation; will matter when SK/BK and
   compartmental work starts. Plateau dynamics (the original
   reason for the voltage diagnostic) are a separate active-
   conductance question that per-edge sign mode does not address.

6. **Citation corrections to apply across docs:**
   - "Gao & Hobert 2020" doesn't exist → use Mellem et al. 2008
     (PMID 18587393)
   - "Rogers 2003" for FLP-18 → use Cohen et al. 2009 (PMID 19356718).
     Rogers 2003 is actually about FLP-21.
   - Nelson 2013 NLP-22 sleep paper IS real (PMID 24301180) — but
     CeNGEN shows zero NLP-22 expression in RIA. Genuine data-method
     mismatch.

7. **T4-5 candidate scope refinement (post-overnight v1):**
   Locked: FLP-13 (Nath 2016), FLP-18 (Cohen 2009), FLP-21 (de Bono
   1998), NLP-40 (Wang 2013), DAF-28 (Li 2003).
   **WARNING:** FLP-13 has Jaccard overlap 0.80 with FLP-11 — likely
   redundant. Reconsider before adding.
   Dropped (A1 expression failed): INS-1, INS-7, INS-22, NLP-22 (zero),
   NLP-24, NLP-29.
   Confirmed database artifacts: nssp-29, Y51H7C.3 (R04A9.1 is real
   but marginal).

8. **RIS is silenced under per-edge mode.** RIS goes from 21.8 Hz
   tonic (default) to 0.8 Hz (per-edge) — a network-equilibrium
   consequence, not a direct sign flip. The April 21 RIS molecular
   audit (FLP-11 release fires correctly, peptidergic targets show
   ~22% disinhibition) was conducted under default mode and does
   not transfer to per-edge without re-running.

9. **Working environment:**
   - Brain env: `/home/rohit/miniconda3/envs/ml/bin/python` (Brian2 2.9)
   - Project repo: `~/Desktop/website/personalwebsite/` (auto-deploys to Vercel)
   - Live site: rohitravi.com/projects/c-elegans-multimodal (P0+P1 shipped 2026-04-21)
   - Compute ratio: ~1.66× wall/simulated under post-volt config
     (was 3.06× in earlier sessions; difference likely environmental)
   - Audit harness: `phase0_audit.py` with `--quick` / default /
     `--audit-long` tiers, plus `--g-gap-ns` and `--use-per-edge-glu`
     CLI flags added during 2026-04-25 work
   - Comparison harness: `phase0_postvolt_compare.py` reads sweep
     CSVs and produces side-by-side cascade + phenotype tables

10. **Before recommending any action,** verify:
    - Current git HEAD (`git log --oneline | head -10`)
    - Whether the live site has changed since 2026-04-21 (`curl -s
      https://www.rohitravi.com/projects/c-elegans-multimodal/`)
    - Read `docs/current-state-summary.md` for the current pending-
      decisions list before assuming any specific work is "next"

11. **Working style (hard constraints):**
    - Plan-first for non-trivial work; ask before executing big refactors
    - No-sugarcoat assessments; honest scope labels
    - Push back on speculative proposals before elaborating
    - Falsifiability required for cross-domain claims
    - No wet-lab bio suggestions (theoretical + computational only)
    - Prefer seed-locked ensemble audits (n≥3) over single-run claims
    - Single-change discipline for diagnostic work — one variable
      per test so effects are attributable
    - Mechanism-not-phenotype as stopping criterion — phenotype
      reproduction is necessary but not sufficient
    - Direct measurement settles disputes between sessions
    - Use Conventional Commits prefixes
    - Don't run Layer B audits for modulator claims; use Layer A
      (molecular) instead per the audit strategy

12. **Pending decisions (post-T0 resolution; see
    docs/current-state-summary.md for current list):**
    - **PVC/AVB handling under per-edge mode.** PVC fires
      Δ +60-70 Hz on touch under per-edge — biologically
      questionable. Two interpretations open: CeNGEN expression-
      vs-function mismatch, or canonical biology more nuanced
      than textbook. Neither yet falsified.
    - **FSM/classifier recalibration under per-edge dynamics.**
      Refined by dPIR finding: question is broader than just
      "retrain to recover dREV" — characterize what behavioral
      signature AVA-ablation produces under correct cascade and
      which FSM channels best reflect the Chalfie phenotype.
    - **Whether per-edge becomes production default or stays
      opt-in.** Depends on resolving above two.
    - **RIS molecular audit re-run under per-edge mode.** RIS
      silenced at 0.8 Hz; April 21 audit transferability unknown.
    - **Carried forward as contingent:** LIF-vs-graded head-to-
      head, citation corrections across docs, FLP-13 reconsider
      (0.80 Jaccard with FLP-11), NLP-22 verification.

13. **Known gotchas to watch for** (`ref_celegans_known_bugs`):
    - `astro dev` SSR fail on C. elegans page — use `astro build`
      + static serve
    - Time-indexed array lookups need length+NaN guards (Zen
      browser race)
    - `Environment.inject_into_brain()` needs peak>0 guard
    - Brian2 quantities need explicit unit imports (pA, mV, etc.)
    - **`phase0_analyze.py` overwrites `current-state-summary.md`
      from scratch** — DO NOT re-run it without backup
    - ActivityFSM diagnostics-collapse on v3 LIF was the original
      T0 framing (see `scripts/brain/artifacts/t0_run_report.md`);
      now resolved at the architectural level by per-edge sign
      convention

**My priority for this session:** [FILL IN HERE — pick one]

  A) PVC/AVB handling under per-edge mode (literature dive +
     possibly per-edge override sweep; two interpretations to
     adjudicate)
  B) FSM/classifier recalibration under per-edge dynamics (bank
     retraining; engineering lift; deferred during overnight v2
     Track B as LOGISTICAL_FAILURE)
  C) Network-stability scan under per-edge mode for non-touch
     scenarios (osmotic_shock, food, chemotaxis, aerotaxis,
     spontaneous)
  D) RIS silencing investigation + RIS molecular audit re-run
     under per-edge
  E) Re-run audited phenotypes (three-mode taxonomy, Mode 3
     modulators) under per-edge to determine which findings
     transfer
  F) Production sign-mode decision (per-edge as default, opt-in,
     or hybrid with curated overrides)
  G) Apply citation corrections across docs (independent of T0
     resolution)
  H) Refine T4-5 scope (FLP-13 redundancy + NLP-22 anomaly;
     independent of T0 resolution)
  I) Compartmental integration + plateau dynamics (independent
     question; per-edge sign mode does not address active
     conductances)
  J) Something else: [describe]

Once you've read the memory files and verified the current git state,
confirm you have the context, state your understanding of what I just
asked for in one paragraph, list any specific file paths or assumptions
you need to verify before starting, and wait for my go-ahead before
making any code changes.

---

## Tips for using this primer

- **Edit the priority line** (the `[FILL IN HERE]` placeholder) to
  match what you actually want to work on.
- **Expect a read-back from Claude** before any code changes — by
  design, per the "plan first" working-style rule.
- **The primer points at the in-repo docs first.** Auto-memory is
  supplementary. When memory and in-repo docs disagree, the in-
  repo docs (especially `docs/t0_resolution_report.md` and
  `docs/current-state-summary.md`) are authoritative.
- **Keep this file in sync.** When PVC/AVB resolves, FSM is
  recalibrated, per-edge becomes default, etc. — update this
  file so future sessions see the new state. Update timestamp at
  the top.

## Changelog

- **2026-04-25 (afternoon, post-T0 resolution):** rewritten to
  reflect T0 resolution work block. Operative cascade corrected
  to ALM/AVM → PVC → AVD/AVE → AVA. Per-edge sign convention
  identified as the architectural fix; voltage fix re-described
  as no-op for LIF (kept for biological documentation). April 21
  phenotype reproduction described as sign-convention artifact
  on dREV with dPIR persistence under per-edge. Pending decisions
  list re-anchored on PVC/AVB handling, FSM recalibration, and
  sign-mode default. Reading-order section re-pointed at in-repo
  docs first.
- **2026-04-25 (morning):** rewritten post-Phase-0 +
  post-overnight-v1 + post-overnight-v2. Added three-mode
  taxonomy as central methodological contribution, voltage-scale
  finding, citation corrections, refined T4-5 scope, Phase 2
  kickoff decision. **Superseded by afternoon revision** —
  voltage fix turned out to be a no-op for LIF, and the
  classifier-mediated phenotype framing was a sign-convention
  bug, not a "the classifier learned distributed pattern"
  abstract finding.
- 2026-04-21: original primer at end of Phase 0 W0 baseline
  measurement pass.
