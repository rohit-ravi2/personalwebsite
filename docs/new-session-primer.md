# New-session primer — paste this into the first turn

Copy everything between the `---` lines into your first message. It
is designed to prime a fresh Claude session with full context on
the C. elegans simulator project and pick up exactly where the
2026-04-25 session left off (Phase 0 complete, two overnight
extension runs complete, Tier 2 not yet started, Phase 2 kickoff
pending).

---

I'm resuming work on my C. elegans multi-modal simulator project.
Before doing anything, please read these auto-memory files in order
so you have full project context:

  1. `project_celegans_session_resume` — **READ FIRST.** Single-doc
     orientation as of 2026-04-25. Captures Phase 0 close-out, two
     overnight run results, current state, pending decisions.
  2. `project_celegans_phase0_complete` — empirical Phase 0 +
     overnight runs (three-mode taxonomy across all 9 modulators)
  3. `ref_celegans_audit_strategy` — three-layer audit framework +
     three readout failure modes
  4. `ref_celegans_citation_corrections` — Gao-Hobert→Mellem 2008,
     Rogers 2003→Cohen 2009 etc.
  5. `project_celegans_classifier_gap` — T0 finding (now refined to
     Mode 3 cascade-via-tonic-shift)
  6. `project_celegans_tier2_plan` — concrete starting points (note:
     refined post-Phase-0; see session_resume for current status)
  7. `ref_celegans_known_bugs` — SSR fail, defensive indexing,
     Brian2 units, ActivityFSM gotcha
  8. `feedback_audit_before_advance` — seed-locked n≥3 audit rule

After reading those memories, also check:
  - `docs/current-state-summary.md` (in repo)
  - `docs/audit-strategy.md` (in repo, locked methodology)
  - `docs/tier2-4-execution-plan.md` (in repo)
  - `docs/citation-audit-checklist.md` (in repo)

Project facts to know before we talk:

1. **Phase 0 is COMPLETE as of 2026-04-22.** Tier 2/4 phases have
   NOT YET STARTED. Phase 2 kickoff requires user decision on
   LIF-vs-graded brain-path comparison.

2. **The three-mode failure-mode taxonomy is the paper's central
   methodological contribution** and is now empirically validated
   across all 9 v3 modulators:
   - Mode 1 (readout-blind): 5 — FLP-11, FLP-1, NLP-12, TA, OA
   - Mode 2 (readout-trivial): 2 — 5HT (NSM in readout), DA (CEPDL in readout)
   - Mode 3 (readout-cascade-via-tonic-shift): 2 — FLP-2 (AIA+RID), PDF-1 (AVB)

3. **The voltage-scale finding is load-bearing for Phase 2.**
   Compartmental scaffold uses `v_rest = -65 mV` (mammalian-cortical
   template). Mellem 2008 (PMID 18587393) reports AVA at −20 to −30 mV.
   40 mV template error explains the plateau failure (Probe 2 of
   Phase 0 plateau diagnostic). LIF brain has the same template
   issue (v_rest=−65, v_thr=−50). Phase 2 needs whole-brain voltage
   regime correction, not just compartmental scaffold tweak.

4. **The fix for the plateau is two-part** (overnight v2 Track F
   confirmed minimal model insufficient):
   - v_rest correction across roster (−65 → −25 mV for command
     interneurons per Mellem 2008)
   - Add Ca-activated K⁺ (SK/BK from SLO-1/SLO-2) for plateau
     termination — current h-inactivation reaches `h_ss = 0.231`,
     too permissive

5. **Citation corrections to apply across docs:**
   - "Gao & Hobert 2020" doesn't exist → use Mellem et al. 2008
     (PMID 18587393)
   - "Rogers 2003" for FLP-18 → use Cohen et al. 2009 (PMID 19356718).
     Rogers 2003 is actually about FLP-21.
   - Nelson 2013 NLP-22 sleep paper IS real (PMID 24301180) — but
     CeNGEN shows zero NLP-22 expression in RIA. Genuine data-method
     mismatch.

6. **T4-5 candidate scope refinement (post-overnight v1):**
   Locked: FLP-13 (Nath 2016), FLP-18 (Cohen 2009), FLP-21 (de Bono
   1998), NLP-40 (Wang 2013), DAF-28 (Li 2003).
   **WARNING:** FLP-13 has Jaccard overlap 0.80 with FLP-11 — likely
   redundant. Reconsider before adding.
   Dropped (A1 expression failed): INS-1, INS-7, INS-22, NLP-22 (zero),
   NLP-24, NLP-29.
   Confirmed database artifacts: nssp-29, Y51H7C.3 (R04A9.1 is real
   but marginal).

7. **Mode 1 robustness confirmed** (overnight v2 Track A): all 4 of
   FLP-1, NLP-12, TA, OA PASS_MODE_1 at n=5 × 60s. BUT 2 of those
   (FLP-1, OA) are MECHANISM_INERT per Track C2 — release barely fires.
   The "Mode 1" label conflates "operating but invisible" with "not
   operating at all." Worth refining for paper clarity.

8. **Working environment:**
   - Brain env: `/home/rohit/miniconda3/envs/ml/bin/python` (Brian2 2.9)
   - Project repo: `~/Desktop/website/personalwebsite/` (auto-deploys to Vercel)
   - Live site: rohitravi.com/projects/c-elegans-multimodal (P0+P1 shipped 2026-04-21)
   - Compute ratio: 3.06× wall/simulated for v3 LIF brain
   - Audit harness: `phase0_audit.py` with `--quick` / default / `--audit-long` tiers

9. **Before recommending any action,** verify:
   - Current git HEAD (`git log --oneline | head -10`)
   - Whether overnight outputs in `scripts/brain/artifacts/overnight_*/` are committed
   - Whether the live site has changed since 2026-04-21 (`curl -s https://www.rohitravi.com/projects/c-elegans-multimodal/ -o /tmp/probe.html`)

10. **Working style (hard constraints):**
   - Plan-first for non-trivial work; ask before executing big refactors
   - No-sugarcoat assessments; honest scope labels
   - Push back on speculative proposals before elaborating
   - Falsifiability required for cross-domain claims
   - No wet-lab bio suggestions (theoretical + computational only)
   - Prefer seed-locked ensemble audits (n≥3) over single-run claims
   - Use Conventional Commits prefixes — Tier 2 work as `feat(T2-#N): …`
   - Don't run Layer B audits for modulator claims; use Layer A (molecular)
     instead per the audit strategy

11. **Pending decisions for user (Phase 2 kickoff):**
   - LIF-vs-graded brain-path head-to-head decision (3-4 weeks realistic)
   - Apply citation corrections across project docs
   - Reconsider FLP-13 in T4-5 given 0.80 Jaccard with FLP-11
   - Verify NLP-22 inclusion (Nelson 2013 paper real, CeNGEN expression absent)
   - Track B retraining (overnight v2 LOGISTICAL_FAILURE) — engineering lift
   - Commit overnight outputs to repo (still uncommitted as of 2026-04-25)

12. **Known gotchas to watch for** (`ref_celegans_known_bugs`):
   - `astro dev` SSR fail on C. elegans page — use `astro build` + static serve
   - Time-indexed array lookups need length+NaN guards (Zen browser race)
   - `Environment.inject_into_brain()` needs peak>0 guard
   - Brian2 quantities need explicit unit imports (pA, mV, etc.)
   - **`phase0_analyze.py` overwrites `current-state-summary.md` from scratch** — DO NOT re-run it without backup
   - ActivityFSM diagnostics-collapse on v3 LIF is the T0 finding, NOT a bug

**My priority for this session:** [FILL IN HERE — pick one]

  A) Phase 2 kickoff — LIF voltage rework + decision framework
  B) Phase 2 kickoff — graded compartmental integration
  C) Address pending citation corrections + commit overnight outputs
  D) Refine T4-5 scope (FLP-13 redundancy + NLP-22 anomaly)
  E) Track B retraining (empirically confirm Mode prediction under alt readouts)
  F) Something else: [describe]

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
- **The primer assumes memory is intact.** If memory is wiped, the
  fallback is `docs/current-state-summary.md` + `docs/audit-strategy.md`
  + reading the overnight MORNING_BRIEF.md files.
- **Keep this file in sync.** When Phase 2 starts, voltages get
  patched, T4-5 ships, etc. — update this file so future sessions
  see the new state. Update timestamp at the top.

## Changelog

- **2026-04-25:** rewritten post-Phase-0 + post-overnight-v1 + post-overnight-v2.
  Adds three-mode taxonomy as central methodological contribution, voltage-scale
  finding, citation corrections, refined T4-5 scope, Phase 2 kickoff decision.
- 2026-04-21: original primer at end of Phase 0 W0 baseline measurement pass.
