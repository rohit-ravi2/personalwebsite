# P1_P2 dated amendment — fly dropped (no Drosophila atlas); worm-only keystone

**Date:** 2026-06-12 · **Authority:** orchestrator review after the close-out workflow's adversarial verifier
flagged a fake-pass on the fly arm.

## The defect caught (and fixed)

`match3_ensemble(org)` loaded the single worm 300-neuron `x_c.json` regardless of `org`, and PR is
scale-invariant, so `match3_ensemble('fly')` silently produced the **worm** result relabeled `"fly"` —
PR identical to the worm value to 15 digits. Because fly is the a-priori payoff organism (WF1 said class
identity does measurable work there), this would have recorded a **fabricated positive on the very organism
the keystone was meant to test.** The fabricated `artifacts/p1_p2/match3_fly.json` has been DELETED and a hard
`NotImplementedError` guard added to `match3_ensemble` for `org != 'worm'`.

## Root cause: no Drosophila cell-type-expression atlas

The per-class expression vectors `x_c` are built from the **C. elegans CeNGEN** single-cell atlas (300 neurons).
There is no equivalent Drosophila-larva (2952-neuron Winding-2023) cell-type-expression atlas wired into the
project. So the cell-type-resolved Match#3 test **cannot be run on fly** without first acquiring/constructing a
fly expression atlas — that is genuine new data acquisition, out of this session's scope. The roadmap's
"worm(300)+fly(2952)" heavy scope was over-specified; it is corrected here to **worm-only**.

## Scope correction

- Match#3 heavy scope: **worm only.** Fly is DEFERRED-DATA-ABSENT (needs a Drosophila expression atlas); mouse
  remains EXCLUDED (generic random graph has no cell types). This is an honest scope limit, not a result.
- The worm closed-form PR arm is run-ready and ran honestly (see `match3_worm.json`); no LIF ensemble is needed
  for the spatial PR statistic, so the earlier "~270 min LIF ensemble" runtime estimate was also wrong — it is
  minutes.

## Verified-honest worm result (kept)

- `G_BIT_IDENTITY` PASS (new operator path bit-identical to V1 at x_c=ones; max per-neuron err 3.55e-15 pA / 36
  cases) — frozen `apply_anesthetic` untouched, non-destructive.
- `G0_RANK_LIFT_REALIZED` PASS (per-neuron drive diverges 17× more under real x_c than under x_c=ones).
- `x_c` is real CeNGEN expression (7×300; complex_i/ii rows all-ones by biology as prereg'd; the other 5 classes
  ~0.22–0.33 per-neuron expression fraction — not all-ones).
- `G_SOL7_able_to_fail` PASS, with a documented prereg-statistic correction: η² is degenerate (x_c rows are
  class-mean-constant ⇒ η²≡1), caught by the able-to-fail screen; **PR** is the operative spread statistic (which
  G1 already used — no threshold change).
- **Worm Match#3: `G1_MATCH3_SPATIAL_SPECIAL = NULL_DEFLATE`** — conserved PR = 0.875 at the **46th percentile**
  of magnitude+SNARE-matched surrogates; `G2_MATCH3_NOT_ENTAILED` PASS (Var over surrogates 2.5e-3 > 1e-9).
