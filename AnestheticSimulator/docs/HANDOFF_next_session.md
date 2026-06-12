# AnestheticSimulator — handoff to next session (as of 2026-06-12)

Read first, with [`SESSION_CLOSEOUT_2026-06-12.md`](SESSION_CLOSEOUT_2026-06-12.md) (results) and
[`wf2_gated_fix_roadmap.md`](wf2_gated_fix_roadmap.md) (the full plan). Memory:
`project_anesthesia_gated_roadmap.md`, `project_anesthesia_strengthening_verdict.md`.

## State of the program
Foundation **certified rank-2** (P18). Phase-1 + keystone **all run** with preregistered accept-either-way gates:
- **P8** Sub-Q1 trichotomy = control-bug artifact → worm 28% / fly 26% / mouse 38% (worm≈fly modest-sig, mouse marginal).
- **P3** mean-field theorem (mouse-at-median derived). **P11** occupancy saturation was a Kp artifact (engagement robust).
- **P1_P2** keystone rank-lift → worm Match#3 **NULL** (cell-type targeting not established even on the lifted substrate).
- **P7** "one free α" narrowed. **P4** Gate-4 uninterpretable/retained. **P17** behavioral readout **DEMOTED** (Paper-2 blocked).
  **SOL28** nca quorum robust (PASS). **P20** 3 genotypes route to V2/Tier4 (constructive positive).
- One **fabrication caught** (fly Match#3 relabel) before shipping; guarded.

## Running / queued right now
- **SOL27 NCA-1 fold** — running locally on CPU (PID 477025, ~21 GB RAM; PID 443396 queue/watcher). Greene NOT needed
  (the OOM was model-load, not length; 47 GB RAM lifts it). When done: `artifacts/p13_sol27_local/sol27_local_verdict.json`
  + `NCA-1_Q9N4D6_ESMFold.pdb`. **CAVEAT:** Q9N4D6 is only **474 aa** — far short of full-length NALCN/NCA-1 (~1600 aa);
  it is a fragment/isoform. Verify the accession / consider the full-length sequence before docking. NCA-2 (G5EDM1) is
  already folded as a paralog proxy. **COMMIT the fold output when it lands.**

## Next actions (priority order)
1. **Collect + commit the SOL27 NCA-1 fold output** (running). Re-run `phase_a_esmfold_local.py unc80` for UNC-80
   if wanted (CPU, slow, low value — ~3000-aa IDR scaffold, poor ESMFold confidence).
2. **Write the P16 heavy scoring loop** — the only genuinely incomplete module. Fast gates PASS (strict
   degree-preserving shuffle invariants, NWB parse, positive control); the heavy `__main__` is a stub. Needs:
   (a) the scoring loop, (b) a NON-DESTRUCTIVE spontaneous spike-export path (read `brain.spikes`; do NOT modify
   `run_single`/`compute_metrics`), (c) Gate-B's two glu-sign arms via a custom brain_factory. Then run (~overnight,
   ml env). Bayesian-likely NULL (connectome≈shuffled prior). `src/state_validation/p16_structure_activity_exam.py`.
3. **Re-ground the behavioral readout (P17 follow-up)** before any Paper-2 work — the immobilization readout is not
   held-out-validated. Either re-validate against a different held-out target or restrict claims to network-state.
4. **Regenerate Wave-P occupancy magnitudes under the corrected frame (P11)** — the `occupancy_matrix.csv` saturation
   values are Kp-inflated; recompute under M0 (no Kp) or report an M0/M2 bracket.
5. **React `AnesthesiaPipeline.tsx`** — surface the corrected Sub-Q1 / close-out numbers (still shows V7-era).
6. **Fly Match#3** — the one data-gated payoff: needs a *Drosophila* cell-type-expression atlas (new data acquisition).

## Repo gotchas (important)
- The website repo (`/mnt/ssd4tb/Desktop/website/personalwebsite`) ALSO holds the **parallel C. elegans Tier4 session's
  uncommitted work**: `scripts/brain/*` (untracked), `scripts/brain/artifacts/modulator_tables.json` (modified).
  **Do NOT commit those** — only `AnestheticSimulator/` paths + the anesthesia page. Use scoped `git add AnestheticSimulator/`.
- The live page is `src/content/projects/anesthesia-pipeline.mdx` — rewritten this session to the corrected findings
  (Adversarial close-out section is authoritative; V5–V7 sections kept as history, marked superseded where they conflict).
- Env: brian2 work = `~/miniconda3/envs/ml/bin/python`; pure-numpy/arithmetic = `python3`. Held-out NWB ~25-27 GB each →
  h5py streaming, never resident.
- Pushing `main` deploys the live site (Astro/Vercel, rohitravi.com).
