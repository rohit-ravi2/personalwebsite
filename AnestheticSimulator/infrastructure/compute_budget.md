# Wave P — Compute budget summary

**Status:** SCAFFOLDED. Estimates only.
**External spend:** **$0.** Wave P runs entirely on local hardware (RTX 4060 Ti) plus free-tier overflow.

---

## 1. Per-phase rollup (zero external spend path)

| Phase | Local GPU-h (4060 Ti) | Local CPU-h | Colab free-tier h (T4) | External USD | Wall-clock |
|---|---|---|---|---|---|
| A | 12 | 4 | 10 (overflow only, pentameric edge cases) | 0 | ~1 week |
| B | 30 | 5 | 8 (overflow only, large DiffDock receptors) | 0 | ~2 weeks |
| C | 0 | 5 | 0 | 0 | ~2 days |
| D | 120 | 4 | 0 | 0 | ~4 weeks |
| E | 0 | 10 | 0 | 0 | ~1 week |
| F | 0 | 5 | 0 | 0 | ~3 days |
| G | 80 | 4 | 0 | 0 | ~2 weeks |
| H | 0 | 8 | 0 | 0 | ~1 week |
| I (stretch) | 20 | 80 | 12 (JAX overflow if needed) | 0 | ~3-4 weeks |
| J (stretch) | 0 | 20 | 0 | 0 | ~1 week |
| **Total** | **~262 GPU-h local** | **~145** | **~30** | **$0** | **~6 months** |

Phase B GPU-h adjusted upward slightly to absorb GNINA workload that was previously routed to a cloud burst alongside FEP. Phase D GPU-h adjusted upward to ~120 to account for the kinetic-shift MD now running fully on local hardware (FEP top-10 cloud step removed; see Phase B preregistration appendix).

---

## 2. Local RTX 4060 Ti utilization plan

The 4060 Ti has 8 GB VRAM and ~22 TFLOPS FP32. Realistic monthly throughput at typical Wave P workload mix (mixed OpenMM MD + Brian2 + ML inference + structure prediction): ~120 GPU-hours/month allowing for cooling, downtime, and shared use with Wave 2.

**Throughput estimates (4060 Ti only, no cloud):**

- **ESMFold monomer prediction:** ~30-60 minutes per *C. elegans* sequence ≤ 700 aa. 25 monomers = ~15 GPU-hours total.
- **Boltz-1 / OpenFold pentamer prediction:** ~2-6 hours per pentamer with chunked attention; tight-but-fits on 8 GB. 6 pentameric Tier-1 targets = ~25 GPU-hours.
- **AutoDock Vina docking (GPU-accelerated path):** ~3-5 minutes per (target, anesthetic) pair at exh=32. 150 pairs = ~10 GPU-hours.
- **DiffDock per pair (truncated receptor):** ~5-10 minutes on 4060 Ti for a 30-Å-truncated receptor pocket. 150 pairs = ~20 GPU-hours.
- **GNINA rescoring:** ~1 minute per pair × 150 = ~3 GPU-hours.
- **OpenMM MD (50,000-atom system, POPC bilayer):** ~30-50 ns/day on 4060 Ti. 8 systems × ~100 ns each = ~120 GPU-hours.
- **Brian2 perturbation runs:** GPU-light; mostly CPU.

**Schedule:**

- Months 1-2: Phase A (~12 GPU-h) + Phase B (~30 GPU-h). Plenty of headroom. Use spare cycles for Phase D MD calibration.
- Months 3-4: Phase D MD = ~120 GPU-h spread over 2 months. Tight but feasible. Phase E/F prep run in CPU-light parallel.
- Month 5: Phase G = ~80 GPU-h. Largest single-month load. May spill into month 6 if needed.
- Month 6: Phase H + paper draft + optional Phase I/J.

---

## 3. Free-tier path (default)

The default path uses **Colab free tier (T4 GPU, ~12 hr/day session caps)** for overflow only. Total cumulative budget: ~30 hours across the program.

When free-tier Colab is invoked:

- Phase A pentameric edge cases where Boltz-1 / ESMFold / OpenFold all fail to fit locally (~10 hours).
- Phase B DiffDock on receptors where 30-Å truncation isn't sufficient (~8 hours).
- Phase I JAX overflow if local 4060 Ti is saturated by concurrent Phase G (~12 hours).

**Scheduling note:** ~30 hr cumulative on a 12-hr/day cap means **~3 calendar days of Colab time** total — easily absorbed across 6 months. Daily quotas may force pacing; budget 1.5× wall-clock buffer for quota waits.

No Colab Pro, no A100, no commercial-tier upgrade is needed for the canonical path.

---

## 4. Acceleration paths (DEFERRED — reverse only on explicit user direction)

The following are **not** part of the Wave P plan. They are documented here so that, if the user later reverses the no-external-spend commitment, the cost-benefit picture is on file:

- **Lambda Labs / AWS A100 cloud burst for FEP top-10 (Phase B/D).** ~$25-40/hr × ~50 hours per burst × 1-2 bursts = **$200-400 total**. Buys absolute ΔG_bind on top-10 hits via OpenMM/YANK FEP. Useful if absolute affinity (rather than relative ranking) becomes load-bearing. The canonical Wave P plan uses GNINA-derived relative Kd ranking, which is sufficient for multi-target occupancy framing — see `preregistration/phase_b_binding_pose.md` §13.
- **Colab Pro ($9.99/month).** Buys longer sessions and better GPUs (V100/A100 access). Useful only if free-tier session caps become a wall-clock bottleneck in Phase A. None of the canonical Phase A workload requires this.
- **Cloud GPU rental for full Complex I assembly (~45 subunits).** Would require >40 GB VRAM and is beyond consumer hardware. The canonical plan deliberately scopes Complex I down to single-subunit-per-anesthetic-site analysis (GAS-1 primary) per Morgan & Sedensky 1995. Full assembly is DEFERRED / SPECULATIVE.

These paths are **gated behind explicit user authorization**. None are required for Phase A — H to ship.

---

## 5. Storage budget

| Phase | Outputs | Size estimate |
|---|---|---|
| A | 25 × ~5 MB PDB + multimer outputs | ~500 MB |
| B | 150 × Vina pdbqt + DiffDock ensembles + GNINA SDF | ~5 GB |
| C | NPZ + CSV + heatmaps | ~50 MB |
| D | 8 MD trajectories @ ~10 GB each | ~80 GB (NOT git-tracked) |
| E | NPZ traces + module | ~100 MB |
| F | NPZ baselines | ~50 MB |
| G | 2,440 NPZ traces + aggregated | ~30 GB (NOT git-tracked) |
| H | Tables + reports | ~10 MB |
| I (stretch) | NPZ inverse occupancy + JAX | ~500 MB |
| J (stretch) | Signatures NPZ + UMAP | ~200 MB |

Total active storage: ~120 GB during peak (months 3-5). Most lives in `artifacts/traces/` and `artifacts/runs/` — not git-tracked. Periodic cleanup of intermediate MD trajectories is expected.

**Storage allocation:** the user has `/mnt/ssd4tb/` (4 TB SSD) already mounted. Wave P's ~120 GB peak fits with substantial headroom. **Cost: $0** (existing hardware). Allocation is a 5-minute verification check, not a procurement task.

---

## 6. Power / thermal considerations

The 4060 Ti at 100% utilization for 96+ hours of MD draws ~165 W and produces heat. Wave P does not run heavier than Wave 2's already-tested workloads. No concern.

---

## 7. Throttling rules

If multi-day Phase D MD is running and the user needs the GPU for other work:

- Brian2 production simulator runs (Wave 2 and notebook pipeline) take priority.
- Wave P MD pauses and resumes when GPU is free.
- Phase G runs are batchable into nights / weekends.

---

## 8. Cost summary

- Local hardware: $0 (already owned, RTX 4060 Ti 8 GB VRAM).
- Colab: $0 (free tier T4, overflow only, ~30 cumulative hours across program).
- Cloud bursts: **$0 (dropped from canonical plan).** Documented in §4 as a deferred enhancement only.
- Software: $0 (all open or free academic; load-bearing tools are MIT / Apache 2.0 / BSD).
- Storage: $0 (pre-existing `/mnt/ssd4tb/` 4 TB SSD; ~120 GB peak allocation).

**Total Wave P out-of-pocket cost: $0.**

If the user later authorizes FEP cloud spend on top-10 hits: $200-400 marginal. Not in the canonical path.
