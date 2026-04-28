# Wave P — Risk register

**Status:** SCAFFOLDED. Risks identified at kickoff. Updated per work block.

---

## Risk matrix

| # | Risk | Likelihood | Impact | Mitigation | Owner phase |
|---|---|---|---|---|---|
| R1 | AlphaFold-Multimer fails on 3+ Tier-1 pentameric receptors | Medium | Medium (open-source fallbacks available) | Open-source predictor ladder: ESMFold (MIT) → Boltz-1 (MIT) → OpenFold (Apache 2.0) → ColabFold free tier T4 → subunit-by-subunit pocket modeling. AF-Multimer is non-load-bearing in the revised plan. | A |
| R2 | Vina, DiffDock, GNINA disagree on > 60% of pose pairs | Medium | High (Phase B reliability) | Photolabel cross-check; GNINA cross-method-agreement Gate B.1.4; document uncertainty per pair; escalate to deferred FEP path only on explicit user authorization | B |
| R3 | MD-derived kinetic shifts diverge from literature by > 2× | Medium | High (Phase D calibration) | Mammalian-control MD on TREK-1/GABA-A/NALCN; if controls fail, fall back to literature-only | D |
| R4 | **Multi-target framing falsified at Gate C.1** (< 5 targets > 10% occupancy) | Low-Medium | Program-level | Plan B documented: pivot to single-target framework; result is publishable as negative finding | C |
| R5 | WT EC50 wrong by > 5× at Phase G | Medium | High (Phase G/H validation) | Revisit Vina box / pocket assignment; recalibrate GNINA-to-Kd conversion; if persistent, escalate to deferred FEP path on top hits per `phase_b_binding_pose.md` §13 | G/H |
| R6 | gas-1 hypersensitivity does not reproduce | Medium | High (anchor 3) | Revisit Phase F K-ATP coupling; alternative mechanism (direct Complex I redox / ROS-based); document | F/H |
| R7 | JAX differentiable simulator does not converge | High (small data, large param space) | Low (stretch) | Defer Phase I; Phase H validates without it; document non-identifiability | I |
| R8 | Compute scope creep | Medium | Medium | Bound each phase by preregistered compute budget; Tier 2 only after Tier 1 ships; **no external spend without explicit user reversal of zero-cost commitment** | program-level |
| R9 | **Citation misattribution propagation** | Medium | High (Wave 2 lost 3-4 weeks to one) | Pre-flight verification queue; PMID/DOI required at point of citation; mark unverified explicitly | program-level |
| R10 | UNC-79 / UNC-80 NCA-complex auxiliary subunits structurally challenging | High (known IDR-rich) | Medium | Predict structured domains only; document IDR regions; focus Phase B on functional domain | A |
| R11 | Wave 2 in-flight channels (IRK, UNC-103) don't ship in time for Phase G | Medium | Low (workaround exists) | Hybrid Brian2+NEURON path documented in `wave2_handoff.md`; switch to pure Brian2 when Wave 2 ships | G |
| R12 | Notebook pipeline artifact format changes mid-Wave-P | Medium | Medium | Pin to specific commit per Wave P phase; adapter layer for format changes | G |
| R13 | RTX 4060 Ti 8 GB VRAM insufficient for some MD systems | Medium | Medium | Reduce system size (truncate to functional domain); free-tier Colab T4 overflow as last resort; never required for canonical Phase D plan | D |
| R14 | **Pentameric AF-Multimer / structure prediction fails to fit in 8 GB VRAM** | Medium-High | Medium (delays Phase A on those targets) | Fallback ladder per R1: ESMFold (MIT, Lin et al. 2023 *Science* DOI 10.1126/science.ade2574) primary; Boltz-1 (MIT, Wohlwend et al. 2024) secondary; OpenFold (Apache 2.0, Ahdritz et al. 2024 *Nat Methods*) tertiary; ColabFold free tier (T4) quaternary; subunit-by-subunit pocket modeling as last resort. | A |
| R15 | License terms on AF-Multimer / RFAA preclude desired use case | Low (open-source substitutes available) | Low (was Medium; demoted) | ESMFold / OpenFold / Boltz-1 are MIT/Apache 2.0 — no commercial restriction. AF-Multimer / RFAA are non-load-bearing in revised plan. | A |
| R16 | PyPhi computational scaling prevents network-level Phi | High (known exponential) | Low (stretch) | Use 5-cell command-neuron subset; aggregate hierarchical Phi; document approximation | J |
| R17 | Halothane Kp partition coefficient varies 2× across sources | Medium | Medium | Use median of 3 published values; sensitivity analysis at 0.5× and 2×; document range | C |
| R18 | Hill coefficient n=1 default underestimates pentameric cooperativity | Medium | Medium | Sensitivity analysis at n=2 for pentameric receptors; document the difference | C |
| R19 | Anesthetic immobilization threshold calibration is sensitive to WT control variability | Medium | Medium | Use 90th percentile of WT firing rates as threshold; sensitivity analysis on ±10% threshold | G |
| R20 | Per-target lesion test inconclusive due to non-linear interactions | Low | Low | Report non-linearity directly; non-linearity is consistent with multi-target framing | G |
| R21 | **Complex I full-assembly (~45 subunits) intractable on local 8 GB VRAM hardware** | Certain | Low (canonical plan already scopes around it) | Phase A canonically targets **single-subunit-per-anesthetic-binding-site** modeling: GAS-1 (NDUFS2 homolog) primary per Morgan & Sedensky 1995 PMID 7549290; NUO-1 through NUO-6 individually as secondary. Full-assembly modeling is DEFERRED / SPECULATIVE; no claim is made about quaternary structure context for anesthetic binding within Wave P. | A/F |
| R22 (DEFERRED) | ~~Lambda Labs / AWS cloud burst exceeds $200 budget~~ — **demoted to deferred enhancement** | Not applicable in canonical plan | Not applicable | No cloud bursts in canonical plan. External spend is $0. If user later authorizes FEP cloud spend, $200-400 budget would apply per `phase_b_binding_pose.md` §13. | DEFERRED |

---

## Most consequential risks

### R4 — Multi-target framing falsified at Gate C.1

This is the **single most consequential risk** for the program. If only 0 or 1 targets exceed 10% occupancy at halothane 1× EC50, Wave P's foundational hypothesis is wrong, and the program pivots to a single-target validation framework.

The mitigation is preparation: Wave P explicitly anticipates this outcome. The Plan B is documented (single-target validation; the negative result is publishable as a finding).

The risk should be **surfaced as a research finding rather than treated as a failure**. The user's working style accommodates this: falsifiability before elaboration; honest negative results.

### R9 — Citation misattribution propagation

Wave 2 lost approximately 3-4 weeks of work to a single misattribution (Mellem 2008 → AVA / "20 mV / 600 ms"). Wave P enforces the citation hygiene from day 1 but the risk recurs because:

- Pre-flight verification is human-in-the-loop work that must happen.
- Quantitative biological values need cross-checking against the cited figure.
- A surface citation can be correct (paper exists, PMID is correct) while the cited claim does not match the paper.

The mitigation is the explicit pre-flight verification queue at the bottom of `validation/empirical_anchors.md`. Each blocking PMID must resolve before the corresponding phase enters its execution work block.

### R5 — WT EC50 wrong by > 5×

If Phase G produces WT halothane EC50 dramatically wrong (e.g., 10× off), the binding-affinity calibration is structurally wrong. The diagnosis path is:

1. Phase B docking miscalibration → GNINA-to-Kd conversion factor wrong.
2. Phase C partition coefficient wrong.
3. Phase D shift form wrong.

Mitigation: factor-of-3 uncertainty bands are propagated from Phase B to Phase C; the deferred FEP path (`phase_b_binding_pose.md` §13) is available if the user later authorizes the cost to disambiguate. In the canonical zero-spend plan, the diagnosis runs against literature-anchored mammalian-control MD (Phase D) and against photolabel cross-validation (Phase B) without invoking FEP.

---

## Risk-update protocol

After each Wave P work block, the relevant phase's `phase_*_completion.md` updates the risk register:

- Risks that surfaced get marked with the date and the decision taken.
- Risks that did not surface in the expected phase get rolled forward to the next phase if applicable.
- New risks discovered get added to the register with date and source.

The risk register is **versioned** — the kickoff version (this file) is the baseline. Updates are appended, not in-place edited.
