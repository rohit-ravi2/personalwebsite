# WF2 — Rock-Solid Gated Implementation Roadmap (2026-06-11)

**Question (user):** scope the addressable (non-Tier4-gated) problems with strong computational backing
through ANY means (new sims, new data acquisition, new pipeline/substrate construction); produce a
gated implementation plan, then implement. First principles only, no literature.

Produced by a 195-agent gated-planning workflow (run wf_5928132c-30d): 18 problems (17 addressable,
1 parked); 45 candidate solutions architected; 16 admitted round 1 (strong + feasible + preregisterable),
+1 in completeness, then space exhausted → 15-17 gated work-blocks.

---

## Executive summary

WF2 admitted 15 gated work-blocks across 12 problems (P18 x3 variants, P7, P1_P2 x3, P8 x2, P16, P11, P3, P13, P4, P17, P19, P20). All run LOCAL on the RTX 4060 Ti / 12-core CPU except two Greene-HPC items (P13 ESMFold UNC-80; optional large ensemble sweeps). One problem (P14, thermodynamic-necessity/entropy) is genuinely full-energetic-Tier4 and is PARKED with its minimal lifting build.

The spine is deflation-risk-first and dependency-ordered. Two cheap honesty/foundation audits run FIRST and gate everything: P18 (write-path closure + dynamic rank certificate + artifact provenance — three complementary checks proving V1 is actually rank-2 and that every load-bearing percentile came from the rank-2 path) and P7 (git-archaeology proving whether the sat_pa ladder + 3 Hz cutoff were frozen before the halothane anchor fit). Both are static/CPU, minutes-to-hours, and both can DEFLATE central honesty claims — exactly why they go first. If P18 fails (hidden per-neuron writer found), the entire ledger re-scopes before any build; if P7 returns POSTHOC/UNDECIDABLE, the 'one free parameter (alpha)' framing is narrowed.

Tier 2 is the deflation-capable empirical core, all reusing the frozen v7 ensemble pipeline: P8 (corrected two-coordinate (total_pa, snare_factor) Match#2b null — can deflate fly's last clean positive), which GATES P16 (held-out Atanas connectome-vs-strict-degree-shuffle exam — can deflate V1's structural claims to mean-field), P3 (mean-field collapse theorem deflating the mouse class-identity claim to a derived corollary), P4 (Gate-4 entailment demotion + SNARE-orthogonal falsifier), P11 (Wave-P K_p reference-frame recompute — can deflate the 30/30 multi-target headline), P20 (genotype-anesthetic two-block reachability partition), and P17 (the C-22 readout-validity gate against Kato HisCl + NeuroPAL — can deflate the V7 immobilization-readout semantics itself and BLOCKS the Paper-2 bridge).

Tier 3 is the keystone BUILD: minimal-delta-V2 (P1_P2) — CeNGEN per-class expression vectors x_c replace the all-ones broadcast, lifting operator rank 2->min(7,N) and making Match#3 cell-type-targeting a genuine accept-either-way test for the first time. It is ~90% pre-built infrastructure, wrapped by two meta-gates (SOL7 spread-statistic-frozen-first harness; SOL8 NeuroPAL external grounding) that prevent it from being a can-only-succeed build.

Tier 4 (Greene): P13 ESMFold NCA-1/UNC-80 + re-dock; the nca magnitude question is settled LOCALLY and independently by P13-SOL28 (biophysics-frozen interval sweep). P19 is a low-priority confirmatory leaf. P14 is PARKED.

Every block carries preregistered, hash-locked, accept-either-way gates with too-special floors where a leak could fake success, FAIL branches pre-written as publishable claim-narrowing, and pseudo-test screens proving the gate can fail on this substrate. No block can only-succeed; several are designed to deflate the project's own positives.

---

## Execution sequence (deflation-risk-first, dependency-ordered)

DEFLATION-RISK-FIRST + DEPENDENCY-ORDERED. Phases gate downstream phases; within a phase, blocks are parallelizable unless noted.

PHASE 0 — FOUNDATION/HONESTY AUDITS (run FIRST, all LOCAL CPU, gate everything):
- P18 bundle (3 blocks, ship together): [A] write-path closure audit (static AST/regex, seconds-to-run), [B] dynamic dual-input rank certificate (~50-case battery, <1 min, no brain.run()), [C] artifact provenance ledger (static import-graph, minutes). G1-closure / G1-current-rank / G1-rank2-attribution must PASS for the rank-2 premise to stand. If ANY fails -> HALT, ledger re-derives operator rank before any build. These are the root: P1_P2, P8, P3, P4, P20 all assume rank-2.
- P7 provenance audit (git-archaeology, minutes) — INDEPENDENT of P18, runs in parallel. GATE P7.A/P7.B. Deflates the 'one free alpha' honesty frame if POSTHOC/UNDECIDABLE. Must precede any re-promotion of knapsack/one-free-alpha claims and the Wave-P ladder work (P11->P10).

PHASE 1 — DEFLATION-CAPABLE EMPIRICAL CORE (LOCAL; gated on Phase 0 PASS):
- P8 Match#2b two-coordinate null (overnight ensemble re-run). MUST precede P16 and P9-arm. G-P8b-1 can DEFLATE fly. Run early — it can remove the project's last clean positive.
- P3 mean-field collapse theorem (analytic + 1 overnight degree sweep). Deflates mouse 8.1 to a corollary. WB3 has a soft dependency on P8's dual-coordinate sampler (use stratify-fallback if P8 not yet merged).
- P11 Wave-P K_p reference-frame recompute (pure arithmetic, seconds). Molecular-layer-only, independent of the network family — can run anytime in Phase 1. Feeds P10 (parked-as-rejected) and the sat_pa ladder honesty.
- P4 Gate-4 entailment + SNARE falsifier (re-analysis + <1hr sim). Shares the WB0 write-path grep with P18.
- P20 genotype x anesthetic two-block reachability (~2hr sim slice). Depends on P18 PASS; feeds PARK routing into P1_P2 and P14.
- P17 readout-validity / C-22 gate (Kato .mat full-load + Atanas streaming, ~few CPU-hours). Independent of rank-2 family. GATES the Paper-2 bridge and the V7 immobilization-readout language.
- P16 held-out Atanas structure->activity exam (overnight, ~1640 spontaneous LIF runs). GATED ON P8 fly verdict (informs whether to expect structural signal) and on its own WB1 strict double-edge-swap unit test. Can deflate V1's structural claims to mean-field.

PHASE 2 — KEYSTONE BUILD (LOCAL; gated on P18 PASS):
- P1_P2 minimal-delta-V2 rank-lift (build + ~4-5 CPU-hr Match#3 ensemble), wrapped by SOL7 (freeze-statistic-first harness) and SOL8 (NeuroPAL external grounding, streaming). G0_RANK_LIFT_REALIZED gates G1_MATCH3; G2 certifies independence. SOL8-G1 (NeuroPAL Jaccard) is a HARD join-correctness gate. Reuses P8's corrected sampler (folded into WB4).

PHASE 3 — GREENE HPC + LEAVES:
- P13 ESMFold NCA-1/UNC-80 (Greene GPU >=24GB, one short job) + local re-dock. The nca MAGNITUDE question is settled independently and locally by P13-SOL28 (biophysics-frozen interval [75,120] pA sweep, overnight) — run SOL28 in Phase 1/2 without waiting on Greene.
- P19 gap-on/gap-off confirmatory (LOCAL, <1hr) — lowest severity, run last; soft-depends on P3/SOL1 eps_MF for its secondary gate.

PARKED: P14 (thermodynamic necessity / entropy) — full-energetic-Tier4, not scheduled; minimal build documented for user decision.

---

## Compute scheduling

LOCAL (RTX 4060 Ti 8GB / 12-core CPU) — everything except two items. The existing full ensemble ran in 410 min on 12 CPU cores; all re-run-based blocks fit that envelope. bf16/gradient-checkpointing are IRRELEVANT here (Brian2 LIF, not deep nets) — the GPU is largely untouched.
- Static/analytic (seconds-to-minutes, CPU): P18-A (AST/regex), P18-C (import-graph), P7 (git log -S), P11 (2160 Hill evals), P3-WB1/WB2 (analytic + 1 instrumented array), P4-WB0/WB2 (grep + frozen-CSV re-analysis), P1_P2-WB1 (CeNGEN CSV stream + join).
- Operator-immediate, no brain.run() (<1 min, CPU): P18-B (~50-case battery, SVD of 50x300).
- Ensemble re-runs (overnight, CPU, ~410-min-class or a slice): P8 Match#2b (50 ens x 3 org x 8 dose x 3 seed), P8b WB3 G-grid (5760 sims/organism, fly dominates ~6-8 CPU-hr — confirmatory, skippable if tight), P3-WB4 degree sweep (4 mouse re-runs), P4-WB5 (1 synthetic compound, <1hr), P20-WB2 (~1080 LIF runs, ~2hr), P13-SOL28 (64 nca-subsets x 3 endpoints, hours-to-overnight), P1_P2-WB5 Match#3 (2 organisms, ~4-5 CPU-hr).
- Streaming held-out reads (CPU, hours, NEVER resident — h5py partial reads only; atanas NWB are ~25-27GB EACH, NOT the 194M the ground map once claimed): P16 (parse + 1640 spontaneous runs, overnight), P17 (Kato .mat ~42-56MB full-load + Atanas streaming, ~few CPU-hr), P1_P2-SOL8 (NeuroPAL ~815MB streaming + Atanas bouts, ~1-2hr).
- GREENE HPC (reservable, free; the ONLY hard-required remote jobs): P13-SOL27 ESMFold/AlphaFold-multimer of NCA-1 (Q6Q762) + UNC-80 (Q9XV66) — UNC-80 OOM'd on 8GB local, needs >=24GB GPU; one short job in miniconda3/envs/ml-equivalent. Subsequent AutoDock-Vina re-dock is CPU -> back to LOCAL. OPTIONAL Greene: large parallel ensemble seed-sweeps for tighter P16 CIs (10 worms x many shuffle seeds) — not on the critical path.
- Notify policy: notify on launch + completion of every overnight job (P8, P16, P20, SOL28, P1_P2-WB5) per the long-task rule; urgent-notify if any HALT gate (P18 fail, P8 G1-formula-fidelity fail, P1_P2 G0 fail) fires.

---

## Open risks

1. P18 is the single point of failure for the whole build program. If G1-closure/G1-current-rank finds a hidden per-neuron writer, P1_P2/P8/P3/P4/P20 ALL re-scope. Mitigation: the verified code read (uniform I_ext broadcast, single global SNARE syn.w scale, run_single mutation set == {apply_genotype, apply_anesthetic}) puts the prior at ~85% PASS, but the FAIL branch is real and pre-accepted (it would REVIVE some positive claims). Do P18 first and do not start any build until it lands.

2. Held-out-data null is the Bayesian-likely outcome for P16 and SOL8-G2. The user's own C. elegans prior (connectome ~ shuffled at d=4/8/16, calcium-only) predicts P16 Gate-A NULL and SOL8-G2 ambiguous. This is pre-accepted and publishable (deflates V1 structural claims to mean-field), but the user should expect the keystone V2 build to land a NULL on worm Match#3 too (G1_MATCH3 pct~50%). The PAYOFF organism is fly (WF1 says identity does measurable work there) — sequence fly Match#3 and weight expectations accordingly.

3. P8 can remove the project's last clean positive. If G-P8b-1 DEFLATES fly (percentile >25-30% under joint SNARE control), the fly-shuffle (P9, not separately admitted here) has no positive to explain and is cancelled, and the cross-organism conservation framing loses its single quantified anchor. This is the intended anti-target-engineering outcome — flagged so it is not a surprise.

4. Streaming-cost reality check: atanas NWB files are ~25-27GB EACH (P17/P16/SOL8 all touch them). Verified the ground-map's '194M' figure was stale. h5py partial reads are mandatory; never cat/load whole. Budget P16/P17/SOL8 as hours-of-streaming-I/O, not minutes.

5. P7 may be PERMANENTLY UNDECIDABLE for the sat_pa ladder (co-committed with the alpha fit; pre-first-commit working-tree tuning is unobservable from git). The honest verdict is UNDECIDABLE_BY_TIMESTAMP, not a vindication — the 'one free alpha' claim narrows regardless. Set expectations: this audit mostly DEFLATES.

6. Gate-power on small held-out samples: SOL8-G3 and P17-Q feasibility forks may find too few clean quiescent bouts / NeuroPAL-named neurons (freely-moving recordings, sparse spontaneous quiescence, ~40-60/300 named). Pre-registered park-to-fewer-worms / descriptive-only branches handle this, but several confirmatory arms may ship UNDERPOWERED rather than decisive.

7. P11 -> P10 propagation question is partly moot: the rejected-solutions analysis (SOL49) verified phase_g_state_validator.py:24 disclaims any Vina/wave2_overlay dependency — the sat_pa ladder is hand-set and decoupled from the molecular layer. So a P11 deflation of the 30/30 headline does NOT mechanically flip any V1 quorum/percentile; it is a Wave-P-layer honesty finding only. Do not over-claim downstream network impact from P11.

8. P13 magnitude vs structure split: folding NCA-1/UNC-80 on Greene resolves only structure-availability/dockability. NALCN has no published Kd (pharmacological outlier), so nca_block stays structurally UNCALIBRATABLE — that residue is Tier4-adjacent and parked. SOL28's interval sweep settles whether the magnitude even matters to the quorum (likely robust across [75,120] pA); run SOL28 locally and do NOT wait on Greene for the load-bearing answer.

9. SOL7/SOL8 parasitism on P1_P2: the Match#3 meta-gates cannot run until the rank-lift operator is merged. WB0 (freeze spread-statistic S in closed form) MUST precede any run to remove the hidden-tunable-DOF the skeptics flagged; if it is skipped the whole V2 result becomes can-only-succeed. Enforce the WB0 hash-lock as a hard predecessor.

---

## Gated work-blocks (summary — each gate is preregistered + accept-either-way)

### P18-A Write-path closure audit — G1-closure (reachable+rank-contributing writers == {apply_genotype,apply_anesthetic}, zero per-neuron-varying writes, all 3 factories) / G2-rot-detector (re-derived seed set superset of 11+ known witnesses) / G3-rate-source-durability (regression lock asserts both no-new-I_ext-writer AND proprio-rate-zero)
### P18-B Dynamic rank certificate — G1-current-rank (#singular-values(D)>1e-6 == 1 AND max rho_k<1e-6, all-ones broadcast) / G2-synapse-rank (S rank-1, every r_k spatially uniform = single global SNARE scalar) / G3-coverage-completeness (7/7 classes, 4/4 genotype branches, 100% if-branches hit before rank gates readable)
### P18-C Artifact provenance ledger — G0-denominator-freeze (grep-reachability predicate, hash-locked, no post-hoc shrink) / G1-rank2-attribution (100% in-scope artifacts trace to run_single rank-2 path by call-graph AND schema) / G1b-untraceable (zero orphan CSVs)
### P7 Provenance git-archaeology — GATE P7.A (3 Hz cutoff: PROVEN_PREREGISTERED iff dated prereg pins rule-form AND value within 10% WT-calibrated band, else POSTHOC / RULE_REDESIGNED_VALUE_DEFENSIBLE) / GATE P7.B (sat_pa ladder: PROVEN iff earlier dated doc pins 8 values + byte-identity, else UNDECIDABLE_BY_TIMESTAMP / ACTIVE_TUNING-halt)
### P8 Match#2b two-coordinate null — G1-formula-fidelity (sampler reproduces operator total_pa to 1e-9, HARD, blocks run) / G2-null-density (>=20 jointly-matched draws/organism else INDETERMINATE) / G3-fly-survival (fly percentile >5% and <=30% PASS; >30% DEFLATE; <=5% too-special quarantine)
### P8b Joint-control + analytic cross-check — G-P8b-1 (fly survives joint SNARE control, <=25% PASS / >25% DEFLATE / <=0.5% quarantine) / G-P8b-2 (analytic-vs-empirical self-consistency, confirmatory) / G-P8b-3 (power gate: both coordinate spreads <=10%)
### P16 Held-out structure->activity exam — Gate-A (delta-R = R(real) - R(strict double-edge-swap), CI-lower > model-internal noise_floor PASS; <= noise_floor NULL/deflate; R(real)>=0.6 too-special HALT-for-leak) / Gate-B (sign-robustness across both glu-sign arms) / Gate-C (positive control recovers known structure through GCaMP/downsample/regression pipeline at R>=0.5, else NULL is weakly-informative)
### P11 K_p reference-frame recompute — G-P11.1 (M0 no-Kp: >=5/30 genes >10% occ at halothane 1xEC50 survives, with saturation drop; <5 deflates) / G-P11.2 (frame-fragility: spread n_M2-n_M0 <=2 robust, >=5 fragile) / G-P11.3 (pseudo-test screen: gate is live on both branches)
### P3 Mean-field collapse theorem — G-P3.1 (Var_i(mu_i)/<mu>^2 slope in [-1.3,-0.7] AND conditional MI <= through-QF finite-N band) / G-P3.2 (dual-coordinate-controlled mouse percentile in [25%,75%]; <=10% too-special contradicts theorem) / G-P3.3 (confirmatory degree sweep, operating-point-invariance-guarded, non-load-bearing)
### P4 Gate-4 entailment + SNARE falsifier — G4-WB0-writepath (exactly 1 orthogonal synaptic channel = SNARE) / G4-A-entailment (R2(Stage3 ~ total_pa+snare_gain)>=0.95 AND zero attrition, demotes 8.4; demonstration only) / G4-B-snare-falsifier (precondition: SNARE-max moves QF>=0.2; then pseudo-NI max_qf>=0.5 Outcome-A-demote / <0.5 Outcome-B-retain)
### P20 Genotype x anesthetic two-block reachability — P20-GATE-A (>=1 of 9 genotypes STRUCTURALLY-UNREACHABLE = real positive; all reachable = epistasis-is-bookkeeping deflation) / P20-GATE-B (Gao-block multiplicative interaction matches prediction, not additive null) / P20-GATE-C (too-special: all 9 ratios <5% error -> circularity flag -> P7 escalation)
### P17 Readout-validity / C-22 gate — Q1-data-sign (command-low activity tracks behavioral-immobility, s>0 CI excludes 0 in BOTH Kato-labeled-states AND Atanas-pose) / Q2-causal-AVA-silencing (HisCl data-side dwell shift matches pre-declared direction; model-side bonus with too-special leak floor) / Q3-threshold-valley (Kato command bimodal deltaBIC>=10 AND 3.0Hz within +/-1Hz of model's OWN valley; no Hz<->dF/F equivalence)
### P1_P2 Minimal-delta-V2 rank-lift — G0_RANK_LIFT_REALIZED (two equal-(total_pa,snare_gain) profiles produce q_neuron differing by L2>1e-6) / G1_MATCH3_SPATIAL_SPECIAL (conserved PR percentile in (1%,10%] PASS; >10% NULL/deflate; <=1% too-special leak) / G2_MATCH3_NOT_ENTAILED (Var(PR over surrogates)>1e-9, not pinned by magnitude)
### P1_P2 SOL7 falsifiable-gate harness — G1_able_to_fail_screen (disjoint-support profiles give different S, identical-support give same S, before any biology) / G2_spread_percentile_two_sided (P3_org<=30% positive, <=5% floor->within-block surrogate, >30% TESTED-AND-NULL) / G3_independence_from_match2 (Spearman(magnitude-pct,spread-pct)<0.95 AND attrition>0)
### P1_P2 SOL8 external NeuroPAL grounding — G1_neuropal_join_correctness (per-class Jaccard>=0.5 for >=5/7 classes, HARD; fail HALTS + re-freeze x_c) / G2_quiescence_structure_confirmatory (Spearman(S_V2,natural-quiescence-drop) CI>0 above gene-marginal-preserving structured null; one-directional support-or-silence) / G3_bout_yield_feasibility_fork (n_worms>=6 full / 3-5 descriptive / <3 park-G2)
### P13-SOL28 nca biophysics-frozen interval sweep — G1-interval-provenance (interval = closed-form of frozen disk constants, [75,120] pA, NOT bracketing legacy 40 pA) / G2-quorum-survival (at nca=75 pA, passing nca-quorum subsets >=80% of baseline AND SNARE-OR-ComplexI universality holds) / G3-substitution-leak-screen (7 non-nca scalars byte-identical across endpoints)
### P13-SOL27 ESMFold NCA-1/UNC-80 (Greene) — pLDDT pocket-confidence gate (fold proceeds to docking iff pocket pLDDT clears threshold; paralog-bridge NCA-2 fallback; structure-availability half only — magnitude half settled by SOL28)
### P19 gap-on/gap-off confirmatory — G1-QF-claim-impact (max_d|QF_gapon-QF_gapoff|<0.10 PASS=gap-inert footnote; >=0.10 FAIL=report worm residual as gap bracket) / G2-eps-consistency (max_d|eps_emp-eps_MF|<0.10 corroborates SOL1 bound; soft-dependent, else DEFERRED)

---

## PARKED — full-energetic-Tier4

P14 — Thermodynamic NECESSITY of reversible reduced-activity states (C3 / sec 8.7 entropy framework). GENUINELY full_energetic_tier4_only; PARKED, not scheduled. Why V1 cannot represent it: V1's state vector is one voltage scalar per neuron + static injected current — no conserved charge, no k_B T, no moving Nernst, no metabolic free-energy budget, no lipid/aqueous coordinate. A nats-valued information entropy without k_B T and a charge ledger CANNOT distinguish reversible quiescence from collapse/death, and 'reversible' is the entire load-bearing content of the necessity claim.

MINIMAL LIFTING BUILD (strictly larger than minimal-delta-V2; this is the 'V8' substrate): per-neuron Na/K ion ledger (conserved charge) + Na/K-ATPase pump current + joule-valued metabolic free-energy budget + a lipid/aqueous partition coordinate + a state-continuation harness for hysteresis/reversibility (so dose-up != dose-down can produce a non-zero loop area). With that substrate: (1) lipid-partitioning becomes a DERIVED Meyer-Overton result rather than an assumed K_p multiply (this also retroactively grounds P5 and P11's reference-frame question from first principles); (2) reversibility becomes a bifurcation/hysteresis property the model can exhibit and measure; (3) a genuine Schnakenberg sigma_EP / Landauer joule-valued entropy production becomes definable on a cyclic >=2-D observable.

HONEST BOUNDARY to enforce if it is ever built: do NOT relabel a dynamical-systems quasi-potential (Phi = -D log p_ss with the fixed 6 mV noise) as thermodynamic free energy — that is a category error the doc already flags. The substrate-free residue available NOW (entropy-sense hygiene + the impossibility-boundary theorem) strengthens the BOUNDARY statement, not the claim, and is documentation-only.

COMPUTE: this is the build that would justify reserving NYU Greene HPC for the heavier numerical integration (ion-pool ODEs + metabolic state per neuron at scale). DECISION IS THE USER'S — it is a parked moonshot, not a live defect, provided sec 8.7 is not overclaimed in the interim. Note: P20's WB5 PARK routing and P13's nca-calibration residue both feed candidates into this same Tier4 build (gas-1 mitochondrial synergy needs the Complex-I/ATP metabolic state; nca_block's missing functional Kd is structurally uncalibratable without it).

---

# AnestheticSimulator WF2 — Rock-Solid Gated Implementation Roadmap

**Principle:** deflation-risk-first, dependency-ordered. Run the audits and tests that can DEFLATE a claim before the builds that rest on them. Every block carries preregistered, hash-locked, accept-either-way gates with FAIL branches pre-written as publishable claim-narrowing, too-special floors where a leak could fake success, and a pseudo-test screen proving the gate can fail on this substrate. No block can only-succeed.

**Hardware:** LOCAL = RTX 4060 Ti 8GB / 12-core CPU (Brian2 LIF — GPU largely idle, bf16 irrelevant). GREENE = NYU HPC, reservable/free, used for exactly two things (ESMFold UNC-80; optional CI-tightening sweeps). Held-out data streamed via h5py, never resident.

**Grounded on disk (verified):** operator is uniform `I_ext[:] += total_pa*pA` + single global SNARE `syn.w *= factor`; `run_single` mutation set == `{apply_genotype, apply_anesthetic}` between construction and `brain.run()`; `connectome.npz` carries `names/klass/iGluR_expr/GluCl_expr`; all 4 CeNGEN threshold CSVs present; frozen v7 stage CSVs present; atanas2023 / randi / kato / neuropal / wormpose all on disk.

---

## PHASE 0 — FOUNDATION & HONESTY AUDITS (run FIRST, LOCAL CPU, gate everything)

These are cheap, deflation-capable, and foundation-gating. **Nothing downstream proceeds until P18 PASSES.** P7 runs in parallel.

### WB-P18 — Is V1 actually rank-2? (ship as one bundle: A static-closure, B dynamic-certificate, C provenance)
Three complementary proofs that (i) the write-path is closed, (ii) the realized values collapse to 2 DOF, (iii) every load-bearing percentile came from the rank-2 path.

**P18-A Write-path closure (static AST/regex; seconds to run):**
- WB0 prereg: fix the Brian2 coupling vocabulary {I_ext, v, w, PoissonGroup.rates}, the machine-re-derived seed-regex set, the 3 in-scope factories (worm `SeededLIFBrain`, fly, mouse), the rank-contribution rule (per-neuron-VARYING write only). Hash-lock.
- WB1 re-derive writer inventory across `AnestheticSimulator/src` + `scripts/brain`; assert superset of 11+ known witnesses (rot-detector).
- WB2 `ast` reachability from `run_single`: assert mutation set == `{apply_genotype@367, apply_anesthetic@368}`; assert ablate/proprio/sensory/modulation never reached, per factory.
- WB3 dormancy classification (proprio = ATTACHED-BUT-RATE-GATED-ZERO; `_push_ablation` = UNATTACHED).
- WB4 regression lock (pytest, CI): clause A (no new I_ext/w writer) + clause B (rate sources unreachable).
- **Gates:** `G1-closure` (reachable+rank-contributing writers == {apply_genotype, apply_anesthetic}, zero per-neuron-varying, all 3 factories). `G2-rot-detector` (re-derived seed set ⊇ known witnesses). `G3-rate-source-durability` (lock asserts BOTH clauses). FAIL on any -> rank-2 wrong, HALT, re-scope ledger (good news for some positive claims).

**P18-B Dynamic rank certificate (~50-case battery, <1 min, NO brain.run()):**
- WB0 prereg statistics (singular spectrum of I_ext-delta matrix D and synapse-ratio matrix S; per-profile uniformity residual rho_k), thresholds (tau=1e-6, one decade above float32 floor), full-line decision rule (>1e-6 FALSIFY / [1e-9,1e-6] ESCALATE / <1e-9 PASS). Hash-lock.
- WB1 coverage-complete battery (7 classes x>=2 engagements, 4 genotype branches, 3/5/7-class mixtures, SNARE alone + co-engaged). WB2 instrument: snapshot pre/post I_ext + syn.w, NO integration. WB3 SVD + reconstruction + uniformity. WB4 verdict.
- **Gates:** `G1-current-rank` (#sv(D)>1e-6 == 1 AND max rho_k<1e-6 = all-ones broadcast). `G2-synapse-rank` (S rank-1, every r_k spatially uniform = single global SNARE scalar). `G3-coverage-completeness` (7/7 classes, 4/4 genotype branches, 100% if-branches hit — VOIDs the rank gates if incomplete). FAIL = hidden per-neuron/synapse structure; rank>2; ledger re-scopes (Match#3/mouse-identity/fly-SNARE may already be partly representable — accepted deflation of the impossibility framing, halt-and-narrow per R6).

**P18-C Artifact provenance ledger (static import-graph; minutes):**
- WB0 grep-reachability denominator freeze (hash-locked, no post-hoc shrink; the known-suspect `phase_g_halothane_dose_response.csv` — written by the higher-rank `apply_to_brain` module with a per-neuron `k2p_max` schema — is recorded in the candidate set, scope decided by predicate). WB1 import+call-graph tracer (does writer reach `apply_to_brain` vs only `apply_anesthetic`/`apply_genotype`). WB2 schema fingerprint cross-check (RANK2-NATIVE vs HIGHER-RANK-NATIVE columns). WB3 verdict + claim re-attribution. WB4 untraceable-branch hardening (orphan CSV -> FAIL, never pass-by-default).
- **Gates:** `G0-denominator-freeze`. `G1-rank2-attribution` (100% in-scope artifacts trace to run_single by call-graph AND schema; contaminated==0). `G1b-untraceable` (untraceable_in_scope==0). FAIL = a load-bearing number silently came from the richer operator; that claim VOID-pending-rederivation (publishable provenance correction). Expected: ensemble/percentile family CLEAN; dose_response.csv HIGHER-RANK but website-only/out-of-scope by the frozen predicate.

**Compute:** 100% local CPU, seconds-to-minutes. **Dependency:** ROOT. Everything else waits on G1-closure + G1-current-rank + G1-rank2-attribution PASS. **Expected:** ~85% PASS (verified code read supports rank-2); FAIL is real and pre-accepted.

### WB-P7 — Were sat_pa ladder + 3 Hz cutoff frozen BEFORE the halothane anchor fit? (git-archaeology; minutes; INDEPENDENT of P18, parallel)
- WB0 prereg verdict-mapping + token list, hash-locked. WB1 token-introduction archaeology (`git log --all -S<token>`). WB2 byte-identity/tuning-signature trace across commits touching `phase_g_state_validator.py` (HALT-if sat_pa value changed in same diff as an alpha re-fit -> ACTIVE_TUNING). WB3 prereg-vs-shipped rule diff (compound WT-calibrated FSM vs single fixed 3.0 Hz). WB4 over-deflation guard (is 3.0 Hz within 10% of the WT-calibrated 90th-pct value?). WB5 signed verdict + PROVENANCE.md disclosure.
- **GATE P7.A (3 Hz cutoff):** PROVEN_PREREGISTERED iff dated prereg pins rule-form AND value within 10% band; else POSTHOC; else RULE_REDESIGNED_VALUE_DEFENSIBLE. **GATE P7.B (sat_pa ladder):** PROVEN iff earlier dated doc pins 8 values + byte-identity; else UNDECIDABLE_BY_TIMESTAMP; ACTIVE_TUNING->halt.
- FAIL/deflate branch pre-written: narrow 'one free parameter (alpha)' to 'one fitted scalar (alpha, itself re-fit 0.22<->0.13) conditional on a ladder of UNDECIDABLE a-priori status and a cutoff rule redesigned post-prereg.' **Compute:** local CPU. **Dependency:** none upstream; must precede re-promotion of knapsack/one-free-alpha honesty + the Wave-P ladder (P11->P10). **Expected:** mostly DEFLATES (P7.A POSTHOC, P7.B UNDECIDABLE).

---

## PHASE 1 — DEFLATION-CAPABLE EMPIRICAL CORE (LOCAL; gated on Phase 0 PASS)

### WB-P8 — Corrected two-coordinate Match#2b null (can deflate fly's last clean positive)
The shipped Match#2 sampler matched only `agg_pa` and left `snare_factor` free; it also adds a phantom SNARE 50 pA term the operator never applies. Fix to match BOTH `(total_pa, snare_factor)`.
- WB0 prereg (exact operator-path formulas for both coordinates, +/-5% joint tolerance, N_min=20 accepted draws/organism, frozen percentile thresholds <=30% pass / <=5% too-special). WB1 single-source refactor (import the operator's dict, DELETE the duplicate + `_aggregate_pa_at_dose`) + unit test (reproduce operator total_pa to 1e-9). WB2 pure-Python null-density dry-run (pre-empt empty-null-after-6h). WB3 full re-run (50 ens x 3 org x 8 dose x 3 seed) emitting NEW match2b artifacts (do not overwrite). WB4 verdict + claim reconciliation.
- **Gates:** `G1-formula-fidelity` (sampler == operator total_pa to 1e-9; HARD, blocks the overnight run). `G2-null-density` (>=20 jointly-matched draws/organism else INDETERMINATE). `G3-fly-survival` (>5% and <=30% PASS-as-identity; >30% DEFLATE; <=5% too-special quarantine).
- Optional companion **P8b** (joint-control + tabulated-G analytic cross-check): `G-P8b-1` fly survives joint control <=25%, `G-P8b-2` analytic-vs-empirical consistency (confirmatory), `G-P8b-3` power gate (coordinate spreads <=10%).
- **Compute:** WB0-2 minutes; WB3 overnight (~410-min-class). **Dependency:** P18 PASS (assumes (total_pa, snare_factor) are the complete sufficient statistic). **GATES P16 and the fly-shuffle arm.** **Expected:** worm stays ~0% (crosses too-special floor, anchor-overfit per P6); mouse stays ~46% (P7-null re-confirmed on airtight control); fly is the live deflation risk.

### WB-P3 — Mean-field collapse theorem (deflates mouse 8.1 to a derived corollary)
Prove the mouse class-identity null is entailed a-priori by the ER + all-ones-broadcast substrate.
- WB0 prereg (closed-form mu_i with Var/<mu>^2 = Theta(1/K); finite-N band propagated THROUGH the QF nonlinearity; W_syn=0.18mV*(40/K) operating-point-preserving renormalization frozen; dual-coordinate matching required). WB1 analytic reduction. WB2 numeric verification on one existing per-neuron array (slope + through-QF conditional MI). WB3 dual-coordinate re-scoring (soft-dep on P8 sampler; stratify-fallback). WB4 confirmatory degree sweep {20,40,80,160} (non-load-bearing, operating-point-invariance-guarded). WB5 rewrite 8.1.
- **Gates:** `G-P3.1` (slope in [-1.3,-0.7] AND conditional MI <= through-QF band — PRIMARY falsifier). `G-P3.2` (dual-coordinate mouse percentile in [25%,75%]; <=10% too-special CONTRADICTS theorem -> escalate). `G-P3.3` (degree sweep convergence to 50% ~1/sqrt(K); VOID if baseline rate drifts >0.5 Hz — non-load-bearing).
- **Compute:** WB1-3 minutes; WB4 overnight (4 mouse re-runs). **Dependency:** P18 PASS (premise = broadcast is complete write-path); WB3 soft-dep on P8. **Expected:** PASS (deflation by demotion — 8.1 'conserved special in mouse' DELETED, replaced by its negation as a corollary that predicts the embarrassing 46% from first principles). FAIL of G-P3.1 re-opens P1/P18 and REVIVES a possible genuine mouse signal.

### WB-P11 — Wave-P K_p reference-frame recompute (can deflate the 30/30 multi-target headline)
Vina Kd is aqueous-1M-standard-state referenced; clinical EC50 is already bath-referenced; the current `conc_eff = Kp*conc_aq` double-counts the partition the EC50 absorbed.
- WB0 derive the standard state, designate M0 (no-Kp) as DECISION-BINDING, M1 (sqrt-Kp)/M2 (full-Kp) as bracketing bands; hash-lock. WB1 freeze denominator (180 pairs = 30 genes x 6 anesthetics, the realized Gate-C.1 matrix; the 30-vs-25 gap is the P13 NCA hole, flagged not patched). WB2 three-way occupancy recompute (reuse frozen `affinity_to_kd_uM`/`occupancy`; vary ONLY the partition map). WB3 re-evaluate Gate C.1 per model + saturation census.
- **Gates:** `G-P11.1` (M0: >=5/30 genes >10% occ at halothane 1xEC50 survives AND saturation drops materially below M2's 60/180; <5 = headline was Kp-manufactured -> deflate; all-30-still-saturated -> HALT methodological error). `G-P11.2` (frame-fragility: spread n_M2-n_M0 <=2 robust / >=5 fragile). `G-P11.3` (pseudo-test screen: prove >=1 pair clears 10% with no Kp AND >=1 partition-only pair, so both branches are reachable).
- **Compute:** local CPU, seconds (2160 Hill evals). **Dependency:** P7 (sat_pa provenance, since P11 feeds ladder honesty); independent of network family. **Note:** P11 deflation does NOT mechanically flip any V1 quorum (sat_pa ladder is hand-set, decoupled from Vina per phase_g:24) — Wave-P-layer honesty only.

### WB-P4 — Gate-4 (Eger) entailment demotion + SNARE-orthogonal falsifier
Zero Stage2->Stage3 attrition means non-immobilizer discrimination falls out of correct volatile total_pa prediction.
- WB0 write-path grep (shares P18). WB1 prereg hash-lock. WB2 Part-A entailment audit (regress Stage3 max_qf on (total_pa, snare_gain)). WB3 SNARE-only pseudo-NI table (total_pa pinned in non-immobilizer band, ONLY SNARE engaged; NCA dropped — it sums into total_pa). WB4 SNARE-lever-sufficiency positive control (can SNARE alone move QF?). WB5 falsifier run.
- **Gates:** `G4-WB0-writepath` (exactly 1 orthogonal synaptic channel = SNARE). `G4-A-entailment` (R2>=0.95 AND zero attrition -> demote 8.4 from 'independent capability'; DEMONSTRATION only). `G4-B-snare-falsifier` (precondition: SNARE-max moves QF>=0.2 else UNINTERPRETABLE; then pseudo-NI max_qf>=0.5 = Outcome-A demote / <0.5 = Outcome-B narrowly retain independence).
- **Compute:** re-analysis + <1hr sim, local. **Dependency:** P18 (rank-2 / single-orthogonal-SNARE premise). **Expected:** Part-A PASS (literal 2-scalar map); Part-B precondition-fail is a live risk (SNARE scales E and I symmetrically) -> P4 falls back to Part-A demonstration honestly.

### WB-P20 — Genotype x anesthetic two-block reachability partition
The untested interaction term. Verified TWO-block composition: 7 I_ext-routed genotypes ADD into total_pa; 2 Gao genotypes MULTIPLY syn.w (alongside the SNARE multiplier).
- WB0 prereg (name the two blocks + which genotype in which; thresholds). WB1 analytic rank-certificate (DEMOTED from gate to cheap confirmation: block-I = horizontal EC50 translation, block-S = multiplicative). WB2 LIVE battery (9 genotypes x 3 volatiles x 8 doses x 5 seeds; 3 DEFERRED rows excluded). WB3 reachability partition vs held-out literature bands. WB4 multiplicative-interaction check on Gao block (circularity-flagged). WB5 claim relocation + PARK routing.
- **Gates:** `P20-GATE-A` (>=1 genotype STRUCTURALLY-UNREACHABLE = real positive routing to V2/Tier4; all reachable = epistasis-is-bookkeeping deflation). `P20-GATE-B` (Gao ratios match multiplicative prediction, distinguishable from additive null). `P20-GATE-C` (too-special: all 9 ratios <5% error -> circularity flag -> P7 escalation).
- **Compute:** ~2hr LIF slice, local. **Dependency:** P18 (two-block theorem valid only if write-path complete); soft-dep P7 (credits the magnitude arm). **Feeds:** PARK routing -> P1_P2 (expression-routable epistasis) and P14 (gas-1 mitochondrial synergy needs metabolic state).

### WB-P17 — Readout-validity / C-22 gate (can deflate the V7 immobilization-readout semantics; BLOCKS Paper-2 bridge)
Ground V7's AVA-containing command-fraction / 3.0 Hz quiescence readout against held-out causal AVA-silencing + labeled behavioral states.
- WB0 prereg (statistics + velocity-immobility threshold DERIVED as histogram valley (procedure not number) + modality-split rule: NEVER compare Hz to dF/F). WB1 loader (Kato .mat full-load; Atanas NWB streamed). WB2 Q1 data-side directional sign (Kato labeled STATES, not a velocity heuristic; AVA bimodal FWD-low/REV-high positive control for ID correctness). WB3 Q2 causal AVA-HisCl silencing (data half airtight; model half bonus with too-special leak floor). WB4 Q3 threshold-valley (within-modality bimodality existence + model-space self-consistency only). WB5 verdict + page-rewrite gate.
- **Gates:** `Q1-data-sign` (command-low activity tracks behavioral-immobility, s>0 CI excludes 0 in BOTH Kato-labeled AND Atanas-pose). `Q2-causal-AVA-silencing` (HisCl dwell shift matches pre-declared direction; model bonus only). `Q3-threshold-valley` (Kato command bimodal deltaBIC>=10 AND 3.0Hz within +/-1Hz of model's OWN valley).
- **Compute:** Kato ~42-56MB full-load + Atanas streaming, ~few CPU-hr, local. **Dependency:** independent of rank-2 family; shares Atanas loader with P16. **Expected:** PASS keeps readout language + unblocks Paper 2; FAIL (plausible given connectome~shuffled prior + C-22 flag) demotes to 'command-interneuron rate suppression — a network statistic, NOT validated behavioral quiescence', Paper 2 BLOCKED-on-readout.

### WB-P16 — Held-out structure->activity exam (can deflate V1 structural claims to mean-field)
V1 connectome vs STRICT degree-preserving double-edge-swap, scored against Atanas calcium covariance.
- WB0 prereg (GCaMP kernel, global+motion regression, overlap-selection, partial-correlation scoring controlling rate+in/out-degree MANDATORY, n_seeds>=20). WB1 LOAD-BEARING REMEDIATION: NEW strict `permute_double_edge_swap` preserving in+out-degree + weight multiset + per-sign count (do NOT reuse the leaky `permute_configuration` which scrambles in-degree and `+=`-collapses ~4% of edges); invariance unit test BLOCKS the run. WB2 parse (stream 10 NWB). WB3 POSITIVE CONTROL (scorer recovers known structure through the observation bottleneck). WB4 run both glu-sign arms. WB5 verdict.
- **Gates:** `Gate-A` (delta-R = R(real) - R(swap-null), CI-lower > model-internal noise_floor PASS; <= NULL/deflate; R(real)>=0.6 too-special HALT-for-leak; CI straddles 0 INCONCLUSIVE-BY-POWER). `Gate-B` (sign-robustness across both arms). `Gate-C` (positive control recovers structure at R>=0.5 else NULL is weakly-informative).
- **Compute:** parse + ~1640 spontaneous LIF runs, overnight, local (stream 25-27GB NWB, never resident). **Dependency:** GATED ON P8 fly verdict (raises/lowers prior on structural signal) + WB1 unit test. **Expected:** Bayesian-likely NULL (connectome~shuffled per user's d=4/8/16 prior) = clean pre-accepted deflation; a PASS is high-value, replicate on Kato/Zimmer.

---

## PHASE 2 — KEYSTONE BUILD (LOCAL; gated on P18 PASS)

### WB-P1_P2 — Minimal-delta-V2 rank-lift (the single highest-leverage addressable build)
CeNGEN per-class expression vectors x_c replace the all-ones broadcast; operator rank 2 -> min(7,N); Match#3 cell-type-targeting becomes a genuine test. ~90% pre-built (neuron-name<->CeNGEN-class join, per-neuron extraction, name<->index map all exist). Wrapped by SOL7 and SOL8 meta-gates so it cannot be can-only-succeed.

**Core build (SOL1):**
- WB0 PREREG-FREEZE (spread statistic = LABEL-FREE participation ratio PR only; distance-to-template variant FORBIDDEN; SNARE-corrected sampler folds in P8; TPM->[0,1] transform + threshold CSV pinned; complex_i/ii keep x_c=ones by biology; two-sided floors). WB1 build x_c (7x300) from CeNGEN markers in `build_cengen_panel.py` PANEL. WB2 operator edit (`I_ext += alpha * sum_c (-sat_c*e_c) * x_c`; SNARE gated by presynaptic unc-64; with x_c=ones must be BIT-IDENTICAL to V1). WB3 readout re-plumbing (return 300-long q_neuron). WB4 sampler fix + Match#3 statistic. WB5 run worm(300)+fly(2952), mouse EXCLUDED (random graph = P3). WB6 rank-lift-bought-something check.
- **Gates:** `G0_RANK_LIFT_REALIZED` (two equal-(total_pa,snare_gain) profiles -> q_neuron L2>1e-6; gates G1). `G1_MATCH3_SPATIAL_SPECIAL` (conserved PR percentile in (1%,10%] PASS = first non-by-construction cell-type-targeting positive; >10% NULL/deflate -> 'does NOT establish' ledger; <=1% too-special leak). `G2_MATCH3_NOT_ENTAILED` (Var(PR over surrogates)>1e-9, not pinned by magnitude).

**SOL7 meta-gate (freeze-statistic-first harness):**
- WB0 freeze spread statistic S in CLOSED FORM (between-cell-type variance fraction eta^2 of the per-neuron input) BEFORE any run. WB1 able-to-fail screen (disjoint-support profiles differ, identical-support same). WB2 SNARE-corrected null. WB3 run gate. WB4 entailment/attrition. WB5 within-block surrogate (conditional, only if floor fires).
- **Gates:** `G1_able_to_fail_screen` (rank-lift mechanically succeeded + S is clean). `G2_spread_percentile_two_sided` (P3_org<=30% positive, <=5% floor->within-block surrogate, >30% TESTED-AND-NULL — strictly better than V1's NOT_TESTED). `G3_independence_from_match2` (Spearman(magnitude-pct, spread-pct)<0.95 AND attrition>0, else demote to entailed-by-magnitude).

**SOL8 external NeuroPAL grounding (streaming; leaf-node corroboration):**
- WB0 prereg (re-stamp SOL1 threshold hash; bout rule = population-activity-percentile valley; structured null = gene-marginal-preserving label permutation; split FAIL-tree). WB1 NeuroPAL connectome-keying. WB2 V1 Jaccard correctness. WB3 bout-yield pre-check. WB4 build S_V2 + measured-drop. WB5 run with STRUCTURED null. WB6 verdict.
- **Gates:** `G1_neuropal_join_correctness` (per-class Jaccard>=0.5 for >=5/7 classes; HARD — fail HALTS + re-freeze x_c, every downstream V2 percentile suspect). `G2_quiescence_structure_confirmatory` (Spearman(S_V2, natural-quiescence-drop) CI>0 above structured null; ONE-DIRECTIONAL — null is 'anesthetic and natural quiescence spatially distinct', a Paper-2 finding, NOT a refutation). `G3_bout_yield_feasibility_fork` (n_worms>=6 full / 3-5 descriptive / <3 park-G2).

- **Compute:** build minutes; Match#3 run ~4-5 CPU-hr; SOL8 streaming ~1-2hr. All local. **Dependency:** P18 PASS (necessity of the lift); folds in P8 sampler; SOL7/SOL8 parasitic on SOL1 merge — WB0 hash-locks MUST precede runs. **Expected:** fly is the a-priori payoff (WF1: identity does measurable work there); worm NULL plausible (overfit + connectome~shuffled prior); every branch ships a committed hash-stamped verdict, claim earned or narrowed, never retuned.

---

## PHASE 3 — GREENE HPC + LEAVES

### WB-P13 — NCA-1 / UNC-80 structures (Greene) + nca magnitude (LOCAL, independent)
Two decoupled halves.

**Magnitude (SOL28, LOCAL, run in Phase 1/2 — do NOT wait on Greene):** the load-bearing question is whether nca_block's docking-orphaned 40 pA matters to the quorum.
- WB0 prereg + provenance freeze. WB1 derive interval from FROZEN passive constants: I_block = block_fraction * g_leak(10nS) * D(15mV) -> [75,120] pA (note: 40 pA legacy sits BELOW lo, so the interval cannot be accused of trivially including it). WB2 freeze the 7 non-nca scalars + override only nca. WB3 RE-SIMULATE the 64 nca-containing worm subsets x 3 endpoints (nonlinear LIF — cannot recompute arithmetically). WB4 verdict.
- **Gates:** `G1-interval-provenance` (interval = closed-form of frozen disk constants, byte-identical on re-derivation). `G2-quorum-survival` (at nca=75 pA, passing nca-quorum subsets >=80% of baseline AND SNARE-OR-ComplexI universality holds; else the quorum was resting on a hand-set scalar -> deflate). `G3-substitution-leak-screen` (7 non-nca scalars byte-identical across endpoints).
- **Compute:** hours-to-overnight, local CPU.

**Structure (SOL27, GREENE):** ESMFold/AlphaFold-multimer of NCA-1 (Q6Q762) + UNC-80 (Q9XV66) — UNC-80 OOM'd on 8GB, needs >=24GB GPU; paralog-bridge fallback (NCA-2 already folded).
- **Gate:** pLDDT pocket-confidence gate decides whether docking proceeds; re-dock is CPU (back to local). Structure-availability half only — magnitude settled by SOL28. NALCN has no published Kd, so nca_block stays structurally uncalibratable (Tier4-adjacent, parked residue).
- **Compute:** one short Greene GPU job + local Vina.

### WB-P19 — Gap-on vs gap-off confirmatory (LOCAL, lowest severity, run last)
- WB0 prereg (primary = max_d|QF_gapon-QF_gapoff|, tau=0.10; bracket-only, no alpha re-fit). WB1 config-flagged voltage monitor (assert only include_gap + monitor changed). WB2 run frozen halothane dose-response twice/dose x seeds. WB3 secondary eps comparison (soft-dep on P3/SOL1 eps_MF; DEFERRED if unavailable). WB4 adjudicate.
- **Gates:** `G1-QF-claim-impact` (max_d|QF diff|<0.10 PASS = gap-inert footnote; >=0.10 FAIL = report worm residual as gap-on/gap-off bracket). `G2-eps-consistency` (max_d|eps_emp-eps_MF|<0.10 corroborates SOL1 bound, else retract SOL1 bound; soft-dependent).
- **Compute:** ~48 short LIF runs, <1hr, local. **Dependency:** schedule AFTER load-bearing fixes; leaf (nothing downstream). **Expected:** PASS (gap inert by WF1 A17 data-processing-inequality argument) -> worm numbers stand unbracketed.

---

## PARKED — full-energetic-Tier4 (user decision, NOT scheduled)

### P14 — Thermodynamic NECESSITY of reversible reduced-activity states (sec 8.7 / C3 entropy framework)
Genuinely substrate-gated. V1 has one voltage scalar/neuron + static current — no conserved charge, no k_B T, no moving Nernst, no metabolic budget, no lipid/aqueous coordinate. A nats-valued entropy without k_B T + a charge ledger CANNOT distinguish reversible quiescence from collapse/death, and 'reversible' is the whole claim.

**Minimal lifting build (the 'V8' substrate, strictly > minimal-delta-V2):** per-neuron Na/K ion ledger + Na/K-ATPase pump current + joule-valued metabolic free-energy budget + lipid/aqueous partition coordinate + state-continuation harness for hysteresis. With it: lipid-partitioning becomes a DERIVED Meyer-Overton result (retroactively grounds P5/P11), reversibility becomes a measurable bifurcation/hysteresis property, and joule-valued Schnakenberg sigma_EP becomes definable on a cyclic >=2-D observable. **Honest boundary:** do NOT relabel the dynamical-systems quasi-potential (Phi = -D log p_ss, fixed 6 mV noise) as thermodynamic free energy.

**Compute:** this is the build that would justify reserving Greene for heavy ion-pool + metabolic-state integration. P20's gas-1 PARK and P13's nca-calibration residue both feed this same Tier4 build. Substrate-free residue available now (entropy-sense hygiene + impossibility-boundary theorem) strengthens the BOUNDARY, not the claim — documentation only.

---

## Open risks (summary)
1. **P18 single-point-of-failure** — a hidden writer re-scopes the whole build program (~85% PASS prior; FAIL pre-accepted, would REVIVE some positives). Do first; build nothing until it lands.
2. **Held-out NULL is Bayesian-likely** for P16 + SOL8-G2 + worm Match#3 (user's connectome~shuffled prior). Payoff organism is fly. Pre-accepted, publishable.
3. **P8 can remove the last clean positive** (fly DEFLATE). Intended anti-target-engineering outcome.
4. **Atanas NWB are ~25-27GB each** (not 194M) — mandatory h5py streaming; budget P16/P17/SOL8 as hours of I/O.
5. **P7 likely UNDECIDABLE** for sat_pa (pre-first-commit tuning unobservable) — narrows the honesty claim regardless.
6. **Gate-power** — SOL8-G3 / P17 feasibility forks may ship UNDERPOWERED (sparse bouts, ~40-60/300 NeuroPAL-named). Park-to-fewer-worms branches handle it.
7. **P11->P10 is moot** — sat_pa ladder is hand-set, decoupled from Vina; P11 deflation is Wave-P-layer honesty only, no downstream network flip.
8. **P13 magnitude vs structure** — Greene fold resolves dockability only; NALCN has no Kd, nca stays uncalibratable (parked). SOL28 settles whether magnitude matters, locally, first.
9. **SOL7/SOL8 parasitism** — enforce WB0 hash-lock (freeze S in closed form) as a hard predecessor or the V2 result becomes can-only-succeed.
