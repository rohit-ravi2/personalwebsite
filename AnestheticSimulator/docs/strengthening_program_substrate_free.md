# Substrate-Free Strengthening Program — scoping verdict (2026-06-11)

**Question (user):** Working on all fronts incl. the entropy framework, from first principles only
(no literature), is there a real fleshed-out way to maximally strengthen the AnestheticSimulator
claims WITHOUT the Tier4 substrate upgrade — or is the substrate the single and only way?

**Outcome:** STRENGTHENING_PATH_EXISTS (binary, per the preregistered termination criteria).
Produced by a 79-agent first-principles scoping workflow (run wf_b18e20f9-65f): 20 avenues
considered, 4 survivors; two completeness rounds added 7 further avenues, 0 survived → survivor
set is stable / space converged.

---

## Executive summary

A real, substrate-independent strengthening path exists, and it is not marginal — it is the correct and only place to state what the V1/V6 claims can and cannot mean. The decisive fact, verified directly in source (phase_g_state_validator.py:245, 308-335), is that the entire pharmacological forward map factors through a 2-dimensional sufficient statistic: a single uniform hyperpolarizing current total_pa (every non-SNARE class summed and broadcast identically to all N neurons) plus a single global synaptic gain snare_gain (SNARE's lone multiplicative channel). apply_genotype factors through the same two scalars. This is a rank-2 operator, provable by elementary functional composition with zero simulation.

From that one verified premise, four high-power avenues survive the adversarial filter, and together they (a) convert the §8.3/Match#3 "not tested" deviation into a one-line PROVEN impossibility (cell-type spread is the constant all-ones vector for every profile → I(spread;readout|count,magnitude)=0 exactly); (b) convert the mouse P7 "embarrassing null" (46% median) into an a-priori PREDICTED consequence of mean-field collapse of a rank-2 operator on a structureless graph — a scope correction, not a refutation; (c) put a calibrated, CI-bearing effect size (ΔR² class-identity-beyond-magnitude) under every percentile, isolating the fly result (A3) as the project's single quantified positive and stress-testing it with a degree-preserving connectome shuffle that is a genuine accept-either-way falsifier; (d) run the knapsack/subset-sum + magnitude-matched combinatorial null that honestly decomposes the EARNED 4-6-class quorum and SNARE-OR-ComplexI universality into magnitude-bookkeeping vs class-identity BEFORE promotion — the disclosure that pre-empts the strongest reviewer objection. None of this needs Tier4; all of it needs the substrate to be exactly what it is, because the whole point is to characterize the forward map V1 already implements.

The crucial honest boundary: Tier4 IS the single and only way for the POSITIVE versions of the by-construction limits — cell-type-resolved targeting (a positive Match#3 result), the mouse class-identity positive, and the metabolic/joule-valued half of the thermodynamic-necessity claim. These are bounded by the rank-2 + no-ion-pool structure as theorems; no re-analysis can exceed an information bound the forward map already destroyed. So "Tier4 is the only way" is true ONLY for raising the operator rank and for adding energetic state variables. For everything the earned claims actually assert, the substrate-free program is not just possible — it is the rigorous, non-target-engineering way to maximally strengthen them, and several pieces would weaken the marketing claim down to exactly what is licensed, which is the opposite of target-engineering.

---

## Decisive reasoning

The verdict is forced by a single asymmetry between two kinds of question the rank-2 operator creates. The forward map provably destroys all information beyond (total_pa, snare_gain): any two 7-class profiles mapping to the same pair produce bit-identical I_ext and weight arrays, hence identical trajectories, EC50s, and percentiles. This is verified, not assumed, against phase_g_state_validator.py:245/308-335.

That destruction cuts cleanly in two directions. For POSITIVE questions that require a higher-dimensional fingerprint than the 2-D summary — cell-type-resolved targeting (a positive Match#3), mouse class-identity beyond magnitude, the joule-valued metabolic-necessity claim — no re-analysis can recover information the map already annihilated. These are bounded by an information ceiling (data-processing inequality) and by the absence of energetic state variables, as THEOREMS. For these, and ONLY these, Tier4 (raising operator rank via a per-neuron expression vector; adding ion/metabolic state) is the single and only way. The orchestrator's 'is it impossible short of Tier4?' is therefore TRUE for exactly this family.

But for CHARACTERIZING what the operator already computed — which is precisely what the EARNED claims assert and what the limitations need honestly stated — the rank-2 destruction is not an obstacle, it is the subject. Proving Match#3≡Match#2 is an algebraic identity (the spread coordinate is a constant vector → zero conditional MI). Predicting mouse P7 is a mean-field corollary. Decomposing the quorum is knapsack arithmetic on the frozen table. Localizing fly's signal is a degree-preserving shuffle on the existing connectome. Every one of these consumes only frozen code + frozen CSVs + at most one config-flag re-run of V1 as-is, and several would WEAKEN the headline claim to exactly what is licensed — which is the definitional opposite of target-engineering. The adversarial filter confirmed four such avenues at 'high' strengthening power while rejecting every avenue whose strengthening lived on the energetic side or was a redundant reframe.

Therefore the binary is decisively STRENGTHENING_PATH_EXISTS: a complete, fleshed-out, multi-front program strengthens the real claims without Tier4. The blanket statement 'anything short of Tier4 is impossible' is FALSE — it holds only for the three by-construction positive limits and the metabolic-entropy measurement, which the program itself names and assigns to Tier4 with a proof of why. The honest framing is not 'Tier4 or nothing' but 'Tier4 for the rank/energetics ceiling; rigorous re-analysis and theory for everything the controls already touched, including bringing the Thesis line into exact agreement with what the operator licenses.'

---

## Entropy / thermodynamic framework verdict (dedicated)

SPLIT, and the split is sharp. The thermodynamic/entropy framework CANNOT be substantiated from first principles without the substrate in the sense that matters to the named claim (C3: thermodynamic NECESSITY of reversible reduced-activity states, with anesthetic susceptibility as its price). That claim is modal and energetic — it requires a free-energy budget in joules, a Na/K-ATPase ion/charge ledger, a metabolic state variable, and a lipid/aqueous partition coordinate to even be REPRESENTABLE. V1 has none: the state vector is one voltage scalar per neuron plus a static injected current; there is no conserved charge, no kT, no moving Nernst, no lipid degree of freedom. Every entropy-framework avenue that tried to deliver the energetic core failed the filter for the SAME reason: A11 (quasi-potential), A12 (Landauer bridge), A13 (Na/K-ATPase budget), A19 (Boltzmann effective-temperature) all equivocate between a dynamical-systems quasi-potential (Φ = -D log p_ss, real but with D = the fixed 6 mV noise, NOT a kT or metabolic free energy) and a thermodynamic free energy with energetic content. Strip the equivocation and the necessity claim is exactly where §8.7 left it: substrate-gated. The single honest, decisive reason: a nats-valued information entropy without a k_B T and a charge ledger cannot distinguish reversible quiescence from collapse/death — and the "reversible" qualifier is the entire load-bearing content of the necessity claim.

What the framework CAN gain substrate-free is strictly bounded and must be sold as honesty/hygiene, not as the theory: (1) the entropy-SENSE hygiene result — proving that spectral entropy (a signal-complexity statistic) and any thermodynamic entropy-production rate are different objects with different units, so the framework stops equivocating; (2) the impossibility/boundary result — proving precisely WHY V1 cannot host the joule-valued claim (rank-of-state-vector argument; power_balance.py's ohmic-dissipation calculation is the proof the missing piece is conductance/ion state and nothing more exotic), which upgrades §8.7 from a bare disclaimer to a precise theorem about the gap. Both are real but both STRENGTHEN THE BOUNDARY, not the claim. The Schnakenberg entropy-production avenue (A9) survived one lens but its sign is largely pre-ordained by the rank-1 bifurcation already established, and the rasters it needs were never persisted — so even the descriptive operationalization is weaker than it looks and at best a paired confirmation, not a discovery. Net entropy verdict: the necessity theorem is in-principle analytic, but on THIS project the energetic instantiation is genuinely Tier4-gated; the only substrate-free entropy deliverables are scope-tightening and the impossibility boundary, and they should be labeled as such rather than dressed as a thermodynamic theory.

---

## Surviving avenues (ranked)

- S1 — Factorization theorem (QF = G(total_pa, snare_gain)) + forward-map sufficiency regression + identity-beyond-magnitude ΔR² with CIs: one coherent move that proves Match#3≡Match#2, predicts mouse P7 a priori, and replaces every percentile with a calibrated effect size. Highest strengthening power; the spine all others hang on.
- S2 — Knapsack/subset-sum theorem + magnitude-matched combinatorial null per subset size: decomposes the EARNED 4-6-class quorum and SNARE-OR-ComplexI universality into magnitude-arithmetic vs class-identity, the disclosure A1/A2 need before promotion. Pure arithmetic on the frozen table; deflates-or-defends honestly.
- S3 — Mean-field collapse theorem (derives mouse P7) + degree-preserving connectome shuffle localizing the fly A3 signal to the residual: the single highest-information experiment, a genuine accept-either-way falsifier on the existing fly/worm connectomes (one edge-permutation, no substrate change).
- S4 — Promote Match#3 from logged DEVIATION to PROVEN-UNDECIDABLE-ON-V1 + re-scope every 'conserved-target' statement to the SUFFICIENCY claim the rank-2 operator licenses, with class-identity credit quarantined to fly: brings the Thesis line into exact agreement with the controls (documentation + the S1/S3 analyses).

---

# Complete Strengthening Program — AnestheticSimulator, substrate-free (no Tier4)

**Organizing principle.** One verified mechanical fact governs everything: the V1 perturbation operator is **rank-2** — `total_pa` (uniform current, all non-SNARE classes summed and broadcast identically) and `snare_gain` (the lone multiplicative synaptic channel). `apply_genotype` factors through the same two scalars. Verified in `phase_g_state_validator.py:245, 260-283, 308-335`. Every survivor is a theorem about, or an analysis conditioned on, this operator. The program is ranked by strengthening power; the entropy front is treated explicitly and honestly as boundary-work, not theory-delivery.

---

## FRONT 1 (highest power) — The Factorization Theorem spine (avenue S1)

**Deliverable:** `docs/factorization_theorem.md` + `artifacts/sufficiency/`.

**Step 0 — Freeze the operator as premises (transcription, no runs).** Record verbatim: (a) `resolve_target_neurons → list(range(brain.N))`, `mechanism_class` never read; (b) `total_pa = alpha * Σ_{c≠snare} (-sat_c · e_c(d))` written identically to `I_ext[:]`; (c) SNARE the sole separate channel, scalar `factor = 1+(snare_max-1)·e_snare` on all weights; (d) genotype path identical. Define `S(profile,d) = (total_pa(d), snare_gain(d))`. **This fixes every quantity before any outcome is read — the procedural anti-target-engineering guarantee.**

**Step 1 — Prove the theorem (½ page, analytic).** For any P, P' with S(P,d)=S(P',d) at all tested d, the Brian2 input arrays are bit-identical → same seed → identical trajectory, QF, EC50. Hence **QF(profile,d) = G(total_pa(d), snare_gain(d))** for a single fixed scalar-functional G.
- *Corollary 1 (Match#3 collapse):* cell-type spread is the constant all-ones vector for every profile → zero bits → **Match#3 ≡ Match#2 algebraically**, not approximately.
- *Corollary 2 (sigmoid forced):* sum of monotone Hills through monotone G ⇒ dose-response is necessarily sigmoidal ⇒ **sigmoid SHAPE is not evidence; only crossing LOCATION is** (sharpens A4).
- *Corollary 3 (mouse a priori):* structureless graph + uniform input ⇒ G depends on total_pa alone ⇒ class identity is a sufficient statistic only via magnitude ⇒ **mouse-at-median is predicted, not a null** (reframes B1).

**Step 2 — Arithmetic extractor on frozen data (no sim).** For every ensemble row (conserved + 50 Match#1 + 50 Match#2 per organism), compute `total_pa(d)` and `snare_gain(d)` from the Hill tables. **Critical audit to log:** the existing Match#2 sampler (`_draw_random_profile_match2`) matched on `agg_pa` ONLY — it EXCLUDES SNARE. Report the spread of `snare_gain` across Match#2 ensembles; if non-trivial, the existing control left one sufficient-statistic coordinate free (a real, fixable gap). Artifact: `S_table_{organism}.csv`.

**Step 3 — Empirically confirm sufficiency.** Fit one low-capacity monotone surrogate `G_hat: (total_pa, snare_gain) → QF` per organism (isotonic / 2-input monotone GAM). Regress simulated QF on `G_hat(S)`; theorem predicts R²→1 up to seed noise. Estimate the seed-noise floor from a handful of same-profile-different-seed reruns (existing substrate, unchanged) or off existing repeated rows. Artifact: `sufficiency_fit_{organism}.json`.

**Step 4 — The decisive regression (identity beyond magnitude).** Two nested models per organism: Model A (2-D sufficient statistic only) vs Model B (+ class-presence indicators + per-class engagement-area terms). Likelihood-ratio / partial-F with bootstrap CI over the 50 ensembles gives the **exact extra variance class identity explains beyond magnitude**. Prediction: mouse ΔR²≈0 (CI spans 0); fly ΔR²>0 (the A3 signal, now a calibrated effect size, not a bare 4.76%). Artifact: `identity_beyond_magnitude_{organism}.json`.

**Step 5 — Re-coordinate the three headline results.** Match#3 → one-line proof of equality; mouse P7 → predicted, ΔR²≈0±CI; fly A3 → ΔR²=Z±W>0, the unique organism where identity does measurable work. Flag (do not yet run) that fly's increment must live in the snare_gain/connectome channel — the hook for Front 3.

**Step 6 — Knapsack honesty check (folds in S2 minimal form).** See Front 2.

**Step 7 — Buckingham-Π audit.** List dimensionful constants (concentration via Hill, sat_pa [pA], C_mem, tau, threshold gap [mV], 3 Hz cutoff). Count independent dimensionless groups; verify "one free α." **Caveat (from adversarial filter):** Π counts DIMENSIONAL not FITTED DOF — it can ENUMERATE a candidate hidden group (e.g. threshold-gap/sat_pa-scale) but cannot adjudicate whether it was tuned. Run it as Π-derivation + a code/prereg provenance check of whether the 3 Hz cutoff and sat_pa ladder were pinned a priori; pre-register the scoping boundary so the boundary itself does not become a free knob.

**Step 8 — Assemble** with an explicit scope box: PROVEN vs DEFLATED vs STILL-NEEDS-TIER4. Cross-link to `v7_final_summary.md` §8.

---

## FRONT 2 (high power) — Knapsack / combinatorial null for the EARNED quorum (avenue S2)

**Deliverable:** `artifacts/sufficiency/knapsack_null.json` + `docs/quorum_knapsack_verdict.md`. **This is the disclosure A1/A2 need before promotion; run it FIRST among the empirical pieces — it could partially deflate the headline.**

**Step 0 — Prereg the inputs and ONE acceptance statistic before computing** (procedural guard): freeze the ladder `{complex_i 60, snare 50, nca 40, k2p/gaba/nachr 30, complex_ii/glucl 20}`, frozen α, halothane Hill rows, MAC targets, the 2× band, and the decision that the verdict (fraction of magnitude-matched random size-k subsets in band vs conserved) is accepted whatever it is.

**Step 1-2 — Reconstruct the threshold band F̂ from frozen data.** The frozen CSV already gives QF at 8 doses for all 317 subsets. Pool `(total_pa, QF)` points; fit monotone F̂ (stratify SNARE-present vs SNARE-absent — that stratification IS the A2/SNARE-separation lever). The 2× rule maps to a `total_pa` band `[T_lo,T_hi]`.

**Step 3 — Analytic minimum cardinality k*.** Greedy subset-sum on the sorted per-class currents `alpha·sat_c·e_c(MAC)` (worm: ComplexI 4.57, NCA 2.76, K2P 2.25, nAChR 2.14, GABA 1.98, GluCl 1.32, ComplexII 1.19 pA). Three diagnostic outcomes: empirical_min == k_greedy ⇒ quorum size is ARITHMETIC (narrow A1); empirical_min > k_greedy ⇒ network rejects magnitude-sufficient subsets = real work, with k_greedy as a derived floor; empirical_min < k_greedy ⇒ SNARE synaptic channel supplies threshold-crossing current cannot = SNARE structurally load-bearing. Explain fly min=6 vs worm min=4 at smaller α from band+ladder, not assertion.

**Step 4 — Full 2^n analytic enumeration vs simulated ground truth.** Predict passes_stage1 for every subset via F̂; join to frozen `v7_stage1_halothane.csv`; confusion matrix. High concordance ⇒ subset search added little beyond knapsack arithmetic. Low concordance localizes exactly which subsets the connectome treats differently = the worm/fly residual targets.

**Step 5 — Magnitude-matched combinatorial null per size k** (the derived per-k generalization of P7): `p_conserved(k)` vs `p_random(k)` with sat_pa multiset held fixed, permuted across labels. If equal for all k ⇒ identity adds nothing beyond count+magnitude in every organism (mouse P7 generalized). If `p_conserved(k) > p_random(k)` in worm/fly but not mouse ⇒ the load-bearing positive, per-k effect size + bootstrap CI.

**Step 6 — Separate the two reasons behind SNARE-OR-ComplexI "100%."** **Grounding correction surfaced:** in frozen `v7_stage1_halothane.csv`, ComplexI is in 32/33 passers; the lone exception is the all-six-non-ComplexI worm set carried by SNARE — so the lever is precisely "ComplexI-OR-SNARE" with ComplexI doing nearly all the work. (a) Remove ComplexI from the ladder, recompute k_greedy and analytic passers; if minimum jumps and passer count collapses, ComplexI universality is LARGEST-SUMMAND arithmetic. (b) **The one substrate-touching step (still V1, a config flag, NOT Tier4):** re-run the existing `v7_subset_search.py` with SNARE forced out of the candidate pool (~63-127 subsets/organism, existing brains); if a pure-current subset still reaches band on worm/fly, SNARE is substitutable; if not, its synaptic channel supplies something current cannot.

**Step 7 — Mouse mean-field falsification** (closes to B1, see Front 3). **Step 8 — Verdict table + rewrite A1/A2 language to exactly what survived.**

**Conditional-on-ladder caveat (must be stated):** the whole knapsack is conditional on the assumed sat_pa magnitudes; report ±50% OAT sensitivity on the analytic knapsack (cheap) but do NOT claim the ladder is biologically correct — that is a separate, possibly substrate-gated question.

---

## FRONT 3 (high power) — Mean-field theorem + fly connectome-shuffle falsifier (avenue S3)

**Deliverable:** `docs/A3_meanfield_residual_results.md` + `artifacts/a3/`. **The single highest-information experiment; a genuine accept-either-way test of the project's last clean positive.**

**Step 0 — FREEZE THE FALSIFIER BEFORE TOUCHING DATA.** Fix from theory: (i) fly's signal declared connectome-borne only if the Match#2 percentile gap is reduced ≥50% under degree-preserving shuffle AND the per-dose residual drops into the finite-N band; (ii) the finite-N band = std of `(r_sim - r_MF)` measured on the MOUSE graph (where theory says residual MUST be noise), NOT fitted on fly; (iii) verdict table maps every outcome in advance. Accept whichever lands.

**Step 1 — Derive the mean-field reduction (pen-and-paper).** From the LIF equation + the rank-2 operator, on a degree-homogeneous graph every neuron sees the same mean input `mu = K·W_syn·s·⟨w⟩·r + total_pa`; self-consistent `r* = Φ(mu(r*), sigma_eff)`; cell-to-cell deviations O(1/√K), O(1/√N). Result: **QF is an exact function of (total_pa, s) and nothing else** on a structureless graph.

**Step 2 — Mouse P7 as corollary** (a priori): magnitude-matched randoms share (total_pa, s) ⇒ identical QF ⇒ conserved profile at the median by construction. Re-label B1 everywhere from "null / hypothesis fails" to "scope limit: untestable on a structureless graph under a rank-2 operator."

**Step 3 — Implement the 1-population solver, validate on mouse** (frozen data). `analysis/a3_meanfield.py` using the brains' own constants. PASS = `r_sim - r_MF` is zero-mean noise at seed-variance scale; this both confirms the theorem and CALIBRATES the Step-0 band.

**Step 4 — Worm/fly residual decomposition** (frozen, no re-run). Apply the same transfer to Cook2019/Winding2023 trajectories; form `R = QF_sim - QF_MF`. Decompose fly's 5.56%→4.76% tightening into a mean-field (magnitude) component and a residual (connectome) component. Same for worm (also quantifies how much of worm's 0% is mean-field-magnitude vs anchor-overfit).

**Step 5 — The decisive re-run: degree-preserving connectome shuffle** (cheap, in-scope, NOT Tier4). Double-edge-swap holding in/out-degree sequences exactly fixed (so `QF_MF` is invariant by construction; only R can move). Re-run ONLY the fly Match#2 percentile pipeline on the shuffled connectome, ≥10 shuffle seeds. One edge permutation + existing `v7_random_ensemble` code — no HH channels, no ion pools, no CeNGEN tables.

**Step 6 — Apply the frozen verdict.** Outcome A (signal survives shuffle): fly result is degree-statistics + finite-N, NOT connectome-structural → **WEAKEN A3 toward artifact**. Outcome B (signal vanishes under shuffle): fly identity IS carried by specific wiring routing SNARE differently than uniform current → **A3 upgraded to a connectome-structural signal with a passed falsification; B5 confirmed precisely.** Outcome A is live and would deflate the project's last clean positive — which is exactly why the test is honest.

**Step 7 — Cross-check against the user's own C. elegans "connectome ≈ shuffled" prior** (d=4/8/16 calcium null): strong Bayesian prior that worm collapses to mean-field + anchor-overfit, making the FLY shuffle the single highest-information experiment.

**Step 8 — Information-ceiling corollary** (free rigor): `I(7-class profile; readout) ≤ H(total_pa, s)` — the validator cannot resolve more than 2 effective perturbation DOF on a homogeneous graph; cell-type targeting (operator rank↑) is the ONLY lever that raises it. Bounds V1 and motivates V8 without overclaiming either.

**Residual limit to state:** "connectome-structural" is strictly weaker than "biologically conserved-target"; even Outcome B shows the Winding2023 graph interacts with the magnitude ladder, not that the class LIST is privileged. n=3 fly passers, 0.8 pp gap — report effect size + CI, no p-value theater.

---

## FRONT 4 (high power, mostly documentation) — Re-scope the Thesis to the SUFFICIENCY claim (avenue S4)

**Deliverable:** edits to `v7_final_summary.md` §7, `anesthesia-pipeline.mdx` Thesis line; a verdict-JSON diff changing Match#3 from `V7-DEVIATION/P8_unfilled` to `PROVEN-UNDECIDABLE-ON-V1`.

- **Replace the Thesis line** with the sufficiency form: "...reproduced by a coordinated partial perturbation reaching an aggregate threshold (total_pa, s) crossable only by a QUORUM of ≥4-6 magnitude-contributing classes at frozen α — no single-target event is sufficient. The model does NOT establish that the specific CONSERVED identities are privileged beyond aggregate magnitude, EXCEPT in fly (the lone organism whose structured connectome lets SNARE route differently). Mouse sits at the magnitude-matched median by construction; worm class-specificity is anchor-confounded."
- **Demote the adjective "conserved" to "multi-class"** in the headline; keep "conserved-target" only inside the fly-scoped sentence.
- **Add the impossibility statement:** "Cell-type-resolved targeting is not under-tested; it is provably undecidable on this substrate (rank-2 operator), and a positive test requires raising operator rank — a per-neuron CeNGEN expression weighting, NOT more data."
- **Scope what Tier4 buys, in operator terms** (doubles as substrate justification): the per-neuron expression vector replaces `1_N` with class-specific `x_c ∈ R^N`, raising operator rank from 2 to ≤min(7,N); ONLY then is `I(spread;readout|count,mag) > 0` possible. Note the EXACT minimal delta lifting the impossibility is a length-N expression weighting on the existing `I_ext`/`w` writes on the already-real worm/fly connectomes — strictly smaller than full Tier4 (no HH channels, ion pools, or metabolic state needed for THIS claim).
- **Audit owed before publishing the theorem as absolute:** grep every write to `brain.neurons.I_ext` and `syn.w` across the codebase to confirm `apply_anesthetic`/`apply_genotype` are the complete write-path (rules out hidden per-neuron state that would weaken rank-2).

---

## ENTROPY / THERMODYNAMIC FRONT (explicit) — boundary-work only, honestly labeled

The named claim (C3: thermodynamic NECESSITY of reversible reduced-activity states) is **genuinely Tier4-gated** and the program does NOT pretend otherwise. What is delivered substrate-free here is strictly scope-tightening:

1. **Entropy-sense hygiene (do this — free, high-value honesty).** Prove that spectral entropy (Phase J, a signal-complexity statistic, currently SCAFFOLD/PENDING) and any thermodynamic entropy-production rate are different objects, different units, different claims. State it cleanly so the framework stops equivocating. A spectral-entropy decrease is the SHADOW the theory would predict, not the theory; even fully implemented it adds nothing the firing-suppression readout doesn't already imply and has no necessity content.

2. **The impossibility boundary (do this — upgrades §8.7 from disclaimer to theorem).** Prove WHY V1 cannot host the joule-valued claim: rank-of-state-vector argument (one voltage scalar per neuron; no conserved charge, no kT, no moving Nernst, no lipid coordinate). `power_balance.py`'s ohmic-dissipation calculation (the one real first-principles energetics that ran: 7e6-3e7 ATP/sec, 4.5× spread) is the PROOF the missing piece is conductance/ion state and "nothing more exotic." This strengthens the BOUNDARY, not the claim — label it as such.

3. **Schnakenberg entropy-production (OPTIONAL, paired only).** If pursued, run it PAIRED with the Front-3 mean-field/bifurcation analysis so the team can attribute how much of the monotone-σ-decrease is analytically pre-ordained by the rank-1 bifurcation vs empirically informative. Two hard caveats from the filter: (i) the sign is largely forced (probability currents collapse as firing collapses), so a peak at EC50 would be the only genuinely informative outcome; (ii) **the frozen rasters it needs were never persisted** — so this is not pure re-analysis; it requires re-running Phase G/J with raster capture (still V1, no substrate change, but not free). Down-scope to: report σ_EP strictly as the inequality + the parameter-free same-sign prediction + partition-sensitivity disclosure. Do NOT sell it as an independent discovery or a thermodynamic state-function.

**Everything energetic beyond this — Landauer joule-bridge, Na/K-ATPase budget, Boltzmann effective-temperature, quasi-potential with energy units, hysteresis/reversibility-as-bifurcation — is Tier4-gated** (or rests on a false theorem; A11/A20 wrongly localize irreversibility to SNARE — a recurrent net under uniform constant current can be bistable with fixed weights, and run_single builds a fresh brain per dose so hysteresis area is identically zero without a new state-continuation harness). Assign these to V8 with the proof of why, do not relabel a dynamical-systems quasi-potential as thermodynamic free energy.

---

## EXECUTION ORDER (ranked by strengthening power, with deflation-risk run first)

1. **S2 knapsack null** (Front 2) — run FIRST; pure arithmetic; could partially deflate the headline, so honesty demands it precede promotion.
2. **S1 factorization theorem + sufficiency regression** (Front 1) — the spine; analytic + frozen-data; converts Match#3/mouse into theorems and puts CIs under every percentile.
3. **S3 mean-field + fly shuffle** (Front 3) — the one decisive accept-either-way experiment on the project's last positive.
4. **S4 re-scope + entropy boundary-work** (Front 4 + entropy front) — documentation + hygiene + the impossibility theorem; brings the Thesis into exact agreement with the controls.

**What this program does NOT do, and assigns to Tier4 with proof:** any positive cell-type-targeting result (Match#3), the mouse class-identity positive, the joule-valued metabolic-necessity claim, lipid/aqueous partitioning as a derived (not assumed) result. These are bounded by the rank-2 + no-ion-pool structure as theorems — the substrate IS the only way for them, and the program proves exactly why, which is itself the cleanest possible justification for building it.

---

## Rejected avenues — why they failed

The rejections cluster into three honest failure modes. (1) SECRETLY REQUIRES THE SUBSTRATE — the entire energetic/thermodynamic-necessity cluster: A11 (quasi-potential), A12 (Landauer cost bridge), A13 (Na/K-ATPase free-energy budget), A20 (operator-level reversibility theorem), and A15 (critical-slowing) all need a joule scale, a charge/ion ledger, a metabolic state variable, or a slow recovery variable that V1 structurally lacks; their substrate-free residue is either a dynamical-systems result mislabeled as thermodynamics, or a boundary-marker that DEFINES the substrate requirement rather than removing it. A11/A20 additionally rest on a false theorem (a recurrent net under uniform constant current CAN be bistable/hysteretic with fixed weights, so irreversibility is not SNARE-localized and the dose-response is path-independent by construction — run_single builds a fresh brain per dose, so hysteresis-loop area is identically zero and unrunnable without a new state-continuation harness). (2) REFRAMING THAT ADDS NO EVIDENCE — A4 (measured MI ceiling sits redundantly atop an EXACT analytic bound and is dominated by estimator bias / dose-collinearity), A6/A7 (Meyer-Overton-residual reframes restate an already-run §8.2/control_2 analysis, and A7 risks provenance circularity — the non-immobilizer EC50s were hand-set from class labels, so 'residual ⊥ P' can confirm an input assumption rather than discover non-lipid information; the docking predecessor failed this exact inference), A8 (Buckingham-Pi counts dimensional, not fitted, DOF — blind to calibration history, so it cannot adjudicate the honesty question and routes its only real value through the knapsack/provenance guard that lives in S2). (3) PHYSICS/MATH WRONG OR INERT — A10 (Harada-Sasa needs a real thermal bath; the LIF noise has no kT, so T_eff is a free knob and the 'non-circular cross-validation' is tunable), A14/A16/A18/A19 (each relocates rather than removes a hidden knob, or derives a quantity — noise-localization variance, superposition identity, Boltzmann slope — that is a deterministic function of the SAME (total_pa, snare_gain) pair already established, hence carries no new information and often contradicts the frozen data, e.g. A16's √N ordering is the inverse of the observed worm-tightest/mouse-loosest pattern). A17 (gap-junction Laplacian) is real but inert: it raises the rank of the RESPONSE operator F downstream of the rank-2 INPUT bottleneck, so by the data-processing inequality it cannot raise class-identity discriminability, and it privileges the anchor-overfit worm over the clean fly.

---

## Open questions that gate the program

1. COMPLETE WRITE-PATH AUDIT (gates publishing the factorization theorem as absolute): grep every assignment to brain.neurons.I_ext and syn_exc.w/syn_inh.w across the whole codebase. I verified apply_anesthetic and apply_genotype verbatim, but the rank-2 theorem is only airtight if those are the COMPLETE perturbation write-path. Any hidden per-neuron or per-class state write would weaken the bound. This is cheap and must be done first.

2. RASTER PERSISTENCE: ground map 3 flags that the Phase G/J rasters the entropy-production avenue would need were never persisted, and the Harada-Sasa response-side estimator needs a timed-kick protocol that run_single (one monolithic brain.run, fresh brain per dose) does not implement. So any entropy-production or FDT work is NOT pure re-analysis — it requires re-running V1 with new monitors. Confirm whether any raster data survived before scoping the entropy front's empirical pieces.

3. PROVENANCE OF THE sat_pa LADDER AND THE 3 Hz CUTOFF: the knapsack analysis is conditional on the magnitude ladder, and the Π-audit can enumerate but not adjudicate whether the 3 Hz cutoff / sat_pa values were pinned a priori or tuned post-hoc to land halothane MAC. Were these frozen BEFORE the halothane anchor was fit? The honesty of A1/A2 and the 'one free parameter' claim depends on the answer; it is a code/prereg-history question, not a physics question.

4. MATCH#2 SNARE GAP: the existing Match#2 sampler matched only on agg_pa, excluding SNARE — so one of the two sufficient-statistic coordinates was left uncontrolled in the frozen control. How large is the resulting snare_gain spread across the Match#2 ensembles, and does correcting it change the fly percentile? This is a real, fixable defect in the existing control surfaced by the factorization view.

5. FLY SNARE ENGAGEMENT (gates the power of the entire S3 falsifier): the residual can only carry class-identity signal through the SNARE channel; if the fly conserved profile engages SNARE weakly at the relevant doses, Outcome A is mechanically forced regardless of biology. Confirm fly's SNARE engagement before betting the project's last positive on the shuffle test.

6. WORM I_gap CAVEAT: the worm mean-field reduction is complicated by gap-junction diffusive coupling (worm-only I_gap); the homogeneous reduction is clean for fly/mouse but the worm residual interpretation must carry this caveat — quantify how much I_gap perturbs the single-population reduction.

7. MINIMAL-DELTA vs FULL TIER4: the program shows the cell-type-targeting claims (Match#3, B1) need only a length-N per-neuron expression vector on the existing connectomes — strictly smaller than full Tier4 (no HH channels, ion pools, or metabolic state). Is that minimal delta worth building as an intermediate (V2) BEFORE committing to full Tier4, given it lifts the single cleanest by-construction impossibility while the metabolic/lipid/thermodynamic family genuinely needs the full V8?
