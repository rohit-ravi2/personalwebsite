# Phase H — Empirical validation

**Phase letter:** H
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phase G (network perturbation runs).
**Successor:** Phase I, J (stretch) and paper draft.
**Compute:** local CPU minutes; no GPU.

---

## 1. Goal

Compare Phase G's network-level predictions against **8 anchor predictions** from published wet-lab data. Compile a pass/fail table; assess Wave P's overall scientific outcome.

The phase delivers the program-level success / failure verdict.

---

## 2. The 8 anchor predictions

| # | Anchor | Prediction | Source | Pass tolerance |
|---|---|---|---|---|
| 1 | WT halothane EC50 | ~3% atm (~340 µM aqueous) | Crowder 1996 PMID 8855256 | within 2× |
| 2 | WT isoflurane EC50 | ~5% atm | Morgan 1995 (PMID lookup needed) | within 2× |
| 3 | gas-1(fc21) iso EC50 leftward | 2-3× lower than WT iso | Morgan & Sedensky 1995 PMID 7549290 | within 50% (so 1.5×-4.5× shift) |
| 4 | unc-79(e1068) halothane EC50 rightward | 2-3× higher than WT halothane | Sedensky 1992 PMID 1346264 | within 50% |
| 5 | unc-80(e1069) similar to unc-79 | EC50 similar shift to unc-79 | Sedensky 1992 PMID 1346264 | within 50% (qualitative match) |
| 6 | twk-18(cn110) halothane resistance | 2-3× higher than WT halothane (gain-of-function K2P) | Sedensky 2001 PMID 11756669 | within 50% |
| 7 | unc-13(s69) halothane hypersensitivity | 2-3× lower than WT halothane | van Swinderen 1999 (PMID lookup needed) | within 50% |
| 8 | propofol immobilization in µM | µM EC50 in aqueous, sub-mM | Boddington 2017 (PMID lookup needed) | order of magnitude (within 10× central) |

### Pass criterion for the program

Wave P passes the program-level test if **≥ 4 of 8 anchors match** within their tolerance. Below 4/8 is a partial-confirmation; below 2/8 is a falsification of the multi-target framing at the network level.

The threshold of 4/8 reflects honest scope: the program is a proof-of-concept of network-level digital pharmacology, not a production-grade drug-discovery platform. 4/8 anchors at 2× tolerance is a meaningful demonstration; 6+/8 would be a strong demonstration.

---

## 3. Failure-mode mapping

For each anchor that fails, Wave P maps to a specific upstream phase rebuild:

| Failure | Diagnosed upstream cause | Rebuild target |
|---|---|---|
| WT EC50 wrong (anchor 1, 2, or 8) | Binding affinity miscalibration | Phase B/C/D rebuild |
| gas-1 hypersensitivity wrong (anchor 3) | Metabolic layer wrong | Phase F rebuild |
| unc-79 / unc-80 wrong (anchors 4, 5) | NCA-complex perturbation wrong | Phase A/B for NCA-complex; or Phase D for shift |
| twk-18 wrong (anchor 6) | TWK-18 gain-of-function effect not captured | Phase D MD on TWK-18 |
| unc-13 wrong (anchor 7) | Markov synapse priming-rate shift wrong | Phase E rebuild for r_prime |
| Single-target lesion reproduces full effect (Gate G.1.5 fails) | Multi-target framing wrong at network level | Wave P pivot |

### 3.1 Failure threshold for program pivot

If 0-1 anchors pass: full pivot. The multi-target framing is unsupported. Document as negative result and reframe paper.

If 2-3 anchors pass: partial pivot. Identify which anchors fail; rebuild specific phases; re-run.

If 4-5 anchors pass: program success at proof-of-concept level. Paper draft proceeds.

If 6+ anchors pass: strong program success. Paper draft + Tier 2 / Phase I / J consideration.

---

## 4. Method

### 4.1 Per-anchor evaluation

For each anchor:

```python
# anchor 1: WT halothane EC50
sim_EC50 = aggregated_ec50.query("anesthetic == 'halothane' and genotype == 'WT' and lesion_class == 'full'")["fitted_EC50"].iloc[0]
pub_EC50 = 340  # µM aqueous
ratio = sim_EC50 / pub_EC50
pass_anchor_1 = 0.5 <= ratio <= 2.0
```

### 4.2 Aggregate verdict

```python
n_pass = sum(anchor_pass[1:9])
if n_pass >= 6:
    verdict = "STRONG_PASS"
elif n_pass >= 4:
    verdict = "PASS"
elif n_pass >= 2:
    verdict = "PARTIAL_FAIL"
else:
    verdict = "FAIL"
```

### 4.3 Lesion test verdict (re-evaluation of Gate G.1.5)

In addition to the 8 anchors, Phase H formally re-evaluates **Gate G.1.5** (the per-target lesion test) at the program level:

```python
full_effect = aggregated_ec50.query("anesthetic == 'halothane' and genotype == 'WT' and lesion_class == 'full'")["fraction_immobilized_mean"].iloc[0]
for lesion in ["GABA", "NCA", "K2P", "SNARE", "complexI", "GluCl", "nAChR"]:
    lesion_effect = aggregated_ec50.query(f"lesion_class == '{lesion}'")["fraction_immobilized_mean"].iloc[0]
    if lesion_effect / full_effect > 0.8:
        # Single-target reproduces full effect: multi-target framing FALSIFIED
        ...
```

Gate G.1.5 + 4/8 anchors = program success. Gate G.1.5 fail = program pivot regardless of anchor count.

---

## 5. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| Anchor evaluation script | local CPU | 1 | $0 |
| Failure-mode mapping document | manual | 2 | $0 |
| Visualization | local CPU | 1 | $0 |
| End-of-program report | manual | 4 | $0 |
| **Total Phase H** | | **~8 hours** | **$0** |

---

## 6. Preregistered success criteria (Gate H.1)

1. **H.1.1 — ≥ 4/8 anchors pass within their tolerances.**
2. **H.1.2 — Gate G.1.5 (multi-target lesion) holds at the program level.**
3. **H.1.3 — Failure-mode mapping is complete:** every failed anchor has a documented upstream-phase diagnosis.

---

## 7. Halting rules

Phase H is a write-up phase; halting rules are about how the verdict is reported:

- If H.1.1 fails (< 4 anchors pass): the program pivot is the next conversation, not Phases I/J.
- If H.1.2 fails (lesion test passes a single class): document, write the negative result, do not proceed to Phase I/J.

---

## 8. Output deliverables

| File | Contents |
|---|---|
| `artifacts/validation/anchor_table.csv` | 8-row anchor evaluation |
| `artifacts/validation/anchor_evaluation.md` | Per-anchor pass/fail with diagnosis |
| `artifacts/validation/lesion_test_program_level.md` | G.1.5 program-level re-evaluation |
| `artifacts/validation/program_verdict.md` | Wave P program-level verdict |
| `artifacts/validation/phase_h_completion.md` | end-of-block report |

---

## 9. Falsifiability checks

The phase's premise: **"Wave P's network-level predictions match published wet-lab data on at least 4 of 8 anchor predictions, AND per-target lesion analysis supports the multi-target framing."**

Falsified if:

- < 4 anchors pass.
- Per-target lesion reproduces > 80% of the full effect.

A program-level falsification is a publishable negative result. The Wave P paper would then be reframed as "predicted multi-target binding profile fails to reproduce *C. elegans* anesthetic phenotypes — implications for the multi-target hypothesis at network scale."

---

## 10. Integration points

**Inputs:** all Phase G outputs.

**Outputs:** the program-level verdict drives the paper draft (`papers/wave_p_paper_outline.md`).

---

## 11. Citation hygiene declaration

- Crowder 1996 — PMID 8855256. [VERIFIED]
- Morgan 1995 — (PMID lookup needed). [BLOCKING]
- Morgan & Sedensky 1995 — PMID 7549290. [VERIFIED]
- Sedensky 1992 — PMID 1346264. [VERIFIED]
- Sedensky 2001 — PMID 11756669. [VERIFIED]
- van Swinderen 1999 — (PMID lookup needed). [BLOCKING]
- Boddington 2017 — (PMID lookup needed). [BLOCKING]

**Pre-flight verification status:** 4 of 7 verified.

---

## 12. Phase H execution plan

1. Pre-flight citation verification (3 PMIDs).
2. Run anchor evaluation script on `aggregated_ec50.csv`.
3. For each failed anchor, write failure-mode diagnosis.
4. Re-evaluate G.1.5 at program level.
5. Produce `program_verdict.md` with overall pass/partial-fail/fail verdict.
6. End-of-program report.
