# Phase C — Occupancy matrix at clinical concentrations

**Phase letter:** C
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phase B (binding pose prediction). Phase C requires Gate B.1 to pass before entering.
**Successor:** Phase D (kinetic shift translation) consumes Phase C's occupancy matrix.
**Compute:** local CPU, minutes (the load-bearing science is not compute-heavy; it is calibration-heavy).
**Significance:** **Gate C.1 is the first program-level falsifiability checkpoint of Wave P. It is the load-bearing decision point in the multi-target framing.**

---

## 1. Goal

Convert per-(target, anesthetic) Kd values from Phase B into **fractional occupancy at clinical concentrations** (0.5×, 1×, 2×, 5× clinical EC50), accounting for membrane partitioning and Hill-equation kinetics. Produce the canonical occupancy matrix that Phase D translates into per-target kinetic shifts.

The phase resolves the load-bearing question of Wave P: **at clinical anesthetic concentrations, do multiple targets show non-trivial occupancy?**

If yes → multi-target framing supported, Phase D begins.
If no (only 0 or 1 targets exceed 10% occupancy at 1× EC50) → multi-target framing is **falsified**. Wave P pauses, the program pivots to a single-target validation framework, and the negative result is documented as a publishable finding.

---

## 2. Background

### 2.1 The Hill equation and fractional occupancy

For a single binding site with no cooperativity:

```
occupancy(C) = C^n / (Kd^n + C^n)        (Hill equation)
```

with n = 1 in the absence of cooperativity. C is the concentration **at the binding site** (not the bulk concentration). For ion channels and GPCRs in the membrane, the binding-site concentration depends on partition coefficient (see §2.2).

For multiple binding sites with positive cooperativity (e.g., Cys-loop pentamers with multiple symmetry-equivalent sites), n > 1. Wave P defaults to **n = 1 unless target-specific literature provides a measured cooperativity coefficient**.

### 2.2 Membrane partitioning is mandatory

Volatile anesthetics are highly lipophilic. Oil/water partition coefficients (K_p):

| Anesthetic | K_p,oil/water | log P_oct (proxy) |
|---|---|---|
| Halothane | ~250 | 2.3 |
| Isoflurane | ~90 | 2.0 |
| Sevoflurane | ~50 | 1.8 |
| Propofol | ~1300 | 3.8 |
| Ketamine | ~10 | 2.2 |
| Etomidate | ~50 | 2.5 |

The membrane-side concentration of an anesthetic at thermodynamic equilibrium with bulk aqueous concentration C_aq is:

```
C_membrane ≈ K_p × C_aq
```

For halothane at 1× clinical EC50 (~340 µM aqueous), the membrane-side concentration is ~85 mM — three orders of magnitude higher. **This is non-negotiable for membrane-embedded targets.** A bulk-aqueous calculation would underestimate occupancy by 250×.

For cytosolic targets (e.g., Complex I subunits matrix-side), bulk aqueous concentration applies. For membrane-embedded portions of those subunits (most of NDUFS2 / GAS-1's pocket lies in the membrane arm, near the Q-binding tunnel), partition still applies. **Per-target compartment assignment** is in `targets/tier1_targets.csv` column `pocket_compartment` (one of `membrane_embedded`, `aqueous_extracellular`, `aqueous_intracellular`, `membrane_interfacial`).

### 2.3 Clinical concentrations

| Anesthetic | Aqueous EC50 used as 1× | Source |
|---|---|---|
| Halothane | 340 µM | Crowder 1996 PNAS PMID 8855256 (3% atm via Henry's law) |
| Isoflurane | 290 µM | Morgan 1995 (5% atm, isoflurane has lower aqueous solubility per atm than halothane) |
| Sevoflurane | 230 µM | Crowder lab |
| Propofol | 1 µM | Boddington 2017 (PMID lookup needed) |
| Ketamine | 5 mM | (PMID lookup needed) |
| Etomidate | 0.3 µM | (PMID lookup needed) |

Wave P evaluates occupancy at 0.5×, 1×, 2×, and 5× these values. The 5× column is for the high-concentration sensitivity test — many physiological-receptor experiments use supraclinical concentrations.

### 2.4 Conversion: Vina/GNINA score → Kd

Vina returns a score in kcal/mol. The empirical relationship to Kd is approximate:

```
ΔG_bind ≈ Vina_score × 0.7   (empirical fudge factor for anesthetics)
Kd = exp(ΔG_bind / RT)        (RT = 0.593 kcal/mol at 298 K)
```

GNINA's CNN-rescored score is in pK_d-like units (higher = better binding). The conversion is documented per the GNINA paper.

In the canonical zero-spend plan, FEP is not run in Phase B (DEFERRED — see `preregistration/phase_b_binding_pose.md` §13). All 150 pairs use GNINA-derived Kd with an explicit factor-of-3 uncertainty band. If the user later authorizes the deferred FEP path, top-10 FEP-derived ΔG_bind would replace GNINA Kd for those 10 pairs with a tighter factor-of-2 band.

---

## 3. Method

### 3.1 Per-(target, anesthetic) calculation

For each pair in `artifacts/binding/binding_matrix.csv`:

```python
# Inputs
target_compartment = "membrane_embedded"  # from tier1_targets.csv
anesthetic_aq_EC50_uM = 340.0             # halothane example
anesthetic_Kp = 250.0                      # halothane oil/water
target_Kd_uM = 80.0                        # from GNINA (canonical) or FEP (if deferred path authorized)
hill_n = 1                                 # default

# Compute membrane-side concentrations at 0.5x, 1x, 2x, 5x EC50
multipliers = [0.5, 1.0, 2.0, 5.0]
for mult in multipliers:
    C_aq = mult * anesthetic_aq_EC50_uM
    if target_compartment in ["membrane_embedded", "membrane_interfacial"]:
        C_at_site = anesthetic_Kp * C_aq
    else:
        C_at_site = C_aq
    occupancy = C_at_site**hill_n / (target_Kd_uM**hill_n + C_at_site**hill_n)
    # store
```

### 3.2 Uncertainty propagation

Phase B's binding affinity estimates carry uncertainty:

- GNINA-derived Kd: factor of 3 (canonical; used for all 150 pairs in the zero-spend plan).
- FEP-derived Kd (DEFERRED, only if user authorizes the deferred path): factor of 2 (literature benchmark on FEP-vs-experiment for anesthetics); would replace GNINA Kd on top 10 hits.
- Photolabel-confirmed pairs: no extra discount.

Wave P propagates this as a per-pair Kd interval [Kd / factor, Kd × factor] and computes occupancy at the lower-Kd (higher-occupancy) and upper-Kd (lower-occupancy) bounds. The Phase C output reports occupancy as a 3-tuple (low, central, high) per pair.

### 3.3 Output matrix structure

```
occupancy_matrix.npz contains:
- targets:          shape (25,)         array of target gene names
- anesthetics:      shape (6,)          array of anesthetic names
- multipliers:      shape (4,)          [0.5, 1.0, 2.0, 5.0]
- Kd_central:       shape (25, 6)       per-pair central Kd in µM
- Kd_low:           shape (25, 6)       lower bound (factor 2 or 3)
- Kd_high:          shape (25, 6)       upper bound
- occupancy_central: shape (25, 6, 4)   per-pair per-multiplier central occupancy
- occupancy_low:    shape (25, 6, 4)    lower bound
- occupancy_high:   shape (25, 6, 4)    upper bound
- compartment:      shape (25,)         per-target compartment label
- Kp:               shape (6,)          per-anesthetic partition coefficient
- hill_n:           shape (25, 6)       per-pair Hill coefficient (default 1)
- meta:             dict                Phase B source matrix path, Phase B version, date
```

### 3.4 Visualization deliverables

For Gate C.1 inspection:

- `artifacts/occupancy/occupancy_heatmap_halothane_1x.png` — 25 targets ranked by occupancy.
- `artifacts/occupancy/occupancy_heatmap_isoflurane_1x.png` — same for isoflurane.
- `artifacts/occupancy/occupancy_dose_response.png` — per-target dose-response curves.
- `artifacts/occupancy/occupancy_uncertainty.png` — per-pair central + bracket plot.
- `artifacts/occupancy/multitarget_count.csv` — per-(anesthetic, multiplier) count of targets exceeding 10% occupancy.

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| Hill equation across 25 × 6 × 4 = 600 cells | local CPU | < 0.1 | $0 |
| Uncertainty bracket propagation | local CPU | < 0.1 | $0 |
| Visualization | local CPU | 0.5 | $0 |
| Write Gate C.1 evaluation | manual review | 1-2 | $0 |
| **Total Phase C** | | **~2 hours** | **$0** |

Phase C is computationally trivial. Its load-bearing role is the **interpretive judgment at Gate C.1** — does the resulting matrix support or falsify the multi-target framing?

---

## 5. Preregistered success criteria (Gate C.1 — load-bearing)

**Gate C.1 is the first program-level falsifiability checkpoint of Wave P.**

Phase C passes Gate C.1 if:

**C.1.1 — Multi-target presence:** At halothane 1× EC50 (membrane-adjusted), **≥ 5 of 25 Tier-1 targets show central occupancy > 10%**. (The threshold and number are pre-committed; modifications require an amendment block at the bottom of this document with date and rationale.)

**C.1.2 — Robustness across anesthetics:** The C.1.1 criterion holds for at least 2 of 3 volatile anesthetics (halothane, isoflurane, sevoflurane). If only 1 of 3 passes, the multi-target framing may be halothane-specific.

**C.1.3 — Robustness across uncertainty:** Even at the lower-occupancy bound (Kd_high), at least 3 of 25 targets exceed 10% occupancy. This guards against C.1.1 passing only at the optimistic edge of the Kd uncertainty band.

**C.1.4 — Lesion preview consistency:** The targets exceeding 10% are distributed across at least 3 of the 5 mechanism classes (Cys-loop, K2P, NCA, SNARE, Complex I). Concentration in a single class would already weaken the multi-target framing.

If C.1.1 through C.1.4 all pass: **Wave P proceeds to Phase D.**

If C.1.1 fails: **multi-target framing is falsified.** Wave P pauses. The result is documented as a publishable negative finding ("predicted occupancy at halothane EC50 shows only N targets > 10%"), and the program pivots to a single-target validation framework focused on the highest-occupancy target.

If C.1.1 passes but C.1.2-C.1.4 fail: surface to user. The multi-target framing is ambiguous; the user decides whether to proceed with caveats or pivot.

---

## 6. Halting rules

**Pause and surface (mid-flight):**

- During Phase C execution, if the membrane-partition adjustment turns out to push every target above 50% occupancy at 1× EC50 → "everything binds everything," the framing is uninformative; suspect a methodological error in the partition adjustment. Halt.
- If a target's central Kd is < 100 nM (sub-micromolar) — far tighter than known anesthetic-class Kd → suspect Phase B over-fitting to a high-affinity local mode; revisit GNINA poses on that pair; consider escalating to deferred FEP path per `phase_b_binding_pose.md` §13 only on user authorization.
- If the Kd uncertainty bracket spans more than one order of magnitude on > 50% of pairs → Phase B's calibration was too noisy; pause and discuss escalation options (deferred FEP path, photolabel re-evaluation, or homolog-only docking).

**Document and continue:**

- A single target's compartment assignment is genuinely ambiguous (e.g., interfacial vs. bilayer-spanning) → flag as low-confidence in `compartment_uncertainty.md`, run both compartment scenarios, report both occupancy values.
- Hill coefficient n > 1 surfaces in literature for a specific target after kickoff → amend the document with date and source, re-run Phase C for that target.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `artifacts/occupancy/occupancy_matrix.npz` | 25 × 6 × 4 occupancy matrix with central + bracket |
| `artifacts/occupancy/occupancy_table.csv` | Same data in human-readable format |
| `artifacts/occupancy/multitarget_count.csv` | Count of targets > 10% per (anesthetic, multiplier) |
| `artifacts/occupancy/gate_c1_evaluation.md` | **Load-bearing**: C.1.1-C.1.4 evaluation with pass/fail per criterion |
| `artifacts/occupancy/occupancy_heatmap_*.png` | Heatmaps per anesthetic at 1× |
| `artifacts/occupancy/occupancy_dose_response.png` | Dose-response curves |
| `artifacts/occupancy/compartment_uncertainty.md` | Targets with ambiguous compartment |
| `artifacts/occupancy/phase_c_completion.md` | End-of-block report |
| `artifacts/logs/phase_c_<DATE>.log` | Execution log |

---

## 8. Falsifiability checks

The phase's premise is the **multi-target framing of Wave P itself**. Phase C is where the framing is empirically tested.

Falsified if:

- C.1.1 fails: < 5 targets > 10% at halothane 1× EC50.
- C.1.3 fails strongly: even with optimistic Kd, < 3 targets > 10%.
- All occupancy values cluster around 99% or 1% (no gradient) → docking-derived Kd is uniformly mis-scaled, calculation is uninformative.

The first two are program-level falsifications (Wave P pivots). The third is a methodological failure (Phase B miscalibration).

---

## 9. Integration points

**Inputs from earlier phases:**

- Phase B `artifacts/binding/binding_matrix.csv` — GNINA-derived Kd estimates (central + factor-of-3 bracket).
- Phase B `artifacts/binding/top10_gnina.csv` — top-10 GNINA hits with cross-method-agreement metrics. (FEP-derived Kd for top hits is DEFERRED; only present if user authorizes the deferred path.)
- `targets/tier1_targets.csv` — `pocket_compartment` per target.
- `anesthetics/anesthetic_panel.csv` — clinical EC50, Kp.

**Outputs consumed by:**

- **Phase D** (`src/phase_d_kinetic_shifts.py`) — reads `occupancy_matrix.npz` to compute per-target kinetic shifts.
- **Phase G** (network runs) — reads occupancy at chosen multipliers for dose-response runs.
- **Phase H** (validation) — uses the central + bracket as the input model for cross-anesthetic predictions.
- **Phase I** (inverse design, stretch) — Phase C is the structural-prior occupancy that Phase I's empirical-derived occupancy is compared against.

---

## 10. Citation hygiene declaration

- Crowder 1996, *PNAS*, halothane EC50 in *C. elegans* — PMID 8855256. [VERIFIED]
- Morgan 1995, isoflurane — (PMID lookup needed; should be 7748540 or near).
- Morgan & Sedensky 1995, gas-1 isoflurane — PMID 7549290. [VERIFIED]
- Sedensky 1992, unc-79 / unc-80 halothane — PMID 1346264. [VERIFIED]
- Boddington 2017, propofol — (PMID lookup needed).
- Halothane oil/water Kp = 250 — established in lipid-bilayer literature; verify primary source (Franks & Lieb 1994 Nature; Eckenhoff & Johansson 1997 *Pharmacological Reviews*).

**Pre-flight verification status:** 3 of 6 verified. 3 PMIDs are blocking items for Phase C entry — Morgan 1995, Boddington 2017, halothane Kp primary source.

---

## 11. Risk register (Phase C)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Membrane partition coefficient varies by 2× between sources | Medium | Use the median of 3 published K_p values per anesthetic; document range |
| Cytosolic vs membrane-embedded compartment mis-assignment | Medium | Per-target review against published binding-site literature; flag ambiguous cases; run both scenarios |
| Hill coefficient n = 1 default underestimates pentameric cooperativity | Medium | Sensitivity analysis: re-run with n = 2 for pentameric receptors; document the difference |
| GNINA-to-Kd conversion has uncalibrated systematic bias | High (known) | Document factor-of-3 uncertainty band per pair; deferred FEP path on top 10 (`phase_b_binding_pose.md` §13) available for anchor disambiguation if user authorizes |
| Gate C.1 fails (multi-target framing falsified) | Low-Medium | Plan B is documented: pivot to single-target framework; result is publishable as negative finding |

---

## 12. Phase C execution plan

1. Pre-flight citation verification (3 blocking items).
2. Verify per-target `pocket_compartment` assignment in `tier1_targets.csv` against published binding-site literature.
3. Load `binding_matrix.csv` from Phase B.
4. Run occupancy calculation across 25 × 6 × 4 = 600 cells; propagate uncertainty.
5. Run sensitivity analysis: alternate K_p values (× 2 / 0.5), alternate Hill n (1 vs 2 for pentamers).
6. Generate visualizations.
7. **Compile Gate C.1 evaluation as a standalone document.** Mark each of C.1.1-C.1.4 as PASS/FAIL with quantitative justification. This is the load-bearing artifact of Phase C.
8. End-of-block report with explicit Phase D readiness assessment OR program-level pivot recommendation.

---

## 13. Special note on Phase C's interpretive role

Phase C is **not** the most computationally expensive phase, but it is **the most consequential**. The occupancy matrix is a 25 × 6 × 4 grid of numbers, but the interpretation at Gate C.1 determines whether Wave P proceeds at all.

Mid-flight surfacing applies aggressively here: any unexpected pattern in the occupancy distribution (e.g., halothane and isoflurane disagree on which targets are highest-occupancy; one target dominates at 80% while all others are < 5%) is surfaced before the formal Gate C.1 evaluation. The user reviews the matrix before the gate is formally evaluated.

The user's standing instruction is: **falsifiability before elaboration**. Gate C.1 is the operational expression of that standing instruction at the program level.
