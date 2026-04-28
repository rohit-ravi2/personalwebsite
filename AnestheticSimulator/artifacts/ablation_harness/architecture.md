# Ablation harness — architecture v1

**Date:** 2026-04-28 (Wave P / Session 2 / WB3 / CP1)
**Status:** PAUSE FOR REVIEW. No code written; no implementation begun.
**Scope:** scaffolding infrastructure. Real ablation experiments deploy post-WB3 + Phase G LIFBrain integration.

---

## Pre-flight findings (all clean)

1. **Methodology literature.** Computational ablation as mechanism-isolation is established methodology — Markram et al. Blue Brain ablation studies (in silico knockout of cell types or channels, observe network response), theoretical neuroscience necessary-condition tests (Tononi 2008 on mechanism isolation; Sporns 2010 on network ablation), and pharmacological systems biology in silico drug target deletion (multi-target kinase inhibitor analyses). The high-level design — *zero out target perturbation, observe phenotype delta, rank causal necessity* — matches the standard protocol; the implementation specifics (per-target vs per-class granularity, statistical correction, pre-WB3 vs post-WB3 dual-substrate) are what need to be decided here.

2. **Overlay schema supports per-target ablation cleanly.** `wave2_overlay_v2.json` has 6 anesthetics × 30 targets = 180 entries. Each entry: `mechanism_class` + `occupancy_1xEC50` + `parameters` dict (e.g., `n_Ca_delta`, `tau_decay_factor`, `rate_factor`) + `occupancy_1xEC50_v1` for trace. Per-target ablation = setting that target's `occupancy_1xEC50` and parameter value to 0 (or removing from the perturbation profile). Atomic, non-side-effecting if done in-memory.

3. **Phase G perturbation manager is consumable without modification.** `AnestheticPerturbation.compute_perturbation_vector(anesthetic, dose_multiplier)` returns a `PerturbationProfile` dataclass whose `per_class` dict groups entries by mechanism class. The harness will modify the profile post-compute (zero out entries for ablated targets) before passing to `apply_to_brain()`. **No Phase G API changes required** — the harness is a wrapper that uses Phase G as-is.

4. **Behavioral readouts available pre-WB3 vs post-WB3.**
   - **Pre-WB3 (current):** Phase G's 50-neuron LIF demo (Brian2 NeuronGroup + Synapses) produces a single scalar `firing_rate_Hz` per run. Adequate for harness mechanics verification; insufficient for behavioral state distribution claims.
   - **Post-WB3 (pending Session 1 WB3 + Phase G LIFBrain integration):** 300-neuron LIFBrain with Cook 2019 connectome → per-neuron firing rates, command interneuron rates, optionally FSM behavioral states via ActivityFSM. The substrate change does not require harness API changes if readouts are abstracted behind a `PhenotypeReadout` callable.

5. **Granularity decision: per-(anesthetic, target) at default; per-mechanism-class as derived analysis.** Per the prompt's explicit guidance, per-(anesthetic, target) is the load-bearing primary deliverable. Per-mechanism-class attribution is computed post-hoc by grouping per-target results by `mechanism_class`. Per-(target, anesthetic) supports the load-bearing question "which targets matter for which anesthetic"; per-class supports "which mechanism classes matter" as a separate output of the same data.

6. **Compute budget.**
   - **Pre-WB3 demo network:** ~2 sec wall-clock per run × 5 seeds × 30 targets × 6 anesthetics + baselines = ~30 min for the full pre-WB3 smoke suite. Trivial.
   - **Post-WB3 LIFBrain (estimate):** 30-120 sec per Brian2 cython run depending on simulation duration. Default suite at n=5 seeds × 30 targets × 6 anesthetics + baselines ≈ 4530 runs. At ~60 sec/run ≈ **75 hours total**, **~12.5 hours per anesthetic sub-suite**. Within overnight batch budget; **state persistence + resume are load-bearing**, not optional.
   - Mitigation: per-anesthetic sub-suite is the deployable unit; n=3 seeds for initial pass + n=5 confirmation pass; resumable batch runner.

**One architectural decision flagged for review (Section 3):** profile-level ablation vs overlay-file-level ablation. Recommendation in Section 3.

---

## Section 1 — Conceptual design

### Mechanism-isolation question

For each of the 6 anesthetics, **which target engagements are causally necessary for the behavioral phenotype, and which are bystanders?**

The Wave P binding pipeline produces a multi-target engagement profile per anesthetic — halothane engages 8 mechanism classes across 30 targets at ≈ 0.99 occupancy. The binding profile alone says nothing about which engagements are *causally necessary* for behavioral immobilization vs which are *binding-positive bystanders* whose removal does not disrupt the phenotype. Resolving this is the load-bearing scientific contribution: it converts "halothane binds many targets" into "halothane requires X, Y, Z; removing W, V, U leaves the phenotype intact."

### Experimental protocol

```
For each anesthetic A:
  baseline = run(brain, full_perturbation(A, dose=1×), n_seeds=5)
  per_target_ablation = {}
  for target T in A's perturbation profile:
      ablated = run(brain, perturbation_minus(A, T, dose=1×), n_seeds=5)
      per_target_ablation[T] = effect_size(baseline, ablated)
  rank targets by effect_size descending → causally-necessary list
```

The harness produces, per anesthetic, a **ranked list of causal necessity** with per-target effect size + confidence interval + multiple-comparison-corrected p-value.

### Statistical analysis

- **n=5 seeds minimum per condition** (n=3 acceptable for pre-WB3 smoke tests; n=5 for production runs).
- **Paired comparison** baseline vs ablation per seed (same RNG seed for baseline and ablation runs to control for stochastic variability).
- **Effect size:** Cohen's d for behavioral state fractions; log-fold-change for firing rates; absolute difference for command interneuron rates.
- **Multiple-comparison correction:** Benjamini-Hochberg FDR at α=0.05 across 30 targets per anesthetic. (Bonferroni is too conservative for 30 hypotheses with anticipated correlation; FDR is the right control for ranked-list interpretation.)
- **Necessity criterion:** target is "causally necessary" if `|d| > 0.8` (large effect) AND BH-corrected p < 0.05.
- **Bystander criterion:** target is "bystander" if `|d| < 0.2` (small effect) AND BH-corrected p > 0.20 (i.e., evidence the ablation did *not* change phenotype meaningfully).
- Targets between necessary and bystander are "ambiguous" — not enough power at n=5 seeds to call definitively.

---

## Section 2 — Input/Output contract

### Inputs

| Input | Type | Notes |
|---|---|---|
| `overlay_path` | path to `wave2_overlay_v2.json` | source of per-(anesthetic, target) kinetic shifts |
| `anesthetic` | str | one of 6 anesthetics in overlay |
| `dose_multiplier` | float (default 1.0) | × clinical EC50 |
| `brain_substrate` | callable | constructs a Brian2 brain (Phase G demo or LIFBrain wrapper) |
| `scenario` | dict | {duration_ms, scenario_name, sensory_input_spec} |
| `ablation_targets` | list[str] or None | which target(s) to zero out; None = baseline (no ablation) |
| `seed` | int | for paired-comparison reproducibility |

### Outputs

Per run, a `RunResult` dict:

```json
{
  "anesthetic": "halothane",
  "dose_multiplier": 1.0,
  "ablation_targets": ["UNC-49"],
  "seed": 42,
  "phenotype": {
    "firing_rate_Hz": 24.0,
    "n_spikes": 2400,
    "fsm_state_fractions": null,
    "command_interneuron_rates": null
  },
  "perturbation_summary": {
    "n_classes_engaged_post_ablation": 7,
    "max_class_occupancy_post_ablation": 0.997,
    "ablated_class": "gaba_potentiation"
  },
  "wall_clock_sec": 2.1,
  "_meta": {
    "substrate": "phase_g_demo_50neuron",
    "git_sha": "<hash>",
    "timestamp": "2026-04-28T12:34:56Z"
  }
}
```

`fsm_state_fractions` and `command_interneuron_rates` are `null` pre-WB3; populated post-WB3.

Per-anesthetic aggregation produces `AblationSuiteResult`:

```json
{
  "anesthetic": "halothane",
  "baseline": {
    "n_seeds": 5,
    "phenotype_means": {"firing_rate_Hz": 0.0},
    "phenotype_stderrs": {"firing_rate_Hz": 0.0}
  },
  "per_target": {
    "UNC-49": {
      "n_seeds": 5,
      "phenotype_means": {...},
      "effect_size_cohens_d": 1.85,
      "p_value_uncorrected": 0.0001,
      "p_value_bh_corrected": 0.003,
      "necessity_class": "necessary"
    },
    ...
  },
  "ranked_causally_necessary": ["UNC-49", "TWK-18", ...],
  "bystanders": ["NLF-1", ...]
}
```

---

## Section 3 — Ablation harness API

```python
class AblationHarness:
    """Mechanism-isolation harness for Wave P anesthetic perturbations.

    Wraps Phase G AnestheticPerturbation with profile-level ablation transforms.
    Substrate-agnostic via PhenotypeReadout callable; pre-WB3 demo and
    post-WB3 LIFBrain modes share the same harness API.
    """

    def __init__(
        self,
        overlay_path: Path,
        substrate: SubstrateProvider,
        readout: PhenotypeReadout,
        out_dir: Path,
    ):
        """Construct from overlay file + substrate provider + readout function.

        substrate: callable returning a fresh Brian2 brain instance per run
                   (signature: substrate() -> Brain). Pre-WB3 demo or LIFBrain.
        readout:   callable extracting metrics from a run
                   (signature: readout(brain, run_state) -> dict).
        out_dir:   directory for per-run JSON state persistence.
        """

    def run_baseline(
        self, anesthetic: str, dose: float = 1.0, seed: int = 42,
        scenario: dict | None = None,
    ) -> RunResult:
        """Full perturbation, no ablation."""

    def run_ablation(
        self, anesthetic: str, target: str | list[str], dose: float = 1.0,
        seed: int = 42, scenario: dict | None = None,
    ) -> RunResult:
        """Zero out specified target(s) before applying perturbation."""

    def run_full_ablation_suite(
        self, anesthetic: str, dose: float = 1.0, n_seeds: int = 5,
        scenario: dict | None = None, resume: bool = True,
    ) -> AblationSuiteResult:
        """Baseline + ablation per target × n_seeds. Resumable from disk."""

    def compute_target_necessity(
        self, suite_result: AblationSuiteResult,
        metric: str = "firing_rate_Hz",
    ) -> dict:
        """Effect size + BH-corrected p per target, ranked descending."""

    def compute_class_attribution(
        self, suite_result: AblationSuiteResult,
    ) -> dict:
        """Group per-target results by mechanism_class; per-class aggregate effect."""
```

### Profile-level ablation (architectural decision)

Two implementation options surfaced in pre-flight:

**Option A — file-level ablation:** copy `wave2_overlay_v2.json`, set the ablated target's `occupancy_1xEC50 = 0` and `parameters.*.value = 0`, write to a temp file, point `AnestheticPerturbation` at the temp file. Pros: no Phase G changes, profile is exactly what real perturbation would be. Cons: filesystem churn, cleanup required, harder to test.

**Option B (recommended) — profile-level ablation:** call Phase G's `compute_perturbation_vector(anesthetic, dose)` to get a `PerturbationProfile`, then mutate the profile in-memory (remove ablated target's entries from the `per_class` dict), then call `apply_to_brain()` with the modified profile. Requires a small Phase G addition: **`apply_to_brain` accepts an optional pre-computed profile parameter** rather than recomputing internally.

**Option B is the load-bearing recommendation.** It's atomic, side-effect-free, testable, and the Phase G change is a 5-line API extension that's backwards-compatible (existing callers pass no profile, current behavior preserved).

**Phase G API change required (Option B):**

```python
# phase_g_network_perturbation.py:
def apply_to_brain(self, brain, anesthetic: str, dose_multiplier: float = 1.0,
                   profile: PerturbationProfile | None = None):
    if profile is None:
        profile = self.compute_perturbation_vector(anesthetic, dose_multiplier)
    # ... existing logic uses profile.per_class instead of recomputing ...
```

If the user prefers Option A (no Phase G change), the harness can ship with file-level ablation; the implementation will just be slightly more verbose. **Default to Option B unless Rohit prefers otherwise.**

---

## Section 4 — Phenotype readouts

### Pre-WB3 demo mode

`PhenotypeReadout` for the Phase G 50-neuron demo (the substrate already used in `phase_g_network_perturbation.py:dose_response_sweep`):

```python
def demo_readout(brain, run_state) -> dict:
    return {
        "firing_rate_Hz": run_state["n_spikes"] / brain.N / run_state["duration_s"],
        "n_spikes": run_state["n_spikes"],
        "fsm_state_fractions": None,        # not available pre-WB3
        "command_interneuron_rates": None,   # not available pre-WB3
    }
```

This is sufficient for harness mechanics verification (smoke tests in CP3) and for pre-WB3 smoke tests of Test 4 / mutant infrastructure.

### Post-WB3 production mode

`PhenotypeReadout` for the 300-neuron LIFBrain integrated with Phase G:

```python
def lifbrain_readout(brain, run_state) -> dict:
    spike_data = brain.spikes  # SpikeMonitor
    n_total = brain.N
    duration_s = run_state["duration_s"]
    # Aggregate firing rate
    firing_rate_Hz = len(spike_data.t) / n_total / duration_s
    # Command interneuron rates (AVA, AVB, AVD, RIM, RIB, AIB, etc.)
    command_neurons = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR",
                       "RIML", "RIMR", "RIBL", "RIBR", "AIBL", "AIBR"]
    command_rates = {}
    for cn in command_neurons:
        if cn in brain.idx:
            i = brain.idx[cn]
            spikes_i = (spike_data.i == i).sum()
            command_rates[cn] = spikes_i / duration_s
    # FSM state fractions (only if ActivityFSM connected)
    fsm_state_fractions = None
    if hasattr(brain, "fsm_state_history"):
        history = brain.fsm_state_history
        from collections import Counter
        c = Counter(history)
        total = sum(c.values()) or 1
        fsm_state_fractions = {state: count / total for state, count in c.items()}
    return {
        "firing_rate_Hz": firing_rate_Hz,
        "n_spikes": len(spike_data.t),
        "command_interneuron_rates": command_rates,
        "fsm_state_fractions": fsm_state_fractions,
    }
```

This is the post-WB3 hookup point. The harness ships with `lifbrain_readout` defined and a TODO marker for the `SubstrateProvider` that constructs the LIFBrain — that provider is implementable once Session 1's WB3 lands and Phase G LIFBrain integration is wired.

### Both modes

Both readouts return the same dict shape (with `null` for unavailable fields). The harness's statistical machinery operates on whichever metric is non-`null`; pre-WB3 uses `firing_rate_Hz`, post-WB3 uses `fsm_state_fractions` as the load-bearing behavioral metric with `firing_rate_Hz` and `command_interneuron_rates` as supporting metrics.

---

## Section 5 — Batch execution + state persistence

### Per-run state file

`out_dir/runs/{anesthetic}_{ablation_target_or_baseline}_{seed}.json` — a `RunResult` written immediately after each run.

Filename convention: `halothane_baseline_42.json`, `halothane_ablate_UNC-49_42.json`.

### Resume semantics

`run_full_ablation_suite(anesthetic, n_seeds=5, resume=True)` enumerates expected files, runs only missing combinations. Crash mid-suite → re-invocation picks up where the disk left off.

### Aggregation

`out_dir/summary/{anesthetic}_suite_summary.json` — `AblationSuiteResult` written after all runs in a suite complete. Aggregation is idempotent (re-runnable from disk without redoing the simulations).

### Concurrency

v1: serial execution per anesthetic suite. Parallel within an anesthetic across seeds is straightforward (independent runs) but deferred to v2 — Brian2 + cython doesn't multiprocess cleanly without process-level isolation.

---

## Section 6 — Statistical methodology

### Effect size

Per metric:

| Metric | Effect size measure | Threshold for "necessary" |
|---|---|---|
| `firing_rate_Hz` | log-fold-change from baseline | \|log10(ablation/baseline)\| > 0.30 (≈2× change) |
| `fsm_state_fractions["FWD"]` | Cohen's d on per-seed fractions | \|d\| > 0.8 (large effect) |
| `command_interneuron_rates["AVAL"]` | Cohen's d on per-seed rates | \|d\| > 0.8 |

The load-bearing metric post-WB3 will be `fsm_state_fractions["QUI"]` (quiescence fraction) since anesthetic immobilization is operationally a quiescence state. The harness reports all three; necessity ranking uses the FSM metric when available, falling back to firing rate pre-WB3.

### Statistical testing

- **Per-target test:** paired t-test on per-seed metric (baseline vs ablation), 5 paired pairs.
- **Multiple-comparison correction:** Benjamini-Hochberg FDR across 30 targets per anesthetic suite.
- **Decision rule:**
  - **Necessary** = |effect_size| > threshold AND BH-corrected p < 0.05
  - **Bystander** = |effect_size| < threshold/4 AND BH-corrected p > 0.20
  - **Ambiguous** = otherwise
- **Power note at n=5:** detecting d=0.8 at n=5 paired-pairs has ~50% power. n=10 reaches 90% power. The harness defaults to n=5 (compute-feasible) and surfaces "ambiguous" classifications honestly; promote to n=10 for production confirmation runs on targets near the boundary.

### Why BH-FDR over Bonferroni

Bonferroni at α=0.05/30 = 0.0017 is too conservative for ranked-list interpretation when targets are biologically correlated (e.g., the 5 nAChR subunits — UNC-29/38/63/LEV-1/ACR-16 — are not independent hypotheses). BH-FDR controls the false discovery rate at 0.05 across the ranked list, which is the right error control for the question "give me the targets most likely to be causally necessary."

---

## Section 7 — Cross-anesthetic comparison

Once per-anesthetic ablation suites are complete, three cross-anesthetic analyses ship as derived outputs:

### 7.1 Convergence — universally necessary targets

Targets that rank "necessary" across multiple anesthetics. Convergent targets are candidates for "common-pathway" anesthesia targets.

```
universal_necessary = intersect(
  necessary_set(halothane), necessary_set(isoflurane), ...
  necessary_set(propofol), necessary_set(etomidate), ...
)
```

Hypothesis: convergent targets cluster around K2P (halothane K2P literature), GABA-A (propofol/etomidate), and SNARE/Complex I (volatiles). Confirmation would be a publishable mechanism-isolation finding.

### 7.2 Divergence — anesthetic-specific targets

Targets necessary for one anesthetic but not others.

```
etomidate_specific = necessary_set(etomidate) - union(necessary_set(others))
ketamine_specific = necessary_set(ketamine) - union(necessary_set(others))
```

Etomidate is biologically GABA-A-selective; ketamine is NMDA-pathway-selective. The harness should reproduce these well-established selectivity patterns. If it doesn't, the calibration has a problem.

### 7.3 Mechanism class decomposition

Per anesthetic, fraction of behavioral effect attributable to each mechanism class:

```
class_attribution[A][C] = sum(effect_size(A, T) for T in C) / sum(effect_size(A, *))
```

Halothane should distribute across multiple classes (K2P + GABA-A + GluCl + Complex I + SNARE + nAChR). Etomidate should concentrate in `gaba_potentiation`. This is the mechanism-class summary that converts "halothane is multi-target" into a quantitative attribution.

---

## Section 8 — Mutant phenotype validation hooks

For each mutant in the published *C. elegans* anesthesia genetics literature, the harness simulates the mutation by modifying a parameter and runs anesthetic dose-response under the mutant background.

| Mutant | Mutation parameter | Expected phenotype | Anchor PMID |
|---|---|---|---|
| `gas-1(fc21)` | reduce GAS-1 Complex I rate factor (e.g., × 0.4 baseline) | hypersensitive: lower halothane EC50 | Morgan & Sedensky 1994 PMID 7943840 |
| `twk-18(cn110gf)` | enhance TWK-18 K2P leak conductance (×2) | hypersensitive (CP6 corrected direction) | Singaram 2011 PMID 22137475 |
| `sup-9(n180lf)` | reduce K2P leak conductance (×0.3) | modestly resistant | Singaram 2011 |
| `unc-79(e1068), unc-80(e1069)` | block NCA-1 Na leak channel | resistant | Sedensky & Meneely 1987 PMID 3576211 |
| `unc-13(s69)` | reduce SNARE release probability globally (×0.2) | hypersensitive (already-low margin) | Nguyen 1995 PMID 7647836 |

### Mutant infrastructure API

```python
class MutantBackground:
    """Encapsulates parameter modifications for a C. elegans mutant."""
    def __init__(self, mutant_name: str, modifications: dict[str, float]):
        # modifications: {"GAS-1.rate_factor": 0.4, ...}

    def apply_to_overlay(self, overlay: dict) -> dict:
        # Return a modified overlay with the mutation applied as a global
        # background change. Modifications stack with anesthetic perturbation.
```

Then in the harness:

```python
harness.run_baseline(anesthetic="halothane", dose=1.0, mutant=MutantBackground("gas-1", ...))
```

### Expected validation outcomes

- **Under WT background**, anesthetic dose-response matches the WT baseline.
- **Under gas-1 background**, halothane EC50 drops by 2-3× (Morgan target).
- **Under twk-18(cn110gf) background**, halothane EC50 drops by ~2× (Singaram target after CP6 direction correction).
- **Under unc-79 background**, halothane requires ~2× higher dose for the same effect (resistance).

Reproducing these phenotypes from primary-source-grounded mutant parameters would be substantive external validation of the harness's mechanism-isolation claims.

### Pre-WB3 vs post-WB3 mutant deployment

- **Pre-WB3:** infrastructure for `gas-1`, `twk-18(cn110)` implemented; smoke tests verify the mutation parameter modification produces a different baseline + different ablation profile. Real validation requires post-WB3 substrate.
- **Post-WB3:** all 5 mutants deployable on LIFBrain; full validation against published phenotypes.
- **`unc-79/80`, `unc-13` notes:** these involve targets that may not have full Phase G hooks pre-Phase-δ (NCA-1 / UNC-80 lack AlphaFold structures per CP6; UNC-13 not in Tier-1 panel). Scaffolded with TODO markers; full implementation deferred.

---

## Section 9 — Test 4 (Eger non-immobilizer) integration

The Eger non-immobilizer puzzle is the load-bearing boundary diagnostic that determines whether anesthetic specificity emerges at the network level or remains unresolved.

### Inputs

- `negative_vina_results.csv` — already shipped. 8 ligands × 30 targets × 3 poses, including hexafluoroethane, cis-1,2-DCE, trans-1,2-DCE.
- A "negative-control overlay" that follows `wave2_overlay_v2.json` schema, derived by applying the same Phase D kinetic-shift translation to negative-control Vina outputs + applying the CP5 f_allo = 2.50× correction.

### Synthetic overlay generation

CP4 will produce `artifacts/kinetics/wave2_overlay_negative_v2.json` — same schema as the anesthetic overlay, populated for hexafluoroethane + cis-DCE + trans-DCE (extending to all 8 negative controls if useful). The overlay generator script reuses Phase D logic; no new methodology needed.

### Test 4 protocol

```
For halothane (anesthetic positive control):
  full ablation suite, identify causally-necessary target list
For hexafluoroethane (Eger non-immobilizer):
  full ablation suite under same protocol
For cis-DCE (anesthetic) vs trans-DCE (non-immobilizer):
  paired comparison

Test 4 outcome interpretation:
  If hexafluoroethane network response < halothane response despite similar
  binding profile (CP3, CP7 confirmed): Eger puzzle resolved at network
  level — the multi-target integration produces anesthetic specificity
  even when binding profiles are similar.

  If hexafluoroethane network response ≈ halothane response: puzzle
  is NOT at network level either. Pivot to documentation of where the
  discriminative information lives (pharmacokinetics, behavioral
  threshold, conformational selectivity).
```

### Pre-WB3 vs post-WB3 Test 4

Pre-WB3: synthetic overlay + harness consumption verified; demo-network response reported for transparency. **Real Test 4 finding requires post-WB3 LIFBrain integration** — demo network is too small for the network-level integration test to be load-bearing.

Test 4 is the single most informative downstream experiment in the Wave P trajectory. The harness ships with Test 4 as a first-class operation (`run_test_4()` method) so post-WB3 deployment is a one-line invocation rather than a custom analysis.

---

## Decision points for review

1. **Profile-level vs file-level ablation (Section 3).** Recommend Option B (profile-level + 5-line Phase G API extension). If preferred, fall back to Option A (file-level, no Phase G change).

2. **Necessity threshold values (Section 6).** `|d| > 0.8` for behavioral state fractions; `|log10 fold| > 0.30` for firing rates. These are Cohen's conventional thresholds + standard 2× fold-change cutoff. If different thresholds are preferred (e.g., stricter for publication), tighten now.

3. **n_seeds default (Section 6).** n=5 default for production; n=3 for smoke tests. Promote to n=10 for confirmation on boundary targets. If different default preferred, adjust.

4. **Mutant scope at v1 (Section 8).** Implement `gas-1` + `twk-18(cn110gf)` fully; scaffold `sup-9 lf`, `unc-79/80`, `unc-13` with TODO. If different prioritization preferred, restate.

5. **Test 4 scope at v1 (Section 9).** Generate synthetic overlay for hexafluoroethane + cis-DCE + trans-DCE only at v1; defer benzene/methanol/n-pentane/cyclohexane/dimethyl_ether to optional v2. If full 8-ligand scope preferred, expand.

---

## What happens after approval

CP2: implement `src/ablation_harness.py` per Sections 3, 5, 6 (~3-4 hours of focused work).
CP3: smoke test on Phase G demo (~1-2 hours).
CP4: synthetic negative-control overlay + Test 4 plumbing (~2 hours).
CP5: mutant infrastructure for gas-1 + twk-18 GoF (~2 hours).
CP6: scaffolding completion doc + 4 commits + push (~30 min).

Total post-approval: ~9-10 hours. Within bounded work-block scope.

The harness is **scaffolding** — it ships ready to consume Phase G LIFBrain integration when Session 1's WB3 lands. **No real mechanism-isolation findings are produced in this work block** — those land in the next work block when WB3 + Phase G LIFBrain are operational. This work block makes that next work block deployable as a one-invocation batch run rather than requiring scaffolding rework.

---

## Honest scope reminder

This is staging infrastructure. The load-bearing scientific deliverables — per-anesthetic causally-necessary target lists, mutant phenotype validations, Test 4 verdict on the Eger non-immobilizer puzzle — depend on Phase G LIFBrain integration (gated on Session 1's WB3 release-event rule). Pre-WB3 smoke tests verify harness mechanics; they do not produce mechanism findings.

The investment is justified by the asymmetry: building the harness now means the load-bearing experimental work block runs cleanly post-WB3 instead of requiring another scaffolding cycle.
