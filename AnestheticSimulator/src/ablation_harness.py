"""Ablation harness — mechanism-isolation infrastructure for Wave P.

Per-(anesthetic, target) computational ablation: zero out one target's
perturbation contribution, observe phenotype delta vs full-perturbation
baseline, rank causally-necessary targets per anesthetic.

Architecture: artifacts/ablation_harness/architecture.md (CP1).

This is scaffolding infrastructure. Real mechanism-isolation experiments
deploy post-WB3 + Phase G LIFBrain integration. Pre-WB3 smoke tests verify
harness mechanics on the Phase G 50-neuron LIF demo network.

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/ablation_harness.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_g_network_perturbation import (
    AnestheticPerturbation,
    PerturbationProfile,
    OVERLAY_V2,
)

OUT_DIR = ROOT / "artifacts" / "ablation_harness"
RUNS_DIR = OUT_DIR / "runs"
SUMMARY_DIR = OUT_DIR / "summary"


# ===== Phenotype readouts ============================================


def demo_readout(brain, run_state: dict) -> dict:
    """Pre-WB3 readout for the Phase G 50-neuron LIF demo network.

    run_state contains: n_spikes, duration_s, n_neurons.
    """
    duration_s = run_state.get("duration_s", 1.0)
    n_neurons = run_state.get("n_neurons", brain.N if hasattr(brain, "N") else 50)
    n_spikes = run_state.get("n_spikes", 0)
    return {
        "firing_rate_Hz": n_spikes / max(1, n_neurons) / max(0.001, duration_s),
        "n_spikes": n_spikes,
        "fsm_state_fractions": None,
        "command_interneuron_rates": None,
    }


def lifbrain_readout(brain, run_state: dict) -> dict:
    """Post-WB3 readout for the 300-neuron LIFBrain.

    TODO: wire to LIFBrain post-WB3. Currently returns the demo shape with
    placeholder None for unavailable fields. The real implementation
    extracts per-neuron firing rates from brain.spikes (SpikeMonitor),
    computes command interneuron rates, and reads FSM state history if
    ActivityFSM is connected.
    """
    duration_s = run_state.get("duration_s", 1.0)
    spike_data = getattr(brain, "spikes", None)
    if spike_data is None or not hasattr(brain, "N"):
        return demo_readout(brain, run_state)
    n_total = brain.N
    n_spikes = len(spike_data.t) if hasattr(spike_data, "t") else 0
    firing_rate_Hz = n_spikes / max(1, n_total) / max(0.001, duration_s)

    command_neurons = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR",
                        "RIML", "RIMR", "RIBL", "RIBR", "AIBL", "AIBR"]
    command_rates: dict[str, float] = {}
    if hasattr(brain, "idx") and hasattr(spike_data, "i"):
        import numpy as np
        spike_i_arr = np.asarray(spike_data.i)
        for cn in command_neurons:
            if cn in brain.idx:
                idx = brain.idx[cn]
                count = int((spike_i_arr == idx).sum())
                command_rates[cn] = count / max(0.001, duration_s)

    fsm_state_fractions: Optional[dict[str, float]] = None
    if hasattr(brain, "fsm_state_history"):
        from collections import Counter
        history = list(getattr(brain, "fsm_state_history"))
        if history:
            c = Counter(history)
            total = sum(c.values()) or 1
            fsm_state_fractions = {str(state): count / total for state, count in c.items()}

    return {
        "firing_rate_Hz": firing_rate_Hz,
        "n_spikes": n_spikes,
        "fsm_state_fractions": fsm_state_fractions,
        "command_interneuron_rates": command_rates if command_rates else None,
    }


# ===== Substrate providers ===========================================


def make_phase_g_demo_substrate(seed: int):
    """Construct a fresh 50-neuron Brian2 LIF demo brain.

    Mirrors the demo network in phase_g_network_perturbation.dose_response_sweep.
    Returned object has: .N, .neurons (NeuronGroup), .net (Network), .W_syn,
    minimal _W_chem_runtime (zeros, so synaptic-weight ablation is a no-op
    on this substrate but harness mechanics still verify).
    """
    import numpy as np
    from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Network, Quantity,
                        ms, mV, nS, pF, Hz, pA, defaultclock, prefs,
                        seed as brian2_seed)
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    brian2_seed(seed)

    N_E, N_I = 40, 10
    eqs = """
    dv/dt = (v_rest - v)/tau + (I_ext)/C_mem : volt (unless refractory)
    I_ext : amp
    """
    params = {"v_rest": -65 * mV, "tau": 20 * ms, "C_mem": 200 * pF}
    G = NeuronGroup(N_E + N_I, eqs, threshold="v > -50*mV", reset="v = -70*mV",
                    refractory=2 * ms, namespace=params, method="exact")
    G.v = -65 * mV
    G.I_ext = 350 * pA

    S_ee = Synapses(G[:N_E], G[:N_E], on_pre="v_post += 0.3*mV")
    S_ee.connect(p=0.1)
    S_ei = Synapses(G[:N_E], G[N_E:], on_pre="v_post += 0.5*mV")
    S_ei.connect(p=0.2)
    S_ie = Synapses(G[N_E:], G[:N_E], on_pre="v_post -= 0.5*mV")
    S_ie.connect(p=0.3)

    spikes = SpikeMonitor(G)
    net = Network(G, S_ee, S_ei, S_ie, spikes)

    class _DemoBrain:
        pass

    brain = _DemoBrain()
    brain.N = N_E + N_I
    brain.names = [f"E{i}" for i in range(N_E)] + [f"I{i}" for i in range(N_I)]
    brain.neurons = G
    brain.spikes = spikes
    brain.net = net
    brain.W_syn = 0.5 * mV  # demo doesn't really use this; provide for API parity
    brain._W_chem_runtime = np.zeros((brain.N, brain.N), dtype=np.float32)
    brain.nt_primary = ["ACh"] * N_E + ["GABA"] * N_I
    brain.idx = {n: i for i, n in enumerate(brain.names)}
    return brain


def make_lifbrain_substrate_TODO(seed: int):
    """Post-WB3 substrate provider — constructs the 300-neuron LIFBrain.

    TODO: wire to scripts/brain/lif_brain.py:LIFBrain once Session 1's WB3
    (cross-coupling release rule) lands and Phase G LIFBrain integration
    is operational. The signature returns a brain compatible with
    AnestheticPerturbation.apply_to_brain — needs .names, .neurons.I_ext,
    ._W_chem_runtime, .W_syn, .nt_primary, .idx.
    """
    raise NotImplementedError(
        "LIFBrain substrate provider is post-WB3. Pending Session 1's "
        "release-event rule + Phase G LIFBrain wiring. See architecture.md "
        "Section 4 for the readout signature; implement once WB3 lands."
    )


# ===== Ablation profile transforms ==================================


def ablate_profile(
    profile: PerturbationProfile, ablation_targets: list[str]
) -> PerturbationProfile:
    """Return a new PerturbationProfile with the specified target entries zeroed.

    The original profile is not mutated. Entries for `ablation_targets` are
    removed entirely from the per_class dict (equivalent to occupancy=0 since
    apply_to_brain skips zero-magnitude entries).
    """
    if not ablation_targets:
        return profile
    targets_set = set(ablation_targets)
    new_per_class: dict[str, list[dict]] = {}
    for cls, entries in profile.per_class.items():
        kept = [e for e in entries if e["target"] not in targets_set]
        if kept:
            new_per_class[cls] = kept
    new_profile = PerturbationProfile(
        anesthetic=profile.anesthetic, dose_multiplier=profile.dose_multiplier
    )
    new_profile.per_class = new_per_class
    return new_profile


# ===== Ablation harness =============================================


@dataclass
class RunResult:
    anesthetic: str
    dose_multiplier: float
    ablation_targets: list[str]
    seed: int
    phenotype: dict
    perturbation_summary: dict
    wall_clock_sec: float
    meta: dict = field(default_factory=dict)


class AblationHarness:
    """Mechanism-isolation harness consuming Phase G perturbation manager."""

    def __init__(
        self,
        overlay_path: Path = OVERLAY_V2,
        substrate: Callable[[int], Any] = make_phase_g_demo_substrate,
        readout: Callable[[Any, dict], dict] = demo_readout,
        out_dir: Path = OUT_DIR,
        substrate_label: str = "phase_g_demo_50neuron",
    ):
        self.pert = AnestheticPerturbation(overlay_path)
        self.substrate = substrate
        self.readout = readout
        self.out_dir = Path(out_dir)
        self.runs_dir = self.out_dir / "runs"
        self.summary_dir = self.out_dir / "summary"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.substrate_label = substrate_label

    def list_anesthetics(self) -> list[str]:
        return self.pert.list_anesthetics()

    def list_targets(self, anesthetic: str) -> list[str]:
        return list(self.pert.overlay["by_anesthetic"][anesthetic].keys())

    def _run_filename(
        self, anesthetic: str, ablation_targets: list[str], seed: int
    ) -> Path:
        if not ablation_targets:
            tag = "baseline"
        else:
            tag = "ablate_" + "+".join(sorted(ablation_targets))
        # Sanitize tag for filename use
        tag = tag.replace("/", "_")
        return self.runs_dir / f"{anesthetic}_{tag}_{seed}.json"

    def _persist(self, result: RunResult) -> None:
        path = self._run_filename(
            result.anesthetic, result.ablation_targets, result.seed
        )
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, allow_nan=False)

    def _maybe_load(
        self, anesthetic: str, ablation_targets: list[str], seed: int
    ) -> Optional[RunResult]:
        path = self._run_filename(anesthetic, ablation_targets, seed)
        if not path.exists():
            return None
        try:
            d = json.load(open(path))
            return RunResult(**d)
        except (json.JSONDecodeError, TypeError):
            return None

    def _do_run(
        self,
        anesthetic: str,
        ablation_targets: list[str],
        dose: float,
        seed: int,
        scenario: dict,
        mutant_overlay_modifications: Optional[dict] = None,
    ) -> RunResult:
        """Single ablation run on a fresh brain."""
        from brian2 import ms, pA

        t_start = time.time()
        brain = self.substrate(seed)

        # Compute base profile, then ablate
        profile = self.pert.compute_perturbation_vector(anesthetic, dose)
        if mutant_overlay_modifications:
            profile = self._apply_mutant_modifications(profile, mutant_overlay_modifications)
        profile = ablate_profile(profile, ablation_targets)

        # On the demo substrate, apply_to_brain's W_chem-based hooks are no-ops
        # (zero matrix); current-injection hooks (Complex I + K2P) and snare W_syn
        # scaling do work. For the demo we additionally collapse multi-class
        # perturbation magnitude to additive hyperpolarizing currents matching
        # the Phase G dose-response sweep convention.
        duration_ms = scenario.get("duration_ms", 2000)
        if self.substrate_label == "phase_g_demo_50neuron":
            self._apply_demo_perturbation(brain, profile)
            brain.net.run(duration_ms * ms)
        else:
            self.pert.apply_to_brain(brain, anesthetic, dose, profile=profile)
            brain.net.run(duration_ms * ms)

        # Readout
        n_spikes = len(brain.spikes.t) if hasattr(brain, "spikes") else 0
        run_state = {
            "n_spikes": n_spikes,
            "duration_s": duration_ms / 1000.0,
            "n_neurons": brain.N if hasattr(brain, "N") else 50,
        }
        phenotype = self.readout(brain, run_state)

        # Perturbation summary post-ablation
        all_occupancies = [
            e["occupancy"] for cls in profile.per_class.values() for e in cls
        ]
        summary = {
            "n_classes_engaged_post_ablation": len(profile.per_class),
            "max_class_occupancy_post_ablation": max(all_occupancies) if all_occupancies else 0.0,
            "n_targets_engaged_post_ablation": sum(
                1 for o in all_occupancies if o > 0.10
            ),
            "ablation_targets": ablation_targets,
        }

        wall = time.time() - t_start
        result = RunResult(
            anesthetic=anesthetic,
            dose_multiplier=dose,
            ablation_targets=ablation_targets,
            seed=seed,
            phenotype=phenotype,
            perturbation_summary=summary,
            wall_clock_sec=round(wall, 3),
            meta={
                "substrate": self.substrate_label,
                "scenario": scenario,
                "mutant": mutant_overlay_modifications,
            },
        )
        self._persist(result)
        return result

    def _apply_demo_perturbation(self, brain, profile: PerturbationProfile) -> None:
        """Demo-network perturbation analog (matches phase_g dose_response_sweep)."""
        from brian2 import pA

        def class_max(name: str) -> float:
            return max((e["perturbation_magnitude"]
                        for e in profile.per_class.get(name, [])), default=0.0)
        complex_i_max = class_max("complex_i_block")
        k2p_max = class_max("k2p_potentiation")
        snare_max = class_max("snare_cooperativity")
        nachr_max = class_max("nachr_antagonism")
        gaba_max = class_max("gaba_potentiation")
        glucl_max = class_max("glucl_potentiation")
        nca_max = class_max("nca_block")
        complex_ii_max = class_max("complex_ii_block")

        ci_pA = -60.0 * complex_i_max - 30.0 * complex_ii_max
        k2p_pA = -30.0 * k2p_max
        snare_pA = -50.0 * snare_max
        nachr_pA = -30.0 * nachr_max
        gaba_pA = -30.0 * gaba_max
        glucl_pA = -30.0 * glucl_max
        nca_pA = +20.0 * nca_max  # nca_block reduces depolarizing leak
        hyperpol_pA = ci_pA + k2p_pA + snare_pA + nachr_pA + gaba_pA + glucl_pA + nca_pA
        brain.neurons.I_ext = brain.neurons.I_ext + hyperpol_pA * pA

    def _apply_mutant_modifications(
        self, profile: PerturbationProfile, mods: dict
    ) -> PerturbationProfile:
        """Apply mutant background modifications to a perturbation profile.

        mods example: {"GAS-1.rate_factor": 0.4} → multiply Complex I block
        magnitude on GAS-1 by 0.4 (reduced Complex I activity baseline).
        Or {"TWK-18.leak_scale": 2.0} → multiply K2P potentiation magnitude
        on TWK-18 by 2.0 (K2P GoF — increased baseline leak amplifies
        halothane potentiation).
        """
        if not mods:
            return profile
        out = PerturbationProfile(
            anesthetic=profile.anesthetic, dose_multiplier=profile.dose_multiplier
        )
        for cls, entries in profile.per_class.items():
            new_entries = []
            for e in entries:
                new_e = dict(e)
                for key, factor in mods.items():
                    if "." in key:
                        target, _ = key.split(".", 1)
                        if e["target"] == target:
                            new_e["perturbation_magnitude"] = (
                                e["perturbation_magnitude"] * float(factor)
                            )
                new_entries.append(new_e)
            if new_entries:
                out.per_class[cls] = new_entries
        return out

    # ---- Public API -------------------------------------------------

    def run_baseline(
        self, anesthetic: str, dose: float = 1.0, seed: int = 42,
        scenario: Optional[dict] = None, mutant_modifications: Optional[dict] = None,
        resume: bool = True,
    ) -> RunResult:
        if resume:
            cached = self._maybe_load(anesthetic, [], seed)
            if cached is not None:
                return cached
        scenario = scenario or {"duration_ms": 2000, "name": "spontaneous"}
        return self._do_run(anesthetic, [], dose, seed, scenario, mutant_modifications)

    def run_ablation(
        self, anesthetic: str, target: str | list[str], dose: float = 1.0,
        seed: int = 42, scenario: Optional[dict] = None,
        mutant_modifications: Optional[dict] = None,
        resume: bool = True,
    ) -> RunResult:
        if isinstance(target, str):
            ablation_targets = [target]
        else:
            ablation_targets = list(target)
        if resume:
            cached = self._maybe_load(anesthetic, ablation_targets, seed)
            if cached is not None:
                return cached
        scenario = scenario or {"duration_ms": 2000, "name": "spontaneous"}
        return self._do_run(
            anesthetic, ablation_targets, dose, seed, scenario, mutant_modifications
        )

    def run_full_ablation_suite(
        self, anesthetic: str, dose: float = 1.0, n_seeds: int = 5,
        scenario: Optional[dict] = None, resume: bool = True,
        mutant_modifications: Optional[dict] = None,
        targets: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """Full ablation suite: baseline + per-target ablation × n_seeds."""
        scenario = scenario or {"duration_ms": 2000, "name": "spontaneous"}
        targets = targets or self.list_targets(anesthetic)
        seeds = list(range(42, 42 + n_seeds))

        baseline_results = []
        for s in seeds:
            r = self.run_baseline(anesthetic, dose, s, scenario, mutant_modifications, resume)
            baseline_results.append(r)
            if verbose:
                print(f"  baseline seed={s} firing_rate={r.phenotype['firing_rate_Hz']:.2f} Hz")

        per_target: dict[str, list[RunResult]] = {}
        for t in targets:
            per_target[t] = []
            for s in seeds:
                r = self.run_ablation(anesthetic, t, dose, s, scenario, mutant_modifications, resume)
                per_target[t].append(r)
            if verbose:
                rates = [r.phenotype["firing_rate_Hz"] for r in per_target[t]]
                print(f"  ablate {t:10s} mean_rate={statistics.mean(rates):.2f} Hz "
                      f"(n={len(rates)})")

        # Build suite result
        suite = self._aggregate_suite(
            anesthetic, dose, baseline_results, per_target, scenario
        )
        suite_path = self.summary_dir / f"{anesthetic}_suite_summary.json"
        with open(suite_path, "w") as f:
            json.dump(suite, f, indent=2, allow_nan=False)
        return suite

    def _aggregate_suite(
        self, anesthetic: str, dose: float, baseline: list[RunResult],
        per_target: dict[str, list[RunResult]], scenario: dict,
    ) -> dict:
        baseline_rates = [r.phenotype["firing_rate_Hz"] for r in baseline]
        baseline_mean = statistics.mean(baseline_rates) if baseline_rates else 0.0
        baseline_stdev = (
            statistics.stdev(baseline_rates) if len(baseline_rates) >= 2 else 0.0
        )

        # Per-target effect size + uncorrected p (paired t-test approximation)
        per_target_stats: dict[str, dict] = {}
        for target, runs in per_target.items():
            rates = [r.phenotype["firing_rate_Hz"] for r in runs]
            n = len(rates)
            if n == 0:
                continue
            mean = statistics.mean(rates)
            stdev = statistics.stdev(rates) if n >= 2 else 0.0
            # Paired diffs (same seed for baseline and ablation)
            paired_diffs = []
            for r_b, r_a in zip(baseline, runs):
                paired_diffs.append(
                    r_a.phenotype["firing_rate_Hz"] - r_b.phenotype["firing_rate_Hz"]
                )
            d_mean = statistics.mean(paired_diffs) if paired_diffs else 0.0
            d_stdev = (
                statistics.stdev(paired_diffs) if len(paired_diffs) >= 2 else 0.0
            )
            # Cohen's d (paired)
            cohens_d = d_mean / d_stdev if d_stdev > 1e-9 else 0.0
            # Log-fold-change (firing rate metric)
            if baseline_mean > 1e-6 and mean > 1e-6:
                log10_fold = math.log10(mean / baseline_mean)
            else:
                log10_fold = 0.0
            # Approximate paired t-statistic + p
            if d_stdev > 1e-9 and n >= 2:
                t_stat = d_mean / (d_stdev / math.sqrt(n))
                p_value = self._t_to_p_two_tailed(t_stat, n - 1)
            else:
                t_stat = 0.0
                p_value = 1.0
            per_target_stats[target] = {
                "n_seeds": n,
                "ablation_mean_rate": round(mean, 4),
                "ablation_stdev_rate": round(stdev, 4),
                "paired_mean_diff": round(d_mean, 4),
                "paired_stdev_diff": round(d_stdev, 4),
                "cohens_d_paired": round(cohens_d, 3),
                "log10_fold_vs_baseline": round(log10_fold, 3),
                "t_stat": round(t_stat, 3),
                "p_value_uncorrected": round(p_value, 5),
            }

        # BH-FDR multiple-comparison correction across targets
        if per_target_stats:
            ranked = sorted(
                per_target_stats.items(), key=lambda kv: kv[1]["p_value_uncorrected"]
            )
            m = len(ranked)
            for rank, (target, s) in enumerate(ranked, start=1):
                bh = s["p_value_uncorrected"] * m / rank
                s["p_value_bh_corrected"] = round(min(1.0, bh), 5)
            # Enforce monotone non-decreasing BH q-values
            min_q = 1.0
            for target, _ in reversed(ranked):
                s = per_target_stats[target]
                if s["p_value_bh_corrected"] > min_q:
                    s["p_value_bh_corrected"] = round(min_q, 5)
                else:
                    min_q = s["p_value_bh_corrected"]

        # Necessity classification
        for target, s in per_target_stats.items():
            d = abs(s["cohens_d_paired"])
            log_fold = abs(s["log10_fold_vs_baseline"])
            p_corr = s["p_value_bh_corrected"]
            if (d > 0.8 or log_fold > 0.30) and p_corr < 0.05:
                s["necessity_class"] = "necessary"
            elif d < 0.2 and log_fold < 0.075 and p_corr > 0.20:
                s["necessity_class"] = "bystander"
            else:
                s["necessity_class"] = "ambiguous"

        ranked_necessary = sorted(
            (t for t, s in per_target_stats.items() if s["necessity_class"] == "necessary"),
            key=lambda t: -abs(per_target_stats[t]["cohens_d_paired"]),
        )
        bystanders = [
            t for t, s in per_target_stats.items() if s["necessity_class"] == "bystander"
        ]
        ambiguous = [
            t for t, s in per_target_stats.items() if s["necessity_class"] == "ambiguous"
        ]

        return {
            "anesthetic": anesthetic,
            "dose_multiplier": dose,
            "scenario": scenario,
            "baseline": {
                "n_seeds": len(baseline),
                "mean_firing_rate_Hz": round(baseline_mean, 4),
                "stdev_firing_rate_Hz": round(baseline_stdev, 4),
            },
            "per_target": per_target_stats,
            "ranked_causally_necessary": ranked_necessary,
            "bystanders": bystanders,
            "ambiguous": ambiguous,
            "n_targets_tested": len(per_target_stats),
            "_meta": {
                "substrate": self.substrate_label,
                "necessity_thresholds": {
                    "cohens_d": 0.8,
                    "log10_fold": 0.30,
                    "alpha_bh": 0.05,
                },
                "bystander_thresholds": {
                    "cohens_d": 0.2,
                    "log10_fold": 0.075,
                    "alpha_bh": 0.20,
                },
            },
        }

    def compute_class_attribution(self, suite: dict) -> dict:
        """Group per-target results by mechanism class for class-level summary."""
        overlay = self.pert.overlay["by_anesthetic"][suite["anesthetic"]]
        by_class: dict[str, list[dict]] = {}
        for target, stats in suite["per_target"].items():
            cls = overlay.get(target, {}).get("mechanism_class", "unknown")
            by_class.setdefault(cls, []).append({"target": target, **stats})
        out = {}
        for cls, entries in by_class.items():
            ds = [abs(e["cohens_d_paired"]) for e in entries]
            n_necessary = sum(1 for e in entries if e["necessity_class"] == "necessary")
            out[cls] = {
                "n_targets": len(entries),
                "n_necessary": n_necessary,
                "max_abs_cohens_d": round(max(ds), 3) if ds else 0.0,
                "mean_abs_cohens_d": round(statistics.mean(ds), 3) if ds else 0.0,
                "necessary_targets": [
                    e["target"] for e in entries if e["necessity_class"] == "necessary"
                ],
            }
        return out

    @staticmethod
    def _t_to_p_two_tailed(t_stat: float, df: int) -> float:
        """Approximate two-tailed p from t-stat using a normal-tail approximation
        for df>=10, and a conservative Student-t correction for small df.

        For small df (<10), uses Hill 1970 approximation. Adequate accuracy for
        decision-rule purposes (necessity classification).
        """
        if df < 1 or t_stat == 0.0:
            return 1.0
        x = abs(t_stat)
        if df >= 30:
            # Standard normal approximation
            return 2.0 * (1.0 - _stdnorm_cdf(x))
        # Hill 1970 approximation for student-t
        a = df - 0.5
        b = 48.0 * a * a
        z = a * math.log(1.0 + x * x / df)
        z = z * (((-3.0 / b - 0.5) / a + 1.0))
        if z >= 0:
            p_one = 1.0 - _stdnorm_cdf(math.sqrt(z))
        else:
            p_one = _stdnorm_cdf(-math.sqrt(-z))
        return min(1.0, max(0.0, 2.0 * p_one))


def _stdnorm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ===== Test 4 (Eger non-immobilizer) ================================


NEGATIVE_OVERLAY_PATH = ROOT / "artifacts" / "kinetics" / "wave2_overlay_negative_v2.json"


def build_negative_control_overlay(
    ligands: list[str] = ("hexafluoroethane", "cis_12_dichloroethylene", "trans_12_dichloroethylene"),
    out_path: Path = NEGATIVE_OVERLAY_PATH,
) -> Path:
    """Generate a wave2_overlay_v2-style overlay for negative-control ligands.

    Reuses Phase D kinetic-shift translation logic on negative_vina_results.csv.
    Applies CP5 f_allo = 2.50× allosteric correction to occupancies for parity
    with wave2_overlay_v2.json.
    """
    NEG_VINA = ROOT / "artifacts" / "calibration" / "negative_vina_results.csv"
    OVERLAY_V2_PATH = ROOT / "artifacts" / "kinetics" / "wave2_overlay_v2.json"

    F_ALLO = 2.50
    R_KCAL = 1.9872041e-3
    T_K = 298.0
    RT = R_KCAL * T_K

    def kd_uM(dg: float) -> float:
        return math.exp(dg / RT) * 1e6

    # Use wave2_overlay_v2 mechanism_class assignments per target as canonical
    # mapping (target → mechanism_class is anesthetic-independent in the v2 schema).
    overlay_v2 = json.load(open(OVERLAY_V2_PATH))
    canon_anesth = "halothane"
    target_meta: dict[str, dict] = {}
    for target, info in overlay_v2["by_anesthetic"][canon_anesth].items():
        target_meta[target] = {
            "mechanism_class": info["mechanism_class"],
            "parameters": info.get("parameters", {}),
        }

    # Negative control concentrations: assume 1 mM aqueous (comparable to Eger
    # non-immobilizer test concentrations). Different from clinical anesthetic
    # 1× EC50; we encode this by computing occupancy at 1 mM with corrected Kd.
    neg_conc_uM = 1000.0

    rows = list(csv.DictReader(open(NEG_VINA)))
    by_lig: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["ligand"] not in ligands:
            continue
        try:
            aff = float(r["affinity_kcal_per_mol"])
        except (ValueError, TypeError):
            continue
        kd_corrected = kd_uM(aff) / F_ALLO
        gene = r["gene"]
        # Keep tightest binding pose per (ligand, gene)
        prev = by_lig.setdefault(r["ligand"], {})
        if gene not in prev or kd_corrected < prev[gene]:
            prev[gene] = kd_corrected

    # Build overlay
    out = {
        "by_anesthetic": {},
        "_meta": {
            "version": "negative_v2",
            "concentration_uM": neg_conc_uM,
            "f_allo_correction": F_ALLO,
            "ligands": list(ligands),
            "comment": (
                "Synthetic negative-control overlay matching wave2_overlay_v2 schema. "
                "Occupancies computed at 1 mM aqueous post-CP5 correction; kinetic shift "
                "parameter values inherited from canonical anesthetic (halothane) "
                "wave2_overlay_v2 entries since the binding pipeline cannot distinguish "
                "non-immobilizer kinetic shifts from anesthetic kinetic shifts at the "
                "binding-pipeline level (CP3, CP7 boundary findings). Test 4 is the "
                "post-WB3 substrate experiment that asks whether network-level "
                "integration distinguishes them despite similar binding profiles."
            ),
        },
    }
    for lig, kds in by_lig.items():
        ligand_entry = {}
        for gene, kd in kds.items():
            occ = neg_conc_uM / (neg_conc_uM + kd)
            meta = target_meta.get(gene, {"mechanism_class": "unknown", "parameters": {}})
            ligand_entry[gene] = {
                "mechanism_class": meta["mechanism_class"],
                "occupancy_1xEC50": occ,
                "parameters": meta["parameters"],
                "predicted_Kd_uM_corrected": round(kd, 2),
                "_negative_control": True,
            }
        out["by_anesthetic"][lig] = ligand_entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    return out_path


# ===== Mutant infrastructure ========================================


MUTANT_DEFINITIONS = {
    "gas-1": {
        "description": "gas-1(fc21) — Complex I activity reduced",
        "modifications": {"GAS-1.rate_factor": 0.4, "NUO-1.rate_factor": 0.4,
                          "NUO-2.rate_factor": 0.4, "NUO-3.rate_factor": 0.4,
                          "NUO-4.rate_factor": 0.4},
        "expected_phenotype": "halothane EC50 drops 2-3× (hypersensitive)",
        "anchor": "Morgan & Sedensky 1994 PMID 7943840 + Kayser 2001 PMID 11278828",
        "implemented": True,
    },
    "twk-18-cn110gf": {
        "description": "twk-18(cn110gf) — K2P GoF amplifies halothane K2P potentiation",
        "modifications": {"TWK-18.k2p_factor": 2.0, "TWK-29.k2p_factor": 2.0},
        "expected_phenotype": "halothane EC50 drops ~2× (hypersensitive, CP6 corrected)",
        "anchor": "Singaram 2011 PMID 22137475 — corrected per CP6",
        "implemented": True,
    },
    "sup-9-lf": {
        "description": "sup-9(n180lf) — K2P LoF reduces tonic inhibition substrate",
        "modifications": {"TWK-18.k2p_factor": 0.3, "TWK-29.k2p_factor": 0.3},
        "expected_phenotype": "modestly resistant",
        "anchor": "Singaram 2011 PMID 22137475",
        "implemented": True,
    },
    "unc-79": {
        "description": "unc-79(e1068) — NCA-1 Na leak channel disrupted",
        "modifications": {"NCA-2.nca_factor": 0.0, "UNC-79.nca_factor": 0.0,
                          "NLF-1.nca_factor": 0.0},
        "expected_phenotype": "halothane resistant",
        "anchor": "Sedensky & Meneely 1987 PMID 3576211",
        "implemented": False,
        "todo": ("Full implementation requires NCA-1/UNC-80 AlphaFold structures "
                 "(deferred to ColabFold T4 fallback per CP6) and Phase G wiring "
                 "for nca_block class on real LIFBrain. Scaffold present."),
    },
    "unc-13-s69": {
        "description": "unc-13(s69) — SNARE priming hypomorph",
        "modifications": {"UNC-64.snare_factor": 0.2, "RIC-4.snare_factor": 0.2,
                          "SNB-1.snare_factor": 0.2},
        "expected_phenotype": "hypersensitive — already-low release-p margin",
        "anchor": "Nguyen 1995 PMID 7647836",
        "implemented": False,
        "todo": ("UNC-13 not in Tier-1 panel; SNARE perturbation as proxy. "
                 "Real validation requires UNC-13 docked + Phase G coverage."),
    },
}


def get_mutant_modifications(mutant_name: str) -> dict:
    """Return the modifications dict for a named mutant, or {} for WT."""
    if not mutant_name or mutant_name == "WT":
        return {}
    if mutant_name not in MUTANT_DEFINITIONS:
        raise KeyError(f"unknown mutant {mutant_name!r}; "
                        f"known: {list(MUTANT_DEFINITIONS)}")
    spec = MUTANT_DEFINITIONS[mutant_name]
    if not spec.get("implemented", False):
        print(f"  WARNING: mutant {mutant_name!r} is scaffolded but not fully "
              f"implemented. Reason: {spec.get('todo', 'pending')}")
    return spec["modifications"]


# ===== Smoke tests ==================================================


def smoke_test_single_ablation():
    """CP3 smoke test 1: single ablation of UNC-49 under halothane."""
    print("=== CP3 smoke test 1: single ablation halothane × UNC-49 ===\n")
    h = AblationHarness()
    baseline = []
    ablated = []
    for s in (42, 43, 44):
        r_b = h.run_baseline("halothane", dose=1.0, seed=s, resume=False)
        r_a = h.run_ablation("halothane", "UNC-49", dose=1.0, seed=s, resume=False)
        baseline.append(r_b.phenotype["firing_rate_Hz"])
        ablated.append(r_a.phenotype["firing_rate_Hz"])
        print(f"  seed={s}: baseline={r_b.phenotype['firing_rate_Hz']:.2f} Hz, "
              f"ablate UNC-49={r_a.phenotype['firing_rate_Hz']:.2f} Hz, "
              f"Δ={r_a.phenotype['firing_rate_Hz']-r_b.phenotype['firing_rate_Hz']:+.2f}")
    print(f"\nMean baseline: {statistics.mean(baseline):.2f} Hz, "
          f"mean ablation: {statistics.mean(ablated):.2f} Hz")
    return baseline, ablated


def smoke_test_mini_suite():
    """CP3 smoke test 2: mini suite — halothane × 5 targets × 3 seeds."""
    print("\n=== CP3 smoke test 2: mini suite halothane × 5 targets × 3 seeds ===\n")
    h = AblationHarness()
    suite = h.run_full_ablation_suite(
        "halothane", dose=1.0, n_seeds=3,
        targets=["UNC-49", "TWK-18", "GAS-1", "ACR-16", "UNC-64"],
    )
    print(f"\nBaseline mean: {suite['baseline']['mean_firing_rate_Hz']:.2f} Hz")
    print(f"Targets necessary: {suite['ranked_causally_necessary']}")
    print(f"Bystanders: {suite['bystanders']}")
    print(f"Ambiguous: {suite['ambiguous']}")
    return suite


def smoke_test_cross_anesthetic():
    """CP3 smoke test 3: cross-anesthetic mini-comparison."""
    print("\n=== CP3 smoke test 3: cross-anesthetic mini-comparison ===\n")
    h = AblationHarness()
    out = {}
    for ane in ("halothane", "propofol", "etomidate"):
        suite = h.run_full_ablation_suite(
            ane, dose=1.0, n_seeds=3,
            targets=["UNC-49", "GAS-1", "TWK-18"],
            verbose=False,
        )
        out[ane] = suite
        rates = [suite['baseline']['mean_firing_rate_Hz']]
        per_target_summary = {
            t: {"d": s["cohens_d_paired"], "log_fold": s["log10_fold_vs_baseline"],
                "necessity": s["necessity_class"]}
            for t, s in suite["per_target"].items()
        }
        print(f"  {ane}: baseline {rates[0]:.2f} Hz, per-target: {per_target_summary}")
    return out


def smoke_test_test_4():
    """CP4 smoke: build negative-control overlay + mini ablation comparison."""
    print("\n=== CP4 smoke: Test 4 (Eger non-immobilizer) infrastructure ===\n")
    out_path = build_negative_control_overlay()
    print(f"Negative-control overlay shipped: {out_path}")
    neg_overlay = json.load(open(out_path))
    for lig in ("halothane",):
        # halothane via main overlay
        h = AblationHarness()
        suite = h.run_full_ablation_suite(
            lig, dose=1.0, n_seeds=3,
            targets=["UNC-49", "TWK-18", "GAS-1"],
            verbose=False,
        )
        print(f"  {lig} (anesthetic POS control): baseline "
              f"{suite['baseline']['mean_firing_rate_Hz']:.2f} Hz")

    # Build a harness pointing at the negative overlay for the non-immobilizers
    h_neg = AblationHarness(
        overlay_path=out_path,
        substrate_label="phase_g_demo_50neuron_neg_overlay",
    )
    for lig in ("hexafluoroethane", "cis_12_dichloroethylene", "trans_12_dichloroethylene"):
        if lig not in h_neg.list_anesthetics():
            print(f"  {lig}: not in negative overlay (skipped)")
            continue
        suite = h_neg.run_full_ablation_suite(
            lig, dose=1.0, n_seeds=3,
            targets=["UNC-49", "TWK-18", "GAS-1"],
            verbose=False,
        )
        eger = "non-immobilizer" if lig != "cis_12_dichloroethylene" else "anesthetic (Eger 2001)"
        print(f"  {lig:32s} ({eger}): baseline "
              f"{suite['baseline']['mean_firing_rate_Hz']:.2f} Hz")


def smoke_test_mutant():
    """CP5 smoke: gas-1 + twk-18 GoF mutant infrastructure on demo."""
    print("\n=== CP5 smoke: mutant phenotype infrastructure ===\n")
    h = AblationHarness()
    print("WT halothane baseline (3 seeds):")
    wt_rates = []
    for s in (42, 43, 44):
        r = h.run_baseline("halothane", dose=1.0, seed=s, resume=False)
        wt_rates.append(r.phenotype["firing_rate_Hz"])
    print(f"  mean firing rate: {statistics.mean(wt_rates):.2f} Hz")

    for mutant in ("gas-1", "twk-18-cn110gf"):
        mods = get_mutant_modifications(mutant)
        print(f"\n{mutant} background ({MUTANT_DEFINITIONS[mutant]['description']}):")
        print(f"  Expected: {MUTANT_DEFINITIONS[mutant]['expected_phenotype']}")
        m_rates = []
        for s in (42, 43, 44):
            r = h.run_baseline(
                "halothane", dose=1.0, seed=s,
                mutant_modifications=mods, resume=False,
            )
            m_rates.append(r.phenotype["firing_rate_Hz"])
        print(f"  Observed mean firing rate: {statistics.mean(m_rates):.2f} Hz "
              f"(WT: {statistics.mean(wt_rates):.2f} Hz)")
        delta = statistics.mean(m_rates) - statistics.mean(wt_rates)
        print(f"  Δ vs WT: {delta:+.2f} Hz "
              f"(hypersensitive expectation = lower firing rate at same dose)")


def smoke_test_subsat():
    """CP3 smoke test 4: sub-saturating dose to show ablation differentiation.

    At 1× clinical EC50 the Phase G demo saturates (binding saturation
    documented in Phase G dose-response). At dose=0.003-0.005× the demo
    has dynamic range and ablation effects are visible. This verifies the
    harness's necessity-classification machinery actually triggers when
    the network response has room to differ.
    """
    print("\n=== CP3 smoke test 4: sub-saturating dose differential ===\n")
    h = AblationHarness()
    suite = h.run_full_ablation_suite(
        "halothane", dose=0.003, n_seeds=3,
        targets=["UNC-49", "TWK-18", "GAS-1", "ACR-16", "UNC-64", "AVR-14",
                  "UNC-79"],
        verbose=False,
    )
    base = suite["baseline"]["mean_firing_rate_Hz"]
    print(f"Baseline (halothane @ 0.003× clinical EC50): {base:.2f} Hz "
          f"(n=3 seeds; demo dynamic range > 0)")
    print(f"\nPer-target ablation results:")
    print(f"  {'target':<10s} {'rate_Hz':>9s} {'log_fold':>9s} {'cohens_d':>9s} "
          f"{'p_corr':>8s} {'class':>12s}")
    for target in ["UNC-49", "TWK-18", "GAS-1", "ACR-16", "UNC-64", "AVR-14", "UNC-79"]:
        if target not in suite["per_target"]:
            continue
        s = suite["per_target"][target]
        print(f"  {target:<10s} {s['ablation_mean_rate']:>9.2f} "
              f"{s['log10_fold_vs_baseline']:>+9.3f} "
              f"{s['cohens_d_paired']:>+9.3f} "
              f"{s['p_value_bh_corrected']:>8.3f} "
              f"{s['necessity_class']:>12s}")
    print(f"\nNecessary: {suite['ranked_causally_necessary']}")
    print(f"Bystanders: {suite['bystanders']}")
    print(f"Ambiguous: {suite['ambiguous']}")
    return suite


def main() -> int:
    smoke_test_single_ablation()
    smoke_test_mini_suite()
    smoke_test_cross_anesthetic()
    smoke_test_test_4()
    smoke_test_mutant()
    smoke_test_subsat()
    print("\nAll smoke tests complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
