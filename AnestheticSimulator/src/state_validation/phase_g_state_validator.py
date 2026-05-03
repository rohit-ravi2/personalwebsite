"""phase_g_state_validator — V1 network-state validator.

Replaces the binding-driven Phase G perturbation manager with a literature-grounded,
connectome-constrained network-state predictor.

Architecture
------------
1. Load the anesthetic_perturbation_table.csv (per-anesthetic, per-mechanism Hill curves).
2. Load LIFBrain (300 neurons, Cook 2019 + per-edge CeNGEN sign override).
3. For (genotype, anesthetic, dose, seed):
   - Apply genotype baseline shift (mutant_baseline_perturbations.csv)
   - Apply per-class anesthetic engagement via Hill(dose, target_EC50)
   - Run 60s simulation (10s warmup + 50s record)
   - Compute network-state metrics
4. Fit Hill curve over doses, extract immobilization EC50 = dose where quiescent_fraction = 0.5

Convention
----------
Effect application — at engagement=e (range [0,1]), multiplicative factor on parameter is:
    effect_factor(e) = 1 + (max - 1) × e
For blocking classes (max < 1), this gives reduction.
For potentiating classes (max > 1), this gives enhancement.

This module does NOT depend on Vina occupancy or the wave2_overlay_v2.json.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

# Parent paths to import C. elegans simulator brain
ANESTH_ROOT = Path('/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator')
SIM_ROOT = Path('/home/rohit/Desktop/website/personalwebsite/scripts')
sys.path.insert(0, str(SIM_ROOT))


# ===== Perturbation table =====

@dataclass
class PerturbationRow:
    anesthetic: str
    mechanism_class: str
    target_EC50_uM: float | None
    max_effect_factor: float | None
    hill_n: float
    source_PMID: str
    evidence_grade: str

    def engagement(self, dose_uM: float) -> float:
        """Hill engagement at given dose."""
        if self.target_EC50_uM is None or self.max_effect_factor is None:
            return 0.0
        if dose_uM <= 0:
            return 0.0
        n = self.hill_n
        return (dose_uM ** n) / (dose_uM ** n + self.target_EC50_uM ** n)

    def factor(self, dose_uM: float) -> float:
        """Multiplicative factor on the target parameter at given dose."""
        if self.target_EC50_uM is None or self.max_effect_factor is None:
            return 1.0
        e = self.engagement(dose_uM)
        return 1.0 + (self.max_effect_factor - 1.0) * e


def load_perturbation_table(path: Path) -> dict[str, dict[str, PerturbationRow]]:
    """Returns {anesthetic: {mechanism_class: PerturbationRow}}."""
    out: dict[str, dict[str, PerturbationRow]] = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            if line.startswith('anesthetic,'):
                continue  # header
            parts = next(csv.reader([line]))
            anest, mech, ec50_s, mxf_s, hn_s, _src, pmid, grade, _notes = parts[:9]
            ec50 = float(ec50_s) if ec50_s else None
            mxf = float(mxf_s) if mxf_s else None
            hn = float(hn_s) if hn_s else 1.0
            row = PerturbationRow(anest, mech, ec50, mxf, hn, pmid, grade)
            out.setdefault(anest, {})[mech] = row
    return out


# ===== Mutant baseline =====

@dataclass
class MutantBaseline:
    gene: str
    direction: str
    complex_i_factor: float
    nca_leak_factor: float
    wsyn_global_factor: float
    wsyn_excitatory_factor: float
    literature_ratio: str
    notes: str
    # V4: optional fly-specific entry point — K2P baseline leak removal
    # (Sandman LoF in fly = no constitutive K2P leak → depolarized baseline → resistant)
    k2p_baseline_factor: float = 1.0


def load_mutant_table(path: Path) -> dict[str, MutantBaseline]:
    """Load a mutant baseline table (worm or fly). Tolerates both worm-V3 schema
    (7 leading numeric fields) and fly-V4 schema (8 fields with k2p_baseline_factor)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            if line.startswith('gene,'):
                # detect schema by checking column count
                continue
            parts = next(csv.reader([line]))
            gene, direction, ci, nca, wsg, wse = parts[0:6]
            # Detect schema: V4 fly has k2p_baseline_factor at index 6, V3 worm has lit_ratio
            try:
                # try parsing index 6 as float — V4 schema
                k2p_baseline = float(parts[6])
                lit_r = parts[7]
                notes = parts[-1] if len(parts) > 8 else ''
            except (ValueError, IndexError):
                # V3 worm schema
                k2p_baseline = 1.0
                lit_r = parts[6]
                notes = parts[-1] if len(parts) > 7 else ''
            out[gene] = MutantBaseline(
                gene=gene, direction=direction,
                complex_i_factor=float(ci), nca_leak_factor=float(nca),
                wsyn_global_factor=float(wsg), wsyn_excitatory_factor=float(wse),
                literature_ratio=lit_r, notes=notes,
                k2p_baseline_factor=k2p_baseline,
            )
    return out


# ===== Network-state metrics =====

# Command-interneuron set for the "quiescent" measurement.
# Locomotion command: AVA, AVB, AVD, AVE, PVC (Chalfie 1985; Gray 2005)
# Plus AIB, RIB, RIM as premotor / state-coordinating interneurons.
COMMAND_NEURONS = ['AVAL','AVAR','AVBL','AVBR','AVDL','AVDR','AVEL','AVER',
                   'PVCL','PVCR','AIBL','AIBR','RIBL','RIBR','RIML','RIMR']

QUIESCENT_RATE_THRESHOLD_HZ = 3.0  # mean command-neuron rate below this = quiescent state


def compute_metrics(spike_times_s: np.ndarray, spike_neuron_ids: np.ndarray,
                    neuron_names: list[str], sim_duration_s: float,
                    record_start_s: float = 10.0,
                    bin_dt_s: float = 0.5,
                    quiescent_threshold_hz: float = QUIESCENT_RATE_THRESHOLD_HZ,
                    command_set: list[str] | list[int] | None = None) -> dict:
    """Compute three primary network-state metrics from a recording.

    Returns:
        quiescent_fraction: fraction of bins where command-neuron mean rate < threshold
        mean_firing_rate_hz: mean across command neurons over the record window
        state_autocorrelation: lag-1 autocorrelation of population state vector
        + per-bin and per-neuron diagnostics for plotting
    """
    # Filter to record window
    record_end_s = sim_duration_s
    mask = (spike_times_s >= record_start_s) & (spike_times_s <= record_end_s)
    ts = spike_times_s[mask]
    ids = spike_neuron_ids[mask]

    # Build command-neuron index list
    if command_set is None:
        cmd_idxs = [i for i, n in enumerate(neuron_names) if n in COMMAND_NEURONS]
    elif command_set and isinstance(command_set[0], int):
        cmd_idxs = list(command_set)  # already indices (e.g., from FlyLarvaBrain)
    else:
        cmd_idxs = [i for i, n in enumerate(neuron_names) if n in command_set]

    # Bin spikes per neuron
    n_bins = int((record_end_s - record_start_s) / bin_dt_s)
    bin_edges = np.linspace(record_start_s, record_end_s, n_bins + 1)
    n_neurons = len(neuron_names)
    counts = np.zeros((n_neurons, n_bins), dtype=np.float64)
    for nid, t in zip(ids, ts):
        if 0 <= nid < n_neurons:
            b = min(n_bins - 1, max(0, int((t - record_start_s) / bin_dt_s)))
            counts[nid, b] += 1
    rates_hz = counts / bin_dt_s  # [neurons × bins], Hz

    # Command-neuron mean rate per bin
    if cmd_idxs:
        cmd_mean_rate = rates_hz[cmd_idxs, :].mean(axis=0)  # [bins]
    else:
        cmd_mean_rate = rates_hz.mean(axis=0)
    quiescent_fraction = float((cmd_mean_rate < quiescent_threshold_hz).mean())
    mean_firing_rate_hz = float(cmd_mean_rate.mean())

    # State autocorrelation (lag-1) on population state vector
    pop_vec = rates_hz.mean(axis=0)  # mean firing rate across neurons per bin
    if len(pop_vec) > 1:
        # Pearson r between pop_vec[:-1] and pop_vec[1:]
        x = pop_vec[:-1] - pop_vec[:-1].mean()
        y = pop_vec[1:] - pop_vec[1:].mean()
        denom = np.sqrt((x**2).sum() * (y**2).sum())
        state_autocorr = float((x * y).sum() / denom) if denom > 0 else 0.0
    else:
        state_autocorr = 0.0

    # Whole-network firing rate (across all neurons, not just command)
    network_mean_rate = float(rates_hz.mean())

    return {
        'quiescent_fraction': quiescent_fraction,
        'command_mean_firing_rate_hz': mean_firing_rate_hz,
        'network_mean_firing_rate_hz': network_mean_rate,
        'state_autocorrelation_lag1': state_autocorr,
        'n_bins': n_bins,
        'n_command_neurons_in_brain': len(cmd_idxs),
    }


# ===== Genotype + anesthetic application =====

# Per-mechanism-class scale factors (pA hyperpolarizing current at saturation × engagement).
# These are the bridge from "the perturbation engaged X% of receptor population" to
# "the network sees Y pA of effective current shift". They're calibrated by the
# alpha tuning step in M3 (Gate 1).
DEFAULT_PER_CLASS_PA_AT_SATURATION = {
    'complex_i_block':       60.0,   # K-ATP coupling — global
    'complex_ii_block':      20.0,
    'k2p_potentiation':      30.0,   # K2P-expressing neurons only
    'nca_block':             40.0,   # NCA-expressing neurons; sign flips on losing leak (depolarizing)
    'gaba_potentiation':     30.0,   # GABA-A-expressing neurons (UNC-49 expression)
    'glucl_potentiation':    20.0,   # GluCl-expressing neurons
    'nachr_antagonism':      30.0,   # ACh-receptor expressing
    'snare_cooperativity':   50.0,   # global W_syn modifier (handled separately)
}

# Neuron-name → list of mechanism classes that target it (rough heuristic by neuron type)
# In V1 we apply globally; refinement to CeNGEN-tagged sets is V2.
def resolve_target_neurons(brain, mechanism_class: str) -> list[int]:
    """Return list of neuron indices in the brain that are targeted by this mechanism class.
    V1 = global for simplicity; V2 will use CeNGEN expression tables."""
    return list(range(brain.N))


def apply_genotype(brain, mutant: MutantBaseline | None, alpha_calib: float) -> None:
    """Apply a mutant baseline shift BEFORE anesthetic perturbation.

    For Complex I LoF mutants (gas-1 etc.): hyperpolarize all neurons via reduced ATP/K-ATP.
    For NCA LoF mutants (unc-79/80): hyperpolarize NCA-expressing neurons (lost leak).
    For Gαo LoF mutants (goa-1, dgk-1): scale W_syn × wsyn_global_factor (>1 = more excitable).
    """
    import brian2
    pA = brian2.pA
    if mutant is None:
        return
    # Complex I: K-ATP coupling = (1 - factor) × baseline pA shift
    if mutant.complex_i_factor < 1.0:
        ci_pa = -50.0 * (1.0 - mutant.complex_i_factor) * alpha_calib
        brain.neurons.I_ext[:] = brain.neurons.I_ext[:] + ci_pa * pA
    # NCA leak loss: depolarizing leak gone → hyperpolarize NCA-expressing neurons
    if mutant.nca_leak_factor < 1.0:
        nca_pa = -30.0 * (1.0 - mutant.nca_leak_factor) * alpha_calib
        # V1: apply to all (NCA expression is broad in interneurons)
        brain.neurons.I_ext[:] = brain.neurons.I_ext[:] + nca_pa * pA
    # Gαo signaling LoF: scale Brian2 synapse weights (live state variables)
    # Bug fix: previous version scaled brain._W_chem_runtime which is a debug-only
    # numpy reference; the actual simulation reads from syn_exc.w / syn_inh.w which
    # are Brian2 state variables that propagate to the cython codegen.
    if mutant.wsyn_global_factor != 1.0:
        f = mutant.wsyn_global_factor
        if hasattr(brain, 'syn_exc') and brain.syn_exc is not None and len(brain.syn_exc) > 0:
            brain.syn_exc.w[:] = np.asarray(brain.syn_exc.w[:]) * f
        if hasattr(brain, 'syn_inh') and brain.syn_inh is not None and len(brain.syn_inh) > 0:
            brain.syn_inh.w[:] = np.asarray(brain.syn_inh.w[:]) * f

    # V4 fly-specific: K2P baseline leak loss (Sandman / ORK1 LoF) modeled as
    # depolarizing baseline current per neuron (loss of hyperpolarizing K leak)
    if getattr(mutant, 'k2p_baseline_factor', 1.0) < 1.0:
        k2p_pa = +30.0 * (1.0 - mutant.k2p_baseline_factor) * alpha_calib
        brain.neurons.I_ext[:] = brain.neurons.I_ext[:] + k2p_pa * pA


def apply_anesthetic(brain, profile: dict[str, PerturbationRow], dose_uM: float,
                     alpha_calib: float) -> None:
    """Apply per-class Hill-engagement perturbations to the brain.

    profile: {mechanism_class: PerturbationRow}
    dose_uM: anesthetic concentration in µM
    alpha_calib: global perturbation strength multiplier (calibrated in M3 Gate 1)
    """
    import brian2
    pA = brian2.pA

    # Compute engagements (0..1) per class
    eng = {cls: row.engagement(dose_uM) for cls, row in profile.items()}

    # Effect-factor mapping per class: how does engagement translate to network-level shift?
    # Convention: blocking classes (factor < 1 at saturation) → hyperpolarize via I_ext shift.
    # Potentiating classes (factor > 1 at saturation) → effect direction depends on class:
    #   gaba_potentiation: enhanced inhibition → hyperpolarize
    #   glucl_potentiation: enhanced inhibition → hyperpolarize
    #   k2p_potentiation: enhanced K leak → hyperpolarize

    # Sum hyperpolarizing effects in pA
    total_pa = 0.0
    for cls, e in eng.items():
        if e == 0:
            continue
        sat_pa = DEFAULT_PER_CLASS_PA_AT_SATURATION.get(cls, 0.0)
        if cls in ('complex_i_block', 'complex_ii_block', 'nachr_antagonism', 'nca_block'):
            # blocking — sign of effect is hyperpolarizing
            total_pa += -sat_pa * e
        elif cls in ('gaba_potentiation', 'glucl_potentiation', 'k2p_potentiation'):
            # potentiating inhibition — also hyperpolarizing
            total_pa += -sat_pa * e
        else:
            pass

    total_pa *= alpha_calib
    if total_pa != 0:
        brain.neurons.I_ext[:] = brain.neurons.I_ext[:] + total_pa * pA

    # SNARE cooperativity: scale Brian2 synapse weights live (same bug-fix as genotype path)
    if 'snare_cooperativity' in eng and eng['snare_cooperativity'] > 0:
        snare_e = eng['snare_cooperativity']
        snare_max = profile['snare_cooperativity'].max_effect_factor
        if snare_max is not None:
            factor = 1.0 + (snare_max - 1.0) * snare_e
            if hasattr(brain, 'syn_exc') and brain.syn_exc is not None and len(brain.syn_exc) > 0:
                brain.syn_exc.w[:] = np.asarray(brain.syn_exc.w[:]) * factor
            if hasattr(brain, 'syn_inh') and brain.syn_inh is not None and len(brain.syn_inh) > 0:
                brain.syn_inh.w[:] = np.asarray(brain.syn_inh.w[:]) * factor


# ===== Main run =====

def run_single(anesthetic: str, dose_uM: float, seed: int, sim_duration_s: float,
               profile: dict[str, PerturbationRow],
               mutant: MutantBaseline | None = None,
               alpha_calib: float = 1.0,
               brain_factory=None,
               quiescent_threshold_hz: float = QUIESCENT_RATE_THRESHOLD_HZ,
               command_set: list[str] | list[int] | None = None) -> dict:
    """Run one (anesthetic, dose, seed, mutant) simulation and return metrics.

    Args:
        brain_factory: callable(seed) → brain instance. Defaults to worm LIFBrain.
                       For fly, pass a callable that returns a SeededFlyLarvaBrain.
        quiescent_threshold_hz: organism-specific quiescent threshold.
                                Worm baseline ~5 Hz → threshold 3.0 Hz.
                                Fly baseline ~2 Hz → threshold ~1.0 Hz.
        command_set: list of command-neuron names or indices. Defaults to worm set.
                     For fly, pass brain.command_neurons_idx.
    """
    np.random.seed(seed)
    if brain_factory is None:
        # Default = worm
        from brain.lif_brain import LIFBrain
        class SeededLIFBrain(LIFBrain):
            _brian2_seed = seed
        brain = SeededLIFBrain(use_per_edge_glu_signs=True)
    else:
        brain = brain_factory(seed)
    apply_genotype(brain, mutant, alpha_calib)
    apply_anesthetic(brain, profile, dose_uM, alpha_calib)
    brain.run(sim_duration_s * 1000.0)  # ms
    # Pull spike data — Brian2 quantities → seconds
    import brian2
    if hasattr(brain, 'spikes') and len(brain.spikes.t) > 0:
        spike_t = np.asarray(brain.spikes.t / brian2.second)
        spike_i = np.asarray(brain.spikes.i, dtype=int)
    else:
        spike_t = np.array([])
        spike_i = np.array([], dtype=int)
    # If command set wasn't passed, fall back to the brain's own attribute
    # (FlyLarvaBrain exposes command_neurons_idx; LIFBrain does not, so worm
    # uses the COMMAND_NEURONS module-level default)
    effective_command = command_set
    if effective_command is None and hasattr(brain, 'command_neurons_idx'):
        effective_command = list(brain.command_neurons_idx)
    metrics = compute_metrics(spike_t, spike_i, brain.names, sim_duration_s,
                              record_start_s=min(10.0, sim_duration_s * 0.2),
                              quiescent_threshold_hz=quiescent_threshold_hz,
                              command_set=effective_command)
    metrics['anesthetic'] = anesthetic
    metrics['dose_uM'] = dose_uM
    metrics['seed'] = seed
    metrics['mutant'] = mutant.gene if mutant else 'WT'
    metrics['alpha_calib'] = alpha_calib
    return metrics


def hill_fit_ec50(doses: np.ndarray, quiescent_fractions: np.ndarray,
                  threshold: float = 0.5) -> float | None:
    """Find dose where quiescent_fraction crosses threshold (0.5 by default).
    Linear interpolation in log-space."""
    if len(doses) < 2 or len(quiescent_fractions) != len(doses):
        return None
    order = np.argsort(doses)
    doses = doses[order]
    qf = quiescent_fractions[order]
    # Find first crossing
    for i in range(len(qf) - 1):
        if qf[i] < threshold <= qf[i+1] or qf[i+1] < threshold <= qf[i]:
            # Interpolate in log dose
            ld0, ld1 = np.log10(doses[i]), np.log10(doses[i+1])
            q0, q1 = qf[i], qf[i+1]
            if q1 == q0:
                return doses[i]
            frac = (threshold - q0) / (q1 - q0)
            return float(10.0 ** (ld0 + frac * (ld1 - ld0)))
    if qf.max() < threshold:
        return None  # threshold never reached
    return float(doses[np.argmin(np.abs(qf - threshold))])
