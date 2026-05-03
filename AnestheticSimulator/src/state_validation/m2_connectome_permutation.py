"""V5 M2 — connectome permutation null tests.

Tests whether the SPECIFIC wiring of the Cook 2019 / Winding 2023 connectomes
is load-bearing for the V3/V4 results.

Three permutation classes per organism:
  P1 — Erdős-Rényi rewiring (preserve total edge count + sign distribution)
  P2 — Configuration model (preserve in/out degree per neuron)
  P3 — Block shuffle within cell-type (preserve cell-type × cell-type aggregate;
       randomize within each block)

Plus:
  P4 — Cross-organism swap (worm perturbation table on fly connectome, vice versa)

For each permutation: re-run Gate 1 (halothane WT @ frozen α) + Gate 4 (Eger
non-immobilizer specificity). Compare to V3/V4 baseline. PASS criterion for the
hypothesis "connectome is load-bearing": permutations should FAIL Gate 1
(predicted EC50 not within 2× of 340 µM at frozen α).
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
SIMV = ANESTH / 'src'
sys.path.insert(0, str(SIMV))
sys.path.insert(0, str(ROOT / 'scripts'))


# ===== Permutation algorithms =====

def permute_er(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erdős-Rényi rewiring: preserve number of edges of each sign,
    randomize which neurons they connect."""
    N = W.shape[0]
    out = np.zeros_like(W)
    for sign in (+1, -1):
        i_orig, j_orig = np.where(np.sign(W) == sign)
        n_edges = len(i_orig)
        weights = W[i_orig, j_orig]
        # Randomize positions
        new_i = rng.integers(0, N, size=n_edges)
        new_j = rng.integers(0, N, size=n_edges)
        for k in range(n_edges):
            out[new_i[k], new_j[k]] = weights[k]
    return out.astype(W.dtype)


def permute_configuration(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Configuration model: preserve in/out degree per neuron + edge weights;
    randomize the bipartite matching of stubs."""
    N = W.shape[0]
    out = np.zeros_like(W)
    for sign in (+1, -1):
        i_orig, j_orig = np.where(np.sign(W) == sign)
        n_edges = len(i_orig)
        weights = W[i_orig, j_orig].copy()
        # Shuffle the j (post) array — preserves out-degree of i
        new_j = j_orig.copy()
        rng.shuffle(new_j)
        for k in range(n_edges):
            out[i_orig[k], new_j[k]] += weights[k]  # += in case of collisions
    return out.astype(W.dtype)


def permute_block_shuffle(W: np.ndarray, celltype_per_neuron: list, rng: np.random.Generator) -> np.ndarray:
    """Block shuffle: preserve cell-type × cell-type aggregate weight; randomize within."""
    N = W.shape[0]
    types = sorted(set(celltype_per_neuron))
    type_to_idxs = {t: [i for i, c in enumerate(celltype_per_neuron) if c == t] for t in types}
    out = np.zeros_like(W)
    for t_pre in types:
        for t_post in types:
            pre_idxs = type_to_idxs[t_pre]
            post_idxs = type_to_idxs[t_post]
            block = W[np.ix_(pre_idxs, post_idxs)]
            # Flatten edges in this block, shuffle their positions
            flat_w = block.flatten()
            n_cells = len(flat_w)
            shuffled = flat_w.copy()
            rng.shuffle(shuffled)
            out[np.ix_(pre_idxs, post_idxs)] = shuffled.reshape(block.shape)
    return out.astype(W.dtype)


# ===== Worm permuted brain factory =====

PERMUTED_WORM_NPZ_CACHE = Path('/tmp/v5_permuted_worm_npz')
PERMUTED_FLY_MATRICES_CACHE = Path('/tmp/v5_permuted_fly_matrices')


def precompute_worm_permutations(perm_seed: int = 7):
    """Build all 3 worm permuted connectome npz files ONCE; cache to /tmp."""
    PERMUTED_WORM_NPZ_CACHE.mkdir(parents=True, exist_ok=True)
    original_npz = ROOT / 'scripts/brain/artifacts/connectome.npz'
    d = dict(np.load(original_npz, allow_pickle=True))
    W_per_edge = d['W_chem_per_edge'].astype(np.float32).copy()
    W_chem_default = d['W_chem'].astype(np.float32).copy()
    klass = [str(c) for c in d['klass']]
    for perm_kind in PERMS:
        out_path = PERMUTED_WORM_NPZ_CACHE / f'{perm_kind}_seed{perm_seed}.npz'
        if out_path.exists():
            continue
        rng = np.random.default_rng(perm_seed)
        rng2 = np.random.default_rng(perm_seed + 100)
        if perm_kind == 'P1_ER':
            W_perm = permute_er(W_per_edge, rng)
            W_chem_perm = permute_er(W_chem_default, rng2)
        elif perm_kind == 'P2_config':
            W_perm = permute_configuration(W_per_edge, rng)
            W_chem_perm = permute_configuration(W_chem_default, rng2)
        elif perm_kind == 'P3_block':
            W_perm = permute_block_shuffle(W_per_edge, klass, rng)
            W_chem_perm = permute_block_shuffle(W_chem_default, klass, rng2)
        out = dict(d)
        out['W_chem'] = W_chem_perm
        out['W_chem_per_edge'] = W_perm
        np.savez(out_path, **out)
    print(f'  precomputed {len(PERMS)} worm permuted npz files in {PERMUTED_WORM_NPZ_CACHE}')


def build_permuted_worm_brain_factory(perm_kind: str, perm_seed: int = 7):
    """Returns a factory(seed) → SeededLIFBrain with the connectome permuted.
    Assumes precompute_worm_permutations has already run."""
    new_npz = PERMUTED_WORM_NPZ_CACHE / f'{perm_kind}_seed{perm_seed}.npz'
    if not new_npz.exists():
        raise FileNotFoundError(f'Run precompute_worm_permutations first; missing {new_npz}')

    def factory(seed):
        from brain import lif_brain
        orig = lif_brain.ARTIFACT
        lif_brain.ARTIFACT = new_npz
        try:
            class SeededLIFBrain(lif_brain.LIFBrain):
                _brian2_seed = seed
            brain = SeededLIFBrain(use_per_edge_glu_signs=True)
        finally:
            lif_brain.ARTIFACT = orig
        return brain
    return factory


# ===== Fly permuted brain factory =====

def precompute_fly_permutations(perm_seed: int = 7):
    """Build all 3 fly permuted matrices ONCE; cache to /tmp."""
    PERMUTED_FLY_MATRICES_CACHE.mkdir(parents=True, exist_ok=True)
    from state_validation import fly_larva_brain as flb
    import pandas as pd
    M = pd.read_csv(flb.WINDING_DIR / 'all-all_connectivity_matrix.csv', index_col=0)
    W_raw = M.values.astype(np.float32)
    names = [str(n) for n in M.columns]
    ann = pd.read_csv(flb.WINDING_DIR / 'annotations.csv')
    celltype_lookup: dict[str, str] = {}
    for _, row in ann.iterrows():
        for col in ('left_id', 'right_id'):
            v = row.get(col)
            if v in (None, '', 'no pair') or (isinstance(v, float) and np.isnan(v)):
                continue
            celltype_lookup[str(v).strip()] = str(row['celltype']).strip()
    celltypes = [celltype_lookup.get(n, 'unknown') for n in names]
    nt_heuristic = flb.load_nt_heuristic(flb.NT_HEURISTIC_PATH)
    signs, _ = flb.assign_signs_by_nt_heuristic(celltypes, nt_heuristic)
    W_signed = signs[:, None].astype(np.float32) * W_raw

    metadata_path = PERMUTED_FLY_MATRICES_CACHE / 'metadata.npz'
    np.savez(metadata_path,
             names=np.array(names),
             celltypes=np.array(celltypes))

    for perm_kind in PERMS:
        out_path = PERMUTED_FLY_MATRICES_CACHE / f'{perm_kind}_seed{perm_seed}.npy'
        if out_path.exists():
            continue
        rng = np.random.default_rng(perm_seed + 1000)
        if perm_kind == 'P1_ER':
            W_perm = permute_er(W_signed, rng)
        elif perm_kind == 'P2_config':
            W_perm = permute_configuration(W_signed, rng)
        elif perm_kind == 'P3_block':
            W_perm = permute_block_shuffle(W_signed, celltypes, rng)
        np.save(out_path, W_perm)
    print(f'  precomputed {len(PERMS)} fly permuted matrices in {PERMUTED_FLY_MATRICES_CACHE}')


def build_permuted_fly_brain_factory(perm_kind: str, perm_seed: int = 7):
    """Returns a factory(seed) → permuted-FlyLarvaBrain.
    Assumes precompute_fly_permutations has already run."""
    from state_validation import fly_larva_brain as flb

    # Load precomputed
    W_perm = np.load(PERMUTED_FLY_MATRICES_CACHE / f'{perm_kind}_seed{perm_seed}.npy')
    meta = np.load(PERMUTED_FLY_MATRICES_CACHE / 'metadata.npz', allow_pickle=True)
    names = list(meta['names'])
    celltypes = list(meta['celltypes'])

    # We need to construct a FlyLarvaBrain that uses W_perm directly as its W_chem.
    # The cleanest way: subclass and override the connectome-loading logic.
    # Here we monkey-patch the module's load_winding_connectome to return a fake matrix.
    # Then the brain rebuilds with W_perm but still goes through assign_signs_by_nt_heuristic
    # which would re-sign — bad. So we need to bypass sign assignment too.

    # Simpler: write a full subclass that takes the precomputed signed matrix.
    from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Network,
                        ms, mV, nS, pF, Hz, second, prefs, seed as brian2_seed)

    def factory(seed):
        prefs.codegen.target = 'cython'
        np.random.seed(seed)
        brian2_seed(seed)
        params = dict(flb.LIF_PARAMS)
        namespace = {**params, 'W_syn': flb.W_SYN_DEFAULT,
                     'C_mem': flb.C_MEM_DEFAULT,
                     'sigma': flb.NOISE_SIGMA_DEFAULT,
                     'v_rest_bias': flb.V_REST_BIAS_DEFAULT}
        eqs = """
        dv/dt = (v_rest - v + v_rest_bias)/tau + I_ext/C_mem + sigma*xi*tau**-0.5 : volt (unless refractory)
        I_ext : amp
        """
        N = W_perm.shape[0]
        neurons = NeuronGroup(N, eqs, threshold='v > v_thr', reset='v = v_reset',
                              refractory='t_ref', method='euler', namespace=namespace)
        neurons.v = flb.LIF_PARAMS['v_rest']
        neurons.I_ext = 0
        exc_pre, exc_post = np.where(W_perm > 0)
        inh_pre, inh_post = np.where(W_perm < 0)
        exc_w = W_perm[exc_pre, exc_post].astype(np.float32)
        inh_w = (-W_perm[inh_pre, inh_post]).astype(np.float32)
        syn_exc = Synapses(neurons, neurons, model='w : 1',
                            on_pre='v_post += W_syn * w',
                            namespace={'W_syn': flb.W_SYN_DEFAULT})
        if len(exc_pre):
            syn_exc.connect(i=exc_pre.tolist(), j=exc_post.tolist())
            syn_exc.w = exc_w.tolist()
        syn_inh = Synapses(neurons, neurons, model='w : 1',
                            on_pre='v_post -= W_syn * w',
                            namespace={'W_syn': flb.W_SYN_DEFAULT})
        if len(inh_pre):
            syn_inh.connect(i=inh_pre.tolist(), j=inh_post.tolist())
            syn_inh.w = inh_w.tolist()
        spikes = SpikeMonitor(neurons)
        net = Network(neurons, syn_exc, syn_inh, spikes)

        # Wrap as a brain-interface-compliant object
        class PermutedFlyBrain:
            def __init__(self):
                self.neurons = neurons
                self.syn_exc = syn_exc
                self.syn_inh = syn_inh
                self.spikes = spikes
                self.net = net
                self.names = names
                self.N = N
                self.idx = {n: i for i, n in enumerate(names)}
                self.command_neurons_idx = [
                    i for i, ct in enumerate(celltypes) if ct in flb.COMMAND_CELLTYPES
                ]
                self.celltypes = celltypes

            def run(self, duration_ms):
                self.net.run(duration_ms * ms)
        return PermutedFlyBrain()

    return factory


# ===== Worker for one (perm_kind, dose, seed) combo =====

def _worker_worm(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import (
        load_perturbation_table, run_single,
    )
    perm_kind, anest, dose, seed, alpha, sim_dur = args
    factory = build_permuted_worm_brain_factory(perm_kind, perm_seed=7)
    prof = load_perturbation_table(
        '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/data/state_validation/anesthetic_perturbation_table.csv'
    )
    metrics = run_single(anest, dose_uM=dose, seed=seed, sim_duration_s=sim_dur,
                          profile=prof[anest], mutant=None, alpha_calib=alpha,
                          brain_factory=factory)
    return {'perm': perm_kind, 'organism': 'worm', 'anesthetic': anest, 'dose_uM': dose,
            'seed': seed, 'qf': metrics['quiescent_fraction'],
            'cmd_rate': metrics['command_mean_firing_rate_hz']}


def _worker_fly(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import (
        load_perturbation_table, run_single,
    )
    from state_validation.fly_state_validator import FLY_QUIESCENT_THRESHOLD_HZ
    perm_kind, anest, dose, seed, alpha, sim_dur = args
    factory = build_permuted_fly_brain_factory(perm_kind, perm_seed=7)
    prof = load_perturbation_table(
        '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/data/state_validation_fly/fly_anesthetic_perturbation_table.csv'
    )
    metrics = run_single(anest, dose_uM=dose, seed=seed, sim_duration_s=sim_dur,
                          profile=prof[anest], mutant=None, alpha_calib=alpha,
                          brain_factory=factory,
                          quiescent_threshold_hz=FLY_QUIESCENT_THRESHOLD_HZ)
    return {'perm': perm_kind, 'organism': 'fly', 'anesthetic': anest, 'dose_uM': dose,
            'seed': seed, 'qf': metrics['quiescent_fraction'],
            'cmd_rate': metrics['command_mean_firing_rate_hz']}


# ===== main =====

WORM_ALPHA = 0.13
FLY_ALPHA = 0.060
SIM_DUR = 60.0
SEEDS = [42, 137, 219, 331, 443]
DOSES_VOLATILE = [10.0, 30.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 3000.0]
DOSES_EGER     = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]

PERMS = ['P1_ER', 'P2_config', 'P3_block']
GATE1_ANESTHETIC = 'halothane'
GATE4_COMPOUNDS = ['cis_12_dichloroethylene', 'trans_12_dichloroethylene', 'hexafluoroethane']


def _dispatch(args):
    """Module-level dispatcher — pickle-able for multiprocessing.Pool."""
    org = args[0]
    worker_args = args[1:]
    if org == 'worm':
        return _worker_worm(worker_args)
    else:
        return _worker_fly(worker_args)


def main():
    out_dir = ANESTH / 'artifacts/v5_controls'
    out_dir.mkdir(parents=True, exist_ok=True)

    # PRECOMPUTE permuted artifacts ONCE (avoids race conditions in workers)
    print('Precomputing permuted connectomes...')
    precompute_worm_permutations(perm_seed=7)
    precompute_fly_permutations(perm_seed=7)

    # Build task list — for each (organism, perm), Gate 1 dose-response + Gate 4 dose-response
    tasks = []
    for perm in PERMS:
        # Worm
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append(('worm', perm, GATE1_ANESTHETIC, d, s, WORM_ALPHA, SIM_DUR))
        for c in GATE4_COMPOUNDS:
            for d in DOSES_EGER:
                for s in SEEDS:
                    tasks.append(('worm', perm, c, d, s, WORM_ALPHA, SIM_DUR))
        # Fly
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append(('fly', perm, GATE1_ANESTHETIC, d, s, FLY_ALPHA, SIM_DUR))
        for c in GATE4_COMPOUNDS:
            for d in DOSES_EGER:
                for s in SEEDS:
                    tasks.append(('fly', perm, c, d, s, FLY_ALPHA, SIM_DUR))

    print(f'M2 connectome permutation tests — {len(tasks)} sims')
    print(f'  permutations: {PERMS}')
    print(f'  organisms: worm + fly')
    print(f'  Gate 1 (halothane WT) + Gate 4 (Eger panel) per condition')
    print()

    t_start = time.time()
    results = []

    with mp.Pool(processes=8) as pool:
        for i, m in enumerate(pool.imap_unordered(_dispatch, tasks, chunksize=1)):
            results.append(m)
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(tasks) - (i + 1))
                print(f'  [{i+1:>4d}/{len(tasks)}] {100*(i+1)/len(tasks):.0f}%  '
                      f'elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min', flush=True)

    elapsed = time.time() - t_start
    print(f'\nAll {len(tasks)} sims complete in {elapsed/60:.1f} min')

    # Write raw + aggregate
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_dir / 'M2_raw.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fieldnames})

    # Aggregate per (organism, perm, anesthetic, dose)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r['organism'], r['perm'], r['anesthetic'], r['dose_uM'])].append(r)

    summary = []
    for (org, perm, anest, dose), runs in sorted(agg.items()):
        qf = np.array([x['qf'] for x in runs])
        summary.append({'organism': org, 'perm': perm, 'anesthetic': anest, 'dose_uM': dose,
                        'qf_mean': float(qf.mean()), 'qf_sd': float(qf.std(ddof=1))})
    with open(out_dir / 'M2_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Compute Gate 1 + Gate 4 verdicts per (organism, perm)
    from state_validation.phase_g_state_validator import hill_fit_ec50
    verdict = {'frozen_alpha': {'worm': WORM_ALPHA, 'fly': FLY_ALPHA},
               'sim_duration_s': SIM_DUR, 'n_seeds': len(SEEDS), 'permutations': {}}
    for org in ('worm', 'fly'):
        verdict['permutations'][org] = {}
        for perm in PERMS:
            # Gate 1 — halothane WT
            hal_rows = sorted([r for r in summary
                              if r['organism']==org and r['perm']==perm
                              and r['anesthetic']==GATE1_ANESTHETIC],
                             key=lambda x: x['dose_uM'])
            if hal_rows:
                doses = np.array([r['dose_uM'] for r in hal_rows])
                qfs = np.array([r['qf_mean'] for r in hal_rows])
                ec50 = hill_fit_ec50(doses, qfs, threshold=0.5)
                published = 340.0
                if ec50:
                    err = max(ec50/published, published/ec50)
                    gate1_pass = err <= 2.0
                else:
                    err = None
                    gate1_pass = False
            else:
                ec50 = err = None
                gate1_pass = False

            # Gate 4 — Eger
            eger_results = []
            for c in GATE4_COMPOUNDS:
                comp_rows = sorted([r for r in summary
                                   if r['organism']==org and r['perm']==perm and r['anesthetic']==c],
                                   key=lambda x: x['dose_uM'])
                if not comp_rows:
                    continue
                max_qf = max(r['qf_mean'] for r in comp_rows)
                expected = 'ANESTHETIC' if c == 'cis_12_dichloroethylene' else 'NON_IMMOBILIZER'
                correct = max_qf >= 0.5 if expected == 'ANESTHETIC' else max_qf < 0.5
                eger_results.append({'compound': c, 'max_qf': max_qf,
                                     'expected': expected, 'correct': correct})
            n_eger_correct = sum(1 for r in eger_results if r['correct'])
            gate4_pass = n_eger_correct == len(eger_results) and len(eger_results) > 0

            verdict['permutations'][org][perm] = {
                'gate1_predicted_EC50_uM': ec50,
                'gate1_published_EC50_uM': 340.0,
                'gate1_fold_error': err,
                'gate1_PASS': bool(gate1_pass),
                'gate4_n_correct': n_eger_correct,
                'gate4_n_tested': len(eger_results),
                'gate4_PASS': bool(gate4_pass),
                'gate4_per_compound': eger_results,
            }

    with open(out_dir / 'M2_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2, default=str)

    # Print headline
    print('\n' + '=' * 78)
    print('M2 CONNECTOME PERMUTATION VERDICT — gates run at FROZEN α')
    print('=' * 78)
    print(f'{"organism":>10s}  {"perm":>10s}  {"Gate 1 EC50":>15s}  {"fold_err":>8s}  {"G1":>3s}  {"G4 (3/3 expected)":>18s}')
    for org in ('worm', 'fly'):
        for perm in PERMS:
            v = verdict['permutations'][org][perm]
            ec50_str = f'{v["gate1_predicted_EC50_uM"]:.0f} µM' if v['gate1_predicted_EC50_uM'] else 'no cross'
            err_str = f'{v["gate1_fold_error"]:.2f}×' if v['gate1_fold_error'] else '—'
            print(f'  {org:>8s}  {perm:>10s}  {ec50_str:>13s}    {err_str:>6s}   '
                  f'{"PASS" if v["gate1_PASS"] else "FAIL":>3s}   '
                  f'{v["gate4_n_correct"]}/{v["gate4_n_tested"]}  '
                  f'{"PASS" if v["gate4_PASS"] else "FAIL"}')

    # Interpretation
    print('\n=== INTERPRETATION ===')
    n_gate1_pass = sum(1 for org in ('worm','fly') for perm in PERMS
                        if verdict['permutations'][org][perm]['gate1_PASS'])
    if n_gate1_pass <= 1:
        print('  → Most permutations FAIL Gate 1 at frozen α. CONNECTOME IS LOAD-BEARING.')
    elif n_gate1_pass >= 4:
        print('  → Most permutations PASS Gate 1 at frozen α. ARCHITECTURE IS OVER-DETERMINED.')
        print('  → "Connectome-constrained" claim needs to be retracted or narrowed.')
    else:
        print('  → Mixed results. Connectome partially load-bearing; refine analysis.')


if __name__ == '__main__':
    main()
