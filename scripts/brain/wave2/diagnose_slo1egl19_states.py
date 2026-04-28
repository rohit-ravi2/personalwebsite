"""Compare slo1egl19 internal state evolution between Brian2 and NEURON."""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import numpy as np

from neuron_reference import _nicoletti_env
from option_alpha_aiy_cell import (
    build_brian2_aiy_7channel,
    AIY_SURF_CM2, AIY_G_SCM2, AIY_ECA_MV, AIY_EK_MV,
)


def neuron_states_at_hold(hold_mV: float, dur_ms: float = 200.0,
                            prestep_ms: float = 50.0, dt_ms: float = 0.025):
    with _nicoletti_env():
        from neuron import h, gui  # noqa: F401
        from g_to_Scm2 import gScm2

        surf = AIY_SURF_CM2
        g0 = [0.14, 1.0, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
        g_scaled = gScm2(g0, surf, 6)
        cm_uFcm2 = float(g_scaled[8])
        e_leak = float(g_scaled[7])
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_aiy_diag2")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        soma.insert("egl19")
        soma.insert("slo1egl19")
        soma.insert("nca")
        soma.insert("leak")
        soma.insert("slo1iso")
        soma.insert("kqt1")
        soma.insert("shl1")

        for seg in soma:
            seg.leak.gbar = float(g_scaled[0])
            seg.slo1iso.gbar = float(g_scaled[1])
            seg.kqt1.gbar = float(g_scaled[2])
            seg.egl19.gbar = float(g_scaled[3])
            seg.slo1egl19.gbar = float(g_scaled[4])
            seg.nca.gbar = float(g_scaled[5])
            seg.shl1.gbar = float(g_scaled[6])
            seg.leak.e = e_leak
            seg.eca = AIY_ECA_MV
            seg.ek = AIY_EK_MV

        stim = h.VClamp(soma(0.5))
        stim.dur[0] = prestep_ms
        stim.dur[1] = dur_ms
        stim.dur[2] = 0
        stim.amp[0] = -60.0
        stim.amp[1] = float(hold_mV)
        stim.amp[2] = -60.0

        rec = {}
        rec['t'] = h.Vector(); rec['t'].record(h._ref_t)
        rec['v'] = h.Vector(); rec['v'].record(soma(0.5)._ref_v)
        rec['m_egl19'] = h.Vector(); rec['m_egl19'].record(soma(0.5).egl19._ref_m)
        rec['h_egl19'] = h.Vector(); rec['h_egl19'].record(soma(0.5).egl19._ref_h)
        rec['m_slo1egl19'] = h.Vector(); rec['m_slo1egl19'].record(soma(0.5).slo1egl19._ref_m)
        # slo1egl19's RANGE vars (may differ from internal computations):
        try:
            rec['mminf_slo1egl19'] = h.Vector(); rec['mminf_slo1egl19'].record(soma(0.5).slo1egl19._ref_mminf)
            rec['tslo1'] = h.Vector(); rec['tslo1'].record(soma(0.5).slo1egl19._ref_tslo1)
            rec['caCALC'] = h.Vector(); rec['caCALC'].record(soma(0.5).slo1egl19._ref_caCALC)
            rec['kcmCALC'] = h.Vector(); rec['kcmCALC'].record(soma(0.5).slo1egl19._ref_kcmCALC)
            rec['kopCALC'] = h.Vector(); rec['kopCALC'].record(soma(0.5).slo1egl19._ref_kopCALC)
            rec['komCALC'] = h.Vector(); rec['komCALC'].record(soma(0.5).slo1egl19._ref_komCALC)
            rec['mca'] = h.Vector(); rec['mca'].record(soma(0.5).slo1egl19._ref_mca)
            rec['hca'] = h.Vector(); rec['hca'].record(soma(0.5).slo1egl19._ref_hca)
            rec['alph'] = h.Vector(); rec['alph'].record(soma(0.5).slo1egl19._ref_alph)
            rec['bet'] = h.Vector(); rec['bet'].record(soma(0.5).slo1egl19._ref_bet)
            rec['taca'] = h.Vector(); rec['taca'].record(soma(0.5).slo1egl19._ref_taca)
        except AttributeError as e:
            print(f"  WARN: slo1egl19 RANGE: {e}")

        h.tstop = prestep_ms + dur_ms
        h.dt = dt_ms
        h.v_init = -60
        h.finitialize(-60)
        h.run()

        out = {k: np.array(v) for k, v in rec.items()}
        del soma
        del stim

    t = out['t']
    in_step = (t >= prestep_ms) & (t <= prestep_ms + dur_ms)
    return {k: v[in_step] for k, v in out.items()}


def brian2_states_at_hold(hold_mV: float, dur_ms: float = 200.0,
                            prestep_ms: float = 50.0, dt_ms: float = 0.025):
    from brian2 import ms, mV, defaultclock, StateMonitor

    factory = build_brian2_aiy_7channel(record_components=False)
    bundle = factory()
    G = bundle["group"]
    # Add a custom state monitor to track internals
    extra_mon = StateMonitor(G, [
        "v",
        "m_egl19", "h_egl19", "egl19_minf", "egl19_mtau",
        "m_slo1egl19", "slo1egl19_mminf", "slo1egl19_tslo1",
        "slo1egl19_caCALC",
        "slo1egl19_kcm", "slo1egl19_kom", "slo1egl19_kop",
        "slo1egl19_alpha1", "slo1egl19_beta1",
    ], record=True)
    # Add to network
    bundle["network"].add(extra_mon)

    defaultclock.dt = dt_ms * ms
    bundle["set_v"](-60.0)
    bundle["network"].run(prestep_ms * ms)
    bundle["set_v"](hold_mV)
    bundle["network"].run(dur_ms * ms)

    t = np.asarray(extra_mon.t) * 1e3  # ms
    in_step = (t >= prestep_ms) & (t <= prestep_ms + dur_ms)
    out = {
        't': t[in_step],
        'v': np.asarray(extra_mon.v[0])[in_step] * 1e3,
        'm_egl19': np.asarray(extra_mon.m_egl19[0])[in_step],
        'h_egl19': np.asarray(extra_mon.h_egl19[0])[in_step],
        'egl19_minf': np.asarray(extra_mon.egl19_minf[0])[in_step],
        'egl19_mtau': np.asarray(extra_mon.egl19_mtau[0])[in_step],
        'm_slo1egl19': np.asarray(extra_mon.m_slo1egl19[0])[in_step],
        'mminf_slo1egl19': np.asarray(extra_mon.slo1egl19_mminf[0])[in_step],
        'tslo1': np.asarray(extra_mon.slo1egl19_tslo1[0])[in_step],
        'caCALC': np.asarray(extra_mon.slo1egl19_caCALC[0])[in_step],
        'kcmCALC': np.asarray(extra_mon.slo1egl19_kcm[0])[in_step],
        'kopCALC': np.asarray(extra_mon.slo1egl19_kop[0])[in_step],
        'komCALC': np.asarray(extra_mon.slo1egl19_kom[0])[in_step],
        'alph': np.asarray(extra_mon.slo1egl19_alpha1[0])[in_step],
        'bet': np.asarray(extra_mon.slo1egl19_beta1[0])[in_step],
    }
    return out


def compare(hold_mV):
    print(f"\n{'='*70}\nState comparison at hold={hold_mV:+.1f} mV\n{'='*70}")
    nrn = neuron_states_at_hold(hold_mV)
    b2 = brian2_states_at_hold(hold_mV)

    def last(arr): return float(arr[-1])

    print(f"\n{'state':<22s} {'NEURON':<14s} {'Brian2':<14s} {'Δ':<14s}")
    print('-'*70)
    pairs = [
        ('v', 'v'),
        ('m_egl19', 'm_egl19'),
        ('h_egl19', 'h_egl19'),
        ('m_slo1egl19', 'm_slo1egl19'),
        ('mminf_slo1egl19', 'mminf_slo1egl19'),
        ('tslo1', 'tslo1'),
        ('caCALC', 'caCALC'),
        ('kcmCALC', 'kcmCALC'),
        ('kopCALC', 'kopCALC'),
        ('komCALC', 'komCALC'),
        ('alph', 'alph'),
        ('bet', 'bet'),
    ]
    for nrn_k, b2_k in pairs:
        if nrn_k in nrn and b2_k in b2:
            n = last(nrn[nrn_k])
            b = last(b2[b2_k])
            d = b - n
            pct = (d/abs(n)*100) if abs(n) > 1e-12 else float('nan')
            print(f"{nrn_k:<22s} {n:+12.6e}  {b:+12.6e}  {d:+12.6e}  ({pct:+.2f}%)")

    # Also: at -60 (initial), what's mca and hca? slo1egl19's own mca = megl19_egl19 (egl19's m).
    # They should be equal.
    if 'mca' in nrn:
        print(f"\nNEURON slo1egl19's mca (=megl19_egl19) at end: {nrn['mca'][-1]:+.6e}")
        print(f"NEURON slo1egl19's hca (=hegl19_egl19) at end: {nrn['hca'][-1]:+.6e}")
        print(f"NEURON slo1egl19's taca (= tactegl19) at end: {nrn['taca'][-1]:+.6e}")
        print(f"  vs egl19's m_egl19: {nrn['m_egl19'][-1]:+.6e} and h_egl19: {nrn['h_egl19'][-1]:+.6e}")


if __name__ == "__main__":
    for h in [+0.0, +20.0, +40.0]:
        compare(h)
