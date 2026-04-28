"""
Diagnose AIY VC divergence — per-channel current comparison Brian2 vs NEURON
at a hold where divergence is largest (+0 mV, +14% ss).

Asks: which channel(s) contribute the systematic outward excess in Brian2?
"""
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
    AIY_SURF_CM2, AIY_CM_UFCM2, AIY_E_LEAK_MV, AIY_ECA_MV, AIY_EK_MV,
    AIY_G_SCM2,
)


def neuron_aiy_per_channel_at_hold(hold_mV: float, dur_ms: float = 200.0,
                                    prestep_ms: float = 50.0, dt_ms: float = 0.025):
    """Voltage-clamp NEURON AIY, return per-channel current traces."""
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

        soma = h.Section(name="soma_aiy_diag")
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

        # Record per-mechanism currents (each channel writes to its own ik/ica/i_<name>)
        # NEURON's `ica` is the SUM over all Ca-using mechanisms; same for `ik`.
        # To get per-channel breakdown, we record each mechanism's individual `curr` or
        # via the segment-level `i_<suffix>` references where available.
        # For our channels:
        #   egl19 writes to ica
        #   slo1egl19 writes to ik (we need to use _ref_ik_slo1egl19 or similar — let's
        #     check if segment-level references exist for individual mechanism currents)
        # Standard NEURON: for a mechanism with USEION k WRITE ik, the ik is summed
        # across mechanisms but each mechanism may also have a RANGE `g` or `curr`
        # that exposes its own contribution.
        rec = {}
        rec['t'] = h.Vector(); rec['t'].record(h._ref_t)
        rec['v'] = h.Vector(); rec['v'].record(soma(0.5)._ref_v)
        rec['ica_total'] = h.Vector(); rec['ica_total'].record(soma(0.5)._ref_ica)
        rec['ik_total'] = h.Vector(); rec['ik_total'].record(soma(0.5)._ref_ik)
        rec['i_leak'] = h.Vector(); rec['i_leak'].record(soma(0.5)._ref_i_leak)
        rec['i_nca'] = h.Vector(); rec['i_nca'].record(soma(0.5)._ref_i_nca)
        # Per-channel curr fields (kqt1.mod has `curr`)
        try:
            rec['kqt1_curr'] = h.Vector(); rec['kqt1_curr'].record(soma(0.5).kqt1._ref_curr)
        except AttributeError:
            print("  WARN: kqt1.curr not exposed")
        # SHL-1, SHK-1, SLO-1 don't have explicit curr in their mod files,
        # but we can compute from gbar*m*h*(v-ek) post-hoc using state recordings:
        try:
            rec['m_shl1'] = h.Vector(); rec['m_shl1'].record(soma(0.5).shl1._ref_m)
            rec['hf_shl1'] = h.Vector(); rec['hf_shl1'].record(soma(0.5).shl1._ref_hf)
            rec['hs_shl1'] = h.Vector(); rec['hs_shl1'].record(soma(0.5).shl1._ref_hs)
        except AttributeError as e:
            print(f"  WARN: shl1 state: {e}")
        try:
            rec['m_kqt1'] = h.Vector(); rec['m_kqt1'].record(soma(0.5).kqt1._ref_m)
            rec['s_kqt1'] = h.Vector(); rec['s_kqt1'].record(soma(0.5).kqt1._ref_s)
        except AttributeError as e:
            print(f"  WARN: kqt1 state: {e}")
        try:
            rec['m_egl19'] = h.Vector(); rec['m_egl19'].record(soma(0.5).egl19._ref_m)
            rec['h_egl19'] = h.Vector(); rec['h_egl19'].record(soma(0.5).egl19._ref_h)
        except AttributeError as e:
            print(f"  WARN: egl19 state: {e}")
        try:
            rec['m_slo1iso'] = h.Vector(); rec['m_slo1iso'].record(soma(0.5).slo1iso._ref_m)
        except AttributeError as e:
            print(f"  WARN: slo1iso state: {e}")
        try:
            rec['m_slo1egl19'] = h.Vector(); rec['m_slo1egl19'].record(soma(0.5).slo1egl19._ref_m)
        except AttributeError as e:
            print(f"  WARN: slo1egl19 state: {e}")

        h.tstop = prestep_ms + dur_ms
        h.dt = dt_ms
        h.v_init = -60
        h.finitialize(-60)
        h.run()

        out = {k: np.array(v) for k, v in rec.items()}
        del soma
        del stim

    # Trim to step window
    t = out['t']
    in_step = (t >= prestep_ms) & (t <= prestep_ms + dur_ms)
    return {k: (v[in_step] if isinstance(v, np.ndarray) and v.ndim == 1 else v) for k, v in out.items()}, surf


def brian2_aiy_per_channel_at_hold(hold_mV: float, dur_ms: float = 200.0,
                                    prestep_ms: float = 50.0, dt_ms: float = 0.025):
    """Voltage-clamp Brian2 AIY, return per-channel current traces."""
    from brian2 import ms, mV, defaultclock

    factory = build_brian2_aiy_7channel(record_components=True)
    bundle = factory()
    defaultclock.dt = dt_ms * ms
    bundle["set_v"](-60.0)
    bundle["network"].run(prestep_ms * ms)
    bundle["set_v"](hold_mV)
    bundle["network"].run(dur_ms * ms)

    mon = bundle["monitor"]
    surf = bundle["config"]["surf_cm2"]
    t = np.asarray(mon.t) * 1e3  # ms
    in_step = (t >= prestep_ms) & (t <= prestep_ms + dur_ms)

    out = {
        't_ms': t[in_step],
        'v_mV': np.asarray(mon.v[0])[in_step] * 1e3,
        'i_leak_mAcm2': np.asarray(mon.i_leak_mAcm2[0])[in_step],
        'ik_nca_mAcm2': np.asarray(mon.ik_nca_mAcm2[0])[in_step],
        'ica_egl19_mAcm2': np.asarray(mon.ica_egl19_mAcm2[0])[in_step],
        'ik_slo1iso_mAcm2': np.asarray(mon.ik_slo1iso_mAcm2[0])[in_step],
        'ik_slo1egl19_mAcm2': np.asarray(mon.ik_slo1egl19_mAcm2[0])[in_step],
        'ik_kqt1_mAcm2': np.asarray(mon.ik_kqt1_mAcm2[0])[in_step],
        'ik_shl1_mAcm2': np.asarray(mon.ik_shl1_mAcm2[0])[in_step],
    }
    return out, surf


def compare_at_hold(hold_mV: float):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: Per-channel currents at hold={hold_mV:+.1f} mV")
    print('='*70)

    print("\n[NEURON]")
    nrn, nrn_surf = neuron_aiy_per_channel_at_hold(hold_mV)

    print("\n[Brian2]")
    b2, b2_surf = brian2_aiy_per_channel_at_hold(hold_mV)

    # Compute per-channel SS values in pA at end of step
    # NEURON gives mA/cm² per mech; B2 gives mA/cm² per channel
    # Convert mA/cm² → pA via ×surf×1e9
    surf = nrn_surf  # = b2_surf
    factor_pA = surf * 1e9  # mA/cm² × cm² × 1e9 = mA × 1e9 = pA. (1 mA = 1e9 pA.)

    n_ss = max(1, int(20.0 / 0.025))  # last 20 ms

    def ss_pA(arr_mAcm2):
        return float(np.mean(arr_mAcm2[-n_ss:])) * factor_pA

    # NEURON per-channel SS
    nrn_ica_total_pA = ss_pA(nrn['ica_total'])
    nrn_ik_total_pA = ss_pA(nrn['ik_total'])
    nrn_i_leak_pA = ss_pA(nrn['i_leak'])
    nrn_i_nca_pA = ss_pA(nrn['i_nca'])

    # Reconstruct NEURON per-channel K from states
    # We'd need gbar values — pull from g_scaled
    from g_to_Scm2 import gScm2  # not imported here, but available via _nicoletti_env scope... use direct
    # Actually let's compute g values directly:
    g_kqt1 = AIY_G_SCM2['kqt1']
    g_shl1 = AIY_G_SCM2['shl1']
    g_slo1iso = AIY_G_SCM2['slo1iso']
    g_slo1egl19 = AIY_G_SCM2['slo1egl19']
    g_egl19 = AIY_G_SCM2['egl19']

    ek = -80.0
    eca = 60.0
    a = 0.8

    if 'm_kqt1' in nrn and 's_kqt1' in nrn:
        nrn_ik_kqt1 = g_kqt1 * np.array(nrn['m_kqt1']) * np.array(nrn['s_kqt1']) * (np.array(nrn['v']) - ek)
        nrn_ik_kqt1_pA = ss_pA(nrn_ik_kqt1)
    else:
        nrn_ik_kqt1_pA = float('nan')

    if 'm_shl1' in nrn and 'hf_shl1' in nrn:
        nrn_ik_shl1 = g_shl1 * np.array(nrn['m_shl1']) * (a * np.array(nrn['hf_shl1']) + (1-a) * np.array(nrn['hs_shl1'])) * (np.array(nrn['v']) - ek)
        nrn_ik_shl1_pA = ss_pA(nrn_ik_shl1)
    else:
        nrn_ik_shl1_pA = float('nan')

    if 'm_slo1iso' in nrn:
        nrn_ik_slo1iso = g_slo1iso * np.array(nrn['m_slo1iso']) * (np.array(nrn['v']) - ek)
        nrn_ik_slo1iso_pA = ss_pA(nrn_ik_slo1iso)
    else:
        nrn_ik_slo1iso_pA = float('nan')

    if 'm_slo1egl19' in nrn and 'h_egl19' in nrn:
        nrn_ik_slo1egl19 = g_slo1egl19 * np.array(nrn['m_slo1egl19']) * np.array(nrn['h_egl19']) * (np.array(nrn['v']) - ek)
        nrn_ik_slo1egl19_pA = ss_pA(nrn_ik_slo1egl19)
    else:
        nrn_ik_slo1egl19_pA = float('nan')

    if 'm_egl19' in nrn and 'h_egl19' in nrn:
        nrn_ica_egl19 = g_egl19 * np.array(nrn['m_egl19']) * np.array(nrn['h_egl19']) * (np.array(nrn['v']) - eca)
        nrn_ica_egl19_pA = ss_pA(nrn_ica_egl19)
    else:
        nrn_ica_egl19_pA = float('nan')

    # Brian2 per-channel SS
    b2_i_leak_pA = ss_pA(b2['i_leak_mAcm2'])
    b2_i_nca_pA = ss_pA(b2['ik_nca_mAcm2'])
    b2_ica_egl19_pA = ss_pA(b2['ica_egl19_mAcm2'])
    b2_ik_slo1iso_pA = ss_pA(b2['ik_slo1iso_mAcm2'])
    b2_ik_slo1egl19_pA = ss_pA(b2['ik_slo1egl19_mAcm2'])
    b2_ik_kqt1_pA = ss_pA(b2['ik_kqt1_mAcm2'])
    b2_ik_shl1_pA = ss_pA(b2['ik_shl1_mAcm2'])

    print(f"\n{'channel':<14s} {'NEURON SS (pA)':<18s} {'Brian2 SS (pA)':<18s} {'Δ (pA)':<12s} {'Δ%':<8s}")
    print("-" * 70)
    for label, n, b in [
        ("leak",       nrn_i_leak_pA,        b2_i_leak_pA),
        ("nca",        nrn_i_nca_pA,         b2_i_nca_pA),
        ("ica_egl19",  nrn_ica_egl19_pA,     b2_ica_egl19_pA),
        ("ik_slo1iso", nrn_ik_slo1iso_pA,    b2_ik_slo1iso_pA),
        ("ik_slo1egl19", nrn_ik_slo1egl19_pA, b2_ik_slo1egl19_pA),
        ("ik_kqt1",    nrn_ik_kqt1_pA,       b2_ik_kqt1_pA),
        ("ik_shl1",    nrn_ik_shl1_pA,       b2_ik_shl1_pA),
    ]:
        d = b - n
        pct = (d / abs(n) * 100) if abs(n) > 1e-6 else float('nan')
        print(f"{label:<14s} {n:+12.3f}      {b:+12.3f}      {d:+8.3f}    {pct:+6.1f}%")

    print("-" * 70)
    nrn_total = nrn_i_leak_pA + nrn_i_nca_pA + nrn_ica_total_pA + nrn_ik_total_pA
    b2_total = b2_i_leak_pA + b2_i_nca_pA + b2_ica_egl19_pA + b2_ik_slo1iso_pA + b2_ik_slo1egl19_pA + b2_ik_kqt1_pA + b2_ik_shl1_pA
    print(f"{'NRN_ica_tot':<14s} {nrn_ica_total_pA:+12.3f}")
    print(f"{'NRN_ik_tot':<14s} {nrn_ik_total_pA:+12.3f}")
    print(f"{'TOTAL':<14s} {nrn_total:+12.3f}      {b2_total:+12.3f}      {b2_total - nrn_total:+8.3f}")


if __name__ == "__main__":
    # Probe holds: low (passive), threshold of K activation, mid-range (max divergence), high
    for hold in [-60.0, -30.0, 0.0, +20.0, +40.0]:
        compare_at_hold(hold)
