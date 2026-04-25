#!/usr/bin/env python3
"""Decompose AVDL's incoming drive at rest, then track which sources
change firing rate on touch.

Read-only diagnostic — no new compute. Uses existing
phase0_scenario_traces/touch_seed{42..51}.npz (full 300-neuron rasters
from the historical scenario audit; pre-voltage-fix but the voltage
fix was a no-op for LIF dynamics so rates are valid).

For each of AVDL and AVDR:
  1. Per-source incoming chemical contribution at rest (pre window
     1-5s) under DEFAULT signs (per-neuron NT) and ALT signs (per-edge).
  2. Per-source incoming gap weight (gap currents at rest are ~0 since
     v_pre ≈ v_post at baseline; weight is the coupling capacity).
  3. Top 5 excitatory + top 5 inhibitory sources at rest.
  4. For each of those top sources: rate change pre → peri touch
     (peri window 5-7s).

The mechanistic question:
  - If AVD's drop on touch comes from "inhibitors firing more on touch":
    we should see top inhibitors increasing rate.
  - If AVD's drop comes from "excitors losing rate on touch":
    we should see top excitors decreasing rate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
TRACES = ART / "phase0_scenario_traces"

# Replicate the lif_brain.py sign-override list (DEFAULT_SIGN_OVERRIDES)
PER_NEURON_OVERRIDES = {
    "ASHL", "ASHR", "ASKL", "ASKR", "ASEL", "ASER",
    "AWCL", "AWCR", "AWAL", "AWAR", "ADLL", "ADLR",
    "AFDL", "AFDR", "ASGL", "ASGR", "AUAL", "AUAR",
    "URYDL", "URYDR", "URYVL", "URYVR",
    "AIYL", "AIYR", "AIBL", "AIBR", "RIAL", "RIAR",
}


def load_connectome():
    d = np.load(ART / "connectome.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    idx = {n: i for i, n in enumerate(names)}
    W_raw = d["W_chem_raw"].astype(np.float32)
    sign_pre = np.array(d["sign"], dtype=np.int8)
    sign_pre_eff = sign_pre.copy()
    for n in PER_NEURON_OVERRIDES:
        if n in idx:
            sign_pre_eff[idx[n]] = +1
    W_signed_default = sign_pre_eff[:, None].astype(np.float32) * W_raw  # current sim path
    W_signed_per_edge = d["W_chem_per_edge"].astype(np.float32)         # alt: turn-on path
    W_gap = d["W_gap"].astype(np.float32)
    return d, names, idx, W_raw, W_signed_default, W_signed_per_edge, W_gap


def compute_rates(npz_path: Path):
    """Returns (pre_rate[300], peri_rate[300]) in Hz, per-neuron."""
    d = np.load(npz_path)
    fsb = d["full_raster"]  # (T, N) uint8
    dt_s = 0.05  # BRAIN_SYNC_MS / 1000
    times = np.arange(fsb.shape[0]) * dt_s
    pre_mask = (times >= 1.0) & (times < 5.0)
    peri_mask = (times >= 5.0) & (times < 7.0)
    pre_rate = fsb[pre_mask].sum(axis=0) / pre_mask.sum() / dt_s
    peri_rate = fsb[peri_mask].sum(axis=0) / peri_mask.sum() / dt_s
    return pre_rate.astype(np.float32), peri_rate.astype(np.float32)


def main():
    d, names, idx, W_raw, W_default, W_per_edge, W_gap = load_connectome()
    N = len(names)

    # Aggregate firing rates across all 10 touch seeds
    seed_files = sorted(TRACES.glob("touch_seed*.npz"))
    print(f"Aggregating rates across {len(seed_files)} touch seeds…")
    pre_acc = np.zeros(N, dtype=np.float64)
    peri_acc = np.zeros(N, dtype=np.float64)
    n_files = 0
    for f in seed_files:
        pre, peri = compute_rates(f)
        pre_acc += pre
        peri_acc += peri
        n_files += 1
    pre_mean = pre_acc / n_files
    peri_mean = peri_acc / n_files
    delta = peri_mean - pre_mean
    print(f"  pre window:  1-5s    peri window: 5-7s    n={n_files} seeds\n")

    # Sanity-check known cascade neurons against the existing baseline
    print("=== Sanity check vs Phase 0 baseline report ===")
    for nm in ["ALML", "AVM", "AIBL", "AVDL", "AVAL", "AVEL", "RIML", "RIS"]:
        if nm in idx:
            i = idx[nm]
            print(f"  {nm:5s}: pre={pre_mean[i]:6.2f}  peri={peri_mean[i]:6.2f}  Δ={delta[i]:+6.2f}")
    print()

    for tgt in ("AVDL", "AVDR"):
        ti = idx[tgt]
        print("=" * 110)
        print(f"AVD target: {tgt}")
        print("=" * 110)

        # Per-source incoming chemical "drive rate" = pre_rate × signed_weight
        # Units: spike-weight per second (not voltage; just for ranking).
        contrib_default = pre_mean * W_default[:, ti]      # (N,)
        contrib_per_edge = pre_mean * W_per_edge[:, ti]    # (N,)
        gap_capacity = W_gap[:, ti]                        # (N,)

        # Total incoming (default path, current sim)
        net_chem_default_pre = contrib_default.sum()
        pos_chem_default_pre = contrib_default[contrib_default > 0].sum()
        neg_chem_default_pre = contrib_default[contrib_default < 0].sum()
        # Total incoming (per-edge path, alt)
        net_chem_per_edge_pre = contrib_per_edge.sum()
        pos_chem_per_edge_pre = contrib_per_edge[contrib_per_edge > 0].sum()
        neg_chem_per_edge_pre = contrib_per_edge[contrib_per_edge < 0].sum()

        print(f"\n  Total chemical drive rate at rest (pre 1-5s window):")
        print(f"    DEFAULT (per-neuron sign):   "
              f"+drive={pos_chem_default_pre:8.1f}  -drive={neg_chem_default_pre:8.1f}  "
              f"net={net_chem_default_pre:+8.1f}")
        print(f"    PER-EDGE (postsyn sign):     "
              f"+drive={pos_chem_per_edge_pre:8.1f}  -drive={neg_chem_per_edge_pre:8.1f}  "
              f"net={net_chem_per_edge_pre:+8.1f}")
        print(f"    Gap weight total: {gap_capacity.sum():8.1f}  (gap CURRENT at rest ≈ 0 because v_pre≈v_post)")

        # Top sources under DEFAULT signs
        order = np.argsort(-np.abs(contrib_default))
        print(f"\n  --- Top sources under DEFAULT signs (ranked by |drive rate at rest|) ---")
        print(f"  {'src':<6}  {'NT':<22}  {'raw':>5}  {'signed_w':>9}  "
              f"{'pre_rate':>9}  {'drive=r·w':>11}  {'peri_rate':>10}  {'Δrate':>7}  {'Δdrive':>9}")
        # Print top 5 excitatory + top 5 inhibitory
        ex_picks = [j for j in order if contrib_default[j] > 0][:5]
        in_picks = [j for j in order if contrib_default[j] < 0][:5]
        all_picks = ex_picks + in_picks
        for j in all_picks:
            tag = "EXC" if contrib_default[j] > 0 else "INH"
            sw = float(W_default[j, ti])
            rw = float(W_raw[j, ti])
            pre_r = float(pre_mean[j])
            peri_r = float(peri_mean[j])
            drive_pre = pre_r * sw
            drive_peri = peri_r * sw
            d_drive = drive_peri - drive_pre
            print(f"  {names[j]:<6}  {d['nt_primary'][j]:<22}  "
                  f"{rw:5.0f}  {sw:+9.1f}  {pre_r:9.2f}  {drive_pre:+11.1f}  "
                  f"{peri_r:10.2f}  {peri_r-pre_r:+7.2f}  {d_drive:+9.1f}  [{tag}]")

        # Δdrive contribution by source (for DEFAULT path) — what changes most on touch?
        d_drive_per_src = (peri_mean - pre_mean) * W_default[:, ti]
        print(f"\n  --- Top Δdrive contributors on touch (DEFAULT signs) ---")
        order_dd = np.argsort(-np.abs(d_drive_per_src))[:10]
        for j in order_dd:
            if abs(d_drive_per_src[j]) < 0.5: continue
            sw = float(W_default[j, ti])
            print(f"  {names[j]:<6}  Δrate={delta[j]:+6.2f}  signed_w={sw:+6.1f}  "
                  f"Δdrive={d_drive_per_src[j]:+8.1f}")

        # And under PER-EDGE
        order_pe = np.argsort(-np.abs(contrib_per_edge))
        print(f"\n  --- Top sources under PER-EDGE signs (the available alt path) ---")
        ex_picks_pe = [j for j in order_pe if contrib_per_edge[j] > 0][:5]
        in_picks_pe = [j for j in order_pe if contrib_per_edge[j] < 0][:5]
        for j in ex_picks_pe + in_picks_pe:
            tag = "EXC" if contrib_per_edge[j] > 0 else "INH"
            sw = float(W_per_edge[j, ti])
            rw = float(W_raw[j, ti])
            pre_r = float(pre_mean[j])
            peri_r = float(peri_mean[j])
            drive_pre = pre_r * sw
            print(f"  {names[j]:<6}  {d['nt_primary'][j]:<22}  raw={rw:5.0f}  "
                  f"signed_w={sw:+6.1f}  pre={pre_r:6.2f}  drive={drive_pre:+8.1f}  "
                  f"peri_r={peri_r:6.2f}  Δrate={peri_r-pre_r:+6.2f}  [{tag}]")

        d_drive_pe = (peri_mean - pre_mean) * W_per_edge[:, ti]
        net_dd = d_drive_per_src.sum()
        net_dd_pe = d_drive_pe.sum()
        print(f"\n  Net Δdrive on touch (DEFAULT signs):  {net_dd:+9.1f}")
        print(f"  Net Δdrive on touch (PER-EDGE signs): {net_dd_pe:+9.1f}")

        # The question: AVD drops on touch. Why?
        ex_drop = sum((peri_mean[j] - pre_mean[j]) * W_default[j, ti]
                      for j in range(N) if W_default[j, ti] > 0)
        in_change = sum((peri_mean[j] - pre_mean[j]) * W_default[j, ti]
                        for j in range(N) if W_default[j, ti] < 0)
        print(f"\n  Decomposition of why AVD changes on touch (DEFAULT signs):")
        print(f"    Δdrive from excitatory inputs (rate change × +w): {ex_drop:+8.1f}")
        print(f"    Δdrive from inhibitory inputs (rate change × −w): {in_change:+8.1f}")
        if ex_drop < 0 and in_change > 0:
            verdict = "AVD loses excitation AND gains inhibition (both contribute)"
        elif ex_drop < 0 and in_change <= 0:
            verdict = "AVD primarily LOSES EXCITATION (inhibitors don't fire more)"
        elif ex_drop >= 0 and in_change > 0:
            verdict = "AVD primarily GAINS INHIBITION (excitators steady)"
        else:
            verdict = "AVD should go UP, but it goes DOWN — anomaly"
        print(f"    → {verdict}")
        print()


if __name__ == "__main__":
    main()
