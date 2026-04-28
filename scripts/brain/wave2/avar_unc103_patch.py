"""
AVAR UNC-103 patch — standalone wave2 patch around Nicoletti's upstream AVAL iclamp.

Upstream defect
---------------
The Nicoletti 2024 GitHub repo (`github.com/ModelDBRepository/2017403`,
ModelDB 2017403) ships `AVAR_simulation.py` which imports
`AVAR_simulation_iclamp` from a module of the same name, but
`AVAR_simulation_iclamp.py` is NOT present in the repo head tree. AVAL has
both a wrapper `AVAL_simulations.py` and an iclamp impl `AVAL_simulation_iclamp.py`;
AVAR has only the wrapper. Running `AVAR_simulation.py` therefore raises
ModuleNotFoundError on the iclamp import, blocking end-to-end AVAR validation.

What this patch does
--------------------
Provides `AVAR_simulation_iclamp(g_AVAR_scaled_with_unc103, s1, s2, ns)` —
a function with the same call signature as `AVAL_simulation_iclamp.AVA_simulation_iclamp`
but which inserts the upstream `unc103` mod (already shipped as `unc103.mod` and
already compiled into the local x86_64 mech library) with the gbar from the AVAR
parameter vector.

The patch is a *standalone module* under our wave2/ directory. It does NOT
modify Nicoletti's upstream code in place — we only `import` from upstream.

UNC-103 conductance value
-------------------------
From `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAR_simulation.py` line 28:

    # CONDUCTANCES: EGL19, LEAK, IRK, NCA, UNC103, ELEAK, CM
    g0=[0.0643372, 0.225225, 0.042079, 0.0493356, 0.0481669, -37, 0.751761]
    gbest=gScm2(g0, surf, 4)

surf = 1121.79e-8 (cm²); index 4 → indices 0-4 (i.e. EGL19, LEAK, IRK, NCA, UNC103)
are rescaled from nS-units to S/cm² via `gScm2` (g[i]*1e-9 / surf), while
indices 5-6 (ELEAK, CM) are passed through unchanged.

So UNC-103 enters the model with gbar (in S/cm²) = 0.0481669 * 1e-9 / 1121.79e-8.

Why this is faithful to upstream behavior
-----------------------------------------
- AVAR_simulation.py imports `AVA_simulation_iclamp` (which it expects to insert
  unc103 — see the channel list comment line 27). The missing-file workaround
  in v2 reused AVAL's iclamp without unc103 because the AVAR-specific iclamp
  script is gone. This patch restores that channel insertion at the AVAL-iclamp
  boundary using AVAR's published conductance.
- The `unc103.mod` file IS shipped in the repo and IS compiled into the local
  mechanism library (`x86_64/libnrnmech.so`). The only missing piece is the
  Python glue.

Confidence
----------
- Channel set is verified (matches the comment in AVAR_simulation.py line 27).
- Conductance value is read directly from line 28.
- Surface area + ELEAK + CM are all from the same g0 vector — using AVAR's,
  not AVAL's.
- The IClamp protocol (delay=1023 ms, dur=1000 ms, simdur=2500 ms) and time
  alignment (subtract 1000 ms baseline) match AVAL_simulation_iclamp exactly,
  consistent with how AVAL_simulation.py invokes it.

Acceptance test
---------------
Resting potential (V at t=0 of figure axis, just before stim onset) should
land at -25 mV ± 5 mV — close to the experimental anchor in Fig 1B AVAR
panel, which is the v2 missing-UNC103 run's +11 mV bias corrected.

Author: Phase β-pre v3 engineering session, 2026-04-26.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


NICOLETTI_DIR = Path("/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024")


# AVAR parameters from upstream AVAR_simulation.py lines 24-28.
# Order: [EGL19, LEAK, IRK, NCA, UNC103, ELEAK, CM]
AVAR_G0 = [0.0643372, 0.225225, 0.042079, 0.0493356, 0.0481669, -37, 0.751761]
AVAR_SURF_CM2 = 1121.79e-8  # cm²
AVAR_GSCM2_INDEX = 4  # indices 0..4 rescaled, 5..6 passed through


def _activate_nicoletti_env():
    """chdir into Nicoletti dir and add to sys.path so compiled mods load."""
    os.chdir(str(NICOLETTI_DIR))
    if str(NICOLETTI_DIR) not in sys.path:
        sys.path.insert(0, str(NICOLETTI_DIR))


def _restore_env(cur_cwd):
    os.chdir(cur_cwd)
    if str(NICOLETTI_DIR) in sys.path:
        sys.path.remove(str(NICOLETTI_DIR))


def AVAR_simulation_iclamp_patched(s1: float, s2: float, ns: int) -> tuple:
    """Drop-in replacement for the missing AVAR_simulation_iclamp.AVA_simulation_iclamp.

    Mirrors AVAL_simulation_iclamp.py (Nicoletti upstream) verbatim except:
      - surface area = AVAR's 1121.79e-8 cm² (not AVAL's 1123.84e-8)
      - inserts `unc103` channel mod with AVAR's gbar
      - parameter vector is AVAR's g0, post-`gScm2`-rescale at index 4

    Returns (v_normalized, time_aligned, iv_peak, iv) matching upstream API.
    Caller is responsible for cwd / sys.path environment.
    """
    from neuron import h, gui  # noqa: F401  (gui needed even if not used)
    from g_to_Scm2 import gScm2

    g_scaled = gScm2(AVAR_G0, AVAR_SURF_CM2, AVAR_GSCM2_INDEX)
    # g_scaled now has units S/cm² for indices 0..4, raw for 5..6.
    # Layout (matches AVAR_G0):
    g_egl19 = g_scaled[0]
    g_leak = g_scaled[1]
    g_irk = g_scaled[2]
    g_nca = g_scaled[3]
    g_unc103 = g_scaled[4]
    e_leak = g_scaled[5]
    cm_uFcm2 = g_scaled[6]

    surf = AVAR_SURF_CM2
    L = math.sqrt(surf / math.pi)
    rsoma = L * 1e4  # microns

    soma = h.Section(name="soma_avar")
    soma.L = rsoma
    soma.diam = rsoma
    soma.Ra = 100
    soma.cm = cm_uFcm2

    # Channel set per AVAR_simulation.py line 27 comment:
    # EGL19, LEAK, IRK, NCA, UNC103
    soma.insert("egl19")
    soma.insert("leak")
    soma.insert("irk")
    soma.insert("nca")
    soma.insert("unc103")

    for seg in soma:
        seg.egl19.gbar = g_egl19
        seg.leak.gbar = g_leak
        seg.irk.gbar = g_irk
        seg.nca.gbar = g_nca
        seg.unc103.gbar = g_unc103
        seg.leak.e = e_leak
        seg.eca = 60
        seg.ek = -80

    stim = h.IClamp(soma(0.5))
    stim.delay = 1023  # ms — matches AVAL_simulation_iclamp.py line 53
    stim.amp = 0.0  # overwritten in loop
    stim.dur = 1000

    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    simdur = 2500
    ref_v: list = []
    ref_t: list = []

    for i in np.linspace(start=s1, stop=s2, num=ns):
        stim.amp = i
        h.tstop = simdur
        h.dt = 0.025
        h.finitialize(-60)
        h.run()

        ref_t_vec = np.zeros_like(t_vec)
        t_vec.to_python(ref_t_vec)
        ref_t.append(ref_t_vec)

        ref_v_vec = np.zeros_like(v_vec)
        v_vec.to_python(ref_v_vec)
        ref_v.append(ref_v_vec)

    v = np.array(list(ref_v))
    time1 = np.array(ref_t)

    resc_ind = np.where(time1[1, :] >= 1000)
    resc_min = int(np.amin(resc_ind))
    resc_max = int(np.amax(resc_ind))
    v_normalized = v[:, resc_min:resc_max]
    time = time1[:, resc_min:resc_max] - 1000

    # SS V-I (averaging window 23-63 ms post-onset, matches upstream)
    ind = np.where(np.logical_and(time[0] >= 23, time[0] <= 63))
    ind_max = int(np.amax(ind))
    ind_min = int(np.amin(ind))
    iv = np.mean(v_normalized[:, ind_min:ind_max], axis=1)

    # PEAKS (window 953-1023 ms post-onset, matches upstream)
    ind2 = np.where(np.logical_and(time[0] >= 953, time[0] <= 1023))
    ind2_max = int(np.amax(ind2))
    ind2_min = int(np.amin(ind2))
    iv_peak: list = []
    for j in range(ns):
        if j <= 3:
            peak = np.amin(v_normalized[j, ind2_min:ind2_max])
        else:
            peak = np.amax(v_normalized[j, ind2_min:ind2_max])
        iv_peak.append(peak)

    return v_normalized, time, iv_peak, iv


def run_AVAR_iclamp_patched() -> dict:
    """Public entry point — runs AVAR iclamp with UNC-103 inserted, returns
    the same dict shape as v2's run_AVAR_iclamp() so it slots into
    layer_b_validation cleanly.
    """
    cur_cwd = os.getcwd()
    try:
        _activate_nicoletti_env()
        v_norm, time_norm, iv_peak, _iv = AVAR_simulation_iclamp_patched(-0.03, 0.03, 7)
        time_aligned = time_norm - 23.0  # same offset convention as v2's AVAR
        steps_pa = np.linspace(-30.0, 30.0, 7).tolist()
        return {
            "cell": "AVAR",
            "n_steps": 7,
            "current_steps_pA": steps_pa,
            "v_traces_mV": [list(v) for v in v_norm],
            "time_ms": [list(t) for t in time_aligned],
            "stim_onset_ms": 0.0,
            "stim_offset_ms": 1000.0,
            "patch_applied": "avar_unc103_patch_v3",
            "patch_provenance": (
                "UNC-103 inserted with gbar from AVAR_simulation.py line 28 "
                f"(raw 0.0481669 nS, rescaled via gScm2 at index 4 with "
                f"surf=1121.79e-8 cm²). Channel set [EGL19, LEAK, IRK, NCA, UNC103] "
                "matches AVAR_simulation.py line 27 comment."
            ),
        }
    finally:
        _restore_env(cur_cwd)


if __name__ == "__main__":
    # Smoke test: run patch and report resting potential per step.
    result = run_AVAR_iclamp_patched()
    print(f"AVAR patched run: {result['n_steps']} steps")
    for i, step_pa in enumerate(result["current_steps_pA"]):
        v = np.asarray(result["v_traces_mV"][i])
        t = np.asarray(result["time_ms"][i])
        # Rest = mean V before stim onset (t < 0)
        pre = v[t < 0]
        rest = float(np.mean(pre)) if len(pre) > 0 else float(v[0])
        # Plateau = mean V in last 30% of stim window (700-1000 ms)
        in_plat = (t >= 700.0) & (t <= 1000.0)
        plat = float(np.mean(v[in_plat])) if in_plat.any() else None
        print(f"  step {step_pa:+5.0f} pA: rest = {rest:6.2f} mV, "
              f"plateau = {plat:6.2f} mV" if plat is not None
              else f"  step {step_pa:+5.0f} pA: rest = {rest:6.2f} mV, plateau = N/A")
