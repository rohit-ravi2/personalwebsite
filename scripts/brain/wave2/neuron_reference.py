"""
NEURONReference — Phase β CP1.A.1 wrapper for Layer A comparisons.

This module provides a programmatic interface to Nicoletti 2024 NEURON cells
suitable for cross-validating Brian2 translations against the upstream NEURON
reference. It encapsulates the cwd/sys.path management required for NEURON to
find the compiled mod library, owns a single `h.Section` per instance lifetime,
and exposes voltage-clamp and current-clamp protocols that mirror what the
Brian2 harnesses produce so the Layer A comparison code reuses the same
output structures.

Design notes
------------

1. **Single section per instance.** Creating a new `h.Section` per holding
   potential pays steep section-construction overhead and risks NEURON
   leaking state between calls. We construct one section in __init__ and
   reuse it across all sweeps; `h.finitialize()` resets state between calls.

2. **Cell-name dispatch.** AVAL/AIY/RIM are built per the canonical Nicoletti
   wrapper functions (re-using their parameter vectors and surface areas
   verbatim). AVAR re-uses the existing `avar_unc103_patch.py` to insert
   `unc103.mod` (missing in upstream repo head). For CP3 we expose a
   `custom` mode that takes an explicit channel list + parameter dict.

3. **State output structure.** Returned dicts mirror Brian2 harness output:
   - voltage_clamp: per-hold dict with {hold_mV, t_ms, V_mV, I_pA, peak_I_pA,
     ss_I_pA, time_to_peak_ms}.
   - current_clamp: per-step dict with {injection_pA, t_ms, V_mV, peak_V_mV,
     plateau_V_mV, baseline_pre_mV, baseline_post_mV}.

4. **Cleanup discipline.** `cleanup()` deletes the section + IClamp/VClamp
   handles and restores cwd/sys.path. NEURON's `h` singleton state cannot
   be fully reset within a process, so the wrapper is single-cell within
   a Python session in practice — for cross-cell comparison runs in one
   session, prefer subprocess isolation OR explicit `cleanup()` between
   instances.

5. **Voltage-clamp protocol.** Mirrors `AVAL_simulation_vclamp.AVA_simulation_vc`
   structure: pre-step at -30 mV (1007.8 ms), test step at hold_mV (250 ms),
   tail step at -30 mV (242.2 ms). This is Nicoletti's canonical Fig 1 protocol.
   For CP3-style custom cells we accept a simpler 200 ms single-step protocol.

6. **Current-clamp protocol.** Mirrors AVAL_simulation_iclamp: 1023 ms baseline,
   1000 ms injection, ~500 ms recovery. We aliases simpler CP3-style
   (`settle_ms` baseline, `injection_duration_ms` step, `post_ms` recovery)
   onto this structure for the Layer A comparison API the spec mandates.

References
----------
- Nicoletti et al. PLoS ONE 2024, https://doi.org/10.1371/journal.pone.0298105
- Phase α deliverable 4 (`voltage_clamp_harness.py`) refactor flag #3
- Phase β-pre v3 results (`comparison_validation_results_v2.json`) — output
  structure this wrapper must reproduce.
"""
from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np


NICOLETTI_DIR = Path("/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024")


# ---------------------------------------------------------------------------
# Per-cell parameter vectors (read directly from upstream wrapper scripts).
# Ordering matches each wrapper's `g0` vector exactly so `gScm2` reproduces
# the upstream rescale.
# ---------------------------------------------------------------------------

CELL_PARAMS = {
    "AVAL": {
        # AVAL_simulations.py line 26: [egl19, leak, irk, nca, eleak, cm]
        "g0": [0.104385, 0.150164, 0.1, 0, -39, 0.859551],
        "gscm2_index": 3,
        "surf_cm2": 1123.84e-8,
        "channels": ["irk", "leak", "egl19", "nca"],
        "channel_param_map": {
            "egl19": ("gbar", 0),
            "leak": ("gbar", 1),
            "irk": ("gbar", 2),
            "nca": ("gbar", 3),
        },
        "leak_e_index": 4,
        "cm_index": 5,
        "eca_mV": 60.0,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
        "iclamp_dt_ms": 0.025,
        "vclamp_dt_ms": 0.01,
    },
    "AIY": {
        # AIY_simulation_iclamp.py: [leak_g, slo1iso, kqt1, egl19, slo1egl19,
        #                            nca, shl1, eleak, cm]
        # Defaults from AIY_simulations.py — read at runtime.
        "channels": ["egl19", "slo1egl19", "nca", "leak", "slo1iso", "kqt1", "shl1"],
        "surf_cm2": 65.89e-8,
        "eca_mV": 60.0,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
        "iclamp_dt_ms": 0.4,
        "vclamp_dt_ms": 0.025,
    },
    "RIM": {
        # RIM_simulation_iclamp.py: [shl1, egl2, irk, cca1, unc2, egl19, leak,
        #                            eleak, cm]
        "channels": ["shl1", "egl2", "irk", "cca1", "unc2", "egl19", "leak"],
        "surf_cm2": 103.34e-8,
        "eca_mV": 60.0,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
        "iclamp_dt_ms": 0.04,
        "vclamp_dt_ms": 0.025,
    },
}


# ---------------------------------------------------------------------------
# Environment management
# ---------------------------------------------------------------------------

@contextmanager
def _nicoletti_env():
    """chdir into Nicoletti dir + add to sys.path so compiled mods load."""
    cur_cwd = os.getcwd()
    added_path = str(NICOLETTI_DIR) not in sys.path
    try:
        os.chdir(str(NICOLETTI_DIR))
        if added_path:
            sys.path.insert(0, str(NICOLETTI_DIR))
        yield
    finally:
        os.chdir(cur_cwd)
        if added_path and str(NICOLETTI_DIR) in sys.path:
            sys.path.remove(str(NICOLETTI_DIR))


# ---------------------------------------------------------------------------
# NEURONReference
# ---------------------------------------------------------------------------

class NEURONReference:
    """Programmatic interface to Nicoletti NEURON models for Layer A comparison.

    Parameters
    ----------
    cell_name : str
        One of {"AVAL", "AIY", "RIM", "AVAR", "custom"}. "AVAR" uses the
        existing `avar_unc103_patch.py`. "custom" requires `custom_spec`.
    custom_spec : dict, optional
        Required iff cell_name == "custom". Keys:
          - "channels": list of mod-suffix strings (e.g. ["leak", "egl19", "cadiff", "caintra1"])
          - "params": dict of {(channel_suffix, param_name): value} in NEURON
            internal units (S/cm² for gbar, mV for e, etc.)
          - "surf_cm2": float surface area
          - "cm_uFcm2": float specific capacitance
          - "eca_mV", "ek_mV", "v_init_mV": float potentials
          - "ra": float (default 100)
    mods_path : str, optional
        Path to Nicoletti dir. Defaults to canonical location.

    Notes
    -----
    The NEURON `h` singleton accumulates state across instances. For
    multi-cell sweeps in one process, call `cleanup()` between cells.
    """

    def __init__(
        self,
        cell_name: str,
        custom_spec: Optional[dict] = None,
        mods_path: str = str(NICOLETTI_DIR),
    ):
        self.cell_name = cell_name
        self.custom_spec = custom_spec
        self.mods_path = Path(mods_path)
        self._soma = None
        self._h = None
        self._initialized = False
        self._cm_pF = None  # cached for capacitance reporting

        if cell_name == "custom":
            if custom_spec is None:
                raise ValueError("custom cell_name requires custom_spec")
            for required_key in ("channels", "params", "surf_cm2", "cm_uFcm2",
                                 "eca_mV", "ek_mV", "v_init_mV"):
                if required_key not in custom_spec:
                    raise ValueError(f"custom_spec missing required key: {required_key}")
        elif cell_name not in ("AVAL", "AIY", "RIM", "AVAR"):
            raise ValueError(f"unsupported cell_name: {cell_name}")

        self._build_section()

    # ------------------------------------------------------------------
    # Internal: section construction
    # ------------------------------------------------------------------

    def _build_section(self) -> None:
        with _nicoletti_env():
            from neuron import h, gui  # noqa: F401

            self._h = h

            if self.cell_name == "custom":
                self._build_custom_section()
            elif self.cell_name == "AVAL":
                self._build_aval_section()
            elif self.cell_name == "AIY":
                self._build_aiy_section()
            elif self.cell_name == "RIM":
                self._build_rim_section()
            elif self.cell_name == "AVAR":
                self._build_avar_section()

        self._initialized = True

    def _build_aval_section(self) -> None:
        from g_to_Scm2 import gScm2
        h = self._h
        spec = CELL_PARAMS["AVAL"]
        g_scaled = gScm2(spec["g0"], spec["surf_cm2"], spec["gscm2_index"])
        cm_uFcm2 = float(g_scaled[spec["cm_index"]])
        e_leak = float(g_scaled[spec["leak_e_index"]])
        surf = spec["surf_cm2"]
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_aval")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        for ch in spec["channels"]:
            soma.insert(ch)

        for seg in soma:
            seg.egl19.gbar = float(g_scaled[0])
            seg.leak.gbar = float(g_scaled[1])
            seg.irk.gbar = float(g_scaled[2])
            seg.nca.gbar = float(g_scaled[3])
            seg.leak.e = e_leak
            seg.eca = spec["eca_mV"]
            seg.ek = spec["ek_mV"]

        self._soma = soma
        self._cm_pF = cm_uFcm2 * surf * 1e6  # (μF/cm²)·cm² = μF → ×1e6 = pF

    def _build_aiy_section(self) -> None:
        # NB: upstream filename is `AIY_simulation.py` (no trailing 's'), unlike
        # AVAL_simulations.py. Inline the g0 vector to avoid the module-import
        # cost of running AIY_simulation.py top-level (which calls os.mkdir
        # and matplotlib.show — side effects we don't want during validation).
        # g0 = [leak, slo1iso, kqt1, egl19, slo1egl19, nca, shl1, eleak, cm]
        # Source: AIY_simulation.py line 25.
        aiy_g0 = [0.14, 1.0, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
        from g_to_Scm2 import gScm2
        h = self._h
        spec = CELL_PARAMS["AIY"]
        # AIY uses index 6 in gScm2 (channels rescaled, eleak + cm raw).
        g_scaled = gScm2(aiy_g0, spec["surf_cm2"], 6)
        cm_uFcm2 = float(g_scaled[8])
        e_leak = float(g_scaled[7])
        surf = spec["surf_cm2"]
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_aiy")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        for ch in spec["channels"]:
            soma.insert(ch)

        for seg in soma:
            seg.leak.gbar = float(g_scaled[0])
            seg.slo1iso.gbar = float(g_scaled[1])
            seg.kqt1.gbar = float(g_scaled[2])
            seg.egl19.gbar = float(g_scaled[3])
            seg.slo1egl19.gbar = float(g_scaled[4])
            seg.nca.gbar = float(g_scaled[5])
            seg.shl1.gbar = float(g_scaled[6])
            seg.leak.e = e_leak
            seg.eca = spec["eca_mV"]
            seg.ek = spec["ek_mV"]

        self._soma = soma
        self._cm_pF = cm_uFcm2 * surf * 1e6

    def _build_rim_section(self) -> None:
        # NB: upstream filename is `RIM_simulation.py` (no trailing 's').
        # Per RIM_simulation.py line 25-27, `g` is already in S/cm² and is
        # passed DIRECTLY to RIM_simulation_iclamp without gScm2() rescaling.
        # So we skip gScm2 here too (passing it would double-divide by surf
        # and produce wrong conductances).
        # g0 = [shl1, egl2, irk, cca1, unc2, egl19, leak, eleak, cm]
        rim_g0 = [
            0.0009048750067326097,    # shl1     (S/cm²)
            0.0001411644285181245,    # egl2     (S/cm²)
            0.0003272854640954744,    # irk      (S/cm²)
            0.0008451919806776876,    # cca1     (S/cm²)
            9.676795045480941e-05,    # unc2     (S/cm²)
            0.00032005818627638106,   # egl19    (S/cm²)
            9.676795045480941e-05,    # leak     (S/cm²)
            -50.0,                    # eleak    (mV)
            1.5,                      # cm       (μF/cm²)
        ]
        h = self._h
        spec = CELL_PARAMS["RIM"]
        g_scaled = rim_g0  # already in S/cm² — no rescaling needed
        cm_uFcm2 = float(g_scaled[8])
        e_leak = float(g_scaled[7])
        surf = spec["surf_cm2"]
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_rim")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        for ch in spec["channels"]:
            soma.insert(ch)

        for seg in soma:
            seg.shl1.gbar = float(g_scaled[0])
            seg.egl2.gbar = float(g_scaled[1])
            seg.irk.gbar = float(g_scaled[2])
            seg.cca1.gbar = float(g_scaled[3])
            seg.unc2.gbar = float(g_scaled[4])
            seg.egl19.gbar = float(g_scaled[5])
            seg.leak.gbar = float(g_scaled[6])
            seg.leak.e = e_leak
            seg.eca = spec["eca_mV"]
            seg.ek = spec["ek_mV"]

        self._soma = soma
        self._cm_pF = cm_uFcm2 * surf * 1e6

    def _build_avar_section(self) -> None:
        # Re-uses the existing avar_unc103_patch logic to construct AVAR
        # with UNC-103 inserted from AVAR's parameter vector.
        from g_to_Scm2 import gScm2
        from avar_unc103_patch import AVAR_G0, AVAR_SURF_CM2, AVAR_GSCM2_INDEX  # noqa: WPS433
        h = self._h
        g_scaled = gScm2(AVAR_G0, AVAR_SURF_CM2, AVAR_GSCM2_INDEX)
        e_leak = float(g_scaled[5])
        cm_uFcm2 = float(g_scaled[6])
        surf = AVAR_SURF_CM2
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_avar")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        for ch in ["egl19", "leak", "irk", "nca", "unc103"]:
            soma.insert(ch)

        for seg in soma:
            seg.egl19.gbar = float(g_scaled[0])
            seg.leak.gbar = float(g_scaled[1])
            seg.irk.gbar = float(g_scaled[2])
            seg.nca.gbar = float(g_scaled[3])
            seg.unc103.gbar = float(g_scaled[4])
            seg.leak.e = e_leak
            seg.eca = 60
            seg.ek = -80

        self._soma = soma
        self._cm_pF = cm_uFcm2 * surf * 1e6

    def _build_custom_section(self) -> None:
        h = self._h
        spec = self.custom_spec
        surf = spec["surf_cm2"]
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_custom")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = float(spec.get("ra", 100))
        soma.cm = spec["cm_uFcm2"]

        ca_using = {"egl19", "cca1", "unc2", "cadiff", "caintra1", "slo1egl19", "slo2egl19"}
        k_using = {"irk", "shl1", "shk1", "kqt1", "kqt3", "kvs1", "egl2", "egl36",
                   "kcnl", "exp2", "slo1iso", "slo1egl19", "slo2iso", "slo2egl19", "unc103"}
        channels = spec["channels"]
        has_ca = any(c in ca_using for c in channels)
        has_k = any(c in k_using for c in channels)

        for ch in channels:
            soma.insert(ch)

        # Some pool mods have GLOBAL params (cadiff: depth, beta) that must
        # be set on h.<param>_<suffix>, not per-segment. We try seg first,
        # fall back to h.<pname>_<ch> if AttributeError.
        for (ch, pname), value in spec["params"].items():
            try:
                seg0 = soma(0.5)
                mech = getattr(seg0, ch)
                setattr(mech, pname, value)
                # If RANGE, we set on every segment uniformly:
                for seg in soma:
                    setattr(getattr(seg, ch), pname, value)
            except AttributeError:
                # GLOBAL parameter — set via h
                global_name = f"{pname}_{ch}"
                if hasattr(h, global_name):
                    setattr(h, global_name, value)
                else:
                    raise

        for seg in soma:
            if has_ca:
                seg.eca = spec["eca_mV"]
            if has_k:
                seg.ek = spec["ek_mV"]

        self._soma = soma
        self._cm_pF = spec["cm_uFcm2"] * 1e6 / surf

    # ------------------------------------------------------------------
    # Public API: voltage clamp
    # ------------------------------------------------------------------

    def voltage_clamp(
        self,
        holding_potentials: list[float],
        duration_ms: float = 250.0,
        prestep_mV: float = -30.0,
        prestep_ms: float = 1007.8,
        tail_mV: float = -30.0,
        tail_ms: float = 242.2,
        dt_ms: Optional[float] = None,
        v_init_mV: Optional[float] = None,
        record_currents: Optional[list[str]] = None,
        capacitance_pF: float = 100.0,
    ) -> dict:
        """Run voltage-clamp protocol; return per-hold currents + features.

        Default protocol mirrors Nicoletti's Fig 1 voltage-clamp:
        prestep (1007.8 ms @ -30 mV) → step (250 ms @ hold) → tail (242.2 ms @ -30 mV).

        For simpler isolated-channel tests, call with prestep_ms=0 and tail_ms=0.

        Parameters
        ----------
        holding_potentials : list[float]
            Test-step voltages in mV.
        record_currents : list[str], optional
            Mechanism current names to record (e.g. ["ica", "ik", "i_leak", "i_nca"]).
            If None, defaults sensibly per cell. We always record total membrane
            current via segment-level summation.

        Returns
        -------
        dict with keys:
            cell : str
            protocol : dict (params used)
            holds : list of per-hold dicts {hold_mV, t_ms, V_mV, I_pA, I_total_pA,
                    peak_I_pA, ss_I_pA, time_to_peak_ms, current_components}
        """
        if not self._initialized:
            raise RuntimeError("NEURONReference not initialized")

        if dt_ms is None:
            dt_ms = self._default_dt_vclamp()
        if v_init_mV is None:
            v_init_mV = self._default_v_init()

        with _nicoletti_env():
            h = self._h
            soma = self._soma

            stim = h.VClamp(soma(0.5))
            stim.dur[0] = prestep_ms
            stim.dur[1] = duration_ms
            stim.dur[2] = tail_ms
            stim.amp[0] = prestep_mV
            stim.amp[2] = tail_mV

            # Record default current set
            if record_currents is None:
                record_currents = self._default_currents_to_record()

            t_vec = h.Vector()
            v_vec = h.Vector()
            v_vec.record(soma(0.5)._ref_v)
            t_vec.record(h._ref_t)

            current_vecs = {}
            for cur_name in record_currents:
                vec = h.Vector()
                ref_attr = f"_ref_{cur_name}"
                ref_obj = getattr(soma(0.5), ref_attr, None)
                if ref_obj is None:
                    # Try to access via mechanism
                    continue
                vec.record(ref_obj)
                current_vecs[cur_name] = vec

            simdur = prestep_ms + duration_ms + tail_ms
            surf = self._surf_cm2()

            results = []
            for v_hold in holding_potentials:
                stim.amp[1] = float(v_hold)
                h.tstop = simdur
                h.dt = dt_ms
                # F14 (run #2): h.run() (from stdrun.hoc) calls init() which
                # re-finitializes via h.v_init. We must set h.v_init explicitly
                # so the finitialize-with-our-value sticks. Otherwise default
                # h.v_init=-65 silently overrides our v_init_mV in NEURON's
                # internal init pass, causing channel gates to settle at hinf(-65)
                # instead of hinf(v_init_mV). This caused the SHL-1 systematic
                # 7.3% peak divergence found in run #2 Phase C.2.
                h.v_init = v_init_mV
                h.finitialize(v_init_mV)
                h.run()

                t_full = np.array(t_vec.to_python())
                v_full = np.array(v_vec.to_python())
                # Total current = sum of per-mechanism currents × surface (S/cm² × mV → A/cm²);
                # convert to pA via × 1e9 × surf_cm2.
                comp_arr = {}
                for cur_name, vec in current_vecs.items():
                    comp_arr[cur_name] = np.array(vec.to_python())
                if comp_arr:
                    i_total_Acm2 = np.sum(np.stack(list(comp_arr.values()), axis=0), axis=0)
                    # NEURON's `ica`, `ik`, `i_leak` etc. are in mA/cm². Convert mA→A then ×surf.
                    i_total_pA = i_total_Acm2 * 1e-3 * surf * 1e12  # mA/cm²·cm² = mA → ×1e-3 = A → ×1e12 = pA
                else:
                    i_total_pA = np.zeros_like(t_full)

                # Trim to test-step window: t in [prestep_ms, prestep_ms + duration_ms]
                step_start = prestep_ms
                step_end = prestep_ms + duration_ms
                in_step = (t_full >= step_start) & (t_full <= step_end)
                t_step = t_full[in_step] - step_start
                v_step = v_full[in_step]
                i_step = i_total_pA[in_step]

                # Peak: signed extremum of the step-window current
                if len(i_step) > 0:
                    peak_idx = int(np.argmax(np.abs(i_step)))
                    peak_I_pA = float(i_step[peak_idx])
                    time_to_peak_ms = float(t_step[peak_idx])
                else:
                    peak_I_pA = 0.0
                    time_to_peak_ms = 0.0

                # SS: mean current in last 20% of step window
                if len(i_step) > 5:
                    ss_n = max(1, int(0.2 * len(i_step)))
                    ss_I_pA = float(np.mean(i_step[-ss_n:]))
                else:
                    ss_I_pA = peak_I_pA

                step_components = {}
                if comp_arr:
                    for k, arr in comp_arr.items():
                        arr_step = arr[in_step]
                        # Convert mA/cm² to pA via ×surf×1e9
                        arr_step_pA = arr_step * 1e-3 * surf * 1e12
                        step_components[k] = {
                            "trace_pA": arr_step_pA.tolist(),
                            "ss_pA": float(np.mean(arr_step_pA[-max(1, int(0.2*len(arr_step_pA))):])) if len(arr_step_pA) else 0.0,
                        }

                results.append({
                    "hold_mV": float(v_hold),
                    "t_ms": t_step.tolist(),
                    "V_mV": v_step.tolist(),
                    "I_total_pA": i_step.tolist(),
                    "peak_I_pA": peak_I_pA,
                    "ss_I_pA": ss_I_pA,
                    "time_to_peak_ms": time_to_peak_ms,
                    "current_components": step_components,
                })

            return {
                "cell": self.cell_name,
                "protocol": {
                    "prestep_mV": prestep_mV,
                    "prestep_ms": prestep_ms,
                    "step_duration_ms": duration_ms,
                    "tail_mV": tail_mV,
                    "tail_ms": tail_ms,
                    "dt_ms": dt_ms,
                    "v_init_mV": v_init_mV,
                },
                "holds": results,
                "surf_cm2": surf,
            }

    # ------------------------------------------------------------------
    # Public API: current clamp
    # ------------------------------------------------------------------

    def current_clamp(
        self,
        injection_pa: float | list[float],
        injection_duration_ms: float = 1000.0,
        settle_ms: float = 1000.0,
        post_ms: float = 500.0,
        v_rest_mv: float = -60.0,
        dt_ms: Optional[float] = None,
    ) -> dict:
        """Run current-clamp protocol.

        Parameters
        ----------
        injection_pa : float or list of floats
            Injection amplitude(s). If scalar, runs single sweep; if list,
            runs one sweep per amplitude.
        injection_duration_ms : float
            Step duration.
        settle_ms : float
            Pre-stim baseline duration.
        post_ms : float
            Post-stim recovery duration.
        v_rest_mv : float
            Initialization potential for `h.finitialize`.
        dt_ms : float, optional
            Defaults per cell.

        Returns
        -------
        dict with keys:
            cell : str
            protocol : dict
            sweeps : list of {injection_pa, t_ms, V_mV, baseline_pre_mV,
                              peak_V_mV, plateau_V_mV, baseline_post_mV,
                              time_to_peak_ms}
        """
        if dt_ms is None:
            dt_ms = self._default_dt_iclamp()

        if isinstance(injection_pa, (int, float)):
            injection_list = [float(injection_pa)]
        else:
            injection_list = [float(x) for x in injection_pa]

        with _nicoletti_env():
            h = self._h
            soma = self._soma

            stim = h.IClamp(soma(0.5))
            stim.delay = settle_ms
            stim.dur = injection_duration_ms
            simdur = settle_ms + injection_duration_ms + post_ms

            v_vec = h.Vector()
            t_vec = h.Vector()
            v_vec.record(soma(0.5)._ref_v)
            t_vec.record(h._ref_t)

            sweeps = []
            for inj_pa in injection_list:
                # IClamp uses nA internally
                stim.amp = inj_pa * 1e-3
                h.tstop = simdur
                h.dt = dt_ms
                # F14: set h.v_init explicitly (stdrun's run() re-finitializes via h.v_init)
                h.v_init = v_rest_mv
                h.finitialize(v_rest_mv)
                h.run()

                t_full = np.array(t_vec.to_python())
                v_full = np.array(v_vec.to_python())

                # Align time so t=0 = stim onset
                t_aligned = t_full - settle_ms

                pre_mask = t_aligned < 0
                step_mask = (t_aligned >= 0) & (t_aligned < injection_duration_ms)
                post_mask = t_aligned >= injection_duration_ms

                baseline_pre = float(np.mean(v_full[pre_mask])) if pre_mask.any() else float(v_full[0])
                baseline_post = float(np.mean(v_full[post_mask][-max(1, int(0.2*post_mask.sum())):])) if post_mask.any() else float(v_full[-1])

                if step_mask.any():
                    v_step = v_full[step_mask]
                    t_step = t_aligned[step_mask]
                    # Peak: signed extremum relative to baseline
                    delta = v_step - baseline_pre
                    peak_idx = int(np.argmax(np.abs(delta)))
                    peak_V_mV = float(v_step[peak_idx])
                    time_to_peak_ms = float(t_step[peak_idx])
                    # Plateau: median of last 20% of step window
                    n_plat = max(1, int(0.2 * len(v_step)))
                    plateau_V_mV = float(np.median(v_step[-n_plat:]))
                else:
                    peak_V_mV = baseline_pre
                    time_to_peak_ms = 0.0
                    plateau_V_mV = baseline_pre

                sweeps.append({
                    "injection_pa": inj_pa,
                    "t_ms": t_aligned.tolist(),
                    "V_mV": v_full.tolist(),
                    "baseline_pre_mV": baseline_pre,
                    "peak_V_mV": peak_V_mV,
                    "plateau_V_mV": plateau_V_mV,
                    "baseline_post_mV": baseline_post,
                    "time_to_peak_ms": time_to_peak_ms,
                })

            return {
                "cell": self.cell_name,
                "protocol": {
                    "injection_duration_ms": injection_duration_ms,
                    "settle_ms": settle_ms,
                    "post_ms": post_ms,
                    "v_rest_mV": v_rest_mv,
                    "dt_ms": dt_ms,
                },
                "sweeps": sweeps,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _surf_cm2(self) -> float:
        if self.cell_name == "custom":
            return self.custom_spec["surf_cm2"]
        if self.cell_name == "AVAR":
            from avar_unc103_patch import AVAR_SURF_CM2  # noqa: WPS433
            return AVAR_SURF_CM2
        return CELL_PARAMS[self.cell_name]["surf_cm2"]

    def _default_dt_vclamp(self) -> float:
        if self.cell_name == "custom":
            return 0.025
        return CELL_PARAMS.get(self.cell_name, {}).get("vclamp_dt_ms", 0.025)

    def _default_dt_iclamp(self) -> float:
        if self.cell_name == "custom":
            return 0.025
        return CELL_PARAMS.get(self.cell_name, {}).get("iclamp_dt_ms", 0.025)

    def _default_v_init(self) -> float:
        if self.cell_name == "custom":
            return self.custom_spec.get("v_init_mV", -60.0)
        return CELL_PARAMS.get(self.cell_name, {}).get("v_init_mV", -60.0)

    def _default_currents_to_record(self) -> list[str]:
        """Cell-specific default current list for total-current summation."""
        if self.cell_name == "AVAL":
            return ["ica", "ik", "i_nca", "i_leak"]
        if self.cell_name == "AIY":
            return ["ica", "ik", "i_nca", "i_leak"]
        if self.cell_name == "RIM":
            return ["ica", "ik", "i_leak"]
        if self.cell_name == "AVAR":
            return ["ica", "ik", "i_nca", "i_leak", "i_unc103"]
        # custom: default to whatever channels are inserted
        if self.cell_name == "custom":
            channels = self.custom_spec["channels"]
            currents = []
            if any(c in channels for c in ("egl19", "cca1", "unc2")):
                currents.append("ica")
            if any(c in channels for c in ("irk", "shl1", "shk1", "kqt1", "kqt3", "kvs1",
                                            "egl2", "egl36", "kcnl", "exp2", "slo1iso",
                                            "slo1egl19", "slo2iso", "slo2egl19", "unc103")):
                currents.append("ik")
            if "leak" in channels:
                currents.append("i_leak")
            if "nca" in channels:
                currents.append("i_nca")
            return currents
        return []

    def cleanup(self) -> None:
        """Drop section reference. NEURON `h` singleton not fully resettable."""
        self._soma = None
        self._initialized = False


# ---------------------------------------------------------------------------
# Convenience: build callable matching voltage_clamp_compare's reference signature
# ---------------------------------------------------------------------------

def make_callable_reference(neuron_ref: NEURONReference,
                            **vclamp_kwargs):
    """Wrap a NEURONReference.voltage_clamp into the
    `(hold_mV, dur_ms, dt_ms) -> (t, V, I_pA)` signature expected by
    voltage_clamp_compare.

    For the new `current_domain` voltage-clamp comparison we won't need this —
    `voltage_clamp_compare_v2` consumes the dict structure directly. This
    helper exists for backward-compat with the legacy voltage_clamp_compare.
    """
    def _ref(hold_mV: float, dur_ms: float, dt_ms: float):
        result = neuron_ref.voltage_clamp(
            holding_potentials=[hold_mV],
            duration_ms=dur_ms,
            dt_ms=dt_ms,
            **vclamp_kwargs,
        )
        hold = result["holds"][0]
        t = np.array(hold["t_ms"])
        V = np.array(hold["V_mV"])
        I_pA = np.array(hold["I_total_pA"])
        return t, V, I_pA
    return _ref


# ---------------------------------------------------------------------------
# Self-test (run as: python neuron_reference.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NEURONReference self-test ===")
    print("[1] Instantiating AVAL...")
    aval = NEURONReference("AVAL")
    print(f"    cell={aval.cell_name}  surf={aval._surf_cm2():.3e} cm²  cm={aval._cm_pF:.2f} pF")

    print("[2] Voltage-clamp at 3 holds (-60, -30, 0 mV)...")
    vc = aval.voltage_clamp(
        holding_potentials=[-60.0, -30.0, 0.0],
        duration_ms=250.0,
    )
    print(f"    n_holds={len(vc['holds'])}")
    for h in vc["holds"]:
        print(f"    hold={h['hold_mV']:+6.1f} mV  peak_I={h['peak_I_pA']:+8.2f} pA  "
              f"ss_I={h['ss_I_pA']:+8.2f} pA  ttp={h['time_to_peak_ms']:6.1f} ms")

    print("[3] Current-clamp 3 sweeps (-20, 0, 20 pA)...")
    cc = aval.current_clamp(
        injection_pa=[-20.0, 0.0, 20.0],
        injection_duration_ms=1000.0,
        settle_ms=1000.0,
        post_ms=500.0,
        v_rest_mv=-60.0,
    )
    for s in cc["sweeps"]:
        print(f"    inj={s['injection_pa']:+5.1f} pA  pre={s['baseline_pre_mV']:+6.2f} mV  "
              f"peak={s['peak_V_mV']:+6.2f} mV  plat={s['plateau_V_mV']:+6.2f} mV  "
              f"post={s['baseline_post_mV']:+6.2f} mV")

    aval.cleanup()
    print("[4] Cleanup complete.")
