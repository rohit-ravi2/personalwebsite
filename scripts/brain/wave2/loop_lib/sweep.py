"""
Subset + full sweep functions for the agentic loop.
"""
from __future__ import annotations

import sys
import json
import importlib
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WAVE2_DIR = THIS_DIR.parent
sys.path.insert(0, str(WAVE2_DIR))

import numpy as np


def _reimport_pipeline():
    """Force-reimport the build pipeline after edits to parameter files."""
    mods_to_reload = [m for m in list(sys.modules)
                      if m.startswith("path2_scale") or m == "layer1_cells"
                      or m.startswith("channels")]
    for m in mods_to_reload:
        del sys.modules[m]
    # Now re-import
    import path2_scale.scalable_builder as sb
    return sb


def sweep_cells(cell_names: list[str], sim_ms: float = 1500.0,
                tag: str = "") -> dict[str, dict]:
    """Run rest sim on each cell; return dict mapping cell -> result.

    Forces reimport before running so parameter file edits take effect.
    """
    sb = _reimport_pipeline()
    from layer1_cells import build_layer1_cell
    from brian2 import ms

    results = {}
    for cls in cell_names:
        try:
            spec_s = sb.build_scalable_spec(cls)
            spec_l = sb.to_layer1_cellspec(spec_s)
            if not spec_l.channels:
                results[cls] = {"status": "no_channels"}
                continue
            bundle = build_layer1_cell(spec_l)
            bundle["network"].run(sim_ms * ms)
            mon = bundle["monitor"]
            V_traj = np.asarray(mon.v[0] / 1e-3)
            ions = {ion: float(getattr(mon, ion)[0][-1])
                    for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
            nan = bool(np.isnan(V_traj).any()
                       or any(np.isnan(np.asarray(getattr(mon, ion)[0])).any()
                              for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")))
            results[cls] = {
                "status": "ok",
                "V_rest_mV": float(V_traj[-1]),
                "V_min_mV": float(np.nanmin(V_traj)),
                "V_max_mV": float(np.nanmax(V_traj)),
                "K_in_mM": ions["K_in"],
                "Na_in_mM": ions["Na_in"],
                "Cl_in_mM": ions["Cl_in"],
                "Ca_in_uM": ions["Ca_in"] * 1e3,
                "nan": nan,
                "channels": list(spec_s.channels.keys()),
                "n_channels": len(spec_s.channels),
                "pump_scale": getattr(spec_l, "pump_NaK_scale", 1.0),
            }
        except Exception as e:
            results[cls] = {"status": "exception",
                            "error": f"{type(e).__name__}: {e}",
                            "tb": traceback.format_exc()[:500]}
    return results


def all_cengen_cells() -> list[str]:
    from path2_scale.cengen_tpm_data import CENGEN_NEURONS
    return list(CENGEN_NEURONS)
