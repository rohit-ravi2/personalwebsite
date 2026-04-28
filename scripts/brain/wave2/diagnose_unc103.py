"""Diagnose: is NEURON's [leak + unc103] custom section actually running UNC-103?"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuron_reference import NEURONReference

surf = 1123.84e-8
g_leak = 0.150164 * 1e-9 / surf

custom_spec = {
    "channels": ["leak", "unc103"],
    "params": {
        ("leak", "gbar"): g_leak,
        ("leak", "e"): -39.0,
        ("unc103", "gbar"): 0.1,
    },
    "surf_cm2": surf,
    "cm_uFcm2": 0.859551,
    "eca_mV": 60.0,
    "ek_mV": -80.0,
    "v_init_mV": -60.0,
}

nref = NEURONReference("custom", custom_spec=custom_spec)
soma = nref._soma
seg = soma(0.5)

print("Section setup:")
print(f"  L={soma.L}, diam={soma.diam}, cm={soma.cm}, Ra={soma.Ra}")
try:
    print(f"  ek={seg.ek}")
except Exception as e:
    print(f"  ek lookup failed: {e}")
try:
    print(f"  eca={seg.eca}")
except Exception as e:
    print(f"  eca not present (expected — UNC-103 doesn't use Ca): {e}")
print(f"  Mechanisms inserted:")
for mech_name in ["leak", "unc103"]:
    has_mech = hasattr(seg, mech_name)
    print(f"    {mech_name}: present={has_mech}")
    if has_mech:
        mech = getattr(seg, mech_name)
        print(f"      gbar={getattr(mech, 'gbar', 'N/A')}")
        if mech_name == "leak":
            print(f"      e={getattr(mech, 'e', 'N/A')}")

# Run a single VC at -10 mV to see what happens
print("\nVC at -10 mV (test step 250 ms with prestep_ms=50, tail_ms=0):")
result = nref.voltage_clamp(
    holding_potentials=[-10.0],
    duration_ms=250.0,
    prestep_ms=50.0,
    prestep_mV=-60.0,
    tail_ms=0.0,
    dt_ms=0.025,
)
hold = result["holds"][0]
import numpy as np
i = np.array(hold["I_total_pA"])
t = np.array(hold["t_ms"])
print(f"  peak_I_pA={hold['peak_I_pA']:.2f}")
print(f"  ss_I_pA={hold['ss_I_pA']:.2f}")
print(f"  I trajectory: t={t[0]:.2f}→{t[-1]:.2f} ms, I_min={i.min():.2f}, I_max={i.max():.2f}")
print(f"  current_components keys: {list(hold['current_components'].keys())}")
for k, v in hold["current_components"].items():
    print(f"    {k}: ss_pA={v['ss_pA']:.2f}")

nref.cleanup()
