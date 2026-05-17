"""
Stratified audit: how well does the substrate fit each cell's BIOLOGICAL
rest, not a uniform criterion?

For cells with published rest data, compute delta(V_substrate - V_published).
Stratify by cell role:
  - plateau / command / modulatory (expected depolarized rest, elevated Ca OK)
  - phasic / sensory (expected hyperpolarized rest, low Ca)
  - unknown (use loose default)

Reports:
  1. Per-cell V delta from published
  2. Cells over-hyperpolarized (likely mis-fit due to our channel-load + CDI
     pushing plateau cells into phasic regime)
  3. Cells correctly fit
  4. Role-specific plausibility counts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


# Published V_rest references (mV). Cell-class level. From literature.
# Format: class -> (V_published_mV, role, source)
PUBLISHED_REST = {
    # Command interneurons — plateau / bistable
    "AVA":  (-25.0, "plateau-command",   "Mellem 2008 -20 to -30; Liu 2020"),
    "AVB":  (-30.0, "plateau-command",   "Pirri/Alkema; forward command"),
    "AVD":  (-30.0, "plateau-command",   "Sun 2014; backward command"),
    "AVE":  (-30.0, "plateau-command",   "Pirri 2009; backward escape"),
    # Modulatory monoamine / peptide — bistable / bursting
    "RIM":  (-50.0, "plateau-modulatory", "Nicoletti 2024 fit; Liu 2020 RIM bistable"),
    "RIS":  (-50.0, "plateau-modulatory", "Turek 2013; sleep burster"),
    "HSN":  (-40.0, "plateau-modulatory", "egg-laying burster; Schafer lab"),
    "NSM":  (-45.0, "plateau-modulatory", "5HT release neuron"),
    "ADF":  (-50.0, "plateau-modulatory", "5HT sensory-modulatory"),
    "ADE":  (-45.0, "plateau-modulatory", "dopaminergic"),
    "CEP":  (-45.0, "plateau-modulatory", "dopaminergic"),
    "PDE":  (-45.0, "plateau-modulatory", "dopaminergic"),
    # Sensory — hyperpolarized rest
    "AIY":  (-89.0, "sensory-deep",       "Nicoletti 2024; deeply hyperpolarized"),
    "AIZ":  (-75.0, "phasic-interneuron", "interneuron"),
    "ASE":  (-65.0, "sensory-phasic",     "Goodman lab chemosensory"),
    "AWC":  (-65.0, "sensory-phasic",     "chemosensory"),
    "AFD":  (-65.0, "sensory-phasic",     "thermosensory"),
    "ASH":  (-65.0, "sensory-phasic",     "polymodal nociceptor"),
    "ASJ":  (-65.0, "sensory-phasic",     "chemosensory"),
    "AWA":  (-70.0, "sensory-phasic",     "chemosensory"),
    "AWB":  (-70.0, "sensory-phasic",     "chemosensory"),
    # Pharyngeal — sustained-tonic
    "M3":   (-70.0, "pharyngeal",         "Avery 1995"),
    "MC":   (-60.0, "pharyngeal-plateau", "Avery 1995 pacemaker"),
    "MI":   (-65.0, "pharyngeal",         "pharyngeal motor"),
    "I3":   (-65.0, "pharyngeal",         "pharyngeal interneuron"),
    "I5":   (-65.0, "pharyngeal",         "pharyngeal interneuron"),
    "I6":   (-65.0, "pharyngeal",         "pharyngeal interneuron"),
    "M5":   (-65.0, "pharyngeal",         "pharyngeal motor"),
    # Body-wall motor (active during locomotion)
    "VA":   (-50.0, "motor-active",       "active during reverse locomotion"),
    "VB":   (-50.0, "motor-active",       "active during forward locomotion"),
    "DA":   (-50.0, "motor-active",       "dorsal reverse motor"),
    "DB":   (-50.0, "motor-active",       "dorsal forward motor"),
    "VD_DD":(-65.0, "motor-GABA",         "GABA cross-inhibition"),
    "VC":   (-50.0, "motor-active",       "egg-laying motor"),
    # Other interneurons (mostly phasic)
    "AIA":  (-70.0, "phasic-interneuron", "first-layer interneuron"),
    "AIB":  (-65.0, "phasic-interneuron", "reversal interneuron"),
    "AIN":  (-70.0, "phasic-interneuron", "interneuron"),
    "AIM":  (-65.0, "phasic-interneuron", "interneuron"),
    "PVC":  (-50.0, "plateau-command",    "forward-locomotion command"),
    "RIA":  (-70.0, "phasic-interneuron", "head-movement interneuron"),
    "RIB":  (-65.0, "phasic-interneuron", "interneuron"),
    "RIC":  (-50.0, "plateau-modulatory", "OA-releasing"),
    "RIP":  (-50.0, "plateau-pharyngeal", "ring-pharynx gateway"),
    "RMD":  (-50.0, "plateau-command",    "head-bend command"),
    "RMD_DV":(-50.0,"plateau-command",    "head-bend"),
    "RMD_LR":(-50.0,"plateau-command",    "head-bend"),
}


CLASS_MAP = {
    # Map individual cell name -> CeNGEN class
    # Most are name → strip-suffix mappings handled below
}


def cell_class(name: str) -> str:
    """Map individual to its broad class (matches PUBLISHED_REST keys)."""
    if name in PUBLISHED_REST:
        return name
    # strip L/R
    if name[:-1] in PUBLISHED_REST and name[-1] in "LR":
        return name[:-1]
    # VD/DD digits
    if name.startswith("VD") and name[2:].isdigit():
        return "VD_DD"
    if name.startswith("DD") and name[2:].isdigit():
        return "VD_DD"
    # CEPDL etc → CEP
    for cls in PUBLISHED_REST:
        if name.startswith(cls):
            return cls
    # strip trailing digits
    s = name.rstrip("0123456789")
    if s in PUBLISHED_REST:
        return s
    return None


def main():
    with open("/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/layer2_validation.json") as f:
        net = json.load(f)
    rest = net["rest_snap"]

    # Per-cell delta vs published
    results = []
    for name, s in rest.items():
        cls = cell_class(name)
        if cls is None:
            continue
        pub_V, role, source = PUBLISHED_REST[cls]
        delta = s["V_mV"] - pub_V
        results.append({
            "name": name, "class": cls, "role": role,
            "V_now": s["V_mV"], "V_pub": pub_V, "delta": delta,
            "Ca_uM": s["Ca_uM"], "K_mM": s["K_mM"], "Na_mM": s["Na_mM"],
        })

    print(f"=== Stratified audit: {len(results)} cells with published rest ===\n")

    # Aggregate by role
    roles = {}
    for r in results:
        roles.setdefault(r["role"], []).append(r)

    print(f"{'role':<25s} {'n':>4s} {'V_pub':>7s} {'V_med':>7s} {'delta_med':>11s} {'|delta|>10':>11s}")
    print("-" * 80)
    for role, cells in sorted(roles.items()):
        deltas = [c["delta"] for c in cells]
        V_meds = [c["V_now"] for c in cells]
        V_pubs = [c["V_pub"] for c in cells]
        big_misfit = sum(1 for d in deltas if abs(d) > 10)
        print(f"{role:<25s} {len(cells):>4d} {np.median(V_pubs):>+7.1f} {np.median(V_meds):>+7.1f} "
              f"{np.median(deltas):>+11.1f} {big_misfit:>11d}")

    print()
    print("=== Most over-hyperpolarized cells (we forced PLATEAU into PHASIC?) ===")
    over_hyp = sorted([r for r in results if r["delta"] < -15], key=lambda r: r["delta"])
    for r in over_hyp[:25]:
        print(f"  {r['name']:8s} ({r['class']:7s} | {r['role']:22s}) "
              f"V_now={r['V_now']:+6.1f}  V_pub={r['V_pub']:+6.1f}  "
              f"delta={r['delta']:+6.1f} mV  Ca={r['Ca_uM']:.3f} μM")

    print()
    print("=== Cells close to published (within ±10 mV) ===")
    close = [r for r in results if abs(r["delta"]) <= 10]
    print(f"  {len(close)} / {len(results)} cells within ±10 mV of published rest")

    print()
    print("=== Cells over-depolarized (true Ca-runaway cases?) ===")
    over_dep = sorted([r for r in results if r["delta"] > 15], key=lambda r: -r["delta"])
    for r in over_dep[:10]:
        print(f"  {r['name']:8s} ({r['class']:7s}) "
              f"V_now={r['V_now']:+6.1f}  V_pub={r['V_pub']:+6.1f}  "
              f"delta={r['delta']:+6.1f}  Ca={r['Ca_uM']:.3f} μM")


if __name__ == "__main__":
    main()
