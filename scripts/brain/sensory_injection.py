#!/usr/bin/env python3
"""Phase 3a step 3 — curated sensory-stimulus presets for the LIF brain.

Each preset is a (neuron_list, Poisson_rate_Hz, EPSP_weight_mV) triple
encoding a canonical C. elegans sensory modality. These let downstream
code say `brain.stimulate("touch_anterior", intensity=0.8)` rather than
hard-coding neuron names per call site.

Sensory-neuron assignments from Bargmann 2006, WormAtlas, and the
standard worm sensory-circuit literature. Where a modality has
left/right sensory pairs with asymmetric responses (e.g., ASE salt-
sensing: ASEL responds to salt ON, ASER to salt OFF), both are
injected together for v1 — asymmetric responses are a v2 refinement.

Usage:
    from lif_brain import LIFBrain
    from sensory_injection import stimulate, SENSORY_PRESETS

    brain = LIFBrain()
    brain.run(200)                     # settle
    stimulate(brain, "touch_anterior", intensity=1.0)
    brain.run(500)
"""
from __future__ import annotations

from typing import Iterable


# Each preset: neurons to inject + base Poisson rate (Hz) at intensity=1.0.
# EPSP weight per spike uses the same convention as LIFBrain.inject_poisson.
SENSORY_PRESETS: dict[str, dict] = {
    # ---- Mechanosensation ----
    "touch_anterior": {
        "neurons": ("ALML", "ALMR", "AVM"),
        "base_rate_hz": 180,
        "weight_mv": 15,
        "note": "Gentle head touch — ALM/AVM (Chalfie 1981).",
    },
    "touch_posterior": {
        "neurons": ("PLML", "PLMR"),
        "base_rate_hz": 180,
        "weight_mv": 15,
        "note": "Gentle tail touch — PLM posterior mechanosensors.",
    },
    "touch_nose_harsh": {
        "neurons": ("ASHL", "ASHR", "FLPL", "FLPR", "OLQDL", "OLQDR",
                    "OLQVL", "OLQVR"),
        "base_rate_hz": 220,
        "weight_mv": 15,
        "note": "Harsh nose touch — ASH+FLP+OLQ polymodal (Kaplan 1993).",
    },

    # ---- Chemosensation / taste ----
    "osmotic_shock": {
        "neurons": ("ASHL", "ASHR"),
        "base_rate_hz": 220,
        "weight_mv": 15,
        "note": "High osmolarity avoidance — ASH (Hart 1995).",
    },
    "salt_attractant": {
        "neurons": ("ASEL", "ASER"),  # asymmetric ON/OFF in reality
        "base_rate_hz": 150,
        "weight_mv": 12,
        "note": "NaCl sensing — ASE (Bargmann 1993); ASEL=ON, ASER=OFF in v2.",
    },
    "bitter_repellent": {
        "neurons": ("ASHL", "ASHR", "ADLL", "ADLR"),
        "base_rate_hz": 200,
        "weight_mv": 14,
        "note": "Bitter/noxious chemicals — ASH+ADL polymodal.",
    },
    "food_signal": {
        "neurons": ("ADFL", "ADFR", "ASIL", "ASIR", "ASJL", "ASJR"),
        "base_rate_hz": 40,
        "weight_mv": 10,
        "note": "Food-availability tonic (low-rate) — ASI/ASJ/ADF feeding-state.",
    },

    # ---- Olfaction ----
    "odor_attractant_awc": {
        "neurons": ("AWCL", "AWCR"),
        "base_rate_hz": 140,
        "weight_mv": 12,
        "note": "Attractive volatile odors — AWC (Bargmann 1993).",
    },
    "odor_attractant_awa": {
        "neurons": ("AWAL", "AWAR"),
        "base_rate_hz": 140,
        "weight_mv": 12,
        "note": "Diacetyl-type attractants — AWA.",
    },
    "odor_repellent_awb": {
        "neurons": ("AWBL", "AWBR"),
        "base_rate_hz": 160,
        "weight_mv": 13,
        "note": "Repellent odors — AWB (Troemel 1997).",
    },

    # ---- Gas sensation ----
    "co2_high": {
        "neurons": ("BAGL", "BAGR"),
        "base_rate_hz": 180,
        "weight_mv": 14,
        "note": "Elevated CO2 — BAG (Hallem 2008).",
    },
    "o2_high": {
        "neurons": ("URXL", "URXR", "AQR", "PQR"),
        "base_rate_hz": 180,
        "weight_mv": 14,
        "note": "High O2 (aerotaxis away) — URX/AQR/PQR (Gray 2004).",
    },

    # ---- Thermosensation ----
    "thermal_warm": {
        "neurons": ("AFDL", "AFDR"),
        "base_rate_hz": 140,
        "weight_mv": 12,
        "note": "Above-Tc warm — AFD primary thermosensor (Mori 1995).",
    },

    # ---- Proprioception (used in closed-loop from body feedback) ----
    "proprioception_stretch": {
        "neurons": ("DVA", "PDA", "PDEL", "PDER"),
        "base_rate_hz": 40,
        "weight_mv": 8,
        "note": "Body-wall stretch proprioception (Li 2006).",
    },
}


def stimulate(brain, preset: str, intensity: float = 1.0,
              neurons: Iterable[str] | None = None) -> list[str]:
    """Inject a named sensory preset into a LIFBrain instance.

    Args:
        brain: a LIFBrain (from `lif_brain.py`).
        preset: key into SENSORY_PRESETS, OR "custom" if `neurons` given.
        intensity: linear scale on base_rate_hz (0 = off, 1 = canonical).
        neurons: iterable of neuron names (only used with preset="custom").

    Returns:
        List of neuron names actually injected (those present in brain).
    """
    if preset == "custom":
        if neurons is None:
            raise ValueError("custom preset requires `neurons=…`")
        targets = list(neurons)
        rate = 150.0 * intensity
        weight = 12.0
    else:
        if preset not in SENSORY_PRESETS:
            raise KeyError(
                f"Unknown preset {preset!r}. "
                f"Known: {sorted(SENSORY_PRESETS)}"
            )
        p = SENSORY_PRESETS[preset]
        targets = list(p["neurons"])
        rate = p["base_rate_hz"] * intensity
        weight = p["weight_mv"]

    injected = []
    for n in targets:
        if n in brain.idx:
            brain.inject_poisson(n, rate, weight_mv=weight)
            injected.append(n)
    return injected


def list_presets() -> None:
    """Human-readable print of all presets."""
    for k, v in SENSORY_PRESETS.items():
        print(f"  {k:<22} {len(v['neurons']):>2}n  "
              f"{v['base_rate_hz']:>3} Hz  — {v['note']}")


if __name__ == "__main__":
    print("Available sensory presets:")
    list_presets()
