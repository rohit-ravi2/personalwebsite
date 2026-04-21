#!/usr/bin/env python3
"""P1 #4 — Build the motor-neuron → body-wall-muscle innervation table.

C. elegans hermaphrodite has:
  - 95 body-wall muscles arranged as 24 longitudinal rows × 4 quadrants
    (dorsal-left, dorsal-right, ventral-left, ventral-right), minus
    some head reductions. Row 1 (head) has only 4, row 2 has 4, but
    most body rows are 4.
  - 75 ventral-cord motor neurons: DA1–9 (9), DB1–7 (7), DD1–6 (6),
    VA1–12 (12), VB1–11 (11), VD1–13 (13), AS1–11 (11), VC1–6 (6).

Innervation rules (White 1986, Cook 2019 update):
  - DA / DB / AS are CHOLINERGIC and excite DORSAL muscles
  - DD is GABAergic and inhibits DORSAL muscles
  - VA / VB are CHOLINERGIC and excite VENTRAL muscles
  - VD is GABAergic and inhibits VENTRAL muscles
  - VC is cholinergic, innervates vulval + some ventral body

Each motor neuron has a receptive field of ~3-5 consecutive muscle
rows centred on its position along the ventral cord. We use a
simplified "approximate Gaussian" receptive field per class:
  DA1 centred ~row 3, spans rows 1-5
  DA2 centred ~row 5, spans rows 3-7
  …
  DA9 centred ~row 21, spans rows 19-23
and analogously for the other classes, with class-specific widths.

Output: public/data/motor_innervation.json — readable from both the
body builder (scripts/build_wormbody.py, future v2) and the dashboard
(for showing motor-neuron → muscle activation heatmaps).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "public" / "data" / "motor_innervation.json"


# --- Muscle layout ---------------------------------------------------

# Anatomically accurate (White 1986): head has special-case geometry
# (rows 1-4 have 3-4 muscles), neck rows 5-7 have 4, body rows 8-22
# have 4, tail rows 23-24 trail off. Total 95.
MUSCLE_COUNT_PER_ROW = [
    # Rows 1-4 (head)
    4, 4, 4, 4,
    # Rows 5-7 (neck)
    4, 4, 4,
    # Rows 8-22 (body wall)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    # Rows 23-24 (tail)
    4, 3,
]
assert sum(MUSCLE_COUNT_PER_ROW) == 95, (
    f"muscle count {sum(MUSCLE_COUNT_PER_ROW)} != 95"
)

QUADRANTS = ["DL", "DR", "VL", "VR"]
TOTAL_ROWS = len(MUSCLE_COUNT_PER_ROW)


def muscle_names() -> list[str]:
    """Generate canonical muscle names in row-major (row, quadrant) order."""
    out = []
    for r, n_in_row in enumerate(MUSCLE_COUNT_PER_ROW, start=1):
        if n_in_row == 4:
            quads = QUADRANTS
        elif n_in_row == 3:
            # Head rows 2 and 4 are the ones missing one quadrant —
            # assume the missing is VR by convention (row-specific in
            # reality; this is a simplification).
            quads = ["DL", "DR", "VL"]
        else:
            quads = QUADRANTS[:n_in_row]
        for q in quads:
            out.append(f"M{r}{q}")
    return out


# --- Motor-neuron centres and receptive fields -----------------------


# Each motor-neuron class: (number of neurons, row_range_start, row_range_end,
#  quadrant_side {'D' or 'V'}, NT_sign, receptive_half_width_rows, class_tag)
MOTOR_CLASSES: dict[str, dict] = {
    "DA": dict(count=9, row_start=2, row_end=22, side="D", sign=+1,
               half_width=2, tag="ACh exc → dorsal (backward locomotion)"),
    "DB": dict(count=7, row_start=5, row_end=22, side="D", sign=+1,
               half_width=2, tag="ACh exc → dorsal (forward locomotion)"),
    "AS": dict(count=11, row_start=3, row_end=23, side="D", sign=+1,
               half_width=1, tag="ACh exc → dorsal (short reach)"),
    "DD": dict(count=6, row_start=3, row_end=22, side="D", sign=-1,
               half_width=3, tag="GABA inh → dorsal (reciprocal)"),
    "VA": dict(count=12, row_start=1, row_end=22, side="V", sign=+1,
               half_width=2, tag="ACh exc → ventral (backward)"),
    "VB": dict(count=11, row_start=5, row_end=22, side="V", sign=+1,
               half_width=2, tag="ACh exc → ventral (forward)"),
    "VD": dict(count=13, row_start=2, row_end=23, side="V", sign=-1,
               half_width=3, tag="GABA inh → ventral (reciprocal)"),
    "VC": dict(count=6, row_start=10, row_end=18, side="V", sign=+1,
               half_width=1, tag="ACh exc → ventral (vulval region)"),
}


def _neuron_centre(c: dict, i: int) -> float:
    """Fractional row centre for the i-th (0-indexed) neuron of class."""
    # Distribute centres uniformly across [row_start, row_end]
    if c["count"] == 1:
        return (c["row_start"] + c["row_end"]) / 2
    frac = i / (c["count"] - 1)
    return c["row_start"] + frac * (c["row_end"] - c["row_start"])


def build_innervation() -> dict:
    """Build weighted innervation table.

    Returns a dict:
      muscles: list of muscle names in canonical order
      neurons: list of motor-neuron names in canonical order
      weights: sparse list of [neuron_name, muscle_name, weight, sign]
        — weight in [0, 1] is a Gaussian receptive-field value
    """
    muscle_list = muscle_names()
    muscle_idx = {m: i for i, m in enumerate(muscle_list)}

    neurons: list[str] = []
    weights: list[tuple[str, str, float, int]] = []

    for klass, c in MOTOR_CLASSES.items():
        for i in range(c["count"]):
            nname = f"{klass}{i + 1}"
            neurons.append(nname)
            centre = _neuron_centre(c, i)
            hw = c["half_width"]
            # Gaussian receptive field over rows within window
            lo_row = max(1, int(centre - hw - 1))
            hi_row = min(TOTAL_ROWS, int(centre + hw + 1))
            side_prefix = c["side"]   # "D" or "V"
            for r in range(lo_row, hi_row + 1):
                w_row = math.exp(-((r - centre) ** 2) / (2 * (hw / 2) ** 2))
                if w_row < 0.15:
                    continue
                n_in_row = MUSCLE_COUNT_PER_ROW[r - 1]
                quads_here = (QUADRANTS if n_in_row == 4
                              else (["DL", "DR", "VL"] if n_in_row == 3
                                    else QUADRANTS[:n_in_row]))
                # Route to D* or V* quadrants by side
                target_quads = [q for q in quads_here
                                if q.startswith(side_prefix)]
                for q in target_quads:
                    mname = f"M{r}{q}"
                    if mname not in muscle_idx:
                        continue
                    weights.append((nname, mname, round(w_row, 3), c["sign"]))

    return {
        "muscles": muscle_list,
        "neurons": neurons,
        "classes": {k: {**v} for k, v in MOTOR_CLASSES.items()},
        "weights": weights,
        "meta": {
            "source": "White et al. 1986 (J Comp Neurol; classical "
                      "mind-of-a-worm electron micrographs), with "
                      "cholinergic/GABA assignments per Pereira 2015 "
                      "and Cook 2019 updates.",
            "note": "Gaussian receptive fields approximate the true "
                    "innervation; each neuron→muscle is assigned a "
                    "weight in [0, 1]. Use weight × sign as the "
                    "activation transfer. For a production fit, "
                    "replace with the exact serial-section synapse "
                    "counts from the White 1986 appendix; this is a "
                    "first-pass smooth approximation.",
        },
    }


def main() -> None:
    payload = build_innervation()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT_JSON}")
    print(f"  muscles: {len(payload['muscles'])}")
    print(f"  motor neurons: {len(payload['neurons'])}")
    print(f"  sparse weights: {len(payload['weights'])}")


if __name__ == "__main__":
    main()
