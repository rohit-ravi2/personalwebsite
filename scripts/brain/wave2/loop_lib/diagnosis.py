"""
Per-cell failure classification for the agentic loop.

A cell sweep result is a dict with keys: V_rest_mV, K_in_mM, Na_in_mM,
Cl_in_mM, Ca_in_uM, nan, channels, ...

This module classifies failures into actionable categories.
"""
from __future__ import annotations


# Plausibility envelope
V_MIN, V_MAX = -110.0, 50.0
K_MIN, K_MAX = 80.0, 200.0
NA_MIN, NA_MAX = 0.5, 50.0
CL_MIN, CL_MAX = 1.0, 30.0
CA_MAX_uM = 1.0  # 1 μM


def is_plausible(r: dict) -> bool:
    if r.get("nan", False):
        return False
    return (V_MIN < r["V_rest_mV"] < V_MAX
            and K_MIN < r["K_in_mM"] < K_MAX
            and NA_MIN < r["Na_in_mM"] < NA_MAX
            and CL_MIN < r["Cl_in_mM"] < CL_MAX
            and 0 < r["Ca_in_uM"] < CA_MAX_uM)


def classify_failure(r: dict) -> list[str]:
    """Return a list of failure categories for a non-plausible cell.

    Categories:
      - "nan"            : NaN in voltage or concentration trace
      - "v_depolarized"  : V_rest > V_MAX or warmer than -30 mV
      - "v_hyperpolarized": V_rest < V_MIN
      - "ca_runaway"     : Ca_in > 1 μM
      - "na_accumulation": Na_in > 50 mM
      - "k_depletion"    : K_in < 80 mM
      - "cl_imbalance"   : Cl_in < 1 or > 30 mM
    """
    cats = []
    if r.get("nan", False):
        cats.append("nan")
        return cats  # NaN supersedes other categorization
    V = r["V_rest_mV"]
    if V > -30.0:
        cats.append("v_depolarized")
    if V < V_MIN:
        cats.append("v_hyperpolarized")
    if r["Ca_in_uM"] > CA_MAX_uM:
        cats.append("ca_runaway")
    if r["Na_in_mM"] > NA_MAX:
        cats.append("na_accumulation")
    if r["K_in_mM"] < K_MIN:
        cats.append("k_depletion")
    if r["Cl_in_mM"] < CL_MIN or r["Cl_in_mM"] > CL_MAX:
        cats.append("cl_imbalance")
    if not cats:
        # plausibility envelope failure with all subfields OK — likely
        # near a threshold or has some other issue
        cats.append("borderline")
    return cats


def dominant_category(failures: dict[str, list[str]]) -> str | None:
    """Return the most common failure category across all failing cells."""
    counts = {}
    for cats in failures.values():
        for c in cats:
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def summarize_failures(failures: dict[str, list[str]]) -> dict[str, int]:
    """Return count per category."""
    counts = {}
    for cats in failures.values():
        for c in cats:
            counts[c] = counts.get(c, 0) + 1
    return counts
