#!/usr/bin/env python3
"""Export C. elegans hermaphrodite class-level connectome to JSON for the site.

Source: Cook et al. 2019, SI 7 "Cell class connectome adjacency matrices,
corrected July 2020". Classes are neuron types (e.g. ASI, AVAL), not individual
left/right instances, which keeps the graph visually tractable (~100 nodes)
while remaining the canonical published substrate.

Output: public/data/connectome.json with {nodes, edges, meta}.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

COOK_XL = Path(
    "/home/rohit/Desktop/C-Elegans/data/connectome/cook2019/"
    "SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx"
)
OUT = Path(__file__).resolve().parents[1] / "public" / "data" / "connectome.json"

# Cook 2019 group codes → coarse 3-class role used by the project's classifier.
CATEGORY_TO_ROLE = {
    "SN1": "sensory", "SN2": "sensory", "SN3": "sensory",
    "SN4": "sensory", "SN5": "sensory", "SN6": "sensory",
    "IN1": "inter", "IN2": "inter", "IN3": "inter", "IN4": "inter",
    "SMN": "motor", "HMN": "motor",
}


def _forward_fill(seq):
    out = []
    cur = None
    for x in seq:
        if pd.notna(x) and isinstance(x, str) and x.strip():
            cur = x.strip()
        out.append(cur)
    return out


def _load_matrix(sheet: str) -> tuple[list[str], list[str], list[str | None], list[str | None], pd.DataFrame]:
    df = pd.read_excel(COOK_XL, sheet_name=sheet, header=None)
    # Row 0: category bands across columns; Row 1: column neuron class names.
    # Col 0: category bands down rows; Col 1: row neuron class names.
    col_cats = _forward_fill(df.iloc[0, 2:].tolist())
    row_cats = _forward_fill(df.iloc[2:, 0].tolist())
    col_neurons = [str(x).strip() for x in df.iloc[1, 2:].tolist()]
    row_neurons = [str(x).strip() for x in df.iloc[2:, 1].tolist()]
    # Matrix values live at [2:, 2:]
    mat = df.iloc[2:, 2:].reset_index(drop=True)
    mat.columns = range(mat.shape[1])
    return col_neurons, row_neurons, col_cats, row_cats, mat


def build():
    chem_cols, chem_rows, chem_col_cats, chem_row_cats, chem_mat = _load_matrix("herm chem grouped")
    gap_cols, gap_rows, gap_col_cats, gap_row_cats, gap_mat = _load_matrix("herm gap jn grouped symmetric")

    # Union of all neuron classes; attach role via whichever category info exists.
    # (chemical data has 106 targets × 95 sources; gap has a slightly different shape.)
    node_cat = {}
    for name, cat in zip(chem_cols, chem_col_cats):
        if name and name not in node_cat:
            node_cat[name] = cat
    for name, cat in zip(chem_rows, chem_row_cats):
        if name and name not in node_cat:
            node_cat[name] = cat
    for name, cat in zip(gap_cols, gap_col_cats):
        if name and name not in node_cat:
            node_cat[name] = cat
    for name, cat in zip(gap_rows, gap_row_cats):
        if name and name not in node_cat:
            node_cat[name] = cat

    nodes = []
    for name, cat in sorted(node_cat.items()):
        role = CATEGORY_TO_ROLE.get(cat, "other")
        nodes.append({"id": name, "role": role, "category": cat})

    # Edges from chemical sheet (directed, weighted by synapse count).
    edges = []
    for ri, src in enumerate(chem_rows):
        if not src:
            continue
        for ci, tgt in enumerate(chem_cols):
            if not tgt or src == tgt:
                continue
            try:
                v = chem_mat.iat[ri, ci]
            except IndexError:
                continue
            if pd.isna(v):
                continue
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w <= 0:
                continue
            edges.append({"source": src, "target": tgt, "weight": int(w) if w == int(w) else round(w, 2), "type": "chemical"})

    # Gap junctions (symmetric). De-dupe A→B / B→A into one undirected edge.
    seen_gap = set()
    for ri, src in enumerate(gap_rows):
        if not src:
            continue
        for ci, tgt in enumerate(gap_cols):
            if not tgt or src == tgt:
                continue
            pair = tuple(sorted((src, tgt)))
            if pair in seen_gap:
                continue
            try:
                v = gap_mat.iat[ri, ci]
            except IndexError:
                continue
            if pd.isna(v):
                continue
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w <= 0:
                continue
            seen_gap.add(pair)
            edges.append({"source": pair[0], "target": pair[1], "weight": int(w) if w == int(w) else round(w, 2), "type": "gap"})

    # Drop nodes with no edges (keeps the graph clean if some classes are unconnected in this slice).
    connected = set()
    for e in edges:
        connected.add(e["source"])
        connected.add(e["target"])
    nodes = [n for n in nodes if n["id"] in connected]

    meta = {
        "source": "Cook et al. 2019 (SI 7, corrected July 2020) — hermaphrodite cell-class connectome",
        "nodes": len(nodes),
        "edges": len(edges),
        "chemical_edges": sum(1 for e in edges if e["type"] == "chemical"),
        "gap_edges": sum(1 for e in edges if e["type"] == "gap"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"nodes": nodes, "edges": edges, "meta": meta}, indent=0, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB)")
    print("meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    build()
