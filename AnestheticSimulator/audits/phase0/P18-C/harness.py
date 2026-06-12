#!/usr/bin/env python3
"""
P18-C Artifact provenance ledger - static import/call-graph tracer.

Proves (or refutes) that every LOAD-BEARING v7 artifact was produced by the
rank-2 run_single path, not by the higher-rank apply_to_brain writer.

Static only: stdlib ast + regex + hashlib. No brain.run(). No brian2 needed.

Gates (frozen in prereg.json):
  G0  denominator-freeze : in-scope set fixed by hash-locked grep-reachability predicate
  G1  rank2-attribution  : 100% in-scope trace to run_single by call-graph AND schema; contaminated==0
  G1b untraceable        : zero orphan in-scope CSVs
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path("/mnt/ssd4tb/Desktop/website/personalwebsite")
ASIM = REPO / "AnestheticSimulator"
SRC = ASIM / "src"
HERE = Path(__file__).resolve().parent

# ---- Frozen knobs (must match prereg.json) ----
LOAD_BEARING_DOCS = [
    ASIM / "docs" / "v7_final_summary.md",
    ASIM / "docs" / "v7_preregistration.md",
]
KNOWN_SUSPECT = "artifacts/phase_g/phase_g_halothane_dose_response.csv"
ARTIFACT_RE = re.compile(r"artifacts/[A-Za-z0-9_./-]+\.(?:csv|json)")

RANK2_TARGET = "run_single"
HIGHER_RANK_TARGETS = {"apply_to_brain", "NetworkPerturbation"}
HIGHER_RANK_MODULE_HINTS = {"phase_g_network_perturbation"}

# Higher-rank per-neuron schema fingerprint columns
HIGHER_RANK_COLS = {
    "k2p_max", "complex_i_max", "gaba_max", "snare_max",
    "nachr_max", "glucl_max", "hyperpol_pA",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# G0: denominator freeze via grep-reachability over load-bearing docs
# ---------------------------------------------------------------------------
def freeze_denominator():
    doc_hashes = {}
    in_scope = set()
    referenced_by = defaultdict(list)
    for doc in LOAD_BEARING_DOCS:
        if not doc.exists():
            raise FileNotFoundError(f"load-bearing doc missing: {doc}")
        doc_hashes[str(doc.relative_to(ASIM))] = sha256(doc)
        text = doc.read_text(errors="replace")
        for m in ARTIFACT_RE.finditer(text):
            rel = m.group(0)
            in_scope.add(rel)
            referenced_by[rel].append(str(doc.relative_to(ASIM)))
    # known suspect: recorded in candidate set, scope decided by predicate
    suspect_in_scope = KNOWN_SUSPECT in in_scope
    return {
        "doc_hashes": doc_hashes,
        "in_scope": sorted(in_scope),
        "referenced_by": {k: sorted(set(v)) for k, v in referenced_by.items()},
        "known_suspect": KNOWN_SUSPECT,
        "known_suspect_in_scope": suspect_in_scope,
    }


# ---------------------------------------------------------------------------
# Map each in-scope artifact -> writer module(s) by static grep of basename
# An artifact A is WRITTEN by module M if M contains a write op (to_csv /
# csv.writer / json.dump / open(...,'w')) AND references the artifact basename
# (without extension is sufficient since stems are unique here).
# ---------------------------------------------------------------------------
def build_writer_index():
    py_files = sorted(SRC.rglob("*.py"))
    file_text = {p: p.read_text(errors="replace") for p in py_files}
    return py_files, file_text


WRITE_TOKENS = ("to_csv", "csv.writer", "json.dump", "DictWriter", "writerow", "writeheader")


def _parametric_pattern(stem: str):
    """Build a regex tolerant of f-string parametrization of an embedded digit run,
    e.g. stem 'v7_match3_random_50' -> matches 'v7_match{match_level}_random_50'."""
    # split on the FIRST standalone digit run that is bounded by non-digits on the
    # interior (the variant index, e.g. the '3' in v7_match3_random_50).
    m = re.search(r"([A-Za-z_]+)(\d+)(_[A-Za-z0-9_]+)?$", stem)
    if not m:
        return None
    # the index digit is the run immediately after a letter token in the middle
    m2 = re.match(r"^(.*[A-Za-z_])(\d+)(_.*)$", stem)
    if not m2:
        return None
    pre, _idx, post = m2.group(1), m2.group(2), m2.group(3)
    return re.compile(re.escape(pre) + r"\{[^}]+\}" + re.escape(post))


def find_writers(rel_artifact: str, py_files, file_text):
    stem = Path(rel_artifact).stem  # e.g. v7_match2_raw
    param = _parametric_pattern(stem)
    writers = []
    for p in py_files:
        t = file_text[p]
        hit = stem in t or (param is not None and param.search(t) is not None)
        if not hit:
            continue
        if any(tok in t for tok in WRITE_TOKENS):
            writers.append(p)
    return writers


# ---------------------------------------------------------------------------
# Transitive import graph over the src package, + per-module name-reachability.
# We answer for a starting module: can it reach a call to RANK2_TARGET
# (run_single) and/or a HIGHER_RANK target (apply_to_brain / NetworkPerturbation),
# following intra-package imports transitively.
# ---------------------------------------------------------------------------
def module_name_of(p: Path) -> str:
    # module path relative to SRC, dotted, e.g. state_validation.v7_subset_search
    rel = p.relative_to(SRC).with_suffix("")
    return ".".join(rel.parts)


def build_import_graph(py_files, file_text):
    by_module = {module_name_of(p): p for p in py_files}
    imports = defaultdict(set)        # module -> set(imported intra-package modules)
    local_calls = defaultdict(set)    # module -> set(call/attr names used)
    for p in py_files:
        mod = module_name_of(p)
        try:
            tree = ast.parse(file_text[p])
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                for alias in node.names:
                    # candidate intra-package module = base, or base.alias
                    for cand in (base, f"{base}.{alias.name}" if base else alias.name):
                        if cand in by_module:
                            imports[mod].add(cand)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in by_module:
                        imports[mod].add(alias.name)
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    local_calls[mod].add(f.id)
                elif isinstance(f, ast.Attribute):
                    local_calls[mod].add(f.attr)
            elif isinstance(node, ast.Attribute):
                local_calls[mod].add(node.attr)
            elif isinstance(node, ast.Name):
                local_calls[mod].add(node.id)
    return by_module, imports, local_calls


def reaches(start_mod, target_names, imports, local_calls, module_hints=None):
    """BFS over transitive intra-package imports; True if any visited module
    locally references one of target_names (or its file path hints a hint module)."""
    module_hints = module_hints or set()
    seen = set()
    stack = [start_mod]
    hit_modules = []
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        # hint: module name itself is a known higher-rank module
        leaf = m.split(".")[-1]
        if leaf in module_hints:
            hit_modules.append(m)
        if local_calls.get(m, set()) & target_names:
            hit_modules.append(m)
        for dep in imports.get(m, set()):
            if dep not in seen:
                stack.append(dep)
    return (len(hit_modules) > 0), sorted(set(hit_modules)), sorted(seen)


# ---------------------------------------------------------------------------
# Schema fingerprint of an artifact file
# ---------------------------------------------------------------------------
def schema_fingerprint(rel_artifact: str):
    path = ASIM / rel_artifact
    if not path.exists():
        return {"exists": False, "higher_rank_cols_present": [], "classification": "MISSING"}
    cols = set()
    if path.suffix == ".csv":
        with path.open() as f:
            r = csv.reader(f)
            try:
                header = next(r)
            except StopIteration:
                header = []
        cols = set(h.strip() for h in header)
    else:  # json
        try:
            obj = json.loads(path.read_text())
        except Exception:
            obj = {}
        def collect_keys(o, acc):
            if isinstance(o, dict):
                for k, v in o.items():
                    acc.add(str(k))
                    collect_keys(v, acc)
            elif isinstance(o, list):
                for v in o[:50]:
                    collect_keys(v, acc)
        collect_keys(obj, cols)
    hr = sorted(cols & HIGHER_RANK_COLS)
    classification = "HIGHER-RANK-NATIVE" if hr else "RANK2-NATIVE"
    return {"exists": True, "n_cols": len(cols), "higher_rank_cols_present": hr,
            "classification": classification}


# ---------------------------------------------------------------------------
def main():
    result = {"block_id": "P18-C", "gates": {}}

    # G0
    g0 = freeze_denominator()
    result["G0_denominator_freeze"] = g0
    result["prereg_sha256"] = sha256(HERE / "prereg.json")

    py_files, file_text = build_writer_index()
    by_module, imports, local_calls = build_import_graph(py_files, file_text)

    in_scope = g0["in_scope"]

    ledger = []
    contaminated = 0
    untraceable = 0
    for rel in in_scope:
        writers = find_writers(rel, py_files, file_text)
        writer_mods = [module_name_of(w) for w in writers]

        # call-graph reachability from each writer module
        reaches_r2_any = False
        reaches_hr_any = False
        r2_hits_all = []
        hr_hits_all = []
        for wm in writer_mods:
            r2, r2_hits, _ = reaches(wm, {RANK2_TARGET}, imports, local_calls)
            hr, hr_hits, _ = reaches(wm, HIGHER_RANK_TARGETS, imports, local_calls,
                                     module_hints=HIGHER_RANK_MODULE_HINTS)
            reaches_r2_any = reaches_r2_any or r2
            reaches_hr_any = reaches_hr_any or hr
            r2_hits_all += [f"{wm}->{h}" for h in r2_hits]
            hr_hits_all += [f"{wm}->{h}" for h in hr_hits]

        schema = schema_fingerprint(rel)

        # Call-graph attribution:
        #  - reaches run_single and NOT apply_to_brain => RANK2
        #  - reaches apply_to_brain => HIGHER-RANK
        #  - reaches neither but writer is a pure re-analysis of frozen rank2 raw
        #    artifacts (no brain target at all) => RANK2-REANALYSIS
        if reaches_hr_any:
            cg_class = "HIGHER-RANK"
        elif reaches_r2_any:
            cg_class = "RANK2"
        elif writer_mods:
            cg_class = "RANK2-REANALYSIS"  # has writer, no brain-build target reached
        else:
            cg_class = "NO-WRITER"

        traceable = len(writer_mods) > 0
        is_csv = rel.endswith(".csv")
        on_disk = schema["classification"] != "MISSING"

        # final per-artifact verdict:
        #   RANK2-NATIVE         : writer reaches run_single/re-analysis AND schema rank2 AND on disk
        #   HIGHER-RANK-NATIVE   : writer reaches apply_to_brain OR schema carries higher-rank cols
        #   MISSING-NEVER-PRODUCED: referenced (planned) but file absent on disk (no schema to check)
        cg_ok = cg_class in ("RANK2", "RANK2-REANALYSIS")
        if not on_disk:
            verdict = "MISSING-NEVER-PRODUCED"
        elif cg_ok and schema["classification"] == "RANK2-NATIVE" and traceable:
            verdict = "RANK2-NATIVE"
        else:
            verdict = "HIGHER-RANK-NATIVE"

        # contamination = a load-bearing artifact that EXISTS and came from the
        # higher-rank operator (call-graph OR schema). MISSING is NOT contamination.
        if verdict == "HIGHER-RANK-NATIVE":
            contaminated += 1
        # G1b orphan/untraceable: an in-scope CSV with no identifiable writer
        # module. (Per prereg: orphan => FAIL, never pass-by-default.)
        if is_csv and not traceable:
            untraceable += 1

        ledger.append({
            "artifact": rel,
            "referenced_by": g0["referenced_by"].get(rel, []),
            "writer_modules": writer_mods,
            "traceable": traceable,
            "callgraph_class": cg_class,
            "reaches_run_single": reaches_r2_any,
            "reaches_apply_to_brain": reaches_hr_any,
            "run_single_hits": sorted(set(r2_hits_all))[:5],
            "higher_rank_hits": sorted(set(hr_hits_all))[:5],
            "schema_class": schema["classification"],
            "schema_higher_rank_cols": schema["higher_rank_cols_present"],
            "verdict": verdict,
        })

    # ---- Known suspect classification (out-of-scope check) ----
    suspect = KNOWN_SUSPECT
    suspect_writers = [module_name_of(w) for w in find_writers(suspect, py_files, file_text)]
    suspect_schema = schema_fingerprint(suspect)
    suspect_hr = False
    suspect_hits = []
    for wm in suspect_writers:
        hr, hits, _ = reaches(wm, HIGHER_RANK_TARGETS, imports, local_calls,
                              module_hints=HIGHER_RANK_MODULE_HINTS)
        suspect_hr = suspect_hr or hr
        suspect_hits += [f"{wm}->{h}" for h in hits]
    # who consumes the suspect (website-only check)
    consumers = [module_name_of(p) for p in py_files
                 if Path(suspect).stem in file_text[p] and "read_csv" in file_text[p]
                 or (Path(suspect).name in file_text[p])]
    result["known_suspect_analysis"] = {
        "artifact": suspect,
        "in_load_bearing_scope": g0["known_suspect_in_scope"],
        "writer_modules": suspect_writers,
        "reaches_apply_to_brain": suspect_hr,
        "higher_rank_hits": sorted(set(suspect_hits))[:5],
        "schema_class": suspect_schema["classification"],
        "schema_higher_rank_cols": suspect_schema["higher_rank_cols_present"],
        "consumer_modules": sorted(set(consumers)),
        "disposition": ("HIGHER-RANK-NATIVE-but-OUT-OF-SCOPE-website-only"
                        if (not g0["known_suspect_in_scope"]) else "HIGHER-RANK-IN-SCOPE-CONTAMINATION"),
    }

    result["ledger"] = ledger

    # ---- Gate evaluation ----
    n = len(in_scope)
    missing_list = [x["artifact"] for x in ledger if x["verdict"] == "MISSING-NEVER-PRODUCED"]
    n_missing = len(missing_list)
    g0_pass = (n > 0) and all(isinstance(x, str) for x in in_scope)
    g1_pass = (contaminated == 0) and (n > 0)
    g1b_pass = (untraceable == 0) and (n_missing == 0)

    result["gates"] = {
        "G0_denominator_freeze": {
            "in_scope_count": n,
            "verdict": "PASS" if g0_pass else "FAIL",
            "criterion": "in-scope set fixed by frozen predicate, non-empty",
        },
        "G1_rank2_attribution": {
            "in_scope_count": n,
            "rank2_native_count": sum(1 for x in ledger if x["verdict"] == "RANK2-NATIVE"),
            "contaminated_count": contaminated,
            "contaminated_artifacts": [x["artifact"] for x in ledger if x["verdict"] == "HIGHER-RANK-NATIVE"],
            "verdict": "PASS" if g1_pass else "FAIL",
            "criterion": "100% in-scope RANK2-NATIVE by call-graph AND schema; contaminated==0",
        },
        "G1b_untraceable": {
            "untraceable_in_scope_csv": untraceable,
            "missing_never_produced_count": n_missing,
            "missing_never_produced_artifacts": missing_list,
            "verdict": "PASS" if g1b_pass else "FAIL",
            "criterion": "zero orphan in-scope CSVs AND zero referenced-but-absent in-scope artifacts (orphan/missing => FAIL, never pass-by-default)",
        },
    }
    overall = "PASS" if (g0_pass and g1_pass and g1b_pass) else "FAIL"
    result["overall_verdict"] = overall

    out = HERE / "result.json"
    out.write_text(json.dumps(result, indent=2))
    print("=" * 70)
    print("P18-C ARTIFACT PROVENANCE LEDGER")
    print("=" * 70)
    print(f"Load-bearing docs frozen: {[str(d.relative_to(ASIM)) for d in LOAD_BEARING_DOCS]}")
    print(f"prereg.json sha256: {result['prereg_sha256']}")
    print(f"In-scope artifact count (G0 denominator): {n}")
    print("-" * 70)
    for x in ledger:
        print(f"[{x['verdict']:18s}] {x['artifact']}")
        print(f"    writer={x['writer_modules']} cg={x['callgraph_class']} "
              f"(run_single={x['reaches_run_single']}, apply_to_brain={x['reaches_apply_to_brain']}) "
              f"schema={x['schema_class']} hr_cols={x['schema_higher_rank_cols']}")
    print("-" * 70)
    s = result["known_suspect_analysis"]
    print("KNOWN SUSPECT:", s["artifact"])
    print(f"    in_load_bearing_scope={s['in_load_bearing_scope']} "
          f"writer={s['writer_modules']} apply_to_brain={s['reaches_apply_to_brain']} "
          f"schema={s['schema_class']} hr_cols={s['schema_higher_rank_cols']}")
    print(f"    consumers={s['consumer_modules']}")
    print(f"    disposition={s['disposition']}")
    print("-" * 70)
    for gname, g in result["gates"].items():
        print(f"{gname}: {g['verdict']}  {g}")
    print("=" * 70)
    print(f"OVERALL: {overall}")
    return result


if __name__ == "__main__":
    main()
