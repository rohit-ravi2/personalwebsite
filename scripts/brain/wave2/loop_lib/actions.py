"""
Action library for the agentic loop.

Each action is a parameterized fix that modifies a parameter file. Each
action returns:
  - description: human-readable text for the log
  - revert: callable that undoes the change

Actions operate on parameter files (not equations). They are bounded by
biological envelopes defined in envelopes.py.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WAVE2_DIR = THIS_DIR.parent

# Envelopes
E_LEAK_MIN, E_LEAK_MAX = -90.0, -30.0
G_LEAK_MIN, G_LEAK_MAX = 1e-6, 1e-3
PUMP_SCALE_MAX = 20.0
C_GLOBAL_MIN, C_GLOBAL_MAX = 1e3, 1e6


def _backup(file_path: Path) -> Path:
    """Make a backup copy of a file; return the backup path."""
    backup = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copyfile(file_path, backup)
    return backup


def _restore(backup_path: Path):
    """Restore from backup."""
    original = backup_path.with_suffix("")
    if original.suffix == ".bak":
        # if double-suffix issue
        original = Path(str(backup_path).rsplit(".bak", 1)[0])
    shutil.copyfile(backup_path, original)


def _replace_in_file(file_path: Path, pattern: str, replacement: str) -> int:
    """Replace pattern (regex) with replacement in file; return # replacements."""
    text = file_path.read_text()
    new_text, n = re.subn(pattern, replacement, text)
    if n > 0:
        file_path.write_text(new_text)
    return n


# ----------------------------------------------------------------------
# Action: raise/lower MAX_PUMP_SCALE in pump_capacity_scaling.py
# ----------------------------------------------------------------------

def set_max_pump_scale(new_value: float):
    if new_value <= 1.0 or new_value > PUMP_SCALE_MAX:
        raise ValueError(f"pump scale {new_value} outside envelope (1, {PUMP_SCALE_MAX}]")
    f = WAVE2_DIR / "path2_scale" / "pump_capacity_scaling.py"
    backup = _backup(f)
    n = _replace_in_file(f, r"MAX_PUMP_SCALE\s*=\s*[\d.]+", f"MAX_PUMP_SCALE = {new_value}")
    if n == 0:
        _restore(backup)
        raise RuntimeError("Could not find MAX_PUMP_SCALE to update")
    return {"description": f"MAX_PUMP_SCALE -> {new_value}",
            "revert": lambda: _restore(backup),
            "backup": str(backup)}


# ----------------------------------------------------------------------
# Action: scale C_GLOBAL_DEFAULT in scalable_builder.py
# ----------------------------------------------------------------------

def set_c_global_default(new_value: float):
    if new_value < C_GLOBAL_MIN or new_value > C_GLOBAL_MAX:
        raise ValueError(f"C_global {new_value} outside envelope")
    f = WAVE2_DIR / "path2_scale" / "scalable_builder.py"
    backup = _backup(f)
    n = _replace_in_file(f, r"C_GLOBAL_DEFAULT\s*=\s*[\d.eE+-]+",
                          f"C_GLOBAL_DEFAULT = {new_value:.3e}")
    if n == 0:
        _restore(backup)
        raise RuntimeError("Could not find C_GLOBAL_DEFAULT to update")
    return {"description": f"C_GLOBAL_DEFAULT -> {new_value:.3e}",
            "revert": lambda: _restore(backup),
            "backup": str(backup)}


# ----------------------------------------------------------------------
# Action: scale DEFAULT_E_LEAK_MV in scalable_builder.py
# ----------------------------------------------------------------------

def set_default_e_leak(new_value_mV: float):
    if new_value_mV < E_LEAK_MIN or new_value_mV > E_LEAK_MAX:
        raise ValueError(f"e_leak {new_value_mV} outside [{E_LEAK_MIN}, {E_LEAK_MAX}] mV")
    f = WAVE2_DIR / "path2_scale" / "scalable_builder.py"
    backup = _backup(f)
    n = _replace_in_file(f, r"DEFAULT_E_LEAK_MV\s*=\s*[-\d.]+",
                          f"DEFAULT_E_LEAK_MV = {new_value_mV}")
    if n == 0:
        _restore(backup)
        raise RuntimeError("Could not find DEFAULT_E_LEAK_MV")
    # Also update DEFAULT_V_INIT to match
    _replace_in_file(f, r"DEFAULT_V_INIT\s*=\s*[-\d.]+",
                     f"DEFAULT_V_INIT = {new_value_mV}")
    return {"description": f"DEFAULT_E_LEAK_MV -> {new_value_mV}",
            "revert": lambda: _restore(backup),
            "backup": str(backup)}


# ----------------------------------------------------------------------
# Action: scale DEFAULT_G_LEAK_SCM2 in scalable_builder.py
# ----------------------------------------------------------------------

def set_default_g_leak(new_value_Scm2: float):
    if new_value_Scm2 < G_LEAK_MIN or new_value_Scm2 > G_LEAK_MAX:
        raise ValueError(f"g_leak {new_value_Scm2} outside envelope")
    f = WAVE2_DIR / "path2_scale" / "scalable_builder.py"
    backup = _backup(f)
    n = _replace_in_file(f, r"DEFAULT_G_LEAK_SCM2\s*=\s*[\d.eE+-]+",
                          f"DEFAULT_G_LEAK_SCM2 = {new_value_Scm2:.3e}")
    if n == 0:
        _restore(backup)
        raise RuntimeError("Could not find DEFAULT_G_LEAK_SCM2")
    return {"description": f"DEFAULT_G_LEAK_SCM2 -> {new_value_Scm2:.3e}",
            "revert": lambda: _restore(backup),
            "backup": str(backup)}


# ----------------------------------------------------------------------
# Action: add a Ca-clearance pump multiplier (similar to pump_NaK_scale)
# ----------------------------------------------------------------------

def set_ca_clearance_scale_for_failing(failing_cells: list[str], scale: float):
    """Add a per-cell Ca-clearance scaling override.

    We accumulate cells in pump_capacity_scaling.py via a class-keyed dict
    CA_CLEAR_OVERRIDE = {"HSN": 5.0, ...}. The scalable_builder reads this
    and applies it via spec.ca_clearance_scale.
    """
    f = WAVE2_DIR / "path2_scale" / "pump_capacity_scaling.py"
    backup = _backup(f)
    text = f.read_text()
    # Look for CA_CLEAR_OVERRIDE dict; if not present, add it.
    if "CA_CLEAR_OVERRIDE" not in text:
        addendum = "\n\n# Per-cell Ca-clearance scaling overrides (added by overnight loop)\nCA_CLEAR_OVERRIDE: dict[str, float] = {}\n"
        text = text + addendum
    # Update overrides
    new_overrides = ", ".join(f'"{c}": {scale}' for c in failing_cells)
    text = re.sub(r"CA_CLEAR_OVERRIDE\s*:\s*dict\[str,\s*float\]\s*=\s*\{[^}]*\}",
                  f"CA_CLEAR_OVERRIDE: dict[str, float] = {{{new_overrides}}}",
                  text, count=1)
    f.write_text(text)
    return {"description": f"CA_CLEAR_OVERRIDE += {failing_cells} @ {scale}x",
            "revert": lambda: _restore(backup),
            "backup": str(backup)}


# ----------------------------------------------------------------------
# Action: implement KVS-1 channel (Kv3 analog of EGL-36)
# ----------------------------------------------------------------------

def implement_kvs1():
    """Create channels/kvs1.py modeled on egl36.py.

    KVS-1 is Kv3 family, similar V_half and slope. Same NMODL structure.
    """
    kvs1_path = WAVE2_DIR / "channels" / "kvs1.py"
    if kvs1_path.exists():
        return {"description": "KVS-1 already implemented", "revert": lambda: None,
                "noop": True}

    content = '''"""
KVS-1 voltage-gated K channel — Brian2 module.

KVS-1 is a C. elegans Kv3 (Shaw) family channel, sibling to EGL-36.
Modeled identically (Kv3 family kinetics conserved). γ = 16 pS.

Default parameters (Kv3 canonical):
  va = 10 mV (depolarization-activated)
  ka = 8 mV (slope)
  mtau = 1.5 ms (fast Kv3 activation)
"""
from __future__ import annotations


KVS1_PARAMS = {
    "va_kvs1":          10.0,
    "ka_kvs1":           8.0,
    "mtau_kvs1":         1.5,
    "gbar_kvs1_Scm2":    1.0e-4,
    "ek_mV":           -80.0,
}


KVS1_EQS = """
# KVS-1 Kv3 Shaw-family K channel, m^4 non-inactivating.
kvs1_minf = 1.0 / (1.0 + exp(-(v_mV - kvs1_va) / kvs1_ka)) : 1
dm_kvs1/dt = (kvs1_minf - m_kvs1) / (kvs1_mtau * ms) : 1
ik_kvs1_mAcm2 = kvs1_gbar * m_kvs1 * m_kvs1 * m_kvs1 * m_kvs1 * (v_mV - kvs1_ek) : 1
# Parameters:
kvs1_va : 1
kvs1_ka : 1
kvs1_mtau : 1
kvs1_gbar : 1
kvs1_ek : 1
"""


def kvs1_apply_params(group, gbar_Scm2=None, ek_mV=None, params_override=None):
    p = dict(KVS1_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_kvs1_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV
    name_map = {
        "va_kvs1":         "kvs1_va",
        "ka_kvs1":         "kvs1_ka",
        "mtau_kvs1":       "kvs1_mtau",
        "gbar_kvs1_Scm2":  "kvs1_gbar",
        "ek_mV":           "kvs1_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])


def kvs1_init_states(group, v_mV=-60.0):
    import numpy as np
    p = KVS1_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_kvs1"]) / p["ka_kvs1"]))
    group.m_kvs1 = float(minf)


NAME = "kvs1"
EQS = KVS1_EQS
apply_params = kvs1_apply_params
init_states = kvs1_init_states
'''
    kvs1_path.write_text(content)

    # Wire into layer1_cells.py
    l1 = WAVE2_DIR / "layer1_cells.py"
    l1_backup = _backup(l1)
    l1_text = l1.read_text()

    if "from channels import kvs1 as kvs1_mod" not in l1_text:
        l1_text = l1_text.replace(
            "from channels import egl36 as egl36_mod",
            "from channels import egl36 as egl36_mod\nfrom channels import kvs1 as kvs1_mod",
        )
    l1_text = l1_text.replace(
        '"ik_egl36_mAcm2"]',
        '"ik_egl36_mAcm2",\n                   "ik_kvs1_mAcm2"]',
    )
    l1_text = l1_text.replace(
        '"egl36":  (egl36_mod, "ik_egl36_mAcm2", "K", ("egl36_ek",)),\n    }',
        '"egl36":  (egl36_mod, "ik_egl36_mAcm2", "K", ("egl36_ek",)),\n        "kvs1":   (kvs1_mod, "ik_kvs1_mAcm2", "K", ("kvs1_ek",)),\n    }',
    )
    l1_text = l1_text.replace(
        'if any(v.endswith("egl36_mAcm2") for v in present_K):  bridges.append("egl36_ek = E_K_mV : 1")',
        'if any(v.endswith("egl36_mAcm2") for v in present_K):  bridges.append("egl36_ek = E_K_mV : 1")\n    if any(v.endswith("kvs1_mAcm2") for v in present_K):   bridges.append("kvs1_ek = E_K_mV : 1")',
    )
    # _CHANNEL_APPLIES entry — add before final closing brace
    kvs1_entry = '''    "kvs1": {
        "params_attr": "KVS1_PARAMS",
        "gbar_key": "gbar_kvs1_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va_kvs1",   "kvs1_va"),  ("ka_kvs1", "kvs1_ka"),
            ("mtau_kvs1", "kvs1_mtau"),
            ("gbar_kvs1_Scm2", "kvs1_gbar"),
        ],
    },
'''
    # Find the closing brace of _CHANNEL_APPLIES (which ends with "}")
    # Match the egl36 entry's closing brace
    l1_text = l1_text.replace(
        '''    "egl36": {
        "params_attr": "EGL36_PARAMS",
        "gbar_key": "gbar_egl36_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va_egl36",   "egl36_va"),  ("ka_egl36", "egl36_ka"),
            ("mtau_egl36", "egl36_mtau"),
            ("gbar_egl36_Scm2", "egl36_gbar"),
        ],
    },
}''',
        '''    "egl36": {
        "params_attr": "EGL36_PARAMS",
        "gbar_key": "gbar_egl36_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va_egl36",   "egl36_va"),  ("ka_egl36", "egl36_ka"),
            ("mtau_egl36", "egl36_mtau"),
            ("gbar_egl36_Scm2", "egl36_gbar"),
        ],
    },
''' + kvs1_entry + "}",
    )
    l1_text = l1_text.replace(
        '"slo2": slo2_mod, "egl36": egl36_mod,\n}',
        '"slo2": slo2_mod, "egl36": egl36_mod, "kvs1": kvs1_mod,\n}',
    )
    # init_states branch
    l1_text = l1_text.replace(
        '''        elif ch_name == "egl36" and hasattr(egl36_mod, "egl36_init_states"):
            egl36_mod.egl36_init_states(group, v_mV=v)''',
        '''        elif ch_name == "egl36" and hasattr(egl36_mod, "egl36_init_states"):
            egl36_mod.egl36_init_states(group, v_mV=v)
        elif ch_name == "kvs1" and hasattr(kvs1_mod, "kvs1_init_states"):
            kvs1_mod.kvs1_init_states(group, v_mV=v)''',
    )
    l1.write_text(l1_text)

    # Wire into scalable_builder.py
    sb = WAVE2_DIR / "path2_scale" / "scalable_builder.py"
    sb_backup = _backup(sb)
    sb_text = sb.read_text()
    sb_text = sb_text.replace(
        '"SLO-2", "EGL-36"}',
        '"SLO-2", "EGL-36", "KVS-1"}',
    )
    sb_text = sb_text.replace(
        'UNSUPPORTED_CHANNELS = {"KVS-1", "SLO-1", "KQT-2", "KQT-3"}',
        'UNSUPPORTED_CHANNELS = {"SLO-1", "KQT-2", "KQT-3"}',
    )
    sb_text = sb_text.replace(
        '("slo-2", "slo2"), ("egl-36", "egl36"),\n    ]:',
        '("slo-2", "slo2"), ("egl-36", "egl36"), ("kvs-1", "kvs1"),\n    ]:',
    )
    # Remove kvs-1 from skipped list
    sb_text = sb_text.replace(
        'for gene_check in ("kvs-1", "slo-1"):',
        'for gene_check in ("slo-1",):',
    )
    sb_text = sb_text.replace(
        'for gene in ("kvs-1", "slo-1"):',
        'for gene in ("slo-1",):',
    )
    sb.write_text(sb_text)

    def revert():
        kvs1_path.unlink(missing_ok=True)
        _restore(l1_backup)
        _restore(sb_backup)

    return {"description": "Implement KVS-1 (Kv3 analog)", "revert": revert,
            "backup": str(l1_backup)}


# ----------------------------------------------------------------------
# Action: allow Nicoletti pump_NaK_scale override (e.g. for RIM)
# ----------------------------------------------------------------------

def nicoletti_pump_scale(cell_name: str, scale: float):
    """Apply pump_NaK_scale to a Nicoletti cell (nuanced deviation).

    Caller is responsible for documenting rationale.
    """
    if cell_name not in ("AVAL", "AVAR", "AIY", "RIM"):
        raise ValueError(f"{cell_name} not a Nicoletti cell")
    if scale <= 0 or scale > PUMP_SCALE_MAX:
        raise ValueError(f"scale {scale} out of bounds")
    # Modify scalable_builder.to_layer1_cellspec to use scale for that cell
    f = WAVE2_DIR / "path2_scale" / "scalable_builder.py"
    backup = _backup(f)
    text = f.read_text()
    # Find and replace the Nicoletti branch
    old = '''    if s.name in ("AVAL", "AVAR", "AIY", "RIM"):
        pump_key = s.name
        pump_scale = 1.0  # Nicoletti cells use their own calibrated pumps'''
    # Build the new branch that overrides for this specific cell
    new = f'''    if s.name in ("AVAL", "AVAR", "AIY", "RIM"):
        pump_key = s.name
        pump_scale = {scale} if s.name == "{cell_name}" else 1.0'''
    if old not in text:
        # May already be in a modified state — try to find existing override
        # and update; otherwise abort
        _restore(backup)
        raise RuntimeError("Cannot locate Nicoletti pump_scale branch (already modified?)")
    text = text.replace(old, new)
    f.write_text(text)
    return {"description": f"Nicoletti override: {cell_name} pump_NaK_scale={scale}",
            "revert": lambda: _restore(backup),
            "backup": str(backup)}
