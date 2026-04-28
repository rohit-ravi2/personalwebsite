#!/usr/bin/env python3
"""Phase α — Wave 2 setup verification + Nicoletti mod compilation runner.

Idempotent: re-running is safe. Verifies:
  1. NEURON + Brian2 + numpy/scipy/matplotlib import in this venv
  2. Nicoletti's 24 .mod files compile via nrnivmodl (cleans + recompiles
     if `--force` given; otherwise only compiles when libnrnmech.so is missing)
  3. The compiled mechanism library exposes all 24 Nicoletti density mechanisms

Designed to be run inside the isolated venv at ~/venvs/wave2-neuron/.

Usage:
    /home/rohit/venvs/wave2-neuron/bin/python setup_neuron.py [--force]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

NICOLETTI_DIR = Path(
    "/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024"
)

EXPECTED_MECHS = [
    "cadiff", "caintra1", "cca1", "egl19", "egl2", "egl36", "exp2",
    "irk", "kcnl", "kqt1", "kqt3", "kvs1", "leak", "nca",
    "shk1", "shl1",
    "slo1egl19", "slo1iso", "slo1unc2",
    "slo2egl19", "slo2iso", "slo2unc2",
    "unc103", "unc2",
]


def verify_imports() -> dict:
    """Import NEURON / Brian2 / scientific stack and return version dict."""
    from neuron import h  # noqa: WPS433
    import brian2  # noqa: WPS433
    import numpy  # noqa: WPS433
    import scipy  # noqa: WPS433
    import matplotlib  # noqa: WPS433

    return {
        "python": sys.version.split()[0],
        "neuron": h.nrnversion(),
        "brian2": brian2.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def compile_mods(force: bool = False) -> dict:
    """Compile Nicoletti mod files via nrnivmodl. Idempotent unless force=True."""
    if not NICOLETTI_DIR.exists():
        raise FileNotFoundError(f"Nicoletti dir not found: {NICOLETTI_DIR}")

    arch_dir = NICOLETTI_DIR / "x86_64"
    libpath = arch_dir / "libnrnmech.so"

    mod_files = sorted(NICOLETTI_DIR.glob("*.mod"))
    if not mod_files:
        raise RuntimeError(f"No .mod files found in {NICOLETTI_DIR}")

    if force and arch_dir.exists():
        shutil.rmtree(arch_dir)

    if libpath.exists() and not force:
        return {
            "compiled": False,
            "reason": "libnrnmech.so already present (use --force to recompile)",
            "n_mods": len(mod_files),
            "libpath": str(libpath),
        }

    nrnivmodl = shutil.which("nrnivmodl")
    if nrnivmodl is None:
        # fall back to venv-local
        venv_bin = Path(sys.executable).parent
        nrnivmodl = str(venv_bin / "nrnivmodl")
    proc = subprocess.run(
        [nrnivmodl],
        cwd=NICOLETTI_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "compiled": True,
        "returncode": proc.returncode,
        "n_mods": len(mod_files),
        "libpath": str(libpath),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-15:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-15:]),
    }


_LIST_MECHS_SCRIPT = """
from neuron import h
mt = h.MechanismType(0)
n = int(mt.count())
for i in range(n):
    name = h.ref('')
    mt.select(i)
    mt.selected(name)
    print(name[0])
"""


def list_mechanisms() -> list[str]:
    """Spawn a subprocess in Nicoletti dir to list density mechanisms.

    Subprocess is required because NEURON's `from neuron import h`
    triggers auto-load of libnrnmech.so from the *current* working
    directory at *import time* — chdir() after import does not retro-
    actively load the library. A clean subprocess starts in Nicoletti dir.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _LIST_MECHS_SCRIPT],
        cwd=NICOLETTI_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force-recompile mod files (delete x86_64/ first)",
    )
    args = parser.parse_args(argv)

    print("=== Phase α — Wave 2 setup verification ===")
    print()
    print("[1/3] Verifying Python imports...")
    versions = verify_imports()
    for k, v in versions.items():
        print(f"  {k}: {v}")

    print()
    print("[2/3] Compiling Nicoletti mod files...")
    compile_info = compile_mods(force=args.force)
    if compile_info["compiled"]:
        print(f"  nrnivmodl returncode: {compile_info['returncode']}")
        if compile_info["returncode"] != 0:
            print("  STDOUT tail:")
            print("    " + compile_info["stdout_tail"].replace("\n", "\n    "))
            print("  STDERR tail:")
            print("    " + compile_info["stderr_tail"].replace("\n", "\n    "))
            return 1
    else:
        print(f"  skipped: {compile_info['reason']}")
    print(f"  mod files: {compile_info['n_mods']}")
    print(f"  library: {compile_info['libpath']}")

    print()
    print("[3/3] Listing mechanisms loaded by NEURON...")
    mechs = list_mechanisms()
    present = [m for m in EXPECTED_MECHS if m in mechs]
    missing = [m for m in EXPECTED_MECHS if m not in mechs]
    print(f"  density mechanisms loaded: {len(mechs)}")
    print(f"  Nicoletti present: {len(present)}/{len(EXPECTED_MECHS)}")
    if missing:
        print(f"  MISSING: {missing}")
        return 1

    print()
    print("=== Setup verified — ready for reference_validation.py ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
