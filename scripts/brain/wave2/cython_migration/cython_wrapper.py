"""Helper: wrap a wave2 cell factory so that prefs.codegen.target is
overridden to a chosen target AFTER the wrapped factory returns its bundle.

The wave2 cell factories (option_alpha_*_cell.py) hardcode
`prefs.codegen.target = "cython"` inside the factory body. To run validation
under cython without modifying production code, wrap the factory:

    factory = build_brian2_aval_4channel(...)
    cython_factory = wrap_factory_with_codegen(factory, "cython")
    voltage_clamp_compare_v2(cython_factory, nref, holds, ...)

The wrapper invokes the inner factory (which sets prefs to numpy), then
overrides prefs.codegen.target to the desired value before the bundle is
used. Brian2 reads codegen target at network.run() time, so this works
correctly.
"""
from __future__ import annotations

from typing import Callable, Any


def wrap_factory_with_codegen(inner_factory: Callable[[], Any],
                              target: str) -> Callable[[], Any]:
    """Return a new factory that calls inner_factory and overrides codegen
    target to `target` (e.g., 'cython' or 'numpy').
    """
    def _wrapped():
        from brian2 import prefs
        bundle = inner_factory()
        prefs.codegen.target = target
        return bundle
    return _wrapped
