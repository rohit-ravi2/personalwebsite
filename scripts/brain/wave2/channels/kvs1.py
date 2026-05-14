"""
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
