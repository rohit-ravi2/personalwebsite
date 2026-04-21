#!/usr/bin/env python3
"""Phase T1e — 2D agar-plate environment with chemical gradients.

Adds spatial ecology to the closed-loop worm simulation:
  - 2D substrate, worm position tracked from MuJoCo body CoM
  - Attractant concentration field (2D Gaussian around food patch)
  - Optional repellent gradient
  - Food landmark at fixed position
  - Sensory neurons (ASE/AWC/AWA) receive Poisson rates proportional to
    concentration AT the worm's head, + gradient sensing component
    (Pierce-Shimomura 1999 navigation: ON-cells respond to dC/dt > 0,
    OFF-cells to dC/dt < 0)

Enables the canonical worm validation: chemotaxis index measurement.
If the worm's brain produces correct navigation behaviour, it should
reach the food patch more often than chance.

Integration: ClosedLoopEnv can optionally hold an Environment; each
sync step queries concentration at the worm's current head position,
converts to sensory drive, injects into brain.
"""
from __future__ import annotations

import math

import numpy as np


class ChemoGradient:
    """2D Gaussian attractant field centred on a food patch."""

    def __init__(self,
                 food_xy: tuple[float, float] = (8.0e-3, 0.0),  # 8 mm from origin
                 peak_conc: float = 1.0,    # arbitrary units, normalised 0-1
                 sigma_m: float = 4.0e-3,    # 4 mm spatial SD
                 ):
        self.food_x, self.food_y = food_xy
        self.peak = peak_conc
        self.sigma = sigma_m

    def concentration(self, x: float, y: float) -> float:
        dx = x - self.food_x
        dy = y - self.food_y
        d2 = dx * dx + dy * dy
        return self.peak * math.exp(-d2 / (2 * self.sigma * self.sigma))

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        """∇C(x, y) — used when we want spatial-gradient sensing rather
        than temporal dC/dt."""
        c = self.concentration(x, y)
        return (
            -c * (x - self.food_x) / (self.sigma * self.sigma),
            -c * (y - self.food_y) / (self.sigma * self.sigma),
        )


# ---------------------------------------------------------------------
# P0 #3 — O2 / CO2 fields and aerotaxis sensory wiring
# ---------------------------------------------------------------------
#
# Biology:
#   URX, AQR, PQR: HIGH-O2 sensors via GCY-35/36 soluble guanylyl
#   cyclases. C. elegans prefers ~10-14% O2; the response to HIGH O2
#   (21% atmospheric) is aversive in wild N2 (NPR-1 215V lab strain
#   is less aversive). URX tonically tracks O2 with a short adaptation
#   timescale (~5 s).
#
#   BAG: O2-decrease + CO2-increase sensor. GCY-9 mediates CO2
#   response. BAG is OFF-cell for O2 (fires when O2 drops) and ON-cell
#   for CO2 (fires when CO2 rises).
#
#   URX → RMG → AVA/AVB command path (Macosko 2009, Coates 2002).
#   BAG → AIY / RIA / RIB path (Zimmer 2009).
#
# References:
#   - Gray 2004 Nature: first description of O2 aerotaxis
#   - Cheung 2005: GCY-35 O2 sensing
#   - Hallem & Sternberg 2008: CO2 via BAG
#   - Zimmer 2009: URX and BAG complementary O2 sensors
#   - Laurent 2015: URX adaptation kinetics


class LinearGasField:
    """1D linear gradient (along +x direction) of a dissolved gas.
    Typical experimental setup: microfluidic chamber with 21% O2 at
    one end, 7% at the other. `atmo_frac(x, y)` returns the O2 (or
    CO2) fraction at the worm's position."""

    def __init__(self,
                 min_frac: float = 0.07,   # 7% O2 (or 0.0004 for CO2 baseline)
                 max_frac: float = 0.21,   # 21% O2 (or 0.05 for high CO2)
                 axis: str = "x",           # gradient along +x
                 x_min_m: float = -10e-3,   # position at min_frac
                 x_max_m: float = 10e-3,    # position at max_frac
                 ):
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.axis = axis
        self.x_min = x_min_m
        self.x_max = x_max_m

    def fraction(self, x: float, y: float) -> float:
        coord = x if self.axis == "x" else y
        t = (coord - self.x_min) / (self.x_max - self.x_min + 1e-12)
        t = max(0.0, min(1.0, t))
        return self.min_frac + t * (self.max_frac - self.min_frac)


class RadialGasField:
    """2D radial gas source/sink centred at a point. Useful for a
    worm colony emitting CO2 or a pharmacological bubble."""

    def __init__(self,
                 center_xy: tuple[float, float] = (0.0, 0.0),
                 baseline_frac: float = 0.21,
                 peak_frac: float = 0.07,   # drops to 7% near centre
                 sigma_m: float = 5e-3,
                 ):
        self.cx, self.cy = center_xy
        self.baseline = baseline_frac
        self.peak = peak_frac
        self.sigma = sigma_m

    def fraction(self, x: float, y: float) -> float:
        dx = x - self.cx
        dy = y - self.cy
        w = math.exp(-(dx * dx + dy * dy) / (2 * self.sigma * self.sigma))
        return self.baseline + w * (self.peak - self.baseline)


class AerotaxisSensory:
    """Injects O2- and CO2-driven Poisson rates into aerotaxis cells.

    Cell wiring (Zimmer 2009 / Hallem & Sternberg 2008 / Laurent 2015):
      - URX, AQR, PQR: ON-cells for HIGH O2 (fire when O2 > preferred)
      - BAG: OFF-cell for O2 (fires when O2 drops below prefered)
              AND ON-cell for CO2 (fires when CO2 rises)

    Rates are tonic responses to the instantaneous gas fraction + a
    derivative response to temporal change, with a 5-s adaptation
    timescale for URX (Laurent 2015) baked into the averaging window.
    """

    # URX/AQR/PQR expressed in connectome
    HIGH_O2_NEURONS = ["URXL", "URXR", "AQR", "PQR"]
    # BAG: left + right
    BAG_NEURONS = ["BAGL", "BAGR"]

    def __init__(self,
                 o2_field: LinearGasField | RadialGasField | None = None,
                 co2_field: LinearGasField | RadialGasField | None = None,
                 preferred_o2_frac: float = 0.12,
                 tonic_max_rate_hz: float = 120.0,
                 deriv_max_rate_hz: float = 80.0,
                 deriv_window_s: float = 5.0,       # URX adaptation
                 ):
        self.o2 = o2_field
        self.co2 = co2_field
        self.pref_o2 = preferred_o2_frac
        self.tonic_max = tonic_max_rate_hz
        self.deriv_max = deriv_max_rate_hz
        self.deriv_window = deriv_window_s

        # Rolling history per channel for dX/dt
        self._o2_hist: list[tuple[float, float]] = []
        self._co2_hist: list[tuple[float, float]] = []

        # Snapshots for telemetry
        self.last_o2: float = preferred_o2_frac
        self.last_co2: float = 0.0004  # atmospheric baseline
        self.last_do2_dt: float = 0.0
        self.last_dco2_dt: float = 0.0

    def _sample_and_deriv(self, field, hist: list, x: float, y: float,
                           t_s: float) -> tuple[float, float]:
        if field is None:
            return (0.0, 0.0)
        f = field.fraction(x, y)
        hist.append((t_s, f))
        cutoff = t_s - self.deriv_window
        while len(hist) > 2 and hist[0][0] < cutoff:
            hist.pop(0)
        d = 0.0
        if len(hist) >= 2:
            dt = hist[-1][0] - hist[0][0]
            if dt > 0:
                d = (hist[-1][1] - hist[0][1]) / dt
        return (f, d)

    def update(self, x: float, y: float, t_s: float) -> None:
        self.last_o2, self.last_do2_dt = self._sample_and_deriv(
            self.o2, self._o2_hist, x, y, t_s
        )
        if self.co2 is not None:
            self.last_co2, self.last_dco2_dt = self._sample_and_deriv(
                self.co2, self._co2_hist, x, y, t_s
            )

    def inject_into_brain(self, brain) -> None:
        # High-O2 neurons: rate scales with (O2 − preferred)_+ and +dO2/dt
        excess = max(0.0, self.last_o2 - self.pref_o2)
        o2_scale = min(1.0, excess / max(0.01, 0.21 - self.pref_o2))
        pos_d = max(0.0, self.last_do2_dt)
        d_scale = min(1.0, pos_d / 0.02)     # 2% O2/s saturates
        high_rate = self.tonic_max * o2_scale + self.deriv_max * d_scale
        for n in self.HIGH_O2_NEURONS:
            brain.set_sensory_rate(n, high_rate, weight_mv=8)

        # BAG: responds to O2 decrease AND CO2 increase
        deficit = max(0.0, self.pref_o2 - self.last_o2)
        o2_deficit_scale = min(1.0, deficit / max(0.01, self.pref_o2))
        neg_d = max(0.0, -self.last_do2_dt)
        d_deficit = min(1.0, neg_d / 0.02)
        co2_rise = max(0.0, self.last_dco2_dt)
        co2_scale = min(1.0, co2_rise / 0.001)  # 0.1%/s saturates
        bag_rate = (self.tonic_max * o2_deficit_scale
                    + self.deriv_max * d_deficit
                    + self.deriv_max * co2_scale)
        for n in self.BAG_NEURONS:
            brain.set_sensory_rate(n, bag_rate, weight_mv=8)

    def telemetry(self) -> dict:
        return {
            "o2_frac": round(self.last_o2, 4),
            "co2_frac": round(self.last_co2, 5),
            "do2_dt": round(self.last_do2_dt, 5),
            "dco2_dt": round(self.last_dco2_dt, 6),
            "preferred_o2": self.pref_o2,
        }


class Environment:
    """2D agar environment around a worm. Tracks worm head position,
    samples concentration field, translates to sensory neuron Poisson
    rates injected into the brain."""

    # ASE/AWC/AWA assignments (Bargmann 1993, Chalasani 2007):
    #   ASE — salt-sensing, primary chemotaxis driver in many assays.
    #   AWC — volatile attractant, "OFF" cell (rate ∝ -dC/dt).
    #   AWA — volatile attractant, "ON" cell (rate ∝ +dC/dt).
    #   ASH — polymodal avoidance (not used for positive chemotaxis).
    ATTRACTANT_ON_NEURONS = ["ASEL", "ASER", "AWAL", "AWAR"]
    ATTRACTANT_OFF_NEURONS = ["AWCL", "AWCR"]

    def __init__(self,
                 gradient: ChemoGradient | None = None,
                 initial_head_xy: tuple[float, float] = (0.0, 0.0),
                 tonic_max_rate_hz: float = 80.0,
                 deriv_max_rate_hz: float = 160.0,
                 deriv_averaging_s: float = 1.0,
                 aerotaxis: AerotaxisSensory | None = None,
                 ):
        self.gradient = gradient or ChemoGradient()
        self.head_x, self.head_y = initial_head_xy
        self.tonic_max_rate = tonic_max_rate_hz
        self.deriv_max_rate = deriv_max_rate_hz
        # Rolling history for dC/dt
        self.deriv_window_s = deriv_averaging_s
        self._hist_t: list[float] = []
        self._hist_c: list[float] = []
        # Logging
        self.trail: list[tuple[float, float, float]] = []  # (t, x, y)
        self.last_c: float = 0.0
        self.last_dcdt: float = 0.0
        # P0 #3 — optional O2/CO2 aerotaxis overlay
        self.aerotaxis: AerotaxisSensory | None = aerotaxis
        self.aero_trail: list[dict] = []

    def update_head_position(self, x: float, y: float, t_s: float):
        """Update head position from MuJoCo body state. Called once per
        sync step. Also updates rolling dC/dt estimate."""
        self.head_x, self.head_y = x, y
        c = self.gradient.concentration(x, y)
        self._hist_t.append(t_s)
        self._hist_c.append(c)
        # Keep ~N seconds of history
        cutoff = t_s - self.deriv_window_s
        while len(self._hist_t) > 2 and self._hist_t[0] < cutoff:
            self._hist_t.pop(0)
            self._hist_c.pop(0)
        # Compute dC/dt from first and last samples in window
        if len(self._hist_t) >= 2:
            dt = self._hist_t[-1] - self._hist_t[0]
            if dt > 0:
                self.last_dcdt = (self._hist_c[-1] - self._hist_c[0]) / dt
        self.last_c = c
        self.trail.append((t_s, x, y))

        # P0 #3 — update aerotaxis sensors
        if self.aerotaxis is not None:
            self.aerotaxis.update(x, y, t_s)
            tm = self.aerotaxis.telemetry()
            self.aero_trail.append({"t": round(t_s, 3), **tm})

    def inject_into_brain(self, brain):
        """Compute Poisson rates for sensory neurons based on current
        concentration + dC/dt, and inject into the brain as Poisson
        spike trains. Called once per sync step after update_head_position.
        """
        # Skip chemotaxis injection when the gradient is a dummy
        # (aerotaxis-only scenarios set peak=0 / sigma=tiny).
        has_chemo = self.gradient.peak > 1e-6 and self.gradient.sigma > 1e-4
        if has_chemo:
            tonic_rate = self.tonic_max_rate * min(1.0, max(0.0, self.last_c))
            pos_deriv = max(0.0, self.last_dcdt)
            neg_deriv = max(0.0, -self.last_dcdt)
            ref_deriv = (self.gradient.peak / self.gradient.sigma
                         / max(0.5, self.deriv_window_s))
            if ref_deriv > 0:
                on_rate = self.deriv_max_rate * min(1.0, pos_deriv / ref_deriv)
                off_rate = self.deriv_max_rate * min(1.0, neg_deriv / ref_deriv)
            else:
                on_rate = off_rate = 0.0
            for n in self.ATTRACTANT_ON_NEURONS:
                brain.set_sensory_rate(n, tonic_rate + on_rate, weight_mv=8)
            for n in self.ATTRACTANT_OFF_NEURONS:
                brain.set_sensory_rate(n, off_rate, weight_mv=8)

        # P0 #3 — aerotaxis injection, if configured
        if self.aerotaxis is not None:
            self.aerotaxis.inject_into_brain(brain)

    def distance_to_food(self) -> float:
        dx = self.head_x - self.gradient.food_x
        dy = self.head_y - self.gradient.food_y
        return math.sqrt(dx * dx + dy * dy)

    def chemotaxis_index(self) -> dict:
        """Simple metric after a simulation: closer-to-food = higher CI.
        Pierce-Shimomura 1999's classical chemotaxis index is
        (N_attract - N_control) / N_total in spatial zones; our single-
        worm version reports fractional displacement toward food vs away.
        """
        if len(self.trail) < 2:
            return {}
        start_t, start_x, start_y = self.trail[0]
        end_t, end_x, end_y = self.trail[-1]
        dx = end_x - start_x
        dy = end_y - start_y
        total_displacement = math.sqrt(dx * dx + dy * dy)
        # Projection onto direction-to-food vector from start
        food_dx = self.gradient.food_x - start_x
        food_dy = self.gradient.food_y - start_y
        food_mag = math.sqrt(food_dx * food_dx + food_dy * food_dy) + 1e-9
        proj = (dx * food_dx + dy * food_dy) / food_mag
        return {
            "start_xy_mm": (start_x * 1e3, start_y * 1e3),
            "end_xy_mm": (end_x * 1e3, end_y * 1e3),
            "start_to_food_mm": food_mag * 1e3,
            "end_to_food_mm": self.distance_to_food() * 1e3,
            "net_displacement_mm": total_displacement * 1e3,
            "displacement_toward_food_mm": proj * 1e3,
            "CI": proj / (total_displacement + 1e-9),  # cos θ to food
        }


if __name__ == "__main__":
    # Smoke test: instantiate, run a short trajectory, compute CI
    import math
    grad = ChemoGradient(food_xy=(8e-3, 0.0))
    env = Environment(grad)
    # Simulate moving in a straight line toward food
    for i, t in enumerate(np.linspace(0, 30, 60)):
        x = (i / 60) * 8e-3
        y = 0
        env.update_head_position(x, y, float(t))
    ci = env.chemotaxis_index()
    for k, v in ci.items():
        print(f"  {k}: {v}")
    print(f"\nFinal C at head: {env.last_c:.3f}")
    print(f"Final dC/dt:     {env.last_dcdt:.3f}")
