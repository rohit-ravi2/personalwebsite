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

    def inject_into_brain(self, brain):
        """Compute Poisson rates for sensory neurons based on current
        concentration + dC/dt, and inject into the brain as Poisson
        spike trains. Called once per sync step after update_head_position.
        """
        # Tonic rate — proportional to absolute concentration (normalized)
        tonic_rate = self.tonic_max_rate * min(1.0, max(0.0, self.last_c))
        # Derivative-coupled rate — ON cells fire on positive dC/dt
        pos_deriv = max(0.0, self.last_dcdt)
        neg_deriv = max(0.0, -self.last_dcdt)
        # Normalise against sigma and peak — a typical dC/dt near the
        # gradient peak is peak / sigma / move_timescale
        ref_deriv = (self.gradient.peak / self.gradient.sigma
                     / max(0.5, self.deriv_window_s))
        on_rate = self.deriv_max_rate * min(1.0, pos_deriv / ref_deriv)
        off_rate = self.deriv_max_rate * min(1.0, neg_deriv / ref_deriv)

        # Inject via persistent-rate Poisson (no Brian2 recompile).
        # ASE and AWA fire on tonic + positive dC/dt (ON cells),
        # AWC fires on negative dC/dt (OFF cells).
        for n in self.ATTRACTANT_ON_NEURONS:
            brain.set_sensory_rate(n, tonic_rate + on_rate, weight_mv=8)
        for n in self.ATTRACTANT_OFF_NEURONS:
            brain.set_sensory_rate(n, off_rate, weight_mv=8)

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
