"""REQ-185 Firedrake gate — a TIME-dependent ``source`` stays LIVE across the
reused weak form.

Firedrake assembles the source weak form ONCE and reuses it every step, so the
``time`` the operator sees must be a live ``fd.Constant`` the solver refreshes
each step (``firedrake_solver.py``: ``sim_time_const`` + ``step`` publishes it),
NOT a value baked at setup.  This gate discriminates directly:

  a still unit pond (h=1, no flow, walls) with an IMPLICIT ``source`` that rains
  into the continuity row ONLY while ``t < T_rain``.  Total water volume must
  RISE during the rain and be FLAT afterwards.  A source frozen at ``t=0`` (the
  pre-REQ-185 bug) would rain forever -> the volume would keep climbing.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import sympy as sp
from sympy import Matrix, Max, Min, sqrt, zeros

import firedrake as fd

import zoomy_core.model.boundary_conditions as bc

sys.path.insert(0, os.path.dirname(__file__))
from _swe2d_common import SystemModelSpec, run_swe2d  # noqa: E402


class RainSWE(SystemModelSpec):
    """2-D SWE whose IMPLICIT ``source`` rains into continuity while t < T_rain."""

    variables = ("b", "h", "hu", "hv")
    parameters = dict(g=9.81, eps=1e-2, u_max=30.0,
                      rain_rate=0.5, T_rain=0.2, ramp=0.01)

    def get_fields(self):
        v, p = self.variables, self.parameters
        hinv = sqrt(2) * v.h / sqrt(v.h ** 4 + Max(v.h, p.eps) ** 4)
        return v.b, v.h, v.hu, v.hv, v.hu * hinv, v.hv * hinv

    def flux(self):
        b, h, hu, hv, u, w = self.get_fields()
        f = zeros(4, 2)
        f[1, 0], f[1, 1] = hu, hv
        f[2, 0], f[2, 1] = hu * u, hu * w
        f[3, 0], f[3, 1] = hv * u, hv * w
        return f

    def hydrostatic_pressure(self):
        h, g = self.variables.h, self.parameters.g
        p = zeros(4, 2)
        p[2, 0] = g * h ** 2 / 2
        p[3, 1] = g * h ** 2 / 2
        return p

    def nonconservative_matrix(self):
        h, g = self.variables.h, self.parameters.g
        bed = sp.MutableDenseNDimArray.zeros(4, 4, 2)
        bed[2, 0, 0] = g * h
        bed[3, 0, 1] = g * h
        return bed

    def source(self):
        # REQ-185: the rain rate is a direct function of TIME ``t``, into the
        # continuity (h) row only.  A frozen-at-0 source rains forever.
        # The time gate is built from Max/Min (UFL-lowerable, and survives
        # SystemModel assembly) -- a bare ``sp.Function("conditional")`` in the
        # source slot collapses to a symbol.  gate ~ 1 while t < T_rain, 0 after
        # (sharp ramp of width ``ramp``).
        t, p = self.time, self.parameters
        gate = Max(sp.S.Zero, Min(sp.S.One, (p.T_rain - t) / p.ramp))
        return Matrix([0, p.rain_rate * gate, 0, 0])

    # NOTE: no symbolic ``eigenvalues`` -> the NSM uses the compiled numerical
    # (dgeev) wave-speed spectrum for the CFL dt.  (The symbolic-eigenvalue CFL
    # path evaluates model.eigenvalues pointwise, which is a separate concern.)

    def update_variables(self):
        v, p = self.variables, self.parameters
        cap = Max(v.h - p.eps, sp.S.Zero) * p.u_max
        clamp = lambda q: Max(-cap, Min(q, cap))
        return Matrix([v.b, v.h, clamp(v.hu), clamp(v.hv)])

    def reconstruction_variables(self):
        v = self.variables
        return Matrix([v.b, v.b + v.h, v.hu, v.hv])

    def boundary_conditions(self):
        return [bc.Wall(tag="wall", momentum_field_indices=[[2, 3]],
                        permeability=0.0, wall_slip=1.0)]


def _still_pond(Q, model):
    Q.dat.data[:, 0] = 0.0   # b (flat bed)
    Q.dat.data[:, 1] = 1.0   # h (still unit pond)
    Q.dat.data[:, 2] = 0.0   # hu
    Q.dat.data[:, 3] = 0.0   # hv


def _volume(res):
    """Total water volume = sum_cells h * cell_area."""
    return float(np.sum(res.Q[:, 1] * res.areas))


def test_rain_source_volume_plateaus_after_T_rain():
    """A time-gated volumetric source drives total volume monotonically while
    t < T_rain, then it must PLATEAU.

    |V(t) - V0| grows as rain_rate * min(t, T_rain) on the unit pond (V0 = 1),
    reaching ~0.5 * 0.2 = 0.10 by t = T_rain = 0.2, then FLAT.  The assertion is
    sign-agnostic on purpose: what REQ-185 must prove is that the source sees
    LIVE time, i.e. the gate CLOSES at T_rain.  A source frozen at t=0 keeps the
    gate open, so |V(0.6) - V0| would keep growing past |V(0.4) - V0| (never
    flat).  (The direction of the volume change follows the solver's implicit
    continuity-source sign convention and is not what this gate tests.)
    """
    mk_model = lambda: RainSWE()
    mk_mesh = lambda: fd.UnitSquareMesh(8, 8)
    V0 = 1.0  # h = 1 on the unit square

    during = _volume(run_swe2d(mk_model(), mk_mesh(), _still_pond,
                               time_end=0.10, cfl=0.4))   # t < T_rain
    post1 = _volume(run_swe2d(mk_model(), mk_mesh(), _still_pond,
                              time_end=0.40, cfl=0.4))     # t > T_rain
    post2 = _volume(run_swe2d(mk_model(), mk_mesh(), _still_pond,
                              time_end=0.60, cfl=0.4))     # later, t > T_rain

    # (1) the time-gated source is ACTIVE before T_rain: volume moved from V0.
    assert abs(during - V0) > 1e-3, f"source inactive during rain: {during} ~= {V0}"
    # (2) it kept driving the volume up to T_rain (post1 further from V0).
    assert abs(post1 - V0) > abs(during - V0) + 1e-3, (
        f"volume stopped moving before T_rain: |post1-V0|={abs(post1-V0):.4f} "
        f"<= |during-V0|={abs(during-V0):.4f}")
    # (3) DECISIVE: FLAT after T_rain -> the source time is LIVE (the gate closed
    #     at T_rain).  A frozen-at-0 source would give |post2-post1| ~ 0.10.
    assert abs(post2 - post1) < 5e-3, (
        f"volume not flat after T_rain: post1={post1:.4f} post2={post2:.4f} "
        f"(a frozen-time bug would move it ~{0.5 * (0.6 - 0.4):.2f} more)")
