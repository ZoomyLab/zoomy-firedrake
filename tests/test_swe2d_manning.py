"""2-D SWE Manning-friction gate — source term stability + momentum damping.

A minimal ``ShallowWater`` with a Manning bed-friction ``source_explicit``
(KP-desingularised ``hinv``, exactly the canonical malpasset form) is run
twice from the SAME uniform-flow initial condition — once with friction
(``n>0``) and once frictionless (``n=0``).  The gate asserts:

- **stability**: the friction run stays finite with non-negative cell-mean
  depth (the explicit friction source does not blow the solve up);
- **damping**: friction removes kinetic energy, so the friction run's final
  momentum is strictly below both its own initial value and the frictionless
  run's final momentum.  Comparing the two runs isolates the friction source
  from wall reflection / numerical diffusion (shared by both).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sympy import Matrix, Max, Min, Rational, sqrt, zeros

import firedrake as fd

import zoomy_core.model.boundary_conditions as bc

sys.path.insert(0, os.path.dirname(__file__))
from _swe2d_common import SystemModelSpec, run_swe2d  # noqa: E402


class ShallowWater(SystemModelSpec):
    """Minimal 2-D SWE with Manning bed friction as an explicit source."""

    variables = ("b", "h", "hu", "hv")
    parameters = dict(g=9.81, n=0.033, eps=1e-2, u_max=30.0)

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
        import sympy as sp
        h, g = self.variables.h, self.parameters.g
        bed = sp.MutableDenseNDimArray.zeros(4, 4, 2)
        bed[2, 0, 0] = g * h
        bed[3, 0, 1] = g * h
        return bed

    def source_explicit(self):
        b, h, hu, hv, u, w = self.get_fields()
        p = self.parameters
        speed = sqrt(u ** 2 + w ** 2)
        rate = -p.g * p.n ** 2 * speed / Max(h, p.eps) ** Rational(1, 3)
        return Matrix([0, 0, rate * u, rate * w])

    def eigenvalues(self):
        import sympy as sp
        b, h, hu, hv, u, w = self.get_fields()
        p, normal = self.parameters, self.normal
        normal_velocity = u * normal.n0 + w * normal.n1
        wave = sqrt(p.g * Max(h, p.eps))
        dry = sp.Function("conditional")
        return Matrix([
            dry(h > p.eps, e, sp.S.Zero)
            for e in (sp.S.Zero, normal_velocity,
                      normal_velocity - wave, normal_velocity + wave)])

    def update_variables(self):
        import sympy as sp
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


def _uniform_flow(Q, model):
    """Uniform depth h=1 flowing in +x (hu=1); flat bed."""
    Q.dat.data[:, 0] = 0.0    # b
    Q.dat.data[:, 1] = 1.0    # h
    Q.dat.data[:, 2] = 1.0    # hu
    Q.dat.data[:, 3] = 0.0    # hv


def _momentum_norm(Q):
    return float(np.sqrt(np.sum(Q[:, 2] ** 2 + Q[:, 3] ** 2)))


def test_swe2d_manning_friction_damps_momentum():
    mesh = fd.UnitSquareMesh(6, 6)
    t_end = 1.0
    n_cells = mesh.num_cells()
    mom_init = float(np.sqrt(n_cells))  # hu=1, hv=0 everywhere

    fric = run_swe2d(ShallowWater(n=0.15), mesh, _uniform_flow,
                     time_end=t_end, cfl=0.4)
    free = run_swe2d(ShallowWater(n=0.0), mesh, _uniform_flow,
                     time_end=t_end, cfl=0.4)

    # Stability of the friction run.
    assert np.all(np.isfinite(fric.Q)), "non-finite state with friction"
    assert fric.Q[:, 1].min() >= -1e-6, "friction run lost positivity"

    mom_fric = _momentum_norm(fric.Q)
    mom_free = _momentum_norm(free.Q)

    # Friction damps momentum below its initial value ...
    assert mom_fric < mom_init, (
        f"friction did not damp momentum: {mom_fric:.4f} >= {mom_init:.4f}")
    # ... and demonstrably below the frictionless run (isolates the source).
    assert mom_fric < mom_free, (
        f"friction run momentum {mom_fric:.4f} not below frictionless "
        f"{mom_free:.4f}")
