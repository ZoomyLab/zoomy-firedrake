"""2-D SWE viscous-diffusion gate — the diffusion operator stays finite.

A minimal ``ShallowWater`` carrying the canonical full-deviatoric-stress
``diffusion_matrix_explicit`` (``div(nu h (grad u + grad u^T))``) is run at
DG0, where the diffusion form is discretised by the solver's two-point flux
(TPFA) branch.  A localised velocity blob on an otherwise still, flat sheet
is smoothed by the viscous stress.  The gate asserts the run stays finite and
bounded with non-negative cell-mean depth — i.e. the diffusion discretisation
does not inject energy or blow up.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sympy import Matrix, Max, Min, sqrt, zeros

import firedrake as fd

import zoomy_core.model.boundary_conditions as bc

sys.path.insert(0, os.path.dirname(__file__))
from _swe2d_common import SystemModelSpec, run_swe2d  # noqa: E402


class ShallowWater(SystemModelSpec):
    """Minimal 2-D SWE with an explicit viscous-diffusion stress."""

    variables = ("b", "h", "hu", "hv")
    parameters = dict(g=9.81, nu=1.0, eps=1e-2, u_max=30.0)

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

    def diffusion_matrix_explicit(self):
        # Full deviatoric stress div(nu h (grad u + grad u^T)); the [i, 1, d, e]
        # entries carry the -velocity part because the moments are hu, hv.
        import sympy as sp
        b, h, hu, hv, u, w = self.get_fields()
        nu = self.parameters.nu
        a = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        # u-momentum
        a[2, 2, 0, 0] = 2 * nu
        a[2, 1, 0, 0] = -2 * nu * u
        a[2, 2, 1, 1] = nu
        a[2, 1, 1, 1] = -nu * u
        a[2, 3, 1, 0] = nu
        a[2, 1, 1, 0] = -nu * w
        # v-momentum
        a[3, 3, 0, 0] = nu
        a[3, 1, 0, 0] = -nu * w
        a[3, 2, 0, 1] = nu
        a[3, 1, 0, 1] = -nu * u
        a[3, 3, 1, 1] = 2 * nu
        a[3, 1, 1, 1] = -2 * nu * w
        return a

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


def _velocity_blob(Q, model):
    """Still, flat sheet (h=1) with a smooth Gaussian x-momentum blob."""
    mesh = Q.function_space().mesh()
    V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=mesh.geometric_dimension)
    xy = np.asarray(fd.Function(V).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro)
    r2 = (xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2
    Q.dat.data[:, 0] = 0.0                    # b
    Q.dat.data[:, 1] = 1.0                    # h
    Q.dat.data[:, 2] = np.exp(-r2 / 0.05)     # hu
    Q.dat.data[:, 3] = 0.0                    # hv


def test_swe2d_diffusion_stays_finite():
    # Explicit (forward-Euler) diffusion carries a parabolic dt limit the
    # hyperbolic CFL calc ignores, so keep nu modest on a coarse mesh — the
    # gate is finiteness of the operator, not a stiff-diffusion stress test.
    mesh = fd.UnitSquareMesh(6, 6)
    model = ShallowWater(nu=0.1)

    out = run_swe2d(model, mesh, _velocity_blob, time_end=0.3, cfl=0.4)
    Q = out.Q

    # Finiteness / stability of the diffusion discretisation.
    assert np.all(np.isfinite(Q)), "non-finite state after diffusion run"
    assert Q[:, 1].min() >= -1e-6, f"lost positivity: h_min={Q[:, 1].min():.3e}"
    # Bounded — no blow-up (diffusion is dissipative; the IC peak is 1).
    assert np.max(np.abs(Q[:, 2:4])) < 2.0, (
        f"momentum blew up under diffusion: max|m|={np.max(np.abs(Q[:, 2:4])):.4f}")
    # Mass unchanged (no source on h, impermeable walls).
    mass = float(np.dot(Q[:, 1], out.areas))
    mass_init = float(np.dot(np.ones(mesh.num_cells()), out.areas))
    assert abs(mass - mass_init) / mass_init < 1e-6, "mass not conserved"
