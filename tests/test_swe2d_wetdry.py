"""2-D SWE wet/dry gate — dam break onto a DRY bed.

A minimal ``ShallowWater`` (flat bed, no friction, no diffusion) authored on
the canonical NSM path is run with ``PositiveNonconservativeHLL`` at DG0.  The
gate asserts the two robustness properties the removed legacy
``test_swe2d_dam_break_wetdry`` used to cover, but on the CURRENT
``[b, h, hu, hv]`` state (no ``aux[b, bx, by]`` layout):

- **positivity**: the cell-mean water height stays ``>= -1e-6`` even as the
  bore advances over the initially dry half (the positivity-preserving
  Riemann solver + the ``update_variables`` dry-state clamp).
- **mass conservation**: ``h`` carries no source and the walls are
  impermeable, so total water mass is conserved to solver tolerance.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sympy import Matrix, Max, Min, sqrt, zeros

import firedrake as fd

import zoomy_core.model.boundary_conditions as bc

# Repo pytest.ini forces --import-mode=importlib, which does not add the test
# dir to sys.path; make the shared sibling harness importable.
sys.path.insert(0, os.path.dirname(__file__))
from _swe2d_common import SystemModelSpec, run_swe2d  # noqa: E402


class ShallowWater(SystemModelSpec):
    """Minimal 2-D SWE — flux + hydrostatic pressure + wet/dry handling."""

    variables = ("b", "h", "hu", "hv")
    parameters = dict(g=9.81, eps=1e-2, u_max=30.0)

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


def _dam_break_dry(Q, model):
    """Left half wet (h=1), right half DRY (h=0); flat bed, at rest."""
    mesh = Q.function_space().mesh()
    V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=mesh.geometric_dimension)
    xy = np.asarray(fd.Function(V).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro)
    h0 = np.where(xy[:, 0] < 0.5, 1.0, 0.0)
    Q.dat.data[:, 0] = 0.0          # b
    Q.dat.data[:, 1] = h0           # h
    Q.dat.data[:, 2] = 0.0          # hu
    Q.dat.data[:, 3] = 0.0          # hv


def test_swe2d_dam_break_wetdry_positivity_and_mass():
    mesh = fd.UnitSquareMesh(10, 10)
    model = ShallowWater()

    # Initial mass (before the run), on the same DG0 weights.
    from _swe2d_common import cell_centroids, cell_areas
    xy, areas = cell_centroids(mesh), cell_areas(mesh)
    h_init = np.where(xy[:, 0] < 0.5, 1.0, 0.0)
    mass_init = float(np.dot(h_init, areas))

    out = run_swe2d(model, mesh, _dam_break_dry, time_end=0.1, cfl=0.4)
    h = out.Q[:, 1]

    # Positivity: cell-mean water height never goes meaningfully negative.
    assert h.min() >= -1e-6, f"positivity violated: h_min={h.min():.3e}"
    # Finiteness of the whole state.
    assert np.all(np.isfinite(out.Q)), "non-finite state after run"
    # The bore must have advanced into the dry half (a real, non-trivial run).
    wet_right = h[out.centroids[:, 0] > 0.5]
    assert wet_right.max() > 1e-3, "bore never reached the dry half"

    # Mass conservation: h has no source, walls are impermeable.
    mass_final = float(np.dot(h, out.areas))
    rel = abs(mass_final - mass_init) / mass_init
    assert rel < 1e-6, f"mass not conserved: rel change {rel:.3e}"
