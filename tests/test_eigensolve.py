"""Unit gate for the compiled LAPACK eigensolver (:mod:`zoomy_firedrake.eigensolve`).

Three levels:

* the raw ``dgeev`` par_loop kernel on a KNOWN matrix — SWE at rest, where
  ``max|λ(A_n)| = √(g h)`` (the REQ-168 [A] check: h=2 → 4.429; the retired
  Gershgorin row-sum gave 39.24), plus the full ``[λ, R, L]`` contract
  (``R Λ L = A``, ``L R = I``) and the non-finite guard;
* :class:`NumericalEigenSpectrum` on a REAL SWE runtime with
  ``eigenvalues=None`` — the ``A_n`` the solver actually lowers, reduced to the
  same ``√(g h)`` cell wave speed by the compiled kernel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sympy import Matrix, Max, sqrt, zeros

import firedrake as fd

sys.path.insert(0, os.path.dirname(__file__))
from _swe2d_common import SystemModelSpec  # noqa: E402

from zoomy_firedrake.eigensolve import (  # noqa: E402
    compute_eigenstack, NumericalEigenSpectrum)

G, H = 9.81, 2.0


def _swe_An_at_rest():
    """Normal-projected quasilinear matrix of 2-D SWE ``[b,h,hu,hv]`` at rest
    (h=H, u=v=0), n=(1,0): rows b,h,hu,hv."""
    A = np.zeros((4, 4))
    A[1, 2] = 1.0            # ∂ₓ(hu) in the mass row
    A[2, 0] = G * H         # bed-slope coupling g h ∂ₓb
    A[2, 1] = G * H         # pressure ∂(g h²/2)/∂h = g h
    return A


def test_dgeev_kernel_swe_at_rest():
    """max|λ| = √(gH) on the SWE-at-rest A_n (REQ-168 [A])."""
    mesh = fd.UnitSquareMesh(2, 2, quadrilateral=True)
    n = 4
    An = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 0, shape=(n, n)))
    eig = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=n + 2 * n * n))
    An.dat.data[:] = _swe_An_at_rest()[None, :, :]

    compute_eigenstack(An, eig)

    lam = eig.dat.data_ro[0][:n]
    assert np.max(np.abs(lam)) == \
        __import__("pytest").approx(np.sqrt(G * H), rel=1e-12)
    # spectrum: {±c, 0, 0}
    assert sorted(np.round(lam, 8)) == sorted(
        np.round([np.sqrt(G * H), -np.sqrt(G * H), 0.0, 0.0], 8))


def test_dgeev_kernel_full_contract():
    """[λ, R, L] honours R Λ L = A and L R = I on a non-degenerate matrix."""
    import pytest
    mesh = fd.UnitSquareMesh(1, 1, quadrilateral=True)
    n = 4
    A = np.array([[3., 1., 0., 0.], [1., 2., 1., 0.],
                  [0., 1., 4., 1.], [0., 0., 1., 5.]])
    An = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 0, shape=(n, n)))
    eig = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=n + 2 * n * n))
    An.dat.data[:] = A[None, :, :]

    compute_eigenstack(An, eig)

    o = eig.dat.data_ro[0]
    lam, R, L = o[:n], o[n:n + n * n].reshape(n, n), o[n + n * n:].reshape(n, n)
    assert sorted(np.round(lam, 8)) == sorted(
        np.round(np.linalg.eigvals(A).real, 8))
    assert np.linalg.norm(L @ R - np.eye(n)) < 1e-12
    assert np.linalg.norm(R @ np.diag(lam) @ L - A) < 1e-12


def test_dgeev_kernel_nonfinite_guard():
    """A non-finite matrix → λ = +inf, R = L = I (REQ-168 inf contract)."""
    mesh = fd.UnitSquareMesh(1, 1, quadrilateral=True)
    n = 4
    An = fd.Function(fd.TensorFunctionSpace(mesh, "DG", 0, shape=(n, n)))
    eig = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=n + 2 * n * n))
    bad = _swe_An_at_rest()
    bad[0, 0] = np.nan
    An.dat.data[:] = bad[None, :, :]

    compute_eigenstack(An, eig)

    o = eig.dat.data_ro[0]
    assert np.all(np.isinf(o[:n]))
    assert np.allclose(o[n:n + n * n].reshape(n, n), np.eye(n))
    assert np.allclose(o[n + n * n:].reshape(n, n), np.eye(n))


class _SWE(SystemModelSpec):
    """Minimal 2-D SWE for the runtime-level spectrum check."""

    variables = ("b", "h", "hu", "hv")
    parameters = dict(g=9.81, eps=1e-2)

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
        bed = sp.MutableDenseNDimArray.zeros(4, 4, 2)   # (n_eq, n_state, n_dim)
        bed[2, 0, 0] = g * h                            # g h ∂ₓ b
        bed[3, 0, 1] = g * h                            # g h ∂_y b
        return bed

    def boundary_conditions(self):
        import zoomy_core.model.boundary_conditions as bc
        return [bc.Wall(tag="wall", momentum_field_indices=[[2, 3]],
                        permeability=0.0, wall_slip=1.0)]


def test_spectrum_swe_at_rest_via_runtime():
    """NumericalEigenSpectrum on the real SWE runtime (eigenvalues=None) →
    cell wave speed = √(gH), computed by the compiled kernel from the A_n the
    solver actually lowers."""
    import pytest
    from zoomy_core.numerics import NumericalSystemModel
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
    from zoomy_core.transformation.to_ufl import UFLRuntimeModel

    sm = _SWE()
    sm.eigenvalues = None                   # force the numerical path
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeHLL)
    rt = UFLRuntimeModel.from_nsm(nsm)

    mesh = fd.UnitSquareMesh(3, 3, quadrilateral=True)
    spec = NumericalEigenSpectrum(mesh, 0, rt)

    n = rt.n_variables
    Q = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=n))
    Qaux = fd.Function(fd.VectorFunctionSpace(
        mesh, "DG", 0, dim=max(rt.n_aux_variables, 1)))
    Q.dat.data[:] = np.array([0.0, H, 0.0, 0.0])[None, :]     # b,h,hu,hv at rest

    spec.refresh(Q, Qaux, rt.parameters)

    assert float(np.max(spec.lam_cell.dat.data_ro)) == \
        pytest.approx(np.sqrt(G * H), rel=1e-9)
    assert float(np.min(spec.lam_cell.dat.data_ro)) == \
        pytest.approx(np.sqrt(G * H), rel=1e-9)


def test_end_to_end_numerical_spectrum_flux():
    """A 2-D SWE dam break with ``eigenvalues=None`` — the case that died with
    ``NameError: eigensystem`` at flux assembly — now runs finite: positive
    height, mass conserved (the compiled ``dgeev`` LLF wave speed feeds the
    convective flux via the ``eigensystem``/``eigenvalues`` module hooks)."""
    import pytest
    from _swe2d_common import run_swe2d, cell_areas, cell_centroids

    def _dam(Q, model):
        mesh = Q.function_space().mesh()
        V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=mesh.geometric_dimension)
        xy = np.asarray(fd.Function(V).interpolate(
            fd.SpatialCoordinate(mesh)).dat.data_ro)
        Q.dat.data[:, 0] = 0.0                              # b
        Q.dat.data[:, 1] = np.where(xy[:, 0] < 0.5, 1.0, 0.1)  # h (wet everywhere)
        Q.dat.data[:, 2] = 0.0
        Q.dat.data[:, 3] = 0.0

    mesh = fd.UnitSquareMesh(8, 8, quadrilateral=True)
    xy, areas = cell_centroids(mesh), cell_areas(mesh)
    mass_init = float(np.dot(np.where(xy[:, 0] < 0.5, 1.0, 0.1), areas))

    model = _SWE()
    model.eigenvalues = None                # force the numerical-spectrum path
    out = run_swe2d(model, mesh, _dam, time_end=0.05, cfl=0.4)

    h = out.Q[:, 1]
    assert np.all(np.isfinite(out.Q))       # was NameError → now finite
    assert float(h.min()) >= -1e-9          # positivity
    mass = float(np.dot(h, areas))
    assert abs(mass - mass_init) / mass_init < 1e-9   # mass conservation
