"""Numerical eigenvalue / eigensystem for Firedrake via a compiled LAPACK kernel.

Firedrake's convective flux and CFL ``dt`` are TSFC-generated UFL kernels.  When
a model carries a **closed-form** spectrum (SWE), the wave speed
``max|λ(A_n)|`` lowers to a UFL expression and TSFC inlines it.  When it does
*not* (VAM has no closed form; SME's Cardano spectrum collapses ``dt``), the
solver runs with ``eigenvalues=None`` and the Riemann solver routes the wave
speed through the opaque :class:`zoomy_core.model.kernel_functions.eigensystem`
/ ``eigenvalues`` kernels — which are *numerical*, not UFL-expressible, so TSFC
cannot inline them (``NameError: eigensystem`` at flux assembly).

This module is the Firedrake analogue of dmplex's ``UserFunctions::eigensystem``
(Eigen) and numpy's ``_eigensystem_numpy`` (``np.linalg.eig``): a **compiled C
kernel that calls LAPACK ``dgeev``** (general real eigenproblem → eigenvalues +
right eigenvectors; ``L = R⁻¹`` via ``dgetrf``/``dgetri``), run per DOF by a
low-level :func:`pyop2.parloop`.  No Python callback, no ``np.linalg`` on
``.dat.data``.

Layout of the per-DOF output stack (row-major, length ``n + 2n²``) mirrors the
core contract (:meth:`SystemModel.eigensystem`)::

    [ eigenvalues λ (n),  right eigenvectors R (n·n),  left L = R⁻¹ (n·n) ]

Non-finite guard (REQ-168 addendum): LAPACK raises / returns garbage on
non-finite input, and the order-2 a-posteriori (MOOD) path evaluates the wave
speed on not-yet-corrected candidate states that are non-finite at a dry front
BY DESIGN.  A non-finite matrix (or a ``dgeev`` non-convergence) yields
``λ = +inf`` with an identity eigenbasis ``R = L = I`` — an infinite wave speed
the ``dt`` clamp / MOOD flag can act on, and a well-defined ``R|Λ|L``.
"""
from __future__ import annotations

import functools

import ufl
import firedrake as fd
from pyop2 import Kernel, parloop, READ, WRITE

__all__ = ["compute_eigenstack", "NumericalEigenSpectrum"]


# ---------------------------------------------------------------------------
# The compiled LAPACK kernel (one specialisation per system size ``n``)
# ---------------------------------------------------------------------------
def _kernel_source(n: int) -> str:
    """Raw-C ``dgeev`` kernel body for an ``n×n`` system.

    ``A`` arrives row-major (Firedrake tensor DOF layout); LAPACK is
    column-major, so the transpose-copy ``af[j*n+i] = A[i*n+j]`` presents
    Fortran order and ``dgeev``'s right eigenvectors ``VR`` are the right
    eigenvectors of ``A`` (columns).  Output is written row-major to match the
    numpy / dmplex contract.
    """
    nn = n * n
    return f"""
#include <math.h>
extern void dgeev_(const char*, const char*, const int*, double*, const int*,
                   double*, double*, double*, const int*, double*, const int*,
                   double*, const int*, int*);
extern void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
extern void dgetri_(const int*, double*, const int*, int*, double*,
                    const int*, int*);

/* fast-math-immune non-finite test: PyOP2 compiles kernels with -ffast-math,
   under which isfinite/isnan are optimised to constant true.  Inspect the IEEE
   exponent bits instead — all-ones exponent ⇒ inf or nan. */
static int zoomy_nonfinite(double x) {{
  union {{ double d; unsigned long long u; }} v;
  v.d = x;
  return (((v.u >> 52) & 0x7FFULL) == 0x7FFULL);
}}

void zoomy_eig(double *out, const double *A) {{
  const int n = {n};
  const int nn = {nn};
  int i, j;
  int finite = 1;
  for (i = 0; i < nn; i++) if (zoomy_nonfinite(A[i])) finite = 0;
  for (i = 0; i < n + 2*nn; i++) out[i] = 0.0;
  if (!finite) {{
    for (i = 0; i < n; i++) out[i] = INFINITY;      /* λ = +inf  */
    for (i = 0; i < n; i++) {{ out[n + i*n + i] = 1.0;          /* R = I */
                              out[n + nn + i*n + i] = 1.0; }}   /* L = I */
    return;
  }}
  double af[{nn}];
  for (i = 0; i < n; i++) for (j = 0; j < n; j++) af[j*n + i] = A[i*n + j];
  double wr[{n}], wi[{n}], vr[{nn}], work[8*{n}];
  int lwork = 8*n, info = 0, ldvl = 1;
  char jobvl = 'N', jobvr = 'V';
  dgeev_(&jobvl, &jobvr, &n, af, &n, wr, wi, (double*)0, &ldvl,
         vr, &n, work, &lwork, &info);
  if (info != 0) {{                                  /* no convergence */
    for (i = 0; i < n; i++) out[i] = INFINITY;
    for (i = 0; i < n; i++) {{ out[n + i*n + i] = 1.0;
                              out[n + nn + i*n + i] = 1.0; }}
    return;
  }}
  for (i = 0; i < n; i++) out[i] = wr[i];            /* real part of λ */
  /* R row-major: R[i][j] = vr[i + j*n]  (column j is eigenvector j) */
  for (i = 0; i < n; i++) for (j = 0; j < n; j++) out[n + i*n + j] = vr[i + j*n];
  /* L = R^{{-1}}: invert the column-major VR in place */
  double Rinv[{nn}], work2[{nn}];
  int ipiv[{n}], info2 = 0, info3 = 0;
  for (i = 0; i < nn; i++) Rinv[i] = vr[i];
  dgetrf_(&n, &n, Rinv, &n, ipiv, &info2);
  dgetri_(&n, Rinv, &n, ipiv, work2, &nn, &info3);
  if (info2 == 0 && info3 == 0)
    for (i = 0; i < n; i++) for (j = 0; j < n; j++)
      out[n + nn + i*n + j] = Rinv[i + j*n];
  else                                               /* defective R → L = I */
    for (i = 0; i < n; i++) out[n + nn + i*n + i] = 1.0;
}}
"""


@functools.lru_cache(maxsize=None)
def _eig_kernel(n: int) -> Kernel:
    """Cached :class:`pyop2.Kernel` (raw C) linked against LAPACK/BLAS."""
    return Kernel(_kernel_source(n), "zoomy_eig",
                  accesses=[WRITE, READ],
                  ldargs=["-llapack", "-lblas"])


def compute_eigenstack(An_fn: fd.Function, eig_fn: fd.Function) -> None:
    """Fill ``eig_fn`` with the per-DOF ``[λ, R, L]`` stack of the row-major
    matrices held in ``An_fn`` (dim ``n²`` in, ``n + 2n²`` out).

    A direct per-DOF :func:`pyop2.parloop` — one ``dgeev`` per node, no maps.
    """
    n2 = An_fn.function_space().value_size
    n = int(round(n2 ** 0.5))
    assert eig_fn.function_space().value_size == n + 2 * n * n, (
        "eig_fn must have dim n + 2n²")
    parloop(_eig_kernel(n), eig_fn.function_space().node_set,
            eig_fn.dat(WRITE), An_fn.dat(READ))


# ---------------------------------------------------------------------------
# Wave-speed field for the numerical-spectrum path
# ---------------------------------------------------------------------------
class NumericalEigenSpectrum:
    """Compiled cell wave-speed field for a model with no closed-form spectrum.

    Holds a DG(``degree``) scalar ``lam_cell`` = ``max_{n_s} max_i |λ_i(A_n)|``
    over a set of sample normals ``n_s`` — the direction-agnostic spectral
    radius of the normal-projected quasilinear matrix, refreshed from the
    current state by the compiled ``dgeev`` kernel.

    Both consumers read this ONE field:

    * the CFL ``dt`` — ``compute_dt`` samples ``lam_cell`` at every DG DOF;
    * the convective flux — with ``eigenvalues=None`` the numerical HLL/Rusanov
      collapses to local Lax-Friedrichs (:mod:`zoomy_core.fvm.riemann_solvers`
      docstring, ``wave_speed_bounds``: ``s_L,s_R = ∓ max|λ|``), whose only
      spectral input is ``max|λ|``.  The ``eigensystem``/``eigenvalues`` module
      hooks return ``lam_cell`` (restricted to the flux's ``'+'/'-'`` side) so
      the face wave speed is ``max(lam_cell('+'), lam_cell('-'))``.

    The exact per-facet-normal spectral radius is NOT representable as a cell
    Function (the facet normal is a facet quantity), so the face speed uses the
    cell bound ``max_{n_s} ρ(A_n)`` — a valid, slightly-more-diffusive LLF
    speed, consistent with the normals ``compute_dt`` already samples, and
    exact at a lake at rest (``ρ = √(gh)`` for every normal, and the LLF
    dissipation vanishes where the reconstructed jump is zero).
    """

    def __init__(self, mesh, degree, runtime_model):
        self.mesh = mesh
        self.rt = runtime_model
        self.n = int(runtime_model.n_variables)
        self.dim = int(mesh.geometric_dimension)
        n = self.n

        self._VT = fd.TensorFunctionSpace(mesh, "DG", degree, shape=(n, n))
        self._Vstack = fd.VectorFunctionSpace(mesh, "DG", degree,
                                              dim=n + 2 * n * n)
        self._Vscalar = fd.FunctionSpace(mesh, "DG", degree)
        self.An_fn = fd.Function(self._VT)
        self.eig_fn = fd.Function(self._Vstack)
        self.lam_cell = fd.Function(self._Vscalar)
        self._lam_tmp = fd.Function(self._Vscalar)

        # Sample normals: axis-aligned + diagonals (the set ``get_compute_dt``
        # already uses, so the ``dt`` bound and the flux speed agree).
        if self.dim == 1:
            self._normals = [fd.as_vector([1.0])]
        else:
            s = 1.0 / (2.0 ** 0.5)
            self._normals = [fd.as_vector([1.0, 0.0]),
                             fd.as_vector([0.0, 1.0]),
                             fd.as_vector([s, s]),
                             fd.as_vector([s, -s])]

    def _An_ufl(self, Q, Qaux, p, n_vec):
        """``A_n = Σ_d n_d · quasilinear_matrix[:, :, d]`` as a UFL matrix.

        ``runtime.quasilinear_matrix`` returns shape ``(n², dim)`` (flat
        row-major rows stacked per axis); reshape to ``(n, n)``."""
        n, dim = self.n, self.dim
        qm = self.rt.quasilinear_matrix(Q, Qaux, p)   # (n*n, dim)
        return ufl.as_matrix(
            [[sum(qm[i * n + j, d] * n_vec[d] for d in range(dim))
              for j in range(n)] for i in range(n)])

    def _maxabs_ufl(self):
        """``max_i |eig_fn[i]|`` over the first ``n`` (eigenvalue) slots."""
        return functools.reduce(
            ufl.max_value, [abs(self.eig_fn[i]) for i in range(self.n)])

    def refresh(self, Q, Qaux, p):
        """Recompute ``lam_cell`` from the current state via the ``dgeev``
        kernel: for each sample normal, interpolate ``A_n`` → run the kernel →
        accumulate the per-DOF ``max_i|λ_i|``, taking the max across normals."""
        for k, n_vec in enumerate(self._normals):
            self.An_fn.interpolate(self._An_ufl(Q, Qaux, p, n_vec))
            compute_eigenstack(self.An_fn, self.eig_fn)
            if k == 0:
                self.lam_cell.interpolate(self._maxabs_ufl())
            else:
                self._lam_tmp.interpolate(
                    ufl.max_value(self.lam_cell, self._maxabs_ufl()))
                self.lam_cell.assign(self._lam_tmp)

    # -- module-dict hooks (eigenvalues / eigensystem) ---------------------
    def _side(self, a_flat):
        """The ``'+'`` / ``'-'`` restriction carried by ``a_flat`` (the flux's
        face side), or ``None`` (a cell / exterior-facet expression)."""
        from ufl.classes import PositiveRestricted, NegativeRestricted
        for e in a_flat:
            if not hasattr(e, "ufl_operands"):
                continue
            for node in ufl.corealg.traversal.unique_pre_traversal(e):
                if isinstance(node, PositiveRestricted):
                    return "+"
                if isinstance(node, NegativeRestricted):
                    return "-"
        return None

    def _lam_restricted(self, a_flat):
        side = self._side(a_flat)
        if side == "+":
            return self.lam_cell("+")
        if side == "-":
            return self.lam_cell("-")
        return self.lam_cell

    def eigenvalues_hook(self, idx, *a_flat):
        """Module-dict ``eigenvalues(idx, *A_flat)``: idx-th eigenvalue.

        Slot 0 carries the spectral-radius bound ``lam_cell`` (so a downstream
        ``max_i|λ_i|`` recovers it); the remaining slots are 0.  ``A_flat`` is
        ignored beyond its restriction side — the numerical wave speed is the
        cell bound, not the facet-exact spectrum (see class docstring)."""
        idx = int(idx)
        if idx == 0:
            return self._lam_restricted(a_flat)
        return ufl.as_ufl(0.0)

    def eigensystem_hook(self, idx, *a_flat):
        """Module-dict ``eigensystem(idx, *A_flat)``: idx-th entry of the
        stacked ``[λ(n), R(n²), L(n²)]``.

        Slot 0 = ``lam_cell`` (the wave speed the LLF flux reads); other
        eigenvalue slots 0; ``R`` / ``L`` = identity.  The identity eigenbasis
        is contract completeness only — the numerical-spectrum HLL/Rusanov is
        LLF and never forms ``R|Λ|L`` (it requests only idx < n)."""
        idx = int(idx)
        n = self.n
        if idx < n:
            return self._lam_restricted(a_flat) if idx == 0 else ufl.as_ufl(0.0)
        r = idx - n
        i, j = divmod(r % (n * n), n)         # R block then L block
        return ufl.as_ufl(1.0 if i == j else 0.0)
