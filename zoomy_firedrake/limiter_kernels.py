"""Compiled per-cell kernels for the Firedrake stabilisation hot-path.

The DG(1) limiter / positivity / MOOD post-processing steps that ``step()`` and
``_step_with_mood()`` run every timestep were pure numpy on ``Function.dat.data``
(cell-mean reductions, θ-scaling, ``exp`` damping, troubled-cell detection).
Numpy on ``.dat.data`` pulls the whole time loop back into the Python
interpreter, bypasses PyOP2's halo dirty-tracking (the root of the
``firedrake_compat`` sub-view bugs) and does not vectorise across cells.

Every routine here is a low-level :func:`pyop2.parloop` over ``mesh.cell_set``
— the same idiom as :mod:`zoomy_firedrake.eigensolve` — that runs a raw-C
kernel per cell, reading/writing DG DOFs through the space's
``cell_node_map()``.  For a ``VectorFunctionSpace(..., dim=nc)`` accessed
through its cell-node map the local kernel argument is the flat, node-major
block ``q[a*nc + c]`` (node ``a``, component ``c``) — verified against the
numpy layout before wiring in.

The arithmetic is a **bit-faithful transcription** of the numpy it replaces:
same cell-mean (``Σ q / nn``), same θ formula, same ``mean + θ (q − mean)``
rescale, same ``mean < −tol`` troubled test.  Only ``exp`` differs by a libm
ULP (C ``exp`` vs ``numpy.exp``).
"""
from __future__ import annotations

import functools

import firedrake as fd
from pyop2 import Kernel, parloop, READ, WRITE, RW, MIN, MAX

__all__ = [
    "zhang_shu_positivity",
    "extract_component",
    "copy_component",
    "copy_component_v2v",
    "cell_mean_finite_or_inf",
    "mood_troubled_indicator",
    "FusedVertexLimiter",
]


# fast-math-immune non-finite test (PyOP2 compiles with -ffast-math, under
# which isfinite/isnan fold to constants) — inspect the IEEE exponent bits.
_NONFINITE = """
static int zoomy_nonfinite(double x) {
  union { double d; unsigned long long u; } v;
  v.d = x;
  return (((v.u >> 52) & 0x7FFULL) == 0x7FFULL);
}
"""


# ---------------------------------------------------------------------------
# Zhang-Shu positivity θ-scaling (replaces the numpy body of
# ``_apply_positivity_scaling``)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _pp_kernel(nn: int, nc: int, hi: int) -> Kernel:
    return Kernel(f"""
void zoomy_pp(double *q) {{
  const int nn = {nn}, nc = {nc}, hi = {hi};
  double mean = 0.0, hmin = q[hi];
  for (int a = 0; a < nn; a++) {{
    double v = q[a*nc + hi];
    mean += v;
    if (v < hmin) hmin = v;
  }}
  mean /= nn;
  double theta;
  if (hmin < 0.0) {{
    if (mean > 0.0) {{
      double denom = mean - hmin;
      double d = (denom > 0.0) ? denom : 1.0;
      double t = mean / d;
      theta = (t < 1.0) ? t : 1.0;
    }} else theta = 0.0;
  }} else theta = 1.0;
  if (theta >= 1.0) return;                       /* cell already positive */
  for (int a = 0; a < nn; a++)
    q[a*nc + hi] = mean + theta*(q[a*nc + hi] - mean);
}}""", "zoomy_pp", accesses=[RW])


def zhang_shu_positivity(Q: fd.Function, h_idx: int) -> None:
    """Per-cell Zhang-Shu θ-scaling of component ``h_idx`` of ``Q`` (DG1).

    ``h ← h̄ + θ (h − h̄)`` with ``θ = min(1, h̄/(h̄ − h_min))`` so every node
    is ``≥ 0`` while the cell mean (and total mass) is unchanged.  Cells whose
    mean is already ``≤ 0`` collapse to their mean (θ=0).  A cell that is
    already positive (θ≥1) is left bit-for-bit untouched.
    """
    V = Q.function_space()
    cnm = V.cell_node_map()
    parloop(_pp_kernel(cnm.arity, V.value_size, int(h_idx)),
            V.mesh().cell_set, Q.dat(RW, cnm))


# ---------------------------------------------------------------------------
# Compiled component copy / extract (replace the numpy ``dat.data`` slices of
# ``firedrake_compat.safe_{extract,assign}_component`` — a par_loop keeps
# PyOP2's halo dirty-tracking, unlike a raw ``.dat.data`` write)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _extract_kernel(nn: int, nc: int, ci: int) -> Kernel:
    return Kernel(f"""
void zoomy_cextract(double *s, const double *q) {{
  const int nn = {nn}, nc = {nc}, ci = {ci};
  for (int a = 0; a < nn; a++) s[a] = q[a*nc + ci];
}}""", "zoomy_cextract", accesses=[WRITE, READ])


@functools.lru_cache(maxsize=None)
def _copy_kernel(nn: int, nc: int, ci: int) -> Kernel:
    return Kernel(f"""
void zoomy_ccopy(double *q, const double *s) {{
  const int nn = {nn}, nc = {nc}, ci = {ci};
  for (int a = 0; a < nn; a++) q[a*nc + ci] = s[a];
}}""", "zoomy_ccopy", accesses=[RW, READ])


def extract_component(scalar: fd.Function, vec: fd.Function, ci: int) -> None:
    """Copy component ``ci`` of vector ``vec`` into scalar ``scalar`` (a
    compiled equivalent of ``scalar.dat.data[:] = vec.dat.data_ro[:, ci]``)."""
    V = vec.function_space()
    parloop(_extract_kernel(V.cell_node_map().arity, V.value_size, int(ci)),
            V.mesh().cell_set,
            scalar.dat(WRITE, scalar.function_space().cell_node_map()),
            vec.dat(READ, V.cell_node_map()))


def copy_component(vec: fd.Function, scalar: fd.Function, ci: int) -> None:
    """Write scalar ``scalar`` into component ``ci`` of vector ``vec``, leaving
    the other components untouched (compiled equivalent of
    ``vec.dat.data[:, ci] = scalar.dat.data_ro[:]``)."""
    V = vec.function_space()
    parloop(_copy_kernel(V.cell_node_map().arity, V.value_size, int(ci)),
            V.mesh().cell_set,
            vec.dat(RW, V.cell_node_map()),
            scalar.dat(READ, scalar.function_space().cell_node_map()))


@functools.lru_cache(maxsize=None)
def _copy_v2v_kernel(nn: int, nc: int, ci: int) -> Kernel:
    return Kernel(f"""
void zoomy_vcopy(double *dst, const double *src) {{
  const int nn = {nn}, nc = {nc}, ci = {ci};
  for (int a = 0; a < nn; a++) dst[a*nc + ci] = src[a*nc + ci];
}}""", "zoomy_vcopy", accesses=[RW, READ])


def copy_component_v2v(dst: fd.Function, src: fd.Function, ci: int) -> None:
    """Copy component ``ci`` of vector ``src`` into component ``ci`` of vector
    ``dst`` (same function space), leaving ``dst``'s other components untouched
    — compiled equivalent of ``dst.dat.data[:, ci] = src.dat.data_ro[:, ci]``."""
    V = dst.function_space()
    parloop(_copy_v2v_kernel(V.cell_node_map().arity, V.value_size, int(ci)),
            V.mesh().cell_set,
            dst.dat(RW, V.cell_node_map()),
            src.dat(READ, src.function_space().cell_node_map()))


# ---------------------------------------------------------------------------
# Fused vector Kuzmin vertex-based limiter (replaces the per-component
# ``for i: extract_component; VertexBasedLimiter.apply; copy_component`` loop
# of ``_apply_slope_limiter``).
#
# Firedrake's :class:`~firedrake.VertexBasedLimiter` is scalar-only: limiting a
# ``VectorFunctionSpace(dim=nc)`` state fires, PER component, an ``extract``
# par_loop + an ``assemble`` + a mass ``solve`` (centroids) + two ``assign``s +
# a min/max par_loop + a limit par_loop + a ``copy`` par_loop — ``O(nc)`` launch
# floods where the arithmetic is a handful of flops per cell.
#
# This fuses the whole thing into **three cell-set par_loops + two assigns per
# call, independent of ``nc``**, looping the components INSIDE each C kernel:
#   1. ``_centroid_kernel``  — per-cell arithmetic mean of the DG1 nodal values
#      → a ``VectorFunctionSpace(DG,0)`` centroid (all components at once).  On a
#      P1 simplex the cell mean IS the vertex arithmetic mean, so this equals
#      Firedrake's L2-projection centroid to a libm ULP (verified ≤1e-15).
#   2. ``_minmax_kernel``    — scatter each cell centroid into the ``CG1`` vertex
#      max/min bounds (``MIN``/``MAX`` reduction over cells sharing a vertex,
#      halo-safe — the same reduction Firedrake's ``compute_bounds`` par_loop
#      does, just batched over components).
#   3. ``_limit_kernel``     — Kuzmin's per-vertex α and the mean-preserving
#      ``qavg + α(q − qavg)`` rescale, for every ACTIVE (non-excluded) component;
#      excluded components (e.g. bathymetry) are never written — bit-preserved.
# The α recurrence transcribes Firedrake's loopy kernel branch-for-branch, so
# the limited field matches the per-component path to roundoff (≤1e-15/cell).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _centroid_kernel(nq: int, nc: int) -> Kernel:
    return Kernel(f"""
void zoomy_centroid(double *cbar, const double *q) {{
  const int nq = {nq}, nc = {nc};
  for (int c = 0; c < nc; c++) {{
    double s = 0.0;
    for (int a = 0; a < nq; a++) s += q[a*nc + c];
    cbar[c] = s / nq;
  }}
}}""", "zoomy_centroid", accesses=[WRITE, READ])


@functools.lru_cache(maxsize=None)
def _minmax_kernel(nq: int, nc: int) -> Kernel:
    return Kernel(f"""
void zoomy_minmax(double *maxq, double *minq, const double *cbar) {{
  const int nq = {nq}, nc = {nc};
  for (int a = 0; a < nq; a++)
    for (int c = 0; c < nc; c++) {{
      double v = cbar[c];
      if (v > maxq[a*nc + c]) maxq[a*nc + c] = v;
      if (v < minq[a*nc + c]) minq[a*nc + c] = v;
    }}
}}""", "zoomy_minmax", accesses=[MAX, MIN, READ])


@functools.lru_cache(maxsize=None)
def _limit_kernel(nq: int, nc: int, active: tuple) -> Kernel:
    # One unrolled block per active component; the α recurrence mirrors
    # Firedrake's ``_limit_kernel`` (fmin(alpha, fmin(1, ...)) on the same
    # branch it selects), so the selected value is computed identically.
    blocks = ""
    for c in active:
        blocks += f"""
  {{
    const int c = {c};
    double qavg = cbar[c];
    double alpha = 1.0;
    for (int i = 0; i < nq; i++) {{
      double qi = q[i*nc + c];
      if (qi > qavg)
        alpha = fmin(alpha, fmin(1.0, (qmax[i*nc + c] - qavg)/(qi - qavg)));
      else if (qi < qavg)
        alpha = fmin(alpha, fmin(1.0, (qavg - qmin[i*nc + c])/(qavg - qi)));
    }}
    for (int ii = 0; ii < nq; ii++)
      q[ii*nc + c] = qavg + alpha*(q[ii*nc + c] - qavg);
  }}"""
    return Kernel(f"""
void zoomy_limit(double *q, const double *cbar,
                 const double *qmax, const double *qmin) {{
  const int nq = {nq}, nc = {nc};{blocks}
}}""", "zoomy_limit", accesses=[RW, READ, READ, READ])


class FusedVertexLimiter:
    """Kuzmin vertex-based slope limiter for a DG(1) ``VectorFunctionSpace``,
    applied to all non-excluded components in three ``pyop2.parloop`` launches
    (plus two ``assign``s) per call — independent of the component count.

    A fused, halo-safe transcription of the per-component
    :class:`firedrake.VertexBasedLimiter` loop it replaces; results match to
    ~1e-15 per cell (arithmetic-mean vs L2-projection centroid, one libm ULP).
    Scratch fields (DG0 centroids, CG1 vertex bounds) are allocated once and
    reused; cache the instance per state (it binds to one function space).
    """

    def __init__(self, V: fd.FunctionSpace):
        mesh = V.mesh()
        self.V = V
        self.nc = V.value_size
        self.nq = V.cell_node_map().arity
        self.P0 = fd.VectorFunctionSpace(mesh, "DG", 0, dim=self.nc)
        self.P1CG = fd.VectorFunctionSpace(mesh, "CG", 1, dim=self.nc)
        self.centroids = fd.Function(self.P0)
        self.max_field = fd.Function(self.P1CG)
        self.min_field = fd.Function(self.P1CG)

    def apply(self, W: fd.Function, exclude=frozenset()) -> None:
        """Limit every component of ``W`` except those in ``exclude`` in place."""
        active = tuple(c for c in range(self.nc) if c not in exclude)
        if not active:
            return
        cs = self.V.mesh().cell_set
        cnm = self.V.cell_node_map()
        cnm0 = self.P0.cell_node_map()
        cnm1 = self.P1CG.cell_node_map()

        # 1. cell centroids (all components) via a single arithmetic-mean pass.
        parloop(_centroid_kernel(self.nq, self.nc), cs,
                self.centroids.dat(WRITE, cnm0), W.dat(READ, cnm))

        # 2. vertex min/max bounds over neighbouring cells (MIN/MAX reduction).
        self.max_field.assign(-1.0e10)
        self.min_field.assign(1.0e10)
        parloop(_minmax_kernel(self.nq, self.nc), cs,
                self.max_field.dat(MAX, cnm1),
                self.min_field.dat(MIN, cnm1),
                self.centroids.dat(READ, cnm0))

        # 3. Kuzmin α-clip + mean-preserving rescale of the active components.
        parloop(_limit_kernel(self.nq, self.nc, active), cs,
                W.dat(RW, cnm),
                self.centroids.dat(READ, cnm0),
                self.max_field.dat(READ, cnm1),
                self.min_field.dat(READ, cnm1))


# ---------------------------------------------------------------------------
# MOOD cell-mean reductions → DG0 fields (replace the numpy
# ``Q.dat.data_ro[conn, h_idx].mean(axis=1)`` reductions)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _cmean_kernel(nn: int, nc: int, hi: int) -> Kernel:
    return Kernel(_NONFINITE + f"""
void zoomy_cmean(double *out, const double *q) {{
  const int nn = {nn}, nc = {nc}, hi = {hi};
  double mean = 0.0;
  for (int a = 0; a < nn; a++) mean += q[a*nc + hi];
  mean /= nn;
  out[0] = zoomy_nonfinite(mean) ? INFINITY : mean;   /* +inf ⇒ ignored by min = nanmin */
}}""", "zoomy_cmean", accesses=[WRITE, READ])


@functools.lru_cache(maxsize=None)
def _mood_kernel(nn: int, nc: int, hi: int, negtol_repr: str) -> Kernel:
    return Kernel(_NONFINITE + f"""
void zoomy_mood(double *ind, const double *q) {{
  const int nn = {nn}, nc = {nc}, hi = {hi};
  double mean = 0.0;
  for (int a = 0; a < nn; a++) mean += q[a*nc + hi];
  mean /= nn;
  ind[0] = (zoomy_nonfinite(mean) || (mean < {negtol_repr})) ? 1.0 : 0.0;
}}""", "zoomy_mood", accesses=[WRITE, READ])


def cell_mean_finite_or_inf(Q: fd.Function, h_idx: int,
                            out_dg0: fd.Function) -> None:
    """Fill DG0 ``out_dg0`` with each cell's arithmetic mean of component
    ``h_idx`` — ``+inf`` for a non-finite mean, so ``PETSc Vec.min`` over it
    reproduces ``numpy.nanmin`` (non-finite cells ignored)."""
    V = Q.function_space()
    parloop(_cmean_kernel(V.cell_node_map().arity, V.value_size, int(h_idx)),
            V.mesh().cell_set,
            out_dg0.dat(WRITE, out_dg0.function_space().cell_node_map()),
            Q.dat(READ, V.cell_node_map()))


def mood_troubled_indicator(Q: fd.Function, h_idx: int, cellmean_tol: float,
                            out_dg0: fd.Function) -> None:
    """Fill DG0 ``out_dg0`` with ``1.0`` where cell-mean ``h`` is non-finite or
    ``< -cellmean_tol`` (troubled), else ``0.0`` — the compiled equivalent of
    ``~isfinite(cmean) | (cmean < -tol)``."""
    V = Q.function_space()
    negtol_repr = repr(-float(cellmean_tol))
    parloop(_mood_kernel(V.cell_node_map().arity, V.value_size, int(h_idx),
                         negtol_repr),
            V.mesh().cell_set,
            out_dg0.dat(WRITE, out_dg0.function_space().cell_node_map()),
            Q.dat(READ, V.cell_node_map()))
