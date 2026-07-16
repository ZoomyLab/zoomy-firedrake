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
from pyop2 import Kernel, parloop, READ, WRITE, RW

__all__ = [
    "zhang_shu_positivity",
    "extract_component",
    "copy_component",
    "copy_component_v2v",
    "cell_mean_finite_or_inf",
    "mood_troubled_indicator",
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
