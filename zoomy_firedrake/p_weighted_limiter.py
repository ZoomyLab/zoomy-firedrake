"""P-weighted slope limiter for DG methods.

Implements the p-weighted limiting strategy from:

    Li, L., Lou, J., Luo, H., & Nishikawa, H. (2020).
    "A new p-weighted limiter for the discontinuous Galerkin method
    on one-dimensional and two-dimensional triangular grids."
    Journal of Computational Physics, 407, 109246.

Key idea
--------
For a DGp solution, each hierarchical mode *k* (k = 1, ..., p) is
limited with a weight that increases with mode number:

    w_k = min(1, theta_k)

where theta_k is a trouble-cell indicator that is more aggressive for
higher modes.  The cell average (k = 0) is *never* modified.

For mode k the indicator compares the mode's contribution against the
range of cell-averaged neighbours:

    theta_k = |q_max - q_min| / (2 * |c_k| + eps)

and the final limited mode is:

    c_k_limited = w_k^(p - k + 1) * c_k

so that the highest mode (k = p) receives the strongest damping while
lower modes are preserved at smooth extrema.

Firedrake integration
---------------------
Firedrake stores DG data per-cell with ``(p+1)`` DoFs per cell in 1-D
(or ``(p+1)(p+2)/2`` in 2-D triangles).  The limiter works in the
Legendre modal basis:

1. Project each cell's nodal DoFs to a Legendre modal basis.
2. Compute the trouble indicator from the cell-averaged neighbourhood.
3. Apply the p-weighted damping to each mode.
4. Project back to the nodal basis.

When the mesh is not available (e.g. testing outside a Firedrake
container) the class provides a pure-NumPy ``limit_array`` method that
operates on coefficient arrays directly.
"""

import numpy as np


# ======================================================================
# Legendre basis utilities (pure NumPy, no Firedrake dependency)
# ======================================================================

def _legendre_vandermonde_1d(p, pts):
    """Vandermonde matrix for 1-D Legendre polynomials on [-1, 1].

    Parameters
    ----------
    p : int
        Polynomial degree (modes 0 .. p).
    pts : array_like, shape (n,)
        Reference-element points in [-1, 1].

    Returns
    -------
    V : ndarray, shape (n, p+1)
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    V = np.zeros((n, p + 1))
    V[:, 0] = 1.0
    if p >= 1:
        V[:, 1] = pts
    for k in range(2, p + 1):
        V[:, k] = ((2 * k - 1) * pts * V[:, k - 1] - (k - 1) * V[:, k - 2]) / k
    return V


def _reference_points_1d(p):
    """Equispaced reference points for DG *p* in [-1, 1].

    Firedrake uses equispaced points for DG by default.
    """
    if p == 0:
        return np.array([0.0])
    return np.linspace(-1.0, 1.0, p + 1)


# ======================================================================
# Core p-weighted limiting (operates on modal coefficients)
# ======================================================================

def _neighbour_ranges(cell_averages, neighbours):
    """Compute (q_min, q_max) over each cell's neighbourhood.

    Parameters
    ----------
    cell_averages : ndarray, shape (n_cells,)
    neighbours : list of list of int
        ``neighbours[i]`` is the list of cell indices adjacent to cell *i*.

    Returns
    -------
    q_min, q_max : ndarray, shape (n_cells,)
    """
    n = len(cell_averages)
    q_min = np.empty(n)
    q_max = np.empty(n)
    for i in range(n):
        nbrs = neighbours[i]
        if len(nbrs) == 0:
            q_min[i] = cell_averages[i]
            q_max[i] = cell_averages[i]
        else:
            vals = np.array([cell_averages[j] for j in nbrs])
            vals = np.append(vals, cell_averages[i])
            q_min[i] = vals.min()
            q_max[i] = vals.max()
    return q_min, q_max


def limit_modes(modal_coeffs, neighbours, eps=1e-12):
    """Apply p-weighted limiting to an array of modal coefficients.

    Parameters
    ----------
    modal_coeffs : ndarray, shape (n_cells, p+1)
        Legendre modal coefficients per cell.  Column 0 is the cell average.
    neighbours : list of list of int
        Adjacency: ``neighbours[i]`` lists indices of cells sharing a face
        with cell *i*.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    limited : ndarray, shape (n_cells, p+1)
        Modal coefficients after p-weighted limiting.
    """
    n_cells, n_modes = modal_coeffs.shape
    p = n_modes - 1
    if p < 1:
        return modal_coeffs.copy()

    cell_averages = modal_coeffs[:, 0]
    q_min, q_max = _neighbour_ranges(cell_averages, neighbours)
    delta = q_max - q_min  # range in the neighbourhood

    limited = modal_coeffs.copy()
    for k in range(1, p + 1):
        c_k = modal_coeffs[:, k]
        # Trouble indicator: ratio of neighbour range to mode magnitude
        theta_k = delta / (2.0 * np.abs(c_k) + eps)
        # Clamp to [0, 1]
        w_k = np.clip(theta_k, 0.0, 1.0)
        # P-weighted exponent: higher modes get damped more
        exponent = p - k + 1
        limited[:, k] = (w_k ** exponent) * c_k

    return limited


# ======================================================================
# PWeightedLimiter -- Firedrake-compatible interface
# ======================================================================

class PWeightedLimiter:
    """P-weighted slope limiter for Firedrake DG function spaces.

    Mirrors the API of ``firedrake.VertexBasedLimiter``: construct once
    with a scalar DG function space, then call ``apply(f)`` to limit a
    :class:`firedrake.Function` in-place.

    Parameters
    ----------
    V : firedrake.FunctionSpace
        A *scalar* DG function space (``"DG"``, degree >= 1).
    eps : float, optional
        Regularisation constant for the indicator (default 1e-12).

    Example
    -------
    >>> V = fd.FunctionSpace(mesh, "DG", 2)
    >>> limiter = PWeightedLimiter(V)
    >>> limiter.apply(my_dg_function)
    """

    def __init__(self, V, eps=1e-12):
        import firedrake as fd  # local import -- not available outside container

        self.V = V
        self.eps = eps
        mesh = V.mesh()
        self.degree = V.ufl_element().degree()

        if self.degree < 1:
            # DG0 has nothing to limit
            self._noop = True
            return
        self._noop = False

        p = self.degree

        # --- Pre-compute nodal <-> modal transform matrices ---
        ref_pts = _reference_points_1d(p)
        Vmat = _legendre_vandermonde_1d(p, ref_pts)
        self._nodal_to_modal = np.linalg.inv(Vmat)  # (p+1, p+1)
        self._modal_to_nodal = Vmat                  # (p+1, p+1)

        # --- Pre-compute cell adjacency (face neighbours) ---
        # Use the Firedrake/PETSc plex to get adjacency.
        plex = mesh.topology_dm
        dim = mesh.topological_dimension()

        # Number of cells
        cstart, cend = plex.getHeightStratum(0)
        n_cells = cend - cstart

        # Build adjacency: two cells are neighbours if they share a facet
        neighbours = [[] for _ in range(n_cells)]
        fstart, fend = plex.getHeightStratum(1)
        for f in range(fstart, fend):
            support = plex.getSupport(f)
            cells = [s - cstart for s in support if cstart <= s < cend]
            if len(cells) == 2:
                c0, c1 = cells
                neighbours[c0].append(c1)
                neighbours[c1].append(c0)

        self._neighbours = neighbours
        self._n_cells = n_cells

    def apply(self, f):
        """Limit a scalar DG function in-place.

        Parameters
        ----------
        f : firedrake.Function
            Must live in the function space ``self.V``.
        """
        if self._noop:
            return

        # Read nodal data: shape (n_cells, dofs_per_cell)
        nodal = np.array(f.dat.data_ro).reshape(self._n_cells, -1)

        # Nodal -> modal
        modal = nodal @ self._nodal_to_modal.T

        # Apply p-weighted limiting
        modal_lim = limit_modes(modal, self._neighbours, eps=self.eps)

        # Modal -> nodal
        nodal_lim = modal_lim @ self._modal_to_nodal.T

        # Write back
        f.dat.data[:] = nodal_lim.reshape(-1)
