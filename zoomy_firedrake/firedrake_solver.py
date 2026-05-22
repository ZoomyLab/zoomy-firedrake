from typing import Optional

import firedrake as fd
from firedrake.functionspaceimpl import MixedFunctionSpace
import ufl
import sympy as sp
from zoomy_core.fvm.solver_numpy import Settings
from attrs import field, define
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import Rusanov
from zoomy_firedrake.firedrake_compat import (
    safe_extract_component, safe_assign_component,
)
import numpy as np
from mpi4py import MPI

from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.misc as misc

import os

import meshio
from time import time as get_time


# TODO: Migrate from attrs (@define) to param.Parameterized per project convention.
#       Deferred because Firedrake is not installed locally and cannot be tested.


@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolver:
    """Generic DG solver for hyperbolic conservation laws using Firedrake.

    Uses Lie splitting (convective solve -> source solve) which is the
    natural IMEX pattern for hyperbolic systems with stiff source terms.

    This base class contains no free-surface (h/b) assumptions and works
    for any conservation law: scalar advection, advection-diffusion,
    user-defined systems, etc.

    Free-surface specifics (well-balancing, wet/dry, velocity capping)
    are no longer the solver's concern — the symbolic Riemann solver
    (e.g. :class:`PositiveRusanov`) and the SystemModel's
    ``update_variables`` slot carry them.  Pick the appropriate
    Riemann variant via ``riemann_solver_cls``.

    Workflow::

        solver = FiredrakeHyperbolicSolver(...)
        solver.setup_simulation(mesh_file, model)
        solver.run_simulation()

    Or for backward compatibility::

        solver.solve(mesh_file, model)
    """

    # --- Solver parameters ---
    CFL: float = field(default=0.9)
    time_end: float = field(default=0.1)
    dg_degree: int = field(default=0)
    limiter: str = field(default="vertex")  # "vertex", "p_weighted", or "none"
    # Fields the slope limiter should **skip** (pass through
    # unmodified).  A list of state-field handles — each accepted by
    # :meth:`SystemModel.field_index`:
    #
    # - ``sp.Symbol``: ``model.variables.b``
    # - ``str``:        ``"b"``
    # - ``int``:        explicit index (discouraged — defeats the point)
    #
    # ``None`` (default) → auto-detect: use the SystemModel's
    # :attr:`stationary_indices` set (fields whose evolution is
    # identically zero by construction, e.g. bathymetry ``b`` in
    # shallow-water).  Pass an empty list ``[]`` to force-limit
    # everything despite auto-detection; pass an explicit list to
    # add fields on top of the auto-detected set.  Resolved at
    # :meth:`setup_simulation` time and cached on ``_state``.
    limiter_exclude_fields: Optional[list] = field(default=None)
    # Symbolic Riemann solver class — must accept a SystemModel and
    # expose ``to_runtime_ufl()``.  Defaults to :class:`Rusanov`.
    # Swap to ``HLL`` / ``HLLC`` / ``PositiveRusanov`` for wet/dry.
    riemann_solver_cls: type = field(default=Rusanov)

    # PETSc solver options exposed as constructor kwargs (escape hatch
    # for problems where the defaults don't fit).  When ``None`` the
    # solver picks the defaults laid out in :meth:`_get_linear_solver`
    # / :meth:`_get_nonlinear_solver`; pass a ``dict`` to override.
    #
    # The defaults were picked from the Malpasset DG(0) optimisation
    # campaign — see ``tutorials/firedrake/bench_*`` — and are
    # GAMG-based: they scale serially (best at 14 s for the Malpasset
    # baseline, vs 130 s with the previous LU defaults — 9× faster)
    # AND under MPI (15 s at N=4 on a 26k-cell mesh).
    linear_solver_parameters: dict = field(default=None)
    nonlinear_solver_parameters: dict = field(default=None)

    # Nested struct with factory
    settings: Zstruct = field(factory=lambda: Settings.default())

    # Internal simulation state, populated by setup_simulation().
    # Declared as an attrs field (init=False) so it has a slot on frozen classes.
    _state: object = field(init=False, default=None, repr=False, eq=False)

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, "settings", defaults)

    # ==================================================================
    # Model introspection helpers
    # ==================================================================

    @staticmethod
    def _model_has_free_surface(model):
        """Check if model has free-surface variables (h and b).

        Works with both symbolic Model (model.variables is Zstruct with keys)
        and UFLRuntimeModel (has model.model.variables).
        """
        sym = model.model if hasattr(model, "model") else model
        keys = list(sym.variables.keys()) if hasattr(sym.variables, "keys") else []
        return "h" in keys and "b" in keys

    @staticmethod
    def _model_source_indices(model):
        """Return indices of variables that receive source terms.

        Generic: every variable index is included.  Free-surface subclasses
        override this to exclude bathymetry.
        """
        sym = model.model if hasattr(model, "model") else model
        keys = list(sym.variables.keys()) if hasattr(sym.variables, "keys") else []
        return list(range(len(keys)))

    @staticmethod
    def _slot_is_nonzero(system_model, slot_name):
        """Check whether ``SystemModel.<slot_name>`` carries a
        non-trivial expression at the **active** parameter values.

        Walks every entry of the slot's symbolic tensor and substitutes
        the SystemModel's numeric ``parameter_values``.  Returns
        ``False`` when the slot is ``None`` or every entry collapses to
        zero once parameters are baked in (e.g. ``ν · h`` with ``ν = 0``
        — symbolically non-zero but numerically off).  This matters
        because the UFL lambdification path simplifies ``0 · h`` to
        ``0`` at code-gen time, losing the mesh reference and breaking
        ``form * dx`` with a "missing integration domain" error.
        """
        T = getattr(system_model, slot_name, None)
        if T is None:
            return False
        try:
            params = system_model.parameters
            values = system_model.parameter_values
            sub_map = {params[k]: float(getattr(values, k, 0.0))
                       for k in params.keys()}
            return not all(
                sp.simplify(sp.sympify(e).xreplace(sub_map)) == 0
                for e in sp.flatten(T)
            )
        except Exception:
            return False

    # ==================================================================
    # Mesh / geometry helpers
    # ==================================================================

    def get_map_boundary_tag_to_boundary_function_index(
            self, model, msh_path, mesh, *, boundary_tag_map=None):
        """Build ``{physical_id: bc_list_index}`` for the BC weak-form loop.

        Two sources for the ``{tag_name: physical_id}`` resolution:

        - When the caller passed a gmsh ``.msh`` path, parse the named
          physical groups via ``meshio`` (legacy behaviour).
        - When the caller passed an in-memory ``fd.MeshGeometry``
          (``msh_path is None``), require ``boundary_tag_map`` —
          a ``{tag_name: physical_id}`` dict.  For
          ``fd.IntervalMesh(N, length=H)`` the conventional markers
          are ``1`` (left endpoint) and ``2`` (right endpoint); the
          caller maps these to BC tag names.
        """
        boundary_function_names = model.boundary_conditions._boundary_tags
        if msh_path is None:
            # In-memory mesh path: caller supplies the tag map.
            if boundary_tag_map is None:
                if boundary_function_names:
                    return {"__all__": 0}
                return {}
            field_data = {
                name: [pid] for name, pid in boundary_tag_map.items()
                if name in boundary_function_names
            }
        else:
            msh = meshio.read(msh_path)
            field_data_raw = msh.field_data  # name -> [id, dim, type]
            field_data = {
                name: data for name, data in field_data_raw.items()
                if name in boundary_function_names
            }
        extracted_facets_firedrake = mesh.exterior_facets.unique_markers
        assert len(extracted_facets_firedrake) == len(field_data), (
            f"Mismatch in number of boundary tags: "
            f"extracted {extracted_facets_firedrake}, expected {field_data.keys()}"
        )

        # 1. Make a list of (name, id)
        name_id_pairs = [(name, data[0]) for name, data in field_data.items()]

        if not name_id_pairs:
            # Mesh has no named physical boundary groups at all — fall
            # back to applying the first declared BC over the entire
            # exterior (every unmarked facet).  Without this fallback
            # the BC loop is silently empty: no boundary flux enters
            # the weak form, and conservation laws can leak mass /
            # momentum freely through ∂Ω.  The sentinel ``"__all__"``
            # is interpreted in the weak-form builder as ``fd.ds``
            # (no subdomain_id).
            if model.boundary_conditions._boundary_tags:
                return {"__all__": 0}
            return {}

        # 2. Sort alphabetically by name
        name_id_pairs.sort(key=lambda x: x[0])

        # 3. Assign consecutive indices (0-based)
        name_to_index = {name: i for i, (name, _) in enumerate(name_id_pairs)}

        # 4. Invert to get physical_id -> index
        physical_id_to_index = {pid: name_to_index[name] for name, pid in name_id_pairs}

        return physical_id_to_index

    def _get_x_and_n(self, mesh):
        x = fd.SpatialCoordinate(mesh)
        dim = mesh.geometric_dimension()

        # Always construct a 3D vector (fill missing components with 0)
        if dim == 1:
            x_3d = fd.as_vector([x[0], fd.Constant(0.0), fd.Constant(0.0)])
        elif dim == 2:
            x_3d = fd.as_vector([x[0], x[1], fd.Constant(0.0)])
        else:
            x_3d = fd.as_vector([x[0], x[1], x[2]])
        n = fd.FacetNormal(mesh)
        return x, x_3d, n

    # ==================================================================
    # Numerical flux and related
    # ==================================================================

    def get_nonconservative_flux(self, model, parameters, mesh):
        """Closure returning ``(Dp, Dm)`` — the NCP path-integral
        fluctuations split into outgoing / incoming contributions.

        This is just a thin shim over
        ``self._state.runtime_numerics.numerical_fluctuations`` from
        the SystemModel-driven symbolic Riemann library.  The library
        already does the Gauss-Legendre path integral, the LF
        stabilisation, and the bed-row mask in
        :meth:`NonconservativeRusanov.get_viscosity_identity_fluctuations`.
        Earlier this method re-implemented the same algorithm in
        Python + UFL — that bypassed the symbolic library's
        well-balancing / bed-mask logic and let DG(0)-discontinuous
        ``b`` drift through the LF stabilisation term.

        For Riemann solvers that don't override
        ``numerical_fluctuations`` (e.g. plain ``Rusanov``, ``HLL``)
        the lambdified callable returns zero and the contribution is
        silently dropped.
        """
        sm = self._state.system_model
        n_var = sm.n_equations

        # Short-circuit: when the model has no NCP at all, the
        # symbolic ``numerical_fluctuations`` lambdifies to numerical
        # zeros that lose their mesh reference; multiplying by
        # ``fd.dS`` then trips "missing integration domain".  Returning
        # a mesh-aware ``fd.Constant(0.0)`` vector directly avoids
        # that and skips the lambdified-call cost.
        ncp_is_zero = all(
            sp.simplify(e) == 0
            for e in sp.flatten(sm.nonconservative_matrix)
        )
        if ncp_is_zero:
            zero_vec = fd.as_vector([fd.Constant(0.0)] * n_var)

            def nc_flux(Ql, Qr, Qauxl, Qauxr, n):
                return zero_vec, zero_vec
            return nc_flux

        rn = self._state.runtime_numerics

        def nc_flux(Ql, Qr, Qauxl, Qauxr, n):
            # ``numerical_fluctuations`` returns a ``(2, n_var)`` UFL
            # tensor: row 0 = Dp (outgoing), row 1 = Dm (incoming).
            # UFL rank-2-tensor[0] does NOT auto-collapse to a
            # length-n_var vector — slice element-wise.
            fl = rn.numerical_fluctuations(
                Ql, Qr, Qauxl, Qauxr, parameters, n,
            )
            Dp = fd.as_vector([fl[0, i] for i in range(n_var)])
            Dm = fd.as_vector([fl[1, i] for i in range(n_var)])
            return Dp, Dm
        return nc_flux

    def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, mesh):
        """Apply the symbolic Riemann solver lowered to UFL.

        ``self._state.runtime_numerics`` is the
        :class:`UFLRuntimeSymbolic` produced by
        ``self.riemann_solver_cls(SystemModel.from_model(model)).to_runtime_ufl()``
        in :meth:`setup_simulation`; its ``numerical_flux`` callable
        takes the face-side states + normals and returns the
        numerical-flux UFL vector.  This routes every well-balancing /
        wet-dry / structure-specific logic (handled symbolically inside
        the Riemann solver via :class:`FieldHandle`) into the same code
        path — no Firedrake-side reconstruction is needed.
        """
        rn = self._state.runtime_numerics
        return rn.numerical_flux(Ql, Qr, Qauxl, Qauxr, parameters, n)

    def max_abs_eigenvalue(self, model, Q, Qaux, n, mesh):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(abs(ev[i]) > max_ev, abs(ev[i]), max_ev)
        return max_ev

    # ==================================================================
    # State update helpers
    # ==================================================================

    def write_state(self, q, qaux, out, time=0.0, names=None):
        mesh = q.function_space().mesh()
        # Always project to DG0 for VTK output regardless of solution degree
        V_scalar = fd.FunctionSpace(mesh, "DG", 0)
        n_dof_q = q.function_space().value_size
        n_dof_aux = qaux.function_space().value_size
        subfuns = [
            fd.project(q[i], V_scalar, name=names[i] if names else f"Q{i}")
            for i in range(n_dof_q)
        ] + [
            fd.project(qaux[i], V_scalar, name=names[n_dof_aux + i] if names else f"Qaux{i}")
            for i in range(n_dof_aux)
        ]
        out.write(*subfuns, time=time)

    def update_Qaux(self, Q, Qaux, runtime_model=None, eps=1e-8):
        """Update auxiliary variables via the model's callback.

        If the model exposes ``update_aux_variables``, that is used.
        Otherwise this is a no-op (generic models may have no aux variables).
        """
        if runtime_model is not None and hasattr(runtime_model, "update_aux_variables"):
            Qaux_new = runtime_model.update_aux_variables(Q, Qaux, runtime_model.parameters)
            Qaux.interpolate(Qaux_new)

    def update_Q(self, Q, Qaux, runtime_model=None):
        """Update conserved variables via the model's callback.

        If the model exposes ``update_variables``, that is used.
        Otherwise this is a no-op (generic models need no post-processing).
        """
        if runtime_model is not None and hasattr(runtime_model, "update_variables"):
            Q_new = runtime_model.update_variables(Q, Qaux, runtime_model.parameters)
            Q.interpolate(Q_new)

    # ==================================================================
    # Function space / coordinate helpers
    # ==================================================================

    def get_function_coordinates(self, Q):
        """Return coordinates (as NumPy array) for each DoF in a Firedrake Function.

        Works for DG/CG and arbitrary polynomial degree, assuming:
            Q.function_space() == VectorFunctionSpace(mesh, family, degree, dim=n_variables)

        Output shape: (num_dofs, geometric_dimension())
        """
        V = Q.function_space()
        mesh = V.mesh()
        element = V.ufl_element()

        family = element.family()
        degree = element.degree()
        dim = mesh.geometric_dimension()

        # Build coordinate field in the same FE family/degree
        V_coords = fd.VectorFunctionSpace(mesh, family, degree)
        x = fd.SpatialCoordinate(mesh)
        coords_func = fd.Function(V_coords).interpolate(fd.as_vector(x))

        # Return as NumPy array (read-only for safety).
        # 1D Firedrake meshes give a rank-1 coords array; reshape to
        # ``(n_dofs, 1)`` so downstream callers always see a 2D layout.
        coords = np.asarray(coords_func.dat.data_ro)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        assert coords.shape[1] == dim, f"Unexpected coordinate shape {coords.shape}, dim={dim}"
        return coords

    def get_compute_dt(self, mesh, model, CFL=0.9):
        """Return a callable that computes ``dt(Q, Qaux)`` for the given
        mesh and model.

        Uses the classical RKDG CFL condition

        .. math::

            \\Delta t \\le \\mathrm{CFL} \\;
                \\frac{h_\\text{cell}}{d \\, (2k+1) \\, |\\lambda|_\\text{max}}

        where ``h_cell`` is the cell diameter, ``d`` the spatial
        dimension, ``k`` the DG polynomial degree, and ``|λ|_max`` the
        cell-local maximum absolute eigenvalue of the normal-projected
        quasilinear matrix.  The factor ``1/(2k+1)`` is the theoretical
        SSP-RK(k+1, k+1) stability constant (Cockburn & Shu, 1991) —
        equivalently ``1/(2p−1)`` in the "polynomial order ``p = k+1``"
        convention.  ``CFL`` is therefore a *safety factor* on top of
        the theoretical limit, ≤ 1; defaults to ``0.9`` (aggressive but
        stable on well-shaped meshes).  Drop it to ~0.5 for skewed
        meshes, strong shocks, or unlimited DG(p≥1).

        ``CellDiameter`` is the cell's longest edge — for triangles
        roughly ``2·inradius`` times a quality factor.  The combination
        ``CFL · h / (d · (2k+1) · λ)`` is the common engineering
        approximation; a fully tight bound would use the inradius.
        """
        degree = self.dg_degree
        dim = mesh.geometric_dimension()

        degree_factor = float(2 * degree + 1)
        dim_factor = float(dim)

        # Sample the per-point CFL bound at every nodal DOF of a
        # ``DG(degree)`` scalar space (i.e. one DOF per Q-DOF in each
        # cell).  Taking ``np.min(dt_local.dat.data_ro)`` then picks
        # the worst CFL constraint anywhere in the field — including
        # sub-cell h-minima at DG(1+) vertices where a cell can have
        # ``h_centroid > eps`` but ``h_vertex`` near zero (the
        # dt-collapse failure mode observed on the DG(1) Malpasset
        # run).  For DG(0) this collapses to the previous centroid
        # sample (one DOF/cell).  ``fd.CellDiameter`` stays a pure UFL
        # expression — the assembler broadcasts the per-cell value to
        # each DOF inside that cell during ``interpolate``.
        V_dt = fd.FunctionSpace(mesh, "DG", degree)
        dt_local = fd.Function(V_dt)
        h_cd = fd.CellDiameter(mesh)

        def compute_dt(Q, Qaux):
            """Compute global stable dt for the given fields Q and Qaux."""
            dt_expr = (
                CFL * h_cd
                / (
                    dim_factor * degree_factor
                    * ufl.max_value(
                        self.max_abs_eigenvalue(
                            model, Q, Qaux,
                            fd.as_vector([1.0, 0.0]), mesh),
                        self.max_abs_eigenvalue(
                            model, Q, Qaux,
                            fd.as_vector([0.0, 1.0]), mesh),
                    )
                    + 1e-8
                )
            )
            dt_local.interpolate(dt_expr)
            dt_min = float(np.min(dt_local.dat.data_ro))
            return mesh.comm.allreduce(dt_min, op=MPI.MIN)

        return compute_dt

    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        x = fd.SpatialCoordinate(mesh)
        coords = self.get_function_coordinates(Q)
        Qarr = Q.dat.data
        Q.dat.data[:] = model.initial_conditions.apply(coords.T, Qarr.T).T

    def _get_functionspaces(self, mesh, runtime_model, degree=None):
        """Build DG function spaces with configurable polynomial degree.

        Parameters
        ----------
        mesh : firedrake.Mesh
        runtime_model : UFLRuntimeModel
        degree : int or None
            Polynomial degree for the DG space.  When *None* the solver's
            ``dg_degree`` attribute is used.
        """
        if degree is None:
            degree = self.dg_degree
        V = fd.VectorFunctionSpace(mesh, "DG", degree, dim=runtime_model.n_variables)
        Vaux = fd.VectorFunctionSpace(mesh, "DG", degree, dim=runtime_model.n_aux_variables)
        Qn = fd.Function(V)
        Qs = fd.Function(V)
        Qnp1 = fd.Function(V)
        Qaux_n = fd.Function(Vaux)
        Qaux_s = fd.Function(Vaux)
        Qaux_np1 = fd.Function(Vaux)

        return V, Vaux, Qn, Qs, Qnp1, Qaux_n, Qaux_s, Qaux_np1

    # ==================================================================
    # Weak form construction
    # ==================================================================

    def _get_weak_form_convective(self, runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1,
                                  n, mesh, map_boundary_tag_to_function_index,
                                  sim_time, dt, x, x_3d):
        test_q = fd.TestFunction(Qn.function_space())
        trial_q = fd.TrialFunction(Qn.function_space())

        Q = Qn
        Qaux = Qaux_n

        weak_form = fd.dot(test_q, (trial_q - Qn) / dt) * fd.dx

        weak_form += (
            fd.dot(
                test_q("+") - test_q("-"),
                self.numerical_flux(
                    runtime_model,
                    Q("+"),
                    Q("-"),
                    Qaux_n("+"),
                    Qaux_n("-"),
                    runtime_model.parameters,
                    n("+"),
                    mesh,
                ),
            )
            * fd.dS
        )

        nc_flux = self.get_nonconservative_flux(runtime_model, runtime_model.parameters, mesh)
        Dp, Dm = nc_flux(Q("-"), Q("+"), Qaux_n("-"), Qaux_n("+"), n("-"))
        weak_form += 0.5 * (
            fd.dot(
                test_q("+"),
                Dp
            )
        ) * fd.dS
        weak_form += 0.5 * (
            fd.dot(
                test_q("-"),
                Dm
            )
        ) * fd.dS

        for tag, idx in map_boundary_tag_to_function_index.items():
            # ``__all__`` is the sentinel emitted by
            # ``get_map_boundary_tag_to_boundary_function_index`` when
            # the mesh has no named physical groups — apply the BC
            # over every exterior facet via ``fd.ds`` (no subdomain).
            ds_measure = fd.ds if tag == "__all__" else fd.ds(tag)

            # Position and "distance" placeholders
            dX = x[0]

            # Evaluate boundary state from model
            Q_bnd = runtime_model.boundary_conditions(
                idx,
                sim_time,
                x_3d,
                dX,
                Q,
                Qaux_n,
                runtime_model.parameters,
                n,
            )
            weak_form += ufl.dot(
                test_q,
                self.numerical_flux(runtime_model, Q, Q_bnd, Qaux_n, Qaux_n,
                                    runtime_model.parameters, n, mesh)
            ) * ds_measure

            Dp, Dm = nc_flux(Q, Q_bnd, Qaux_n, Qaux_n, n)
            weak_form += (
                fd.dot(
                    test_q,
                    Dm
                )
            ) * ds_measure

        # --- Explicit IMEX slot: ``diffusion_matrix_explicit`` →
        # evaluated at Qn here.  The implicit ``diffusion_matrix`` slot
        # is added inside the source step at Qnp1 (see
        # _get_weak_form_source).
        sm = self._state.system_model
        if self._slot_is_nonzero(sm, "diffusion_matrix_explicit"):
            weak_form += self._get_weak_form_diffusion(
                runtime_model, Q, Qaux_n, test_q, mesh, n,
                map_boundary_tag_to_function_index, sim_time, x, x_3d,
                slot="diffusion_matrix_explicit",
            )

        # --- Explicit IMEX slot: ``source_explicit`` → evaluated at Qn
        # here (Forward-Euler).  The implicit ``source`` slot is added
        # inside the source step at Qnp1.
        if self._slot_is_nonzero(sm, "source_explicit"):
            S_expl = runtime_model.source_explicit(
                Q, Qaux_n, runtime_model.parameters)
            weak_form -= fd.dot(test_q, S_expl) * fd.dx

        return weak_form

    def _get_weak_form_source(
        self,
        runtime_model,
        Q_star,      # this is Qs from the convective step
        Qnp1,        # unknown for the source step
        Qaux_star,
        Qaux_np1,
        n, mesh, map_boundary_tag_to_function_index,
        sim_time, dt, x, x_3d,
        theta=1.0
    ):
        """Source step: evolve only the variables that receive source terms.

        The source-active indices are determined by ``_model_source_indices``:
        - Generic models: all variable indices

        We build a residual like
            (Qnp1[i] - Q_star[i])/dt = S_i(...)
        for i in source_indices, and 0 for other components.
        """

        # Derive source indices from the model instead of hardcoding [1,2,3]
        source_indices = self._model_source_indices(runtime_model)

        V = Qnp1.function_space()
        test_q = fd.TestFunction(V)
        ncomp = V.value_size

        # theta-method interpolation
        Q_theta = theta * Qnp1 + (1.0 - theta) * Q_star
        Qaux_theta = theta * Qaux_np1 + (1.0 - theta) * Qaux_star

        # Full source vector from the model
        source_full = runtime_model.source(Q_theta, Qaux_theta, runtime_model.parameters)

        zero = fd.Constant(0.0)

        # Always include the (Qnp1[i] − Q_star[i]) identity row for
        # every component — otherwise the Jacobian rows for components
        # without a source term are identically zero and the Newton/KSP
        # solve diverges (DIVERGED_LINEAR_SOLVE).  Only the source RHS
        # is restricted to ``source_indices``; non-source components
        # propagate the convective step's value unchanged.
        diff_restricted = []
        source_restricted = []
        for i in range(ncomp):
            diff_restricted.append(Qnp1[i] - Q_star[i])
            source_restricted.append(source_full[i] if i in source_indices else zero)

        diff_restricted = fd.as_vector(diff_restricted)
        source_restricted = fd.as_vector(source_restricted)

        # Weak form: only source-active components contribute
        weak_form = fd.dot(test_q, diff_restricted / dt) * fd.dx
        weak_form -= fd.dot(test_q, source_restricted) * fd.dx

        # Implicit IMEX slot: ``diffusion_matrix`` evaluated at Qnp1.
        # Newton drives both friction and diffusion to their implicit
        # fixed-point — no parabolic CFL.
        sm = self._state.system_model
        if self._slot_is_nonzero(sm, "diffusion_matrix"):
            weak_form += self._get_weak_form_diffusion(
                runtime_model, Qnp1, Qaux_np1, test_q, mesh, n,
                map_boundary_tag_to_function_index, sim_time, x, x_3d,
                slot="diffusion_matrix",
            )

        return weak_form

    # ------------------------------------------------------------------
    # Diffusive flux — TPFA (DG0) / IP-DG (DG1+)
    # ------------------------------------------------------------------

    def _get_weak_form_diffusion(self, runtime_model, Q, Qaux, test_q, mesh, n,
                                 map_boundary_tag_to_function_index,
                                 sim_time, x, x_3d,
                                 slot="diffusion_matrix"):
        """Discretise ``-∇·(A:∇Q)`` — TPFA for DG(0), IP-DG for DG(1+).

        ``A = sm.<slot>`` (default ``diffusion_matrix``, i.e. the
        implicit slot) is a rank-4 UFL tensor of shape
        ``(n_eq, n_state, n_dim, n_dim)``.  Pass ``slot=
        "diffusion_matrix_explicit"`` to build the explicit form.

        The two DG cases use different formulations because the cell-
        interior gradient is ``≡ 0`` for DG(0):

        - **DG(0) — TPFA** (two-point flux approximation).  Face-normal
          directional derivative approximated by the centroid-to-
          centroid finite difference; bilinear contribution
          ``+∫ [v] · M_avg · [Q] / d_face dS`` over interior facets,
          with ``M[i,j] = Σ_{d,e} A[i,j,d,e]·n[d]·n[e]``.
        - **DG(1+) — symmetric IP-DG**: volume + consistency +
          symmetry + penalty.

        Boundary diffusive contributions are zero (homogeneous Neumann)
        — adequate for wall/outflow patterns.  Extend when a Dirichlet
        diffusive boundary is needed.
        """
        sm = self._state.system_model
        if not self._slot_is_nonzero(sm, slot):
            return fd.Constant(0.0) * fd.dot(test_q, Q) * fd.dx

        n_eq = sm.n_equations
        n_st = sm.n_state
        gdim = mesh.geometric_dimension()
        assert n_st == Q.function_space().value_size, (
            f"{slot} expects n_state == ncomp(Q) for the contraction "
            "with grad(Q)."
        )

        if self.dg_degree == 0:
            return self._diffusion_form_tpfa(
                sm, Q, Qaux, test_q, mesh, n, runtime_model, gdim, slot,
            )

        # Lambdify A slab-by-slab through the UFL module so that
        # ``A_slabs[d][e]`` is an (n_eq, n_state) UFL tensor.
        A_slabs = self._build_diffusion_slabs(sm, Q, Qaux, runtime_model, slot)

        # Volume tensor contraction.
        grad_Q = fd.grad(Q)            # (n_state, gdim)
        grad_v = fd.grad(test_q)       # (n_eq,    gdim)

        # F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]
        def F_diff(d):
            terms = [A_slabs[d][e] * grad_Q[:, e] for e in range(gdim)]
            # each term is (n_eq,) UFL vector; sum element-wise
            return sum(terms[1:], terms[0])

        # Penalty parameter (standard IP-DG): σ = c · p² scaled by h_F.
        sigma = fd.Constant(float(10.0 * self.dg_degree ** 2))
        h_F = fd.CellDiameter(mesh)

        # +∫ F_diff : ∇v dx
        vol_form = fd.Constant(0.0) * fd.dot(test_q, Q) * fd.dx
        for d in range(gdim):
            vol_form += fd.inner(F_diff(d), grad_v[:, d]) * fd.dx

        # Interior face terms.  Build per-side F_diff and A·∇v from the
        # ``+`` / ``-`` traces; jump and average are the standard DG ops.
        A_plus = self._build_diffusion_slabs(sm, Q("+"), Qaux("+"), runtime_model, slot)
        A_minus = self._build_diffusion_slabs(sm, Q("-"), Qaux("-"), runtime_model, slot)

        def F_diff_side(A_side, Q_side, d):
            gQ = fd.grad(Q_side)
            terms = [A_side[d][e] * gQ[:, e] for e in range(gdim)]
            return sum(terms[1:], terms[0])

        n_plus = n("+")
        jump_v = test_q("+") - test_q("-")
        jump_Q = Q("+") - Q("-")

        face_form = fd.Constant(0.0) * fd.dot(test_q("+"), Q("+")) * fd.dS

        # Consistency: -∫ avg(F_diff·n) · [v] dS
        avg_Fn = sum(
            0.5 * (F_diff_side(A_plus,  Q("+"), d)
                   + F_diff_side(A_minus, Q("-"), d)) * n_plus[d]
            for d in range(gdim)
        )
        face_form -= fd.dot(avg_Fn, jump_v) * fd.dS

        # Symmetry: -∫ avg(A·∇v·n) · [Q] dS
        gv_plus = fd.grad(test_q("+"))
        gv_minus = fd.grad(test_q("-"))
        Av_plus = sum(
            sum(A_plus[d][e] * gv_plus[:, e] for e in range(gdim)) * n_plus[d]
            for d in range(gdim)
        )
        Av_minus = sum(
            sum(A_minus[d][e] * gv_minus[:, e] for e in range(gdim)) * n_plus[d]
            for d in range(gdim)
        )
        face_form -= 0.5 * fd.dot(Av_plus + Av_minus, jump_Q) * fd.dS

        # Penalty: +∫ (σ / avg_h) · [Q] · [v] dS
        avg_h = (h_F("+") + h_F("-")) / 2.0
        face_form += (sigma / avg_h) * fd.dot(jump_Q, jump_v) * fd.dS

        return vol_form + face_form

    def _diffusion_form_tpfa(self, sm, Q, Qaux, test_q, mesh, n,
                             runtime_model, gdim, slot="diffusion_matrix"):
        """Two-point flux approximation diffusion form for DG(0).

        On a piecewise-constant DG(0) space ``∇Q = 0`` inside each cell,
        so a volume integral of ``F_diff : ∇v`` vanishes and the
        diffusion has to enter *through* the faces.  TPFA approximates
        the face-normal directional derivative by the centroid-to-
        centroid finite difference

        .. math::

            (A : \\nabla Q \\cdot n)[i] \\approx
              M_{ij} \\, \\frac{Q[j]_+ - Q[j]_-}{d_\\text{face}},
              \\quad M_{ij} = \\sum_{d,e} A_{i,j,d,e}\\, n_d\\, n_e.

        The cell-centroid distance ``d_face`` is approximated by the
        average of the per-side ``CellDiameter`` — exact on regular
        orthogonal quad meshes and a reasonable approximation on
        triangular ones.  The arithmetic mean of the ``+`` / ``-``
        contracted matrices is used for ``M_avg``; harmonic averaging
        could be substituted if needed for strongly heterogeneous A.

        Residual contribution per interior facet:

        .. math::

            +\\int_S [v] \\cdot \\frac{M_\\text{avg} \\, [Q]}{d_\\text{face}}\\, dS.

        Boundary diffusive contributions are zero (homogeneous Neumann).
        """
        A_plus = self._build_diffusion_slabs(
            sm, Q("+"), Qaux("+"), runtime_model, slot,
        )
        A_minus = self._build_diffusion_slabs(
            sm, Q("-"), Qaux("-"), runtime_model, slot,
        )
        n_plus = n("+")

        def _normal_contract(A_slabs):
            """``M[i, j] = Σ_{d, e} A[d][e][i, j] · n[d] · n[e]``."""
            return sum(
                A_slabs[d][e] * n_plus[d] * n_plus[e]
                for d in range(gdim) for e in range(gdim)
            )

        M_avg = 0.5 * (_normal_contract(A_plus) + _normal_contract(A_minus))

        h_F = fd.CellDiameter(mesh)
        d_face = (h_F("+") + h_F("-")) / 2.0

        jump_Q = Q("+") - Q("-")
        jump_v = test_q("+") - test_q("-")
        flux_face = fd.dot(M_avg, jump_Q) / d_face  # (n_eq,) UFL vector

        # Residual ∫_K (-∇·F_diff) v dx integrates to
        # ``-∮_∂K F·n_∂K · v ds`` (no volume part on DG0).  Summing the
        # two cell traces on an interior face f with ``n("+")`` outward
        # of the "+" cell gives ``-∫_f F̂·n("+") · [v] dS``.  In the
        # TPFA approximation F̂·n("+") ≈ M_avg · [Q] / d_face.
        return -fd.dot(jump_v, flux_face) * fd.dS

    @staticmethod
    def _build_diffusion_slabs(sm, Q_side, Qaux_side, runtime_model,
                               slot="diffusion_matrix"):
        """Lambdify ``sm.<slot>`` per ``(d, e)`` slab through the UFL
        module and substitute the UFL traces ``Q_side`` / ``Qaux_side``
        for the state symbols.

        ``slot`` selects which SystemModel slot to lower:
        ``"diffusion_matrix"`` (implicit) or
        ``"diffusion_matrix_explicit"`` (explicit IMEX companion).

        Returns ``A_slabs[d][e]`` — each an ``(n_eq, n_state)`` UFL
        tensor.  Outer index is the flux direction ``d``; inner is the
        gradient direction ``e``.
        """
        from zoomy_core.misc.misc import Zstruct
        from zoomy_core.model.basefunction import Function

        A_tensor = getattr(sm, slot)
        std_sig = Zstruct(variables=sm.variables,
                          aux_variables=sm.aux_variables,
                          parameters=sm.parameters)

        n_eq = sm.n_equations
        n_st = sm.n_state
        gdim = sm.n_dim

        slabs = []
        for d in range(gdim):
            row = []
            for e in range(gdim):
                slab_expr = sp.Matrix(
                    n_eq, n_st,
                    lambda i, j, _d=d, _e=e: A_tensor[i, j, _d, _e],
                )
                fn = Function(
                    name=f"{slot}__d{d}_e{e}",
                    args=std_sig,
                    definition=slab_expr,
                )
                callable_fn = runtime_model._lambdify_function(
                    fn, [runtime_model.module])
                row.append(callable_fn(Q_side, Qaux_side, runtime_model.parameters))
            slabs.append(row)
        return slabs

    # ------------------------------------------------------------------
    # Slope limiter for DG1+
    # ------------------------------------------------------------------

    def _resolve_limiter_exclude_indices(self, system_model) -> frozenset:
        """Translate ``self.limiter_exclude_fields`` (a list of field
        handles, or ``None``) into a frozenset of integer state indices.

        Default behavior (``limiter_exclude_fields = None``) returns
        :attr:`SystemModel.stationary_indices` — fields whose
        evolution is identically zero by construction.  This is the
        right thing for the canonical case (bathymetry ``b`` in
        shallow-water): the limiter never touches ``b``, MPI rank
        boundaries stay consistent, no spurious topography is created
        or destroyed.

        When the user passes an explicit list, each entry is resolved
        via :meth:`SystemModel.field_index` (accepts symbol, name, or
        int).  Passing an empty list disables the exclusion entirely
        (forces the limiter onto every component).
        """
        if self.limiter_exclude_fields is None:
            return system_model.stationary_indices
        return frozenset(
            system_model.field_index(f) for f in self.limiter_exclude_fields
        )

    def _apply_slope_limiter(self, Q):
        """Apply slope limiter for DG degree >= 1.

        For vector function spaces the limiter is applied component-wise
        because both ``VertexBasedLimiter`` and ``PWeightedLimiter``
        operate on scalar spaces.

        Components flagged in ``self._state.limiter_exclude_indices``
        are **passed through unmodified** — limiting them would
        spuriously perturb fields the model declares stationary
        (e.g. bathymetry ``b`` in shallow-water), which under MPI
        rank-splitting then differs across processes and silently
        creates / destroys topography.  The exclude set is computed
        at setup time via :meth:`_resolve_limiter_exclude_indices`.

        Supported modes (set via ``self.limiter``):

        - ``"vertex"``     -- Kuzmin-type vertex-based limiter (Firedrake built-in).
        - ``"p_weighted"`` -- p-weighted limiter (Li et al. 2020).  Higher-order
          modes are damped more aggressively, preserving accuracy at smooth
          extrema while controlling oscillations at shocks.
        - ``"none"``       -- no limiting.
        """
        if self.dg_degree < 1:
            return
        if self.limiter == "none":
            return

        V = Q.function_space()
        mesh = V.mesh()
        ncomp = V.value_size

        # Build a scalar DG space with the same degree for the limiter
        V_scalar = fd.FunctionSpace(mesh, "DG", self.dg_degree)

        if self.limiter == "p_weighted":
            from zoomy_firedrake.p_weighted_limiter import PWeightedLimiter
            # Cache the limiter on _state to avoid recomputing adjacency
            s = self._state
            if s is not None and hasattr(s, "_pw_limiter"):
                limiter = s._pw_limiter
            else:
                limiter = PWeightedLimiter(V_scalar)
                if s is not None:
                    s._pw_limiter = limiter
        else:
            limiter = fd.VertexBasedLimiter(V_scalar)

        exclude = getattr(self._state, "limiter_exclude_indices",
                          frozenset())

        # Per-component limiting with **explicit dat.data slicing on
        # both read and write**.  Excluded components (e.g. bathymetry
        # ``b``) are never touched — bit-preserved.
        #
        # **Why direct dat slicing on the read side**: the obvious
        # ``qi.interpolate(Q.sub(i))`` triggers a Firedrake stride bug
        # in the interpolation kernel for ``VectorFunctionSpace.sub(i)``
        # at ``i ≥ 2``: it silently zeroes scattered values in
        # ``Q.sub(0)`` (b) on rank-boundary partitions.  Reproduced
        # this session via the one-step diagnostic with a per-step
        # b-probe inside the loop — b stayed clean through
        # ``i = 0, 1``, then jumped to ~0 in select cells at the
        # ``qi.interpolate(Q.sub(2))`` call.  Direct slicing avoids
        # the kernel entirely.
        #
        # **Why direct dat slicing on the write side**: the old
        # ``Q.interpolate(fd.as_vector(limited))`` re-vector path
        # injected ~1% mass per step (the vector interpolation back
        # into ``Q`` perturbs cell averages even though Kuzmin's
        # kernel itself is cell-average-preserving — see the
        # Firedrake VertexBasedLimiter source: ``q[ii] = qavg + alpha
        # * (q[ii] - qavg)`` is conservative by construction).
        #
        # ``Q.dat.data[:, i] = qi.dat.data_ro[:]`` is an exact,
        # bit-for-bit numpy slice copy mirroring how the IC writes
        # ``Q`` (see ``MalpassetSolver.set_initial_condition``).  No
        # projection, no interpolation, no layout ambiguity.
        # Per-component limiting via the ``safe_*_component`` wrappers
        # (see ``zoomy_firedrake.firedrake_compat`` for the bug
        # documentation + the standalone reproducer).  When Firedrake
        # fixes ``VectorFunctionSpace.sub(i).interpolate`` /
        # ``.assign``, the wrappers collapse to plain
        # ``.interpolate(Q.sub(i))`` and ``Q.sub(i).assign(qi)``.
        #
        # **Phase ordering matters**: all reads must complete before
        # any writes.  Interleaving a write (direct ``dat.data``
        # slice) and a subsequent read (``interpolate`` on a different
        # subview) triggers a separate Firedrake bug where the read's
        # halo handling fetches stale data and corrupts other
        # columns.  Reproduced standalone — see the reproducer's
        # final loop variant.
        limited = {}
        for i in range(ncomp):
            if i in exclude:
                continue
            qi = fd.Function(V_scalar)
            safe_extract_component(qi, Q, i)
            limiter.apply(qi)
            limited[i] = qi
        for i, qi in limited.items():
            safe_assign_component(Q, qi, i)

    # ==================================================================
    # Solver construction
    # ==================================================================

    # ── Default PETSc solver parameters ───────────────────────────────
    # Picked from the DG(0) Malpasset optimisation campaign (see
    # ``tutorials/firedrake/bench_*``).  Honoured by
    # :meth:`_get_linear_solver` / :meth:`_get_nonlinear_solver` when
    # the user does NOT override via the constructor kwargs
    # ``linear_solver_parameters`` / ``nonlinear_solver_parameters``.

    # Convective (explicit-IMEX) step: ``M (Qnp1 - Qn)/dt + R(Qn) = 0``.
    # For DG the mass matrix is block-diagonal per cell ⇒ block-Jacobi
    # with sub-LU is an exact direct solve and the cheapest possible
    # PC choice; ``ksp_type=preonly`` skips the iterative outer loop.
    DEFAULT_LINEAR_SOLVER_PARAMETERS = {
        "ksp_type": "preonly",
        "pc_type": "bjacobi", "sub_pc_type": "lu",
    }

    # Source (implicit) step: Newton over Manning friction (block-
    # diagonal) ± optional implicit diffusion (face-coupling).
    # GAMG handles both regimes well — degenerates cheaply on
    # block-diagonal matrices and scales properly under MPI.
    # ``basic`` line search (no backtracking) + relaxed Newton
    # tolerances cut Newton iterations to ~1 for the Malpasset case
    # without harming mass conservation (verified machine-precision).
    DEFAULT_NONLINEAR_SOLVER_PARAMETERS = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_max_it": 15,
        "snes_rtol": 1e-6,
        "snes_atol": 1e-8,
        "snes_stol": 1e-10,
        "ksp_type": "gmres", "ksp_rtol": 1e-6,
        "pc_type": "gamg",
    }

    def _get_linear_solver(self, weak_form, Qnp1, Qaux_np1):
        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        sp_ = self.linear_solver_parameters or self.DEFAULT_LINEAR_SOLVER_PARAMETERS
        return fd.LinearVariationalSolver(problem, solver_parameters=dict(sp_))

    def _get_nonlinear_solver(self, weak_form, Qnp1, Qaux_np1):
        J = fd.derivative(weak_form, Qnp1)
        problem = fd.NonlinearVariationalProblem(weak_form, Qnp1, J=J)
        solver_parameters = (
            self.nonlinear_solver_parameters
            or self.DEFAULT_NONLINEAR_SOLVER_PARAMETERS
        )
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=dict(solver_parameters)
        )
        return solver

    def _build_problem(self, weak_form, Qnp1, with_jacobian: bool):
        if with_jacobian:
            J = fd.derivative(weak_form, Qnp1)
            return fd.NonlinearVariationalProblem(weak_form, Qnp1, J=J)
        else:
            return fd.NonlinearVariationalProblem(weak_form, Qnp1)  # no J

    def _get_solver_picard(self, weak_form, Qnp1, Qaux):
        problem = self._build_problem(weak_form, Qnp1, with_jacobian=False)
        picard_sp = {
            # Picard / fixed-point, matrix-free
            "snes_type": "nrichardson",
            "snes_mf_operator": True,
            "snes_max_it": 5,        # do up to 5 iterations

            # linear solve
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1.0e-6,

            # turn off convergence tests that cause DTOL
            "snes_rtol": 0.0,             # ignore relative residual
            "snes_atol": 0.0,             # ignore absolute residual
            "snes_stol": 0.0,             # ignore step tolerance
            "snes_dtol": 0.0,             # disable difference test

            # keep going even if SNES still complains
            "error_on_nonconvergence": False,
        }
        picard = fd.NonlinearVariationalSolver(problem, solver_parameters=picard_sp)

        def cb(snes, it, rnorm):
            self.update_Q(Qnp1, Qaux)
            self.update_Qaux(Qnp1, Qaux)
        picard.snes.setMonitor(cb)
        return picard

    def _get_solver_newton(self, weak_form, Qnp1, Qaux):
        problem = self._build_problem(weak_form, Qnp1, with_jacobian=True)

        newton_sp = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_linesearch_damping": 0.8,
            "snes_rtol": 1.0e-6,
            "snes_atol": 1.0e-8,
            "snes_dtol": 0.0,                 # disable DTOL
            "snes_max_it": 50,

            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_error_if_not_converged": False,
            "error_on_nonconvergence": False,
        }
        newton = fd.NonlinearVariationalSolver(problem, solver_parameters=newton_sp)

        def cb(snes, it, rnorm):
            self.update_Q(Qnp1, Qaux)
            self.update_Qaux(Qnp1, Qaux)
        newton.snes.setMonitor(cb)

        return newton

    # ==================================================================
    # Weak-form / solver registration (used by AMR subclass)
    # ==================================================================

    def _register_weak_forms(self, runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1,
                             n, mesh, map_boundary_tag_to_function_index,
                             sim_time, dt, x, x_3d):
        """Build and return a list of weak forms (convective, then source).

        This is the hook that ``FiredrakeHyperbolicSolverAMR.solve()`` expects.
        It also needs *Qs* and *Qaux_s* intermediates, so we create them here
        from the same function spaces.
        """
        V = Qn.function_space()
        Vaux = Qaux_n.function_space()
        Qs = fd.Function(V)
        Qaux_s = fd.Function(Vaux)

        wf_convective = self._get_weak_form_convective(
            runtime_model, Qn, Qs, Qaux_n, Qaux_s,
            n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d,
        )
        wf_source = self._get_weak_form_source(
            runtime_model, Qnp1, Qs, Qaux_np1, Qaux_s,
            n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d,
            theta=1.0,
        )
        return [wf_convective, wf_source]

    def _register_solvers(self, weak_forms, Qnp1, Qaux_np1):
        """Build solver objects from a list of weak forms.

        Returns a list of solvers in the same order as *weak_forms*.
        The first form (convective) uses a linear solver; the second (source)
        uses a nonlinear solver.  Additional forms are treated as nonlinear.
        """
        solvers = []
        for i, wf in enumerate(weak_forms):
            if i == 0:
                solvers.append(self._get_linear_solver(wf, Qnp1, Qaux_np1))
            else:
                solvers.append(self._get_nonlinear_solver(wf, Qnp1, Qaux_np1))
        return solvers

    # ==================================================================
    # setup_simulation / run_simulation / step
    # ==================================================================

    def setup_simulation(self, mesh_file, model, *, boundary_tag_map=None):
        """Set up all simulation state: mesh, model, spaces, forms, solvers.

        Stores state on ``self`` (via ``object.__setattr__`` since the class is
        frozen) so that ``run_simulation()`` and ``step()`` can access it.

        Parameters
        ----------
        mesh_file : str or fd.MeshGeometry
            Either a path to a Gmsh ``.msh`` file (legacy path) **or**
            an in-memory Firedrake mesh object (e.g. ``fd.IntervalMesh``).
            When an in-memory mesh is passed, supply ``boundary_tag_map``.
        model : zoomy_core Model
            Symbolic model with flux, source, boundary_conditions, etc.
        boundary_tag_map : dict[str, int], optional
            Required when ``mesh_file`` is an in-memory mesh.  Maps each
            BC tag name (as declared on ``model.boundary_conditions``)
            to the Firedrake physical-id of its facet group.
        """
        # -- Phase 1: Load mesh, freeze SystemModel, build runtimes --
        if isinstance(mesh_file, str):
            mesh = fd.Mesh(mesh_file)
            mesh_path = mesh_file
        else:
            mesh = mesh_file
            mesh_path = None
        if isinstance(model, SystemModel):
            raise NotImplementedError(
                "FiredrakeHyperbolicSolver currently requires a Model "
                "instance.  SystemModel-only ingestion would need a "
                "SystemModel-driven UFL runtime that handles every "
                "operator (boundary kernels included); the Model-based "
                "path is the only one that does so today.  Pass the "
                "Model — SystemModel.from_model is run internally."
            )
        system_model = SystemModel.from_model(model)
        # The Model-based UFL runtime stays the source of truth for the
        # per-operator UFL emissions (flux, source,
        # nonconservative_matrix, boundary kernels) — its lambdify path
        # already handles the Model.functions registry end-to-end.
        runtime_model = UFLRuntimeModel(model)

        # Symbolic Riemann solver → UFL runtime.  Wire ``max_wavespeed``
        # to a UFL builder that pulls the eigenvalues from the
        # Model-based runtime and takes the max of their absolute
        # values; this is the same handshake the numpy backend uses
        # (``solver_numpy.py`` line ~401).
        def _max_wavespeed_ufl(*args, _rt=runtime_model, _sm=system_model):
            n_eq = _sm.n_equations
            n_aux = len(_sm.aux_state)
            n_par = _sm.parameters.length()
            Q = ufl.as_vector(list(args[:n_eq]))
            Qaux = ufl.as_vector(list(args[n_eq:n_eq + n_aux])) \
                if n_aux > 0 else ufl.as_vector([fd.Constant(0.0)])
            p = ufl.as_vector(list(args[n_eq + n_aux:n_eq + n_aux + n_par]))
            n_vec = ufl.as_vector(list(args[n_eq + n_aux + n_par:]))
            ev = _rt.eigenvalues(Q, Qaux, p, n_vec)
            out = abs(ev[0])
            for i in range(1, n_eq):
                out = ufl.max_value(abs(ev[i]), out)
            return out

        from zoomy_core.transformation.to_ufl import UFLRuntimeSymbolic
        UFLRuntimeSymbolic.module["max_wavespeed"] = _max_wavespeed_ufl
        numerics = self.riemann_solver_cls(system_model)
        runtime_numerics = numerics.to_runtime_ufl()

        # Stub state — populated incrementally below so that
        # weak-form helpers reached during Phase 5 can already see
        # ``self._state.system_model`` / ``self._state.runtime_numerics``.
        limiter_exclude_indices = self._resolve_limiter_exclude_indices(
            system_model)
        object.__setattr__(self, "_state",
                           Zstruct(system_model=system_model,
                                   runtime_numerics=runtime_numerics,
                                   runtime_model=runtime_model,
                                   limiter_exclude_indices=limiter_exclude_indices))

        # -- Phase 2: Build function spaces --
        V, Vaux, Qnp1, Qs, Qn, Qaux_np1, Qaux_s, Qaux_n = (
            self._get_functionspaces(mesh, runtime_model)
        )

        # -- Phase 3: Set initial conditions --
        self.set_initial_condition(Qn, model)
        self.set_initial_condition(Qs, model)
        self.set_initial_condition(Qnp1, model)
        # Use model-driven updates when available
        self.update_Qaux(Qn, Qaux_n, runtime_model)
        self.update_Qaux(Qs, Qaux_s, runtime_model)
        self.update_Qaux(Qnp1, Qaux_np1, runtime_model)
        self.update_Q(Qn, Qaux_n, runtime_model)
        self.update_Q(Qs, Qaux_s, runtime_model)
        self.update_Q(Qnp1, Qaux_np1, runtime_model)

        # -- Phase 4: Boundary tag mapping --
        map_boundary_tag_to_function_index = (
            self.get_map_boundary_tag_to_boundary_function_index(
                model, mesh_path, mesh, boundary_tag_map=boundary_tag_map)
        )

        # -- Phase 5: Build weak forms --
        x, x_3d, n = self._get_x_and_n(mesh)
        dt = fd.Constant(0.1)
        sim_time = 0.0

        wf_convective = self._get_weak_form_convective(
            runtime_model, Qn, Qs, Qaux_n, Qaux_s,
            n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d,
        )
        wf_source = self._get_weak_form_source(
            runtime_model, Qnp1, Qs, Qaux_np1, Qaux_s,
            n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d,
            theta=1.0,
        )

        # -- Phase 6: Build solvers --
        solver_convective = self._get_linear_solver(wf_convective, Qs, Qaux_s)
        solver_source = self._get_nonlinear_solver(wf_source, Qnp1, Qaux_np1)

        # -- Phase 7: Build dt calculator --
        compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)

        # -- Store simulation state (frozen attrs -> object.__setattr__) --
        # NOTE: carry ``limiter_exclude_indices`` over from the stub
        # built at Phase 1 — otherwise the final state Zstruct drops
        # it and ``_apply_slope_limiter`` falls back to an empty
        # exclude set, re-introducing the b-flicker bug.
        state = Zstruct(
            mesh=mesh,
            runtime_model=runtime_model,
            system_model=system_model,
            runtime_numerics=runtime_numerics,
            limiter_exclude_indices=limiter_exclude_indices,
            model=model,
            V=V,
            Vaux=Vaux,
            Qn=Qn,
            Qs=Qs,
            Qnp1=Qnp1,
            Qaux_n=Qaux_n,
            Qaux_s=Qaux_s,
            Qaux_np1=Qaux_np1,
            map_boundary_tag_to_function_index=map_boundary_tag_to_function_index,
            dt=dt,
            sim_time=sim_time,
            compute_dt=compute_dt,
            solver_convective=solver_convective,
            solver_source=solver_source,
        )
        object.__setattr__(self, "_state", state)

    def step(self, dt_value):
        """Advance one time step using Lie splitting: convective -> source.

        Parameters
        ----------
        dt_value : float
            Time step size (already computed by the CFL condition).
        """
        s = self._state

        # Assign current state as the starting point
        s.Qn.assign(s.Qnp1)
        s.Qaux_n.assign(s.Qaux_np1)

        # Update the UFL Constant with the new dt
        s.dt.assign(dt_value)

        # --- Convective step (linear solve) ---
        s.solver_convective.solve()
        self._apply_slope_limiter(s.Qs)
        self.update_Q(s.Qs, s.Qaux_s, s.runtime_model)
        self.update_Qaux(s.Qs, s.Qaux_s, s.runtime_model)

        # --- Source step (nonlinear Newton solve) ---
        s.solver_source.solve()
        self._apply_slope_limiter(s.Qnp1)
        self.update_Q(s.Qnp1, s.Qaux_np1, s.runtime_model)
        self.update_Qaux(s.Qnp1, s.Qaux_np1, s.runtime_model)

    def run_simulation(self):
        """Run the time loop until ``time_end``.

        Requires ``setup_simulation()`` to have been called first.
        """
        s = self._state
        start_time = get_time()

        # Output setup
        main_dir = misc.get_main_directory()
        out = fd.VTKFile(
            os.path.join(main_dir, self.settings.output.directory, "simulation.pvd")
        )
        self.write_state(s.Qnp1, s.Qaux_n, out, time=s.sim_time)

        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot

        while s.sim_time < self.time_end:
            # Compute stable dt
            dt_value = s.compute_dt(s.Qnp1, s.Qaux_np1)

            # Advance one step
            self.step(dt_value)

            s.sim_time += dt_value
            iteration += 1

            # Write output at snapshot intervals
            if s.sim_time > next_write_time or s.sim_time >= self.time_end:
                next_write_time += dt_snapshot
                self.write_state(s.Qnp1, s.Qaux_np1, out, time=s.sim_time)

            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {int(iteration)}, time: {float(s.sim_time):.6f}, "
                    f"dt: {float(dt_value):.6f}, "
                    f"next write at time: {float(next_write_time):.6f}"
                )

        execution_time = get_time() - start_time
        logger.info(f"Finished simulation in {execution_time:.3f} seconds")
        # Expose final iteration count + wall time so a caller can
        # produce per-step throughput summaries without re-parsing the
        # log.  Kept on ``_state`` to avoid changing the public method
        # signature.
        s.last_iteration_count = int(iteration)
        s.last_execution_time = float(execution_time)

    # ==================================================================
    # Backward-compatible entry point
    # ==================================================================

    def solve(self, mshfile, model, *, boundary_tag_map=None):
        """Run a full simulation: setup + time loop.

        Equivalent to ``setup_simulation()`` followed by
        ``run_simulation()``.  Accepts either a ``.msh`` path or an
        in-memory ``fd.MeshGeometry`` (passes through to
        :meth:`setup_simulation`).
        """
        self.setup_simulation(mshfile, model, boundary_tag_map=boundary_tag_map)
        self.run_simulation()

    # ------------------------------------------------------------------
    # Legacy _setup (kept for AMR subclass compatibility)
    # ------------------------------------------------------------------

    def _setup(self, mshfile, model, *, boundary_tag_map=None):
        """Legacy setup returning a tuple of simulation objects.

        Prefer ``setup_simulation()`` for new code.  This method is retained
        so that ``FiredrakeHyperbolicSolverAMR`` continues to work.  Same
        mesh-input semantics as :meth:`setup_simulation` (path or
        ``fd.MeshGeometry``).
        """
        if isinstance(mshfile, str):
            mesh = fd.Mesh(mshfile)
            mesh_path = mshfile
        else:
            mesh = mshfile
            mesh_path = None
        if isinstance(model, SystemModel):
            raise NotImplementedError(
                "FiredrakeHyperbolicSolver requires a Model instance; "
                "SystemModel.from_model is called internally."
            )
        system_model = SystemModel.from_model(model)
        runtime_model = UFLRuntimeModel(model)

        # Symbolic Riemann solver → UFL runtime (mirrors setup_simulation).
        def _max_wavespeed_ufl(*args, _rt=runtime_model, _sm=system_model):
            n_eq = _sm.n_equations
            n_aux = len(_sm.aux_state)
            n_par = _sm.parameters.length()
            Q = ufl.as_vector(list(args[:n_eq]))
            Qaux = ufl.as_vector(list(args[n_eq:n_eq + n_aux])) \
                if n_aux > 0 else ufl.as_vector([fd.Constant(0.0)])
            p = ufl.as_vector(list(args[n_eq + n_aux:n_eq + n_aux + n_par]))
            n_vec = ufl.as_vector(list(args[n_eq + n_aux + n_par:]))
            ev = _rt.eigenvalues(Q, Qaux, p, n_vec)
            out = abs(ev[0])
            for i in range(1, n_eq):
                out = ufl.max_value(abs(ev[i]), out)
            return out

        from zoomy_core.transformation.to_ufl import UFLRuntimeSymbolic
        UFLRuntimeSymbolic.module["max_wavespeed"] = _max_wavespeed_ufl
        numerics = self.riemann_solver_cls(system_model)
        runtime_numerics = numerics.to_runtime_ufl()
        limiter_exclude_indices = self._resolve_limiter_exclude_indices(
            system_model)
        object.__setattr__(self, "_state",
                           Zstruct(system_model=system_model,
                                   runtime_numerics=runtime_numerics,
                                   runtime_model=runtime_model,
                                   limiter_exclude_indices=limiter_exclude_indices))

        V, Vaux, Qnp1, Qs, Qn, Qaux_np1, Qaux_s, Qaux_n = (
            self._get_functionspaces(mesh, runtime_model)
        )

        self.set_initial_condition(Qn, model)
        self.set_initial_condition(Qs, model)
        self.set_initial_condition(Qnp1, model)
        # Use model-driven updates when available
        self.update_Qaux(Qn, Qaux_n, runtime_model)
        self.update_Qaux(Qs, Qaux_s, runtime_model)
        self.update_Qaux(Qnp1, Qaux_np1, runtime_model)
        self.update_Q(Qn, Qaux_n, runtime_model)
        self.update_Q(Qs, Qaux_s, runtime_model)
        self.update_Q(Qnp1, Qaux_np1, runtime_model)

        # Collect all boundary tags
        map_boundary_tag_to_function_index = (
            self.get_map_boundary_tag_to_boundary_function_index(
                model, mesh_path, mesh, boundary_tag_map=boundary_tag_map)
        )

        return (mesh, runtime_model, V, Vaux, Qn, Qs, Qnp1,
                Qaux_n, Qaux_s, Qaux_np1, map_boundary_tag_to_function_index)


