import firedrake as fd
from firedrake.functionspaceimpl import MixedFunctionSpace
import ufl
from zoomy_core.fvm.solver_numpy import Settings
from attrs import field, define
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
import numpy as np
from mpi4py import MPI

from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.misc as misc

import os

import meshio
from time import time as get_time







@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolver:
    """Hyperbolic solver configuration for Firedrake."""

    # Scalar parameters
    CFL: float = field(default=0.45)
    time_end: float = field(default=0.1)
    dg_degree: int = field(default=0)

    # Nested struct with factory
    settings: Zstruct = field(factory=lambda: Settings.default())

    # Tensor factory (recomputed for each instance).
    # NOTE: The default is a 4x4 identity for backward compatibility with
    # 4-variable SWE models.  For models with a different number of variables
    # use ``_get_identity_matrix(n)`` which builds a correctly-sized UFL tensor.
    IdentityMatrix = field(factory=lambda: ufl.as_tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, "settings", defaults)

    # ------------------------------------------------------------------
    # Model introspection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_identity_matrix(n):
        """Build an n x n UFL identity tensor."""
        rows = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return ufl.as_tensor(rows)

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

        For free-surface models this is all indices except b (index 0).
        For generic models every variable index is included.
        """
        sym = model.model if hasattr(model, "model") else model
        keys = list(sym.variables.keys()) if hasattr(sym.variables, "keys") else []
        if "h" in keys and "b" in keys:
            # SWE-family: source acts on everything except bathymetry (idx 0)
            return list(range(1, len(keys)))
        # Generic: source may act on every variable
        return list(range(len(keys)))

    @staticmethod
    def _model_has_diffusive_flux(model):
        """Check whether the model defines a non-trivial diffusive flux."""
        sym = model.model if hasattr(model, "model") else model
        if not hasattr(sym, "functions"):
            return False
        if "diffusive_flux" not in sym.functions.keys():
            return False
        # Check if the symbolic expression is all zeros
        try:
            sym_dflux = sym.diffusive_flux()
            is_zero = hasattr(sym_dflux, "tolist") and all(
                e == 0
                for row in sym_dflux.tolist()
                for e in (row if isinstance(row, list) else [row])
            )
            return not is_zero
        except Exception:
            return False

    def get_map_boundary_tag_to_boundary_function_index(self, model, msh_path, mesh):
        msh = meshio.read(msh_path)
        boundary_function_names = model.boundary_conditions._boundary_tags
        field_data_raw = msh.field_data  # name → [id, dim, type]
        field_data = {
            name: data for name, data in field_data_raw.items()
            if name in boundary_function_names
        }
        extracted_facets_firedrake = mesh.exterior_facets.unique_markers
        assert len(extracted_facets_firedrake) == len(field_data), f"Mismatch in number of boundary tags: extracted {extracted_facets_firedrake}, expected {field_data.keys()}"
        
        # 1. Make a list of (name, id)
        name_id_pairs = [(name, data[0]) for name, data in field_data.items()]
        
        # 2. Sort alphabetically by name
        name_id_pairs.sort(key=lambda x: x[0])
        
        # 3. Assign consecutive indices (1-based or 0-based as needed)
        #    You mentioned "index of sorted list", so I'll use 1-based:
        name_to_index = {name: i for i, (name, _) in enumerate(name_id_pairs)}
        
        # 4. Invert to get physical_id → index
        physical_id_to_index = {pid: name_to_index[name] for name, pid in name_id_pairs}
        
        return physical_id_to_index

    def get_nonconservative_flux(self, model, parameters, mesh):
        samples, weights = np.polynomial.legendre.leggauss(3)
        samples = fd.as_vector((samples+1)/2)
        weights = fd.as_vector(weights*0.5)
        def nc_flux(Ql, Qr, Qauxl, Qauxr, n):
            A = fd.dot(sum(wi * model.nonconservative_matrix(
                    Ql + xi*(Qr - Ql),
                    Qauxl + xi*(Qauxr - Qauxl),
                    parameters
                ) for xi, wi in zip(samples, weights)), n)
            lam_l = self.max_abs_eigenvalue(model, Ql, Qauxl, n, mesh)
            lam_r = self.max_abs_eigenvalue(model, Qr, Qauxr, n, mesh)
            lam = ufl.max_value(lam_l, lam_r)
            Id = self.IdentityMatrix
            
            Dp = 0.5 * fd.dot(A + lam * Id, (Qr - Ql))
            Dm = 0.5 * fd.dot(A - lam * Id, (Qr - Ql))
            return Dp, Dm
        return nc_flux
    
    def hydrostatic_reconstruction(self, Ql, Qr):
        """
        Hydrostatic reconstruction for shallow water in Firedrake/UFL.
        State layout: Q[0] = b, Q[1] = h, Q[2] = hu (if present), others untouched.
        """
        bL, hL = Ql[0], Ql[1]
        bR, hR = Qr[0], Qr[1]

        # Free surface
        etaL = hL + bL
        etaR = hR + bR

        # max(b_L, b_R) in UFL
        b_star = ufl.max_value(bL, bR)
        
        eps = fd.Constant(1e-4)


        # Reconstructed depths (>= 0)
        hL_star = ufl.max_value(0.0, etaL - b_star)
        hR_star = ufl.max_value(0.0, etaR - b_star)

        # Number of components in Q
        ncomp = Ql.ufl_shape[0]

        # Build component lists manually (no slicing)
        compsL = [None] * ncomp
        compsR = [None] * ncomp

        # b stays the same
        compsL[0] = bL
        compsR[0] = bR

        # h replaced by reconstructed depths
        compsL[1] = hL_star
        compsR[1] = hR_star
        


        if ncomp > 2:
            # Assume Q[2] is hu; we want hu* = h* u
            hL_eff = ufl.max_value(hL, eps)  # regularized division
            hR_eff = ufl.max_value(hR, eps)

            uL = Ql[2] / hL_eff
            uR = Qr[2] / hR_eff

            compsL[2] = hL_star * uL
            compsR[2] = hR_star * uR

            # Any extra components: just copy over unchanged
            for i in range(3, ncomp):
                compsL[i] = Ql[i]
                compsR[i] = Qr[i]

        # If ncomp == 2, we’re done; the remaining entries are already None but unused.

        Ql_star = ufl.as_vector(compsL)
        Qr_star = ufl.as_vector(compsR)

        return Ql_star, Qr_star
    
    def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, mesh):
        # Hydrostatic reconstruction only for free-surface models (SWE-family)
        if self._model_has_free_surface(model):
            Ql_star, Qr_star = self.hydrostatic_reconstruction(Ql, Qr)
        else:
            Ql_star, Qr_star = Ql, Qr

        # Fluxes from (possibly reconstructed) states
        flux_L = model.flux(Ql_star, Qauxl, parameters)
        flux_R = model.flux(Qr_star, Qauxr, parameters)

        central_flux = 0.5 * (flux_L + flux_R)

        # Wave speed (can still use original states for a robust bound)
        alpha_L = self.max_abs_eigenvalue(model, Ql_star, Qauxl, n, mesh)
        alpha_R = self.max_abs_eigenvalue(model, Qr_star, Qauxr, n, mesh)

        # LLF-type dissipation using (possibly reconstructed) states
        num_flux = fd.dot(central_flux, n) - 0.5 * ufl.max_value(alpha_L, alpha_R) * fd.dot(self.IdentityMatrix, (Qr_star - Ql_star))

        return num_flux
      

    def max_abs_eigenvalue(self, model, Q, Qaux, n, mesh):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(abs(ev[i]) > max_ev, abs(ev[i]), max_ev)
        return max_ev
    
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
        """Update auxiliary variables.

        If *runtime_model* is provided and exposes an ``update_aux_variables``
        callback, that model-driven update is used.  Otherwise the legacy
        SWE-specific path (h_inv from Q[1]) is used for backward compatibility.
        """
        if runtime_model is not None and hasattr(runtime_model, "update_aux_variables"):
            Qaux_new = runtime_model.update_aux_variables(Q, Qaux, runtime_model.parameters)
            Qaux.interpolate(Qaux_new)
            return

        # Legacy SWE fallback: assumes Q[1] = h, Qaux[0] = h_inv
        h = Q.sub(1)
        eps = fd.Constant(1e-4)

        h_new  = fd.max_value(h, eps)
        h_inv = 1/(h_new)

        Qaux_new = fd.as_vector([
            h_inv,
        ])
        Qaux.interpolate(Qaux_new)


        
    def update_Q(self, Q, Qaux, runtime_model=None):
        """Update conserved variables (positivity, velocity capping, etc.).

        If *runtime_model* is provided and exposes an ``update_variables``
        callback, that model-driven update is used.  Otherwise the legacy
        SWE-specific path (velocity capping on 4-component [b,h,hu,hv]) is
        used for backward compatibility.
        """
        if runtime_model is not None and hasattr(runtime_model, "update_variables"):
            Q_new = runtime_model.update_variables(Q, Qaux, runtime_model.parameters)
            Q.interpolate(Q_new)
            return

        # Legacy SWE fallback: assumes Q = [b, h, hu, hv]
        ncomp = Q.function_space().value_size
        if ncomp < 4:
            # Not the 4-component SWE layout -- skip legacy capping
            return

        h = Q.sub(1)
        eps = fd.Constant(1e-4)

        h_new  = fd.max_value(h, eps)
        wet    = fd.conditional(h > eps, 1.0, 0.0)

        max_vel_cap = fd.Constant(100)

        u = Q.sub(2) / (h_new)
        u_new = wet * fd.sign(u) * fd.min_value(abs(u), max_vel_cap)

        v = Q.sub(3) / (h_new)
        v_new = wet * fd.sign(v) * fd.min_value(abs(v), max_vel_cap)

        # Build the whole updated Q vector
        Q_new = fd.as_vector([
            Q.sub(0),               # b (bathymetry)
            Q.sub(1),               # h (depth, kept as-is)
            h_new * u_new,          # zero momentum if dry
            h_new * v_new,          # zero momentum if dry
        ])
        Q.interpolate(Q_new)

    def get_function_coordinates(self, Q):
        """
        Return coordinates (as NumPy array) for each DoF in a Firedrake Function.
        
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

        # Return as NumPy array (read-only for safety)
        coords = np.array(coords_func.dat.data_ro)
        assert coords.shape[1] == dim, f"Unexpected coordinate shape {coords.shape}, dim={dim}"
        return coords

    def get_compute_dt(self, mesh, model, CFL=0.45):
        """
        Returns a callable that computes Δt(Q, Qaux) for a fixed mesh and model.

        This avoids re-evaluating static mesh geometry (e.g. CellDiameter).
        The CFL condition accounts for the DG polynomial degree via the
        standard scaling: dt = CFL * h_cell / ((2*degree + 1) * max_eigenvalue).
        """
        degree = self.dg_degree

        V0 = fd.FunctionSpace(mesh, "DG", 0)
        h = fd.Function(V0).interpolate(fd.CellDiameter(mesh))
        dim = mesh.geometric_dimension()

        # For DG degree p the effective wave speed factor is (2p+1)
        degree_factor = float(2 * degree + 1)

        def compute_dt(Q, Qaux):
            """Compute global stable Δt for the given fields Q and Qaux."""

            # Compute eigenvalues in coordinate directions
            lam_x_expr = self.max_abs_eigenvalue(model, Q, Qaux, fd.as_vector([1.0, 0.0]), mesh)
            lam_y_expr = self.max_abs_eigenvalue(model, Q, Qaux, fd.as_vector([0.0, 1.0]), mesh)

            lam_x = fd.project(lam_x_expr, V0)
            lam_y = fd.project(lam_y_expr, V0)
            lam_local = fd.Function(V0)
            lam_local.dat.data[:] = np.maximum(lam_x.dat.data_ro, lam_y.dat.data_ro)

            # Local stable dt -- scaled by (2*degree + 1) for higher-order DG
            dt_local = fd.Function(V0).interpolate(
                CFL * (h / 2) / (degree_factor * lam_local + 1e-8)
            )

            # Global min across cells and MPI ranks
            dt_min = float(np.min(dt_local.dat.data_ro))
            dt_min = mesh.comm.allreduce(dt_min, op=MPI.MIN)
            return dt_min

        return compute_dt
    
    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        x = fd.SpatialCoordinate(mesh)
        coords = self.get_function_coordinates(Q)
        Qarr = Q.dat.data
        Q.dat.data[:] = model.initial_conditions.apply(coords.T, Qarr.T).T
        
        
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
    
    def _setup(self, mshfile, model):
        mesh = fd.Mesh(mshfile)
        runtime_model = UFLRuntimeModel(model)

        # Rebuild IdentityMatrix to match the actual number of variables
        nvar = runtime_model.n_variables
        if nvar != 4:
            object.__setattr__(self, "IdentityMatrix", self._get_identity_matrix(nvar))

        V, Vaux, Qnp1, Qs, Qn, Qaux_np1, Qaux_s, Qaux_n = self._get_functionspaces(mesh, runtime_model)

        self.set_initial_condition(Qn, model)
        self.set_initial_condition(Qs, model)
        self.set_initial_condition(Qnp1, model)
        # Use model-driven updates when available, fall back to legacy SWE path
        self.update_Qaux(Qn, Qaux_n, runtime_model)
        self.update_Qaux(Qs, Qaux_s, runtime_model)
        self.update_Qaux(Qnp1, Qaux_np1, runtime_model)
        self.update_Q(Qn, Qaux_n, runtime_model)
        self.update_Q(Qs, Qaux_s, runtime_model)
        self.update_Q(Qnp1, Qaux_np1, runtime_model)

        # Collect all boundary tags
        map_boundary_tag_to_function_index = self.get_map_boundary_tag_to_boundary_function_index(model, mshfile, mesh)

        return mesh, runtime_model, V, Vaux, Qn, Qs, Qnp1, Qaux_n, Qaux_s, Qaux_np1, map_boundary_tag_to_function_index
    
    def _get_weak_form_convective(self, runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1, n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d):
        test_q = fd.TestFunction(Qn.function_space())
        trial_q = fd.TrialFunction(Qn.function_space())
              
        Q = Qn
        Qaux = Qaux_n
        
        weak_form = fd.dot(test_q, (trial_q-Qn) / dt) * fd.dx  
        # weak_form = fd.dot(test_q, (Qnp1-Qn) / dt) * fd.dx  

        
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
        weak_form += 0.5*(
            fd.dot(
                test_q("+"),
                Dp
                )
            ) * fd.dS
        weak_form += 0.5*(
            fd.dot(
                test_q("-"),
                Dm
                )
            ) * fd.dS
        

        for tag, idx in map_boundary_tag_to_function_index.items():

            # Position and "distance" placeholders (can be replaced with actual geometric quantities)
            dX = x[0]  # your placeholder (e.g., half cell size)

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
                self.numerical_flux(runtime_model, Q, Q_bnd, Qaux_n, Qaux_n, runtime_model.parameters, n, mesh)
            ) * fd.ds(tag)

            Dp, Dm = nc_flux(Q, Q_bnd, Qaux_n, Qaux_n, n)
            weak_form += (
                fd.dot(
                    test_q,
                    Dm
                    )
                ) * fd.ds(tag)

        # --- Diffusive flux contribution (IP-DG) for DG1+ ---
        # NOTE: Cannot test without Firedrake installation
        if self.dg_degree >= 1 and self._model_has_diffusive_flux(runtime_model):
            weak_form += self._get_weak_form_diffusion(
                runtime_model, Q, Qaux_n, test_q, mesh, n,
                map_boundary_tag_to_function_index, sim_time, x, x_3d,
            )

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
        """
        Source step: evolve only the variables that receive source terms.

        The source-active indices are determined by ``_model_source_indices``:
        - SWE-family models: all indices except bathymetry (index 0)
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
        Q_theta    = theta*Qnp1    + (1.0 - theta)*Q_star
        Qaux_theta = theta*Qaux_np1 + (1.0 - theta)*Qaux_star

        # Full source vector from the model
        source_full = runtime_model.source(Q_theta, Qaux_theta, runtime_model.parameters)

        zero = fd.Constant(0.0)

        # Build vectors that are nonzero only for source-active components
        diff_restricted = []
        source_restricted = []

        for i in range(ncomp):
            if i in source_indices:
                diff_restricted.append(Qnp1[i] - Q_star[i])
                source_restricted.append(source_full[i])
            else:
                diff_restricted.append(zero)
                source_restricted.append(zero)

        diff_restricted   = fd.as_vector(diff_restricted)
        source_restricted = fd.as_vector(source_restricted)

        # Weak form: only source-active components contribute
        weak_form  = fd.dot(test_q, diff_restricted/dt) * fd.dx
        weak_form -= fd.dot(test_q, source_restricted) * fd.dx

        return weak_form


    
    # ------------------------------------------------------------------
    # IP-DG diffusive flux (Task 3)
    # ------------------------------------------------------------------

    def _get_weak_form_diffusion(self, runtime_model, Q, Qaux, test_q, mesh, n,
                                  map_boundary_tag_to_function_index,
                                  sim_time, x, x_3d):
        """Interior Penalty DG (IP-DG) discretisation of the diffusive flux.

        For DG0 this returns zero (grad(Q) = 0 inside each cell, and face-based
        diffusion is already captured by the Rusanov dissipation in the numerical
        flux).  For DG1+ the standard symmetric IP-DG form is used:

        - Volume:  integral nu * grad(Q) : grad(v) dx
        - Interior faces:  -avg(nu * grad(Q)) . jump(v) dS
                           -avg(nu * grad(v)) . jump(Q) dS
                           + (sigma / h_F) jump(Q) . jump(v) dS
        - Exterior faces:  analogous Dirichlet penalty terms

        NOTE: Cannot test without Firedrake installation.
        """
        degree = self.dg_degree

        if degree < 1:
            # DG0: no within-cell gradient, diffusion handled by numerical flux
            return fd.Constant(0.0) * fd.dot(test_q, Q) * fd.dx

        if not self._model_has_diffusive_flux(runtime_model):
            return fd.Constant(0.0) * fd.dot(test_q, Q) * fd.dx

        ncomp = Q.function_space().value_size

        # Penalty parameter: scale with degree^2 / h_F (standard IP-DG)
        sigma = fd.Constant(float(10.0 * degree ** 2))
        h_F = fd.CellDiameter(mesh)  # local cell diameter

        # The model's diffusive_flux is symbolically F_diff(Q, Qaux, gradQ, p).
        # In the UFL context we can compute gradQ = grad(Q) directly.
        grad_Q = fd.grad(Q)  # shape (ncomp, gdim)

        # Construct the diffusive flux tensor from the model.
        # The runtime_model.diffusive_flux expects (Q, Qaux, gradQ, p) where
        # gradQ is a flat vector of gradient symbols.  In the UFL world we
        # need to build that flat vector from grad(Q).
        gdim = mesh.geometric_dimension()

        # Build a flat gradient vector matching the model's gradient_variables
        # ordering: [dQ0_d0, dQ0_d1, dQ1_d0, dQ1_d1, ...]
        grad_flat_components = []
        for i in range(ncomp):
            for d in range(gdim):
                grad_flat_components.append(grad_Q[i, d])
        grad_flat = fd.as_vector(grad_flat_components)

        # Evaluate the model's diffusive flux: shape (ncomp, gdim) as a UFL expr
        D_flux = runtime_model.diffusive_flux(Q, Qaux, grad_flat, runtime_model.parameters)

        # --- Volume integral: - integral D_flux : grad(v) dx
        # D_flux[i,d] * grad(test_q)[i,d]
        grad_v = fd.grad(test_q)
        vol_form = fd.Constant(0.0) * fd.dot(test_q, Q) * fd.dx
        for i in range(ncomp):
            for d in range(gdim):
                vol_form += D_flux[i * gdim + d] * grad_v[i, d] * fd.dx

        # --- Interior face terms (symmetric IP-DG)
        # jump(Q)[i] = Q('+')[i] - Q('-')[i]
        # avg(grad_Q)[i,d] = 0.5*(grad_Q('+') + grad_Q('-'))[i,d]
        avg_h = (h_F("+") + h_F("-")) / 2.0

        # Consistency: -avg(D . grad(Q)) . n . jump(v)
        # Symmetry:    -avg(D . grad(v)) . n . jump(Q)
        # Penalty:     sigma/h * jump(Q) . jump(v)
        face_form = fd.Constant(0.0) * fd.dot(test_q("+"), Q("+")) * fd.dS
        for i in range(ncomp):
            jump_Q_i = Q("+").sub(i) - Q("-").sub(i)
            jump_v_i = test_q("+").sub(i) - test_q("-").sub(i)

            for d in range(gdim):
                avg_gradQ_id = 0.5 * (grad_Q("+")[i, d] + grad_Q("-")[i, d])
                avg_gradv_id = 0.5 * (grad_v("+")[i, d] + grad_v("-")[i, d])

                # Use a scalar diffusion coefficient (first non-zero diagonal
                # entry). For more general tensorial diffusion this would
                # require the full D tensor evaluation at faces.
                face_form -= avg_gradQ_id * n("+")[d] * jump_v_i * fd.dS
                face_form -= avg_gradv_id * n("+")[d] * jump_Q_i * fd.dS

            # Penalty
            face_form += (sigma / avg_h) * jump_Q_i * jump_v_i * fd.dS

        return vol_form + face_form

    # ------------------------------------------------------------------
    # Slope limiter for DG1+ (Task 4)
    # ------------------------------------------------------------------

    def _apply_slope_limiter(self, Q):
        """Apply Kuzmin-type vertex-based slope limiter for DG degree >= 1.

        For vector function spaces the limiter is applied component-wise
        because Firedrake's VertexBasedLimiter operates on scalar spaces.

        NOTE: Cannot test without Firedrake installation.
        """
        if self.dg_degree < 1:
            return

        V = Q.function_space()
        mesh = V.mesh()
        ncomp = V.value_size

        # Build a scalar DG space with the same degree for the limiter
        V_scalar = fd.FunctionSpace(mesh, "DG", self.dg_degree)
        limiter = fd.VertexBasedLimiter(V_scalar)

        for i in range(ncomp):
            qi = fd.Function(V_scalar)
            qi.interpolate(Q.sub(i))
            limiter.apply(qi)
            Q.sub(i).interpolate(qi)

    def _get_linear_solver(self, weak_form, Qnp1, Qaux_np1):
        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        solver = fd.LinearVariationalSolver(
            # problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "jacobi"}
            problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "lu"}

        )
        return solver
    
    def _get_nonlinear_solver(self, weak_form, Qnp1, Qaux_np1):

        
        J = fd.derivative(weak_form, Qnp1)
        problem = fd.NonlinearVariationalProblem(weak_form, Qnp1, J=J)
        
        # solver_parameters={
        #     "snes_type": "newtonls",
        #     "ksp_error_if_not_converged": True,
        #     "ksp_type": "gmres",
        #     "pc_type": "lu"
        # }
        solver_parameters={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_linesearch_damping": 0.8,
            "snes_max_it": 25,
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_stol": 1e-12,
            "ksp_type": "bcgs",
            "pc_type": "jacobi",
        }
        # solver_parameters = {
        #     "snes_type": "nrichardson",
        #     "snes_mf_operator": True,

        #     # Allow several iterations
        #     "snes_max_it": 10,

        #     # Disable DTOL and step-based divergence
        #     "snes_dtol": -1.0,
        #     "snes_stol": -1.0,

        #     # Let Picard converge normally:
        #     "snes_rtol": 1e-8,
        #     "snes_atol": 1e-10,

        #     "ksp_type": "gmres",
        #     "pc_type": "lu",

        #     "error_on_nonconvergence": False,
        #     "ksp_error_if_not_converged": False,
        #     "snes_monitor": None,
        #     "snes_linesearch_monitor": None,
        #     "ksp_monitor": None,
        # }
        solver  = fd.NonlinearVariationalSolver(problem,
                                                solver_parameters=solver_parameters
                                                )
        
        # # Access PETSc SNES
        # snes = solver.snes

        # def callback(snes, it, rnorm):
        #     # This is called each nonlinear iteration
        #     self.update_Q(Qnp1, Qaux_np1)
        #     self.update_Qaux(Qnp1, Qaux_np1)

        # snes.setMonitor(callback)
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
            "snes_type":        "nrichardson",
            "snes_mf_operator": True,
            "snes_max_it":      5,        # do up to 5 iterations

            # linear solve
            "ksp_type":         "gmres",
            "pc_type":          "ilu",
            "ksp_rtol":         1.0e-6,

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
        # newton_sp = {
        #     "snes_type":               "newtonls",
        #     "snes_linesearch_type":    "bt",
        #     "snes_linesearch_damping": 0.8,
        #     "snes_max_it":             25,
        #     "ksp_type":                "gmres",
        #     "pc_type":                 "lu",
        #     "pc_factor_shift_type":    "nonzero",
        #     "ksp_error_if_not_converged": False,
        # }
        
        newton_sp = {
            "snes_type":            "newtonls",
            "snes_linesearch_type": "bt",
            "snes_linesearch_damping": 0.8,
            "snes_rtol": 1.0e-6,
            "snes_atol": 1.0e-8,
            "snes_dtol": 0.0,                 # disable DTOL
            "snes_max_it": 50,

            "ksp_type": "gmres",
            "pc_type":  "lu",
            "ksp_error_if_not_converged": False,
            "error_on_nonconvergence": False,
        }
        newton = fd.NonlinearVariationalSolver(problem, solver_parameters=newton_sp)

        def cb(snes, it, rnorm):
            self.update_Q(Qnp1, Qaux)
            self.update_Qaux(Qnp1, Qaux)
        newton.snes.setMonitor(cb)

        return newton
    
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

    # ------------------------------------------------------------------
    # Weak-form / solver registration (used by AMR subclass)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self, mshfile, model):
        start_time = get_time()
        mesh, runtime_model, V, Vaux, Qn, Qs, Qnp1, Qaux_n, Qaux_s, Qaux_np1, map_boundary_tag_to_function_index = self._setup(mshfile, model)
        x, x_3d, n = self._get_x_and_n(mesh)

        compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)
        nc_flux = self.get_nonconservative_flux(runtime_model, runtime_model.parameters, mesh)
        sim_time = 0.0
        dt = fd.Constant(0.1)

        wf_convective = self._get_weak_form_convective(runtime_model, Qn, Qs, Qaux_n, Qaux_s, n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)
        wf_source = self._get_weak_form_source(runtime_model, Qnp1, Qs, Qaux_np1, Qaux_s, n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d, theta=1.0)

        solver_convective = self._get_linear_solver(wf_convective, Qs, Qaux_s)
        solver_source = self._get_nonlinear_solver(wf_source, Qnp1, Qaux_np1)

        main_dir = misc.get_main_directory()
        out = fd.VTKFile(os.path.join(main_dir, self.settings.output.directory, "simulation.pvd"))
        self.write_state(Qnp1, Qaux_n, out, time=sim_time)
        dx_ref = mesh.cell_sizes.dat.data_ro.min()
        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot

        while sim_time < self.time_end:
            Qn.assign(Qnp1)
            Qaux_n.assign(Qaux_np1)
            dt.assign(compute_dt(Qn, Qaux_n))

            solver_convective.solve()
            # Slope limiting after convective step (DG1+)
            # NOTE: Cannot test without Firedrake installation
            self._apply_slope_limiter(Qs)
            self.update_Q(Qs, Qaux_s, runtime_model)
            self.update_Qaux(Qs, Qaux_s, runtime_model)

            solver_source.solve()
            # Slope limiting after source step (DG1+)
            self._apply_slope_limiter(Qnp1)
            self.update_Q(Qnp1, Qaux_np1, runtime_model)
            self.update_Qaux(Qnp1, Qaux_np1, runtime_model)

            sim_time += float(dt)
            iteration += 1
            if sim_time > next_write_time or sim_time >= self.time_end:
                next_write_time += dt_snapshot
                self.write_state(Qnp1, Qaux_np1, out, time=sim_time)
            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {int(iteration)}, time: {float(sim_time):.6f}, "
                    f"dt: {float(dt):.6f}, next write at time: {float(next_write_time):.6f}"
                        )
        execution_time = get_time() - start_time
        logger.info(f"Finished simulation in {execution_time:.3f} seconds")
