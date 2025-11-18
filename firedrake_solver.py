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







@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolver:
    """Hyperbolic solver configuration for Firedrake."""

    # Scalar parameters
    CFL: float = field(default=0.45)
    time_end: float = field(default=0.1)
    
    # Nested struct with factory
    settings: Zstruct = field(factory=lambda: Settings.default())

    # Tensor factory (recomputed for each instance)
    IdentityMatrix = field(factory=lambda: ufl.as_tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, "settings", defaults)
        
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
        # Hydrostatic reconstruction
        Ql_star, Qr_star = self.hydrostatic_reconstruction(Ql, Qr)

        # Fluxes from reconstructed states
        flux_L = model.flux(Ql_star, Qauxl, parameters)
        flux_R = model.flux(Qr_star, Qauxr, parameters)

        central_flux = 0.5 * (flux_L + flux_R)

        # Wave speed (can still use original states for a robust bound)
        alpha_L = self.max_abs_eigenvalue(model, Ql_star, Qauxl, n, mesh)
        alpha_R = self.max_abs_eigenvalue(model, Qr_star, Qauxr, n, mesh)


        # LLF-type dissipation using reconstructed states
        num_flux = fd.dot(central_flux, n) - 0.5 * ufl.max_value(alpha_L, alpha_R) * fd.dot(self.IdentityMatrix, (Qr_star - Ql_star))

        return num_flux
    
    # def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, mesh):
    #     return fd.dot(
    #         0.5
    #         * (model.flux(Ql, Qauxl, parameters) + model.flux(Qr, Qauxr, parameters)),
    #         n,
    #     ) - 0.5 * self.max_abs_eigenvalue(model, Ql, Qauxl, n, mesh) * (Qr - Ql)
        

    def max_abs_eigenvalue(self, model, Q, Qaux, n, mesh):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(abs(ev[i]) > max_ev, abs(ev[i]), max_ev)
        return max_ev
    
    def write_state(self, q, qaux, out, time=0.0, names=None):
        mesh = q.function_space().mesh()
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
        
    def update_Qaux(self, Q, Qaux, eps=1e-8):
        h = Q.sub(1)
        eps = fd.Constant(1e-4)

        h_new  = fd.max_value(h, eps)
        h_inv = 1/(h_new)
        wet    = fd.conditional(h > eps, 1.0, 0.0)

        # Build the whole updated Q vector
        Qaux_new = fd.as_vector([
            h_inv,            # whatever this component is
        ])
        Qaux.interpolate(Qaux_new)


        
    def update_Q(self, Q, Qaux):
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
            Q.sub(0),            # whatever this component is
            Q.sub(1),               # updated h
            h_new * u_new,      # zero momentum if dry
            h_new * v_new,      # zero momentum if dry
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
        """

        V0 = fd.FunctionSpace(mesh, "DG", 0)
        h = fd.Function(V0).interpolate(fd.CellDiameter(mesh))
        dim = mesh.geometric_dimension()

        def compute_dt(Q, Qaux):
            """Compute global stable Δt for the given fields Q and Qaux."""

            # Compute eigenvalues in coordinate directions
            lam_x_expr = self.max_abs_eigenvalue(model, Q, Qaux, fd.as_vector([1.0, 0.0]), mesh)
            lam_y_expr = self.max_abs_eigenvalue(model, Q, Qaux, fd.as_vector([0.0, 1.0]), mesh)

            lam_x = fd.project(lam_x_expr, V0)
            lam_y = fd.project(lam_y_expr, V0)
            lam_local = fd.Function(V0)
            lam_local.dat.data[:] = np.maximum(lam_x.dat.data_ro, lam_y.dat.data_ro)

            # Local stable dt
            dt_local = fd.Function(V0).interpolate(CFL * (h/2) / (lam_local + 1e-8))

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
        
    def solve_with_callback(self, problem, Qnp1, Qaux, update_fn, solver_parameters=None):
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_parameters or {})

        # Access the underlying SNES object
        snes = solver.snes

        # Define callback
        def update_callback(snes, it, rnorm):
            update_fn(Qnp1, Qaux)

        # Attach it
        snes.setMonitor(update_callback)

        # Solve
        solver.solve()
        
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


        V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=runtime_model.n_variables)
        Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=runtime_model.n_aux_variables)
        Qn = fd.Function(V)
        Qnp1 = fd.Function(V)
        Qaux_n = fd.Function(Vaux)        
        Qaux_np1 = fd.Function(Vaux)

        self.set_initial_condition(Qn, model)
        self.set_initial_condition(Qnp1, model)
        self.update_Qaux(Qn, Qaux_n)
        self.update_Qaux(Qnp1, Qaux_np1)
        self.update_Q(Qn, Qaux_n)
        self.update_Q(Qnp1, Qaux_np1)
        
        # Collect all boundary tags
        map_boundary_tag_to_function_index = self.get_map_boundary_tag_to_boundary_function_index(model, mshfile, mesh)
        
        return mesh, runtime_model, V, Vaux, Qn, Qnp1, Qaux_n, Qaux_np1, map_boundary_tag_to_function_index 
    
    def _get_weak_form(self, runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1, n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d):
        test_q = fd.TestFunction(Qn.function_space())
        trial_q = fd.TrialFunction(Qn.function_space())
              
        Q = Qn
        Qaux = Qaux_n

        # linear problem
        # weak_form = fd.dot(test_q, (trial_q-Qn) / dt) * fd.dx      
        # nonlinear version
        weak_form = fd.dot(test_q, (Qnp1-Qn) / dt) * fd.dx

        
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

        theta = 0.0
        # source = runtime_model.source(theta*(Qnp1 + Qn), (1-theta)*(Qaux_np1+Qaux_n), runtime_model.parameters)
        # if not isinstance(source, ufl.constantvalue.Zero):
        #     weak_form -= fd.dot(
        #         test_q,
        #         source,
        #     ) * fd.dx
            
        return weak_form
    
    def _get_solver(self, weak_form, Qnp1, Qaux_np1):
        # a = fd.lhs(weak_form)
        # L = fd.rhs(weak_form)
        # problem = fd.LinearVariationalProblem(a, L, Qnp1)
        # solver = fd.LinearVariationalSolver(
        #     # problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "jacobi"}
        #     problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "lu"}

        # )
        
        J = fd.derivative(weak_form, Qnp1)
        problem = fd.NonlinearVariationalProblem(weak_form, Qnp1, J=J)
        solver  = fd.NonlinearVariationalSolver(problem,
                                                solver_parameters={
                                                    "snes_type": "newtonls",
                                                    "ksp_error_if_not_converged": True,
                                                    "ksp_type": "gmres",
                                                    "pc_type": "lu",
                                                    # "snes_type": "newtonls",
                                                    # "snes_linesearch_type": "bt",
                                                    # "snes_linesearch_damping": 0.8,
                                                    # "snes_max_it": 25,
                                                    # "snes_rtol": 1e-8,
                                                    # "snes_atol": 1e-10,
                                                    # "snes_stol": 1e-12,
                                                    # "ksp_type": "bcgs",
                                                    # "pc_type": "jacobi",
                                                },
                                                )
        
        # Access PETSc SNES
        snes = solver.snes

        def callback(snes, it, rnorm):
            # This is called each nonlinear iteration
            self.update_Q(Qnp1, Qaux_np1)
            self.update_Qaux(Qnp1, Qaux_np1)

        snes.setMonitor(callback)
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

    def solve(self, mshfile, model):
        mesh, runtime_model, V, Vaux, Qn, Qnp1, Qaux_n, Qaux_np1, map_boundary_tag_to_function_index = self._setup(mshfile, model)
        x, x_3d, n = self._get_x_and_n(mesh)
        
        compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)
        nc_flux = self.get_nonconservative_flux(runtime_model, runtime_model.parameters, mesh)
        sim_time = 0.0
        dt = fd.Constant(0.1)

        weak_form = self._get_weak_form(runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1, n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)
        
        # solver = self._get_solver(weak_form, Qnp1, Qaux)
        solver_picard = self._get_solver_picard(weak_form, Qnp1, Qaux_n)
        solver_newton = self._get_solver_newton(weak_form, Qnp1, Qaux_n)
        
        
        main_dir = misc.get_main_directory()
        out = fd.VTKFile(os.path.join(main_dir,self.settings.output.directory, "simulation.pvd"))
        self.write_state(Qnp1, Qaux_n, out, time=sim_time)
        dx_ref = mesh.cell_sizes.dat.data_ro.min()
        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot
        
        
        while sim_time < self.time_end:
            Qn.assign(Qnp1)
            self.update_Q(Qn, Qaux_n)
            self.update_Qaux(Qn, Qaux_n)
            dt.assign(compute_dt(Qn, Qaux_n))
            try:
                solver_picard.solve()
            except :
                logger.info("Picard pre-solve exited with DTOL; continuing with Newton")
            solver_newton.solve()
            self.update_Q(Qnp1, Qaux_n)
            sim_time += float(dt)
            iteration += 1
            if sim_time > next_write_time or sim_time >= self.time_end:
                next_write_time += dt_snapshot
                self.write_state(Qnp1, Qaux_n, out, time=sim_time)
            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {int(iteration)}, time: {float(sim_time):.6f}, "
                    f"dt: {float(dt):.6f}, next write at time: {float(next_write_time):.6f}"
                        )
                
        logger.info(f"Finished simulation with in {sim_time:.3f} seconds")
