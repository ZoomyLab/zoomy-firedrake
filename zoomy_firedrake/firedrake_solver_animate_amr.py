import os
import traceback

import firedrake as fd
from firedrake.functionspaceimpl import MixedFunctionSpace
import ufl
import numpy as np
from mpi4py import MPI

from attrs import field, define

from time import time as get_time
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.misc as misc
import zoomy_firedrake.firedrake_solver as fd_solver

from pyop2 import op2

from firedrake_animate.utility import VTKFile
from firedrake_animate.metric import RiemannianMetric, P0Metric
from firedrake_animate.adapt import adapt

from firedrake.projection import Projector
from firedrake.petsc import PETSc




# ---------------------------------------------------------------------------
# Helper: grow a DG0 indicator by a given number of cell layers
# ---------------------------------------------------------------------------

def grow_indicator(ind_dg0: fd.Function, width: int = 1) -> fd.Function:
    """
    Return a new DG-0 Function in which the set {value==1} has been
    dilated by exactly 'width' cell layers.

    Strategy: vertex averaging (DG0 → CG1) followed by thresholding.
    A cell is added *only* if tmp_dg > 0.49 (all vertices already touched).
    """
    V0 = ind_dg0.function_space()
    mesh = V0.mesh()
    Vcg = fd.FunctionSpace(mesh, "CG", 1)

    grown = fd.Function(V0, name="grown_indicator")
    grown.assign(ind_dg0)

    for _ in range(width):
        tmp_cg = fd.project(grown, Vcg)   # DG0 → CG1
        tmp_dg = fd.project(tmp_cg, V0)   # CG1 → DG0
        grown.interpolate(fd.conditional(tmp_dg > 0.49, 1.0, grown))
    return grown


@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolverAMR(fd_solver.FiredrakeHyperbolicSolver):
    refine_every: int = field(default=20)      # time steps between refinements
    enable_amr: bool = field(default=True)    # total refinement steps

    # ------------------------------------------------------------------
    # Metric helpers based on RiemannianMetric
    # ------------------------------------------------------------------

    def _metric_wet_dry_smooth(self, Q, h_dry=1.0e-4, width=3, gamma=20.0):
        """
        Wet–dry metric using RiemannianMetric:

        - Start from DG0 indicator (wet = 1, dry = 0)
        - Grow by 'width' layers
        - Lift to CG1 and scale
        - Build isotropic RiemannianMetric via compute_isotropic_metric
        """
        mesh = Q.function_space().mesh()
        V0 = fd.FunctionSpace(mesh, "DG", 0)
        V1 = fd.FunctionSpace(mesh, "CG", 1)

        h = Q.sub(1)

        # DG0 wet indicator
        indicator = fd.Function(V0, name="wet_indicator")
        indicator.interpolate(fd.conditional(h > h_dry, 1.0, 0.0))

        # Grow by 'width' layers
        indicator = grow_indicator(indicator, width)

        # Lift to CG1
        smooth_indicator = fd.Function(V1, name="smooth_indicator")
        smooth_indicator.interpolate(indicator)

        # Build isotropic Riemannian metric
        metric = RiemannianMetric(mesh, name="wet_dry_metric")
        # 1 + gamma * indicator ∈ [1, 1+gamma]
        scaled = fd.Function(V1)
        scaled.interpolate(1.0 + gamma * smooth_indicator)
        metric.compute_isotropic_metric(scaled, interpolant="L2")

        # Enforce SPD + size / anisotropy bounds
        metric.enforce_spd(restrict_anisotropy=True, restrict_sizes=True)

        return metric, smooth_indicator

    def _metric_bathymetry(self, Q, grad_thres=0.05, L_ref=0.3, L_bkg=1.5, L0=1.0):
        """
        Bathymetry-based refinement metric:

        - Detect large |grad b|
        - Build indicator
        - Turn into isotropic RiemannianMetric with a length scale
          transition L_bkg -> L_ref in regions of high gradient.
        """
        mesh = Q.function_space().mesh()
        b = Q.sub(0)

        # CG1 bathymetry
        Vcg1 = fd.FunctionSpace(mesh, "CG", 1)
        b_cg = fd.interpolate(b, Vcg1)

        gnorm = fd.Function(Vcg1)
        gnorm.interpolate(fd.sqrt(fd.inner(fd.grad(b_cg), fd.grad(b_cg))))

        V0 = fd.FunctionSpace(mesh, "DG", 0)
        ind = fd.Function(V0)
        ind.interpolate(fd.conditional(gnorm > grad_thres, 1.0, 0.0))

        # Use a smooth CG1 indicator
        ind_cg = fd.Function(Vcg1)
        ind_cg.interpolate(ind)

        # Scale lengths by L0
        L_ref *= L0
        L_bkg *= L0

        # L field: smaller where ind ~ 1 (refinement)
        L = fd.Function(Vcg1)
        L.interpolate(L_bkg - ind_cg * (L_bkg - L_ref))

        # Metric weight ~ 1 / L^2
        weight = fd.Function(Vcg1)
        eps = 1.0e-6
        weight.interpolate(1.0 / (L * L + eps))

        metric = RiemannianMetric(mesh, name="bathymetry_metric")
        metric.compute_isotropic_metric(weight, interpolant="L2")
        metric.enforce_spd(restrict_anisotropy=True, restrict_sizes=True)
        return metric
    
    def _metric_shock(self, Q,
                  gamma=20.0,            # refinement factor between weak/strong shocks
                  aniso=0.5,             # anisotropy ratio h_tang / h_norm (0<aniso≤1)
                  grad_eps=1.0e-12):     # to avoid division by zero
        """
        Build an anisotropic metric that refines across steep fronts of h.

        Parameters
        ----------
        Q      : Vector Function with components [b, h, hu, hv, …]
        gamma  : float   – maximal contrast in edge length between regions
        aniso  : float   – ratio of tangential vs. normal edge length (≤1)
        grad_eps : float – floor for |∇h| when determining directions
        """
        mesh = Q.function_space().mesh()
        dim  = mesh.geometric_dimension()     # expect 2

        # --- 1. Compute gradient of h on CG1 for smoothness -------------
        h_dg  = Q.sub(1)                      # DG0
        Vcg1  = fd.FunctionSpace(mesh, "CG", 1)
        h_cg  = fd.interpolate(h_dg, Vcg1)

        grad_h = fd.grad(h_cg)
        grad_sq= fd.Function(Vcg1)
        grad_sq.interpolate(fd.inner(grad_h, grad_h))  # |∇h|²

        # --- 2. Bring everything to DG0 ---------------------------------
        V0  = fd.FunctionSpace(mesh, "DG", 0)
        gmag = fd.Function(V0)
        gmag.interpolate(fd.sqrt(grad_sq))    # |∇h| in DG0

        # normal vector components in DG0
        gx_dg = fd.Function(V0); gy_dg = fd.Function(V0)
        gx_dg.interpolate(grad_h[0]); gy_dg.interpolate(grad_h[1])

        # --- 3. Normalise gradient to get shock normal ------------------
        gx = gx_dg.dat.data_ro
        gy = gy_dg.dat.data_ro
        mag= gmag.dat.data_ro

        # --- 4. Prepare metric function ---------------------------------
        Vten = fd.TensorFunctionSpace(mesh, "DG", 0, shape=(dim, dim))
        M    = P0Metric(Vten)
        data = M.dat.data
        I2   = np.eye(dim)

        # global max gradient (MPI-safe) – used for scaling
        gmax = mesh.comm.allreduce(float(mag.max()), op=MPI.MAX)
        if gmax < grad_eps:
            # flow is essentially smooth everywhere; return coarse metric
            data[:] = I2
            return M

        for c in range(mesh.num_cells()):
            g = mag[c]
            if g < grad_eps:
                data[c,:,:] = I2                        # no shock
                continue

            # normalised normal vector
            nx = gx[c]/g
            ny = gy[c]/g
            nvec = np.array([nx, ny])
            tvec = np.array([-ny, nx])                 # tangential direction

            # scale factor grows with normalised gradient strength
            s = g / gmax               # 0 … 1
            h_norm  = 1.0 / (1.0 + gamma*s)            # small across shock
            h_tang  = h_norm / aniso                   # larger along front

            lam_n = 1.0 / h_norm**2
            lam_t = 1.0 / h_tang**2

            data[c,:,:] = lam_n*np.outer(nvec, nvec) + \
                        lam_t*np.outer(tvec, tvec)

        return M            # already a P0Metric, no extra project/normalise here

    def _metric_wave(self, Q, gamma=1.0, aniso=1.0):
        """
        Build a flow-aligned anisotropic Riemannian metric.

        Parameters:
        -----------
        Q : Function
            Current solution vector (assumed DG/VF with [h, hu, hv] ordering).
        gamma : float
            Controls refine strength between low- and high-speed flow.
            - Higher gamma means more contrast in scale between fast/slow regions.
        aniso : float in (0,1]
            Controls directional anisotropy.
            - aniso = 1.0 → isotropic metric.
            - aniso < 1.0 → cells are stretched along flow direction.
        """
        mesh = Q.function_space().mesh()
        h, hu, hv = Q.sub(1), Q.sub(2), Q.sub(3)

        # Compute velocity (u, v) and speed
        u = fd.Function(h.function_space(), name="u").interpolate(hu / (h + 1e-14))
        v = fd.Function(h.function_space(), name="v").interpolate(hv / (h + 1e-14))
        speed = np.sqrt(u.dat.data_ro**2 + v.dat.data_ro**2)

        # Global normalized magnitude for refinement scaling
        max_speed = mesh.comm.allreduce(float(speed.max()), op=MPI.MAX)
        norm_speed = speed / max_speed if max_speed > 0 else speed  # ∈ [0,1]

        # Metric in DG0
        Vten0 = fd.TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2))
        M_p0 = P0Metric(Vten0)

        for c in range(mesh.num_cells()):
            s = norm_speed[c]
            if s == 0.0:
                # Isotropic tensor (unrefined base)
                M_p0.dat.data[c, :, :] = np.eye(2)
                continue

            # Velocity direction
            nx, ny = u.dat.data_ro[c] / (speed[c] + 1e-14), v.dat.data_ro[c] / (speed[c] + 1e-14)
            vdir = np.array([nx, ny])
            tdir = np.array([-ny, nx])

            # Scaling based on speed: 1 + gamma*s ∈ [1, 1+gamma]
            scale = 1.0 + gamma * s

            # Anisotropic eigenvalues (stream aligned smaller = more refined)
            lam_stream = scale * aniso
            lam_cross = scale

            M_p0.dat.data[c, :, :] = (
                lam_stream * np.outer(vdir, vdir) +
                lam_cross * np.outer(tdir, tdir)
            )

        # Project to P1 Metric
        metric = RiemannianMetric(mesh, name="wave_metric")
        metric.project(M_p0)

        # Enforce positive-definite constraints
        metric.enforce_spd(restrict_anisotropy=True, restrict_sizes=True)
        return metric


    # ------------------------------------------------------------------
    # Prolongation: conservative + positivity preserving
    # ------------------------------------------------------------------

    def _prolong_vector_function(self, old_fun, new_space,
                                 h_dry=0, pos_idx=[1], clip=True):
        """
        Conservatively prolong vector field old_fun into new_space
        while enforcing positivity (e.g., for water height).
        """
        new_fun = fd.Function(new_space)

        for i in range(old_fun.function_space().value_size):
            comp_old = old_fun.sub(i)
            V_old_scalar = comp_old.function_space()
            tmp_old = fd.Function(V_old_scalar)
            tmp_old.assign(comp_old)

            comp_new = new_fun.sub(i)

            projector = Projector(tmp_old, comp_new)
            projector.project()

            # Enforce positivity for specific components
            if clip and i in pos_idx:
                comp_new.dat.data[:] = np.maximum(comp_new.dat.data_ro, h_dry)

        return new_fun

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def write_state(self, q, qaux, vtk_file, time=0.0, names=None):
        """
        Write state variables q, qaux to a VTK adaptive file.
        Each call writes a full snapshot at a given time.
        """
        mesh = q.function_space().mesh()
        V_scalar = fd.FunctionSpace(mesh, "DG", 0)
        n_q = q.function_space().value_size
        n_aux = qaux.function_space().value_size

        fields = [
            fd.project(q[i], V_scalar, name=names[i] if names else f"Q{i}")
            for i in range(n_q)
        ] + [
            fd.project(qaux[i], V_scalar, name=names[n_q + i] if names else f"Qaux{i}")
            for i in range(n_aux)
        ]

        vtk_file.write(*fields, time=time)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, mshfile, model):
        
        start_time = get_time()
        # ----- 1. Setup -----
        mesh, runtime_model, V, Vaux, Qn, Qs, Qnp1, Qaux_n, Qaux_s, Qaux_np1, map_boundary_tag_to_function_index = \
            self._setup(mshfile, model)
        x, x_3d, n = self._get_x_and_n(mesh)

        compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)
        sim_time = 0.0
        dt = fd.Constant(0.1)


        # ---- 2. Output setup ----
        main_dir = misc.get_main_directory()
        out = VTKFile(os.path.join(main_dir, self.settings.output.directory,
                                   "simulation.pvd"))
        out_id = VTKFile(os.path.join(main_dir, self.settings.output.directory,
                                      "indicator.pvd"))
        out_metric = VTKFile(os.path.join(main_dir, self.settings.output.directory,
                                          "metric.pvd"))

        # Write initial snapshot at t=0
        self.write_state(Qnp1, Qaux_n, out, time=sim_time)


        # ---- 3. Main time loop with AMR ----
        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot
        
        V1 = fd.FunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(mesh, name="initial_metric")
        scaled = fd.Function(V1)
        scaled.interpolate(1.0)
        metric.compute_isotropic_metric(scaled, interpolant="L2")
        mesh_complexity = metric.complexity()


        weak_forms = self._register_weak_forms(runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1, n, mesh, map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)
        solvers = self._register_solvers(weak_forms, Qnp1, Qaux_n)

        while sim_time < self.time_end:
                        # 3b. AMR every `refine_every` iterations (up to max_refinements)
            if (iteration % self.refine_every == 0 and self.enable_amr):
                logger.info(
                    f"Refining at iteration {int(iteration)}"
                )

                old_num_cells = mesh.num_cells()
                try:
                    M_wave = self._metric_wave(Qnp1, gamma=1.0, aniso=0.5)  # Strong refinement in fast regions
                    M_wd, _ = self._metric_wet_dry_smooth(Qnp1, gamma=10000, width=1)
                    # M_shock = self._metric_shock(Qnp1, gamma=1.0, aniso=0.5)

                    # Combine and normalize via Animate:
                    combined_metric = M_wd
                    # combined_metric = M_wd.combine(M_wave, average=True)  # intersection
                    # combined_metric = M_wd.combine(M_shock, average=True)  # intersection

                    combined_metric.set_parameters({
                        "dm_plex_metric_target_complexity": mesh_complexity,
                        "dm_plex_metric_p": np.inf,
                    })
                    combined_metric.normalise()
                    refined_mesh = adapt(mesh, combined_metric)
                    mesh = refined_mesh

                    # Rebuild function spaces on new mesh
                    nvar = runtime_model.n_variables
                    naux = runtime_model.n_aux_variables

                    V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=nvar)
                    Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=naux)

                    # Prolong solutions
                    Qn_new = self._prolong_vector_function(Qn, V)
                    Qnp1_new = self._prolong_vector_function(Qnp1, V)

                    Qaux_new_n = fd.Function(Vaux, name="Qaux")
                    Qaux_new_np1 = fd.Function(Vaux, name="Qaux")

                    
                    self.update_Qaux(Qn_new, Qaux_new_n)
                    self.update_Qaux(Qnp1_new, Qaux_new_np1)
                    self.update_Q(Qn_new, Qaux_new_n)
                    self.update_Q(Qnp1_new, Qaux_new_np1)

                    Qn, Qnp1, Qaux_n , Qaux_np1= Qn_new, Qnp1_new, Qaux_new_n, Qaux_new_np1
                    compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)

                    x, x_3d, n = self._get_x_and_n(mesh)
                    weak_form = self._get_weak_form(
                        runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1, n, mesh,
                        map_boundary_tag_to_function_index, sim_time, dt, x, x_3d
                    )
                    
                    solvers = self._register_solvers(weak_forms, Qnp1, Qaux_n)

                except Exception as e:
                    print("Mesh refinement failed:", e)
                    traceback.print_exc()
                    raise

                new_num_cells = mesh.num_cells()
                logger.info(""f"Mesh refined: {old_num_cells} → {new_num_cells} cells")
                
            # 3b. Advance in time
            Qn.assign(Qnp1)
            self.update_Q(Qn, Qaux_n)
            self.update_Qaux(Qn, Qaux_n)
            self.update_Q(Qnp1, Qaux_np1)
            self.update_Qaux(Qnp1, Qaux_np1)

            dt.assign(compute_dt(Qn, Qaux_n))
            for solver in solvers:
                solver.solve()

            sim_time += float(dt)
            iteration += 1

            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {int(iteration)}, time: {float(sim_time):.6f}, "
                    f"dt: {float(dt):.6f}, next write at time: {float(next_write_time):.6f}"
                )


            # 3c. Output only when reaching the next snapshot time
            if sim_time + 1e-12 >= next_write_time:
                self.write_state(Qnp1, Qaux_n, out, time=sim_time)
                next_write_time += dt_snapshot
        execution_time = get_time() - start_time
        logger.info(f"Finished simulation in {execution_time:.3f} seconds")
