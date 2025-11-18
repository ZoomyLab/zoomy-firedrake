import os
import traceback

import firedrake as fd
from firedrake.functionspaceimpl import MixedFunctionSpace
import ufl
import numpy as np
from mpi4py import MPI

from attrs import field, define

from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.misc as misc
import zoomy_firedrake.firedrake_solver as fd_solver

from pyop2 import op2

from animate.utility import VTKFile
from animate.metric import RiemannianMetric, P0Metric
from animate.adapt import adapt

from firedrake.projection import Projector
from firedrake.petsc import PETSc


# Optional: inspect adaptor options (mostly for debugging)
opts = PETSc.Options()
print("Available options containing 'adaptor':")
for k in opts.getAll().keys():
    if b"adaptor" in k:
        print(k.decode())


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
    max_refinements: int = field(default=6)    # total refinement steps

    # ------------------------------------------------------------------
    # Metric helpers based on RiemannianMetric
    # ------------------------------------------------------------------

    def _normalize_metric(self, metric: RiemannianMetric, mesh, target_factor=1.0, p=np.inf):
        """
        Use Animate's RiemannianMetric.normalise() to scale the metric
        to a desired complexity: target_factor × (#cells).
        """
        target_complexity = float(target_factor * mesh.num_cells())
        params = metric.metric_parameters.copy()
        params["dm_plex_metric_target_complexity"] = target_complexity
        params["dm_plex_metric_p"] = p
        metric.set_parameters(params)
        metric.normalise()
        return metric

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

    def _metric_wave(self, Q,
                     L_stream=0.5,     # multiples of L0
                     L_cross=1.6,
                     vel_floor=1e-8,
                     L0=1.0):
        """
        Flow-aligned anisotropic metric:

        - Compute velocity (u, v)
        - Build a P0Metric per cell with eigenvectors aligned to flow
        - Project to CG1 RiemannianMetric
        """
        L_stream *= L0
        L_cross *= L0

        mesh = Q.function_space().mesh()
        h, hu, hv = Q.sub(1), Q.sub(2), Q.sub(3)

        V0 = h.function_space()
        u = fd.Function(V0)
        v = fd.Function(V0)

        u.interpolate(fd.conditional(h > vel_floor, hu / h, 0.0))
        v.interpolate(fd.conditional(h > vel_floor, hv / h, 0.0))

        u_arr = u.dat.data_ro
        v_arr = v.dat.data_ro
        speed = np.sqrt(u_arr**2 + v_arr**2)

        # global max speed (MPI-safe)
        max_speed = mesh.comm.allreduce(float(speed.max()), op=MPI.MAX)
        max_speed = max(max_speed, vel_floor)

        # Build DG0 tensor metric first (P0Metric)
        Vten0 = fd.TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2))
        M_p0 = P0Metric(Vten0)
        data = M_p0.dat.data
        I2 = np.eye(2)

        lam_cross = 1.0 / L_cross**2
        for c in range(mesh.num_cells()):
            s = speed[c] / max_speed    # 0…1
            if s < 0.05:                # almost stagnant
                data[c, :, :] = I2 * lam_cross
                continue
            nx, ny = u_arr[c] / speed[c], v_arr[c] / speed[c]
            vdir = np.array([nx, ny])
            tdir = np.array([-ny, nx])
            lam_stream = 1.0 / (L_stream * (1 - 0.8 * s))**2
            data[c, :, :] = (
                lam_stream * np.outer(vdir, vdir)
                + lam_cross * np.outer(tdir, tdir)
            )

        # Project to CG1 Riemannian metric
        metric = RiemannianMetric(mesh, name="wave_metric")
        metric.project(M_p0)
        metric.enforce_spd(restrict_anisotropy=True, restrict_sizes=True)
        return metric

    def _combine_metrics(self, metrics, weights=None, average=True):
        """
        Combine several RiemannianMetrics using Animate's combine/average machinery.

        metrics  – list[RiemannianMetric]
        weights  – optional list of floats; len = nmetrics
        """
        assert len(metrics) > 0
        base = metrics[0].copy(deepcopy=True)
        others = metrics[1:]
        if not others:
            return base
        if average:
            if weights is None:
                return base.average(*others)
            else:
                if len(weights) != len(metrics):
                    raise ValueError(
                        f"weights has length {len(weights)}, but metrics has {len(metrics)}"
                    )
                return base.average(*others, weights=weights)
        else:
            # Intersection
            return base.intersect(*others)

    # ------------------------------------------------------------------
    # Prolongation: conservative + positivity preserving
    # ------------------------------------------------------------------

    def _prolong_vector_function(self, old_fun, new_space,
                                 h_dry=1e-10, pos_idx=[1], clip=True):
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
        print(f"[VTK] Exported timestep {time:.4f} with {mesh.num_cells()} cells.")

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, mshfile, model):
        # ----- 1. Setup -----
        mesh, runtime_model, V, Vaux, Qn, Qnp1, Qaux, map_boundary_tag_to_function_index = \
            self._setup(mshfile, model)
        x, x_3d, n = self._get_x_and_n(mesh)

        compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)
        sim_time = 0.0
        dt = fd.Constant(0.1)

        weak_form = self._get_weak_form(
            runtime_model, Qn, Qnp1, Qaux, n, mesh,
            map_boundary_tag_to_function_index, sim_time, dt, x, x_3d
        )
        solver = self._get_solver(weak_form, Qnp1, Qaux)

        # ---- 2. Output setup ----
        main_dir = misc.get_main_directory()
        out = VTKFile(os.path.join(main_dir, self.settings.output.directory,
                                   "simulation.pvd"))
        out_id = VTKFile(os.path.join(main_dir, self.settings.output.directory,
                                      "indicator.pvd"))
        out_metric = VTKFile(os.path.join(main_dir, self.settings.output.directory,
                                          "metric.pvd"))

        # Write initial snapshot at t=0
        self.write_state(Qnp1, Qaux, out, time=sim_time)

        # ---- 2b. Optional: initial adaptive refinement before time stepping ----
        refinements_done = 0
        if self.max_refinements > 0:
            try:
                logger.info("Initial adaptive refinement at t=0.0")
                # Build metrics
                M_wd, smooth_indicator = self._metric_wet_dry_smooth(Qnp1)
                M_wave = self._metric_wave(Qnp1)

                # Combine metrics (average here; could also intersect)
                combined_metric = self._combine_metrics(
                    [M_wd, M_wave], average=True
                )

                # Normalise to desired complexity (~ factor * ncells)
                combined_metric = self._normalize_metric(
                    combined_metric, mesh, target_factor=1.0, p=np.inf
                )

                # Save metric / indicator for debugging (optional)
                # out_id.write(smooth_indicator, time=sim_time)
                # out_metric.write(combined_metric, time=sim_time)

                refined_mesh = adapt(mesh, combined_metric)
                mesh = refined_mesh

                # Rebuild function spaces on new mesh
                nvar = runtime_model.n_variables
                naux = runtime_model.n_aux_variables

                V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=nvar)
                Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=naux)

                # Prolong state conservatively
                Qn = self._prolong_vector_function(Qn, V)
                Qnp1 = self._prolong_vector_function(Qnp1, V)

                Qaux = fd.Function(Vaux, name="Qaux")
                self.update_Qaux(Qn, Qaux)
                self.update_Q(Qn, Qaux)
                self.update_Q(Qnp1, Qaux)

                compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)

                x, x_3d, n = self._get_x_and_n(mesh)
                weak_form = self._get_weak_form(
                    runtime_model, Qn, Qnp1, Qaux, n, mesh,
                    map_boundary_tag_to_function_index, sim_time, dt, x, x_3d
                )
                solver = self._get_solver(weak_form, Qnp1, Qaux)
                refinements_done += 1
                print(f"[AMR] Initial mesh refined to {mesh.num_cells()} cells")
            except Exception as e:
                print("Initial mesh refinement failed:", e)
                traceback.print_exc()
                # continue with unadapted mesh

        # ---- 3. Main time loop with AMR ----
        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot

        while sim_time < self.time_end:
            # 3a. Advance in time
            Qn.assign(Qnp1)
            self.update_Q(Qn, Qaux)
            self.update_Qaux(Qn, Qaux)

            dt.assign(compute_dt(Qn, Qaux))
            solver.solve()
            self.update_Q(Qnp1, Qaux)

            sim_time += float(dt)
            iteration += 1

            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {int(iteration)}, time: {float(sim_time):.6f}, "
                    f"dt: {float(dt):.6f}, next write at time: {float(next_write_time):.6f}"
                )

            # 3b. AMR every `refine_every` iterations (up to max_refinements)
            if (
                iteration % self.refine_every == 0
                and iteration > 0
                and refinements_done < self.max_refinements
            ):
                logger.info(
                    f"Refining at iteration {int(iteration)}, "
                    f"sim_time {float(sim_time):.6f}"
                )

                old_num_cells = mesh.num_cells()
                try:
                    # Build metrics from current state
                    M_wd, smooth_indicator = self._metric_wet_dry_smooth(Qnp1)
                    M_wave = self._metric_wave(Qnp1)
                    # Optionally: M_bath = self._metric_bathymetry(Qnp1)

                    combined_metric = self._combine_metrics(
                        [M_wd, M_wave], average=True
                    )
                    combined_metric = self._normalize_metric(
                        combined_metric, mesh, target_factor=1.0, p=np.inf
                    )

                    # Optional debugging output
                    # out_id.write(smooth_indicator, time=float(sim_time))
                    # out_metric.write(combined_metric, time=float(sim_time))

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

                    Qaux_new = fd.Function(Vaux, name="Qaux")
                    self.update_Qaux(Qn_new, Qaux_new)
                    self.update_Q(Qn_new, Qaux_new)
                    self.update_Q(Qnp1_new, Qaux_new)

                    Qn, Qnp1, Qaux = Qn_new, Qnp1_new, Qaux_new
                    compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)

                    x, x_3d, n = self._get_x_and_n(mesh)
                    weak_form = self._get_weak_form(
                        runtime_model, Qn, Qnp1, Qaux, n, mesh,
                        map_boundary_tag_to_function_index, sim_time, dt, x, x_3d
                    )
                    solver = self._get_solver(weak_form, Qnp1, Qaux)

                    refinements_done += 1
                except Exception as e:
                    print("Mesh refinement failed:", e)
                    traceback.print_exc()
                    raise

                new_num_cells = mesh.num_cells()
                print(f"Mesh refined: {old_num_cells} → {new_num_cells} cells")

            # 3c. Output only when reaching the next snapshot time
            if sim_time + 1e-12 >= next_write_time:
                self.write_state(Qnp1, Qaux, out, time=sim_time)
                next_write_time += dt_snapshot

        logger.info(f"Finished simulation in {sim_time:.3f} seconds")
