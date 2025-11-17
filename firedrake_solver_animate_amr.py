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
import zoomy_firedrake.firedrake_solver as fd_solver

from animate.utility import VTKFile

import os
import meshio
import traceback

# Import animate for AMR
from animate.adapt import adapt
from animate.metric import P0Metric


@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolverAMR(fd_solver.FiredrakeHyperbolicSolver):
    refine_every: int = field(default=20)      # time steps between refinements
    max_refinements: int = field(default=6)    # total refinement steps
    
    def build_spd_metric_from_indicator(self, metric, indicator, safety=1e-3):
        """Build a safe SPD metric where indicator is 0 or 1."""
        ndim = metric.ufl_shape[-1]  # 2 in 2D, 3 in 3D
        ncells = len(indicator)

        # Initialize as isotropic metric
        values = np.zeros((ncells, ndim, ndim))

        # At minimum: identity scaled by safety factor
        values[:] = np.eye(ndim) * safety

        # Mark refinement regions (indicator == 1) with stronger metric
        refining = indicator > 0.5
        values[refining, :, :] = np.eye(ndim)  # Could increase entries for stronger refinement

        # Assign into metric object
        metric.dat.data[:] = values
    
    def _marker_simple_slope(self, Q, threshold=0.05):
        """
        Basic refinement marker based on magnitude of grad(h) squared,
        using interpolation and avoiding project() to prevent integration domain errors.
        """
        mesh = Q.function_space().mesh()

        # h is in DG0 (component 1 of Q)
        h = Q.sub(1)

        # Reconstruct h in CG1 so grad(h) is nonzero and well-resolved
        Vcg1 = fd.FunctionSpace(mesh, "CG", 1)
        h_cg1 = fd.Function(Vcg1)
        h_cg1.interpolate(h)

        # Compute grad(h) in CG1 and interpolate its squared norm into DG0
        grad_h_sq = fd.inner(fd.grad(h_cg1), fd.grad(h_cg1))
        Vdg0 = fd.FunctionSpace(mesh, "DG", 0)
        grad_norm_sq = fd.Function(Vdg0)
        grad_norm_sq.interpolate(grad_h_sq)  # interpolate into DG0

        # Create marker: 1 if |grad(h)| > threshold, 0 otherwise
        marker = fd.Function(Vdg0)
        marker.interpolate(fd.conditional(grad_norm_sq > threshold**2, 1.0, 0.0))

        return marker



    def _marker_from_grad_h(self, Q, threshold=0.05):
        """
        AMR marker based on ||grad(h)||, where h = Q[1].

        Steps:
        - Take h (in DG0)
        - Interpolate to CG1
        - Compute grad(h_cg1)
        - Build indicator and threshold to create DG0 0/1 marker
        """
        mesh = Q.function_space().mesh()

        # h is in DG0 (component 1 of Q)
        h_dg0 = Q.sub(1)

        # Reconstruct h in CG1 so grad(h) is nonzero and well-resolved
        Vcg1 = fd.FunctionSpace(mesh, "CG", 1)
        h_cg1 = fd.Function(Vcg1, name="h_cg1")
        h_cg1.interpolate(h_dg0)

        # Compute the norm of grad(h)
        grad_h_norm_expr = fd.sqrt(fd.inner(fd.grad(h_cg1), fd.grad(h_cg1)))

        # Indicator in CG1
        indicator = fd.Function(Vcg1, name="grad_h_norm")
        indicator.interpolate(grad_h_norm_expr)

        # Threshold and put into DG0 for marking
        Vdg0 = fd.FunctionSpace(mesh, "DG", 0)
        marker = fd.Function(Vdg0, name="amr_marker")
        marker.interpolate(fd.conditional(indicator > threshold, 1.0, 0.0))

        return marker
    
    def _prolong_vector_function(self, old_fun, new_space):
        new_fun = fd.Function(new_space)
        new_fun.interpolate(old_fun)
        return new_fun

    # def _prolong_vector_function(self, old_fun, new_V):
    #     """
    #     Prolong a vector Function old_fun onto new mesh / space new_V
    #     by interpolating each component separately.

    #     old_fun: Function on old mesh, vector-valued.
    #     new_V:   VectorFunctionSpace on new mesh with same value_size.
    #     """
    #     nvar = old_fun.function_space().value_size
    #     assert new_V.value_size == nvar

    #     new_fun = fd.Function(new_V, name=old_fun.name())

    #     for i in range(nvar):
    #         new_fun.sub(i).interpolate(old_fun.sub(i))

    #     return new_fun
    
    def write_state(self, q, qaux, vtk_file, time=0.0, names=None):
        """
        Write state variables q, qaux and mesh to a VTK adaptive file.
        Each call writes a full snapshot at a given time step.
        """
        mesh = q.function_space().mesh()
        V_scalar = fd.FunctionSpace(mesh, "DG", 0)
        n_q = q.function_space().value_size
        n_aux = qaux.function_space().value_size

        # Project components of q and qaux to scalar DG0 functions
        fields = [
            fd.project(q[i], V_scalar, name=names[i] if names else f"Q{i}")
            for i in range(n_q)
        ] + [
            fd.project(qaux[i], V_scalar, name=names[n_q + i] if names else f"Qaux{i}")
            for i in range(n_aux)
        ]

        # Write fields (mesh automatically inferred from first function)
        vtk_file.write(*fields, time=time)
        print(f"[VTK] Exported timestep {time:.4f} with {mesh.num_cells()} cells.")

    # def write_state(self, q, qaux, out, time=0.0, names=None):
    #     mesh = q.function_space().mesh()
    #     V_scalar = fd.FunctionSpace(mesh, "DG", 0)
    #     n_dof_q = q.function_space().value_size
    #     n_dof_aux = qaux.function_space().value_size

    #     # Project components of q and qaux to scalar DG0 functions
    #     subfuns = [
    #         fd.project(q[i], V_scalar, name=names[i] if names else f"Q{i}")
    #         for i in range(n_dof_q)
    #     ] + [
    #         fd.project(qaux[i], V_scalar, name=names[n_dof_q + i] if names else f"Qaux{i}")
    #         for i in range(n_dof_aux)
    #     ]

    #     # Also include mesh coordinates for consistent output
    #     mesh_coords = mesh.coordinates
    #     mesh_coords.rename("mesh_coords")  # Naming prevents clashes

    #     # Write mesh + scalar fields to output
    #     # out.write(mesh_coords, *subfuns, time=time)
    #     out.write(*subfuns, time=time)

    def solve(self, mshfile, model):
        # ----- 1. Setup -----
        mesh, runtime_model, V, Vaux, Qn, Qnp1, Qaux, map_boundary_tag_to_function_index = self._setup(mshfile, model)
        x, x_3d, n = self._get_x_and_n(mesh)

        compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)
        sim_time = 0.0
        dt = fd.Constant(0.1)

        weak_form = self._get_weak_form(runtime_model, Qn, Qnp1, Qaux, n, mesh,
                                        map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)
        solver = self._get_solver(weak_form, Qnp1, Qaux)

        # ---- 2. Write Output ----
        main_dir = misc.get_main_directory()
        out = VTKFile(os.path.join(main_dir, self.settings.output.directory, "simulation.pvd"))
        self.write_state(Qnp1, Qaux, out, time=sim_time)

        # ---- 3. Main time loop with AMR ----
        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot

        while sim_time < self.time_end:
            # advance in time
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

            # ---- 5. AMR every `refine_every` iterations ----
            if iteration % self.refine_every == 0 and iteration > 0:
                logger.info(f"Refining at iteration {int(iteration)}, sim_time {float(sim_time):.6f}")
                
                old_num_cells = mesh.num_cells()
                try:    
                    # (a) Compute indicator based on ∥∇h∥
                    marker = self._marker_simple_slope(Qnp1, threshold=0.05)
                    
                    # Create tensor function space: DG0 with shape (2,2) for 2D
                    V0 = fd.TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2))  

                    # Instantiate the P0Metric explicitly on that space
                    metric = P0Metric(V0)

                    # (c) Assign isotropic metric values
                    indicator = marker.dat.data_ro        # shape: (ncells,)
                    self.build_spd_metric_from_indicator(metric, indicator)
                    
                    values = np.zeros((len(indicator), 2, 2)) 
                    values[:, 0, 0] = 0.8                # M_xx
                    values[:, 1, 1] = 0.8                # M_yy
                    metric.dat.data[:] = values           # assign all tensors

                    # (d) Adapt mesh using this metric
                    refined_mesh = adapt(mesh, metric)
                    mesh = refined_mesh
                    # (d) Rebuild function spaces on the new mesh
                    nvar = runtime_model.n_variables
                    naux = runtime_model.n_aux_variables

                    V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=nvar)
                    Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=naux)

                    # (e) Prolong Qn and Qnp1
                    Qn_new = self._prolong_vector_function(Qn, V)
                    Qnp1_new = self._prolong_vector_function(Qnp1, V)

                    # (f) Rebuild Qaux
                    Qaux_new = fd.Function(Vaux, name="Qaux")
                    self.update_Qaux(Qn_new, Qaux_new)
                    self.update_Q(Qn_new, Qaux_new)
                    self.update_Q(Qnp1_new, Qaux_new)

                    # (g) Replace references
                    Qn, Qnp1, Qaux = Qn_new, Qnp1_new, Qaux_new
                    compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)

                    # (h) Rebuild weak form and solver
                    x, x_3d, n = self._get_x_and_n(mesh)
                    weak_form = self._get_weak_form(runtime_model, Qn, Qnp1, Qaux, n, mesh,
                                                    map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)
                    solver = self._get_solver(weak_form, Qnp1, Qaux)
                except Exception as e:
                    print("Mesh refinement failed:", e)
                    traceback.print_exc()
                    raise 
                new_num_cells = mesh.num_cells()

                print(f"Mesh refined: {old_num_cells} → {new_num_cells} cells")
            self.write_state(Qnp1, Qaux, out, time=sim_time)


        logger.info(f"Finished simulation in {sim_time:.3f} seconds")

