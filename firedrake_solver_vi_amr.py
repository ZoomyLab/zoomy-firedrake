import firedrake as fd
from firedrake.functionspaceimpl import MixedFunctionSpace
import ufl
from zoomy_core.fvm.solver_numpy import Settings
from attrs import field, define
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
import numpy as np
from mpi4py import MPI

from viamr import VIAMR

from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.misc as misc
import zoomy_firedrake.firedrake_solver as fd_solver

import os

import meshio




@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolverAMR(fd_solver.FiredrakeHyperbolicSolver):
    refine_every: int = field(default=20)      # time steps between refinements
    max_refinements: int = field(default=6)    # total refinement steps

    def _marker_from_grad_h(self, Q, threshold=0.05):
        """
        AMR marker based on ||grad(h)||, where h = Q[1].

        Steps:
        - take h (in DG0)
        - interpolate to CG1
        - compute grad(h_cg1)
        - build indicator and threshold to create DG0 0/1 marker
        """
        mesh = Q.function_space().mesh()

        # h is in DG0 (component 1 of Q)
        h_dg0 = Q.sub(1)

        # Reconstruct h in CG1 so grad(h) is nonzero and well-resolved
        Vcg1 = fd.FunctionSpace(mesh, "CG", 1)
        h_cg1 = fd.Function(Vcg1, name="h_cg1")
        h_cg1.interpolate(h_dg0)

        # Now grad(h_cg1) is a genuine UFL expression
        grad_h_norm_expr = fd.sqrt(fd.inner(fd.grad(h_cg1), fd.grad(h_cg1)))

        # Indicator in CG1
        indicator = fd.Function(Vcg1, name="grad_h_norm")
        indicator.interpolate(grad_h_norm_expr)

        # Threshold and put into DG0 for marking
        Vdg0 = fd.FunctionSpace(mesh, "DG", 0)
        marker = fd.Function(Vdg0, name="amr_marker")
        marker.interpolate(fd.conditional(indicator > threshold, 1.0, 0.0))

        return marker
    
    def _prolong_vector_function(self, old_fun, new_V):
        """
        Prolong a vector Function old_fun onto new mesh / space new_V
        by interpolating each component separately.

        old_fun: Function on old mesh, vector-valued.
        new_V:   VectorFunctionSpace on new mesh with same value_size.
        """
        nvar = old_fun.function_space().value_size
        assert new_V.value_size == nvar

        new_fun = fd.Function(new_V, name=old_fun.name())

        for i in range(nvar):
            new_fun.sub(i).interpolate(old_fun.sub(i))

        return new_fun
    
    def write_state(self, q, qaux, out, time=0.0, names=None):
        mesh = q.function_space().mesh()
        V_scalar = fd.FunctionSpace(mesh, "DG", 0)
        n_dof_q = q.function_space().value_size
        n_dof_aux = qaux.function_space().value_size

        # Project components of q and qaux to scalar DG0 functions
        subfuns = [
            fd.project(q[i], V_scalar, name=names[i] if names else f"Q{i}")
            for i in range(n_dof_q)
        ] + [
            fd.project(qaux[i], V_scalar, name=names[n_dof_q + i] if names else f"Qaux{i}")
            for i in range(n_dof_aux)
        ]

        # Also include mesh coordinates for consistent output
        mesh_coords = mesh.coordinates
        mesh_coords.rename("mesh_coords")  # Naming prevents clashes

        # Write mesh + scalar fields to output
        out.write(mesh_coords, *subfuns, time=time)



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
        out = fd.VTKFile(os.path.join(main_dir, self.settings.output.directory, "simulation.pvd"))
        self.write_state(Qnp1, Qaux, out, time=sim_time)


        # ---- 3. AMR setup ----
        amr = VIAMR()
        iteration = 0
        dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
        next_write_time = dt_snapshot

        # ---- 4. Main time loop ----
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

            self.write_state(Qnp1, Qaux, out, time=sim_time)

            # # Write output
            # if sim_time > next_write_time or sim_time >= self.time_end:
            #     next_write_time += dt_snapshot
            #     self.write_state(Qnp1, Qaux, out, time=sim_time)

                
            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {int(iteration)}, time: {float(sim_time):.6f}, "
                    f"dt: {float(dt):.6f}, next write at time: {float(next_write_time):.6f}"
                        )


            # ---- 5. AMR every `refine_every` iterations ----
            if iteration % self.refine_every == 0 and iteration > 0:
                logger.info(f"Refining at iteration {int(iteration)}, sim_time {float(sim_time):.6f}")

                # (a) Compute scalar indicator based on ∥∇h∥
                marker = self._marker_from_grad_h(Qnp1, threshold=0.05)

                # (b) Refine mesh based on marker
                mesh = amr.refinemarkedelements(mesh, marker)
                
                amr.meshreport(mesh)


                # (c) Rebuild function spaces on the new mesh
                nvar = runtime_model.n_variables
                naux = runtime_model.n_aux_variables

                V    = fd.VectorFunctionSpace(mesh, "DG", 0, dim=nvar)
                Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=naux)

                # (d) Prolong Qn and Qnp1 component-wise
                Qn_new   = self._prolong_vector_function(Qn,   V)
                Qnp1_new = self._prolong_vector_function(Qnp1, V)

                # (e) Rebuild Qaux on the new mesh from Qn_new
                Qaux_new = fd.Function(Vaux, name="Qaux")
                self.update_Qaux(Qn_new, Qaux_new)
                self.update_Q(Qn_new, Qaux_new)
                self.update_Q(Qnp1_new, Qaux_new)


                # (f) Replace old references
                mesh = V.mesh()
                Qn, Qnp1, Qaux = Qn_new, Qnp1_new, Qaux_new
                
                # (f.1) Rebuild dt-computation on the *new* mesh
                compute_dt = self.get_compute_dt(mesh, runtime_model, CFL=self.CFL)

                # (g) Rebuild weak form and solver for new mesh
                x, x_3d, n = self._get_x_and_n(mesh)
                weak_form = self._get_weak_form(runtime_model, Qn, Qnp1, Qaux, n, mesh,
                                                map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)
                solver = self._get_solver(weak_form, Qnp1, Qaux)
        
        logger.info(f"Finished simulation in {sim_time:.3f} seconds")
