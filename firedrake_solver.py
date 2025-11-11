import firedrake as fd
import ufl
from zoomy_core.fvm.solver_numpy import Settings
from attrs import field, define
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
import numpy as np
from mpi4py import MPI



@define(frozen=True, slots=True, kw_only=True)
class FiredrakeHyperbolicSolver:
    """Hyperbolic solver configuration for Firedrake."""

    # Scalar parameters
    CFL: float = field(default=0.45)
    time_end: float = field(default=0.1)
    
    # Nested struct with factory
    settings: Zstruct = field(factory=lambda: Settings.default())

    # Tensor factory (recomputed for each instance)
    IdentityMatrix = field(
        factory=lambda: ufl.as_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, "settings", defaults)

    def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, mesh):
        return fd.dot(
            0.5
            * (model.flux(Ql, Qauxl, parameters) + model.flux(Qr, Qauxr, parameters)),
            n,
        ) - 0.5 * self.max_abs_eigenvalue(model, Ql, Qauxl, n, mesh) * (Qr - Ql)

    def max_abs_eigenvalue(self, model, Q, Qaux, n, mesh):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(ev[i] > max_ev, ev[i], max_ev)
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
        
    def compute_dt(self, model, Q, Qaux, ref_dx, CFL=0.45):
        mesh = Q.function_space().mesh()
        
        V = fd.FunctionSpace(mesh, "DG", 0)
        lam_x_expr = self.max_abs_eigenvalue(model, Q, Qaux, fd.as_vector([1.0, 0.0]), mesh)
        lam_y_expr = self.max_abs_eigenvalue(model, Q, Qaux, fd.as_vector([0.0, 1.0]), mesh)
        lam_x = fd.project(lam_x_expr, V)
        lam_y = fd.project(lam_y_expr, V)
        lam_max = max(np.max(lam_x.dat.data_ro), np.max(lam_y.dat.data_ro))
        lam_max = mesh.comm.allreduce(lam_max, op=MPI.MAX)
        return float(CFL * ref_dx / (lam_max + 1e-8))
    
    def set_initial_condition(self, Q, model):
        mesh = Q.function_space().mesh()
        x = fd.SpatialCoordinate(mesh)
        Q_init = model.initial_condition(fd.as_vector(x))
        Q.interpolate(Q_init)
        
    def initial_condition(self, V):
        x, y = fd.SpatialCoordinate(V.mesh())
        h0 = fd.conditional(x < 5, 2.0, 1.0)
        u0 = 0.0
        v0 = 0.0
        b = 0.0
        ic_expr = fd.as_vector([b, h0, h0*u0, h0*v0])
        Q0 = fd.Function(V, name="Q")
        Q0.interpolate(ic_expr)
        return Q0

        

    def solve(self, mshfile, model):
        mesh = fd.Mesh(mshfile)
        runtime_model = UFLRuntimeModel(model)

        x = fd.SpatialCoordinate(mesh)
        n = fd.FacetNormal(mesh)

        V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=runtime_model.n_variables)
        Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=runtime_model.n_aux_variables)
        Qn = fd.Function(V)
        Qnp1 = fd.Function(V)
        Qaux = fd.Function(Vaux)
        
        # self.set_initial_condition(Qn, model)
        # self.set_initial_condition(Qnp1, model)
        Qn = self.initial_condition(V)
        Qnp1 = self.initial_condition(V)
        


        t = 0.0
        # dt = self.compute_dt(model, Qn, Qaux, ref_dx=mesh.cell_sizes.dat.data_ro.min(), CFL=self.CFL)
        dt = fd.Constant(0.1)

        test_q = fd.TestFunction(V)
        trial_q = fd.TrialFunction(V)
        
        # F =  dot(test_q, (trial_q - q_n)/dt) * dx
        # F += dot(jump(test_q), numerical_flux(q_n('+'), q_n('-'), n('+'))) * dS
        # F += dot(test_q, numerical_flux(q_n, q_ext, n)) * ds
        # F -= dot(test_q, source(q_n, db_dx, db_dy)) * dx
        
        # ghost/extrapolation BC
        q_ext = Qn
        q_aux_ext = Qaux

        weak_form = fd.dot(test_q, (trial_q - Qn) / dt) * fd.dx
        weak_form += (
            fd.dot(
                test_q("+") - test_q("-"),
                self.numerical_flux(
                    runtime_model,
                    Qn("+"),
                    Qn("-"),
                    Qaux("+"),
                    Qaux("-"),
                    runtime_model.parameters,
                    n("+"),
                    mesh,
                ),
            )
            * fd.dS
        )
        weak_form += fd.dot(test_q, self.numerical_flux(
            runtime_model,
            Qn,
            q_ext,
            Qaux,
            q_aux_ext,
            runtime_model.parameters, n, mesh)) * fd.ds
        # weak_form -= fd.dot(
        #     test_q,
        #     runtime_model.source(Qn, Qaux, runtime_model.parameters),
        # ) * fd.dx

        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        solver = fd.LinearVariationalSolver(
            problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "jacobi"}
        )
        
        out = fd.VTKFile("simulation.pvd")
        self.write_state(Qnp1, Qaux, out, time=t)
        dx_ref = mesh.cell_sizes.dat.data_ro.min()

        while t < self.time_end:
            solver.solve()
            Qn.assign(Qnp1)
            dt.assign(self.compute_dt(runtime_model, Qn, Qaux, ref_dx=dx_ref, CFL=self.CFL))
            # t.assign(float(t) + float(dt))
            t += float(dt)
            self.write_state(Qnp1, Qaux, out, time=t)
            print(f"t = {t:.3f}, dt = {float(dt):.3e}")
