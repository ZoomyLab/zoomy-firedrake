import firedrake as fd
import ufl
from zoomy_core.fvm.solver_numpy import Settings
from attrs import field, define
from zoomy_core.misc.misc import Zstruct
from zoomy_core.transformation.to_ufl import UFLRuntimeModel



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
        max_ev = abs(ev[0, 0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(ev[i, 0] > max_ev, ev[i, 0], max_ev)
        return max_ev

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

        t = fd.Constant(0.0)
        dt = fd.Constant(0.1)

        test_q = fd.TestFunction(V)
        trial_q = fd.TrialFunction(V)

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

        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        solver = fd.LinearVariationalSolver(
            problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "jacobi"}
        )

        outfile = fd.File("sim_firedrake.pvd")
        outfile.write(Qn, time=float(t))

        while float(t) < self.time_end:
            solver.solve()
            Qn.assign(Qnp1)
            t.assign(float(t) + float(dt))
            outfile.write(Qn, time=float(t))
