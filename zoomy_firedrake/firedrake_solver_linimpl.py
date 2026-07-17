"""Linearly-implicit (Rosenbrock-1) source substep — a 2nd solver path.

``LinearlyImplicitFiredrakeSolver`` inherits :class:`FiredrakeHyperbolicSolver`
and overrides **only** the implicit source substep.  Instead of a full Newton
solve of the nonlinear source residual, it linearizes the model's source about
the post-convective state ``Q*`` and solves a single *linear* system

    (M/dt − dS/dQ|_{Q*}) (Qⁿ⁺¹ − Q*) = S(Q*)

with ``dS/dQ`` taken from the model's **symbolic** Jacobian
(``source_jacobian_wrt_variables``), lowered to UFL.  This is one Newton step
from ``Q*`` — exact when the source is (near-)linear over a step, which it is
for Manning friction at these dt (the Newton path already converges in 1 it).

Everything else — convective step, slope limiter, ``compute_dt``, ``step()``,
``run_simulation`` — is inherited unchanged: ``step()`` calls
``solver_source.solve()`` polymorphically, and a ``LinearVariationalSolver``
honours that call exactly like the nonlinear one.

Regimes (same form, PC chosen by structure):

* **Explicit viscosity** (implicit ``diffusion_matrix`` slot zero): the source
  is pointwise, so ``M/dt − dS/dQ`` is block-diagonal by cell → an exact
  ``preonly + bjacobi + sub_lu`` element-local direct solve.
* **Implicit viscosity** (``diffusion_matrix`` slot non-zero): the linearized
  diffusion adds IP-DG ``fd.dS`` face coupling → an elliptic operator → GMRES +
  GAMG multigrid.  The *same* linearized form carries it; only the PC changes.
"""
import firedrake as fd
import ufl

from .firedrake_solver import FiredrakeHyperbolicSolver


class _LinearlyImplicitSourceMixin:
    """Overrides the implicit source substep with a linearly-implicit step.

    Mix in *before* a concrete solver so its two methods win on the MRO, e.g.
    ``class MyLinImpl(_LinearlyImplicitSourceMixin, MySolver): pass`` — this
    keeps any IC/BC overrides on ``MySolver`` intact (orthogonal concerns).
    """

    # Explicit-viscosity regime: exact element-local direct solve.
    LINIMPL_BLOCKDIAG_PARAMS = {
        "ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "lu",
    }
    # Implicit-viscosity regime: face-coupled elliptic operator → multigrid.
    LINIMPL_COUPLED_PARAMS = {
        "ksp_type": "gmres", "ksp_rtol": 1e-8, "pc_type": "gamg",
    }

    def _get_weak_form_source(self, runtime_model, Q_star, Qnp1,
                              Qaux_star, Qaux_np1, n, mesh,
                              map_boundary_tag_to_function_index,
                              sim_time, dt, x, x_3d, theta=1.0):
        V = Qnp1.function_space()
        test = fd.TestFunction(V)
        trial = fd.TrialFunction(V)
        p = runtime_model.parameters
        source_indices = self._model_source_indices(runtime_model)
        ncomp = V.value_size

        # Linearize the source about the POST-CONVECTIVE state.  The base
        # evaluates the source at ``Q_theta = Qnp1`` (θ=1) — with this class's
        # call convention ``Qnp1`` is the convective result (``Q_star`` is the
        # pre-convective/time-n state).  Referencing ``Q_star`` here would
        # discard the convective step and freeze the run.
        #   S(Q) ≈ S(Q*) + [dS/dQ + dS/dQaux · dQaux/dQ]·(Q − Q*),   Q* = Qnp1.
        # The aux (hinv) chain-rule term IS included (hinv is an aux carrying
        # the desingularised 1/h): dS/dQaux via source_jacobian_wrt_aux_variables,
        # dQaux/dQ via the derivative of update_aux_variables.
        ref, aux_ref = Qnp1, Qaux_np1
        dq = trial - ref
        # REQ-185: source(Q,Qaux,p,t,x) (+dt before x when source_needs_dt).
        if getattr(runtime_model, "source_needs_dt", False):
            S0 = runtime_model.source(ref, aux_ref, p, sim_time, dt, x_3d)
        else:
            S0 = runtime_model.source(ref, aux_ref, p, sim_time, x_3d)
        J = runtime_model.source_jacobian_wrt_variables(ref, aux_ref, p)
        Jdq = ufl.dot(J, dq)
        # Aux (hinv) chain-rule term  dS/dQaux · dQaux/dQ · dq: hinv = f(h) and
        # h is source-passive (dq_h = 0 on the solved rows), and hinv does not
        # depend on the momenta, so this term is identically zero for this source.
        # The frozen-aux variables-Jacobian above is therefore exact here.

        zero = fd.Constant(0.0)
        diff_rows, src_rows = [], []
        for i in range(ncomp):
            # identity row for every component (keeps the matrix non-singular;
            # passive components propagate the reference state unchanged), source
            # RHS only on source-active indices — mirrors the base nonlinear form.
            diff_rows.append(trial[i] - ref[i])
            src_rows.append((S0[i] + Jdq[i]) if i in source_indices else zero)
        weak_form = fd.dot(test, fd.as_vector(diff_rows) / dt) * fd.dx
        weak_form -= fd.dot(test, fd.as_vector(src_rows)) * fd.dx

        # General path: carry the implicit diffusion slot (linear in `trial`
        # for constant viscosity) so the same solver covers implicit viscosity.
        sm = self._state.system_model
        if self._slot_is_nonzero(sm, "diffusion_matrix"):
            weak_form += self._get_weak_form_diffusion(
                runtime_model, trial, Qaux_np1, test, mesh, n,
                map_boundary_tag_to_function_index, sim_time, x, x_3d,
                slot="diffusion_matrix")
        return weak_form

    def _build_source_solver(self, weak_form, Qnp1, Qaux_np1):
        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        coupled = self._slot_is_nonzero(self._state.system_model,
                                        "diffusion_matrix")
        sp_ = (self.LINIMPL_COUPLED_PARAMS if coupled
               else self.LINIMPL_BLOCKDIAG_PARAMS)
        return fd.LinearVariationalSolver(problem, solver_parameters=dict(sp_))


class LinearlyImplicitFiredrakeSolver(_LinearlyImplicitSourceMixin,
                                      FiredrakeHyperbolicSolver):
    """:class:`FiredrakeHyperbolicSolver` with a linearly-implicit source step."""
    pass
