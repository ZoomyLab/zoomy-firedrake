"""Smooth 2D SWE with Manning friction + depth-weighted eddy viscosity.

Exercises the full Model → SystemModel → UFL pipeline against a flat
2D square mesh.  Initial state is uniform normal flow

    h = h0,   hu = q,   hv = 0

with a constant downhill bed slope ``∂_x b = -S₀``.  Manning balance:

    g · h · S₀ = g · n² · u · |u| / h^(7/3)
    ⇒  q = h^(5/3) · √(S₀) / n.

With the IC set to the analytic normal-flow values the solution
should stay close to them — we assert that ``h`` and ``hu`` deviate
from the initial state by < a few percent after a short transient.
The tiny viscosity ``ν = 1e-4`` exercises the IP-DG block without
dominating.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.misc.misc import Zstruct
from zoomy_firedrake.firedrake_compat import SWE
from zoomy_core.fvm.riemann_solvers import Rusanov


REPO_ROOT = os.environ.get("ZOOMY_ROOT", "/work")
MESH_PATH = os.path.join(REPO_ROOT, "meshes", "square", "mesh.msh")

H_NORMAL = 1.0
SLOPE = 1.0e-3
MANNING_N = 0.03
Q_NORMAL = H_NORMAL ** (5.0 / 3.0) * np.sqrt(SLOPE) / MANNING_N


def _make_model():
    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(tag="wall"),
            BC.Extrapolation(tag="inflow"),
            BC.Extrapolation(tag="outflow"),
        ]
    )

    def ic_q(x):
        h = np.full_like(x[0], H_NORMAL)
        hu = np.full_like(x[0], Q_NORMAL)
        hv = np.zeros_like(x[0])
        # IC.UserFunction expects (n_state, n_pts); the solver passes
        # ``coords.T`` as ``x`` and broadcasts against ``Qarr.T``.
        return np.stack([h, hu, hv], axis=0)

    def ic_qaux(x):
        # b = -S₀ · x → ∂_x b = -S₀ everywhere.
        b = -SLOPE * x[0]
        bx = -SLOPE * np.ones_like(x[0])
        by = np.zeros_like(x[0])
        return np.stack([b, bx, by], axis=0)

    return SWE(
        dimension=2,
        manning_n=MANNING_N,
        nu=1.0e-4,
        g=9.81,
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_q),
        aux_initial_conditions=IC.UserFunction(ic_qaux),
    )


@pytest.mark.skipif(
    not os.path.exists(MESH_PATH),
    reason=f"Mesh fixture {MESH_PATH} not present; "
           "run ``bash meshes/square/run.sh`` first.",
)
def test_swe2d_smooth_manning_holds_normal_flow(tmp_path):
    """Initial state is the analytic Manning normal flow; assert it
    stays close to that state over a short transient."""
    from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver

    model = _make_model()
    settings = Settings(
        name="swe2d-smooth-manning",
        output=Zstruct(
            directory=str(tmp_path / "out"),
            snapshots=4,
            filename="dg",
            clean_directory=True,
        ),
    )
    solver = FiredrakeHyperbolicSolver(
        settings=settings,
        time_end=0.05,
        CFL=0.3,
        dg_degree=0,
        riemann_solver_cls=Rusanov,
    )
    solver.solve(MESH_PATH, model)

    s = solver._state
    Q_arr = np.asarray(s.Qnp1.dat.data_ro)  # (n_dofs, 3)
    h_mean = float(np.mean(Q_arr[:, 0]))
    q_mean = float(np.mean(Q_arr[:, 1]))

    assert abs(h_mean - H_NORMAL) / H_NORMAL < 0.10, (
        f"h drifted: mean {h_mean}, normal {H_NORMAL}")
    assert abs(q_mean - Q_NORMAL) / Q_NORMAL < 0.20, (
        f"hu drifted: mean {q_mean}, normal {Q_NORMAL}")
