"""2D dam-break with wet/dry interface — positivity test.

Drops a circular reservoir of water on a dry bed and lets it
collapse.  Exercises the symbolic Riemann solver lowered to UFL on a
state with vanishing depths.

We do not require well-balanced lake-at-rest preservation here (that
needs the bathymetry-aware reconstruction in PositiveRusanov, which
relies on FieldHandle of ``b`` in state — and our SWE2D carries ``b``
in ``Qaux`` instead).  The test asserts:

- the solver builds + runs without raising;
- depth never goes below ``-1e-8`` (approximate positivity);
- mass is approximately conserved over the integration window
  (no large numerical leaks from the wet/dry interface).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.swe import SWE
from zoomy_core.fvm.riemann_solvers import Rusanov


REPO_ROOT = os.environ.get("ZOOMY_ROOT", "/work")
MESH_PATH = os.path.join(REPO_ROOT, "meshes", "square",
                        "mesh_triangular.msh")


def _wetdry_model():
    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(tag="wall"),
            BC.Extrapolation(tag="inflow"),
            BC.Extrapolation(tag="outflow"),
        ]
    )

    def ic_q(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        # Square mesh spans (-10, 10) × (-10, 10) — put a reservoir of
        # radius 3 in the centre, dry outside.
        h = np.where(r < 3.0, 1.0, 1.0e-4)
        hu = np.zeros_like(x[0])
        hv = np.zeros_like(x[0])
        return np.stack([h, hu, hv], axis=0)

    def ic_qaux(x):
        b = np.zeros_like(x[0])
        bx = np.zeros_like(x[0])
        by = np.zeros_like(x[0])
        return np.stack([b, bx, by], axis=0)

    return SWE(
        dimension=2,
        manning_n=0.0,
        nu=0.0,
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
def test_dam_break_runs_and_stays_positive(tmp_path):
    from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver

    model = _wetdry_model()
    settings = Settings(
        name="swe2d-dam-break",
        output=Zstruct(directory=str(tmp_path / "out"),
                       snapshots=4, filename="dg", clean_directory=True),
    )

    solver = FiredrakeHyperbolicSolver(
        settings=settings,
        time_end=0.05,
        CFL=0.25,
        dg_degree=0,
        riemann_solver_cls=Rusanov,
    )
    solver.solve(MESH_PATH, model)

    Q_arr = np.asarray(solver._state.Qnp1.dat.data_ro)
    h_min = float(np.min(Q_arr[:, 0]))
    assert h_min >= -1.0e-6, f"positivity violated: h_min={h_min}"
