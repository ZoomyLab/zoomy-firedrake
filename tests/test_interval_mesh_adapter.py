"""IntervalMesh adapter for FiredrakeHyperbolicSolver — boundary-tag mapping.

The solver historically required a Gmsh ``.msh`` file path.  9a extends
``setup_simulation`` / ``solve`` / ``_setup`` to accept an in-memory
``fd.MeshGeometry`` (e.g. ``fd.IntervalMesh``) plus an explicit
``boundary_tag_map`` dict.  Without a .msh file the legacy meshio path
is bypassed.

This test exercises the boundary-tag resolution in isolation — it
doesn't need a full solve, just confirms that
``get_map_boundary_tag_to_boundary_function_index`` returns the same
``{physical_id: bc_list_index}`` shape when fed an IntervalMesh + tag
map as it does for a .msh + meshio.
"""

from __future__ import annotations

import pytest


def test_interval_mesh_boundary_tag_mapping():
    import firedrake as fd

    import zoomy_core.model.boundary_conditions as BC

    from zoomy_firedrake.firedrake_solver import (
        FiredrakeHyperbolicSolver,
    )
    from zoomy_core.fvm.riemann_solvers import Rusanov

    # Minimal stand-in for a Model — only used for its
    # ``boundary_conditions._boundary_tags`` attribute.
    class _StubModel:
        def __init__(self):
            self.boundary_conditions = BC.BoundaryConditions(
                [BC.Extrapolation(tag="bottom"),
                 BC.Extrapolation(tag="surface")]
            )

    solver = FiredrakeHyperbolicSolver(
        time_end=1.0, CFL=0.5, riemann_solver_cls=Rusanov,
    )

    mesh = fd.IntervalMesh(8, length_or_left=0.0, right=1.0)
    # IntervalMesh tags its endpoints 1 (left) and 2 (right).
    boundary_tag_map = {"bottom": 1, "surface": 2}
    model = _StubModel()

    out = solver.get_map_boundary_tag_to_boundary_function_index(
        model, msh_path=None, mesh=mesh, boundary_tag_map=boundary_tag_map,
    )
    # BC list sorted alphabetically: "bottom" (id=0), "surface" (id=1).
    # Returned dict is {physical_id: bc_list_index}.
    assert out == {1: 0, 2: 1}


def test_interval_mesh_without_tag_map_falls_back():
    """Without a boundary_tag_map, the in-memory branch returns the
    catch-all ``{"__all__": 0}`` sentinel just like the original .msh
    path does when no named groups exist."""
    import firedrake as fd  # noqa: F401  (imported only for the firedrake_available fixture)

    import zoomy_core.model.boundary_conditions as BC
    from zoomy_firedrake.firedrake_solver import (
        FiredrakeHyperbolicSolver,
    )
    from zoomy_core.fvm.riemann_solvers import Rusanov

    class _StubModel:
        def __init__(self):
            self.boundary_conditions = BC.BoundaryConditions(
                [BC.Extrapolation(tag="boundary")]
            )

    solver = FiredrakeHyperbolicSolver(
        time_end=1.0, CFL=0.5, riemann_solver_cls=Rusanov,
    )

    out = solver.get_map_boundary_tag_to_boundary_function_index(
        _StubModel(), msh_path=None, mesh=None, boundary_tag_map=None,
    )
    assert out == {"__all__": 0}
