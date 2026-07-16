"""Shared authoring harness + run helper for the 2-D SWE Firedrake gates.

The three ``test_swe2d_*`` gates each author a *minimal* inline
``ShallowWater`` model on the canonical NSM path (copied down from
``tutorials/firedrake/malpasset_firedrake.py``).  The two pieces that are
pure boilerplate — the ``SystemModelSpec`` authoring shim and the
solver-wiring / read-back helper — live here so each gate reads as
model → run → assert.

Nothing SWE-specific lives in this module: ``SystemModelSpec`` is the
generic "write each operator as a method" harness, and ``run_swe2d`` just
builds ``NumericalSystemModel.from_system_model(..., PositiveNonconservativeHLL)``
+ ``FiredrakeHyperbolicSolver`` and runs the time loop.
"""

from __future__ import annotations

import tempfile

import numpy as np
import sympy as sp
from sympy import eye, zeros

import firedrake as fd

import zoomy_core.model.boundary_conditions as bc
import zoomy_core.model.aux_boundary_conditions as aux_bc
from zoomy_core.misc.misc import Zstruct
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL

from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver


class SystemModelSpec(SystemModel):
    """Author a SystemModel by writing each operator as a method.

    Declare ``variables``, ``aux_variables``, ``parameters`` (name -> default),
    then define any of the operators below as a method returning its symbolic
    tensor.  Verbatim from ``tutorials/firedrake/malpasset_firedrake.py`` — the
    canonical Firedrake authoring shim.
    """

    variables = ()
    aux_variables = ()
    parameters = {}
    _operators = ("flux", "hydrostatic_pressure", "nonconservative_matrix",
                  "source", "source_explicit", "diffusion_matrix_explicit",
                  "eigenvalues", "update_variables", "update_aux_variables",
                  "reconstruction_variables")

    def __init__(self, **parameter_overrides):
        cls = type(self)
        self.time = sp.Symbol("t", real=True)
        self.space = list(sp.symbols("x y", real=True))
        self._distance = sp.Symbol("distance", real=True)
        self.position = Zstruct(X0=sp.Symbol("X0"), X1=sp.Symbol("X1"),
                                X2=sp.Symbol("X2"))
        self.position._symbolic_name = "X"
        self.normal = Zstruct(n0=sp.Symbol("n0", real=True),
                              n1=sp.Symbol("n1", real=True))
        self.normal._symbolic_name = "n"

        state = [sp.Symbol(k, real=True) for k in cls.variables]
        aux = [sp.Symbol(k, real=True) for k in cls.aux_variables]
        self.variables = Zstruct(**dict(zip(cls.variables, state)))
        self.variables._symbolic_name = "Q"
        self.aux_variables = Zstruct(**dict(zip(cls.aux_variables, aux)))
        self.aux_variables._symbolic_name = "Qaux"

        values = {**dict(cls.parameters), **parameter_overrides}
        self.parameters = Zstruct(**{k: sp.Symbol(k, positive=True)
                                     for k in values})
        self.parameters._symbolic_name = "p"

        neq = len(state)
        fields = dict(
            time=self.time, space=self.space, state=state, aux_state=aux,
            parameters=self.parameters, parameter_values=Zstruct(**values),
            normal=self.normal, mass_matrix=eye(neq), source=zeros(neq, 1),
            hydrostatic_pressure=zeros(neq, 2),
            nonconservative_matrix=sp.MutableDenseNDimArray.zeros(neq, neq, 2))
        for name in self._operators:
            method = getattr(cls, name, None)
            if callable(method):
                fields[name] = method(self)

        if callable(getattr(cls, "boundary_conditions", None)):
            walls = bc.BoundaryConditions(self.boundary_conditions())
            args = (self.time, self.position, self._distance, self.variables,
                    self.aux_variables, self.parameters, self.normal)
            fields["boundary_conditions"] = walls.get_boundary_condition_function(
                *args, function_name="boundary_conditions")
            fields["boundary_gradients"] = walls.get_boundary_gradient_function(
                *args, function_name="boundary_gradients")
            aux_walls = bc.BoundaryConditions(
                [aux_bc.Extrapolation(tag=w.tag)
                 for w in walls.boundary_conditions_list])
            fields["aux_boundary_conditions"] = aux_walls.get_boundary_condition_function(
                *args, function_name="aux_boundary_conditions")
            self._boundary_tags = walls._boundary_tags

        super().__init__(**fields)
        self.expose_aux_atoms()


def cell_centroids(mesh):
    """(n_cells, gdim) DG0 cell centroids — same cell ordering as a DG0 state."""
    gdim = mesh.geometric_dimension
    V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=gdim)
    return np.asarray(fd.Function(V).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro)


def cell_areas(mesh):
    """Per-cell area (DG0 lumped mass) — the weights for the mass integral."""
    V = fd.FunctionSpace(mesh, "DG", 0)
    return np.asarray(fd.assemble(fd.TestFunction(V) * fd.dx).dat.data_ro)


def run_swe2d(model, mesh, ic, *, time_end, cfl=0.4):
    """Build the canonical NSM + Firedrake solver, run to ``time_end``.

    ``ic(Q, model)`` is the ``initial_condition_overwrite`` hook — it writes
    the DG0 state (columns ``b, h, hu, hv``).  Returns a ``Zstruct`` with the
    final state array ``Q`` (n_cells, 4), the cell ``centroids`` and ``areas``.
    """
    nsm = NumericalSystemModel.from_system_model(
        model, riemann=PositiveNonconservativeHLL)
    outdir = tempfile.mkdtemp(prefix="swe2d_gate_")
    solver = FiredrakeHyperbolicSolver(
        settings=Settings(output=Zstruct(
            directory=outdir, filename="dg", snapshots=2,
            clean_directory=True)),
        time_end=time_end, CFL=cfl, dg_degree=0, limiter="none",
        riemann_solver_cls=PositiveNonconservativeHLL,
        initial_condition_overwrite=ic)
    # In-memory mesh with a single Wall BC → the "__all__" fallback applies
    # it over the whole exterior (no boundary_tag_map needed).
    solver.setup_simulation(mesh, nsm)
    solver.run_simulation()
    Q = np.asarray(solver._state.Qnp1.dat.data_ro).reshape(-1, model.n_variables)
    return Zstruct(Q=Q, centroids=cell_centroids(mesh), areas=cell_areas(mesh),
                   solver=solver)
