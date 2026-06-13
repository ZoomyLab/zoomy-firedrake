"""Workarounds for Firedrake bugs encountered in the zoomy_firedrake
backend.

These wrappers keep the bug locations identifiable: every call site
uses an explicit ``safe_*`` helper whose docstring documents the
upstream issue and its reproducer.  When Firedrake fixes the
underlying bug, removing the workaround is a single grep:

    grep -rn safe_extract_component  library/zoomy_firedrake
    grep -rn safe_assign_component   library/zoomy_firedrake

Each finding is a single line that can be reverted to the plain
Firedrake call.
"""

from __future__ import annotations

import firedrake as fd

__all__ = ["safe_extract_component", "safe_assign_component"]


def safe_extract_component(
    target_scalar: fd.Function,
    source_vector: fd.Function,
    component_index: int,
) -> None:
    """Copy component ``component_index`` of a vector-valued Function
    into a scalar-valued Function — workaround for a Firedrake bug.

    Equivalent (mathematically) to::

        target_scalar.interpolate(source_vector.sub(component_index))

    but bypasses a Firedrake bug in the ``VectorFunctionSpace.sub(i)``
    interpolation kernel: when called **after** a direct numpy write
    to ``source_vector.dat.data[:, j]`` (any ``j``), a subsequent
    ``target_scalar.interpolate(source_vector.sub(i))`` for
    ``i ≠ j`` silently corrupts scattered values in
    ``source_vector.sub(0)`` on rank-boundary partitions under MPI.

    The trigger is the **interleave** of dat-level writes and
    sub-view interpolations on the same vector function — the direct
    numpy write bypasses PyOP2's halo dirty-tracking, and the next
    interpolation reads stale halo state to "correct" what it thinks
    are inconsistent DOFs.

    Workaround
    ----------
    Two parts, both required to be safe:

    1. Use **direct dat slicing** for both read and write of vector
       components (this function for read; :func:`safe_assign_component`
       for write).
    2. At call sites that touch multiple components, **complete all
       reads before any writes** — never interleave a write to one
       column with a read from another within the same loop pass.
       (See :meth:`FiredrakeHyperbolicSolver._apply_slope_limiter` for
       the pattern: gather all limited scalars into a dict, then
       write back in a second pass.)

    Reproducer: ``tutorials/firedrake/firedrake_subcomp_interpolate_bug.py``.
    File against Firedrake when filed; revert this wrapper to a plain
    ``target_scalar.interpolate(source_vector.sub(component_index))``
    AND drop the two-pass discipline at call sites once the upstream
    fix is in a Firedrake release that ``zoomy_firedrake`` pins.

    Parameters
    ----------
    target_scalar : fd.Function
        Scalar-valued DG(p) Function on the same mesh as ``source_vector``.
    source_vector : fd.Function
        Vector-valued DG(p) Function (``VectorFunctionSpace(..., dim=N)``).
    component_index : int
        Index ``0 ≤ component_index < N`` of the vector component to copy.
    """
    target_scalar.dat.data[:] = source_vector.dat.data_ro[:, component_index]


def safe_assign_component(
    target_vector: fd.Function,
    source_scalar: fd.Function,
    component_index: int,
) -> None:
    """Write a scalar-valued Function into component ``component_index``
    of a vector-valued Function — workaround for a Firedrake bug.

    Equivalent (mathematically) to either of::

        target_vector.sub(component_index).assign(source_scalar)
        target_vector.interpolate(fd.as_vector([... source_scalar at component_index ...]))

    but bypasses two distinct Firedrake bugs:

    1. ``target_vector.sub(i).assign(source_scalar)`` misinterprets the
       scalar ``source_scalar.dat.data`` layout against the vector's
       ``(n_DOFs, n_components)`` ``dat.data`` shape, scrambling
       values across components.
    2. ``target_vector.interpolate(fd.as_vector(...))`` triggers an
       L2-projection-style re-assembly that perturbs cell averages of
       components that were supposed to be passed through unchanged,
       breaking mass conservation by ~1% per step under MPI for the
       Malpasset DG(1) test.

    The direct ``dat.data[:, i] = scalar.dat.data_ro[:]`` slice
    mirrors how the rest of zoomy_firedrake writes per-component data
    (see ``MalpassetSolver.set_initial_condition`` in
    ``tutorials/firedrake/malpasset_viscous_v2.py``).

    Companion constraint
    --------------------
    This write bypasses PyOP2's halo dirty-tracking.  A subsequent
    ``.interpolate(target_vector.sub(j))`` for **any** ``j`` will then
    trigger the read-side bug documented in
    :func:`safe_extract_component` (stale halos corrupt other
    columns).  Always finish all reads of ``target_vector`` first,
    then do all writes.

    Reproducer for (1), (2) and the read+write interleaving: see
    ``tutorials/firedrake/firedrake_subcomp_interpolate_bug.py``.
    File against Firedrake when filed; revert this wrapper to plain
    ``target_vector.sub(i).assign(source_scalar)`` once fixed.

    Parameters
    ----------
    target_vector : fd.Function
        Vector-valued DG(p) Function (``VectorFunctionSpace(..., dim=N)``).
    source_scalar : fd.Function
        Scalar-valued DG(p) Function on the same mesh as ``target_vector``.
    component_index : int
        Index ``0 ≤ component_index < N`` of the vector component to write.
    """
    target_vector.dat.data[:, component_index] = source_scalar.dat.data_ro[:]


# (No legacy-model compat.  All Firedrake SWE cases use the canonical
# ``zoomy_core.model.models`` SWE / MalpassetSWE; this module no longer
# references ``zoomy_core.model.models.legacy`` in any way — the whole
# Firedrake stack (solver AND tests) is legacy-free.)
