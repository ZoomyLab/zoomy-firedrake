"""Shared fixtures for the zoomy_firedrake test suite.

Every test in this package requires Firedrake; if the import fails the
whole module is skipped via the ``firedrake_available`` fixture below.
The Riemann/UFL plumbing in ``zoomy_core`` is exercised by its own
test suite which does not need Firedrake.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def firedrake_available():
    """Skip the entire suite when Firedrake is not importable."""
    try:
        import firedrake  # noqa: F401
    except ImportError:
        pytest.skip(
            "Firedrake is not installed in the current environment; "
            "zoomy_firedrake tests need a Firedrake/PETSc build.",
            allow_module_level=True,
        )
