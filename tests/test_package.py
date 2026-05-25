"""Package import smoke tests."""

from __future__ import annotations


def test_package_imports() -> None:
    import hgraphml

    assert hgraphml.__version__
