"""Every public name renders in the docs: ``__all__`` members must carry docstrings.

Sphinx ``automodule :members:`` silently omits objects without a docstring,
so a missing docstring is a missing API-reference entry on openactuarial.org.
This turns that silent omission into a test failure.
"""
from __future__ import annotations

import inspect

import extremeloss


def test_every_public_name_has_a_docstring():
    missing = []
    for name in extremeloss.__all__:
        obj = getattr(extremeloss, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue  # autodoc documents plain data attributes differently
        doc = inspect.getdoc(obj)
        if not doc or not doc.strip():
            missing.append(name)
    assert not missing, (
        f"public names missing docstrings (silently absent from autodoc): {missing}"
    )
