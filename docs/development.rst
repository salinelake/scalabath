Development
===========

Set up an editable checkout with all development tools:

.. code-block:: console

   $ python -m pip install -e ".[dev,docs]"

Fast validation
---------------

The current tests are marked ``unit``. The ``cpu`` and ``gpu`` markers are
registered for longer runs, and no test uses them yet. Until those tests
exist, both selections below run the same unit suite:

.. code-block:: console

   $ JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"
   $ JAX_ENABLE_X64=1 python -m pytest -m "not gpu"
   $ python -m ruff check src tests
   $ python -m ruff format --check src tests

Run any future test marked ``gpu`` on a CUDA compute node. On Perlmutter, keep
heavy simulations off the login node.

Documentation
-------------

Build the HTML documentation and fail on warnings:

.. code-block:: console

   $ sphinx-build -M html docs docs/_build -W --keep-going

The generated site is written to ``docs/_build/html``. Read the Docs uses the
same ``docs/conf.py`` configuration through the repository-level
``.readthedocs.yaml`` file.

Documentation changes should keep public signatures, array shapes, dtype
requirements, and solver limitations explicit. API pages are generated from
public docstrings, so update the implementation docstring whenever a public
interface changes.
