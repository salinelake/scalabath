Development
===========

Set up an editable checkout with all development tools:

.. code-block:: console

   $ python -m pip install -e ".[dev,docs]"

Fast validation
---------------

Run the unit tier and static checks before submitting changes:

.. code-block:: console

   $ JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"
   $ python -m ruff check src tests
   $ python -m ruff format --check src tests

Longer CPU tests are selected with:

.. code-block:: console

   $ JAX_ENABLE_X64=1 python -m pytest -m "not gpu"

Tests marked ``gpu`` require a CUDA device. On Perlmutter, submit them to a GPU
compute node; do not execute heavy simulations on a login node.

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
