scalabath
=========

Scalable JAX simulations of open quantum dynamics on lattice systems
---------------------------------------------------------------------

``scalabath`` provides dense and tensorized solvers for quantum lattice models
coupled to bosonic environments. It is the reference software implementation
supporting the stochastic phase algorithm (SPA), a bath-reduction strategy for
single-quasiparticle transport with identical, independent local
environments.

The package combines host-side operator construction with JAX-compiled
time-evolution kernels. Its tensorized solvers keep states in the layout
``(batch, system_dim, *boson_dims)`` and accept optional JAX sharding for GPU
execution.

Start with :doc:`installation` and :doc:`quickstart`. Readers using the method
from the accompanying paper should also read :doc:`spa` and the convergence
guidance in :doc:`simulation_guide`.

.. note::

   ``scalabath`` is research software. Validate physical models and converge
   all numerical approximations for the observable and parameter regime being
   studied.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   spa
   simulation_guide
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   development
   citation

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
