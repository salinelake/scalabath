Quick start
===========

This example constructs an open five-site tight-binding chain, initializes a
particle on its center site, and performs dense unitary propagation.

.. code-block:: python

   import jax.numpy as jnp

   from scalabath import UnitarySimulation, tight_binding_1d

   lattice = tight_binding_1d(5, periodic=False)
   hamiltonian = -lattice.nearest_neighbor_hopping(amplitude=1.0)

   simulation = UnitarySimulation(
       hilbert_dim=5,
       dt=0.05,
       hamiltonian=hamiltonian,
   )
   initial_state = jnp.zeros(5, dtype=jnp.complex64).at[2].set(1.0)
   simulation.state = initial_state

   simulation.step(n_steps=20)
   populations = jnp.abs(simulation.state[0]) ** 2
   center_population = simulation.observe(lattice.on_site(2))

   print(populations)
   print(center_population)

The leading state axis is always the ensemble batch. A one-dimensional initial
state is broadcast when ``batch_size`` is greater than one:

.. code-block:: python

   ensemble = UnitarySimulation(
       hilbert_dim=5,
       batch_size=32,
       dt=0.05,
       hamiltonian=hamiltonian,
   )
   ensemble.state = initial_state
   assert ensemble.state.shape == (32, 5)

The first call to :meth:`~scalabath.simulations_unitary.UnitarySimulation.step`
may be slower because JAX compiles the evolution kernel. Subsequent calls with
compatible array shapes reuse compiled code.

Next steps
----------

* Read :doc:`simulation_guide` to select a solver and understand its state
  layout.
* Read :doc:`spa` before constructing a stochastic-phase bath model.
* Use :doc:`examples` for complete Rubrene and bacteriochlorophyll workflows.
* Consult :doc:`api/index` for constructor arguments and method signatures.
