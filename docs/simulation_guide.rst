Simulation guide
================

Solver overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 23 25 52

   * - Class
     - State layout
     - Use
   * - :class:`~scalabath.simulations_unitary.UnitarySimulation`
     - ``(batch, hilbert_dim)``
     - Dense pure-state evolution for a small Hilbert space.
   * - :class:`~scalabath.simulations_lindblad.LindbladSimulation`
     - ``(batch, dim, dim)``
     - Dense Lindblad master-equation evolution.
   * - :class:`~scalabath.simulations_tcl2.TCL2Simulation`
     - ``(batch, system_dim, system_dim)``
     - Finite-time nonsecular TCL2 reduced dynamics with exponentially
       decomposed correlations.
   * - :class:`~scalabath.simulations_unitary.SystemBathUnitarySimulation`
     - ``(batch, system_dim, *boson_dims)``
     - Tensorized system–bath unitary propagation using a second-order split
       step.
   * - :class:`~scalabath.simulations_lindblad.CoupledLindbladTrajectorySimulation`
     - ``(batch, system_dim, *boson_dims)``
     - Non-Hermitian propagation and stochastic jumps for damped bath modes.

Dense Lindblad evolution
------------------------

Hamiltonians and jump operators may be supplied at construction time. A
rank-two input is broadcast across the ensemble batch.

.. code-block:: python

   import jax.numpy as jnp

   from scalabath import LindbladSimulation, tls

   local = tls()
   gamma = 0.2
   simulation = LindbladSimulation(
       hilbert_dim=2,
       dt=0.01,
       hamiltonian=0.5 * local.sigma_z,
       jump_operators=[jnp.sqrt(gamma) * local.sigma_minus],
   )
   simulation.density_matrices = jnp.asarray(
       [[1.0, 0.0], [0.0, 0.0]],
       dtype=jnp.complex64,
   )

   density_matrices = simulation.step(n_steps=100)
   expectation = simulation.observe(local.sigma_z)

TCL2 evolution
--------------

The TCL2 solver accepts bath correlations decomposed as sums of complex
exponentials,

.. math::

   C_a(t) = \sum_r c_{ar}\exp(-\nu_{ar}t).

Coefficients and exponents use shape ``(n_channels, n_terms)``. A shared
one-dimensional term array is broadcast across channels.

.. code-block:: python

   import jax.numpy as jnp

   from scalabath import TCL2Simulation

   hamiltonian = jnp.asarray([[0.0, 0.2], [0.2, 0.1]], dtype=jnp.complex64)
   coupling = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)
   rho0 = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)

   simulation = TCL2Simulation(
       hamiltonian,
       coupling,
       correlation_coefficients=[0.4, 0.1],
       correlation_exponents=[0.05 + 0.8j, 0.05 - 0.8j],
       dt=0.002,
       density_matrices=rho0,
   )
   simulation.step(n_steps=100)
   diagnostics = simulation.diagnostics()

TCL2 is not generally in GKSL form and need not preserve positivity. Inspect
``diagnostics["minimum_eigenvalue"]`` rather than silently projecting the
state back onto the positive cone.

Tensorized system–bath evolution
--------------------------------

Tensorized states store each local bosonic dimension as a separate array axis.
The coupling for one system–mode pair may be a dense matrix of shape
``(system_dim * mode_dim, system_dim * mode_dim)`` or a compact block array of
shape ``(system_dim, mode_dim, mode_dim)`` when it is diagonal in the system
basis.

.. code-block:: python

   import jax.numpy as jnp

   from scalabath import SystemBathUnitarySimulation, tight_binding_1d

   system_dim = 5
   mode_dim = 4
   omega = 0.8
   coupling_strength = 0.15

   lattice = tight_binding_1d(system_dim, periodic=False)
   simulation = SystemBathUnitarySimulation(
       system_dim=system_dim,
       boson_dims=(mode_dim,),
       boson_freqs=(omega,),
       dt=0.01,
   )
   simulation.set_system_hamiltonian(-lattice.nearest_neighbor_hopping())
   simulation.set_bath_harmonic_hamiltonians()

   displacement = simulation.boson_basis[0].creation + simulation.boson_basis[0].annihilation
   coupling_blocks = jnp.stack(
       [coupling_strength * displacement for _ in range(system_dim)]
   )
   simulation.set_system_bath_hamiltonians([coupling_blocks])

   system_state = jnp.zeros(system_dim, dtype=jnp.complex64).at[system_dim // 2].set(1.0)
   bath_state = jnp.zeros(mode_dim, dtype=jnp.complex64).at[0].set(1.0)
   simulation.set_product_state(system_state, [bath_state])

   simulation.step(n_steps=10)
   rho_system = simulation.reduced_system_density_matrix()

Multi-GPU sharding
------------------

The tensorized state can be sharded over its system axis. The partition
specification must match ``(batch, system, *bath_modes)``:

.. code-block:: python

   import jax
   import numpy as np
   from jax.sharding import Mesh, NamedSharding
   from jax.sharding import PartitionSpec as P

   devices = np.asarray([device for device in jax.devices() if device.platform == "gpu"])
   mesh = Mesh(devices, ("system",))
   state_sharding = NamedSharding(mesh, P(None, "system", None))

   simulation = SystemBathUnitarySimulation(
       system_dim=system_dim,
       boson_dims=(mode_dim,),
       dt=0.01,
       sharding=state_sharding,
   )

For multiple bath modes, append one ``None`` entry per bath axis. The system
dimension must be compatible with the device mesh. Verify placement with
``simulation.state.sharding`` after assigning the initial state.

Accuracy and reproducibility
----------------------------

Record the following with every result:

* package and JAX versions, device type, and dtype;
* lattice size, boundary conditions, Hamiltonian parameters, and unit
  convention;
* time step, total propagation time, and observation interval;
* each bosonic cutoff and any bath-compression fitting window or tolerance;
* all NumPy and JAX random seeds;
* ensemble size, SPA bath-copy count, and trajectory count;
* convergence comparisons for the observable being reported.

The first call at a new combination of shapes and dtypes includes JAX
compilation time. Separate compilation from steady-state timing in performance
measurements.
