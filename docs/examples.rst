Examples
========

The repository contains two end-to-end application directories. Their default
parameters are intended for scientific calculations, not quick local tests.
Reduce the state dimensions and propagation time first, estimate memory use,
and run production GPU workloads on a compute node.

Rubrene carrier transport
-------------------------

The `Rubrene example
<https://github.com/salinelake/scalabath/tree/main/examples/rubrene>`_ simulates
carrier transport on a one-dimensional Holstein chain. It reads nine
vibrational frequencies and reorganization energies from
``coupling_const.csv``, samples thermal bath occupations and stochastic
phases, and propagates the tensorized wavefunction with
:class:`~scalabath.simulations_unitary.SystemBathUnitarySimulation`.

A small CPU smoke run is:

.. code-block:: console

   $ cd examples/rubrene
   $ JAX_ENABLE_X64=1 python 01.main.py \
       --chain-length 8 \
       --num-modes 1 \
       --boson-dims 2 \
       --sample-time-fs 0.2 \
       --sample-period-fs 0.1 \
       --batch-size 1 \
       --run-id 0

The script records sampled occupations, phases, populations, and a JSON
metadata file. Use distinct ``run-id`` values for independent ensembles.

Bacteriochlorophyll chain
-------------------------

The `BChl example
<https://github.com/salinelake/scalabath/tree/main/examples/BCHL_chain>`_ evolves
a 19-site chain coupled to six compressed, damped bosonic modes. The coupled
bath Hamiltonian, couplings, and damping parameters are read from
``parameters.json`` and propagated with
:class:`~scalabath.simulations_lindblad.CoupledLindbladTrajectorySimulation`.

A reduced-cutoff smoke command is:

.. code-block:: console

   $ cd examples/BCHL_chain
   $ JAX_ENABLE_X64=1 python 01.main.py \
       --boson-dims 2,2,2,2,2,2 \
       --sample-time-fs 0.1 \
       --sample-period-fs 0.1 \
       --batch-size 1 \
       --run-id 0

Postprocessing
--------------

Each example includes a numbered postprocessing script. Run it only after the
expected ensemble files have been generated. The reference CSV files are
small comparison data; generated ``.npz`` trajectories and plots are excluded
from version control.
