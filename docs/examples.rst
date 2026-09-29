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

The script writes sampled occupations, phases, populations, and a JSON
metadata file. ``--run-id`` selects the output filename. Phases and thermal
occupations come from NumPy's global random state, and this script does not
seed it from ``--run-id``. Set that state before starting the process when a
run must be reproduced. A fresh process begins from the same default state, so
a different ``--run-id`` repeats the same samples.

Bacteriochlorophyll chain
-------------------------

The `BChl example
<https://github.com/salinelake/scalabath/tree/main/examples/BCHL_chain>`_ evolves
a 19-site chain coupled to six compressed, damped bosonic modes.
``00.preprocess.py`` writes ``parameters.json`` by compressing the original
50-mode bath with ``realtimebath``. ``01.main.py`` reads that file and
propagates the chain with
:class:`~scalabath.simulations_lindblad.CoupledLindbladTrajectorySimulation`.
``--run-id`` seeds the phase sampler.

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
