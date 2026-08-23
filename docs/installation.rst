Installation
============

Requirements
------------

``scalabath`` supports Python 3.11 and newer. JAX and NumPy are runtime
dependencies. A CPU installation is sufficient for the dense solvers, small
examples, and API development; research-scale tensorized calculations usually
benefit from a CUDA GPU.

Install from source
-------------------

Clone the public repository and install the package into a Conda environment:

.. code-block:: console

   $ git clone https://github.com/salinelake/scalabath.git
   $ cd scalabath
   $ conda create -n scalabath python=3.11 -y
   $ conda activate scalabath
   $ python -m pip install --upgrade pip
   $ python -m pip install .

To work on the source tree, install the development and documentation extras:

.. code-block:: console

   $ python -m pip install -e ".[dev,docs]"

CUDA-enabled JAX
----------------

On a compatible Linux/NVIDIA system, install one of the CUDA extras:

.. code-block:: console

   $ python -m pip install -e ".[gpu-cuda13]"

   # Or, for a CUDA 12 environment:
   $ python -m pip install -e ".[gpu-cuda12]"

The driver and wheel requirements are maintained by JAX. Check the `official
JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
before selecting a wheel on a new machine.

Verify the installation
-----------------------

.. code-block:: console

   $ python -c "import jax, scalabath; print(scalabath.__version__); print(jax.devices())"

The second line identifies the devices visible to JAX. Seeing only a CPU there
means that the current process will not use a GPU.

64-bit calculations
-------------------

JAX commonly defaults to 32-bit arrays. Enable 64-bit support before Python
starts when a calculation or test requires ``complex128``:

.. code-block:: console

   $ export JAX_ENABLE_X64=1

The requested dtype, time step, and all energy/time conversion factors should
be recorded with production results.
