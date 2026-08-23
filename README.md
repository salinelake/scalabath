# scalabath

[![Documentation Status](https://readthedocs.org/projects/scalabath/badge/?version=latest)](https://scalabath.readthedocs.io/en/latest/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`scalabath` is a JAX-based Python package for simulating open quantum dynamics
on lattice systems. It provides the simulation components used by the
stochastic phase algorithm (SPA), which replaces many site-resolved bosonic
environments with a small number of shared baths carrying stochastic,
site-dependent phases.

The package accompanies the paper *Scalable simulation of non-Markovian
quantum transport by stochastic-phase bath reduction*. In the setting studied
there—single-quasiparticle transport with identical, independent local
environments—SPA restores the target two-point bath correlations by phase
averaging while avoiding one explicit bath copy per lattice site.

## Highlights

- Dense Schrödinger, Lindblad, and finite-time nonsecular TCL2 evolution.
- Tensorized system–bath state propagation without constructing the complete
  system-plus-bath Hamiltonian for every local term.
- Stochastic Lindblad trajectories for coupled pseudomode baths.
- Batched simulations with complex, realization-dependent operator
  prefactors.
- Optional JAX `NamedSharding` for distributing the system axis across GPUs.
- Operator builders for bosons, two-level systems, 1D chains, 2D lattices, and
  composed Hilbert spaces.

## Installation

`scalabath` requires Python 3.11 or newer. To install the released source on a
CPU machine:

```bash
git clone https://github.com/salinelake/scalabath.git
cd scalabath
conda create -n scalabath python=3.11 -y
conda activate scalabath
python -m pip install --upgrade pip
python -m pip install .
```

For development, including tests and documentation:

```bash
python -m pip install -e ".[dev,docs]"
```

On Linux systems with NVIDIA GPUs, select the extra matching the installed CUDA
runtime:

```bash
python -m pip install -e ".[gpu-cuda13]"

# CUDA 12 alternative:
# python -m pip install -e ".[gpu-cuda12]"
```

JAX compilation and device selection are controlled by JAX. Enable 64-bit
arrays before starting Python when a calculation requires them:

```bash
export JAX_ENABLE_X64=1
```

## Quick start

The following example evolves a particle initially localized at the center of
an open five-site chain:

```python
import jax.numpy as jnp

from scalabath import UnitarySimulation, tight_binding_1d

lattice = tight_binding_1d(5, periodic=False)
hamiltonian = -lattice.nearest_neighbor_hopping(amplitude=1.0)

simulation = UnitarySimulation(5, dt=0.05, hamiltonian=hamiltonian)
initial_state = jnp.zeros(5, dtype=jnp.complex64).at[2].set(1.0)
simulation.state = initial_state

simulation.step(n_steps=20)
populations = jnp.abs(simulation.state[0]) ** 2
print(populations)
```

All state containers are batch-first. Dense pure states have shape
`(batch, hilbert_dim)`, dense density matrices have shape
`(batch, hilbert_dim, hilbert_dim)`, and tensorized system–bath states have
shape `(batch, system_dim, *boson_dims)`.

## Choosing a solver

| Solver | State representation | Intended use |
| --- | --- | --- |
| `UnitarySimulation` | Dense pure states | Exact dense unitary propagation of small Hilbert spaces |
| `LindbladSimulation` | Dense density matrices | Small open systems in Lindblad form |
| `TCL2Simulation` | Dense density matrices | Finite-time, nonsecular second-order reduced dynamics |
| `SystemBathUnitarySimulation` | Tensorized pure states | Unitary system–bath dynamics and SPA with explicit modes |
| `CoupledLindbladTrajectorySimulation` | Tensorized trajectories | SPA with damped or coupled pseudomodes |

The tensorized solvers avoid dense operators spanning every bath mode, but the
bath-state size still scales as the product of the bosonic cutoff dimensions.
Converge the time step, every bosonic cutoff, trajectory/phase sample count,
and—when using multiple shared bath copies—the copy count for the observable of
interest.

## Examples

Two end-to-end applications are included:

- [`examples/rubrene`](examples/rubrene) simulates carrier transport in a
  one-dimensional Holstein model with a nine-mode molecular bath.
- [`examples/BCHL_chain`](examples/BCHL_chain) simulates exciton population
  dynamics in a 19-site bacteriochlorophyll chain coupled to compressed,
  damped bosonic modes.

For example, a small CPU smoke run of the Rubrene script is:

```bash
cd examples/rubrene
JAX_ENABLE_X64=1 python 01.main.py \
  --chain-length 8 \
  --num-modes 1 \
  --boson-dims 2 \
  --sample-time-fs 0.2 \
  --sample-period-fs 0.1 \
  --batch-size 1 \
  --run-id 0
```

The default example dimensions are research-scale calculations and may require
a GPU and substantial memory. Start with reduced cutoffs and propagation times
before launching production runs.

## Documentation

The full documentation is available at
[scalabath.readthedocs.io](https://scalabath.readthedocs.io/en/latest/). It
includes installation instructions, the SPA formulation, solver and sharding
guides, reproducibility guidance, examples, and the complete Python API.

To build it locally:

```bash
python -m pip install -e ".[docs]"
sphinx-build -M html docs docs/_build -W --keep-going
```

Open `docs/_build/html/index.html` after the build completes.

## Citation

If `scalabath` contributes to published work, please cite both the software
release and the accompanying paper:

> *Scalable simulation of non-Markovian quantum transport by stochastic-phase
> bath reduction.*

The arXiv identifier and complete BibTeX entry will be added when the preprint
is public.

## Development

Contributions and issue reports are welcome. The fast validation suite is:

```bash
JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"
python -m ruff check src tests
python -m ruff format --check src tests
```

GPU checks are marked `gpu` and should be run on a CUDA compute node rather
than a shared login node. See the documentation for the full testing workflow.

## License

`scalabath` is distributed under the [MIT License](LICENSE).
