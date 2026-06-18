# scalabath

`scalabath` is a JAX-based Python package for scalable simulation of quantum dynamics on lattice systems coupled to bosonic environments. It provides dense
Schrodinger-equation and Lindblad master-equation solvers, tensor-product
system-bath evolution, and operator builders for bosons, two-level systems, and
tight-binding lattices.

## Installation

The package requires Python 3.11 or newer. For CPU development from a source
checkout:

```bash
git clone https://github.com/salinelake/scalabath.git
cd scalabath
conda create -n scalabath python=3.11 -y
conda activate scalabath
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For Linux systems with NVIDIA GPUs, install a CUDA-enabled JAX wheel through one
of the project extras. Choose the CUDA extra that matches the machine:

```bash
conda create -n scalabath python=3.11 -y
conda activate scalabath
python -m pip install -U pip
python -m pip install -e ".[dev,gpu-cuda13]"

# CUDA 12 systems can use:
# python -m pip install -e ".[dev,gpu-cuda12]"
```

For a runtime-only editable install, omit the `dev` extra:

```bash
python -m pip install -e .
```

## Highlighted features

- JAX-backed state containers for batched pure-state and density-matrix
  ensembles.
- Dense unitary and Lindblad simulation classes with JIT-compiled step kernels.
- Tensor-product system-bath evolution for states shaped as
  `(batch, system_dim, *boson_dims)`, avoiding construction of one full dense
  Hamiltonian for every system-bath term.
- Operator builders for bosonic modes, two-level systems, one-dimensional
  tight-binding chains, two-dimensional tight-binding lattices, and composed
  subsystem operators.
- Batch-aware operator prefactors and random-phase system-bath couplings for
  ensemble simulations.

## Code Structure

- `src/scalabath/systems.py`: pure-state and density-matrix ensemble
  containers, including tensor-product layouts and reduced density matrices.
- `src/scalabath/operators_base.py`: local boson, two-level-system, and
  tight-binding operator matrices.
- `src/scalabath/operators_groups.py`: static sums of many-body operators and
  tensor products of subsystem operator groups.
- `src/scalabath/simulations.py`: dense unitary evolution, dense Lindblad
  evolution, tensorized system-bath unitary evolution, and coupled Lindblad
  trajectory evolution.
- `src/scalabath/utilities.py`: shared linear algebra helpers such as adjoints,
  Kronecker products, traces, and expectation values.
- `src/scalabath/constants.py`: physical constants and unit conversions used by
  examples.
- `examples/`: runnable model scripts.
- `tests/`: unit tests for operators, state containers, simulation classes, and
  utilities.

## Applications

One major application of the package is to simulate a reduced open quantum
system where the multi-site, multi-mode environment is replaced by a single
multi-mode bosonic bath through random phase approximation. There are two
scenarios: (1) the Hamiltonian case and (2) the Lindblad case.

### Hamiltonian case

Consider the following problem:

$$
\hat{H}_{B}=\sum_{k=1}^{N}\omega_{k}\hat{b}_{k}^{\dagger}\hat{b}_{k},
$$

and

$$
\hat{H}_{SB}=\sum_{j=1}^{n} \sum_{k=1}^{N} |j\rangle\langle j|  \otimes
(g_{k}e^{i\phi_{jk}}\hat{b}_{k}
+g_{k}^{*} e^{-i\phi_{jk}^{*}}\hat{b}_{k}^{\dagger}).
$$

Then the BCF is given by:

$$
C_{jj^{\prime}}(t)=\sum_{k}g_{k}g_{k}^{*}
e^{i(\phi_{jk}-\phi_{j^{\prime}k})}c_{k}(t),
$$

where $c_{k}(t)$ corresponds to the k-th mode of the bath. This
$C_{jj^{\prime}}(t)$ is not diagonal.

To make it diagonal, draw $\phi_{jk}$ as random numbers uniformly distributed in
$[0,2\pi)$. Then:

$$
\mathbb{E}(C_{jj^{\prime}}(t))
=\sum_{k}g_{k}g_{k}^{*}
\mathbb{E}(e^{i(\phi_{jk}-\phi_{j^{\prime}k})})c_{k}(t)
=\sum_{k}g_{k}g_{k}^{*}c_{k}(t)\delta_{jj^{\prime}}.
$$

### Coupled Lindbladian case

In the coupled Lindbladian case, we have:

$$
\begin{aligned}
\partial_{t}\hat{\rho}
&= -i[\hat{H}_{S}+\hat{H}_{B}+\hat{H}_{SB},\hat{\rho}] \\
&\quad + \sum_{k,k^{\prime}}\Gamma_{kk^{\prime}}
\left(2\hat{b}_{k}\hat{\rho}_{k^{\prime}}^{\dagger}
-\{\hat{b}_{k^{\prime}}^{\dagger}\hat{b}_{k},\hat{\rho}\}\right).
\end{aligned}
$$

$$
\hat{H}_{B}=\sum_{k,k^{\prime}}h_{kk^{\prime}}
\hat{b}_{k}^{\dagger}\hat{b}_{k^{\prime}},
$$

and

$$
\hat{H}_{SB}=\sum_{j=1}^{n}\sum_{k=1}^{N}|j\rangle\langle j|\otimes
(g_{k}e^{i\phi_{j}}\hat{b}_{k}
+g_{k}^{*}e^{-i\phi_{j}^{*}}\hat{b}_{k}^{\dagger}).
$$

where again, $\phi_{j}$ and $\phi_{j^{\prime}}$ are random numbers uniformly
distributed in $[0,2\pi)$. Then:

$$
C_{jj^{\prime}}(t)
=g^{\dagger}e^{(-ih-\Gamma)t}ge^{i(\phi_{j}-\phi_{j^{\prime}})}.
$$

Taking expectation, we have:

$$
\mathbb{E}(C_{jj^{\prime}}(t))
=g^{\dagger}e^{(-ih-\Gamma)t}g
\mathbb{E}(e^{i(\phi_{j}-\phi_{j^{\prime}})})
=g^{\dagger}e^{(-ih-\Gamma)t}g\delta_{jj^{\prime}}.
$$

In practice, we will use diagonal dissipation. So $\Gamma$ is a diagonal matrix.

## Example: Carrier transport in Rubrene crystal

The Rubrene example in `examples/rubrene` simulates carrier transport in a
one-dimensional Holstein-type model. The system is a tight-binding chain, and
the bath mode frequencies and reorganization energies are read from
`coupling_const.csv`. The simulation samples thermal bath occupations, draws
site-dependent random phases, evolves the tensor-product wavefunction with
`SystemBathUnitarySimulation`, and saves site populations for postprocessing.

The implemented Hamiltonian has the form:

$$
\begin{aligned}
\hat{H}
&= J\sum_{j}\left(|j\rangle\langle j+1|
+|j+1\rangle\langle j|\right) \\
&\quad + \sum_{m}\omega_{m}\hat{b}_{m}^{\dagger}\hat{b}_{m} \\
&\quad + \sum_{j,m}|j\rangle\langle j|\otimes g_{m}
\left(e^{i\phi_{jm}}\hat{b}_{m}
+e^{-i\phi_{jm}}\hat{b}_{m}^{\dagger}\right).
\end{aligned}
$$

Run a small CPU smoke test from the example directory:

```bash
cd examples/rubrene
JAX_ENABLE_X64=1 python 01.main.py \
  --chain-length 8 \
  --num-modes 1 \
  --sample-time-fs 2 \
  --sample-period-fs 1 \
  --batch-size 1 \
  --run-id 0
```

A production-scale run matching the current postprocessing script uses the
default Rubrene dimensions:

```bash
cd examples/rubrene
JAX_ENABLE_X64=1 python 01.main.py \
  --temperature 300 \
  --chain-length 150 \
  --sample-time-fs 300 \
  --sample-period-fs 1 \
  --batch-size 5 \
  --run-id 0
```


## Example: Exciton transport in bio-complex BCHL chain

The BCHL chain example in `examples/BCHL_chain` simulates exciton transport in a one-dimensional BCHL chain. The system is a n-site tight-binding chain coupled to an effective bosonic bath. For BCHL, n=19. The number of bosonic modes is denoted by N. 

The bath Hamiltonian is not diagonal and is given by:
$$
\hat{H}_{B}=\sum_{k,k^{\prime}}h_{kk^{\prime}}
\hat{b}_{k}^{\dagger}\hat{b}_{k^{\prime}},
$$

The system-bath coupling is given by the same form as the Rubrene example, but the coupling strengths and phases are different.
$$
\hat{H}_{SB}=\sum_{j=1}^{n}\sum_{k=1}^{N}|j\rangle\langle j|\otimes
(g_{k}e^{i\phi_{j}}\hat{b}_{k}
+g_{k}^{*}e^{-i\phi_{j}^{*}}\hat{b}_{k}^{\dagger}).
$$

The evolution of the density matrix is given by:
$$
\partial_{t}\hat{\rho}
&= -i[\hat{H}_{S}+\hat{H}_{B}+\hat{H}_{SB},\hat{\rho}] \\
&\quad + \sum_{k}\gamma_{k}
\left(\hat{b}_{k}\hat{\rho}\hat{b}_{k}^{\dagger}
-0.5\{\hat{b}_{k}^{\dagger}\hat{b}_{k},\hat{\rho}\}\right).
$$
 
The matrix $h_{kk^{\prime}}$, the coupling strengths $g_{k}$, and the damping rates $\gamma_{k}$ are read from `parameters.json`.

The initial state of the system is a product state. The initial state of the bath is the thermal state at temperature T. The initial state of the system is the center site of the chain.
