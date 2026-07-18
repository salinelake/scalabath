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
- `src/scalabath/simulations_unitary.py`: dense unitary evolution and
  tensorized system-bath unitary evolution.
- `src/scalabath/simulations_lindblad.py`: dense Lindblad evolution and
  coupled Lindblad trajectory evolution.
- `src/scalabath/simulations.py`: compatibility re-exports for the simulation
  classes.
- `src/scalabath/utilities.py`: shared linear algebra helpers such as adjoints,
  Kronecker products, traces, and expectation values.
- `src/scalabath/constants.py`: physical constants and unit conversions used by
  examples.
- `examples/`: runnable model scripts.
- `tests/`: unit tests for operators, state containers, simulation classes, and
  utilities.

## Applications

One major application of the package is to simulate quantum transport in extended systems where the multi-site, multi-mode environment is replaced by a single
multi-mode bosonic bath through random phase approximation.  We call this the stochastic phase algorithm (SPA).

### Quantum transport models
We focus on quantum transport described by an $N_{\text{s}}$-site Holstein Hamiltonian, $\hat H=\hat H_{\text{s}}+\hat H_{\text{b}}+\hat H_{\text{sb}}$. 
The tight-binding Hamiltonian is 
$\hat H_{\text{s}}=\sum_{i=1}^{N_{\text{s}}}U_i |i\rangle\langle i|+\sum_{  i\neq j}^{N_{\text{s}}}V_{ij}|i\rangle\langle j|$, 
where $|i\rangle$ denotes the tight-binding state on site-$i$, $U_i$ is the site energy, and $V_{ij}$ is the hopping amplitude. The connectivity of the tight-binding network is arbitrary.
The vibronic environment of each site consists of $n$  bosonic modes, $\hat H_{\rm b}=\sum_{i=1}^{N_{\text{s}}}\sum_{\alpha=1}^{n}\omega_{\alpha}\hat b_{i\alpha}^\dagger \hat b_{i\alpha}$, where $\hat b_{i\alpha}^\dagger$ creates a vibronic excitation in mode $\alpha$ attached to site $i$. The system-bath coupling is local and diagonal,
$$
    \hat H_{\text{sb}}=\sum_{i=1}^{N_{\text{s}}}  |i\rangle\langle i| \otimes \sum_{\alpha=1}^{n}g_{\alpha}\omega_{\alpha}(\hat b_{i\alpha}^\dagger+\hat b_{i\alpha}),
$$
with dimensionless coupling $g_{\alpha}$.
The local reorganization energy $\lambda=\sum_{\alpha=1}^{n}g_\alpha^2\omega_\alpha$ measures the energetic stabilization due to EVC. 
The intermediate-coupling regime corresponds to $\lambda$ being of the same order as a typical hopping amplitude, so neither weak-coupling nor strong-coupling descriptions are reliable.

We assume the initial state is a tensor product of a system state $\rho_{\text{s}}(0)$ and a thermal environmental state $\rho_{\text{b},\beta}\propto\exp(-\beta \hat H_{\text{b}})$. We denote the reduced system density operator by $\rho_{\text{s}}(t)$. The environmental influence on $\rho_{\text{s}}(t)$ is encoded in the BCF $C_{ij}(t)$ between site-$i$ and $j$, which can be obtained from first-principles calculations or from vibronic spectral densities inferred from spectroscopic data. 
For homogeneous Gaussian baths, $C(t)$ is diagonal. 
$C_{ij}(t)=c(t)\delta_{ij}$, 
with
$$
c(t)=\sum_{\alpha=1}^{n} g_\alpha^2\omega_\alpha^2
[\coth(\beta\omega_\alpha/2)\cos(\omega_\alpha t)-\mathrm{i}\sin(\omega_\alpha t)].
$$
For a continuous bath with spectral density $J(\omega)$, this discrete sum is replaced by an integral over $\omega$. 

### Stochastic phase algorithm (SPA)
SPA starts from a representation of a single local vibronic environment whose scalar BCF is $c(t)$. This representation may be the original set of physical bath modes or a compressed auxiliary bath obtained from existing bath-compression methods. We write both cases in a unified form using $n_{\text{b}}$ bath modes, a bath energy matrix $K$, a damping matrix $\Gamma$, and a coupling vector $ \epsilon$. For an uncompressed harmonic bath, $K_{kl}=\omega_k\delta_{kl}$ ($k,l\in[1,n_{\text{b}}]$), $\Gamma=0$, and $\epsilon_k=g_k\omega_k$. For a compressed coupled-Lindblad bath, $K$ is generally not diagonal. $K$, $\Gamma$, and $\epsilon$ are optimized so that 
$c(t)=\epsilon^\dagger e^{(-iK-\Gamma)t}\epsilon$ for 
$0\leq t\leq \tau$

 
SPA deals with an extended system with $N_{\text{s}}$ sites and $N_{\text{s}}$ such local baths. SPA replaces the $N_{\text{s}}$ local baths with $R$ independent copies of the local bath and defines the SPA total Hamiltonian 
$\hat H_{\text{r}}^{(R)} = \hat H_{\text{s}}+ \hat H_{\text{b,r}}^{(R)}  + \hat H_{\text{sb,r}}^{(R)}$.
The reduced bath Hamiltonian is $\hat H_{\text{b,r}}^{(R)}=\sum_{a=1}^{R}\sum_{k,l=1}^{n_{\text {b}}} K_{kl}\hat b_{a,k}^\dagger \hat b_{a,l}$, where $\hat b_{a,k}^\dagger$ creates an excitation in mode $k$ associated with the global bath $a$. 
The reduced system-bath coupling Hamiltonian is
$$
\hat H_{\text{sb, r}}^{(R)}
= \sum_{i=1}^{N_{\text{s}}} |i\rangle\langle i|  \otimes
\sum_{a=1}^{R}\sum_{k=1}^{n_{\text b}}
\frac{\epsilon_k}{\sqrt R}
\left(
r_i^{(a)} \hat b_{a,k}^\dagger + r_i^{(a)*} \hat b_{a,k}
\right).
$$
The stochastic phase factor $r_i^{(a)}=\mathrm e^{i\theta_i^{(a)}}$ is sampled at each site $i$ and for each bath channel $a$, with $\theta_i^{(a)}$ uniformly distributed in $[0,2\pi)$.  
Because $\mathbb E_\theta[R^{-1}\sum_{a=1}^{R}r_i^{(a)}r_j^{(a)*}]=\delta_{ij}$, the averaged BCF of the reduced system satisfies 
$\mathbb E_\theta\!\left[C_{ij}^{(R)}(t)\right]
= c(t)\delta_{ij}$, and thus reproduces the original BCFs without cross-site correlations.

The quantum dynamics in SPA is governed by the Lindblad equation $\dot{\rho}=-\mathrm i [\hat H_{\text{r}}^{(R)},\rho] + \mathcal D^{(R)}(\rho)$, where 
$$
    \mathcal D^{(R)}(\rho) = \sum_{a=1}^{R}\sum_{k,l=1}^{n_{\text r}}\Gamma_{kl}\left(2\hat b_{a,k}\rho \hat b_{a,l}^\dagger - \{\hat b_{a,l}^\dagger \hat b_{a,k},\rho\}\right).
$$
In the coupled-Lindblad-mode convention, $\Gamma$ differs by a factor of two from the standard Lindblad convention, so that $c(t)=g^\dagger e^{-iHt-\Gamma t}g$ contains no additional factor of two in the damping term.

When $\Gamma=0$, the dissipator vanishes and the dynamics reduce to ordinary unitary evolution. 


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
