# AGENTS.md

## Project overview

`scalabath` is a scientific Python package for simulating quantum dynamics
on lattice systems. The intended model families are:

- Schrodinger-equation dynamics with bosonic environments.
- Lindblad master-equation dynamics with pseudomodes.
- Lattice Hamiltonians, baths, couplings, observables, and time-evolution
  solvers.

The implementation should be based on JAX so that core kernels can use `jit` 
and GPU execution (including multi-GPU via NamedSharding). NumPy is allowed for host-side setup, plotting,
fixtures, and reference calculations, but JIT-critical paths should use
`jax.numpy`.

## Expected repository layout

Keep the repository organized like this unless there is a strong reason to
change it:

```text
scalabath/
  pyproject.toml
  AGENTS.md
  README.md
  LICENSE
  .gitignore
  src/
    scalabath/
      __init__.py
      systems.py
      operators_base.py
      operators_groups.py
      simulations.py
      utilities.py
      constants.py
  tests/
    unit/
    cpu/
    gpu/
    conftest.py
  examples/
  docs/
    plans/
  jobs/
    perlmutter_gpu_test.slurm
```

Do not commit large simulation outputs, checkpoints, logs, virtual
environments, `.jax_cache`, or machine-specific files. Commit small JSON/CSV
summaries only when they are useful for reproducibility.

## Code Structure

The code is organized into the following modules:

- `systems`: Definition of the physical system.
    Essential classes and functions:
    - `PureStatesEnsemble`: Base class for pure state ensembles (PSE), stored as `(batch, hilbert_dim)` arrays. Main methods: `get_pse`, `set_pse`,`normalize`.
    - `DensityMatrixEnsemble`: Base class for density matrix ensembles (DME), stored as `(batch, hilbert_dim, hilbert_dim)` arrays. Main methods: `get_dme`, `set_dme`,`normalize`.
    Based on these classes, we can also define the `TensorProductPureStatesEnsemble` and `TensorProductDensityMatrixEnsemble` for tensor product of Hilbert spaces.

- `operators_base`: Definition of basic classes of operators. 
    Essential classes and functions:
    - `boson`: class for basic bosonic operator matrices: identity, annihilation, creation, number.
    - `tls`: class for basic Pauli matrices: sigma_x, sigma_y, sigma_z, identity, sigma_plus, sigma_minus.
    - `tight_binding_1d`: class for basic 1D tight-binding operator matrices: hopping, on-site, identity.
    - `tight_binding_2d`: class for basic 2D tight-binding operator matrices: hopping, on-site, identity.

- `operators_groups`: Definition of classes for composite operators. 
    Essential classes and functions:
    - `OperatorGroup`: Base class for general many-body operators acting on a subsystem.
    - `BosonOperatorGroup`: subclass of `OperatorGroup` for many-body boson operators acting on a bosonic subsystem.
    - `SpinOperatorGroup`: subclass of `OperatorGroup` for many-body Pauli operators acting on a spin subsystem.
    - `TightBindingChainOperatorGroup`: subclass of `OperatorGroup` for tight-binding operators acting on a tight-binding chain subsystem.
    - `ComposedOperatorGroups`: subclass of `OperatorGroup` for gluing two or more subsystems together and forming a composite many-body operator that acts on the combined system.
  
  All these classes has a method `sum_operators` to sum up the operators in the group and return the total operator matrix.

- `simulations`: Simulations of the system and environment. Essential classes and functions:
    - `UnitarySimulation`: unitary time-evolution simulation of the system. Main methods: `add_operator_group_to_hamiltonian` (add a static operator group to the Hamiltonian), `step` (perform a time-step), `observe` (get the expectation value of an operator).
    - `LindbladSimulation`: Lindblad master equation simulation of the system. Main methods: `add_operator_group_to_hamiltonian` (add a static operator group to the Hamiltonian),`add_operator_group_to_jumping` (add a static operator group to the jumping operators), `step` (perform a time-step), `observe` (get the expectation value of an operator).
    In addition to these general classes that do not prescribes the structure of the quantum system, we also have specialized classes for systems that are composed of several subsystems, for example, a tight binding chain and a bosonic environment. These specialized classes include `SystemBathUnitarySimulation` and `CoupledLindbladTrajectorySimulation`.

- `utilities`: Utilities for the system and environment. 

**Guidelines**
- Use JAX for the implementation of the core kernels.
- `operators_base.py` and `operators_groups.py` are for definition of the basic operator and many-body operator groups. They are called only for once to generate the big operator matrices before the simulation starts. So they don't need to be JIT-able.
- `simulations.py` is the main module for the simulation of the system and environment. It will store the state of the system and the Hamiltonian, jumping operators, and the time-evolution operator as jax arrays. The most time-consuming part of the simulation is `step` method. So it needs to be JIT-able and allows for multi-GPU execution. In addition to the general classes, the specialized classes for systems that are composed of several subsystems should also be JIT-able.
- What we want to achieve is to (1) allow users to construct physical systems that combines several subsystems, for example, a tight binding system and a bosonic environment containing several modes, and (2) implement efficient, multi-GPU simulations of the physical system with unitary time-evolution or Lindblad master equation time-evolution.


## Local development commands

The conda environment for this project is `scalabath`.
If this environment is not installed, use these commands on a Linux machine with only CPU support:

```bash
conda create -n scalabath python=3.11 -y
conda activate scalabath
pip install -U jax
pip install -e .
```

Use these commands on a Linux machine with CUDA support:

```bash
conda create -n scalabath python=3.11 -y
conda activate scalabath
pip install -U "jax[cuda13]"
pip install -e .
```

For fast local validation:

```bash
JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"
python -m ruff check src tests
python -m ruff format --check src tests
```

For formatting:

```bash
python -m ruff format src tests
python -m ruff check --fix src tests
```
 
## Coding guidelines

- Prefer small, composable functions over large solver scripts.
- Format docstrings for future readthedocs documentation.
- Public APIs should have docstrings and type hints where practical.
- Keep array shape conventions explicit in docstrings. For example, document
  whether a state is shaped as `(n_sites,)`, `(hilbert_dim,)`, or
  `(batch, hilbert_dim)`.
- Use complex dtypes deliberately. For quantum dynamics, tests should include
  complex-valued states/operators.
- Keep JAX-transformable code pure: avoid mutation, file I/O, Python iterators,
  and data-dependent Python control flow inside functions expected to be `jit`ed.
- Use NumPy for reference implementations in tests when that makes correctness
  easier to verify.
- Add tests with every behavior change.
- Do not add new runtime dependencies without a short explanation in the commit
  or PR description.
- Unless there is a strong reason, do not use advanced features of JAX, such as `jax.vmap`, `jax.scan`, `jax.while_loop`, etc, or write compact code that is difficult to understand.
- Unless it causes efficiency issues, do not use jnp.einsum to compute tensor contractions because it causes issues with JIT-compilation. Use jax.numpy's matrix multiplications when possible.
## Testing tiers

Use pytest markers consistently:

- `unit`: small CPU-only tests.
- `cpu`: longer CPU tests.
- `gpu`: requires a CUDA GPU; normally run on Perlmutter.

Examples:

```bash
# Fast checks on a Mac
JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"

# All CPU checks on a Mac or Perlmutter CPU node
JAX_ENABLE_X64=1 python -m pytest -m "not gpu"

# GPU checks, normally inside a Slurm job
JAX_ENABLE_X64=1 python -m pytest -m "gpu"
```

## Perlmutter principles

This code will often be developed on Perlmutter supercomputer. In that case, Codex CLI will run on login node.

Never run heavy CPU/GPU simulations on Perlmutter login nodes. On login nodes,
only do lightweight actions such as `git pull`, editing small files, inspecting
logs, or submitting/monitoring Slurm jobs.

For quick validation that may require GPU, you can use the following command to activate an interactive session on Perlmutter:
```bash
salloc -N1 -n32 -t 04:00:00 -C gpu -q shared_interactive --gres=gpu:1 -A m1027
```
This will give you a login node with 32 CPU cores and 1 GPU. This is typically sufficient for most of the tests.

salloc may be forbidden if you are in sandbox mode. In that case, get out of the sandbox. 

In rare cases, you may need to run a job that lasts for a long time. In that case, you should submit a job to the Perlmutter queue using a slurm script. 
Use `jobs/perlmutter_gpu_test.slurm` as a template, and modify the script to suit your needs.

## Codex working rules

- Start by reading `AGENTS.md`, `pyproject.toml`, `README.md` and relevant tests.
- Keep diffs small and reviewable.
- For nontrivial solver changes, write or update a short plan in `docs/plans/`.
- Run the fastest relevant local tests after edits.
- If GPU validation is required, update the Perlmutter instructions or job script
  but do not claim GPU validation passed unless a Perlmutter job actually ran.
- Do not invent physical formulas. If adding a method, include a reference in
  comments, docs, or the PR description when appropriate.
- Preserve reproducibility: record random seeds, dtype assumptions, lattice size,
  time step, and solver tolerances in tests and examples.
- Never print, commit, or request secrets, tokens, private keys, or credentials.

## Done criteria

A change is done when:

- The package imports successfully.
- Fast CPU tests pass with `JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"`.
- Ruff checks pass.
- New or changed behavior has tests.
- GPU-sensitive changes either have a passing Perlmutter job ID recorded in the
  PR/commit notes or clearly state that GPU validation remains to be run.