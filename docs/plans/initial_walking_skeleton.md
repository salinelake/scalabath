# Initial Walking Skeleton

This first package skeleton establishes the public modules named in
`AGENTS.md` without trying to solve the full modeling problem at once.

Implemented scope:

- JAX array utilities for adjoints, `A B A^dagger`, Kronecker composition, and
  ensemble expectations.
- Pure-state and density-matrix ensemble containers with explicit shape
  conventions.
- Basic boson, two-level-system, 1D tight-binding, and 2D tight-binding
  operators.
- Static operator groups and tensor-product composition across subsystems.
- Minimal unitary and Lindblad simulation classes with JIT-compiled stepping
  kernels.

Near-term follow-up:

- Add higher-order Lindblad integrators and regression tests against NumPy
  references.
- Add sharding-oriented APIs once the single-device data model stabilizes.
- Build examples for the 1D tight-binding chain coupled to bosonic modes.
