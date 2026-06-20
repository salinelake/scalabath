# Tensor-product system-bath simulations

## Context

The dense simulation path stores pure states as `(batch, hilbert_dim)` and
operators as dense `(batch, hilbert_dim, hilbert_dim)` arrays. This remains a
useful general fallback, but it is inefficient for the multi-site bath models in
`multi_site_bath_reduction.pdf`, where most factors act only on the system, one
bath mode, or a system-mode pair.

The scratch implementations in `inspirations/` use a more efficient layout:

```text
(batch_size, system_dim, mode_0_dim, ..., mode_n_dim)
```

Local propagators are then applied by reshaping only the relevant tensor axes.

## Plan

1. Keep `operators_base.py` and `operators_groups.py` as host-side operator
   builders. They are still useful for constructing local subsystem matrices or
   dense reference operators, and they do not need to become JIT-critical.
2. Preserve the existing flat `PureStatesEnsemble` and `DensityMatrixEnsemble`
   APIs for general dense simulations.
3. Add general tensor-product pure-state and density-matrix ensembles with
   explicit subsystem dimensions and batch-first storage.
4. Keep system-plus-boson behavior as simulation-level convenience methods on
   top of the generic tensor-product ensembles.
5. Keep dense `UnitarySimulation` and `LindbladSimulation` as broad fallbacks.
6. Add `SystemBathUnitarySimulation` for second-order Trotter evolution with
   local system, bath-mode, and system-mode coupling factors.
7. Add `CoupledLindbladTrajectorySimulation` for the non-Hermitian coupled
   Lindbladian trajectory pattern used by the scratch code.

## Notes

The tensor-product path avoids materializing full system-bath operators during
time stepping. It still accepts matrices produced by the existing operator
builders when those matrices are local to the system, a mode, or a system-mode
pair.

When a state `NamedSharding` is provided, prepared propagators should be placed
on the same mesh with replicated sharding. This keeps the tensor-product step
kernel from receiving single-device propagator inputs while preserving the
state sharding over the system axis.
