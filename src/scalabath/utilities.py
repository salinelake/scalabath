"""Small linear-algebra utilities used across :mod:`scalabath`.

The helpers in this module are intentionally thin wrappers around
``jax.numpy`` so that arrays created during host-side model construction are
already compatible with JAX transformations used by the simulation kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import jax.numpy as jnp
from jax import Array

def positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value

def nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value

def complex_dtype(dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError("operators require a complex dtype")
    return dtype

def adjoint(operator: Array) -> Array:
    """Return the Hermitian adjoint of an operator.

    Args:
        operator: Array with matrix axes in the final two dimensions.

    Returns:
        The conjugate transpose over the final two axes.
    """

    return jnp.swapaxes(jnp.conjugate(operator), -1, -2)


def ABAd(A: Array, B: Array) -> Array:
    """Compute ``A @ B @ A.conj().T``.

    The function follows JAX/NumPy broadcasting rules for leading batch axes.

    Args:
        A: Operator array with shape ``(..., dim_out, dim_in)``.
        B: Operator or density array with shape ``(..., dim_in, dim_in)``.

    Returns:
        The transformed operator with shape ``(..., dim_out, dim_out)``.
    """

    return A @ B @ adjoint(A)


def compose_rank_2(operators: Sequence[Array]) -> Array:
    """Compose operators with a Kronecker product.

    Args:
        operators: Non-empty sequence of rank-2 arrays. The first operator acts
            on the leftmost subsystem in the product Hilbert space.

    Returns:
        Kronecker product of all input operators.

    Raises:
        ValueError: If no operators are provided or an operator is not rank 2.
    """

    ops = tuple(jnp.asarray(operator) for operator in operators)
    if not ops:
        raise ValueError("compose requires at least one operator")

    result = ops[0]
    if result.ndim != 2:
        raise ValueError("compose only accepts rank-2 operators")

    for operator in ops[1:]:
        if operator.ndim != 2:
            raise ValueError("compose only accepts rank-2 operators")
        result = jnp.kron(result, operator)
    return result

def compose(operators: Sequence[Array]) -> Array:
    """Compose operators with a Kronecker product.

    Args:
        operators: Non-empty sequence of rank-2 or rank-3 arrays. The first operator acts
            on the leftmost subsystem in the product Hilbert space.

    Returns:
        Kronecker product of all input operators.

    Raises:
        ValueError: If no operators are provided or an operator is not rank 2 or rank 3.
    """

    ops = tuple(jnp.asarray(operator) for operator in operators)
    if not ops:
        raise ValueError("compose requires at least one operator")
    
    for operator in ops:
        if operator.ndim != ops[0].ndim:
            raise ValueError("compose only accepts operators with the same rank")
    if ops[0].ndim == 2:
        return compose_rank_2(ops)
    elif ops[0].ndim == 3:
        batch_size = ops[0].shape[0]
        result = []
        for batch_idx in range(batch_size):
            result.append(compose_rank_2([op[batch_idx] for op in ops]))
        result = jnp.stack(result, axis=0)
        return result
    else:
        raise ValueError("compose only accepts rank-2 or rank-3 operators")


def batch_trace(density_matrices: Array) -> Array:
    """Return traces for a batch of density matrices.

    Args:
        density_matrices: Array shaped ``(batch, hilbert_dim, hilbert_dim)``.

    Returns:
        Array shaped ``(batch,)``.
    """

    return jnp.trace(density_matrices, axis1=-2, axis2=-1)


def batch_expectation_pure(states: Array, operator: Array) -> Array:
    """Compute ``<psi|O|psi>`` for a pure-state ensemble.

    Args:
        states: Pure states shaped ``(batch, hilbert_dim)``.
        operator: Operator shaped ``(hilbert_dim, hilbert_dim)``.

    Returns:
        Expectation values shaped ``(batch,)``.
    """

    return jnp.einsum("bi,ij,bj->b", jnp.conjugate(states), operator, states)


def batch_expectation_density(density_matrices: Array, operator: Array) -> Array:
    """Compute ``Tr(rho O)`` for a density-matrix ensemble.

    Args:
        density_matrices: Density matrices shaped
            ``(batch, hilbert_dim, hilbert_dim)``.
        operator: Operator shaped ``(hilbert_dim, hilbert_dim)``.

    Returns:
        Expectation values shaped ``(batch,)``.
    """

    return jnp.einsum("bij,ji->b", density_matrices, operator)


__all__ = [
    "ABAd",
    "adjoint",
    "batch_expectation_density",
    "batch_expectation_pure",
    "batch_trace",
    "compose",
]
