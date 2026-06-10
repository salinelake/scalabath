from __future__ import annotations

import argparse
import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scalabath.constants import Constants
from scalabath.operators_base import boson, tls
from scalabath.simulations import SystemBathUnitarySimulation
from scalabath.systems import TensorProductPureStatesEnsemble
from scalabath.utilities import compose

DEFAULT_NMAX = np.asarray([8, 3, 1, 1, 1, 1, 1, 1, 1], dtype=int)
DTYPES = {
    "complex64": jnp.complex64,
    "complex128": jnp.complex128,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tensor-product spin-boson simulation with local system-bath updates."
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="temperature in K")
    parser.add_argument("--dt", type=float, default=0.01, help="time step in fs")
    parser.add_argument("--epsilon", type=float, default=0.1, help="initial spin population")
    parser.add_argument("--sample_time", type=float, default=100.0, help="simulation time in fs")
    parser.add_argument("--sample_period", type=float, default=1.0, help="observation period in fs")
    parser.add_argument("--batchsize", type=int, default=1, help="batch size")
    parser.add_argument("--run_id", type=int, default=0, help="run id used in output filenames")
    parser.add_argument("--seed", type=int, default=0, help="random seed for thermal sampling")
    parser.add_argument("--num_modes", type=int, default=9, help="number of bath modes to include")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="complex64",
        help="complex dtype for JAX arrays",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="disable per-step state normalization in the Trotter solver",
    )
    return parser.parse_args()


def tensor_product_state(factors: list[np.ndarray]) -> np.ndarray:
    product = factors[0]
    for factor in factors[1:]:
        product = product[..., None] * factor.reshape((1,) * product.ndim + (factor.size,))
    return product


def sample_initial_ensemble(
    *,
    batch_size: int,
    epsilon: float,
    num_modes: int,
    nmax: np.ndarray,
    omega: np.ndarray,
    kbT: float,
    seed: int,
    dtype: jnp.dtype,
) -> tuple[TensorProductPureStatesEnsemble, list[list[int]]]:
    rng = np.random.default_rng(seed)
    numpy_dtype = np.complex128 if dtype == jnp.complex128 else np.complex64
    spin_state = np.zeros(2, dtype=numpy_dtype)
    spin_state[0] = np.sqrt(1.0 - epsilon)
    spin_state[1] = np.sqrt(epsilon)

    states = []
    chosen_levels_by_batch = []
    for _ in range(batch_size):
        factors = [spin_state]
        chosen_levels = []
        for mode_index in range(num_modes):
            nstate = int(nmax[mode_index]) + 1
            energy_levels = np.arange(nstate) * omega[mode_index]
            probabilities = np.exp(-energy_levels / kbT)
            probabilities = probabilities / probabilities.sum()
            chosen_level = int(rng.choice(nstate, p=probabilities))

            mode_state = np.zeros(nstate, dtype=numpy_dtype)
            mode_state[chosen_level] = 1.0
            factors.append(mode_state)
            chosen_levels.append(chosen_level)

        states.append(tensor_product_state(factors))
        chosen_levels_by_batch.append(chosen_levels)

    subsystem_dims = (2, *tuple((nmax[:num_modes] + 1).tolist()))
    ensemble = TensorProductPureStatesEnsemble(subsystem_dims, batch_size=batch_size, dtype=dtype)
    ensemble.set_pse(jnp.asarray(np.stack(states, axis=0)))
    ensemble.normalize()
    return ensemble, chosen_levels_by_batch


def build_spin_boson_terms(
    *,
    nmax: np.ndarray,
    omega: np.ndarray,
    spin_boson_coupling: np.ndarray,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    spin = tls(dtype=dtype)
    system_hamiltonian = jnp.zeros((2, 2), dtype=dtype)
    bath_hamiltonians = []
    system_bath_hamiltonians = []

    for mode_nmax, mode_omega, mode_coupling in zip(nmax, omega, spin_boson_coupling, strict=True):
        local_boson = boson(int(mode_nmax), dtype=dtype)
        bath_hamiltonians.append(jnp.asarray(mode_omega, dtype=dtype) * local_boson.number)
        displacement = local_boson.creation + local_boson.annihilation
        system_bath_hamiltonians.append(
            jnp.asarray(mode_coupling, dtype=dtype) * compose([spin.number, displacement])
        )

    return (
        system_hamiltonian,
        bath_hamiltonians,
        system_bath_hamiltonians,
        spin.number,
        spin.sigma_x,
    )


def main() -> None:
    args = parse_args()
    debug = True

    dtype = DTYPES[args.dtype]
    data = np.loadtxt('coupling_const.csv', delimiter=',')
    omega_in_cm = data[: args.num_modes, 0]
    lambda_in_cm = data[: args.num_modes, 1]
    g_factor = np.sqrt(lambda_in_cm / omega_in_cm)

    batch_size = args.batchsize
    dt = args.dt * Constants.fs
    temperature = args.temperature
    nmax = DEFAULT_NMAX[: args.num_modes]
    omega = omega_in_cm * Constants.cm_inverse_energy
    spin_boson_coupling = g_factor * omega
    kbT = Constants.kb * temperature

    total_num_states = 2 * int(np.prod(nmax + 1))
    print(f"nmodes={args.num_modes}, total_num_states={total_num_states}")
    print("g factor:", g_factor)
    print("omega [cm^-1]:", omega_in_cm, "in internal units:", omega)
    print(
        "polaron binding energy [cm^-1]:",
        g_factor**2 * omega_in_cm,
        "sum in meV:",
        (g_factor**2 * omega_in_cm).sum() * Constants.cm_inverse_energy / Constants.meV,
    )

    if debug:
        assert batch_size == 1, "batch size must be 1 for debug mode"
        if os.path.exists('../spin_boson/system_init_test.npy'):
            initial_ensemble = np.load('../spin_boson/system_init_test.npy')
            chosen_levels = np.load('../spin_boson/chosen_levels_test.npy')
    else:
        initial_ensemble, chosen_levels = sample_initial_ensemble(
            batch_size=batch_size,   
            epsilon=args.epsilon,
            num_modes=args.num_modes,
            nmax=nmax,
            omega=omega,
            kbT=kbT,
            seed=args.seed,
            dtype=dtype,
        )
    for batch_index, levels in enumerate(chosen_levels):
        print(f"batch {batch_index} chosen levels:", levels)

    (
        system_hamiltonian,
        bath_hamiltonians,
        system_bath_hamiltonians,
        spin_number,
        spin_x,
    ) = build_spin_boson_terms(
        nmax=nmax,
        omega=omega,
        spin_boson_coupling=spin_boson_coupling,
        dtype=dtype,
    )
    simulation = SystemBathUnitarySimulation(
        2,
        tuple((nmax + 1).tolist()),
        dt,
        batch_size=batch_size,
        system_hamiltonian=system_hamiltonian,
        bath_hamiltonians=bath_hamiltonians,
        system_bath_hamiltonians=system_bath_hamiltonians,
        normalize=not args.no_normalize,
        dtype=dtype,
    )
    simulation.state = initial_ensemble if debug else initial_ensemble.get_pse()

    nsteps = int(round(args.sample_time / args.dt))
    sample_freq = max(1, int(round(args.sample_period / args.dt)))

    t_list = []
    sigma_n_list = []
    sigma_x_list = []
    print(
        "number of steps:",
        nsteps,
        f"observing every {sample_freq} step(s)",
        f"with seed {args.seed}",
    )
    steps_done = 0
    while steps_done < nsteps:
        steps_this_sample = min(sample_freq, nsteps - steps_done)
        simulation.step(n_steps=steps_this_sample)
        steps_done += steps_this_sample

        sigma_n = np.asarray(simulation.observe_system(spin_number).real)
        sigma_x = np.asarray(simulation.observe_system(spin_x).real)
        t_fs = steps_done * args.dt
        print(f"t={t_fs:.3f}fs, sigma_n={sigma_n}, sigma_x={sigma_x}")
        t_list.append(t_fs)
        sigma_n_list.append(sigma_n)
        sigma_x_list.append(sigma_x)

    sigma_n_array = np.asarray(sigma_n_list)  # (nsteps, batchsize)
    sigma_x_array = np.asarray(sigma_x_list)  # (nsteps, batchsize)
    t_array = np.asarray(t_list)
    np.save(f"t_run{args.run_id}.npy", t_array)
    np.save(f"sigma_n_run{args.run_id}.npy", sigma_n_array)
    np.save(f"sigma_x_run{args.run_id}.npy", sigma_x_array)

    sigma_n_mean = np.mean(sigma_n_array, axis=1)  # (nsteps,)
    sigma_x_mean = np.mean(sigma_x_array, axis=1)  # (nsteps,)
    sigma_x_std = np.std(sigma_x_array, axis=1)  # (nsteps,)
    sigma_n_std = np.std(sigma_n_array, axis=1)  # (nsteps,)

    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].plot(t_array, sigma_n_mean)
    ax[0].fill_between(t_array, sigma_n_mean - sigma_n_std, sigma_n_mean + sigma_n_std, alpha=0.5)
    ax[0].set_xlabel("Time (fs)")
    ax[0].set_ylabel("Sigma_n")
    ax[1].plot(t_array, sigma_x_mean)
    ax[1].fill_between(t_array, sigma_x_mean - sigma_x_std, sigma_x_mean + sigma_x_std, alpha=0.5)
    ax[1].set_xlabel("Time (fs)")
    ax[1].set_ylabel("Sigma_x")
    plt.savefig(f"spin_boson_test_run{args.run_id}.png")
if __name__ == "__main__":
    main()
