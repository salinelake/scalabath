from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scalabath.constants import Constants
from scalabath.operators_base import boson, tight_binding_1d
from scalabath.simulations import LindbladSimulation

from helpers import save_metadata

DEFAULT_BOSON_DIMS = np.asarray([2, 2, 3, 3, 2, 2], dtype=int)
DTYPES = {
    "complex64": jnp.complex64,
    "complex128": jnp.complex128,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dense Lindblad BCHL tight-binding chain bath simulation.")
    parser.add_argument("--temperature", type=float, default=300.0, help="bath temperature in K")
    parser.add_argument("--chain-length", type=int, default=19, help="number of tight-binding sites")
    parser.add_argument("--hopping-mev", type=float, default=363.0, help="nearest-neighbor hopping amplitude in cm^-1")
    parser.add_argument("--dt-fs", type=float, default=0.1, help="time step in fs")
    parser.add_argument("--sample-time-fs", type=float, default=300.0, help="total time in fs")
    parser.add_argument("--sample-period-fs", type=float, default=1.0, help="save period in fs")
    parser.add_argument("--batch-size", type=int, default=1, help="number of random phase samples")
    parser.add_argument("--run-id", type=int, default=0, help="run id used in the output filename")
    parser.add_argument("--seed", type=int, default=0, help="random seed for site phases")
    parser.add_argument("--num-modes", type=int, default=6, help="number of bath modes to include")
    parser.add_argument("--boson-dims", default=None, help="comma-separated local boson dimensions, e.g. 2,2,3,3,2,2")
    parser.add_argument("--coupling-csv", type=Path, default="coupling_const.csv", help="CSV with columns omega_cm_inverse, huang_rhys")
    parser.add_argument("--gamma-cm-inverse", default="10.0", help="scalar or comma-separated bath damping rates in cm^-1")
    parser.add_argument("--periodic", action="store_true", help="use periodic boundary conditions for the tight-binding chain")
    parser.add_argument("--dtype", choices=tuple[str, ...](DTYPES), default="complex128", help="complex dtype for JAX arrays")
    return parser.parse_args()


def main() -> None:
    """
    Setup the simulation parameters.
    """
    args = parse_args()
    dtype = DTYPES[args.dtype]
    np_dtype = np.complex128 if jnp.dtype(dtype) == jnp.complex128 else np.complex64
    boson_dims = args.boson_dims if args.boson_dims is not None else DEFAULT_BOSON_DIMS
    boson_dims = np.asarray(boson_dims, dtype=int)[:args.num_modes]
    coupling_data = np.loadtxt(args.coupling_csv, delimiter=',', comments='#')
    omega_cm = coupling_data[: args.num_modes, 0]
    huang_rhys = coupling_data[: args.num_modes, 1]
    omega = omega_cm * Constants.cm_inverse_energy
    coupling = np.sqrt(huang_rhys) * omega
    kbT = Constants.kb * args.temperature
    hopping = args.hopping_mev * Constants.meV
    dt = args.dt_fs * Constants.fs

    gamma_cm = np.fromstring(args.gamma_cm_inverse, sep=",", dtype=float)
    if gamma_cm.size == 0:
        raise ValueError("gamma-cm-inverse must contain at least one value")
    if gamma_cm.size == 1:
        gamma_cm = np.full(args.num_modes, gamma_cm[0], dtype=float)
    else:
        gamma_cm = gamma_cm[: args.num_modes]
    if gamma_cm.size != args.num_modes:
        raise ValueError("gamma-cm-inverse must be scalar or contain num-modes entries")
    if np.any(gamma_cm < 0.0):
        raise ValueError("gamma-cm-inverse values must be non-negative")
    gamma = gamma_cm * Constants.cm_inverse_energy

    bath_dim = int(np.prod(boson_dims))
    hilbert_dim = args.chain_length * bath_dim

    data_folder = Path(f"data_L{args.chain_length}_{int(args.sample_time_fs)}fs")
    data_folder.mkdir(parents=True, exist_ok=True)
    output_path = data_folder / f"batch{args.batch_size}_run{args.run_id}.npz"

    """
    Build the Hamiltonian of the system and the bath.
    """
    system_identity = np.eye(args.chain_length, dtype=np_dtype)
    bath_identity = np.eye(bath_dim, dtype=np_dtype)

    bath_annihilations = []
    for mode_index, _mode_dim in enumerate(boson_dims):
        factors = []
        for local_index, local_dim in enumerate(boson_dims):
            if local_index == mode_index:
                local_operator = boson(int(local_dim) - 1, dtype=dtype).annihilation
                factors.append(np.asarray(local_operator, dtype=np_dtype))
            else:
                factors.append(np.eye(int(local_dim), dtype=np_dtype))
        bath_annihilations.append(reduce(np.kron, factors))

    bath_hamiltonian = np.zeros((bath_dim, bath_dim), dtype=np_dtype)
    for mode_index, annihilation in enumerate(bath_annihilations):
        creation = annihilation.conj().T
        bath_hamiltonian += omega[mode_index] * (creation @ annihilation)

    tb_chain = tight_binding_1d(args.chain_length, periodic=args.periodic, dtype=dtype)
    system_hamiltonian = np.asarray(tb_chain.nearest_neighbor_hopping(hopping), dtype=np_dtype)
    base_hamiltonian = np.kron(system_hamiltonian, bath_identity)
    base_hamiltonian += np.kron(system_identity, bath_hamiltonian)

    rng = np.random.default_rng(args.seed + args.run_id)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(args.batch_size, args.chain_length))
    hamiltonian = np.empty((args.batch_size, hilbert_dim, hilbert_dim), dtype=np_dtype)
    for batch_index in range(args.batch_size):
        matrix = base_hamiltonian.copy()
        for site in range(args.chain_length):
            bath_block = np.zeros((bath_dim, bath_dim), dtype=np_dtype)
            phase = phases[batch_index, site]
            for mode_index, annihilation in enumerate(bath_annihilations):
                creation = annihilation.conj().T
                bath_block += coupling[mode_index] * (
                    np.exp(-1j * phase) * annihilation + np.exp(1j * phase) * creation
                )
            start = site * bath_dim
            stop = start + bath_dim
            matrix[start:stop, start:stop] += bath_block
        hamiltonian[batch_index] = matrix

    jump_operators = []
    for mode_index, annihilation in enumerate(bath_annihilations):
        if gamma[mode_index] > 0.0:
            jump_operator = np.sqrt(2.0 * gamma[mode_index]) * np.kron(
                system_identity,
                annihilation,
            )
            jump_operators.append(jnp.asarray(jump_operator, dtype=dtype))

    """
    Initialize the dense Lindblad simulation.
    """
    simulation = LindbladSimulation(
        hilbert_dim=hilbert_dim,
        dt=dt,
        hamiltonian=jnp.asarray(hamiltonian, dtype=dtype),
        jump_operators=jump_operators,
        batch_size=args.batch_size,
        dtype=dtype,
    )

    center_site = (args.chain_length - 1) // 2
    initial_state = np.zeros(hilbert_dim, dtype=np_dtype)
    initial_state[center_site * bath_dim] = 1.0
    simulation.density_matrices = np.outer(initial_state, initial_state.conj())

    """
    Run the simulation and save site populations.
    """
    nsteps = int(round(args.sample_time_fs / args.dt_fs))
    sample_freq = max(1, int(round(args.sample_period_fs / args.dt_fs)))
    time_fs = [0.0]
    diagonal = jnp.diagonal(simulation.density_matrices, axis1=-2, axis2=-1).real
    populations = diagonal.reshape(args.batch_size, args.chain_length, bath_dim).sum(axis=-1)
    site_populations = [np.asarray(populations)]

    print(f"hilbert_dim={hilbert_dim}, bath_dim={bath_dim}, batch_size={args.batch_size}")
    print(f"number of steps: {nsteps}, observing every {sample_freq} step(s)")
    steps_done = 0
    while steps_done < nsteps:
        steps_this_sample = min(sample_freq, nsteps - steps_done)
        simulation.step(n_steps=steps_this_sample)
        simulation.normalize()

        steps_done += steps_this_sample
        t_fs = steps_done * args.dt_fs
        diagonal = jnp.diagonal(simulation.density_matrices, axis1=-2, axis2=-1).real
        populations = diagonal.reshape(args.batch_size, args.chain_length, bath_dim).sum(axis=-1)
        center_population = np.asarray(populations[:, center_site]).mean()
        trace = np.asarray(jnp.trace(simulation.density_matrices, axis1=-2, axis2=-1).real).mean()
        print(
            f"t={t_fs:.3f}fs, center_population_mean={center_population:.6f}, "
            f"trace_mean={trace:.6f}"
        )
        time_fs.append(t_fs)
        site_populations.append(np.asarray(populations))

    metadata_path = output_path.with_suffix(".json")
    save_metadata(args, boson_dims, center_site, gamma_cm, omega_cm, huang_rhys, output_path)
    np.savez_compressed(
        output_path,
        time_fs=np.asarray(time_fs, dtype=float),
        site_populations=np.asarray(site_populations, dtype=float),
        phases=phases,
    )
    print(f"Results written to {output_path}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
