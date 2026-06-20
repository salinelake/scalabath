from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from helpers import save_metadata, site_populations

from scalabath.constants import Constants
from scalabath.operators_base import boson, tight_binding_1d
from scalabath.simulations_lindblad import CoupledLindbladTrajectorySimulation
from scalabath.utilities import compose

DEFAULT_BOSON_DIMS = np.asarray([4, 4, 4, 4, 4, 4], dtype=int)

DTYPES = {
    "complex64": jnp.complex64,
    "complex128": jnp.complex128,
}
DEFAULT_CHAIN_LENGTH = 19
DEFAULT_HOPPING_CM = 363.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BCHL tight-binding chain trajectory simulation.")
    parser.add_argument("--dt-fs", type=float, default=0.1, help="time step in fs")
    parser.add_argument("--sample-time-fs", type=float, default=100.0, help="total time in fs")
    parser.add_argument("--sample-period-fs", type=float, default=1.0, help="save period in fs")
    parser.add_argument("--batch-size", type=int, default=8, help="number of random phase samples")
    parser.add_argument("--run-id", type=int, default=0, help="run id used in the output filename")
    parser.add_argument("--parameters-json", type=Path, default="parameters.json", help="JSON with info for effective bath modes")
    parser.add_argument("--dtype", choices=tuple[str, ...](DTYPES), default="complex128", help="complex dtype for JAX arrays")
    return parser.parse_args()

def main() -> None:
    """
    Setup the simulation parameters.
    """
    args = parse_args()
    dtype = DTYPES[args.dtype]
    chain_length = DEFAULT_CHAIN_LENGTH
    boson_dims = DEFAULT_BOSON_DIMS
    boson_dim_str = "".join([str(dim) for dim in boson_dims])
    nmodes = len(boson_dims)
    with open(args.parameters_json, "r", encoding="utf-8") as f:
        parameters_json = json.load(f)
    bath_hamiltonians = np.array(parameters_json["Hamiltonian_H"], dtype=complex) * Constants.cm_inverse_energy
    coupling = np.array(parameters_json["Coupling_g"], dtype=float) * Constants.cm_inverse_energy
    gamma_cm = np.array(parameters_json["Dissipation_gamma"], dtype=float) ## note that this is half of the damping in standard Lindblad form.
    gamma = gamma_cm * Constants.cm_inverse_energy
    hopping = DEFAULT_HOPPING_CM * Constants.cm_inverse_energy
    dt = args.dt_fs * Constants.fs
    bath_dim = int(np.prod(boson_dims))

    rng = np.random.default_rng(args.run_id)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(args.batch_size, chain_length))


    data_folder = Path(f"data_{boson_dim_str}")
    data_folder.mkdir(parents=True, exist_ok=True)
    output_path = data_folder / f"batch{args.batch_size}_run{args.run_id}.npz"

    """
    Build the Hamiltonian.
    """
    tb_basis = tight_binding_1d(chain_length, periodic=False, dtype=dtype)
    bath_basis = [boson(int(dim) - 1, dtype=dtype) for dim in boson_dims]

    tb_sub_hamiltonian = tb_basis.nearest_neighbor_hopping(hopping)

    bath_sub_hamiltonian = jnp.zeros((bath_dim, bath_dim), dtype=dtype)
    for i in range(nmodes):
        for j in range(nmodes):
            bath_operators = [boson.identity for boson in bath_basis]
            if i == j:
                bath_operators[i] = bath_basis[i].number
            else:
                bath_operators[i] = bath_basis[i].creation
                bath_operators[j] = bath_basis[j].annihilation
            bath_sub_hamiltonian += bath_hamiltonians[i, j] * compose(bath_operators)
        bath_operators = [boson.identity for boson in bath_basis]
        bath_operators[i] = bath_basis[i].number
        bath_sub_hamiltonian += -1j * gamma[i] * compose(bath_operators)

    system_bath_hamiltonians = []
    for mode_idx in range(nmodes):
        mode_dim = int(boson_dims[mode_idx])
        local_hamiltonian = jnp.zeros(
            (args.batch_size, chain_length * mode_dim, chain_length * mode_dim),
            dtype=dtype,
        )
        for site_idx in range(chain_length):
            creation_factor = coupling[mode_idx] * np.exp(1j * phases[:, site_idx])
            creation_operator = compose([tb_basis.on_site(site_idx), bath_basis[mode_idx].creation])
            local_hamiltonian += creation_factor[:, None, None] * creation_operator[None, :, :]

            annihilation_factor = coupling[mode_idx] * np.exp(-1j * phases[:, site_idx])
            annihilation_operator = compose([tb_basis.on_site(site_idx), bath_basis[mode_idx].annihilation])
            local_hamiltonian += annihilation_factor[:, None, None] * annihilation_operator[None, :, :]
        system_bath_hamiltonians.append(local_hamiltonian)

    """
    Initialize the trajectory simulation.
    """
    simulation = CoupledLindbladTrajectorySimulation(
        system_dim=chain_length,
        boson_dims=boson_dims,
        batch_size=args.batch_size,
        dt=dt,
        system_hamiltonian=tb_sub_hamiltonian,
        bath_hamiltonian=bath_sub_hamiltonian,
        system_bath_hamiltonians=system_bath_hamiltonians,
        key=jax.random.PRNGKey(args.run_id),
        dtype=dtype,
    )
    center_site = (chain_length - 1) // 2
    initial_state = jnp.zeros((args.batch_size, chain_length, *boson_dims), dtype=dtype)
    initial_state = initial_state.at[(slice(None), center_site, *([0] * nmodes))].set(1.0)
    simulation.state = initial_state

    """
    Run the simulation and save site populations.
    """
    nsteps = int(round(args.sample_time_fs / args.dt_fs))
    sample_freq = max(1, int(round(args.sample_period_fs / args.dt_fs)))
    time_fs = [0.0]
    populations = site_populations(simulation.state)
    populations_list = [np.asarray(populations)]

    print(f"bath_dim={bath_dim}, batch_size={args.batch_size}")
    print(f"number of steps: {nsteps}, observing every {sample_freq} step(s)")
    steps_done = 0
    while steps_done < nsteps:
        steps_this_sample = min(sample_freq, nsteps - steps_done)
        simulation.step(n_steps=steps_this_sample)

        steps_done += steps_this_sample
        t_fs = steps_done * args.dt_fs
        populations = site_populations(simulation.state)
        center_population = populations[:, center_site].mean()
        print(f"t={t_fs:.3f}fs, center_population_mean={center_population:.6f}")
        time_fs.append(t_fs)
        populations_list.append(np.asarray(populations))

    metadata_path = output_path.with_suffix(".json")
    save_metadata(args, boson_dims, center_site, gamma_cm, metadata_path)
    np.savez_compressed(
        output_path,
        time_fs=np.asarray(time_fs, dtype=float),
        site_populations=np.asarray(populations_list, dtype=float),
        phases=phases,
    )
    print(f"Results written to {output_path}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
