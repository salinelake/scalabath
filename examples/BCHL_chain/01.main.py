from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path
import json
import jax.numpy as jnp
import numpy as np
import jax.scipy as jsp

from scalabath.constants import Constants
from scalabath.operators_base import boson, tight_binding_1d
from scalabath.simulations import LindbladSimulation
from scalabath.utilities import compose

from helpers import save_metadata, site_populations

DEFAULT_BOSON_DIMS = np.asarray([2, 2, 3, 3, 2, 2], dtype=int)
DTYPES = {
    "complex64": jnp.complex64,
    "complex128": jnp.complex128,
}
DEFAULT_TEMPERATURE = 300.0
DEFAULT_CHAIN_LENGTH = 19
DEFAULT_HOPPING_CM = 363.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dense Lindblad BCHL tight-binding chain bath simulation.")
    parser.add_argument("--dt-fs", type=float, default=0.1, help="time step in fs")
    parser.add_argument("--sample-time-fs", type=float, default=300.0, help="total time in fs")
    parser.add_argument("--sample-period-fs", type=float, default=1.0, help="save period in fs")
    parser.add_argument("--batch-size", type=int, default=1, help="number of random phase samples")
    parser.add_argument("--run-id", type=int, default=0, help="run id used in the output filename")
    parser.add_argument("--boson-dims", default=None, help="comma-separated local boson dimensions, e.g. 2,2,3,3,2,2")
    parser.add_argument("--parameters-json", type=Path, default="parameters.json", help="JSON with info for effective bath modes")
    parser.add_argument("--damping-epsilon", type=float, default=1e-8, help="damping rate lower than this is considered to be zero")
    # parser.add_argument("--periodic", action="store_true", help="use periodic boundary conditions for the tight-binding chain")
    parser.add_argument("--dtype", choices=tuple[str, ...](DTYPES), default="complex128", help="complex dtype for JAX arrays")
    return parser.parse_args()


def main() -> None:
    """
    Setup the simulation parameters.
    """
    args = parse_args()
    dtype = DTYPES[args.dtype]
    np_dtype = np.complex128 if jnp.dtype(dtype) == jnp.complex128 else np.complex64
    chain_length = DEFAULT_CHAIN_LENGTH
    kbT = Constants.kb * DEFAULT_TEMPERATURE
    boson_dims = np.asarray(args.boson_dims, dtype=int) if args.boson_dims is not None else DEFAULT_BOSON_DIMS
    nmodes = len(boson_dims)
    with open(args.parameters_json, "r", encoding="utf-8") as f:
        parameters_json = json.load(f)
        bath_hamiltonians = np.array(parameters_json["Hamiltonian_H"], dtype=complex) * Constants.cm_inverse_energy 
        coupling = np.array(parameters_json["Coupling_g"], dtype=float) * Constants.cm_inverse_energy  ## what is this? coupling or unitless g_factor? 
        gamma = np.array(parameters_json["Dissipation_gamma"], dtype=float) * Constants.cm_inverse_energy ## the bath damping rates, diagonal.  
        if nmodes != len(gamma):
            raise ValueError("boson_dims and gamma must have the same length")
    hopping = DEFAULT_HOPPING_CM * Constants.cm_inverse_energy
    dt = args.dt_fs * Constants.fs
    bath_dim = int(np.prod(boson_dims))
    hilbert_dim = chain_length * bath_dim

    data_folder = Path(f"data_{int(args.sample_time_fs)}fs")
    data_folder.mkdir(parents=True, exist_ok=True)
    output_path = data_folder / f"batch{args.batch_size}_run{args.run_id}.npz"

    """
    Build the Hamiltonian
    """
    tb_basis = tight_binding_1d(chain_length, periodic=False, dtype=dtype)
    bath_basis = [boson(int(dim) - 1, dtype=dtype) for dim in boson_dims]

    ## first, get the pure system Hamiltonian 
    tb_sub_hamiltonian = tb_basis.nearest_neighbor_hopping(hopping)
    tb_hamiltonian = compose([tb_sub_hamiltonian] + [boson.identity for boson in bath_basis])  # (hilbert_dim, hilbert_dim)

    ## then, add the pure bath Hamiltonian
    bath_sub_hamiltonian = jnp.zeros((bath_dim, bath_dim), dtype=dtype)
    for i in range(nmodes):
        for j in range(nmodes):
            bath_operators = [boson.identity for boson in bath_basis]
            bath_operators[i] = bath_basis[i].creation
            bath_operators[j] = bath_basis[j].annihilation
            bath_sub_hamiltonian += bath_hamiltonians[i, j] * compose(bath_operators)
    bath_hamiltonian = compose([tb_basis.identity, bath_sub_hamiltonian]) # (hilbert_dim, hilbert_dim)

    ## last, add the system-bath coupling
    phases = np.random.uniform(0.0, 2.0 * np.pi, size=(args.batch_size, args.chain_length))
    couple_hamiltonian = jnp.zeros((args.batch_size, hilbert_dim, hilbert_dim), dtype=dtype)
    for site_idx in range(chain_length):
        for mode_idx in range(nmodes):
            ## get the g_j^* * exp(-1j * phi_j) * |i><i| b^\dagger_j
            creation_factor = coupling[mode_idx] * np.exp(-1j * phases[:, site_idx])  # (batch_size,)
            bath_operators = [boson.identity for boson in bath_basis]
            bath_operators[mode_idx] = bath_basis[mode_idx].creation
            creation_operator = compose([tb_basis.on_site(site_idx)] + bath_operators) # (hilbert_dim, hilbert_dim)
            couple_hamiltonian += creation_factor[:, None, None] * creation_operator[None, :, :] # (batch_size, hilbert_dim, hilbert_dim)
            ## get the g_j * exp(1j * phi_j) * |i><i| b_j
            annihilation_factor = coupling[mode_idx] * np.exp(1j * phases[:, site_idx]) # (batch_size,)
            bath_operators[mode_idx] = bath_basis[mode_idx].annihilation
            annihilation_operator = compose([tb_basis.on_site(site_idx)] + bath_operators) # (hilbert_dim, hilbert_dim)
            couple_hamiltonian += annihilation_factor[:, None, None] * annihilation_operator[None, :, :] # (batch_size, hilbert_dim, hilbert_dim)
    total_hamiltonian = tb_hamiltonian + bath_hamiltonian + couple_hamiltonian
    ## the jump operators
    jump_operators = []
    for mode_index, damping_rate in enumerate(gamma):
        if damping_rate < 0.0:
            raise ValueError("damping rate must be non-negative")
        if damping_rate > args.damping_epsilon:
            bath_operators = [boson.identity for boson in bath_basis]
            bath_operators[mode_index] = bath_basis[mode_index].annihilation
            jump_operator = compose([tb_basis.identity] + bath_operators) # (hilbert_dim, hilbert_dim)
            jump_operators.append(jump_operator)

    """
    Initialize the dense Lindblad simulation.
    """
    simulation = LindbladSimulation(
        hilbert_dim=hilbert_dim,
        batch_size=args.batch_size,
        dt=dt,
        hamiltonian=total_hamiltonian,
        jump_operators=jump_operators,
        dtype=dtype,
    )
    ## set the initial state: |center><center| \otimes exp(-H_b/k_B T)
    center_site = (args.chain_length - 1) // 2
    tb_sub_initial = tb_basis.on_site(center_site) # (chain_length, chain_length)
    bath_sub_initial = jsp.linalg.expm(-bath_sub_hamiltonian / kbT)   # (bath_dim, bath_dim)
    bath_sub_initial = bath_sub_initial / jnp.trace(bath_sub_initial) # normalize
    simulation.density_matrices = compose([tb_sub_initial, bath_sub_initial])

    """
    Run the simulation and save site populations.
    """
    nsteps = int(round(args.sample_time_fs / args.dt_fs))
    sample_freq = max(1, int(round(args.sample_period_fs / args.dt_fs)))
    time_fs = [0.0]
    populations = site_populations(simulation.density_matrices, args.batch_size, chain_length, bath_dim)
    populations_list = [np.asarray(populations)]

    print(f"hilbert_dim={hilbert_dim}, bath_dim={bath_dim}, batch_size={args.batch_size}")
    print(f"number of steps: {nsteps}, observing every {sample_freq} step(s)")
    steps_done = 0
    while steps_done < nsteps:
        steps_this_sample = min(sample_freq, nsteps - steps_done)
        simulation.step(n_steps=steps_this_sample)
        simulation.normalize()

        steps_done += steps_this_sample
        t_fs = steps_done * args.dt_fs
        populations = site_populations(simulation.density_matrices, args.batch_size, chain_length, bath_dim)
        center_population = populations[center_site].mean()
        print(f"t={t_fs:.3f}fs, center_population_mean={center_population:.6f}")
        time_fs.append(t_fs)
        populations_list.append(np.asarray(populations))

    metadata_path = output_path.with_suffix(".json")
    save_metadata(args, boson_dims, center_site, gamma / Constants.cm_inverse_energy, output_path)
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
