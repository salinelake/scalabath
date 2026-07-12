from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import time as get_time

import jax.numpy as jnp
import numpy as np
from helpers import build_upper_level_system_bath_hamiltonians

from scalabath.constants import Constants
from scalabath.simulations_unitary import SystemBathUnitarySimulation

DEFAULT_BOSON_DIMS = np.asarray([12, 6, 4, 3, 3, 4, 3, 3, 4], dtype=int)  ## reference boson dimensions
SYSTEM_DIM = 2
UPPER_LEVEL = 1


DTYPES = {"complex64": jnp.complex64, "complex128": jnp.complex128}
NP_DTYPES = {"complex64": np.complex64, "complex128": np.complex128}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tensor-product two-level-system bath simulation."
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="bath temperature in K")
    parser.add_argument("--dt-fs", type=float, default=0.1, help="time step in fs")
    parser.add_argument("--sample-time-fs", type=float, default=100.0, help="total time in fs")
    parser.add_argument("--sample-period-fs", type=float, default=1.0, help="save period in fs")
    parser.add_argument("--batch-size", type=int, default=128, help="number of random trajectories")
    parser.add_argument("--run-id", type=int, default=0, help="run id used in the output filename")
    parser.add_argument("--coupling-csv", type=Path, default="../coupling_const.csv", help="CSV with columns omega_cm_inverse, lambda_cm_inverse")
    parser.add_argument("--dtype", choices=tuple[str, ...](DTYPES), default="complex128", help="complex dtype for JAX arrays")
    return parser.parse_args()


def main() -> None:
    """
    setup the simulation parameters
    """
    args = parse_args()
    dtype = DTYPES[args.dtype]
    boson_dims = DEFAULT_BOSON_DIMS
    boson_dim_str = "-".join([str(dim) for dim in boson_dims])
    coupling_data = np.loadtxt(args.coupling_csv, delimiter=",", comments="#")
    omega_cm = coupling_data[:, 0]
    lambda_cm = coupling_data[:, 1]
    omega = omega_cm * Constants.cm_inverse_energy
    g_factor = np.sqrt(lambda_cm / omega_cm)  ## unitless bath-mode coupling strength
    coupling = g_factor * omega  ## the coupling strength of the bath modes in internal units.
    kbT = Constants.kb * args.temperature
    dt = args.dt_fs * Constants.fs  ## the time step in internal units.
    data_folder = f"data_dim{boson_dim_str}"
    os.makedirs(data_folder, exist_ok=True)
    output_path = f"{data_folder}/T{args.temperature:.0f}_batch{args.batch_size}_run{args.run_id}.npz"

    """
    initialize the simulation object.
    """
    simulation = SystemBathUnitarySimulation(
        system_dim=SYSTEM_DIM,
        boson_dims=boson_dims.tolist(),
        dt=dt,
        boson_freqs=omega.tolist(),
        batch_size=args.batch_size,
        dtype=dtype,
    )
    ## set the two-level-system Hamiltonian to zero.
    system_hamiltonian = jnp.zeros((SYSTEM_DIM, SYSTEM_DIM), dtype=dtype)
    simulation.set_system_hamiltonian(system_hamiltonian)

    ## set the bath harmonic Hamiltonians
    simulation.set_bath_harmonic_hamiltonians()

    ## set the system-bath Hamiltonians: only |upper><upper| couples to the bath.
    system_bath_hamiltonians = build_upper_level_system_bath_hamiltonians(
        simulation,
        coupling=coupling,
        upper_level=UPPER_LEVEL,
    )
    simulation.set_system_bath_hamiltonians(system_bath_hamiltonians)

    ## set the initial state
    phi = 0.1
    system_init_state = jnp.array([np.sqrt(phi), np.sqrt(1-phi)], dtype=dtype)
    initial_ensemble, chosen_levels = simulation.sample_thermal_bath_state(system_init_state, kbT)
    simulation.state = initial_ensemble.get_pse()

    """
    run the simulation and save the results.
    """
    nsteps = int(round(args.sample_time_fs / args.dt_fs))
    sample_freq = max(1, int(round(args.sample_period_fs / args.dt_fs)))
    time_fs = [0.0]
    rho_s_list = [simulation.reduced_system_density_matrix()]

    print(f"number of steps: {nsteps}, observing every {sample_freq} step(s)")
    steps_done = 0
    while steps_done < nsteps:
        time_start = get_time()
        steps_this_sample = min(sample_freq, nsteps - steps_done)
        simulation.step(n_steps=steps_this_sample)
        simulation.normalize()

        steps_done += steps_this_sample
        t_fs = steps_done * args.dt_fs
        print(f"t={t_fs:.3f}fs, time taken: {get_time() - time_start:.3f}s")
        print(
            f"projected wall time: "
            f"{(nsteps // sample_freq) * (get_time() - time_start) / 3600:.3f}h"
        )
        rho_s = simulation.reduced_system_density_matrix()
        sigma_x = rho_s[:, 0, 1].mean(0).real * 2.0
        sigma_y = rho_s[:, 0, 1].mean(0).imag * 2.0
        number = rho_s[:, 0, 0].mean(0).real
        print(f"sigma_x={sigma_x}, sigma_y={sigma_y}, upper occupation={number}")
        time_fs.append(t_fs)
        rho_s_list.append(rho_s)

    ## save the results and metadata
    np.savez_compressed(
        output_path,
        time_fs=np.asarray(time_fs, dtype=float),
        rho_s_list=np.asarray(rho_s_list, dtype=NP_DTYPES[args.dtype]),
        chosen_levels=chosen_levels,
    )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
