from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import time as get_time

import jax.numpy as jnp
import numpy as np
from helpers import *

from scalabath.constants import Constants
from scalabath.operators_base import tight_binding_2d
from scalabath.simulations_unitary import SystemBathUnitarySimulation

# DEFAULT_BOSON_DIMS = np.asarray([12, 6, 4, 3, 3, 4, 3, 3, 4], dtype=int)
DEFAULT_BOSON_DIMS = np.asarray([12, 6, 4, 3, 2, 4, 2, 3, 4], dtype=int)

DTYPES = { "complex64": jnp.complex64, "complex128": jnp.complex128 }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tensor-product Rubrene tight-binding lattice bath simulation.")
    parser.add_argument("--temperature", type=float, default=300.0, help="bath temperature in K")
    parser.add_argument("--lattice-l1", type=int, default=128, help="number of tight-binding sites along easy direction")
    parser.add_argument("--lattice-l2", type=int, default=36, help="number of tight-binding sites along hard direction")
    parser.add_argument("--hopping-l1-mev", type=float, default=83.0, help="nearest-neighbor hopping amplitude along easy direction in meV")
    parser.add_argument("--hopping-l2-mev", type=float, default=14.1, help="nearest-neighbor hopping amplitude along hard direction in meV")
    parser.add_argument("--dt-fs", type=float, default=0.1, help="time step in fs")
    parser.add_argument("--sample-time-fs", type=float, default=100.0, help="total time in fs")
    parser.add_argument("--sample-period-fs", type=float, default=1.0, help="save period in fs")
    parser.add_argument("--batch-size", type=int, default=1, help="number of random trajectories")
    parser.add_argument("--run-id", type=int, default=0, help="run id used in the output filename")
    parser.add_argument("--num-modes", type=int, default=9, help="number of bath modes to include")
    parser.add_argument("--boson-dims", default=None, help="comma-separated local boson dimensions, e.g. 9,4,2,2,2,2,2,2,2")
    parser.add_argument("--coupling-csv", type=Path, default="coupling_const.csv", help="CSV with columns omega_cm_inverse, lambda_cm_inverse")
    parser.add_argument("--periodic", type=bool, default=True, help="use periodic boundary conditions for the tight-binding lattice")
    parser.add_argument("--dtype", choices=tuple[str, ...](DTYPES), default="complex64", help="complex dtype for JAX arrays")
    return parser.parse_args()


def main() -> None:
    """
    setup the simulation parameters
    """
    args = parse_args()
    dtype = DTYPES[args.dtype]
    boson_dims = args.boson_dims if args.boson_dims is not None else DEFAULT_BOSON_DIMS
    boson_dims = np.asarray(boson_dims, dtype=int)[:args.num_modes]
    system_dim = args.lattice_l1 * args.lattice_l2
    coupling_data = np.loadtxt(args.coupling_csv, delimiter=',', comments='#')
    omega_cm = coupling_data[:args.num_modes, 0]
    lambda_cm = coupling_data[:args.num_modes, 1]
    omega = omega_cm * Constants.cm_inverse_energy
    g_factor = np.sqrt(lambda_cm / omega_cm) ## the unitless coupling strength of the bath modes. shape: (num_modes,)
    coupling = g_factor * omega  ## the coupling strength of the bath modes in internal units. shape: (num_modes,)
    kbT = Constants.kb * args.temperature 
    hopping_l1 = args.hopping_l1_mev * Constants.meV  ## the hopping amplitude in internal units.
    hopping_l2 = args.hopping_l2_mev * Constants.meV  ## the hopping amplitude in internal units.
    dt = args.dt_fs * Constants.fs  ## the time step in internal units.
    data_folder = f"data_L{args.lattice_l1}x{args.lattice_l2}_{int(args.sample_time_fs)}fs"
    os.makedirs(data_folder, exist_ok=True)
    output_path = f"{data_folder}/T{args.temperature:.0f}_batch{args.batch_size}_run{args.run_id}.npz"
    
    """
    initialize the simulation object.
    """
    simulation = SystemBathUnitarySimulation(
        system_dim=system_dim,
        boson_dims=boson_dims.tolist(),
        dt=dt,
        boson_freqs=omega.tolist(),
        batch_size=args.batch_size,
        dtype=dtype,
    )
    ## set the state sharding for multi-GPU parallelization
    state_sharding = state_sharding_from_gpus(len(boson_dims))
    if state_sharding is not None:
        simulation._pse.sharding = state_sharding

    ## set the system Hamiltonian, triangular hopping is used here because rubrene crystal  has the herringbone structure.
    tb_lattice = tight_binding_2d(args.lattice_l1, args.lattice_l2, periodic=args.periodic, dtype=dtype)
    system_hamiltonian = tb_lattice.triangular_hopping((hopping_l1, hopping_l2))  # 2D array shape: (system_dim, system_dim)
    simulation.set_system_hamiltonian(system_hamiltonian)

    ## set the bath harmonic Hamiltonians
    simulation.set_bath_harmonic_hamiltonians()
    
    ## set the system-bath Hamiltonians
    phases = np.random.uniform(0.0, 2.0 * np.pi, size=(args.batch_size, system_dim))
    system_bath_hamiltonians = build_system_bath_hamiltonians(simulation, coupling=coupling, phases=phases)  ## a num_modes-list of (batch_size, system_dim, boson_dim, boson_dim) arrays
    simulation.set_system_bath_hamiltonians(system_bath_hamiltonians)
    
    ## set the initial state
    center_site = [args.lattice_l1 // 2, args.lattice_l2 // 2]
    center_site_index = tb_lattice.site_index(center_site)
    system_init_state = jnp.zeros(system_dim, dtype=dtype).at[center_site_index].set(1.0)
    initial_ensemble, chosen_levels = simulation.sample_thermal_bath_state(system_init_state, kbT)
    simulation.state = initial_ensemble.get_pse()

    """
    run the simulation and save the results.
    """
    nsteps = int(round(args.sample_time_fs / args.dt_fs))
    sample_freq = max(1, int(round(args.sample_period_fs / args.dt_fs)))
    time_fs = [0.0]
    site_populations = [site_populations_from_state(simulation.state)]

    print(f"number of steps: {nsteps}, observing every {sample_freq} step(s)")
    steps_done = 0
    while steps_done < nsteps:
        time_start = get_time()
        steps_this_sample = min(sample_freq, nsteps - steps_done)
        simulation.step(n_steps=steps_this_sample)
        simulation.normalize()

        steps_done += steps_this_sample
        t_fs = steps_done * args.dt_fs
        populations = site_populations_from_state(simulation.state)
        center_population = populations[:, center_site_index].mean()
        print(f"t={t_fs:.3f}fs, center_population_mean={center_population:.6f}")
        print(f"time taken: {get_time() - time_start:.3f}s")
        print(f"projected wall time: {(nsteps//sample_freq) * (get_time() - time_start) / 3600:.3f}h")
        time_fs.append(t_fs)
        site_populations.append(populations)
    
    ## save the results and metadata
    metadata_path = Path(output_path).with_suffix(".json")
    save_metadata(args, boson_dims, center_site_index, lambda_cm, omega_cm, metadata_path)
    np.savez_compressed(
        output_path,
        time_fs=np.asarray(time_fs, dtype=float),
        site_populations=np.asarray(site_populations, dtype=float),
        chosen_levels=chosen_levels,
        phases=phases,
    )
    print(f"Results written to {output_path}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
