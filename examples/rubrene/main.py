from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
from rubrene_helpers import (
    DTYPES,
    build_bath_hamiltonians,
    build_system_bath_hamiltonians,
    load_coupling_data,
    parse_boson_dims,
    sample_initial_ensemble,
    site_populations_from_state,
    tight_binding_hamiltonian,
)

from scalabath.constants import Constants
from scalabath.simulations import SystemBathUnitarySimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tensor-product Rubrene tight-binding chain bath simulation.")
    parser.add_argument("--temperature", type=float, default=300.0, help="bath temperature in K")
    parser.add_argument("--chain-length", type=int, default=31, help="number of tight-binding sites")
    parser.add_argument("--hopping-mev", type=float, default=83.0, help="nearest-neighbor hopping amplitude in meV")
    parser.add_argument("--dt-fs", type=float, default=0.1, help="time step in fs")
    parser.add_argument("--sample-time-fs", type=float, default=100.0, help="total time in fs")
    parser.add_argument("--sample-period-fs", type=float, default=1.0, help="save period in fs")
    parser.add_argument("--batch-size", type=int, default=4, help="number of random trajectories")
    parser.add_argument("--seed", type=int, default=0, help="random seed for phases and bath sampling")
    parser.add_argument("--run-id", type=int, default=0, help="run id used in the output filename")
    parser.add_argument("--num-modes", type=int, default=9, help="number of bath modes to include")
    parser.add_argument("--boson-dims", default=None, help="comma-separated local boson dimensions, e.g. 9,4,2,2,2,2,2,2,2")
    parser.add_argument("--coupling-csv", type=Path, default="coupling_const.csv", help="CSV with columns omega_cm_inverse, lambda_cm_inverse")
    parser.add_argument("--periodic", action="store_true", help="use periodic boundary conditions for the tight-binding chain")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="complex64", help="complex dtype for JAX arrays")
    parser.add_argument("--no-normalize", action="store_true", help="disable per-step state normalization in the Trotter solver")
    return parser.parse_args()

def main() -> None:
    """
    setup the simulation parameters
    """
    args = parse_args()
    dtype = DTYPES[args.dtype]
    boson_dims = parse_boson_dims(args.boson_dims, args.num_modes)  ## (num_modes,)
    omega_cm, lambda_cm = load_coupling_data(args.coupling_csv, args.num_modes)  ## the frequency and the lambda value of the bath modes in cm^-1. shape: (num_modes,)
    g_factor = np.sqrt(lambda_cm / omega_cm)  ## the unitless coupling strength of the bath modes. shape: (num_modes,)
    omega = omega_cm * Constants.cm_inverse_energy
    coupling = g_factor * omega  ## the coupling strength of the bath modes in internal units. shape: (num_modes,)
    kbT = Constants.kb * args.temperature 
    hopping = args.hopping_mev * Constants.meV  ## the hopping amplitude in internal units.
    dt = args.dt_fs * Constants.fs  ## the time step in internal units.
    os.makedirs("data", exist_ok=True)
    output_path = f"data/T{args.temperature:.0f}_L{args.chain_length}_modes{args.num_modes}_batch{args.batch_size}_run{args.run_id}.npz"
    
    """
    sample the initial product state of the ensemble: |ψ_S⟩ ⊗ |ψ_b^1⟩ ⊗ ... ⊗ |ψ_b^n⟩. |ψ_S⟩ is a single carrier at the central site. |ψ_b^i⟩ is |k⟩ with k drawn from Boltzmann distribution of the bath mode i.
    """
    initial_ensemble, chosen_levels = sample_initial_ensemble(
        batch_size=args.batch_size,
        chain_length=args.chain_length,
        boson_dims=boson_dims,
        omega=omega,
        kbT=kbT,
        seed=args.seed,
        dtype=dtype,
    )
    
    """
    sample the random phases for the system-bath coupling. 
    """
    rng = np.random.default_rng(args.seed + 1)
    phases = rng.uniform(
        0.0, 2.0 * np.pi, size=(args.batch_size, args.num_modes, args.chain_length)
    )
    
    """
    initialize the simulation object.
    """
    simulation = SystemBathUnitarySimulation(system_dim=args.chain_length, boson_dims=boson_dims.tolist(), dt=dt, batch_size=args.batch_size, normalize=not args.no_normalize, dtype=dtype)
    simulation.set_system_hamiltonian(
        tight_binding_hamiltonian(args.chain_length, hopping, periodic=args.periodic, dtype=dtype)
        )  ## the system Hamiltonian. shape: (chain_length, chain_length) or (batch_size, chain_length, chain_length)
    simulation.set_bath_hamiltonians(
        build_bath_hamiltonians(boson_dims=boson_dims, omega=omega, dtype=dtype)
        )  ## the bath Hamiltonians. shape: (num_modes, boson_dim, boson_dim) or (batch_size, num_modes, boson_dim, boson_dim)
    simulation.set_system_bath_hamiltonians(build_system_bath_hamiltonians(
        batch_size=args.batch_size, chain_length=args.chain_length, boson_dims=boson_dims, coupling=coupling, phases=phases, dtype=dtype)
        )  ## the system-bath Hamiltonians. shape: (num_modes, chain_length * boson_dim, chain_length * boson_dim)
    simulation.state = initial_ensemble.get_pse()

    """
    run the simulation and save the results.
    """
    nsteps = int(round(args.sample_time_fs / args.dt_fs))
    sample_freq = max(1, int(round(args.sample_period_fs / args.dt_fs)))
    center_site = (args.chain_length - 1) // 2
    time_fs = [0.0]
    site_populations = [site_populations_from_state(simulation.state)]

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

        t_fs = steps_done * args.dt_fs
        populations = site_populations_from_state(simulation.state)
        center_population = populations[:, center_site].mean()
        print(f"t={t_fs:.3f}fs, center_population_mean={center_population:.6f}")
        time_fs.append(t_fs)
        site_populations.append(populations)

    metadata = {
        "batch_size": args.batch_size,
        "boson_dims": boson_dims.tolist(),
        "center_site": center_site,
        "chain_length": args.chain_length,
        "coupling_csv": str(args.coupling_csv),
        "dtype": args.dtype,
        "dt_fs": args.dt_fs,
        "g_factor": g_factor.tolist(),
        "hopping_internal": float(hopping),
        "hopping_mev": args.hopping_mev,
        "lambda_cm_inverse": lambda_cm.tolist(),
        "normalize": not args.no_normalize,
        "num_modes": args.num_modes,
        "omega_cm_inverse": omega_cm.tolist(),
        "omega_internal": omega.tolist(),
        "periodic": args.periodic,
        "run_id": args.run_id,
        "sample_period_fs": args.sample_period_fs,
        "sample_time_fs": args.sample_time_fs,
        "seed": args.seed,
        "temperature_K": args.temperature,
    }
    np.savez_compressed(
        output_path,
        time_fs=np.asarray(time_fs, dtype=float),
        site_populations=np.asarray(site_populations, dtype=float),
        chosen_levels=chosen_levels,
        phases=phases,
        metadata=np.asarray(json.dumps(metadata, indent=2)),
    )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
