from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from BCHL import run_one_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCHL.py trajectories using phases from BCHL_chain outputs."
    )
    parser.add_argument(
        "--input-glob",
        default="../examples/BCHL_chain/data_223322/batch8_run*.npz",
        help="glob for BCHL_chain .npz files whose phases should be reused",
    )
    parser.add_argument(
        "--parameters-json",
        type=Path,
        default=Path("../examples/BCHL_chain/parameters.json"),
        help="BCHL_chain parameters JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_223322"),
        help="directory, relative to scratch_code unless absolute, for scratch outputs",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="optional maximum number of input files to process",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="optional maximum number of phases per input file to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch_dir = Path(__file__).resolve().parent
    input_paths = sorted((scratch_dir / args.input_glob).parent.glob(Path(args.input_glob).name))
    if args.max_files is not None:
        input_paths = input_paths[: args.max_files]
    if not input_paths:
        raise FileNotFoundError(f"no input files matched {args.input_glob!r}")

    parameters_path = args.parameters_json
    if not parameters_path.is_absolute():
        parameters_path = scratch_dir / parameters_path
    with parameters_path.open(encoding="utf-8") as f:
        parameters = json.load(f)
    h_b = np.asarray(parameters["Hamiltonian_H"], dtype=np.complex128)
    coupling = np.asarray(parameters["Coupling_g"], dtype=np.float64)
    gamma = np.asarray(parameters["Dissipation_gamma"], dtype=np.float64)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = scratch_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        metadata_path = input_path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing metadata file {metadata_path}")
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        with np.load(input_path, allow_pickle=False) as data:
            time_fs = np.asarray(data["time_fs"], dtype=np.float64)
            phases = np.asarray(data["phases"], dtype=np.float64)

        if args.max_trajectories is not None:
            phases = phases[: args.max_trajectories]

        dt_fs = float(metadata["dt_fs"])
        sample_indices = np.rint(time_fs / dt_fs).astype(int)
        nt = int(sample_indices[-1])
        chain_length = int(metadata.get("chain_length", 19))
        boson_dims = [int(dim) for dim in metadata["boson_dims"]]

        site_populations = np.empty((len(time_fs), phases.shape[0], chain_length), dtype=np.float64)
        for phase_index, phase in enumerate(phases):
            print(
                f"{input_path.name}: trajectory {phase_index + 1}/{phases.shape[0]}",
                flush=True,
            )
            rho_s = run_one_trajectory(
                N_S=chain_length,
                V=0.0,
                Nt=nt,
                dt=dt_fs,
                h_b=h_b,
                g=coupling,
                gamma=gamma,
                boson_dim_list=boson_dims,
                dt_in_fs=True,
                phi_array=phase,
            )
            populations = np.real(np.diagonal(rho_s, axis1=0, axis2=1))
            site_populations[:, phase_index, :] = populations[sample_indices]

        norms = site_populations.sum(axis=-1, keepdims=True)
        normalized_site_populations = np.divide(
            site_populations,
            norms,
            out=np.zeros_like(site_populations),
            where=norms > 0.0,
        )
        output_path = output_dir / input_path.name
        scratch_metadata = {
            **metadata,
            "source_input": str(input_path),
            "source_parameters_json": str(parameters_path),
            "scratch_code": "BCHL.py",
            "num_saved_trajectories": int(phases.shape[0]),
            "site_populations": "raw diagonal of scratch reduced density matrix",
            "normalized_site_populations": (
                "site_populations normalized over sites at each saved time"
            ),
        }
        np.savez_compressed(
            output_path,
            time_fs=time_fs,
            site_populations=site_populations,
            normalized_site_populations=normalized_site_populations,
            phases=phases,
        )
        output_path.with_suffix(".json").write_text(
            json.dumps(scratch_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
