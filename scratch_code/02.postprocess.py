from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "scalabath-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "scalabath-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["lines.linewidth"] = 2


def normalized_site_populations(site_populations: np.ndarray) -> np.ndarray:
    populations = np.asarray(site_populations, dtype=float)
    if populations.ndim != 3:
        raise ValueError("site_populations must have shape (time, batch, chain_length)")
    populations = np.where((populations < 0.0) & (populations > -1e-10), 0.0, populations)
    if np.any(populations < -1e-10):
        raise ValueError("site_populations contains significantly negative entries")

    norms = populations.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("site_populations contains invalid normalization")
    return populations / norms


def load_metadata(path: Path) -> dict[str, object]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scratch BCHL.py site populations.")
    parser.add_argument(
        "--input-glob",
        default="data_223322/batch8_run*.npz",
        help="glob for scratch population files",
    )
    parser.add_argument(
        "--population-key",
        default="normalized_site_populations",
        help="population array key to plot; falls back to normalized site_populations",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("../examples/BCHL_chain/reference_populations.csv"),
        help="optional CSV with reference columns: time, center, center+1, ...",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("populations-223322.png"),
        help="output figure path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(Path(".").glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"no input files matched {args.input_glob!r}")

    time_fs = None
    metadata = load_metadata(input_paths[0])
    populations_by_run = []
    for path in input_paths:
        with np.load(path, allow_pickle=False) as data:
            run_time_fs = np.asarray(data["time_fs"], dtype=float)
            if args.population_key in data:
                populations = np.asarray(data[args.population_key], dtype=float)
            else:
                populations = normalized_site_populations(np.asarray(data["site_populations"]))
        if populations.ndim == 2:
            populations = populations[:, None, :]
        if time_fs is None:
            time_fs = run_time_fs
        elif not np.array_equal(time_fs, run_time_fs):
            raise ValueError(f"{path} has a different time grid")
        populations_by_run.append(populations)

    site_populations = np.concatenate(populations_by_run, axis=1)
    if args.population_key != "normalized_site_populations":
        site_populations = normalized_site_populations(site_populations)
    populations_mean = site_populations.mean(axis=1)
    populations_sem = np.std(site_populations, axis=1, ddof=1) / np.sqrt(site_populations.shape[1])

    chain_length = populations_mean.shape[-1]
    center_site = int(metadata.get("center_site", (chain_length - 1) // 2))
    if center_site + 3 >= chain_length:
        raise ValueError("chain is too short to plot center through center + 3")

    reference = None
    if args.reference_csv.exists():
        reference = np.loadtxt(args.reference_csv, delimiter=",", skiprows=1)

    fig, ax = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
    for offset, axis in enumerate(ax.flat):
        site = center_site + offset
        axis.plot(time_fs, populations_mean[:, site], label="BCHL.py")
        axis.fill_between(
            time_fs,
            populations_mean[:, site] - populations_sem[:, site],
            populations_mean[:, site] + populations_sem[:, site],
            color="C0",
            alpha=0.25,
        )
        if reference is not None and reference.shape[1] > offset + 1:
            axis.plot(
                reference[:, 0],
                reference[:, offset + 1],
                linestyle="--",
                color="tab:orange",
                label="Ref",
            )
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Time (fs)")
        axis.set_ylabel("Population")
        axis.set_title("Center" if offset == 0 else f"Center + {offset}")
        axis.legend(frameon=False)

    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)
    print(f"Plotted {site_populations.shape[1]} trajectories from {len(input_paths)} file(s)")
    print(f"Figure written to {args.output_path}")


if __name__ == "__main__":
    main()
