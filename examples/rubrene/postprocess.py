from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and plot Rubrene mean-square displacement from run.py output."
    )
    parser.add_argument("input", type=Path, help="Rubrene .npz output from examples/rubrene/run.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="processed .npz output path; defaults to <input>_msd.npz",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="MSD figure path; defaults to <input>_msd.png",
    )
    parser.add_argument(
        "--show-trajectories",
        action="store_true",
        help="draw individual trajectory MSD curves behind the mean",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_msd.npz")


def default_plot_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_msd.png")


def load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata" not in data:
        return {}
    raw = data["metadata"]
    return json.loads(str(raw.item() if raw.shape == () else raw))


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


def calculate_msd(
    site_populations: np.ndarray,
    *,
    center_site: int,
) -> tuple[np.ndarray, np.ndarray]:
    populations = normalized_site_populations(site_populations)
    chain_length = populations.shape[-1]
    if center_site < 0 or center_site >= chain_length:
        raise ValueError("center_site is outside the chain")

    displacement = np.arange(chain_length, dtype=float) - float(center_site)
    mean_position = np.sum(populations * displacement[None, None, :], axis=-1)
    second_moment = np.sum(populations * displacement[None, None, :] ** 2, axis=-1)
    msd = second_moment - mean_position**2
    return msd, mean_position


def summarize_msd(msd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(msd, axis=1)
    if msd.shape[1] <= 1:
        sem = np.full_like(mean, np.nan)
    else:
        sem = np.std(msd, axis=1, ddof=1) / np.sqrt(msd.shape[1])
    return mean, sem


def plot_msd(
    *,
    time_fs: np.ndarray,
    msd: np.ndarray,
    msd_mean: np.ndarray,
    msd_sem: np.ndarray,
    output_path: Path,
    show_trajectories: bool,
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "scalabath-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    if show_trajectories:
        ax.plot(time_fs, msd, color="0.75", linewidth=0.8, alpha=0.65)
    ax.plot(time_fs, msd_mean, color="C0", linewidth=2.0, label="mean MSD")

    valid_sem = np.isfinite(msd_sem)
    if np.any(valid_sem):
        ax.fill_between(
            time_fs[valid_sem],
            (msd_mean - msd_sem)[valid_sem],
            (msd_mean + msd_sem)[valid_sem],
            color="C0",
            alpha=0.25,
            label="SEM",
        )

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("MSD (site$^2$)")
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = (
        args.output.expanduser() if args.output is not None else default_output_path(input_path)
    )
    plot_path = args.plot.expanduser() if args.plot is not None else default_plot_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        site_populations = np.asarray(data["site_populations"], dtype=float)
        metadata = load_metadata(data)

    center_site = int(metadata.get("center_site", (site_populations.shape[-1] - 1) // 2))
    msd, mean_position = calculate_msd(site_populations, center_site=center_site)
    msd_mean, msd_sem = summarize_msd(msd)

    np.savez_compressed(
        output_path,
        time_fs=time_fs,
        msd=msd,
        msd_mean=msd_mean,
        msd_sem=msd_sem,
        mean_position=mean_position,
        center_site=np.asarray(center_site),
        metadata=np.asarray(json.dumps(metadata, indent=2)),
    )
    plot_msd(
        time_fs=time_fs,
        msd=msd,
        msd_mean=msd_mean,
        msd_sem=msd_sem,
        output_path=plot_path,
        show_trajectories=args.show_trajectories,
    )
    print(f"Processed MSD written to {output_path}")
    print(f"MSD plot written to {plot_path}")


if __name__ == "__main__":
    main()
