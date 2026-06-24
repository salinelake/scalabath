from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scalabath.constants import Constants

RUBRENE_LATTICE_SPACING = [7.19, 14.43]  # in angstrom

mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["lines.linewidth"] = 2


def load_metadata(input_path: Path) -> dict:
    metadata_path = input_path.with_suffix(".json")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_site_populations(site_populations: np.ndarray) -> np.ndarray:
    populations = np.asarray(site_populations, dtype=float)
    if populations.ndim != 3:
        raise ValueError("site_populations must have shape (time, batch, n_sites)")

    populations = np.where((populations < 0.0) & (populations > -1e-10), 0.0, populations)
    if np.any(populations < -1e-10):
        raise ValueError("site_populations contains significantly negative entries")

    norms = populations.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("site_populations contains invalid normalization")
    return populations / norms


def lattice_shape_and_center(metadata: dict, n_sites: int) -> tuple[int, int, tuple[int, int]]:
    lattice_l1 = int(metadata["lattice_l1"])
    lattice_l2 = int(metadata["lattice_l2"])
    if lattice_l1 * lattice_l2 != n_sites:
        raise ValueError("metadata lattice size does not match site_populations")

    center_site_index = int(metadata["center_site_index"])
    center_site = divmod(center_site_index, lattice_l2)
    return lattice_l1, lattice_l2, center_site


def calculate_position_covariance(
    site_populations: np.ndarray,
    *,
    lattice_l1: int,
    lattice_l2: int,
    center_site: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    populations = normalized_site_populations(site_populations)
    populations_2d = populations.reshape(
        populations.shape[0], populations.shape[1], lattice_l1, lattice_l2
    )

    x0, y0 = center_site
    x_grid = (np.arange(lattice_l1, dtype=float) - float(x0))[:, None]
    y_grid = (np.arange(lattice_l2, dtype=float) - float(y0))[None, :]

    mean_x = np.sum(populations_2d * x_grid[None, None, :, :], axis=(-2, -1))
    mean_y = np.sum(populations_2d * y_grid[None, None, :, :], axis=(-2, -1))
    second_xx = np.sum(populations_2d * x_grid[None, None, :, :] ** 2, axis=(-2, -1))
    second_yy = np.sum(populations_2d * y_grid[None, None, :, :] ** 2, axis=(-2, -1))
    second_xy = np.sum(
        populations_2d * x_grid[None, None, :, :] * y_grid[None, None, :, :],
        axis=(-2, -1),
    )

    covariance = np.zeros((*mean_x.shape, 2, 2), dtype=float)
    covariance[..., 0, 0] = second_xx - mean_x**2
    covariance[..., 1, 1] = second_yy - mean_y**2
    covariance[..., 0, 1] = second_xy - mean_x * mean_y
    covariance[..., 1, 0] = covariance[..., 0, 1]
    mean_position = np.stack((mean_x, mean_y), axis=-1)
    return covariance, mean_position


def fit_long_time_slope(
    time_fs: np.ndarray, values: np.ndarray, fit_window_fs: float
) -> np.ndarray:
    fit_start = max(time_fs[0], time_fs[-1] - fit_window_fs)
    fit_mask = time_fs >= fit_start
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("not enough samples in the mobility fit window")

    fit_time = time_fs[fit_mask]
    fit_values = values[fit_mask]
    centered_time = fit_time - fit_time.mean()
    time_index = (slice(None),) + (None,) * (fit_values.ndim - 1)
    return np.sum(centered_time[time_index] * fit_values, axis=0) / np.sum(centered_time**2)


def mobility_tensor(
    covariance_slope: np.ndarray,
    *,
    temperature: float,
    lattice_spacing: tuple[float, float],
) -> np.ndarray:
    spacings_cm = np.asarray(lattice_spacing) * Constants.Angstrom / Constants.cm
    spacing_products = spacings_cm[:, None] * spacings_cm[None, :]
    factor = Constants.elementary_charge * Constants.eV / (2.0 * Constants.kb * temperature)
    return covariance_slope * spacing_products * factor * Constants.s / Constants.fs


def plot_population_snapshots(
    input_path: Path, output_path: Path, *, snapshot_count: int = 5
) -> None:
    from matplotlib.colors import LogNorm

    metadata = load_metadata(input_path)
    with np.load(input_path, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        site_populations = np.asarray(data["site_populations"], dtype=float)

    populations = normalized_site_populations(site_populations).mean(axis=1)
    lattice_l1, lattice_l2, center_site = lattice_shape_and_center(metadata, populations.shape[-1])
    populations_2d = populations.reshape(populations.shape[0], lattice_l1, lattice_l2)

    snapshot_indices = np.unique(
        np.linspace(0, len(time_fs) - 1, min(snapshot_count, len(time_fs)), dtype=int)
    )
    # Use logarithmic scaling; avoid log(0) by setting a small floor value
    epsilon = 1e-8
    snapshot_values = [populations_2d[idx] for idx in snapshot_indices]
    vmin = min(float(np.min(values[values > 0])) for values in snapshot_values)
    vmax = max(float(np.max(values)) for values in snapshot_values)

    # The actual physical size
    data_aspect = (
        lattice_l1 * RUBRENE_LATTICE_SPACING[0] / (lattice_l2 * RUBRENE_LATTICE_SPACING[1])
    )
    # Make the spatial snapshot figure
    fig, axes = plt.subplots(
        1,
        len(snapshot_indices),
        figsize=(3.0 * len(snapshot_indices), 3.0),
        layout="constrained",
        squeeze=False,
    )
    image = None
    for ax, time_index, values in zip(axes[0], snapshot_indices, snapshot_values, strict=True):
        disp = np.clip(values, a_min=epsilon, a_max=None)
        image = ax.imshow(
            disp.T,
            origin="lower",
            aspect=data_aspect,
            norm=LogNorm(vmin=max(epsilon, vmin), vmax=vmax),
            extent=(-0.5, lattice_l1 - 0.5, -0.5, lattice_l2 - 0.5),
            cmap="plasma",  # Use a high-contrast colormap
        )
        ax.plot(center_site[0], center_site[1], "+", color="white", markersize=8)
        ax.set_title(f"t={time_fs[time_index]:.0f} fs")
        ax.set_xlabel("lattice_l1 site")
        ax.set_ylabel("lattice_l2 site")

    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="log(population)", shrink=0.8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    # Now, make the projection figure
    # Compute projection along x and y for each snapshot
    fig_proj, axes_proj = plt.subplots(
        2, 1, figsize=(4.8, 7), layout="constrained", sharex=False
    )
    for i, (time_index, values) in enumerate(zip(snapshot_indices, snapshot_values)):
        proj_x = np.sum(values, axis=1)  # Sum along y, shape: (lattice_l1,)
        proj_y = np.sum(values, axis=0)  # Sum along x, shape: (lattice_l2,)
        axes_proj[0].plot(
            np.arange(lattice_l1), np.sqrt(proj_x), label=f"t={time_fs[time_index]:.0f} fs", linewidth=2 if i == 0 or i == len(snapshot_indices) - 1 else 1
        )
        axes_proj[1].plot(
            np.arange(lattice_l2), np.sqrt(proj_y), label=f"t={time_fs[time_index]:.0f} fs", linewidth=2 if i == 0 or i == len(snapshot_indices) - 1 else 1
        )

    axes_proj[0].set_title("Population projected along x-axis")
    axes_proj[0].set_xlabel("lattice_l1 site (x)")
    axes_proj[0].set_ylabel("sqrt(Population)")
    axes_proj[0].legend(fontsize=9)

    axes_proj[1].set_title("Population projected along y-axis")
    axes_proj[1].set_xlabel("lattice_l2 site (y)")
    axes_proj[1].set_ylabel("sqrt(Population)")
    axes_proj[1].legend(fontsize=9)

    fig_proj.savefig(output_path.with_name(f"{output_path.stem}_projections.png"), dpi=200)
    plt.close(fig_proj)


if __name__ == "__main__":
    temp_list = np.array([300])
    nrun = 4
    lattice_l1 = 100
    lattice_l2 = 20
    data_template = "data_L{l1}x{l2}_300fs/T{T}_batch1_run{run_id}.npz"
    fit_window_fs = 50.0
    components = {"xx": (0, 0), "xy": (0, 1), "yy": (1, 1)}
    mobility_results = []

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), layout="constrained")
    for T in temp_list:
        population_runs = []
        for run_id in range(nrun):
            input_path = Path(data_template.format(l1=lattice_l1, l2=lattice_l2, T=T, run_id=run_id))
            metadata = load_metadata(input_path)

            plot_population_snapshots(
                input_path,
                input_path.with_name(f"{input_path.stem}_population_snapshots.png"),
            )

            with np.load(input_path, allow_pickle=False) as data:
                run_time_fs = np.asarray(data["time_fs"], dtype=float)
                site_populations = np.asarray(data["site_populations"], dtype=float)

            time_fs = run_time_fs
            _, _, center_site = lattice_shape_and_center(metadata, site_populations.shape[-1])

            population_runs.append(normalized_site_populations(site_populations))

        population_batch = np.concatenate(population_runs, axis=1)
        print(f"Calculating position covariance based on population density data of shape {population_batch.shape} for temperature T={T} K")
        population_density = np.mean(population_batch, axis=1, keepdims=True)
        position_covariance, _ = calculate_position_covariance(
            population_density,
            lattice_l1=lattice_l1,
            lattice_l2=lattice_l2,
            center_site=center_site,
        )
        assert position_covariance.shape == (len(time_fs), 1, 2, 2)
        position_covariance = position_covariance[:,0]
        covariance_slope = fit_long_time_slope(time_fs, position_covariance, fit_window_fs)
        mobility = mobility_tensor(
            covariance_slope,
            temperature=float(T),
            lattice_spacing=RUBRENE_LATTICE_SPACING,
        )
        mobility_results.append((float(T), mobility, covariance_slope))

        d_covariance_dt = np.diff(position_covariance, axis=0) / np.diff(time_fs)[:, None, None]
        slope_time_fs = 0.5 * (time_fs[1:] + time_fs[:-1])
        for label, (j, k) in components.items():
            ax[0].plot(time_fs, position_covariance[:, j, k], label=f"T={T}K C{label}")
            ax[1].plot(slope_time_fs, d_covariance_dt[:, j, k], label=f"T={T}K dC{label}/dt")

        print(f"T={T} K mobility tensor (cm^2/V/s):")
        print(np.array2string(mobility, precision=6))

    ax[0].set_xlabel("Time (fs)")
    ax[0].set_ylabel("Covariance (site$^2$)")
    ax[1].set_xlabel("Time (fs)")
    ax[1].set_ylabel("Covariance slope (site$^2$/fs)")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    fig.savefig("covariance_tensor_comparison.png", dpi=200)
    plt.close(fig)

    with open("mobility_tensor.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["T (K)", "mu_xx (cm^2/V/s)", "mu_xy (cm^2/V/s)", "mu_yx (cm^2/V/s)", "mu_yy (cm^2/V/s)"]
        )
        for T, mobility, covariance_slope in mobility_results:
            writer.writerow(
                [T, mobility[0, 0], mobility[0, 1], mobility[1, 0], mobility[1, 1]]
            )
