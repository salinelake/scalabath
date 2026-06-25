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


def triangular_lattice_coordinates(
    lattice_l1: int,
    lattice_l2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Cartesian site coordinates in Angstrom for the triangular lattice."""

    spacing_l1, spacing_l2 = np.asarray(RUBRENE_LATTICE_SPACING, dtype=float)
    l1_indices, l2_indices = np.meshgrid(
        np.arange(lattice_l1, dtype=float),
        np.arange(lattice_l2, dtype=float),
        indexing="ij",
    )
    x_grid = (l1_indices - 0.5 * l2_indices) * spacing_l1
    y_grid = 0.5 * l2_indices * spacing_l2
    return x_grid, y_grid


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

    x_grid, y_grid = triangular_lattice_coordinates(lattice_l1, lattice_l2)
    center_x = x_grid[center_site]
    center_y = y_grid[center_site]
    x_grid = x_grid - center_x
    y_grid = y_grid - center_y

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
) -> np.ndarray:
    """Return the mobility tensor in cm^2/(V s).

    ``covariance_slope`` is the fitted slope of the Cartesian position
    covariance in Angstrom^2/fs. The Einstein relation is
    ``mu = q D / (k_B T)`` with ``D = 0.5 * dC/dt``.
    """

    thermal_energy_eV = Constants.kb * temperature / Constants.eV
    diffusion_cm2_per_s = (
        0.5
        * covariance_slope
        * (Constants.Angstrom / Constants.cm) ** 2
        * (Constants.s / Constants.fs)
    )
    return Constants.elementary_charge * diffusion_cm2_per_s / thermal_energy_eV


if __name__ == "__main__":
    temp_list = np.array([300])
    nrun = 16
    lattice_l1 = 120
    lattice_l2 = 40
    data_template = "data_L{l1}x{l2}_350fs/T{T}_batch1_run{run_id}.npz"
    fit_window_fs = 50.0
    components = {"xx": (0, 0), "xy": (0, 1), "yy": (1, 1)}
    mobility_results = []

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), layout="constrained")
    for T in temp_list:
        population_runs = []
        for run_id in range(nrun):
            input_path = Path(
                data_template.format(l1=lattice_l1, l2=lattice_l2, T=T, run_id=run_id)
            )
            metadata = load_metadata(input_path)

            with np.load(input_path, allow_pickle=False) as data:
                run_time_fs = np.asarray(data["time_fs"], dtype=float)
                site_populations = np.asarray(data["site_populations"], dtype=float)

            time_fs = run_time_fs
            _, _, center_site = lattice_shape_and_center(metadata, site_populations.shape[-1])

            population_runs.append(normalized_site_populations(site_populations))

        population_batch = np.concatenate(population_runs, axis=1)
        population_density = np.mean(population_batch, axis=1, keepdims=True)
        assert population_density.shape == (len(time_fs), 1, lattice_l1 * lattice_l2)
        position_covariance, _ = calculate_position_covariance(
            population_density,
            lattice_l1=lattice_l1,
            lattice_l2=lattice_l2,
            center_site=center_site,
        )
        assert position_covariance.shape == (len(time_fs), 1, 2, 2)
        position_covariance = position_covariance[:, 0]
        covariance_slope = fit_long_time_slope(time_fs, position_covariance, fit_window_fs)
        mobility = mobility_tensor(
            covariance_slope,
            temperature=float(T),
        )
        mobility_results.append((float(T), mobility, covariance_slope))

        d_covariance_dt = np.diff(position_covariance, axis=0) / np.diff(time_fs)[:, None, None]
        slope_time_fs = 0.5 * (time_fs[1:] + time_fs[:-1])
        for label, (j, k) in components.items():
            ax[0].plot(time_fs, np.sqrt(position_covariance[:, j, k]), label=f"T={T}K sqrt(C{label}) (Angstrom)")
            ax[1].plot(slope_time_fs, d_covariance_dt[:, j, k], label=f"T={T}K dC{label}/dt (Angstrom/fs)")

        print(f"T={T} K mobility tensor (cm^2/V/s):")
        print(np.array2string(mobility, precision=6))

    ax[0].set_xlabel("Time (fs)")
    ax[0].set_ylabel("Covariance (Angstrom$^2$)")
    ax[1].set_xlabel("Time (fs)")
    ax[1].set_ylabel("Covariance slope (Angstrom$^2$/fs)")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    fig.savefig("covariance_tensor_comparison.png", dpi=200)
    plt.close(fig)

    with open("mobility_tensor.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "T (K)",
                "mu_xx (cm^2/V/s)",
                "mu_xy (cm^2/V/s)",
                "mu_yx (cm^2/V/s)",
                "mu_yy (cm^2/V/s)",
            ]
        )
        for T, mobility, _covariance_slope in mobility_results:
            writer.writerow(
                [T, mobility[0, 0], mobility[0, 1], mobility[1, 0], mobility[1, 1]]
            )

"""
The mobility tensor we get here is 
[[ 47.01371 -0.04441342]
 [-0.04441342  1.389214]] (cm^2/V/s). 

One can estimate the mobility along the hard axis using Marcus theory.

First, we consider the rate constant for electron transfer along the easy axis.
Using Marcus theory, the rate constant is:
  k(J) = (2 pi / hbar) |J|^2 / sqrt(4 pi lambda kBT)
         * exp[-lambda / (4 kBT)]
For the easy axis, J = 83meV and we get mu_xx ~ 42.8 cm^2/V/s

Then, along the hard axis, we consider the triangular hopping between main chains, and the ratio between lattice constants, we get 
mu_yy ~= mu_xx * 2 * (b/2a)^2 * (J2/J1)^2
    ~= 47 * 2 * (7.215/7.19)^2 * (14.1/83)^2
    ~= 2.72 cm^2/V/s

This is consistent with the mobility tensor we get here.
"""