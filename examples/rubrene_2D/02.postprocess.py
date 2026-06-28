from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from helpers import normalized_site_populations
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["lines.linewidth"] = 2



RUBRENE_LATTICE_SPACING = [7.19, 14.43]  # in angstrom
temp_list = np.array([200, 250, 300, 350, 400])
nrun = 32
lattice_l1 = 128
lattice_l2 = 36
data_template = "data_L{l1}x{l2}_350fs/T{T}_batch1_run{run_id}.npz"
components = {"xx": (0, 0), "xy": (0, 1), "yy": (1, 1)}


def triangular_lattice_coordinates(
    lattice_l1: int,
    lattice_l2: int,
) -> tuple[np.ndarray, np.ndarray]:
    spacing_l1, spacing_l2 = np.asarray(RUBRENE_LATTICE_SPACING, dtype=float)
    l1_indices, l2_indices = np.meshgrid(
        np.arange(lattice_l1, dtype=float),
        np.arange(lattice_l2, dtype=float),
        indexing="ij",
    )
    x_grid = (l1_indices - 0.5 * l2_indices) * spacing_l1
    y_grid = 0.5 * l2_indices * spacing_l2
    return x_grid, y_grid


def triangular_lattice_triangulation(x_grid: np.ndarray, y_grid: np.ndarray) -> Triangulation:
    site_indices = np.arange(x_grid.size).reshape(x_grid.shape)
    triangles = []
    for i in range(x_grid.shape[0] - 1):
        for j in range(x_grid.shape[1] - 1):
            lower_left = site_indices[i, j]
            lower_right = site_indices[i + 1, j]
            upper_left = site_indices[i, j + 1]
            upper_right = site_indices[i + 1, j + 1]
            triangles.append((lower_left, lower_right, upper_left))
            triangles.append((lower_right, upper_right, upper_left))

    return Triangulation(x_grid.ravel(), y_grid.ravel(), np.asarray(triangles, dtype=int))


for T in temp_list:
    population_runs = []
    for run_id in range(nrun):
        input_path = Path(data_template.format(l1=lattice_l1, l2=lattice_l2, T=T, run_id=run_id))
        with np.load(input_path, allow_pickle=False) as data:
            time_fs = np.asarray(data["time_fs"], dtype=float)
            site_populations = np.asarray(data["site_populations"], dtype=float)
        population_runs.append(normalized_site_populations(site_populations))
    population_runs = np.concatenate(population_runs, axis=1)
    population_density = np.mean(population_runs, axis=1)
    populations_2d = population_density.reshape(population_density.shape[0], lattice_l1, lattice_l2)

    # Use logarithmic scaling; avoid log(0) by setting a small floor value
    epsilon = 1e-8
    snapshot_count = 5
    snapshot_indices = np.unique(
        np.linspace(0, len(time_fs) - 1, min(snapshot_count, len(time_fs)), dtype=int)
    )
    snapshot_values = [populations_2d[idx] for idx in snapshot_indices]
    vmin = min(float(np.min(values[values > 0])) for values in snapshot_values)
    vmax = max(float(np.max(values)) for values in snapshot_values)

    x_grid, y_grid = triangular_lattice_coordinates(lattice_l1, lattice_l2)
    triangulation = triangular_lattice_triangulation(x_grid, y_grid)
    x_margin = 0.5 * RUBRENE_LATTICE_SPACING[0]
    y_margin = 0.5 * RUBRENE_LATTICE_SPACING[1]

    fig, axes = plt.subplots(
        len(snapshot_indices),
        1,
        figsize=(7, 2.6 * len(snapshot_indices)),
        layout="constrained",
        squeeze=False,
    )
    image = None
    for ax, time_index, values in zip(
        axes.flatten(), snapshot_indices, snapshot_values, strict=True
    ):
        disp = np.clip(values, a_min=epsilon, a_max=None)
        image = ax.tripcolor(
            triangulation,
            disp.ravel(),
            shading="gouraud",
            norm=LogNorm(vmin=max(epsilon, vmin), vmax=vmax),
            cmap="plasma",
        )
        ax.set_xlim(float(np.min(x_grid) - x_margin), float(np.max(x_grid) + x_margin))
        ax.set_ylim(float(np.min(y_grid) - y_margin), float(np.max(y_grid) + y_margin))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(r"Log($\rho$) at t="+f"{time_fs[time_index]:.0f} fs", fontsize=18)
        ax.set_xlabel("x (Angstrom)")
        ax.set_ylabel("y (Angstrom)")

    if image is not None:
        # Let the colorbar span all axes from axes[0] to axes[-1]
        fig.colorbar(
            image,
            ax=axes.flatten().tolist(),
            label="log(population)",
            shrink=0.6,
        )


    fig.savefig(f"T{T}L{lattice_l1}x{lattice_l2}_snapshots.png", dpi=200)
    plt.close(fig)

    # Now, make the 1D figure
    # Compute population along x and y (passing through the center site) for each snapshot
    center_site = (lattice_l1 // 2, lattice_l2 // 2)
    fig_proj, axes_proj = plt.subplots(
        2, 1, figsize=(4.8, 7), layout="constrained", sharex=False
    )
    for i, (time_index, values) in enumerate(zip(snapshot_indices, snapshot_values, strict=True)):
        proj_x = values[:, center_site[1]]  # x-axis population, shape: (lattice_l1,)
        proj_y = values[center_site[0], :]  # y-axis population, shape: (lattice_l2,)
        linewidth = 2 if i == 0 or i == len(snapshot_indices) - 1 else 1
        label = f"t={time_fs[time_index]:.0f} fs"
        axes_proj[0].plot(
            np.arange(lattice_l1),
            np.sqrt(proj_x),
            label=label,
            linewidth=linewidth,
        )
        axes_proj[1].plot(
            np.arange(lattice_l2),
            np.sqrt(proj_y),
            label=label,
            linewidth=linewidth,
        )

    axes_proj[0].set_title("Population projected along x-axis")
    axes_proj[0].set_xlabel("lattice_l1 site (x)")
    axes_proj[0].set_ylabel("sqrt(Population)")
    axes_proj[0].legend(fontsize=9)

    axes_proj[1].set_title("Population projected along y-axis")
    axes_proj[1].set_xlabel("lattice_l2 site (y)")
    axes_proj[1].set_ylabel("sqrt(Population)")
    axes_proj[1].legend(fontsize=9)

    fig_proj.savefig(f"T{T}L{lattice_l1}x{lattice_l2}_projections.png", dpi=200)
    plt.close(fig_proj)
