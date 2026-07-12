from __future__ import annotations

import os
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from qepsilon import Constants_Metal as Constants
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linewidth'] = 2

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
    populations = normalized_site_populations(site_populations)  # shape: (time, batch, chain_length)
    chain_length = populations.shape[-1]
    if center_site < 0 or center_site >= chain_length:
        raise ValueError("center_site is outside the chain")

    displacement = np.arange(chain_length, dtype=float) - float(center_site)
    mean_position = np.sum(populations * displacement[None, None, :], axis=-1)  # shape: (time, batch)
    second_moment = np.sum(populations * displacement[None, None, :] ** 2, axis=-1)  # shape: (time, batch)
    msd = second_moment - mean_position**2 
    return msd, mean_position

def plot_msd(
    *,
    time_fs: np.ndarray,
    msd: np.ndarray,
    msd_mean: np.ndarray,
    msd_sem: np.ndarray,
    output_path: Path,
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "scalabath-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    # ax.plot(time_fs, msd, color="0.75", linewidth=0.8, alpha=0.65)
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

def plot_population(population, output_path):
    """
    population: shape: (time, chain_length)
    """
    ntime = population.shape[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(population[0], linewidth=2.0, label="t=0 fs")
    ax.plot(population[100], linewidth=2.0, label=f"t=100 fs")
    ax.plot(population[200], linewidth=2.0, label=f"t=200 fs")
    ax.plot(population[300], linewidth=2.0, label=f"t=300 fs")
    ax.plot(population[400], linewidth=2.0, label=f"t=400 fs")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1.0)

    ax.set_xlabel("Site")
    ax.set_ylabel("log(Population)")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":

    temp_list = np.array([200, 250, 300, 350, 400])
    # temp_list = np.array([250, 300, 350])
    time_fs = np.arange(0, 450, 1)
    dt = time_fs[1] - time_fs[0]

    ## plot comparison
    fig, ax = plt.subplots(2,1, figsize=(7, 8))
    msd_mean = []
    msd_sem = []
    msd_slope = []
    for idx, T in enumerate(temp_list):
        ## plot population
        population_path = f"T{T}_ns200/site_occupation_traj.npy"
        output_path = f"T{T}_ns200/population.png"
        site_populations = np.load(population_path) # shape: (time, batch, chain_length)
        plot_population(site_populations.mean(1), output_path)
        ## get MSD trajectory
        msd_path = f"T{T}_ns200/MSD_traj.npy"
        msd_traj = np.load(msd_path) # shape: (time, batch)
        _msd_slope = (msd_traj[1:] - msd_traj[:-1]) / dt
        chain_length = site_populations.shape[-1]
        center_site = chain_length // 2
        msd_mean.append(msd_traj.mean(-1))
        msd_sem.append(np.std(msd_traj, axis=1, ddof=1) / np.sqrt(msd_traj.shape[1]))
        msd_slope.append(_msd_slope.mean(-1))

    msd_mean = np.array(msd_mean)  # shape: (num_temp, time)
    msd_sem = np.array(msd_sem)  # shape: (num_temp, time)
    msd_slope = np.array(msd_slope)  # shape: (num_temp, time - 1)


    fig, ax = plt.subplots(2, len(temp_list), figsize=(5*len(temp_list), 8))
    ax_array =  ax.reshape(2, len(temp_list))
    for idx, T in enumerate(temp_list):
        ax_array[0, idx].plot(time_fs, msd_mean[idx], linewidth=2.0, label=f"T={T}K")
        ax_array[0, idx].fill_between(time_fs, msd_mean[idx] - msd_sem[idx], msd_mean[idx] + msd_sem[idx], color="C0", alpha=0.25)
        ax_array[0, idx].set_xlabel("Time (fs)")
        ax_array[0, idx].set_ylabel("MSD (site$^2$)")
        ax_array[0, idx].legend()
        ax_array[0, idx].grid(True)
        ax_array[1, idx].plot(time_fs[1:], msd_slope[idx], linewidth=2.0, label=f"T={T}K")
        ax_array[1, idx].set_xlabel("Time (fs)")
        ax_array[1, idx].set_ylabel("MSD slope (site$^2$/fs)")
        ax_array[1, idx].legend()
        ax_array[1, idx].grid(True)
    fig.savefig(f"msd_comparison.png", dpi=200)
    plt.close(fig)
 

    ## calculate mobility
    window = 150
    msd_slope_mean = msd_slope[:, -1-window: -1].mean(axis=1)  
    rubrene_R = 7.19 * Constants.Angstrom
    factor = rubrene_R ** 2/ Constants.cm**2 * Constants.eV / Constants.kb / temp_list / 2.0 * Constants.s / Constants.fs
    mobility = msd_slope_mean * factor
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(temp_list, mobility, '*-', label='This work', markersize=10)
    ax.plot(300, 40, marker='v', markersize=7, linewidth=0,color='blue', label='EXP')
    ax.legend()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mobility (cm$^2$/V/s)")
    fig.savefig(f"mobility.png", dpi=200)
    plt.close(fig)
    np.savetxt("mobility.csv", np.column_stack((temp_list, mobility)), delimiter=',', header="T (K), Mobility (cm^2/V/s)")
 