from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scalabath.constants import Constants

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

def plot_population(input_path, output_path):
    with np.load(input_path, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        site_populations = np.asarray(data["site_populations"], dtype=float)
    site_populations = site_populations.mean(axis=-2) # shape: (time, chain_length)
    ntime = site_populations.shape[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(site_populations[0]**0.5, linewidth=2.0, label="t=0 fs")
    ax.plot(site_populations[ntime//4]**0.5, linewidth=2.0, label=f"t={ntime//4} fs")
    ax.plot(site_populations[ntime//2]**0.5, linewidth=2.0, label=f"t={ntime//2} fs")
    ax.plot(site_populations[3*ntime//4]**0.5, linewidth=2.0, label=f"t={3*ntime//4} fs")
    ax.plot(site_populations[ntime-1]**0.5, linewidth=2.0, label=f"t={ntime-1} fs")

    ax.set_xlabel("Site")
    ax.set_ylabel("Population")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":

    ## simulation parameters
    # temp_list = np.array([200, 250, 300, 350, 400])
    temp_list = np.array([200, 300, 400])
    L = 150
    nrun = 32
    batch = 5
    slope_list = np.zeros(len(temp_list))
    data_folder = f"data_L{L}_300fs"

    ## plot comparison
    fig, ax = plt.subplots(2,1, figsize=(7, 8))
    for idx, T in enumerate(temp_list):
        msd_batch = []
        for run_id in range(nrun):
            input_path = f'{data_folder}/T{T}_batch{batch}_run{run_id}.npz'
            ## load metadata
            metadata_path = input_path.replace(".npz", ".json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            center_site = int(metadata.get("center_site", None))
            ## plot population
            population_path = input_path.replace(".npz", "_population.png")
            plot_population(input_path, population_path)
            ## load site populations
            with np.load(input_path, allow_pickle=False) as data:
                time_fs = np.asarray(data["time_fs"], dtype=float)
                site_populations = np.asarray(data["site_populations"], dtype=float)
            ## calculate MSD
            msd, mean_position = calculate_msd(site_populations, center_site=center_site)
            msd_batch.append(msd) # shape: (time, batch)
        msd_batch = np.concatenate(msd_batch, axis=1) # shape: (time, batch * 4)
        msd_mean = np.mean(msd_batch, axis=1)
        msd_slope = (msd_mean[1:] - msd_mean[:-1]) / (time_fs[1:] - time_fs[:-1])
        slope_list[idx] = (msd_mean[-1] - msd_mean[-51]) / (time_fs[-1] - time_fs[-51])
        msd_sem = np.std(msd_batch, axis=1, ddof=1) / np.sqrt(msd_batch.shape[1])
        ax[0].plot(time_fs, msd_mean, linewidth=2.0, label=f"T={T}K")
        ax[0].fill_between(time_fs, msd_mean - msd_sem, msd_mean + msd_sem, color="C0", alpha=0.25)
        ax[1].plot(time_fs[1:], msd_slope, linewidth=2.0, label=f"T={T}K")
    ax[0].set_xlabel("Time (fs)")
    ax[0].set_ylabel("MSD (site$^2$)")
    ax[0].legend()
    ax[1].set_xlabel("Time (fs)")
    ax[1].set_ylabel("MSD slope (site$^2$/fs)")
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    fig.savefig("L150_msd.png", dpi=200)
    plt.close(fig)

    ## load mobility reference data
    mobility_ref = np.loadtxt("mobility.csv", delimiter=',')
    ref_temp = mobility_ref[:, 0]
    ref_dmrg = (mobility_ref[:, 1] + mobility_ref[:, 2])/2
    ref_fgr = (mobility_ref[:, 3] + mobility_ref[:, 4])/2

    ## calculate mobility
    rubrene_R = 7.19 * Constants.Angstrom
    factor = rubrene_R ** 2/ Constants.cm**2 * Constants.eV / Constants.kb / temp_list / 2.0 * Constants.s / Constants.fs
    mobility = slope_list * factor
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temp_list, mobility, linewidth=2.0, label='This work', marker='o')
    ax.plot(ref_temp, ref_dmrg, linewidth=2.0, label="DMRG")
    ax.plot(ref_temp, ref_fgr, linewidth=2.0, label="FGR")
    ax.legend()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mobility (cm$^2$/V/s)")
    fig.savefig("L150_mobility.png", dpi=200)
    plt.close(fig)
