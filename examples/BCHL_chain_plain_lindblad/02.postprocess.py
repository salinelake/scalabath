from __future__ import annotations

import json
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


if __name__ == "__main__":
    ## get the reference data from JPCL paper
    ref_c0_path = 'reference_c0.csv'
    ref_c0 = np.loadtxt(ref_c0_path, delimiter=',', skiprows=1)
    ref_c1_path = 'reference_c1.csv'
    ref_c1 = np.loadtxt(ref_c1_path, delimiter=',', skiprows=1)
    ref_c2_path = 'reference_c2.csv'
    ref_c2 = np.loadtxt(ref_c2_path, delimiter=',', skiprows=1)
    ref_c3_path = 'reference_c3.csv'
    ref_c3 = np.loadtxt(ref_c3_path, delimiter=',', skiprows=1)
    ## load the data from the simulation
    batch_size = 1
    nrun = 128
    boson_dims = [3, 3, 3, 3, 3, 3]
    boson_dim_str = ''.join([str(dim) for dim in boson_dims])
    populations_list = []
    for runid in range(nrun):
        data_path = f"data_{boson_dim_str}/batch{batch_size}_run{runid}.npz"
        with np.load(data_path, allow_pickle=False) as data:
            time_fs = np.asarray(data["time_fs"], dtype=float)
            populations_list.append(np.asarray(data["site_populations"], dtype=float))  # (time, batch, chain_length)
    populations_list = np.concatenate(populations_list, axis=1)  # (time, batch * nrun, chain_length)

    populations_list = normalized_site_populations(populations_list)
    populations_mean = populations_list.mean(axis=1)  # (time, chain_length)
    populations_sem = np.std(populations_list, axis=1, ddof=1) / np.sqrt(populations_list.shape[1])
    chain_length = populations_mean.shape[-1]
    center_site = (chain_length - 1) // 2

    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    ax[0, 0].plot(time_fs, populations_mean[:, center_site], label='This work')
    ax[0, 0].fill_between(time_fs, populations_mean[:, center_site] - populations_sem[:, center_site], populations_mean[:, center_site] + populations_sem[:, center_site], color='C0', alpha=0.25)
    ax[0, 0].plot(ref_c0[:, 0], ref_c0[:, 1], linestyle='--', color='tab:orange', label='Ref')
    ax[0, 1].plot(time_fs, populations_mean[:, center_site + 1], label='This work')
    ax[0, 1].fill_between(time_fs, populations_mean[:, center_site + 1] - populations_sem[:, center_site + 1], populations_mean[:, center_site + 1] + populations_sem[:, center_site + 1], color='C0', alpha=0.25)
    ax[0, 1].plot(ref_c1[:, 0], ref_c1[:, 1], linestyle='--', color='tab:orange', label='Ref')
    ax[1, 0].plot(time_fs, populations_mean[:, center_site + 2], label='This work')
    ax[1, 0].fill_between(time_fs, populations_mean[:, center_site + 2] - populations_sem[:, center_site + 2], populations_mean[:, center_site + 2] + populations_sem[:, center_site + 2], color='C0', alpha=0.25)
    ax[1, 0].plot(ref_c2[:, 0], ref_c2[:, 1], linestyle='--', color='tab:orange', label='Ref')
    ax[1, 1].plot(time_fs, populations_mean[:, center_site + 3], label='This work')
    ax[1, 1].fill_between(time_fs, populations_mean[:, center_site + 3] - populations_sem[:, center_site + 3], populations_mean[:, center_site + 3] + populations_sem[:, center_site + 3], color='C0', alpha=0.25)
    ax[1, 1].plot(ref_c3[:, 0], ref_c3[:, 1], linestyle='--', color='tab:orange', label='Ref')
    for idx, _ax in enumerate(ax.flat):
        _ax.set_ylim(0, 1.05)
        _ax.legend(frameon=False)
        _ax.set_xlabel("Time (fs)")
        _ax.set_ylabel("Population")
        if idx == 0:
            _ax.set_title("Center")
        else:
            _ax.set_title(f"Center + {idx}")
    plt.tight_layout()
    fig.savefig(f'populations-{boson_dim_str}.png', dpi=200)
