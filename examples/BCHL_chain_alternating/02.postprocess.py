from __future__ import annotations

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15
mpl.rcParams["axes.labelsize"] = 15
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
    ref_s8_path = 'reference_s8.csv'
    ref_s8 = np.loadtxt(ref_s8_path, delimiter=',', skiprows=1)
    ref_s9_path = 'reference_s9.csv'
    ref_s9 = np.loadtxt(ref_s9_path, delimiter=',', skiprows=1)
    ref_s10_path = 'reference_s10.csv'
    ref_s10 = np.loadtxt(ref_s10_path, delimiter=',', skiprows=1)
    ref_s11_path = 'reference_s11.csv'
    ref_s11 = np.loadtxt(ref_s11_path, delimiter=',', skiprows=1)
    ## load the data from the simulation
    batch_size = 8
    nrun = 16
    boson_dims = [5, 5, 5, 5, 5, 5]
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
    
    titles = ["Site 10", "Site 11", "Site 9", "Site 8"]
    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    ax[0, 0].plot(time_fs, populations_mean[:, center_site], label='SPA')
    ax[0, 0].fill_between(time_fs, populations_mean[:, center_site] - populations_sem[:, center_site], populations_mean[:, center_site] + populations_sem[:, center_site], color='C0', alpha=0.25)
    ax[0, 0].plot(ref_s10[:, 0], ref_s10[:, 1], linestyle='--', color='tab:orange', label='Ref')
    ax[0, 1].plot(time_fs, populations_mean[:, center_site + 1], label='SPA')
    ax[0, 1].fill_between(time_fs, populations_mean[:, center_site + 1] - populations_sem[:, center_site + 1], populations_mean[:, center_site + 1] + populations_sem[:, center_site + 1], color='C0', alpha=0.25)
    ax[0, 1].plot(ref_s11[:, 0], ref_s11[:, 1], linestyle='--', color='tab:orange', label='Ref')
    ax[1, 0].plot(time_fs, populations_mean[:, center_site -1], label='SPA')
    ax[1, 0].fill_between(time_fs, populations_mean[:, center_site -1] - populations_sem[:, center_site -1], populations_mean[:, center_site -1] + populations_sem[:, center_site -1], color='C0', alpha=0.25)
    ax[1, 0].plot(ref_s9[:, 0], ref_s9[:, 1], linestyle='--', color='tab:orange', label='Ref')
    ax[1, 1].plot(time_fs, populations_mean[:, center_site -2], label='SPA')
    ax[1, 1].fill_between(time_fs, populations_mean[:, center_site -2] - populations_sem[:, center_site -2], populations_mean[:, center_site -2] + populations_sem[:, center_site -2], color='C0', alpha=0.25)
    ax[1, 1].plot(ref_s8[:, 0], ref_s8[:, 1], linestyle='--', color='tab:orange', label='Ref')
    for idx, _ax in enumerate(ax.flat):
        _ax.set_ylim(0, 1.05)
        _ax.legend(frameon=False)
        _ax.set_xlabel("Time (fs)")
        _ax.set_ylabel("Population")
        _ax.set_title(titles[idx])
    plt.tight_layout()
    fig.savefig(f'populations-{boson_dim_str}.png', dpi=200)


    fig, ax = plt.subplots(figsize=(6, 3))

    lines = []
    labels = []
    sim_colors = ['tab:red', 'tab:gray', 'tab:green', 'tab:blue']
    ref_colors = ['red', 'gray', 'green', 'blue']

    shift_labels = ['Site 10', 'Site 11', 'Site 9', 'Site 8']
    pop_indices = [center_site, center_site + 1, center_site -1, center_site -2]
    ref_pops = [ref_s10, ref_s11, ref_s9, ref_s8]


    # Plot "This work" population curves and range
    for i, idx in enumerate(pop_indices):
        # The main population curves
        # l = ax.plot(time_fs, populations_mean[:, idx], linestyle=":", color=sim_colors[i], linewidth=3, alpha=0.8, label=f"This work ({shift_labels[i]})")[0]
        l = ax.plot(time_fs, populations_mean[:, idx], linestyle="-", linewidth=0, marker='o', markersize=3, color=sim_colors[i], alpha=0.8, label=f"SPA ({shift_labels[i]})")[0]

        lines.append(l)
        labels.append(f"SPA ({shift_labels[i]})")

    # Plot references
    for i, (ref, label) in enumerate(zip(ref_pops, shift_labels)):
        l = ax.plot(ref[:, 0], ref[:, 1],  linestyle="-",linewidth=1,  alpha=1,   color=ref_colors[i], label=f"MPI ({label})")[0]
        lines.append(l)
        labels.append(f"MPI ({label})")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("Population")
    ax.tick_params(axis='x', which='both', pad=6)

    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11)
    plt.tight_layout()
    fig.savefig(f'all-populations-{boson_dim_str}.png', dpi=300)
