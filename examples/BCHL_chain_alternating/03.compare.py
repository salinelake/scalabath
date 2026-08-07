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
mpl.rcParams["lines.linewidth"] = 1.5

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
    fig, ax = plt.subplots(2,1, figsize=(4, 4), sharex=True, sharey=True, constrained_layout=True)
    sim_colors = ['tab:red', 'tab:gray', 'tab:green', 'tab:blue']
    ref_colors = ['red', 'gray', 'green', 'blue']

    ## get the reference data for homogeneous chain
    homo_ref_path = '../BCHL_chain/'
    ref_c0_path = homo_ref_path + 'reference_c0.csv'
    ref_c0 = np.loadtxt(ref_c0_path, delimiter=',', skiprows=1)
    ref_c1_path = homo_ref_path + 'reference_c1.csv'
    ref_c1 = np.loadtxt(ref_c1_path, delimiter=',', skiprows=1)
    ref_c2_path = homo_ref_path + 'reference_c2.csv'
    ref_c2 = np.loadtxt(ref_c2_path, delimiter=',', skiprows=1)
    ref_c3_path = homo_ref_path + 'reference_c3.csv'
    ref_c3 = np.loadtxt(ref_c3_path, delimiter=',', skiprows=1)

    ## get the reference data for inhomogeneous chain
    ref_s8_path = 'reference_s8.csv'
    ref_s8 = np.loadtxt(ref_s8_path, delimiter=',', skiprows=1)
    ref_s9_path = 'reference_s9.csv'
    ref_s9 = np.loadtxt(ref_s9_path, delimiter=',', skiprows=1)
    ref_s10_path = 'reference_s10.csv'
    ref_s10 = np.loadtxt(ref_s10_path, delimiter=',', skiprows=1)
    ref_s11_path = 'reference_s11.csv'
    ref_s11 = np.loadtxt(ref_s11_path, delimiter=',', skiprows=1)

    ## load the data from the simulation for inhomogeneous chain
    batch_size = 8
    nrun = 16
    boson_dims = [5, 5, 5, 5, 5, 5]
    boson_dim_str = ''.join([str(dim) for dim in boson_dims])
    inhomo_populations_list = []
    for runid in range(nrun):
        data_path = f"data_{boson_dim_str}/batch{batch_size}_run{runid}.npz"
        with np.load(data_path, allow_pickle=False) as data:
            time_fs = np.asarray(data["time_fs"], dtype=float)
            inhomo_populations_list.append(np.asarray(data["site_populations"], dtype=float))  # (time, batch, chain_length)
    inhomo_populations_list = np.concatenate(inhomo_populations_list, axis=1)  # (time, batch * nrun, chain_length)
    inhomo_populations_list = normalized_site_populations(inhomo_populations_list)
    inhomo_populations_mean = inhomo_populations_list.mean(axis=1)  # (time, chain_length)
    chain_length = inhomo_populations_mean.shape[-1]
    center_site = (chain_length - 1) // 2
    ## load the data from the simulation for homogeneous chain
    homo_populations_list = []
    for runid in range(nrun):
        data_path = f"../BCHL_chain/data_{boson_dim_str}/batch{batch_size}_run{runid}.npz"
        with np.load(data_path, allow_pickle=False) as data:
            time_fs = np.asarray(data["time_fs"], dtype=float)
            homo_populations_list.append(np.asarray(data["site_populations"], dtype=float))  # (time, batch, chain_length)
    homo_populations_list = np.concatenate(homo_populations_list, axis=1)  # (time, batch * nrun, chain_length)
    homo_populations_list = normalized_site_populations(homo_populations_list)
    homo_populations_mean = homo_populations_list.mean(axis=1)  # (time, chain_length)
    assert homo_populations_mean.shape[-1] == chain_length

    ## plot the inhomogeneous chain
    shift_labels = ['Site 10', 'Site 11', 'Site 9', 'Site 8' ]
    pop_indices = [center_site, center_site + 1, center_site -1, center_site -2]
    ref_pops = [ref_s10, ref_s11, ref_s9, ref_s8]
    for idx in range(len(pop_indices)):
        site_idx = pop_indices[idx]
        ax[1].plot(
            time_fs, 
            inhomo_populations_mean[:, site_idx], 
            linestyle="-", 
            linewidth=0, 
            marker='o', 
            markersize=2, 
            color=sim_colors[idx], 
            alpha=0.8, 
            label=f"SPA ({shift_labels[idx]})"
            )
        ax[1].plot(
            ref_pops[idx][:, 0], 
            ref_pops[idx][:, 1], 
            linestyle="-",
            alpha=0.5,  
            color=ref_colors[idx], 
            # label=f"MPI ({shift_labels[idx]})"
            )

    ## plot the homogeneous chain
    shift_labels = ['Site 10', 'Site 9&11', 'Site 8&12', 'Site 7&13' ]
    pop_indices = [center_site, center_site + 1, center_site + 2, center_site + 3]
    ref_pops = [ref_c0, ref_c1, ref_c2, ref_c3]

    for idx in range(len(pop_indices)):
        site_idx = pop_indices[idx]
        ax[0].plot(
            time_fs, 
            homo_populations_mean[:, site_idx], 
            linestyle="-", 
            linewidth=0, 
            marker='o', 
            markersize=2, 
            color=sim_colors[idx], 
            alpha=0.8, 
            label=f"SPA ({shift_labels[idx]})"
            )
        ax[0].plot(
            ref_pops[idx][:, 0], 
            ref_pops[idx][:, 1], 
            linestyle="-",
            alpha=0.5,  
            color=ref_colors[idx], 
            # label=f"MPI ({shift_labels[idx]})"
            )
    ax[0].set_ylabel("Population")
    ax[1].set_ylabel("Population")
    ax[1].set_xlabel(r"$t$ (fs)")
    ax[0].set_ylim(0, 0.65)
    ax[1].set_ylim(0, 0.65)
    ax[0].set_title("Homogeneous Chain")
    ax[1].set_title("Inhomogeneous Chain")
    ax[0].set_yticks([0, 0.2, 0.4, 0.6])
    ax[1].set_yticks([0, 0.2, 0.4, 0.6])
    ax[0].legend(frameon=False, loc="upper right", fontsize=9)
    ax[1].legend(frameon=False, loc="upper right", fontsize=9)
    for a in ax:
        a.tick_params(direction='in', which='both')
    fig.savefig(f'compare-populations.png', dpi=300)