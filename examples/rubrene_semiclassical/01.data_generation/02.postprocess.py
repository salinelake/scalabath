from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linewidth'] = 2

from scalabath.constants import Constants

if __name__ == "__main__":
    ## simulation parameters
    boson_dims = np.asarray([12, 6, 4, 3, 3, 4, 3, 3, 4], dtype=int)
    boson_dim_str = "-".join([str(dim) for dim in boson_dims])
    data_folder = f"data_dim{boson_dim_str}"

    temp_list = np.array([ 200, 250, 300, 350, 400 ])
    batch_size = 128
    run_id = 0

    ## plot comparison
    sigma_x_list = []
    sigma_y_list = []
    sigma_x_std_list = []
    sigma_y_std_list = []
    fig, ax = plt.subplots(2,1, figsize=(7, 8))
    for idx, T in enumerate(temp_list):
        input_path = f'{data_folder}/T{T}_batch{batch_size}_run{run_id}.npz'
        ## load site populations
        with np.load(input_path, allow_pickle=False) as data:
            time_fs = np.asarray(data["time_fs"], dtype=float)
            rho_s_list = np.asarray(data["rho_s_list"], dtype=complex)  # shape: (time, batch, 2, 2)
        sigma_x = rho_s_list[:, :, 0, 1].real.mean(1) * 2.0
        sigma_y = rho_s_list[:, :, 0, 1].imag.mean(1) * 2.0
        sigma_x_std = rho_s_list[:,:,0,1].real.std(1) * 2.0
        sigma_y_std = rho_s_list[:,:,0,1].imag.std(1) * 2.0
        sigma_x_list.append(sigma_x)
        sigma_y_list.append(sigma_y)
        sigma_x_std_list.append(sigma_x_std)
        sigma_y_std_list.append(sigma_y_std)
    sigma_x_list = np.array(sigma_x_list)
    sigma_y_list = np.array(sigma_y_list)
    sigma_x_std_list = np.array(sigma_x_std_list)
    sigma_y_std_list = np.array(sigma_y_std_list)
    np.save(os.path.join(data_folder, "sigma_x_list.npy"), sigma_x_list)
    np.save(os.path.join(data_folder, "sigma_y_list.npy"), sigma_y_list)
    np.save(os.path.join(data_folder, "sigma_x_std_list.npy"), sigma_x_std_list)
    np.save(os.path.join(data_folder, "sigma_y_std_list.npy"), sigma_y_std_list)
    fig, ax = plt.subplots(2, len(temp_list), figsize=(4*len(temp_list), 6), layout="constrained")
    for i, T in enumerate(temp_list):
        ax[0,i].plot(time_fs, sigma_x_list[i], linewidth=2.0, label=f"T={T}K")
        ax[0,i].fill_between(time_fs, sigma_x_list[i] - sigma_x_std_list[i], sigma_x_list[i] + sigma_x_std_list[i], color="C0", alpha=0.25)
        ax[1,i].plot(time_fs, sigma_y_list[i], linewidth=2.0, label=f"T={T}K")
        ax[1,i].fill_between(time_fs, sigma_y_list[i] - sigma_y_std_list[i], sigma_y_list[i] + sigma_y_std_list[i], color="C0", alpha=0.25)
        ax[0,i].set_xlabel("t (fs)")
        ax[0,i].set_ylabel(r"$\langle\sigma_x(t)\rangle$")
        ax[1,i].set_xlabel("t (fs)")
        ax[1,i].set_ylabel(r"$\langle\sigma_y(t)\rangle$") 
        ax[0,i].set_title(f"T={T}K")
        ax[0,i].set_ylim(-0.3, 0.65)
    fig.savefig("sigma_comparison.png", dpi=200)
    plt.close(fig)