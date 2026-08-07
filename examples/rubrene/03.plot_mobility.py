import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.linewidth'] = 2


if __name__ == "__main__":
    ## load our results
    our_results = np.loadtxt("mobility_SPA.csv", delimiter=',')
    sim_temp = our_results[:, 0]
    sim_mobility = our_results[:, 1]

    ## load mobility reference data
    mobility_ref = np.loadtxt("mobility_JPCL.csv", delimiter=',')
    ref_temp = mobility_ref[:, 0]
    ref_dmrg = mobility_ref[:, 1] 
    ref_fgr = mobility_ref[:, 2]
    ref_boltzmann = mobility_ref[:, 3]

    diqcd_ref = np.loadtxt("mobility_DIQCD.csv", delimiter=',')
    ref_diqcd = diqcd_ref[:, 1]
    ehrenfest_ref = np.loadtxt("mobility_Ehrenfest.csv", delimiter=',')
    ref_ehrenfest = ehrenfest_ref[:, 1]

    ## plot the results
    fig, ax = plt.subplots(figsize=(6.2, 4))
    # Plot each group and keep the line handles for two legends
    ln_fgr, = ax.plot(ref_temp, ref_fgr, 'o-', label="FGR", color='tab:blue')
    ln_ehrenfest, = ax.plot(ref_temp, ref_ehrenfest, 'o-', label="Ehrenfest", color='tab:green')
    ln_boltzmann, = ax.plot(ref_temp, ref_boltzmann, 'o-', label="Boltzmann", color='tab:cyan')
    ln_diqcd, = ax.plot(ref_temp, ref_diqcd, 'o-', label="DIQCD", color='tab:purple')
    ln_spa, = ax.plot(sim_temp, sim_mobility, '*--', label='SPA', markersize=13, color='tab:orange')
    ln_dmrg, = ax.plot(ref_temp, ref_dmrg, 'o-', label="TD-DMRG", color='black', markersize=8, linewidth=2, markerfacecolor='none')

    # Group 1: Ehrenfest, FGR, Boltzmann (legend 1, center left, a little to the right)
    group1_lines = [ln_fgr, ln_ehrenfest, ln_boltzmann]
    group1_labels = [l.get_label() for l in group1_lines]
    legend1 = ax.legend(
        group1_lines, group1_labels, frameon=False, fontsize=13, ncol=1, 
        loc='center left', bbox_to_anchor=(0.1, 0.85)
    )

    # Group 2: DIQCD, SPA, TD-DMRG (legend 2, lower left)
    group2_lines = [ln_diqcd, ln_spa, ln_dmrg]
    # group2_lines = [ln_diqcd, ln_dmrg]
    group2_labels = [l.get_label() for l in group2_lines]
    legend2 = ax.legend(group2_lines, group2_labels, frameon=False, fontsize=13, ncol=1, loc='lower left')

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_ylim(30, 100)
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\mu_\text{b}$ ($\mathrm{cm^2/(V\cdot s)}$)")

    # axins.tick_params(axis='both', which='both', labelsize=14)
    plt.tight_layout()

    fig.savefig(f"mobility_comparison.png", dpi=200)
    plt.close(fig)