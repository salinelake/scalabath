"""Compress the BChl vibronic environment with the RealTimeBath package. This happens before the SPA simulation.

The original 50-mode BChl intramolecular bath is compressed at T = 300 K into a six-mode coupled
Lindblad representation

Units
-----
Mode frequencies are tabulated in cm^-1.  The phase of a mode is

    omega[rad/s] * t[s] = 2 pi c[cm/s] * omega_tilde[cm^-1] * t[s],

so with time measured in fs the dimensionless phase is

    omega_tilde[cm^-1] * (2 pi c[cm/fs] * t[fs]).

We therefore fit on the rescaled time axis t_cm = t[fs] * 2*pi*c[cm/fs], which
has units of cm.  Because t_cm is conjugate to cm^-1, every fitted rate that
comes back (the entries of K and Gamma) is directly in cm^-1, and eps is in
cm^-1 as well since c(t) carries units of energy^2.  The 100 fs benchmark
window is 0.018837 cm on this axis.
"""

import json

import numpy as np
import realtimebath as rtb
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
# --------------------------------------------------------------------------
# Physical constants and unit conversion
# --------------------------------------------------------------------------
KB_CM = 0.6950348004          # Boltzmann constant in cm^-1 / K
C_CM_PER_FS = 2.99792458e-5   # speed of light in cm / fs
FS_TO_CM = 2.0 * np.pi * C_CM_PER_FS  # t[fs] -> t[cm], conjugate to cm^-1

TEMPERATURE = 300.0           # K
T_MAX_FS = 100.0              # fitting window of the paper
N_MODES = 6                   # target compressed bath size
N_SAMPLES = 1001              # uniform samples on [0, T_MAX_FS]

# --------------------------------------------------------------------------
# Original bath: Table II / bcf/bcf.md
# --------------------------------------------------------------------------
FREQUENCY_CM = np.array([
      84,  167,  183,  191,  214,  239,  256,  345,  368,  388,
     407,  423,  442,  473,  506,  565,  587,  623,  684,  696,
     710,  727,  776,  803,  845,  858,  890,  915,  967,  980,
    1001, 1019, 1066, 1089, 1105, 1117, 1137, 1158, 1180, 1190,
    1211, 1229, 1252, 1289, 1378, 1466, 1519, 1539, 1648, 1680,
], dtype=float)

HUANG_RHYS = np.array([
    0.0151, 0.0081, 0.0072, 0.0196, 0.0046, 0.0078, 0.0055, 0.0161, 0.0060, 0.0041,
    0.0052, 0.0029, 0.0023, 0.0017, 0.0016, 0.0081, 0.0039, 0.0058, 0.0023, 0.0025,
    0.0018, 0.0266, 0.0095, 0.0042, 0.0025, 0.0025, 0.0284, 0.0048, 0.0027, 0.0031,
    0.0040, 0.0097, 0.0025, 0.0021, 0.0021, 0.0103, 0.0042, 0.0103, 0.0025, 0.0031,
    0.0025, 0.0021, 0.0025, 0.0087, 0.0086, 0.0021, 0.0017, 0.0023, 0.0025, 0.0027,
])


def thermal_bcf(t_cm, temperature=TEMPERATURE):
    """Original 50-mode thermal BCF, evaluated on a time axis in cm.

    c(t) = sum_m S_m omega_m^2 [coth(beta omega_m / 2) cos(omega_m t)
                                - i sin(omega_m t)]
    """
    t_cm = np.asarray(t_cm, dtype=float)
    w = FREQUENCY_CM
    coth = 1.0 / np.tanh(w / (2.0 * KB_CM * temperature))
    phase = np.multiply.outer(t_cm, w)
    real = (HUANG_RHYS * w**2 * coth * np.cos(phase)).sum(axis=-1)
    imag = (HUANG_RHYS * w**2 * np.sin(phase)).sum(axis=-1)
    return real - 1j * imag


# --------------------------------------------------------------------------
# Gauge fixing: rotate to the basis in which the damping matrix is diagonal
# --------------------------------------------------------------------------
def to_diagonal_damping(model):
    """Return (K, gamma, eps) in the paper's gauge, Gamma = diag(gamma).

    c_fit(t) is invariant under eps -> U^dagger eps, K -> U^dagger K U,
    D -> U^dagger D U for any unitary U.  Choosing U from the eigenvectors of
    D makes the damping diagonal, which is the form quoted in the paper.
    """
    gamma, U = np.linalg.eigh(model.damping)
    order = np.argsort(gamma)[::-1]          # largest damping first
    gamma, U = gamma[order].real, U[:, order]
    # residual gauge: each mode carries a free phase; fix it so eps is real >= 0.
    # With V = U @ diag(conj(p)) the transform is eps -> diag(p) U^dagger eps.
    p = np.exp(-1j * np.angle(U.conj().T @ model.coupling))
    V = U @ np.diag(p.conj())
    K = V.conj().T @ model.hamiltonian @ V
    eps = V.conj().T @ model.coupling
    gamma_check = V.conj().T @ model.damping @ V
    assert np.allclose(gamma_check, np.diag(gamma), atol=1e-8 * max(1.0, gamma.max()))
    assert np.allclose(eps.imag, 0.0, atol=1e-10 * np.linalg.norm(eps))
    return (K + K.conj().T) / 2.0, np.maximum(gamma, 0.0), eps.real


def _complex_str(value):
    """Format one complex entry the way ``parameters.json`` stores ``Hamiltonian_H``."""
    return np.array2string(np.complex128(value))


def main():
    lam = float((HUANG_RHYS * FREQUENCY_CM).sum())
    t_fs = np.linspace(0.0, T_MAX_FS, N_SAMPLES)
    t_cm = t_fs * FS_TO_CM
    c_exact = thermal_bcf(t_cm)

    fit = rtb.fit_correlation(
        t_cm,
        c_exact,
        n_modes=N_MODES,
        method="sdp",
        optimize=True,                 # physical time-domain refinement
        optimization_max_nfev=5000,
        optimization_max_points=N_SAMPLES,
    )

    K, gamma, eps = to_diagonal_damping(fit.model)
    c_fit = fit.evaluate(t_cm)

    # the gauge rotation must be exact: eps^dag exp[(-iK - diag(gamma))t] eps == c_fit
    rotated = rtb.LindbladModel(K, np.diag(gamma.astype(complex)), eps.astype(complex))
    gauge_residual = float(np.abs(rotated.evaluate(t_cm) - c_fit).max())
    assert gauge_residual < 1e-6 * np.abs(c_fit).max(), gauge_residual

    err = c_fit - c_exact
    rel_rms = float(np.sqrt(np.mean(np.abs(err) ** 2))
                    / np.sqrt(np.mean(np.abs(c_exact) ** 2)))

    ### Logging
    log = {
        "units": {
            "Hamiltonian_H": "cm^-1",
            "Coupling_g": "cm^-1",
            "Dissipation_gamma": "cm^-1",
            "eigenvalues_H": "cm^-1",
        },
        "original_bath": {
            "modes": int(FREQUENCY_CM.size),
            "temperature [K]": float(TEMPERATURE),
            "kT [cm^-1]": float(KB_CM * TEMPERATURE),
            "reorganization_energy [cm^-1]": lam,
            "reorganization_energy_paper [cm^-1]": 217.66,
            "fit_window [fs]": [0.0, float(T_MAX_FS)],
            "fit_window [cm]": [0.0, float(T_MAX_FS * FS_TO_CM)],
            "samples": N_SAMPLES,
            "c(0) [cm^-2]": float(c_exact[0].real),
        },
        "n_modes": N_MODES,
        "method": str(fit.diagnostics.method),
        "relative_rms_error": rel_rms,
        "max_absolute_error [cm^-2]": float(np.abs(err).max()),
        "min_damping_eigenvalue [cm^-1]": float(fit.diagnostics.min_damping_eigenvalue),
        "stability_abscissa [cm^-1]": float(fit.diagnostics.stability_abscissa),
        "warnings": [str(w) for w in (fit.diagnostics.warnings or [])],
        "Hamiltonian_H": [[_complex_str(v) for v in row] for row in K],
        "Coupling_g": [float(v) for v in eps],
        "Dissipation_gamma": [float(v) for v in gamma],
        "eigenvalues_H": [float(v) for v in np.linalg.eigvalsh(K)],
    }
    log_path = "parameters.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_path}")

    ### Plot the comparison between the original and the fitted correlation functions

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t_fs, c_exact.real * 1e-5, "k-", lw=2, label=r"Re $c(t)$")
    ax.plot(t_fs, c_exact.imag * 1e-5, "-", color="0.55", lw=2, label=r"Im $c(t)$")
    ax.plot(t_fs, c_fit.real * 1e-5, "C3--", lw=1.7, label=r"Re $c_{\text{fit}}(t)$")
    ax.plot(t_fs, c_fit.imag * 1e-5, "C0--", lw=1.7, label=r"Im $c_{\text{fit}}(t)$")
    ax.set_xlabel("$t$ [fs]", fontsize=12)
    ax.set_ylabel(r"BCF [$\mathrm{10^{5} cm^{-2}}$]", fontsize=10)
    ax.set_xlim(0, T_MAX_FS)
    ax.legend(ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig("bath_compression_comparison.png", dpi=300)

if __name__ == "__main__":
    main()
