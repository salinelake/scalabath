import numpy as np


class Constants:
    r"""
    Class whose members are fundamental constants.

    Inner Unit:

    - Length: :math:`\text{pm}`

    - Time: :math:`\text{ps}`

    - Energy: :math:`\hbar \times \text{THz}`
    """

    ## energy units in hbar*THz
    hbar = 1.0  # hbar = hbar_THz * ps = 1.0
    hbar_THz = 1.0
    hbar_MHz = 1e-6 * hbar_THz
    hbar_Hz = 1e-6 * hbar_MHz

    eV = hbar_Hz / 6.582119569e-16
    meV = 1e-3 * eV
    Ry = 13.605693123 * eV
    mRy = 1e-3 * Ry
    Hartree = 27.211386245981 * eV
    Joule = 6.241509e18 * eV
    amu_cc = 931.49410372e6 * eV

    ## physical constants
    kb = 8.6173303e-5 * eV  # hbar * THz / K
    speed_of_light = 299792458  # pm/ps
    amu = amu_cc / speed_of_light**2  # hbar * THz / (pm/ps)^2
    epsilon0 = None  # e^2(hbar*THz*pm)^-1
    elementary_charge = 1.0  # electron charge

    ## length units in pm
    cm = 1e10
    mm = 1e9
    um = 1e6
    nm = 1e3
    Angstrom = 100
    pm = 1
    bohr_radius = 0.52917721092 * Angstrom

    ## time units in ps
    s = 1e12
    ms = 1e9
    us = 1e6
    ns = 1e3
    ps = 1
    fs = 1e-3
    As = 1e-6
    time_au = 2.4188843265864e-2 * fs

    ## additional constants
    cm_inverse_energy = speed_of_light / cm * 2 * np.pi * hbar
