"""BCHL chain helpers (see notes.md). Stubs only — implement step by step."""

from __future__ import annotations

from pathlib import Path

import re

import numpy as np
from scipy import constants as const
from scipy.linalg import expm

_DEFAULT_LINDBLAD_TXT = Path(__file__).resolve().parent / "bcf" / "lindblad_fit_results_nmode_all_6.txt"


def Intial_state(
    N_S: int = 19,
    N_B: int = 6,
    boson_dim_list: list[int] | None = None,
    *,
    psi_S: np.ndarray | None = None,
) -> np.ndarray:
    """Product state ψ_S ⊗ |0⟩ ⊗ … ⊗ |0⟩ as a full tensor.

    Shape (N_S, d0, d1, …) with d_j = boson_dim_list[j]. Each bath mode is |0⟩.

    If ``psi_S`` is None (default), ψ_S is the same localized state as before: 1 at
    index ``N_S // 2``, 0 elsewhere. If ``psi_S`` is given, it must be a length-``N_S``
    wavefunction; it is normalized to unit Euclidean norm before tensoring with the bath
    vacuum.

    Default boson dims: [2, 2, 3, 3, 2, 2].
    """
    if boson_dim_list is None:
        boson_dim_list = [2, 2, 3, 3, 2, 2]
    if len(boson_dim_list) != N_B:
        raise ValueError("len(boson_dim_list) must equal N_B")

    if psi_S is None:
        psi_S_coeffs = np.zeros(N_S, dtype=np.complex128)
        psi_S_coeffs[N_S // 2] = 1.0
    else:
        psi_S_arr = np.asarray(psi_S, dtype=np.complex128).ravel()
        if psi_S_arr.size != N_S:
            raise ValueError(f"psi_S must have length N_S={N_S}, got {psi_S_arr.size}")
        psi_S_coeffs = psi_S_arr
        nrm = np.linalg.norm(psi_S_coeffs)
        if not np.isfinite(nrm) or nrm == 0.0:
            raise ValueError("psi_S must have finite, nonzero norm")
        psi_S_coeffs = psi_S_coeffs / nrm

    shape = (N_S,) + tuple(boson_dim_list)
    psi = np.zeros(shape, dtype=np.complex128)
    idx0 = (slice(None),) + (0,) * N_B
    psi[idx0] = psi_S_coeffs
    return psi


def _balanced_brackets(text: str, start: int) -> str:
    """First '[' at/after start through its matching ']', inclusive."""
    i = text.index("[", start)
    depth = 0
    for k in range(i, len(text)):
        if text[k] == "[":
            depth += 1
        elif text[k] == "]":
            depth -= 1
            if depth == 0:
                return text[i : k + 1]
    raise ValueError("unclosed '[' in file")


def load_data(
    filename: str | Path = _DEFAULT_LINDBLAD_TXT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse Lindblad dump: h_b (N_B×N_B), g (N_B,), gamma (N_B,).

    Default ``filename`` is ``bcf/lindblad_fit_results_nmode_all_6.txt`` next to this module.

    File is NumPy's text format (spaces, not commas; row splits like ``] [``).
    """
    raw = Path(filename).read_text(encoding="utf-8")
    if "Hamiltonian H" not in raw or "Coupling g:" not in raw:
        raise ValueError("unexpected fit file layout")

    h_raw = _balanced_brackets(raw, raw.index("[[", raw.index("Hamiltonian H")))
    inner = re.sub(r"[\[\]]", " ", h_raw.strip()[2:-2])
    toks = inner.split()
    flat: list[complex] = []
    i = 0
    while i < len(toks):
        if toks[i].endswith("j"):
            flat.append(complex(toks[i]))
            i += 1
        else:
            flat.append(complex(toks[i] + toks[i + 1]))
            i += 2
    n = int(len(flat) ** 0.5)
    if n * n != len(flat):
        raise ValueError("h_b not square")
    h_b = np.asarray(flat, dtype=np.complex128).reshape(n, n)

    g_raw = _balanced_brackets(raw, raw.index("Coupling g:"))
    g = np.asarray([float(x) for x in g_raw.strip()[1:-1].replace("\n", " ").split()], dtype=np.float64)

    gamma_key = "Dissipation γ (cm^-1):"
    if gamma_key not in raw:
        raise ValueError("missing gamma section")
    gamma_raw = _balanced_brackets(raw, raw.index(gamma_key) + len(gamma_key))
    gamma = np.asarray([float(x) for x in gamma_raw.strip()[1:-1].replace("\n", " ").split()], dtype=np.float64)

    nb = h_b.shape[0]
    if h_b.shape[1] != nb or g.shape != (nb,) or gamma.shape != (nb,):
        raise ValueError("h_b, g, gamma size mismatch")

    return h_b, g, gamma


def _annihilation_truncated(dim: int) -> np.ndarray:
    """Number basis |0⟩,…,|dim−1⟩; matrix elements ⟨m−1|a|m⟩ = √m."""
    a = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(float(n))
    return a


def _embed_bath_op(op_k: np.ndarray, mode_k: int, boson_dim_list: list[int]) -> np.ndarray:
    """Embed ``op_k`` (d_k×d_k) as acting on bath mode ``mode_k`` in the full product space.

    Basis matches ``psi.reshape(N_S, d0, …)``: axis ``d0`` is slowest, ``d_{N_B-1}`` fastest
    (same order as ``np.ravel`` on the bath tensor axes).
    """
    factors: list[np.ndarray] = []
    for m, d in enumerate(boson_dim_list):
        if m == mode_k:
            factors.append(op_k.astype(np.complex128, copy=False))
        else:
            factors.append(np.eye(d, dtype=np.complex128))
    out = factors[0]
    for M in factors[1:]:
        out = np.kron(out, M)
    return out


def make_local_hamiltonian(
    phi_array: np.ndarray,
    h_b: np.ndarray,
    g: np.ndarray,
    gamma: np.ndarray,
    boson_dim_list: list[int] | None = None,
    V_on_site_array: np.ndarray | None = None,
    V_hopping_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return H_S (N_S×N_S), H_B (N_B_tot×N_B_tot), H_SB_list (length N_B).

    ``H_B`` is in the basis where bath indices follow ``boson_dim_list[0]`` as the **slowest**
    axis and ``boson_dim_list[-1]`` the **fastest**, i.e. the same layout as
    ``psi.reshape(N_S, *boson_dim_list).reshape(N_S, -1)`` for fixed system index.

    If V_on_site_array / V_hopping_array are None, use 0 on-site energies and 363
    hopping as in notes (hopping only), with lengths N_S and N_S-1.
    """
    phi_array = np.asarray(phi_array, dtype=np.float64).ravel()
    N_S = phi_array.size
    if boson_dim_list is None:
        boson_dim_list = [2, 2, 3, 3, 2, 2]
    N_B = len(boson_dim_list)
    n_tot_b = int(np.prod(boson_dim_list))

    h_b = np.asarray(h_b, dtype=np.complex128)
    g = np.asarray(g, dtype=np.float64).ravel()
    gamma = np.asarray(gamma, dtype=np.float64).ravel()

    if h_b.shape != (N_B, N_B):
        raise ValueError("h_b must be N_B×N_B")
    if g.shape != (N_B,) or gamma.shape != (N_B,):
        raise ValueError("g and gamma must have length N_B")

    if V_on_site_array is None:
        V_on_site_array = np.zeros(N_S, dtype=np.float64)
    else:
        V_on_site_array = np.asarray(V_on_site_array, dtype=np.float64).ravel()
    if V_hopping_array is None:
        V_hopping_array = np.full(max(N_S - 1, 0), 363.0, dtype=np.float64)
    else:
        V_hopping_array = np.asarray(V_hopping_array, dtype=np.float64).ravel()

    if V_on_site_array.size != N_S or V_hopping_array.size != max(N_S - 1, 0):
        raise ValueError("V_on_site_array / V_hopping_array length mismatch with N_S")

    H_S = np.zeros((N_S, N_S), dtype=np.float64)
    np.fill_diagonal(H_S, V_on_site_array)
    for i in range(N_S - 1):
        t = V_hopping_array[i]
        H_S[i, i + 1] = t
        H_S[i + 1, i] = t

    B_ops: list[np.ndarray] = []
    for k in range(N_B):
        a_k = _annihilation_truncated(boson_dim_list[k])
        B_ops.append(_embed_bath_op(a_k, k, boson_dim_list))

    H_B = np.zeros((n_tot_b, n_tot_b), dtype=np.complex128)
    for i in range(N_B):
        Bi_dag = B_ops[i].conj().T
        for j in range(N_B):
            H_B += h_b[i, j] * (Bi_dag @ B_ops[j])
        H_B -= 1j * gamma[i] * (Bi_dag @ B_ops[i])

    H_SB_list: list[np.ndarray] = []
    for k in range(N_B):
        n_k = boson_dim_list[k]
        a_k = _annihilation_truncated(n_k)
        a_k_dag = a_k.conj().T
        Hk = np.zeros((N_S * n_k, N_S * n_k), dtype=np.complex128)
        for j in range(N_S):
            blk = g[k] * (np.exp(-1j * phi_array[j]) * a_k + np.exp(1j * phi_array[j]) * a_k_dag)
            ja = j * n_k
            Hk[ja : ja + n_k, ja : ja + n_k] = blk
        H_SB_list.append(Hk)

    return H_S, H_B, H_SB_list


def partial_trace(psi: np.ndarray) -> np.ndarray:
    """ρ_S = Tr_b |ψ⟩⟨ψ|, shape (N_S, N_S). Fast contraction over bath axes.

    Same as ``np.tensordot(psi, psi.conj(), axes=(range(1, ndim), range(1, ndim)))``.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    if psi.ndim < 2:
        raise ValueError("psi must have shape (N_S, d0, ...) with at least one bath axis")
    m = psi.shape[0]
    psi_m = psi.reshape(m, -1)
    return psi_m @ psi_m.conj().T


def _annihilate_bath_axis(psi: np.ndarray, bath_axis: int) -> np.ndarray:
    """Apply truncated ``a`` on ``psi`` along axis ``bath_axis`` (≥ 1; axis 0 is system)."""
    d = psi.shape[bath_axis]
    a = _annihilation_truncated(d)
    moved = np.moveaxis(psi, bath_axis, -1)
    head = moved.shape[:-1]
    flat = moved.reshape(-1, d)
    out_flat = flat @ a.T
    out = out_flat.reshape(*head, d)
    return np.moveaxis(out, -1, bath_axis)


def norm_bi(psi: np.ndarray, i: int) -> float:
    """‖ b_i ψ ‖ with b_i the truncated annihilation on mode i (0-based)."""
    psi = np.asarray(psi, dtype=np.complex128)
    if psi.ndim < 2:
        raise ValueError("psi must have shape (N_S, d0, ...) with at least one bath axis")
    if i < 0 or i >= psi.ndim - 1:
        raise ValueError("bath mode index i must satisfy 0 <= i < number of bath axes")
    ap = _annihilate_bath_axis(psi, i + 1)
    return float(np.linalg.norm(ap.ravel()))


def _apply_exp_H_S(psi: np.ndarray, U_S: np.ndarray) -> np.ndarray:
    """Apply ``U_S`` on the system axis (axis 0); ``psi`` is ``(N_S, *bath)``."""
    shp = psi.shape
    m = psi.reshape(shp[0], -1)
    out = (U_S @ m).reshape(shp)
    return np.asarray(out, dtype=np.complex128)


def _apply_exp_H_B(psi: np.ndarray, U_B: np.ndarray) -> np.ndarray:
    """Apply ``U_B`` on the flattened bath (C order); same basis as ``H_B`` / ``partial_trace``."""
    shp = psi.shape
    m = psi.reshape(shp[0], -1)
    out = (m @ U_B.T).reshape(shp)
    return np.asarray(out, dtype=np.complex128)


def _apply_exp_H_SB(
    psi: np.ndarray,
    U_sb: np.ndarray,
    mode_index: int,
    N_S: int,
    boson_dim_list: list[int],
) -> np.ndarray:
    """Apply ``U_sb`` on (system, bath mode ``mode_index``); block order ``s * n_k + t``."""
    n_k = boson_dim_list[mode_index]
    p = np.moveaxis(psi, mode_index + 1, 1)
    tail = p.shape[2:]
    r = int(np.prod(tail)) if tail else 1
    blk = p.reshape(N_S * n_k, r)
    out_blk = U_sb @ blk
    out = out_blk.reshape(N_S, n_k, *tail)
    return np.asarray(np.moveaxis(out, 1, mode_index + 1), dtype=np.complex128)


def run_one_trajectory(
    N_S: int,
    V: float,
    Nt: int,
    dt: float,
    h_b: np.ndarray,
    g: np.ndarray,
    gamma: np.ndarray,
    boson_dim_list: list[int] | None = None,
    V_on_site_array: np.ndarray | None = None,
    V_hopping_array: np.ndarray | None = None,
    *,
    dt_in_fs: bool = False,
    phi_array: np.ndarray | None = None,
    psi_S: np.ndarray | None = None,
) -> np.ndarray:
    """Single quantum trajectory (notes §6). Returns ``rho_S_list`` shape ``(N_S, N_S, Nt+1)``.

    Steps: ``phi`` for ``make_local_hamiltonian`` (random uniform ``[0,2π)`` per site unless
    ``phi_array`` is given with length ``N_S``); ``Intial_state`` → ``rho[:,:,0]``; build ``H``
    and short-time propagators; draw ``s``; each step apply ``U_S``, ``U_B``, then each ``U_SB[k]``;
    if ``‖ψ‖ < s`` resample ``s``, pick channel ``k`` with probabilities ``∝ ‖b_k ψ‖``, apply ``b_k``,
    normalize; store ``partial_trace``.

    If ``dt_in_fs`` is True, ``H`` is in **cm⁻¹** and ``dt`` is the step in **femtoseconds**;
    propagators use ``expm(-1j · 2π c Δt · H)`` with ``c`` in cm/s and ``Δt`` in seconds.

    If ``dt_in_fs`` is False (default), use ``expm(-1j · dt · H)`` (dimensionless pairing).

    ``V`` is kept for the notes API (unused; RNG is not seeded from it).

    Optional ``psi_S`` is passed to ``Intial_state`` (length ``N_S`` subsystem wavefunction).
    """
    _ = V

    if boson_dim_list is None:
        boson_dim_list = [2, 2, 3, 3, 2, 2]
    N_B = len(boson_dim_list)
    rng = np.random.default_rng()

    # 6.1 on-site phases (optional fixed ``phi_array``)
    if phi_array is None:
        phi_used = rng.uniform(0.0, 2.0 * np.pi, size=N_S)
    else:
        phi_used = np.asarray(phi_array, dtype=np.float64).ravel()
        if phi_used.size != N_S:
            raise ValueError("phi_array must have length N_S")

    # 6.2 initial ψ and reduced state at step 0
    psi = Intial_state(N_S=N_S, N_B=N_B, boson_dim_list=boson_dim_list, psi_S=psi_S)
    rho_S_list = np.zeros((N_S, N_S, Nt + 1), dtype=np.complex128)
    rho_S_list[:, :, 0] = partial_trace(psi)

    # 6.3–6.4 local Hamiltonians and propagators
    H_S, H_B, H_SB_list = make_local_hamiltonian(
        phi_used,
        h_b,
        g,
        gamma,
        boson_dim_list=boson_dim_list,
        V_on_site_array=V_on_site_array,
        V_hopping_array=V_hopping_array,
    )
    if dt_in_fs:
        dt_s = float(dt) * 1e-15
        phase_scale = 2.0 * np.pi * (const.c * 100.0) * dt_s
    else:
        phase_scale = float(dt)
    U_S = expm(-1j * phase_scale * np.asarray(H_S, dtype=np.complex128))
    U_B = expm(-1j * phase_scale * H_B)
    U_SB_list = [expm(-1j * phase_scale * Hk) for Hk in H_SB_list]

    # 6.5 jump threshold in (0, 1)
    s_thr = float(rng.uniform(0.0, 1.0))

    # 6.6 propagate Nt steps
    for step in range(Nt):
        # 6.6.1 split-step maps: system, bath tensor, then each joint factor
        psi = _apply_exp_H_S(psi, U_S)
        psi = _apply_exp_H_B(psi, U_B)
        for k in range(N_B):
            psi = _apply_exp_H_SB(psi, U_SB_list[k], k, N_S, boson_dim_list)

        nrm = float(np.linalg.norm(psi.ravel()))
        if nrm < s_thr:
            # 6.6.2.1
            s_thr = float(rng.uniform(0.0, 1.0))
            # 6.6.2.2 probabilities ∝ ‖ b_k ψ ‖
            weights = np.array([norm_bi(psi, k) for k in range(N_B)], dtype=np.float64)
            wsum = float(weights.sum())
            if wsum > 0.0:
                ch = int(rng.choice(N_B, p=weights / wsum))
            else:
                ch = int(rng.integers(0, N_B))
            psi = _annihilate_bath_axis(psi, ch + 1)
            # 6.6.2.3
            n2 = float(np.linalg.norm(psi.ravel()))
            if n2 > 0.0:
                psi /= n2

        rho_S_list[:, :, step + 1] = partial_trace(psi)

    return rho_S_list
