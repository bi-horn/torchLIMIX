import numpy as np
from typing import Tuple
from numpy import finfo, sqrt
import torch

def prepare_kinship_pipeline(
    G: np.ndarray,
    epsilon: float = sqrt(finfo(float).eps),
    debug: bool = False,
    chunk_size: int = 100_000,
) -> Tuple[
    np.ndarray,
    Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
    np.ndarray,
]:
    """
    Prepare kinship matrix with memory-efficient chunked computation (NumPy).

    Expects a pre-standardized genotype matrix (mean=0, std=1 per column).
    Computes K = G @ G.T / S, then eigendecomposes and enforces PSD.

    Parameters
    ----------
    G : np.ndarray
        Standardized genotype matrix (N individuals × S SNPs)
    epsilon : float
        Small value for numerical stability
    debug : bool
        Print debug information
    chunk_size : int
        Number of SNPs to process at once

    Returns
    -------
    K : np.ndarray
        PSD-corrected kinship matrix (N × N)
    QS : Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
        ((Q0, Q1), S0) eigendecomposition components
    G_lowrank : np.ndarray
        Low-rank representation such that K = G_lowrank @ G_lowrank.T
    """
    G = np.asarray(G)
    if G.ndim != 2:
        raise ValueError("G must be a 2D array (N × S)")

    N, S = G.shape
    dtype = G.dtype

    if debug:
        print(f"[DEBUG] Computing kinship: N={N}, S={S}")
        print(f"[DEBUG] Input G: mean={G.mean():.4e}, std={G.std():.4f}")
        if S > chunk_size:
            print(f"[DEBUG] Computing kinship in chunks of {chunk_size} SNPs...")

    # Compute K = G @ G.T / S in chunks
    K_full = np.zeros((N, N), dtype=dtype)

    num_chunks = (S + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, S)

        G_chunk = G[:, start_idx:end_idx]
        K_full += G_chunk @ G_chunk.T

        if debug and num_chunks > 1:
            if (i + 1) % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                print(
                    f"[DEBUG] Kinship progress: "
                    f"{end_idx}/{S} SNPs ({100 * end_idx / S:.1f}%)"
                )

    K_full /= S

    if debug:
        print(f"[DEBUG] K_full shape: {K_full.shape}")
        print(f"[DEBUG] K_full diagonal mean: {np.diag(K_full).mean():.4f}")

    # Eigendecomposition (symmetric PSD)
    S_full, Q_full = np.linalg.eigh(K_full)

    if debug:
        print(f"[DEBUG] Original eigenvalues: {len(S_full)} total")
        print(
            f"[DEBUG] Eigenvalue range: "
            f"[{S_full.min():.6e}, {S_full.max():.6f}]"
        )
        print(
            f"[DEBUG] Eigenvalues below threshold ({epsilon:.2e}): "
            f"{np.sum(S_full < epsilon)}"
        )

    # Filter small eigenvalues
    ok = S_full >= epsilon
    nok = ~ok

    S0 = S_full[ok]
    Q0 = Q_full[:, ok]
    Q1 = Q_full[:, nok] if np.any(nok) else np.empty((N, 0), dtype=dtype)

    if debug:
        print(f"[DEBUG] Filtered {np.sum(nok)} eigenvalues below {epsilon:.2e}")
        print(f"[DEBUG] Kept {np.sum(ok)} eigenvalues")

    # Reconstruct PSD-corrected K via low-rank factors
    G_lowrank = Q0 @ np.diag(np.sqrt(S0))
    K = G_lowrank @ G_lowrank.T

    return K, ((Q0, Q1), S0), G_lowrank


def prepare_kinship_pipeline_torch(
    G: torch.Tensor,
    epsilon: float = sqrt(finfo(float).eps),
    debug: bool = False,
    chunk_size: int = 100000,
) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:
    """
    Prepare kinship matrix with memory-efficient chunked computation.

    Expects a pre-standardized genotype matrix (mean=0, std=1 per column).
    Computes K = G @ G.T / S, then eigendecomposes.

    Parameters
    ----------
    G : torch.Tensor
        Standardized genotype matrix (N individuals × S SNPs)
    epsilon : float
        Small value for numerical stability
    debug : bool
        Print debug information
    chunk_size : int
        Number of SNPs to process at once (reduce if memory issues persist)

    Returns
    -------
    K : torch.Tensor
        Kinship matrix (N × N)
    QS : Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        ((Q0, Q1), S0) - Eigendecomposition components
    G_lowrank : torch.Tensor
        Low-rank representation
    """
    N, S = G.shape
    device = G.device
    dtype = G.dtype

    if debug:
        print(f"[DEBUG] Computing kinship: N={N}, S={S}")
        print(f"[DEBUG] Input G: mean={G.mean():.4e}, std={G.std():.4f}")
        if S > chunk_size:
            print(f"[DEBUG] Computing kinship in chunks of {chunk_size} SNPs...")

    # Compute K = G @ G.T / S in chunks
    K_full = torch.zeros((N, N), dtype=dtype, device=device)

    num_chunks = (S + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, S)

        G_chunk = G[:, start_idx:end_idx]
        K_full += G_chunk @ G_chunk.T

        if debug and num_chunks > 1:
            if (i + 1) % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                print(f"[DEBUG] Kinship progress: {end_idx}/{S} SNPs ({100*end_idx/S:.1f}%)")

        del G_chunk

    K_full = K_full / S

    if debug:
        print(f"[DEBUG] K_full shape: {K_full.shape}")
        print(f"[DEBUG] K_full diagonal mean: {K_full.diagonal().mean():.4f}")

    # Eigendecomposition
    S_full, Q_full = torch.linalg.eigh(K_full)

    if debug:
        print(f"[DEBUG] Original eigenvalues: {len(S_full)} total")
        print(f"[DEBUG] Eigenvalue range: [{S_full.min():.6e}, {S_full.max():.6f}]")
        print(f"[DEBUG] Eigenvalues below threshold ({epsilon:.2e}): {(S_full < epsilon).sum()}")

    # Filter small eigenvalues (matching economic_qs)
    ok = S_full >= epsilon
    nok = ~ok

    S0 = S_full[ok]
    Q0 = Q_full[:, ok]
    Q1 = Q_full[:, nok] if nok.sum() > 0 else torch.empty(
        (N, 0), dtype=dtype, device=device
    )

    if debug:
        print(f"[DEBUG] Filtered {nok.sum()} eigenvalues below {epsilon:.2e}")
        print(f"[DEBUG] Kept {ok.sum()} eigenvalues")

    # Reconstruct PSD-corrected K
    G_lowrank = Q0 @ torch.diag(torch.sqrt(S0))
    K = G_lowrank @ G_lowrank.T

    return K, ((Q0, Q1), S0), G_lowrank