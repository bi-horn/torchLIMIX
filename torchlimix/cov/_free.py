'''Torch reimplementation from LIMIX FreeFormCov class: https://github.com/limix/glimix-core/blob/master/glimix_core/cov/_free.py'''

import torch
import torch.nn as nn
from typing import Dict, Tuple
from torchlimix.torch_cache import VersionedCacheMixin
from torchlimix.optimizer._optimizer import TorchFunction
from numpy import sqrt, inf, log, finfo

class FreeFormCovTorch(TorchFunction, VersionedCacheMixin):
    """
    General definite positive matrix, K = LLᵀ + εI.
    
    A d×d covariance matrix K will have ((d+1)·d)/2 parameters defining the lower
    triangular elements of a Cholesky matrix L such that:
    
        K = LLᵀ + εI,
    
    for a very small positive number ε.
    """
    
    def __init__(self, dim: int, device: str = "cpu"):
        TorchFunction.__init__(self, "FreeFormCov")
        VersionedCacheMixin.__init__(self)
        
        self.dim = dim
        self.device = device
        self._epsilon = sqrt(finfo(float).eps) * 1000
        
        tsize = ((dim + 1) * dim) // 2
        
        self.tril_idx = torch.tril_indices(dim, dim, offset=-1)
        self.diag_idx = torch.arange(dim)
        
        self.n_offdiag = self.tril_idx.shape[1]
        self.n_diag = dim
        self.n_params = tsize
        
        assert self.n_offdiag + self.n_diag == tsize
        
        init = torch.zeros(tsize, device=device, dtype=torch.double)
        init[:self.n_offdiag] = 1.0
        init[self.n_offdiag:] = 0.0
        self.Lu = nn.Parameter(init)
        self._version_tracker.track("Lu", self.Lu)
        
        # Off-diagonal: unconstrained. Diagonal: log-space with lower bound = log(epsilon)
        self.bounds = [(-inf, +inf)] * (tsize - dim) \
                + [(log(self._epsilon), +11.0)] * dim

    @property
    def nparams(self) -> int:
        return self.n_params

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.dim, self.dim)

    @property  
    def L(self) -> torch.Tensor:
        """Lower-triangular matrix L such that K = LLᵀ + εI."""
        L = torch.zeros(
            (self.dim, self.dim), 
            device=self.device, 
            dtype=self.Lu.dtype
        )
        L[self.tril_idx[0], self.tril_idx[1]] = self.Lu[:self.n_offdiag]
        L[self.diag_idx, self.diag_idx] = torch.exp(self.Lu[self.n_offdiag:])
        return L

    @L.setter
    def L(self, value: torch.Tensor):
        """Set L matrix, converting diagonal to log space."""
        offdiag = value[self.tril_idx[0], self.tril_idx[1]]
        diag = value[self.diag_idx, self.diag_idx]
        log_diag = torch.log(torch.clamp(diag, min=1e-10))
        
        with torch.no_grad():
            self.Lu.data[:self.n_offdiag] = offdiag
            self.Lu.data[self.n_offdiag:] = log_diag

    def value(self) -> torch.Tensor:
        """Covariance matrix K = LLᵀ + εI."""
        L = self.L
        return L @ L.T + self._epsilon * torch.eye(
            self.dim, device=self.device, dtype=L.dtype
        )

    def eigh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eigen decomposition of K.
        
        Returns
        -------
        S : Tensor
            The eigenvalues in ascending order.
        U : Tensor
            Normalized eigenvectors.
        """
        return self._get_cached("eigh", self._compute_eigh)
    
    def _compute_eigh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigendecomposition."""
        L = self.L
        U, S, _ = torch.linalg.svd(L, full_matrices=True)
        
        eigenvalues = S ** 2 + self._epsilon
        eigenvalues = torch.clamp(eigenvalues, min=self._epsilon)
        
        return (eigenvalues, U)

    def logdet(self) -> torch.Tensor:
        """Log-determinant of K via eigendecomposition."""
        S, _ = self.eigh()
        return torch.log(S).sum()

    def gradient(self) -> Dict[str, torch.Tensor]:
        """
        Derivative of K with respect to Lu parameters.
        
        Returns
        -------
        dict
            {"Lu": gradient tensor of shape (dim, dim, n_params)}
        """
        L = self.L
        n = self.dim
        grad = torch.zeros(
            (n, n, self.n_params), 
            device=L.device, 
            dtype=L.dtype
        )

        # Gradient w.r.t. off-diagonal elements
        for i in range(self.n_offdiag):
            row = self.tril_idx[0][i].item()
            col = self.tril_idx[1][i].item()
            grad[row, :, i] = L[:, col]
            grad[:, row, i] += L[:, col]

        # Gradient w.r.t. diagonal elements (log-parameterized)
        m = self.n_offdiag
        for i in range(self.n_diag):
            L_ii = L[i, i]
            grad[i, :, m + i] = L_ii * L[:, i]
            grad[:, i, m + i] += L_ii * L[:, i]

        return {"Lu": grad}

    def clear_cache(self):
        """Clear cached computations."""
        if hasattr(self, '_cache'):
            self._cache.clear()
        if hasattr(self, '_versioned_property_cache'):
            self._versioned_property_cache.clear()

    def __str__(self):
        return f"FreeFormCov(dim={self.dim}): {self.name}\n  L: {self.L}"

