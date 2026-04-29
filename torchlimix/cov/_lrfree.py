"""
Torch reimplementation from LIMIX LRFreeFormCov class.
https://github.com/limix/glimix-core/blob/master/glimix_core/cov/_lrfree.py
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from torchlimix.torch_cache import VersionedCacheMixin
from torchlimix.optimizer._optimizer import TorchFunction
import numpy as np

class LRFreeFormCovTorch(TorchFunction, VersionedCacheMixin):
    """
    Low-rank free-form covariance: K = LLᵀ
    
    L is a (dim x rank) matrix, so K is at most rank `rank`.
    No epsilon regularization (semi-definite, not positive definite).

    matching the NumPy implementation exactly.
    """
    
    def __init__(self, dim: int, rank: int, original_L_init: bool, device: str = "cpu"):
        # Initialize both parent classes
        TorchFunction.__init__(self, "LRFreeFormCov")
        VersionedCacheMixin.__init__(self)
        
        self.dim = dim
        self.rank = rank
        self.device = device
        self.n_params = dim * rank
        
        # NumPy default: ones((dim, rank))
        if original_L_init:
            L_init = torch.ones(dim, rank, device=device, dtype=torch.double)
        else:
            # QR initialization for better numerical properties when rank >= dim
            rng = np.random.RandomState(0)
            Q, _ = np.linalg.qr(rng.randn(dim, rank))
            L_init = torch.from_numpy(Q).to(device=device, dtype=torch.double)

        init = L_init.flatten()
        self.Lu = nn.Parameter(init)
        
        # Track the parameter for automatic cache invalidation
        self._version_tracker.track("Lu", self.Lu)
        
        self.bounds = [(None, None)] * self.n_params

    @property
    def nparams(self) -> int:
        return self.n_params

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.dim, self.dim)

    @property
    def L(self) -> torch.Tensor:
        """
        Matrix L from K = LLᵀ.
        
        Returns (dim, rank) matrix reshaped from flat Lu parameter.
        Uses row-major order to match NumPy's behavior.
        """
        return self.Lu.reshape(self.dim, self.rank)

    @L.setter
    def L(self, value: torch.Tensor):
        """Set L matrix, flattening to update Lu."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.double, device=self.device)
        with torch.no_grad():
            self.Lu.data = value.reshape(-1).to(dtype=torch.double, device=self.device)

    def value(self) -> torch.Tensor:
        """Covariance matrix K = LLᵀ (no epsilon)."""
        L = self.L
        return L @ L.T

    def gradient(self) -> Dict[str, torch.Tensor]:
        """
        Derivative of K = LLᵀ with respect to Lu parameters.
        
        Matches NumPy implementation exactly:
            for ii in range(n * m):
                row = ii // m
                col = ii % m
                grad["Lu"][row, :, ii] = L[:, col]
                grad["Lu"][:, row, ii] += L[:, col]
        
        Returns
        -------
        dict
            {"Lu": gradient tensor of shape (dim, dim, n_params)}
        """
        L = self.L
        d, r = self.dim, self.rank
        
        grad = torch.zeros((d, d, self.n_params), device=L.device, dtype=L.dtype)
        
        # Match NumPy indexing exactly
        for ii in range(self.n_params):
            row = ii // r  # row = ii // rank (m in NumPy)
            col = ii % r   # col = ii % rank
            grad[row, :, ii] = L[:, col]
            grad[:, row, ii] += L[:, col]
        
        return {"Lu": grad}

    def clear_cache(self):
        """Clear cached computations."""
        if hasattr(self, '_cache'):
            self._cache.clear()
        if hasattr(self, '_versioned_property_cache'):
            self._versioned_property_cache.clear()
            
    def __str__(self):
        L_str = str(self.L.detach().cpu().numpy())
        return f"LRFreeFormCov(n={self.dim}, m={self.rank}): {self.name}\n  L: {L_str}"
