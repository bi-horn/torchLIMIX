'''Torch implementation of multi-trait prediciton
Inspired by univariate prediciton: https://github.com/limix/glimix-core/blob/master/glimix_core/lmm/_lmm_predict.py
'''

import torch
from torch import Tensor
from typing import Dict, Optional


class MultiTraitLMMPredict:
    """
    Multi-trait BLUP prediction with Kronecker covariance structure.
    
    Model: vec(Y) ~ N(Xβ, C0 ⊗ K_G + C1 ⊗ I)
    """
    
    def __init__(
        self,
        Y_train: Tensor,
        beta: Tensor,
        C0: Tensor,
        C1: Tensor,
        K_train: Tensor,
        X_train: Tensor,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64  
    ):
        self.device = torch.device(device)
        self.dtype = dtype 
        
        # Convert all tensors to same dtype and device
        self._Y_train = Y_train.to(dtype=self.dtype, device=self.device)
        self._beta = beta.to(dtype=self.dtype, device=self.device)
        self._C0 = C0.to(dtype=self.dtype, device=self.device)
        self._C1 = C1.to(dtype=self.dtype, device=self.device)
        self._K_train = K_train.to(dtype=self.dtype, device=self.device)
        self._X_train = X_train.to(dtype=self.dtype, device=self.device)
        
        self._n_train, self._P = Y_train.shape
        
        # Precompute training covariance and factorization
        print("Building training covariance matrix...")
        self._K_full = self._build_covariance(self._K_train, self._n_train)
        
        # Add jitter for numerical stability
        jitter = 1e-6 * torch.trace(self._K_full) / self._K_full.shape[0]
        self._K_full = self._K_full + jitter * torch.eye(
            self._K_full.shape[0], dtype=self.dtype, device=self.device  
        )
        
        print("Computing Cholesky factorization...")
        self._L = torch.linalg.cholesky(self._K_full)
        
        # Precompute training residuals
        self._mean_train = self._compute_mean(self._X_train)
        self._residual = (self._Y_train - self._mean_train).T.flatten()
        
        print(f"MultiTraitLMMPredict initialized: n={self._n_train}, P={self._P}, dtype={self.dtype}")
        
    def _build_covariance(self, K_genetic: Tensor, n: int) -> Tensor:
        """Build K = C0 ⊗ K_genetic + C1 ⊗ I"""
        I_n = torch.eye(n, dtype=self.dtype, device=self.device)  
        return torch.kron(self._C0, K_genetic) + torch.kron(self._C1, I_n)
    
    def _compute_mean(self, X: Tensor) -> Tensor:
        """Fixed effects prediction: X @ β"""
        X = X.to(dtype=self.dtype, device=self.device)
        return X @ self._beta
    
    def predict(
        self, 
        X_test: Tensor,
        K_test_train: Tensor,
        K_test_test: Optional[Tensor] = None,
        return_variance: bool = True
    ) -> Dict[str, Tensor]:
        """
        Predict phenotypes for test samples.
        """
        X_test = X_test.to(dtype=self.dtype, device=self.device)
        K_test_train = K_test_train.to(dtype=self.dtype, device=self.device)
        n_test = K_test_train.shape[0]
        
        # Fixed effects prediction
        mean_test = self._compute_mean(X_test)
        
        # Cross-covariance: C0 ⊗ K_test_train
        K_cross = torch.kron(self._C0, K_test_train)
        
        # BLUP adjustment: K_cross @ K⁻¹ @ residual
        solved = torch.cholesky_solve(
            self._residual.unsqueeze(-1), self._L
        ).squeeze(-1)
        adjustment = K_cross @ solved
        
        # Reshape to (n_test, P)
        adjustment = adjustment.reshape(self._P, n_test).T
        
        pred_mean = mean_test + adjustment
        result = {'mean': pred_mean}
        
        if return_variance and K_test_test is not None:
            K_test_test = K_test_test.to(dtype=self.dtype, device=self.device)  
            
            # Prior covariance
            I_test = torch.eye(n_test, dtype=self.dtype, device=self.device)  
            K_prior = torch.kron(self._C0, K_test_test) + torch.kron(self._C1, I_test)
            
            # Posterior reduction
            K_cross_solved = torch.linalg.solve(self._L, K_cross.T)
            reduction = K_cross_solved.T @ K_cross_solved
            K_posterior = K_prior - reduction
            
            # Extract per-sample, per-trait variances
            variance = torch.zeros(n_test, self._P, dtype=self.dtype, device=self.device)  
            for p in range(self._P):
                for i in range(n_test):
                    idx = p * n_test + i
                    variance[i, p] = K_posterior[idx, idx].clamp(min=0)
            
            result['variance'] = variance
            result['covariance'] = K_posterior
        
        return result