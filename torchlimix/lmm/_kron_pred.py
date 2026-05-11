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
        
        self._nsamples_train, self._ntraits = Y_train.shape
        
        # Precompute training covariance and factorization
        print("Building training covariance matrix...")
        self._K_full = self._build_covariance(self._K_train, self._nsamples_train)
        
        jitter = 1e-6 * torch.trace(self._K_full) / self._K_full.shape[0]
        self._K_full = self._K_full + jitter * torch.eye(
            self._K_full.shape[0], dtype=self.dtype, device=self.device  
        )
        
        print("Computing Cholesky factorization...")
        self._L = torch.linalg.cholesky(self._K_full)
        
        # Precompute training residuals
        self._meansamples_train = self._compute_mean(self._X_train)
        self._residual = (self._Y_train - self._meansamples_train).T.flatten()
        
        print(f"MultiTraitLMMPredict initialized: n={self._nsamples_train}, p={self._ntraits}, dtype={self.dtype}")
        
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
        nsamples_test = K_test_train.shape[0]
        
        # Fixed effects prediction
        meansamples_test = self._compute_mean(X_test)
        
        # Cross-covariance: C0 ⊗ K_test_train
        K_cross = torch.kron(self._C0, K_test_train)
        
        # BLUP adjustment: K_cross @ K⁻¹ @ residual
        solved = torch.cholesky_solve(
            self._residual.unsqueeze(-1), self._L
        ).squeeze(-1)
        adjustment = K_cross @ solved
        
        # Reshape to (nsamples_test, p)
        adjustment = adjustment.reshape(self._ntraits, nsamples_test).T
        
        pred_mean = meansamples_test + adjustment
        result = {'mean': pred_mean}
        
        if return_variance and K_test_test is not None:
            K_test_test = K_test_test.to(dtype=self.dtype, device=self.device)

            K_cross_solved = torch.linalg.solve_triangular(
                self._L, K_cross.T, upper=False
            )  # (ntraits*nsamples_train, ntraits*nsamples_test)

            #   diag(C0 ⊗ K_test_test)[p, i] = C0[p,p] * K_test_test[i,i]
            #   diag(C1 ⊗ I_test)[p, i]      = C1[p,p]
            c0_diag = torch.diag(self._C0)        # (P,)
            c1_diag = torch.diag(self._C1)        # (P,)
            k_tt_diag = torch.diag(K_test_test)   # (nsamples_test,)

            diag_prior = (
                c0_diag.repeat_interleave(nsamples_test) * k_tt_diag.repeat(self._ntraits)
                + c1_diag.repeat_interleave(nsamples_test)
            )  # (ntraits*nsamples_test,)

            # Diagonal of K_cross_solved.T @ K_cross_solved is the column-wise squared norm.
            diag_reduction = (K_cross_solved ** 2).sum(dim=0)  # (ntraits*nsamples_test,)

            diag_posterior = (diag_prior - diag_reduction).clamp(min=0)
            variance = diag_posterior.reshape(self._ntraits, nsamples_test).T  # (nsamples_test, p)

            result['variance'] = variance
        
        return result