'''Torch implementation of the following multi-trait glimix-core class
Kron2SumCov (https://github.com/limix/glimix-core/blob/master/glimix_core/cov/_kron2sum.py)
'''

import torch
import torch as nn
from torch import nn
from typing import Dict
from torchlimix.cov._free import FreeFormCovTorch
from torchlimix.cov._lrfree import LRFreeFormCovTorch
from torchlimix.torch_cache import (
    VersionedCacheMixin,
    HierarchicalVersionTracker,
    versioned_cached_property
)

class Kron2SumCovTorch(nn.Module, VersionedCacheMixin):
    """
    Implements K = C₀ ⊗ GGᵀ + C₁ ⊗ I.
    
    This implementation matches the NumPy Kron2SumCov class exactly.
    """
    
    def __init__(self, G, dim, rank=None, original_L_init=False, device="cpu"):
        nn.Module.__init__(self)

        self._device = device
        self._G = G.to(device)
        self.dim = dim
        self.rank = rank
        self.n, self.m = self._G.shape
        
        # SVD cache - computed once since G is constant
        self._Lx = None
        self._Sx = None
        self._Sxe = None
        self._LxG = None
        self._diag_LxGGLx = None  
        self._LxGe = None
        self._diag_LxGGLxe = None

        self._C0 = LRFreeFormCovTorch(dim, rank, original_L_init, device=device)
        self._C0.name = "C₀"
        self._C1 = FreeFormCovTorch(dim, device=device)
        self._C1.name = "C₁"

        # Create hierarchical version tracker that monitors C0 and C1
        self._version_tracker = HierarchicalVersionTracker()
        self._version_tracker.track_child(self._C0._version_tracker)
        self._version_tracker.track_child(self._C1._version_tracker)
        self._cache: dict = {}
        self._versioned_property_cache: dict = {} 
        
        self.name = "Kron2SumCov"

    def clear_cache(self):
        """Clear all cached computations."""
        self._cache.clear()
        self._versioned_property_cache.clear()
        
        if hasattr(self._C0, 'clear_cache'):
            self._C0.clear_cache()
        if hasattr(self._C1, 'clear_cache'):
            self._C1.clear_cache()

    def _init_svd(self):
        """
        Initialize SVD components if not already cached.
        
        Matches NumPy exactly:
        - Pads _Sx with 0.0 (not 1e-6)
        - Computes _diag_LxGGLx for full Lx (not just Lxe)
        """
        if self._Lx is not None:
            return

        U, S, _ = torch.linalg.svd(self._G, full_matrices=False)
        
        # S contains singular values, square them for eigenvalues of GG^T
        self._Sxe = S ** 2
        
        pad_len = max(0, self.n - self._Sxe.shape[0])
        if pad_len > 0:
            self._Sx = torch.cat([self._Sxe, torch.zeros(pad_len, device=self.device, dtype=self._Sxe.dtype)])
        else:
            self._Sx = self._Sxe.clone()
        
        # Lx = U^T
        self._Lx = U.T
        
        # LxG = Lx @ G
        self._LxG = self._Lx @ self._G
        
        # dotd(A, B) computes diag(A @ B)
        self._diag_LxGGLx = torch.sum(self._LxG * self._LxG, dim=1)  # diag(LxG @ LxG.T)
        
        # Economical versions (only non-zero singular values)
        self._Lxe = U[:, :S.shape[0]].T
        self._LxGe = self._Lxe @ self._G
        self._diag_LxGGLxe = torch.sum(self._LxGe * self._LxGe, dim=1)  # diag(LxGe @ LxGe.T)

    def _normalize_gradient_shape(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Normalize gradient tensor to shape (n_params, d, d) for iteration.
        
        NumPy convention: (d, d, n_params)
        PyTorch convention: (n_params, d, d)
        """
        if grad.ndim != 3:
            raise ValueError(f"Expected 3D gradient tensor, got shape {grad.shape}")
        
        if grad.shape[0] == grad.shape[1] and grad.shape[0] != grad.shape[2]:
            return grad.permute(2, 0, 1)
        else:
            return grad

    def _get_gradient_numpy_convention(self, grad_raw: torch.Tensor) -> torch.Tensor:
        """
        Ensure gradient is in NumPy convention (d, d, n_params) for direct indexing.
        """
        if grad_raw.ndim != 3:
            raise ValueError(f"Expected 3D gradient tensor, got shape {grad_raw.shape}")
        
        # If already (d, d, n_params)
        if grad_raw.shape[0] == grad_raw.shape[1] and grad_raw.shape[0] != grad_raw.shape[2]:
            return grad_raw
        # If (n_params, d, d), convert
        elif grad_raw.shape[1] == grad_raw.shape[2]:
            return grad_raw.permute(1, 2, 0)
        else:
            raise ValueError(f"Cannot determine gradient convention for shape {grad_raw.shape}")

    @property
    def Lx(self):
        self._init_svd()
        return self._Lx

    @versioned_cached_property
    def Ge(self):
        """Result of US from SVD G = USV^T."""
        U, S, _ = torch.linalg.svd(self._G, full_matrices=False)
        if U.shape[1] < self._G.shape[1]:
            return U * S.unsqueeze(0)
        return self._G
    
    @versioned_cached_property
    def _GG(self):
        return self._G @ self._G.T

    @versioned_cached_property
    def _X(self):
        return self._G @ self._G.T

    @versioned_cached_property
    def _I(self):
        return torch.eye(self.n, device=self.device, dtype=self._G.dtype)

    @property
    def _LhD(self):
        return self._get_cached("LhD", self._compute_LhD)

    def _compute_LhD(self):
        self._init_svd()
        S1, U1 = self._C1.eigh()
        S1_inv_sqrt = 1.0 / torch.sqrt(S1)
        U1S1 = U1 * S1_inv_sqrt.unsqueeze(0)
        C0_val = self._C0.value()
        Ch = U1S1.T @ C0_val @ U1S1
        Sh, Uh = torch.linalg.eigh(Ch)
        Lh = (U1S1 @ Uh).T
        D = 1.0 / (torch.kron(Sh, self._Sx) + 1.0)
        De = 1.0 / (torch.kron(Sh, self._Sxe) + 1.0)
        return {"Lh": Lh, "D": D, "De": De}

    @property
    def Lh(self):
        return self._LhD["Lh"]

    @property
    def D(self):
        return self._LhD["D"]

    @property
    def _De(self):
        return self._LhD["De"]

    @property
    def G(self):
        return self._G
    
    @property
    def device(self): 
        return self._device
    
    @property
    def C0(self): 
        return self._C0
    
    @property
    def C1(self): 
        return self._C1
    
    @property
    def nparams(self): 
        return self._C0.nparams + self._C1.nparams

    def value(self):
        """K = kron(C0, GG^T) + kron(C1, I)"""
        C0 = self._C0.value() 
        C1 = self._C1.value()
        return torch.kron(C0, self._GG) + torch.kron(C1, self._I)

    def solve(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute K^{-1} @ v using K^{-1} = L.T @ D @ L.
        
        Matches NumPy: L.T @ torch_ddot(D, L @ v, left=True)
        """
        self._init_svd()
        L = torch.kron(self.Lh, self.Lx)
        D = self.D
        
        Lv = L @ v
        # torch_ddot(D, Lv, left=True) = diag(D) @ Lv = D * Lv (element-wise)
        if v.ndim == 1:
            DLv = D * Lv
        else:
            DLv = D.unsqueeze(-1) * Lv
        
        return L.T @ DLv

    def logdet(self) -> torch.Tensor:
        """
        log|K| = -log(De).sum() + n * C1.logdet()
        
        Matches NumPy exactly.
        """
        self._init_svd()
        
        De = self._De
        
        # -sum(log(De))
        logdet_D_term = -torch.sum(torch.log(De))
        
        # n * log|C1|
        logdet_C1 = self._C1.logdet()
        logdet_C1_term = self.n * logdet_C1
        
        result = logdet_D_term + logdet_C1_term
        
        return result
        
    def gradient(self) -> Dict[str, torch.Tensor]:
        """
        Gradient of K.
        
        NumPy:
            C0 = self._C0.gradient()["Lu"].T
            C1 = self._C1.gradient()["Lu"].T
            grad = {"C0.Lu": kron(C0, self._X).T, "C1.Lu": kron(C1, self._I).T}
            
        Output shape: (n*d, n*d, n_params)
        """
        self._init_svd()
        
        # Get gradients - keep in NumPy convention (d, d, n_params)
        dC0_np = self._get_gradient_numpy_convention(self._C0.gradient()["Lu"])
        dC1_np = self._get_gradient_numpy_convention(self._C1.gradient()["Lu"])
        
        # NumPy: C0 = gradient()["Lu"].T reverses all dims: (d,d,n) -> (n,d,d)
        dC0_T = dC0_np.permute(2, 0, 1)  # (n_params, d, d)
        dC1_T = dC1_np.permute(2, 0, 1)
        
        # kron(C0, self._X) with C0 (n_params, d, d), _X (n, n) -> (n_params, d*n, d*n)
        # PyTorch kron supports batched operation
        grad_C0 = torch.stack([torch.kron(dC, self._X) for dC in dC0_T])
        grad_C1 = torch.stack([torch.kron(dC, self._I) for dC in dC1_T])
        
        # .T reverses all dims: (n_params, d*n, d*n) -> (d*n, d*n, n_params)
        grad_C0 = grad_C0.permute(1, 2, 0)
        grad_C1 = grad_C1.permute(1, 2, 0)
        
        return {"C0.Lu": grad_C0, "C1.Lu": grad_C1}

    def gradient_dot(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ∂K⋅v for all parameters.
        
        NumPy:
            V = torch_unvec(v, (n, -1) + v.shape[1:])
            C = gradient()["Lu"]  # (d, d, n_params)
            r["C0.Lu"] = tensordot(V.T @ G @ G.T, C, axes=([-2], [0]))
            r["C0.Lu"] = r["C0.Lu"].reshape(V.shape[2:] + (-1,) + (C.shape[-1],), order="F")
            
        Output: (n*d, n_params) for unbatched, with possible batch dims in between
        """
        self._init_svd()
        
        n = self.n
        d = self.dim
        
        # Determine batch dimensions
        if v.ndim == 1:
            batch_shape = ()
            V = v.view(n, d)  # (n, d)
        else:
            batch_shape = v.shape[1:]
            V = v.view(n, d, *batch_shape)  # (n, d, batch...)
        
        # Get gradients in NumPy convention (d, d, n_params)
        dC0 = self._get_gradient_numpy_convention(self._C0.gradient()["Lu"])
        dC1 = self._get_gradient_numpy_convention(self._C1.gradient()["Lu"])
        
        n_params_C0 = dC0.shape[2]
        n_params_C1 = dC1.shape[2]
        
        result = {}
        
        # C0: tensordot(V.T @ GGT, dC0, axes=([-2], [0])) 
        GGT = self._G @ self._G.T  # (n, n)
        
        if V.ndim == 2:
            # V is (n, d), V.T is (d, n)
            VT_GGT = V.T @ GGT  # (d, n)
            out_C0 = torch.einsum('dn,dkp->nkp', VT_GGT, dC0)  # (n, d, n_params)
            out_C0 = out_C0.permute(1, 0, 2).reshape(-1, n_params_C0)  # (n*d, n_params)
            
        else:
            # V is (n, d, batch...), handle batch dimensions
            # V.T conceptually transposes first two dims: (d, n, batch...)
            VT = V.permute(1, 0, *range(2, V.ndim))  # (d, n, batch...)
            
            # VT @ GGT: (d, n, batch...) @ (n, n) -> need einsum
            VT_GGT = torch.einsum('dn...,nm->dm...', VT, GGT)  # (d, n, batch...)
        
            out_C0 = torch.einsum('dn...,dkp->nk...p', VT_GGT, dC0)  # (n, d, batch..., n_params)
            
            perm = [1, 0] + list(range(2, out_C0.ndim))
            out_C0 = out_C0.permute(*perm)  # (d, n, batch..., n_params)
            out_C0 = out_C0.reshape(-1, *batch_shape, n_params_C0)  # (n*d, batch..., n_params)
        
        result["C0.Lu"] = out_C0
        
        # C1: tensordot(V.T, dC1, axes=([-2], [0])) 
        if V.ndim == 2:
            VT = V.T  # (d, n)
            out_C1 = torch.einsum('dn,dkp->nkp', VT, dC1)  # (n, d, n_params)
            out_C1 = out_C1.permute(1, 0, 2).reshape(-1, n_params_C1)  # (n*d, n_params)
        else:
            VT = V.permute(1, 0, *range(2, V.ndim))  # (d, n, batch...)
            out_C1 = torch.einsum('dn...,dkp->nk...p', VT, dC1)  # (n, d, batch..., n_params)
            perm = [1, 0] + list(range(2, out_C1.ndim))
            out_C1 = out_C1.permute(*perm)
            out_C1 = out_C1.reshape(-1, *batch_shape, n_params_C1)
        
        result["C1.Lu"] = out_C1
        
        return result

    def logdet_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Compute ∂log|K| = Tr[K^{-1} ∂K].
        
        NumPy:
            grad_C0 = zeros_like(self._C0.Lu)
            for i in range(self._C0.Lu.shape[0]):
                t = kron(dotd(Lh, dC0[...,i] @ Lh.T), self._diag_LxGGLxe)
                grad_C0[i] = (De * t).sum()
                
        Output shape: matches self._C0.Lu.shape and self._C1.Lu.shape
        """
        self._init_svd()
        
        Lh = self.Lh  # (d, d)
        De = self._De  # (d * p,)
        p = self._Sxe.shape[0]
        n_minus_p = self.n - p
        
        grad = {}

        C0_Lu_shape = self._C0.Lu.shape
        C1_Lu_shape = self._C1.Lu.shape
        n_params_C0 = C0_Lu_shape[0] if C0_Lu_shape else 0
        n_params_C1 = C1_Lu_shape[0] if C1_Lu_shape else 0
        
        # Get gradients in NumPy convention (d, d, n_params)
        dC0 = self._get_gradient_numpy_convention(self._C0.gradient()["Lu"])
        dC1 = self._get_gradient_numpy_convention(self._C1.gradient()["Lu"])
        
        grad_C0 = torch.zeros(n_params_C0, device=self.device, dtype=self._G.dtype)
        
        for i in range(n_params_C0):
            # dC0[..., i] is (d, d)
            dC_i = dC0[:, :, i]
            
            # dotd(Lh, dC @ Lh.T) = diag(Lh @ dC @ Lh.T)
            LhdCLhT = Lh @ dC_i @ Lh.T
            diag_LhdCLhT = torch.diagonal(LhdCLhT)  # (d,)
            
            # kron of two 1D vectors
            t = torch.kron(diag_LhdCLhT, self._diag_LxGGLxe)  # (d * p,)
            
            grad_C0[i] = (De * t).sum()
        
        # Reshape to match Lu shape
        grad["C0.Lu"] = grad_C0.view(C0_Lu_shape)
        grad_C1 = torch.zeros(n_params_C1, device=self.device, dtype=self._G.dtype)
        
        for i in range(n_params_C1):
            dC_i = dC1[:, :, i]
            
            LhdCLhT = Lh @ dC_i @ Lh.T
            diag_LhdCLhT = torch.diagonal(LhdCLhT)  # (d,)
            
            t = (diag_LhdCLhT * n_minus_p).sum()
            eye_p = torch.eye(p, device=self.device, dtype=self._G.dtype)
            t1 = torch.kron(diag_LhdCLhT.unsqueeze(0), eye_p)  # (p, d*p)
            t += (De.unsqueeze(0) * t1).sum()
            
            grad_C1[i] = t
        
        grad["C1.Lu"] = grad_C1.view(C1_Lu_shape)
        
        return grad

    def LdKL_dot(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute L(∂K)L^T ⋅ v.
        
        NumPy:
            V = torch_unvec(v, (n, -1) + v.shape[1:])
            for i in range(n_params):
                t = dot(LxG, dot(LxG.T, dot(V, Lh @ dC0[...,i] @ Lh.T)))
                result[..., i] = t.reshape((-1,) + t.shape[2:], order="F")
                
        Output shape: (n*d,) + batch_shape + (n_params,)
        """
        self._init_svd()
        
        Lh = self.Lh  # (d, d)
        LxG = self._LxG  # (n, m)
        n = self.n
        d = self.dim
        
        # Determine shapes
        if v.ndim == 1:
            batch_shape = ()
            V = v.view(n, d)  # (n, d)
        else:
            batch_shape = v.shape[1:]
            V = v.view(n, d, *batch_shape)  # (n, d, batch...)
        
        # Get gradients in NumPy convention (d, d, n_params)
        dC0 = self._get_gradient_numpy_convention(self._C0.gradient()["Lu"])
        dC1 = self._get_gradient_numpy_convention(self._C1.gradient()["Lu"])
        
        n_params_C0 = dC0.shape[2]
        n_params_C1 = dC1.shape[2]
        
        # Allocate output - matches NumPy: (v.shape[0],) + v.shape[1:] + (n_params,)
        # = (n*d,) + batch_shape + (n_params,)
        result = {
            "C0.Lu": torch.empty((n * d,) + batch_shape + (n_params_C0,), 
                                  device=self.device, dtype=self._G.dtype),
            "C1.Lu": torch.empty((n * d,) + batch_shape + (n_params_C1,), 
                                  device=self.device, dtype=self._G.dtype),
        }
        
        # ===== C0: LxG @ LxG.T @ V @ (Lh @ dC @ Lh.T) =====
        for i in range(n_params_C0):
            dC_i = dC0[:, :, i]
            LhdCLhT = Lh @ dC_i @ Lh.T  # (d, d)
            
            if V.ndim == 2:
                t = LxG @ (LxG.T @ (V @ LhdCLhT))  # (n, d)
                
                # Reshape with Fortran order: (n, d) -> (n*d,)
                t_vec = t.T.reshape(-1)
            else:
                t = torch.einsum('nd...,de->ne...', V, LhdCLhT)  # (n, d, batch...)
                
                # dot(LxG.T, t): LxG.T (m, n) with t (n, d, batch...)
                t = torch.einsum('mn,nd...->md...', LxG.T, t)  # (m, d, batch...)
                
                # dot(LxG, t): LxG (n, m) with t (m, d, batch...)
                t = torch.einsum('nm,md...->nd...', LxG, t)  # (n, d, batch...)
                
                # Reshape with Fortran order on (n, d)
                perm = [1, 0] + list(range(2, t.ndim))
                t_vec = t.permute(*perm).reshape(-1, *batch_shape)  # (n*d, batch...)
            
            result["C0.Lu"][..., i] = t_vec
        
        # ===== C1: V @ (Lh @ dC @ Lh.T) =====
        # Note: Lx @ Lx.T = I (orthogonal), so no LxG term
        for i in range(n_params_C1):
            dC_i = dC1[:, :, i]
            LhdCLhT = Lh @ dC_i @ Lh.T
            
            if V.ndim == 2:
                t = V @ LhdCLhT  # (n, d)
                t_vec = t.T.reshape(-1)
            else:
                t = torch.einsum('nd...,de->ne...', V, LhdCLhT)
                perm = [1, 0] + list(range(2, t.ndim))
                t_vec = t.permute(*perm).reshape(-1, *batch_shape)
            
            result["C1.Lu"][..., i] = t_vec
        
        return result

    def __str__(self):
        return (
            f"Kron2SumCov(G=..., dim={self.dim}, rank={self.rank}): {self.name}\n"
            f"  {self._C0}\n"
            f"  {self._C1}"
        )

