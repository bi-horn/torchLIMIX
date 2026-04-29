'''Torch implementation of the following multi-trait glimix-core class:
Kron2Sum (https://github.com/limix/glimix-core/blob/master/glimix_core/lmm/_kron2sum.py)
'''
import numpy as np
import torch
import warnings
from typing import Dict, Optional, Any
from torchlimix.utils._torch_helpers import torch_is_all_finite, torch_lu_slogdet, torch_vec, torch_unvec, torch_safe_logdet, torch_lu_solve, torch_sum2diag, torch_trace, torch_mdot, torch_mkron, torch_ddot, torch_lu_factor
from torchlimix.optimizer._optimizer import TorchFunction
from torchlimix.torch_cache import (
    VersionedCacheMixin,
    HierarchicalVersionTracker,
    versioned_cached_property
)
from torchlimix.lmm._kron_mean import KronMeanTorch
from torchlimix.lmm._kron_cov import Kron2SumCovTorch
from torchlimix.lmm._kron_scanner import KronFastScannerTorch

# Suppress internal PyTorch deprecation warning often triggered by tensordot
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch._prims_common.check.*")
torch.set_default_dtype(torch.float64)
LOG2PI = torch.tensor(1.837877066409345339081937709124758839607238769531250, dtype=torch.float64)

class Kron2SumTorch(TorchFunction, VersionedCacheMixin):
    """
    PyTorch implementation of Kron2Sum LMM for multi-trait analysis.
    """
    
    def __init__(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        X: torch.Tensor,
        G: torch.Tensor,
        data_meta: Optional[dict] = None,
        rank: int = 0,
        device: str = 'cuda',
        restricted: bool = False,
        config: dict = None
    ):
        TorchFunction.__init__(self, "Kron2Sum")

        self.device = device
        self._restricted = restricted
        
        self._Y = Y.double().to(device)
        self.A = A.double().to(device)
        self.X = X.double().to(device)
        self.G = G.double().to(device)
        self.config = config if config is not None else {}
            
        self._validate_inputs()

        # Extract normalization parameters
        if data_meta is not None and 'G_norm' in data_meta and 'G_scaling' in data_meta:
            self.G_norm = data_meta['G_norm']
            self.C0_scale_factor = data_meta['G_scaling']['scale_to_original_C0']
        else: 
            raise ValueError(
                'G_norm or G_scaling not found in data_meta. '
                'Ensure get_data_multitask_snp properly populates these fields.'
            )

        original_L_init = config.get('original_L_init', False)
        # Create covariance component
        self._cov = Kron2SumCovTorch(
            G=self.G,
            dim=Y.shape[1],
            rank=rank,
            original_L_init = original_L_init,
            device=self.device
        )
        if hasattr(self._cov, 'to'):
            self._cov = self._cov.double()

        # Create mean component
        self._mean = KronMeanTorch(self.A, self.X, device=self.device)

        # This tracks C0.Lu and C1.Lu through the hierarchy
        self._version_tracker = HierarchicalVersionTracker()
        self._version_tracker.track_child(self._cov._version_tracker)
        self._cache: dict = {}
        self._versioned_property_cache: dict = {} 
        
        # Optimizer state
        self._flat_gradient: Optional[np.ndarray] = None
        self._variable_registry: Dict[str, tuple] = {}
        
        # Register variables for optimization
        self._variable_registry["C0.Lu"] = ("Lu", self._cov._C0)
        self._variable_registry["C1.Lu"] = ("Lu", self._cov._C1)
        
        # Warn if too many parameters
        nparams = self._mean.nparams + self._cov.nparams
        if nparams > Y.numel():
            warnings.warn(
                f"The number of parameters ({nparams}) exceeds the outcome size. "
                f"Convergence might be problematic.",
                UserWarning,
            )

    def clear_cache(self):
        """Clear all cached computations - REQUIRED for optimizer."""
        # Clear own caches
        self._cache.clear()
        self._versioned_property_cache.clear()
        
        # Clear covariance caches
        if hasattr(self._cov, '_cache'):
            self._cov._cache.clear()
        if hasattr(self._cov, '_versioned_property_cache'):
            self._cov._versioned_property_cache.clear()
        
        # Clear child covariance component caches
        if hasattr(self._cov._C0, '_cache'):
            self._cov._C0._cache.clear()
        if hasattr(self._cov._C0, '_versioned_property_cache'):
            self._cov._C0._versioned_property_cache.clear()
        if hasattr(self._cov._C1, '_cache'):
            self._cov._C1._cache.clear()
        if hasattr(self._cov._C1, '_versioned_property_cache'):
            self._cov._C1._versioned_property_cache.clear()
            
    def _validate_inputs(self):
        """Validate input matrices."""
        Y = self._Y
        
        # Check Y rank
        try:
            yrank = torch.linalg.matrix_rank(Y).item()
            if Y.shape[1] > yrank:
                warnings.warn(
                    f"Y is not full column rank: rank(Y)={yrank}. "
                    f"Convergence might be problematic.",
                    UserWarning,
                )
        except Exception as e:
            warnings.warn(f"Could not compute matrix rank of Y: {e}", UserWarning)
        
        # Check X rank
        try:
            xrank = torch.linalg.matrix_rank(self.X).item()
            if self.X.shape[1] > xrank:
                warnings.warn(
                    f"X is not full column rank: rank(X)={xrank}. "
                    f"Convergence might be problematic.",
                    UserWarning,
                )
        except Exception as e:
            warnings.warn(f"Could not compute matrix rank of X: {e}", UserWarning)
        
        # Check for non-finite values
        if not torch_is_all_finite(self._Y):
            raise ValueError("There are non-finite values in the outcome matrix.")
        if not torch_is_all_finite(self.A):
            raise ValueError("There are non-finite values in the trait-by-trait design matrix.")
        if not torch_is_all_finite(self.X):
            raise ValueError("There are non-finite values in the covariates matrix.")
        if not torch_is_all_finite(self.G):
            raise ValueError("There are non-finite values in the G matrix.")
    
    @property
    def _nsamples(self) -> int:
        return self._Y.shape[0]
    
    @property
    def _ntraits(self) -> int:
        return self._Y.shape[1]
    
    @property
    def _ncovariates(self) -> int:
        return self.X.shape[1]
    
    @property
    def C0(self) -> torch.Tensor:
        return self._cov.C0.value() * self.C0_scale_factor
    
    @property
    def C1(self) -> torch.Tensor:
        return self._cov.C1.value()

    @property
    def C0_trace(self):
        return torch_trace(self.C0)
    
    @property
    def C1_trace(self): 
        return torch_trace(self.C1)  
        
    @property
    def B(self) -> torch.Tensor:
        return self._mean.B.detach()
    
    @property
    def beta(self) -> torch.Tensor:
        return torch_vec(self.B)
    
    @property
    def beta_covariance(self) -> torch.Tensor:
        terms = self._terms
        return torch.linalg.inv(terms["H"])
    
    @property
    def _df(self):
        np = self._nsamples * self._ntraits
        if not self._restricted:
            return np
        cp = self._ncovariates * self._ntraits
        return np - cp
    
    @property
    def Ge(self) -> torch.Tensor:
        return self._cov.Ge

    @versioned_cached_property
    def _XX(self) -> torch.Tensor:
        return self._mean.X.T @ self._mean.X

    @versioned_cached_property
    def _GY(self) -> torch.Tensor:
        return self.Ge.T @ self._Y

    @versioned_cached_property
    def _GG(self) -> torch.Tensor:
        return self.Ge.T @ self.Ge

    @versioned_cached_property
    def _trGG(self) -> torch.Tensor:
        Ge = self.Ge
        return torch_trace(Ge @ Ge.T)

    @versioned_cached_property
    def _GX(self) -> torch.Tensor:
        return self.Ge.T @ self._mean.X

    @versioned_cached_property
    def _XY(self) -> torch.Tensor:
        return self._mean.X.T @ self._Y

    @versioned_cached_property
    def _XGGY(self) -> torch.Tensor:
        return self._GX.T @ self._GY

    @versioned_cached_property
    def _XGGX(self) -> torch.Tensor:
        return self._GX.T @ self._GX

    @versioned_cached_property
    def _GGGY(self) -> torch.Tensor:
        return self._GG @ self._GY

    @versioned_cached_property
    def _GGGG(self) -> torch.Tensor:
        return self._GG @ self._GG

    @versioned_cached_property
    def _XGGG(self) -> torch.Tensor:
        return self._GX.T @ self._GG

    @versioned_cached_property
    def _logdetH(self) -> float:
        if not self._restricted:
            return 0.0
        return torch_safe_logdet(self._terms["H"])

    @versioned_cached_property
    def _logdet_MM(self) -> float:
        if not self._restricted:
            return 0.0
        M = self._mean.AX
        return torch_safe_logdet(M.T @ M)
 
    def value(self) -> torch.Tensor:
        """Log of the marginal likelihood."""
        return self.lml()
    
    def gradient(self) -> Dict[str, torch.Tensor]:
        """Gradient of the log of the marginal likelihood."""
        return self._lml_gradient()

    @torch.no_grad()
    def lml(self) -> torch.Tensor:
        """Compute log marginal likelihood."""
        terms = self._terms
        
        yKiy = terms["yKiy"]
        mKiy = terms["mKiy"]
        mKim = terms["mKim"]
        
        lml = -self._df * LOG2PI + self._logdet_MM - self._logdetK  
        lml -= self._logdetH 
        lml += -yKiy - mKim + 2 * mKiy
        
        return (lml / 2)

    @property
    def _logdetK(self) -> torch.Tensor:
        """Log determinant of K, matching NumPy implementation."""
        terms = self._terms
        return terms["logdetK"]
    
    @torch.no_grad()
    def _lml_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Compute gradient of log-marginal likelihood w.r.t. covariance parameters.

        Memory optimization: Kronecker products are computed one parameter at a
        time inside a loop, so at most ~3 (chol_dim × chol_dim) matrices exist
        simultaneously, instead of 6 × n_params of them.
        """

        terms = self._terms
        dC0 = self._cov.C0.gradient()["Lu"]  # (p, p, n_params_C0)
        dC1 = self._cov.C1.gradient()["Lu"]  # (p, p, n_params_C1)

        b = terms["b"]
        W = terms["W"]
        Lh, pivH = terms["Lh"], terms["pivH"]
        Lz, pivZ = terms["Lz"], terms["pivZ"]
        WA = terms["WA"]
        WL0 = terms["WL0"]
        YW = terms["YW"]
        MRiM = terms["MRiM"]
        MRiy = terms["MRiy"]
        XRiM = terms["XRiM"]
        XRiy = terms["XRiy"]
        ZiXRiM = terms["ZiXRiM"]
        ZiXRiy = terms["ZiXRiy"]

        # Parameter-independent intermediates
        XRim = torch_mdot(XRiM, b)
        MRim = torch_mdot(MRiM, b)
        mKiM = MRim - torch_mdot(XRim, ZiXRiM)
        yKiM = MRiy - torch_mdot(XRiy, ZiXRiM)

        GY_W = self._GY @ W
        GY_W_vec = torch_vec(GY_W)
        YW_T = YW.T

        n_params_C0 = dC0.shape[-1]
        n_params_C1 = dC1.shape[-1]

        device = b.device
        dtype = b.dtype

        grad_C0 = torch.zeros(n_params_C0, device=device, dtype=dtype)
        grad_C1 = torch.zeros(n_params_C1, device=device, dtype=dtype)

        # C0 gradient: loop over parameters 

        for i in range(n_params_C0):
            dC0_i = dC0[:, :, i]  # (p, p)

            # Small (p × p) intermediates
            WdC0_i = W @ dC0_i
            AWdC0_i = WA.T @ dC0_i
            AWdC0_WA_i = AWdC0_i @ WA
            AWdC0_WL0_i = AWdC0_i @ WL0
            WdC0_WA_i = WdC0_i @ WA
            WL0T_dC0_WL0_i = WL0.T @ dC0_i @ WL0

            # Kronecker products (one param at a time) 
            MR0M_i = torch.kron(AWdC0_WA_i, self._XGGX)
            MR0X_i = torch.kron(AWdC0_WL0_i, self._XGGG)
            XR0X_i = torch.kron(WL0T_dC0_WL0_i, self._GGGG)

            # Non-Kronecker per-param terms 
            MR0y_i = torch_vec(torch_mdot(self._XGGY, WdC0_WA_i))
            XR0y_i = torch_vec(torch_mdot(self._GGGY, WdC0_i, WL0))

            GY_WdC0_i_vec = torch_vec(self._GY @ WdC0_i)
            yR0y_i = GY_WdC0_i_vec @ GY_W_vec

            # LU solves 
            ZiXR0X_i = torch_lu_solve((Lz, pivZ), XR0X_i)
            ZiXR0y_i = torch_lu_solve((Lz, pivZ), XR0y_i)

            # Trace 
            trace_WdC0_i = torch.trace(WdC0_i)
            trace_ZiXR0X_i = torch.trace(ZiXR0X_i)

            # MK0y_i 
            MK0y_i = (MR0y_i
                      - torch_mdot(XRiM.T, ZiXR0y_i)
                      - torch_mdot(MR0X_i, ZiXRiy)
                      + torch_mdot(XRiM.T, ZiXR0X_i, ZiXRiy))

            # yK0y_i (scalar) 
            yK0y_i = (yR0y_i
                      - 2 * (XR0y_i @ ZiXRiy)
                      + ZiXRiy @ torch_mdot(XR0X_i, ZiXRiy))

            # MK0M_i 
            MR0X_ZiXRiM_i = torch_mdot(MR0X_i, ZiXRiM)
            MK0M_i = (MR0M_i
                      - MR0X_ZiXRiM_i
                      - MR0X_ZiXRiM_i.T
                      + torch_mdot(ZiXRiM.T, XR0X_i, ZiXRiM))

            # Free Kronecker products
            del MR0M_i, MR0X_i, XR0X_i, ZiXR0X_i, MR0X_ZiXRiM_i

            # Remaining terms
            MK0m_i = torch_mdot(MK0M_i, b)
            mK0y_i = b @ MK0y_i
            mK0m_i = b @ torch_mdot(MK0M_i, b)

            db_C0_i = torch_lu_solve((Lh, pivH), MK0m_i - MK0y_i)

            mKiM_db_C0_i = mKiM @ db_C0_i
            yKiM_db_C0_i = yKiM @ db_C0_i

            c0_i = (yK0y_i - 2 * mK0y_i + mK0m_i
                    - 2 * mKiM_db_C0_i + 2 * yKiM_db_C0_i)

            # Accumulate gradient 
            grad_C0[i] = -trace_WdC0_i * self._trGG + trace_ZiXR0X_i

            if self._restricted:
                HiMK0M_i = torch_lu_solve((Lh, pivH), MK0M_i)
                grad_C0[i] = grad_C0[i] + torch.trace(HiMK0M_i)
                del HiMK0M_i

            grad_C0[i] = (grad_C0[i] + c0_i) / 2

            del MK0M_i, MK0y_i, ZiXR0y_i, db_C0_i

        # C1 gradient: loop over parameters 

        for i in range(n_params_C1):
            dC1_i = dC1[:, :, i]  # (p, p)

            # Small (p × p) intermediates
            WdC1_i = W @ dC1_i
            AWdC1_i = WA.T @ dC1_i
            AWdC1_WA_i = AWdC1_i @ WA
            AWdC1_WL0_i = AWdC1_i @ WL0
            WdC1_WA_i = WdC1_i @ WA
            WL0T_dC1_WL0_i = WL0.T @ dC1_i @ WL0

            # Kronecker products (one param at a time) 
            MR1M_i = torch.kron(AWdC1_WA_i, self._XX)
            MR1X_i = torch.kron(AWdC1_WL0_i, self._GX.T.contiguous())
            XR1X_i = torch.kron(WL0T_dC1_WL0_i, self._GG)

            # Non-Kronecker per-param terms
            MR1y_i = torch_vec(torch_mdot(self._XY, WdC1_WA_i))
            XR1y_i = torch_vec(torch_mdot(self._GY, WdC1_i, WL0))

            Y_WdC1_i = self._Y @ WdC1_i
            yR1y_i = (YW_T * Y_WdC1_i.T).sum()

            # LU solves 
            ZiXR1X_i = torch_lu_solve((Lz, pivZ), XR1X_i)
            ZiXR1y_i = torch_lu_solve((Lz, pivZ), XR1y_i)

            # Trace
            trace_WdC1_i = torch.trace(WdC1_i)
            trace_ZiXR1X_i = torch.trace(ZiXR1X_i)

            # MK1y_i
            MK1y_i = (MR1y_i
                      - torch_mdot(XRiM.T, ZiXR1y_i)
                      - torch_mdot(MR1X_i, ZiXRiy)
                      + torch_mdot(XRiM.T, ZiXR1X_i, ZiXRiy))

            # yK1y_i (scalar) 
            yK1y_i = (yR1y_i
                      - 2 * (XR1y_i @ ZiXRiy)
                      + ZiXRiy @ torch_mdot(XR1X_i, ZiXRiy))

            # MK1M_i
            MR1X_ZiXRiM_i = torch_mdot(MR1X_i, ZiXRiM)
            MK1M_i = (MR1M_i
                      - MR1X_ZiXRiM_i
                      - MR1X_ZiXRiM_i.T
                      + torch_mdot(ZiXRiM.T, XR1X_i, ZiXRiM))

            # Free Kronecker products
            del MR1M_i, MR1X_i, XR1X_i, ZiXR1X_i, MR1X_ZiXRiM_i

            # Remaining terms 
            MK1m_i = torch_mdot(MK1M_i, b)
            mK1y_i = b @ MK1y_i
            mK1m_i = b @ torch_mdot(MK1M_i, b)

            db_C1_i = torch_lu_solve((Lh, pivH), MK1m_i - MK1y_i)

            mKiM_db_C1_i = mKiM @ db_C1_i
            yKiM_db_C1_i = yKiM @ db_C1_i

            c1_i = (yK1y_i - 2 * mK1y_i + mK1m_i
                    - 2 * mKiM_db_C1_i + 2 * yKiM_db_C1_i)

            # Accumulate gradient
            grad_C1[i] = -trace_WdC1_i * self._nsamples + trace_ZiXR1X_i

            if self._restricted:
                HiMK1M_i = torch_lu_solve((Lh, pivH), MK1M_i)
                grad_C1[i] = grad_C1[i] + torch.trace(HiMK1M_i)
                del HiMK1M_i

            grad_C1[i] = (grad_C1[i] + c1_i) / 2

            del MK1M_i, MK1y_i, ZiXR1y_i, db_C1_i

        return {"C0.Lu": grad_C0, "C1.Lu": grad_C1}
      
    @property
    def _terms(self) -> Dict[str, Any]:
        """
        Compute terms for mixed model likelihood optimization.
        
        Automatically cached and invalidated when parameters change.
        """
        return self._get_cached("terms", self._compute_terms)
    
    @torch.no_grad()
    def _compute_terms(self):
        """Compute terms for mixed model likelihood optimization"""

        A, Y = self._mean.A, self._Y
        S, U = self._cov.C1.eigh()

        W = torch_ddot(U, 1.0 / S) @ U.T
        S = 1.0 / torch.sqrt(S)

        YW = Y @ W
        WA = W @ A

        # Kronecker products for ML/REML
        MRiM = torch.kron(A.T @ WA, self._XX)
        MRiy = torch_vec(self._XY @ WA)
        yRiy = (YW * Y).sum()

        L0 = self._cov.C0.L
        WL0 = W @ L0
        L0WA = L0.T @ WA

        # Compute Z matrix and regularize
        Z = torch.kron(L0.T @ WL0, self._GG)
        XRiM = torch.kron(L0WA, self._GX)
        XRiy = torch_vec(self._GY @ WL0)
        Z = torch_sum2diag(Z, 1.0) # Regularization

        # LU factorization
        Lz, pivZ = torch_lu_factor(Z)

        # Solve linear systems
        ZiXRiM = torch_lu_solve((Lz, pivZ), XRiM)
        ZiXRiy = torch_lu_solve((Lz, pivZ), XRiy)

        # Compute log determinant from LU factorization
        sign_z, logdet_z = torch_lu_slogdet((Lz, pivZ))  
        logdetK = logdet_z - 2 * torch.log(S).sum() * self._nsamples

        # Woodbury matrix identity terms
        MRiXZiXRiM = XRiM.T @ ZiXRiM
        MRiXZiXRiy = ZiXRiM.T @ XRiy
        H = MRiM - MRiXZiXRiM
        MKiy = MRiy - MRiXZiXRiy
        yKiy = yRiy - XRiy @ ZiXRiy

        # Solve for regression coefficients
        Lh, pivH = torch_lu_factor(H)
        b = torch_lu_solve((Lh, pivH), MKiy.flatten())
        B = torch_unvec(b, (self._ncovariates, -1))
        self._mean.B = B.detach().clone()

        # Final derived terms
        XRim = XRiM @ b
        ZiXRim = ZiXRiM @ b
        mRiy = b @ MRiy      
        mRim = b @ MRiM @ b  

        mKiy = mRiy - torch.dot(XRim, ZiXRiy)  
        mKim = mRim - torch.dot(XRim, ZiXRim)  

        terms = {
            "Ge": self.Ge,
            "logdetK": logdetK,
            "MKiy": MKiy,
            "mKiy": mKiy,
            "mKim": mKim,
            "A": A,
            "b": b,
            "B": B,
            "Z": Z,
            "Lz": Lz,
            "pivZ": pivZ,
            "S": S,
            "W": W,
            "WA": WA,
            "YW": YW,
            "WL0": WL0,
            "yRiy": yRiy,
            "MRiM": MRiM,
            "XRiy": XRiy,
            "XRiM": XRiM,
            "ZiXRiM": ZiXRiM,
            "ZiXRiy": ZiXRiy,
            "ZiXRim": ZiXRim,
            "MRiy": MRiy,
            "mRim": mRim,
            "mRiy": mRiy,
            "XRim": XRim,
            "yKiy": yKiy,
            "H": H,
            "Lh": Lh,
            "pivH": pivH,
            "MRiXZiXRiy": MRiXZiXRiy,
            "MRiXZiXRiM": MRiXZiXRiM,
            "_mode": "REML" if self._restricted else "ML"
        }
        return terms 

    def mean(self) -> torch.Tensor:
        """Mean 𝐦 = (A ⊗ X) torch_vec(B)."""
        return self._mean.value()
    
    def covariance(self) -> torch.Tensor:
        """Covariance K = C₀ ⊗ GGᵀ + C₁ ⊗ I."""
        return self._cov.value()

    def get_fast_scanner(self, compile_mode="reduce_overhead"):
        """Return KronFastScannerTorch for efficient association testing."""
        terms = self._terms
        return KronFastScannerTorch(
            Y=self._Y,
            A=self._mean.A,
            X=self._mean.X,
            G=self._cov.Ge,
            terms=terms,
        )

    def fit(self) -> Dict[str, float]:
        """
        Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool, optional
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.

        Returns
        -------
        dict
            'lml': final log marginal likelihood
            'grad_norm': final gradient norm
            'iterations': number of optimization iterations
        """

        try:
            # Run optimization
            self._maximize()

            return {
                'lml': self.value().item(),
                'grad_norm': self._final_grad_norm,
                'iterations': self._n_iterations
            }

        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'lml': float('nan'),
                'grad_norm': float('inf'),
                'iterations': 0
            }
        
    @property
    def final_gradient_norm(self) -> float:
        if self._flat_gradient is not None:
            return float(np.linalg.norm(self._flat_gradient))
        return float('inf')

    @property  
    def final_lml(self) -> float:
        return self.value().item()