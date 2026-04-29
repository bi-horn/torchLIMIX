'''Torch implementation of the following multi-trait glimix-core class:
KronFastScanner (https://github.com/limix/glimix-core/blob/master/glimix_core/lmm/_kron2sum_scan.py)
'''

import torch
from torch import Tensor
from torchlimix.utils._torch_helpers import torch_safe_log, torch_rsolve
from typing import Dict, Optional
import warnings
import tempfile, os
import numpy as np
from torchlimix.torch_cache import versioned_cached_property
from torchlimix.utils.helper_functions import _create_mmap_results, _mmap_results_to_tensors
# Suppress internal PyTorch deprecation warning often triggered by tensordot
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch._prims_common.check.*")
torch.set_default_dtype(torch.float64)
LOG2PI = torch.tensor(1.837877066409345339081937709124758839607238769531250, dtype=torch.float64)
 
class KronFastScannerTorch:
    def __init__(
        self,
        Y: Tensor,
        A: Tensor,
        X: Tensor,
        G: Tensor,
        terms: Dict[str, Tensor],
    ):
        self._Y = Y.double()
        self._A = A.double()
        self._X = X.double()
        self._G = G.double()

        # Precomputed terms
        self._Ge = terms["Ge"]
        self._H = terms["H"]
        self._logdetK = terms["logdetK"]
        self._W = terms["W"]
        self._yKiy = terms["yKiy"]
        self._WA = terms["WA"]
        self._WL0 = terms["WL0"]
        self._Z = terms["Z"]
        self._XRiM = terms["XRiM"]
        self._ZiXRiy = terms["ZiXRiy"]
        self._ZiXRiM = terms["ZiXRiM"]
        self._MRiM = terms["MRiM"]
        self._MRiXZiXRiM = terms["MRiXZiXRiM"] 
        self._MRiy = terms["MRiy"]
        self._MRiXZiXRiy = terms["MRiXZiXRiy"]

    @property
    def _nsamples(self):
        return self._Y.shape[0]

    @property
    def _ntraits(self):
        return self._Y.shape[1]

    @property
    def np(self):
        return self._nsamples * self._ntraits
    
    @property
    def _ncovariates(self):
        return self._X.shape[1]

    @property
    def null_lml(self):
        return self._null_lml

    @versioned_cached_property
    def _null_lml(self):
        scale = self.null_scale
        return self._static_lml / 2 - self.np * torch_safe_log(scale) / 2 - self.np / 2

    @versioned_cached_property
    def null_scale(self):
        beta = self.null_beta.flatten()
        mKiy = torch.dot(beta, self._MKiy.flatten())
        sqrtdot = self._yKiy - mKiy
        scale = sqrtdot / self.np 
        return scale

    @versioned_cached_property
    def null_beta_covariance(self):
        return self.null_scale * torch.linalg.pinv(self._H)

    @versioned_cached_property
    def null_beta(self):
        return torch_rsolve(self._MKiM, self._MKiy) 
    
    @versioned_cached_property
    def null_beta_se(self):
        return torch.sqrt(torch.diag(self.null_beta_covariance))

    @property
    def _df(self):
        np = self._nsamples * self._ntraits
        return np

    @versioned_cached_property
    def _static_lml(self):
        np_val = torch.tensor(self.np, dtype=torch.double)
        logdetK = self._logdetK.to(torch.double)
        return -np_val * LOG2PI - logdetK

    @versioned_cached_property
    def _MKiM(self):
        return self._MRiM - self._XRiM.T @ self._ZiXRiM

    @versioned_cached_property
    def _MKiy(self):
        return self._MRiy - self._XRiM.T @ self._ZiXRiy

    def scan_batched_gpu(
        self,
        A1: Tensor,
        X1: Tensor,
        chunk_size: int = None,
        cache_clear_interval: int = 4,
        progress_callback: callable = None,
    ) -> Dict[str, Tensor]:
        """
        Fully optimized GPU scanner with memory-mapped result storage.
        """
    
        X1 = X1.double().to(self._Y.device)
        A1 = A1.double().to(self._Y.device)
        _, n_snps = X1.shape
        device = X1.device
    
        # Empty A1 guard
        if A1.shape[1] == 0:
            if progress_callback is not None:
                progress_callback(n_snps, n_snps)
    
            beta_se = torch.sqrt(self.null_beta_covariance.diagonal()).cpu()
            return {
                "lml": (self._null_lml.cpu()
                        if isinstance(self._null_lml, Tensor) else self._null_lml),
                "effsizes0": self.null_beta.cpu().reshape(self._ncovariates, -1),
                "effsizes0_se": beta_se.reshape(self._ncovariates, -1),
                "effsizes1": torch.empty(0),
                "effsizes1_se": torch.empty(0),
                "scale": (self.null_scale.cpu()
                        if isinstance(self.null_scale, Tensor) else self.null_scale),
            }
    
        if chunk_size is None:
            chunk_size = min(256, n_snps)
    
        cp = self._ntraits * self._ncovariates
        np_total = self._nsamples * self._ntraits
        yKiy_scalar = float(self._yKiy.item())
        static_lml_scalar = float(self._static_lml.item())
        epsilon_tiny = torch.finfo(torch.float64).tiny
        n_traits = self._ntraits
        n_covariates = self._ncovariates
        n_samples = self._nsamples
        a1_cols = A1.shape[1]
        total_matrix_size = cp + a1_cols
    
        # Pre-compute constant matrices 
        with torch.no_grad():
            AWA1 = self._WA.T @ A1
            A1W = A1.T @ self._W
            A1W_A1 = A1W @ A1
            WL0T_A1 = self._WL0.T @ A1
            Y_A1W_T = self._Y @ A1W.T
    
            MRiM_exp = self._MRiM.unsqueeze(0)
            MRiXZiXRiM_exp = self._MRiXZiXRiM.unsqueeze(0)
            MRiy_exp = self._MRiy.unsqueeze(0)
            MRiXZiXRiy_exp = self._MRiXZiXRiy.unsqueeze(0)
            XRiM_T = self._XRiM.T
    
            try:
                Z_chol = torch.linalg.cholesky(self._Z)
                use_cholesky = True
            except Exception:
                use_cholesky = False
                Z_inv_cached = torch.linalg.pinv(self._Z)
    
        mmaps = _create_mmap_results(
            n_snps, n_covariates, n_traits, a1_cols
        )
    
        # Pre-allocate GPU workspace
        workspace_MKiM = torch.empty(
            chunk_size, total_matrix_size, total_matrix_size,
            device=device, dtype=torch.double,
        )
        workspace_MKiy = torch.empty(
            chunk_size, total_matrix_size, 1,
            device=device, dtype=torch.double,
        )
        workspace_beta = torch.empty(
            chunk_size, total_matrix_size,
            device=device, dtype=torch.double,
        )
        I_se = torch.eye(
            total_matrix_size, device=device, dtype=torch.double,
        ).unsqueeze(0)
    
        # Vectorized Kronecker
        def vectorized_kron(A: Tensor, B_batch: Tensor) -> Tensor:
            batch_size, r, s = B_batch.shape
            p, q = A.shape
            if p == 0 or q == 0 or r == 0 or s == 0:
                return torch.empty(
                    batch_size, p * r, q * s, device=A.device, dtype=A.dtype
                )
            return (torch.einsum('ij,bkl->bikjl', A, B_batch)
                    .reshape(batch_size, p * r, q * s))
    
        with torch.no_grad():
            for chunk_start in range(0, n_snps, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_snps)
                cs = chunk_end - chunk_start
    
                MKiM = workspace_MKiM[:cs]
                MKiy = workspace_MKiy[:cs]
                beta = workspace_beta[:cs]
    
                X1_chunk = X1[:, chunk_start:chunk_end]
    
                geno_var = X1_chunk.var(dim=0)
                X1X1 = torch.sum(X1_chunk ** 2, dim=0)
                XX1 = (self._X.T @ X1_chunk).T.unsqueeze(-1)
                GX1 = (self._G.T @ X1_chunk).T.unsqueeze(-1)
                M1Riy = X1_chunk.T @ Y_A1W_T
    
                MRiM1 = vectorized_kron(AWA1, XX1)
                M1RiM1 = vectorized_kron(A1W_A1, X1X1.view(cs, 1, 1))
                XRiM1 = vectorized_kron(WL0T_A1, GX1)
    
                # Solve Z⁻¹ XRiM1
                if use_cholesky:
                    _b, cd, k = XRiM1.shape
                    XRiM1_flat = XRiM1.permute(1, 0, 2).reshape(cd, _b * k)
                    ZiXRiM1_flat = torch.cholesky_solve(XRiM1_flat, Z_chol)
                    ZiXRiM1 = ZiXRiM1_flat.reshape(cd, _b, k).permute(1, 0, 2)
                else:
                    ZiXRiM1 = Z_inv_cached.unsqueeze(0) @ XRiM1
    
                MRiXZiXRiM1 = XRiM_T.unsqueeze(0) @ ZiXRiM1
                M1RiXZiXRiM1 = XRiM1.transpose(-2, -1) @ ZiXRiM1
                M1RiXZiXRiy = (
                    XRiM1.transpose(-2, -1)
                    @ self._ZiXRiy.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1)
    
                # Build block matrices 
                T0_top = torch.cat(
                    [MRiM_exp.expand(cs, -1, -1), MRiM1], dim=2
                )
                T0_bot = torch.cat(
                    [MRiM1.transpose(1, 2), M1RiM1], dim=2
                )
                T0 = torch.cat([T0_top, T0_bot], dim=1)
    
                T1_top = torch.cat(
                    [MRiXZiXRiM_exp.expand(cs, -1, -1), MRiXZiXRiM1], dim=2
                )
                T1_bot = torch.cat(
                    [MRiXZiXRiM1.transpose(1, 2), M1RiXZiXRiM1], dim=2
                )
                T1 = torch.cat([T1_top, T1_bot], dim=1)
    
                T2 = torch.cat([MRiy_exp.expand(cs, -1), M1Riy], dim=1)
                T3 = torch.cat(
                    [MRiXZiXRiy_exp.expand(cs, -1), M1RiXZiXRiy], dim=1
                )
    
                MKiM[:] = T0 - T1
                MKiy[:] = (T2 - T3).unsqueeze(-1)
    
                # Solve for beta
                try:
                    beta[:] = torch.linalg.solve(MKiM, MKiy.squeeze(-1))
                except Exception:
                    try:
                        beta[:] = (torch.linalg.pinv(MKiM) @ MKiy).squeeze(-1)
                    except Exception:
                        beta[:] = float('nan')
                        if progress_callback:
                            progress_callback(chunk_end, n_snps)
                        continue
    
                mKiy_val = (beta.unsqueeze(-2) @ MKiy).squeeze()
                sqrtdot = yKiy_scalar - mKiy_val
                scale = torch.clamp(sqrtdot / np_total, min=epsilon_tiny)
                lml = (static_lml_scalar / 2
                    - np_total * torch.log(scale) / 2
                    - np_total / 2)
    
                try:
                    I_batch = I_se.expand(cs, -1, -1)
                    MKiM_inv = torch.linalg.solve(MKiM, I_batch)
                    diag_inv = torch.diagonal(MKiM_inv, dim1=-2, dim2=-1)
                except Exception:
                    diag_inv = torch.ones(
                        cs, total_matrix_size, device=device
                    )
    
                se = torch.sqrt(
                    torch.clamp(scale.unsqueeze(-1) * diag_inv, min=epsilon_tiny)
                )
    
                beta1 = beta[:, cp:]
                pve = scale.unsqueeze(-1) * geno_var.unsqueeze(-1) * beta1 ** 2
    
                # Stream to mmap (tiny CPU spike = chunk_size only)
                sl = slice(chunk_start, chunk_end)
                mmaps["lml"][sl] = lml.cpu().numpy()
                mmaps["scale"][sl] = scale.cpu().numpy()
                mmaps["effsizes0"][sl] = (
                    beta[:, :cp]
                    .view(cs, n_traits, n_covariates)
                    .transpose(-2, -1)
                    .cpu().numpy()
                )
                mmaps["effsizes0_se"][sl] = (
                    se[:, :cp]
                    .view(cs, n_traits, n_covariates)
                    .transpose(-2, -1)
                    .cpu().numpy()
                )
                mmaps["effsizes1"][sl] = (
                    beta1.view(cs, a1_cols, 1)
                    .transpose(-2, -1)
                    .cpu().numpy()
                )
                mmaps["effsizes1_se"][sl] = (
                    se[:, cp:]
                    .view(cs, a1_cols, 1)
                    .transpose(-2, -1)
                    .cpu().numpy()
                )
                mmaps["pve"][sl] = pve.cpu().numpy()
    
                if progress_callback:
                    progress_callback(chunk_end, n_snps)
    
                # Periodic GPU cache clear 
                if (chunk_start > 0
                        and chunk_start % (chunk_size * cache_clear_interval) == 0):
                    torch.cuda.empty_cache()
    
        del workspace_MKiM, workspace_MKiy, workspace_beta, I_se
        torch.cuda.empty_cache()

        return _mmap_results_to_tensors(mmaps)
    
    # CPU fallback function if cuda is not available
    def scan_batched_cpu(self, A1: Tensor, X1: Tensor, chunk_size: int = None,
                            debug: bool = False, progress_callback: callable = None) -> Dict[str, Tensor]:
            """
            CPU version with batched operations for speed.
            Aligned with scan_batched_gpu for consistent results.
            """
            assert X1.dim() == 2, "X1 must be of shape (n_samples, n_snps)"

            # Force everything to CPU and double
            X1 = X1.detach().cpu().double().contiguous()
            A1 = A1.detach().cpu().double().contiguous()
            n_samples, n_snps = X1.shape

            if debug:
                print(f"\n=== BATCHED CPU SCAN DEBUG START ===")
                print(f"  X1 shape: {X1.shape}, A1 shape: {A1.shape}")

            if chunk_size is None:
                chunk_size = min(500, n_snps)

            # Handle empty A1 case
            if A1.shape[1] == 0:
                if debug:
                    print("DEBUG: Empty A1 case")
                if progress_callback is not None:
                    progress_callback(n_snps, n_snps)

                beta_se = torch.sqrt(self.null_beta_covariance.diagonal())
                return {
                    "lml": self._null_lml,
                    "effsizes0": self.null_beta.reshape(self._ncovariates, -1),
                    "effsizes0_se": beta_se.reshape(self._ncovariates, -1),
                    "effsizes1": torch.empty(0),
                    "effsizes1_se": torch.empty(0),
                    "scale": self.null_scale,
                }

            # Cache constants
            cp = self._ntraits * self._ncovariates
            np_total = self._nsamples * self._ntraits
            a1_cols = A1.shape[1]
            total_matrix_size = cp + a1_cols
            yKiy_scalar = float(self._yKiy.item())
            static_lml_scalar = float(self._static_lml.item())
            epsilon_tiny = torch.finfo(torch.float64).tiny
            n_traits = self._ntraits
            n_covariates = self._ncovariates

            # Move all class tensors to CPU once
            Y_cpu = self._Y.detach().cpu().double().contiguous()
            X_cpu = self._X.detach().cpu().double().contiguous()
            G_cpu = self._G.detach().cpu().double().contiguous()
            W_cpu = self._W.detach().cpu().double().contiguous()
            WA_cpu = self._WA.detach().cpu().double().contiguous()
            WL0_cpu = self._WL0.detach().cpu().double().contiguous()
            Z_cpu = self._Z.detach().cpu().double().contiguous()
            XRiM_cpu = self._XRiM.detach().cpu().double().contiguous()
            MRiM_cpu = self._MRiM.detach().cpu().double().contiguous()
            MRiy_cpu = self._MRiy.detach().cpu().double().contiguous()
            ZiXRiy_cpu = self._ZiXRiy.detach().cpu().double().contiguous()
            MRiXZiXRiM_cpu = self._MRiXZiXRiM.detach().cpu().double().contiguous()
            MRiXZiXRiy_cpu = self._MRiXZiXRiy.detach().cpu().double().contiguous()

            # Pre-compute constant matrices ONCE
            AWA1 = (WA_cpu.T @ A1).contiguous()
            A1W = (A1.T @ W_cpu).contiguous()
            A1W_A1 = (A1W @ A1).contiguous()
            WL0T_A1 = (WL0_cpu.T @ A1).contiguous()
            Y_A1W_T = (Y_cpu @ A1W.T).contiguous()

            # Pre-expand constant tensors (avoid repeated .expand() / .unsqueeze() in loop)
            MRiM_exp = MRiM_cpu.unsqueeze(0)
            MRiXZiXRiM_exp = MRiXZiXRiM_cpu.unsqueeze(0)
            MRiy_exp = MRiy_cpu.unsqueeze(0)
            MRiXZiXRiy_exp = MRiXZiXRiy_cpu.unsqueeze(0)
            XRiM_T = XRiM_cpu.T

            # Z inverse — pinv is fine for CPU (Cholesky offers less benefit here)
            try:
                Z_inv = torch.linalg.pinv(Z_cpu)
            except Exception:
                Z_inv = torch.eye(Z_cpu.shape[0], dtype=torch.double)

            # Pre-allocate identity for SE computation (solve instead of pinv)
            I_se = torch.eye(total_matrix_size, dtype=torch.double).unsqueeze(0)

            # Pre-allocate workspace tensors (reused across chunks)
            workspace_MKiM = torch.empty(chunk_size, total_matrix_size, total_matrix_size, dtype=torch.double)
            workspace_MKiy = torch.empty(chunk_size, total_matrix_size, 1, dtype=torch.double)
            workspace_beta = torch.empty(chunk_size, total_matrix_size, dtype=torch.double)

            if debug:
                print(f"  Constants computed. Processing {n_snps} SNPs in chunks of {chunk_size}")

            # Result storage
            results_dict = {
                "lml": torch.empty(n_snps, dtype=torch.double),
                "scale": torch.empty(n_snps, dtype=torch.double),
                "effsizes0": torch.empty(n_snps, n_covariates, n_traits, dtype=torch.double),
                "effsizes0_se": torch.empty(n_snps, n_covariates, n_traits, dtype=torch.double),
                "effsizes1": torch.empty(n_snps, 1, a1_cols, dtype=torch.double),
                "effsizes1_se": torch.empty(n_snps, 1, a1_cols, dtype=torch.double),
                "pve": torch.empty(n_snps, a1_cols, dtype=torch.double),
            }

            def batched_kron_cpu(A: Tensor, B_batch: Tensor) -> Tensor:
                """Batched Kronecker product via einsum (no contiguity issues)."""
                batch_size, r, s = B_batch.shape
                p, q = A.shape
                if p == 0 or q == 0 or r == 0 or s == 0:
                    return torch.empty(batch_size, p * r, q * s, dtype=A.dtype)
                return torch.einsum('ij,bkl->bikjl', A, B_batch).reshape(batch_size, p * r, q * s)

            with torch.no_grad():
                for chunk_start in range(0, n_snps, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_snps)
                    cs = chunk_end - chunk_start

                    # Get workspace slices for this chunk
                    MKiM = workspace_MKiM[:cs]
                    MKiy = workspace_MKiy[:cs]
                    beta = workspace_beta[:cs]

                    X1_chunk = X1[:, chunk_start:chunk_end].contiguous()

                    # Genotypic variance for PVE
                    geno_var = X1_chunk.var(dim=0)

                    # Vectorized computation
                    X1X1 = torch.sum(X1_chunk ** 2, dim=0)
                    XX1 = (X_cpu.T @ X1_chunk).T.unsqueeze(-1)
                    GX1 = (G_cpu.T @ X1_chunk).T.unsqueeze(-1)
                    M1Riy = X1_chunk.T @ Y_A1W_T

                    # Batched Kronecker products
                    MRiM1 = batched_kron_cpu(AWA1, XX1)
                    M1RiM1 = batched_kron_cpu(A1W_A1, X1X1.view(cs, 1, 1))
                    XRiM1 = batched_kron_cpu(WL0T_A1, GX1)

                    # Solve Z⁻¹ XRiM1 (batched matmul with pre-computed inverse)
                    ZiXRiM1 = torch.bmm(
                        Z_inv.unsqueeze(0).expand(cs, -1, -1),
                        XRiM1
                    )

                    MRiXZiXRiM1 = XRiM_T.unsqueeze(0) @ ZiXRiM1
                    M1RiXZiXRiM1 = XRiM1.transpose(-2, -1) @ ZiXRiM1
                    M1RiXZiXRiy = (XRiM1.transpose(-2, -1) @ ZiXRiy_cpu.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

                    # Build block matrices
                    T0_top = torch.cat([MRiM_exp.expand(cs, -1, -1), MRiM1], dim=2)
                    T0_bot = torch.cat([MRiM1.transpose(1, 2), M1RiM1], dim=2)
                    T0 = torch.cat([T0_top, T0_bot], dim=1)

                    T1_top = torch.cat([MRiXZiXRiM_exp.expand(cs, -1, -1), MRiXZiXRiM1], dim=2)
                    T1_bot = torch.cat([MRiXZiXRiM1.transpose(1, 2), M1RiXZiXRiM1], dim=2)
                    T1 = torch.cat([T1_top, T1_bot], dim=1)

                    T2 = torch.cat([MRiy_exp.expand(cs, -1), M1Riy], dim=1)
                    T3 = torch.cat([MRiXZiXRiy_exp.expand(cs, -1), M1RiXZiXRiy], dim=1)

                    # Final MKiM and MKiy
                    MKiM[:] = T0 - T1
                    MKiy[:] = (T2 - T3).unsqueeze(-1)

                    # Solve for beta
                    try:
                        beta[:] = torch.linalg.solve(MKiM, MKiy.squeeze(-1))
                    except Exception:
                        try:
                            beta[:] = (torch.linalg.pinv(MKiM) @ MKiy).squeeze(-1)
                        except Exception:
                            beta[:] = float('nan')
                            if progress_callback:
                                progress_callback(chunk_end, n_snps)
                            continue

                    # Scale and likelihood
                    mKiy_val = (beta.unsqueeze(-2) @ MKiy).squeeze()
                    if mKiy_val.dim() == 0:
                        mKiy_val = mKiy_val.unsqueeze(0)
                    sqrtdot = yKiy_scalar - mKiy_val
                    scale = torch.clamp(sqrtdot / np_total, min=epsilon_tiny)
                    lml = static_lml_scalar / 2 - np_total * torch.log(scale) / 2 - np_total / 2

                    # Standard errors via solve (matches GPU, avoids SVD in pinv)
                    try:
                        I_batch = I_se.expand(cs, -1, -1)
                        MKiM_inv = torch.linalg.solve(MKiM, I_batch)
                        diag_inv = torch.diagonal(MKiM_inv, dim1=-2, dim2=-1)
                    except Exception:
                        diag_inv = torch.ones(cs, total_matrix_size, dtype=torch.double)

                    se = torch.sqrt(torch.clamp(scale.unsqueeze(-1) * diag_inv, min=epsilon_tiny))

                    # PVE
                    beta1 = beta[:, cp:]
                    pve = geno_var.unsqueeze(-1) * beta1 ** 2

                    # Store results
                    results_dict["lml"][chunk_start:chunk_end] = lml
                    results_dict["scale"][chunk_start:chunk_end] = scale
                    results_dict["effsizes0"][chunk_start:chunk_end] = beta[:, :cp].view(cs, n_traits, n_covariates).transpose(-2, -1)
                    results_dict["effsizes0_se"][chunk_start:chunk_end] = se[:, :cp].view(cs, n_traits, n_covariates).transpose(-2, -1)
                    results_dict["effsizes1"][chunk_start:chunk_end] = beta1.view(cs, a1_cols, 1).transpose(-2, -1)
                    results_dict["effsizes1_se"][chunk_start:chunk_end] = se[:, cp:].view(cs, a1_cols, 1).transpose(-2, -1)
                    results_dict["pve"][chunk_start:chunk_end] = pve

                    if progress_callback is not None:
                        progress_callback(chunk_end, n_snps)

            return results_dict