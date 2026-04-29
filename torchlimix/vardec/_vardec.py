import numpy as np
import warnings
from typing import Dict, Optional


class VarDecMultiTrait:
    """
    Variance decomposition for multi-trait genetic covariance.

    Block model (overall):
        shared        = P × mean(off-diagonal of C0)
        heterogeneity = max(0, trace(C0) - shared)
        noise         = trace(C1)

    Per-trait:
        genetic_j = C0(j,j)
        noise_j   = C1(j,j)
        h²_j      = C0(j,j) / (C0(j,j) + C1(j,j))

    Genetic correlations:
        rg(i,j) = C0(i,j) / sqrt(C0(i,i) × C0(j,j))
    """

    _EPS = 1e-10
    _W = 62  # column width for formatted output

    def __init__(self, C0, C1, trait_names=None, nsamples=None):
        """
        Parameters
        ----------
        C0 : array-like or torch.Tensor, shape (P, P)
            Genetic trait covariance matrix.
        C1 : array-like or torch.Tensor, shape (P, P)
            Residual/noise trait covariance matrix.
        trait_names : list of str, optional
            Human-readable labels for traits (defaults to trait_0 .. trait_{P-1}).
        nsamples : int, optional
            Number of samples (stored for reference / downstream use).
        """
        self.C0 = self._to_numpy(C0)
        self.C1 = self._to_numpy(C1)
        self.P = self.C0.shape[0]
        self.nsamples = nsamples
        self.trait_names = trait_names or [f"trait_{j}" for j in range(self.P)]
        self.failed = False

        self._overall: Optional[Dict] = None
        self._per_trait: Optional[Dict] = None
        self._rg: Optional[Dict] = None

        try:
            self._decompose()
        except Exception as e:
            warnings.warn(f"VarDecMultiTrait: decomposition failed – {e}")
            self.failed = True

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """Convert torch tensor or array-like to numpy."""
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    @staticmethod
    def _safe_divide(num, den, eps=1e-10):
        return np.where(np.abs(den) > eps, num / den, 0.0)

    def _pct(self, value, total):
        return 100.0 * value / total if total > self._EPS else 0.0
    
    def _decompose(self):
        self._decompose_overall()
        self._decompose_per_trait()
        self._compute_genetic_correlations()

    def _decompose_overall(self):
        """Block model: shared / heterogeneity / noise."""
        P = self.P
        mask = ~np.eye(P, dtype=bool)
        mean_offdiag = np.mean(self.C0[mask])

        var_shared = P * mean_offdiag
        var_het = max(0.0, np.trace(self.C0) - var_shared)
        var_noise = np.trace(self.C1)
        var_total = var_shared + var_het + var_noise

        self._overall = {
            "var_shared": var_shared,
            "var_het": var_het,
            "var_noise": var_noise,
            "var_total": var_total,
            "pct_shared": self._pct(var_shared, var_total),
            "pct_het": self._pct(var_het, var_total),
            "pct_noise": self._pct(var_noise, var_total),
            "trace_C0": np.trace(self.C0),
            "trace_C1": np.trace(self.C1),
            "mean_offdiag_C0": mean_offdiag,
        }

    def _decompose_per_trait(self):
        """Per-trait genetic variance, noise variance, and heritability."""
        diag_C0 = np.diag(self.C0)
        diag_C1 = np.diag(self.C1)
        total = diag_C0 + diag_C1
        h2 = self._safe_divide(diag_C0, total)

        self._per_trait = {
            "genetic": diag_C0.copy(),
            "noise": diag_C1.copy(),
            "total": total.copy(),
            "h2": h2,
            "h2_mean": np.mean(h2),
            "h2_median": np.median(h2),
            "h2_min": np.min(h2),
            "h2_max": np.max(h2),
        }

    def _compute_genetic_correlations(self):
        """rg(i,j) = C0(i,j) / sqrt(C0(i,i) * C0(j,j))"""
        diag = np.diag(self.C0)

        if not np.all(diag > self._EPS):
            self._rg = {
                "matrix": np.full((self.P, self.P), np.nan),
                "offdiag": np.array([]),
                "mean": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
            return

        denom = np.sqrt(np.outer(diag, diag))
        rg = self.C0 / denom
        triu = np.triu_indices(self.P, k=1)
        offdiag = rg[triu]

        self._rg = {
            "matrix": rg,
            "offdiag": offdiag,
            "mean": np.mean(offdiag),
            "median": np.median(offdiag),
            "min": np.min(offdiag),
            "max": np.max(offdiag),
        }

    def get_results(self, verbose=False) -> Dict:
        """
        Full decomposition: overall block model + per-trait + genetic correlations.

        Returns dict with keys: 'overall', 'per_trait', 'rg', 'failed'.
        """
        if self.failed:
            return {"failed": True}

        results = {
            "overall": dict(self._overall),
            "per_trait": {k: (v.copy() if isinstance(v, np.ndarray) else v)
                         for k, v in self._per_trait.items()},
            "rg": {k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in self._rg.items()},
            "failed": False,
        }

        return results

    def get_overall(self) -> Dict:
        """Block model variance components only."""
        if self.failed:
            return {"failed": True}
        return dict(self._overall)

    def get_per_trait(self) -> Dict:
        """Per-trait genetic/noise variance and heritability."""
        if self.failed:
            return {"failed": True}
        return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self._per_trait.items()}

    def get_genetic_correlations(self) -> Dict:
        """Genetic correlation matrix and summary statistics."""
        if self.failed:
            return {"failed": True}
        return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self._rg.items()}