import torch
from torch import Tensor
from typing import Dict, Any, Optional, List
import logging
import json
import os
import gc
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictionResultStore:
    """
    Store and manage prediction results for two scenarios:

    A) **Internal test split** – ground-truth phenotypes are available.
       Saves: predictions CSV, metrics JSON, summary JSON, tensor archive.

    B) **External prediction** – only new genotypes, no ground truth.
       Saves: predictions CSV only (with pred_mean and pred_std).

    Folder structure
    ----------------
    - Real data:       ``{output_dir}/``
    - Simulation:      ``{output_dir}/{sim_folder}/rep{rep_idx:04d}/``
    """

    def __init__(
        self,
        output_dir: str,
        uid: str,
        method: str,
        n_traits: int,
        rep_idx: Optional[int] = None,
        eta: Optional[float] = None,
        corr_bounds: Optional[int] = None,
        use_heterogeneity: bool = False,
        simulation_info: Optional[Dict[str, Any]] = None,
        correction_metadata: Optional[Dict[str, Any]] = None,
        phenotype_data: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.uid = uid
        self.method = method
        self.n_traits = n_traits
        self.rep_idx = rep_idx
        self.eta = eta
        self.corr_bounds = corr_bounds
        self.use_heterogeneity = use_heterogeneity
        self.simulation_info = simulation_info or {}
        self.correction_metadata = correction_metadata or {}
        self.phenotype_data = phenotype_data or {}

        self.predictions: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.confidence_intervals: Dict[str, Any] = {}
        self._tensors: Dict[str, np.ndarray] = {}
        self._has_ground_truth: bool = False

        # Setup directory structure
        self.base_dir = self._setup_directories()

        logger.info(f"PredictionResultStore initialized: {self.base_dir}")

        # Persist preprocessing provenance immediately
        self._save_data_preprocessing_info()
        self._save_phenotype_data()

    def _setup_directories(self) -> str:
        if self.rep_idx is not None:
            if self.use_heterogeneity and self.corr_bounds is not None:
                sim_folder = f"corr{self.corr_bounds:01d}"
            elif self.eta is not None:
                eta_str = f"{self.eta:.2f}"
                sim_folder = f"eta{eta_str}"
            else:
                sim_folder = "simulation"
            rep_folder = f"rep{self.rep_idx:04d}"
            base_dir = os.path.join(self.output_dir, sim_folder, rep_folder)
        else:
            base_dir = self.output_dir

        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    def _save_data_preprocessing_info(self):
        preprocessing_path = os.path.join(self.base_dir, "data_preprocessing.json")

        info: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'uid': self.uid,
            'data_type': 'simulated' if self.eta is not None else 'real',
        }

        if self.eta is not None:
            info['simulation_params'] = {
                'eta': self.eta,
                'rep_idx': self.rep_idx,
            }

        if self.correction_metadata:
            info['corrections'] = {
                'batch_correction': self.correction_metadata.get('batch_correction', {}),
                'covariate_correction': self.correction_metadata.get('covariate_correction', {}),
                'transformations': self.correction_metadata.get('transformations', {}),
            }

        corrections_applied = self.phenotype_data.get('corrections_applied', False)
        if corrections_applied:
            info['phenotype_files'] = {
                'corrected': 'phenotypes_corrected.csv',
                'uncorrected': 'phenotypes_uncorrected.csv',
                'comparison': 'phenotype_comparison.txt',
            }
        else:
            info['phenotype_files'] = {'phenotypes': 'phenotypes.csv'}

        with open(preprocessing_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved data preprocessing info to: {preprocessing_path}")

    def _save_phenotype_data(self):
        corrected = self.phenotype_data.get('corrected')
        uncorrected = self.phenotype_data.get('uncorrected')
        applied = self.phenotype_data.get('corrections_applied', False)

        if corrected is None:
            return

        col_names = None
        if uncorrected is not None and hasattr(uncorrected, 'columns'):
            col_names = uncorrected.columns

        corrected = self._ensure_dataframe(corrected, col_names)

        if uncorrected is not None:
            uncorrected = self._ensure_dataframe(uncorrected, col_names)

        if applied and uncorrected is not None:
            cp = os.path.join(self.base_dir, "phenotypes_corrected.csv")
            up = os.path.join(self.base_dir, "phenotypes_uncorrected.csv")
            corrected.to_csv(cp, index=False)
            uncorrected.to_csv(up, index=False)
            logger.info(f"Saved corrected: {cp}, uncorrected: {up}")
            self._save_phenotype_comparison(uncorrected, corrected)
        else:
            pp = os.path.join(self.base_dir, "phenotypes.csv")
            corrected.to_csv(pp, index=False)
            logger.info(f"Saved phenotypes to: {pp}")

    @staticmethod
    def _ensure_dataframe(obj, columns=None) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, 'detach'):
            arr = obj.detach().cpu().numpy()
        elif hasattr(obj, 'numpy'):
            arr = obj.numpy()
        else:
            arr = obj
        return pd.DataFrame(arr, columns=columns)

    def _save_phenotype_comparison(
        self, df_before: pd.DataFrame, df_after: pd.DataFrame
    ):
        path = os.path.join(self.base_dir, "phenotype_comparison.txt")
        with open(path, 'w') as f:
            f.write("=" * 70 + "\nPHENOTYPE CORRECTION COMPARISON\n" + "=" * 70 + "\n\n")
            for label, df in [("BEFORE CORRECTIONS", df_before),
                              ("AFTER CORRECTIONS", df_after)]:
                f.write(f"{label}:\n" + "-" * 70 + "\n")
                for col in df.columns:
                    m, s = df[col].mean(), df[col].std()
                    lo, hi = df[col].min(), df[col].max()
                    f.write(f"{col}:\n  mean={m:.6f}, std={s:.6f}\n"
                            f"  min={lo:.6f}, max={hi:.6f}\n")
                f.write("\n")

            f.write("CHANGES:\n" + "-" * 70 + "\n")
            for col in df_before.columns:
                mb, ma = df_before[col].mean(), df_after[col].mean()
                sb, sa = df_before[col].std(), df_after[col].std()
                dm, ds = ma - mb, sa - sb
                pm = (dm / mb * 100) if abs(mb) > 1e-8 else 0
                ps = (ds / sb * 100) if abs(sb) > 1e-8 else 0
                f.write(f"{col}:\n  Δ mean={dm:.6f} ({pm:.2f}%)\n"
                        f"  Δ std={ds:.6f} ({ps:.2f}%)\n")
            f.write("\n" + "=" * 70 + "\n")
        logger.info(f"Saved phenotype comparison to: {path}")

    def store_predictions(
        self,
        pred_mean: Tensor,
        pred_var: Tensor,
        C0: Optional[Tensor] = None,
        C1: Optional[Tensor] = None,
        y_true: Optional[Tensor] = None,
        ci_multiplier: float = 2.0,
    ) -> None:
        """Store prediction tensors and (optionally) ground truth.

        Parameters
        ----------
        pred_mean : Tensor, shape (n_test, n_traits)
        pred_var  : Tensor, shape (n_test, n_traits)
        C0, C1    : Optional trait covariance matrices.
        y_true    : Optional ground-truth phenotypes.
                    Pass ``None`` for external-prediction mode.
        ci_multiplier : Multiplier for confidence interval width
                        (default 2.0 ≈ 95.45 %).
        """
        n_test, n_traits = pred_mean.shape
        pred_std = torch.sqrt(pred_var)
        ci_lower = pred_mean - ci_multiplier * pred_std
        ci_upper = pred_mean + ci_multiplier * pred_std

        # Determine scenario
        self._has_ground_truth = (
            y_true is not None and not torch.isnan(y_true).all()
        )

        self._tensors = {
            'pred_mean': self._to_numpy(pred_mean),
            'pred_var': self._to_numpy(pred_var),
            'pred_std': self._to_numpy(pred_std),
            'ci_lower': self._to_numpy(ci_lower),
            'ci_upper': self._to_numpy(ci_upper),
        }
        if C0 is not None:
            self._tensors['C0'] = self._to_numpy(C0)
        if C1 is not None:
            self._tensors['C1'] = self._to_numpy(C1)

        if self._has_ground_truth:
            self._tensors['y_true'] = self._to_numpy(y_true)

        self.predictions = {
            'n_test': n_test,
            'n_traits': n_traits,
            'method': self.method,
            'has_ground_truth': self._has_ground_truth,
            'pred_mean_summary': self._tensor_summary(pred_mean),
            'pred_var_summary': self._tensor_summary(pred_var),
        }

        if self._has_ground_truth:
            self.predictions['y_true_summary'] = self._tensor_summary(y_true)

        # Confidence intervals
        from scipy import stats as sp_stats
        ci_level = float(2 * sp_stats.norm.cdf(ci_multiplier) - 1)
        ci_width = ci_upper - ci_lower

        self.confidence_intervals = {
            'level': ci_level,
            'ci_width_mean': float(ci_width.mean().item()),
            'ci_width_std': float(ci_width.std().item()),
            'ci_width_per_trait': [
                float(ci_width[:, p].mean().item()) for p in range(n_traits)
            ],
        }

        if self._has_ground_truth:
            self.confidence_intervals['coverage'] = self._compute_coverage(
                y_true, ci_lower, ci_upper
            )

        # Covariance info 
        if C0 is not None and C1 is not None:
            diag0 = torch.diag(C0)
            diag1 = torch.diag(C1)
            self.predictions['covariance'] = {
                'C0_trace': float(torch.trace(C0).item()),
                'C1_trace': float(torch.trace(C1).item()),
                'C0_diag': self._to_list(diag0),
                'C1_diag': self._to_list(diag1),
                'total_variance_per_trait': self._to_list(diag0 + diag1),
                'heritability_per_trait': self._to_list(
                    diag0 / (diag0 + diag1 + 1e-8)
                ),
            }

    def compute_metrics(
        self,
        pred_mean: Any,
        y_true: Any,
    ) -> Dict[str, Any]:
        """Compute prediction quality metrics (requires ground truth)."""
        if isinstance(pred_mean, np.ndarray):
            pred_mean = torch.from_numpy(pred_mean)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)

        n_test, n_traits = pred_mean.shape

        metrics: Dict[str, Any] = {
            'n_test': n_test,
            'n_traits': n_traits,
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
            'per_trait': {},
            'overall': {},
        }

        mse_l, mae_l, rmse_l, nrmse_l, corr_l, r2_l = ([] for _ in range(6))

        for p in range(n_traits):
            y_p = y_true[:, p]
            pr_p = pred_mean[:, p]

            mse = float(torch.mean((y_p - pr_p) ** 2).item())
            mae = float(torch.mean(torch.abs(y_p - pr_p)).item())
            rmse = float(torch.sqrt(torch.tensor(mse)).item())

            y_std = y_p.std().item()
            nrmse = rmse / y_std if y_std > 1e-8 else float('nan')

            if y_p.std() > 1e-8 and pr_p.std() > 1e-8:
                corr = float(
                    torch.corrcoef(torch.stack([y_p, pr_p]))[0, 1].item()
                )
            else:
                corr = 0.0

            ss_res = torch.sum((y_p - pr_p) ** 2)
            ss_tot = torch.sum((y_p - y_p.mean()) ** 2)
            r2 = float((1 - ss_res / ss_tot).item()) if ss_tot > 1e-8 else 0.0

            metrics['per_trait'][f'trait_{p}'] = {
                'mse': mse, 'mae': mae, 'rmse': rmse,
                'nrmse': nrmse, 'correlation': corr, 'r2': r2,
            }
            mse_l.append(mse); mae_l.append(mae); rmse_l.append(rmse)
            nrmse_l.append(nrmse); corr_l.append(corr); r2_l.append(r2)

        residuals = y_true - pred_mean
        metrics['overall'] = {
            'mse': float(torch.mean(residuals ** 2).item()),
            'mae': float(torch.mean(torch.abs(residuals)).item()),
            'rmse': float(torch.sqrt(torch.mean(residuals ** 2)).item()),
            'mse_mean': float(np.mean(mse_l)),
            'mae_mean': float(np.mean(mae_l)),
            'rmse_mean': float(np.mean(rmse_l)),
            'nrmse_mean': float(np.nanmean(nrmse_l)),
            'correlation_mean': float(np.mean(corr_l)),
            'correlation_std': float(np.std(corr_l)),
            'r2_mean': float(np.mean(r2_l)),
            'r2_std': float(np.std(r2_l)),
        }

        metrics['mse_per_trait'] = mse_l
        metrics['mae_per_trait'] = mae_l
        metrics['rmse_per_trait'] = rmse_l
        metrics['correlation_per_trait'] = corr_l
        metrics['r2_per_trait'] = r2_l

        self.metrics = metrics
        return metrics

    def save(self) -> None:
        """Save results to disk.

        Scenario A (ground truth):
            predicted_phenotypes.csv, metrics.json, summary.json
        Scenario B (external prediction):
            predicted_phenotypes.csv, summary.json
        """
        self.metadata = {
            'uid': self.uid,
            'method': self.method,
            'n_traits': self.n_traits,
            'has_ground_truth': self._has_ground_truth,
            'rep_idx': self.rep_idx,
            'eta': self.eta,
            'corr_bounds': self.corr_bounds,
            'use_heterogeneity': self.use_heterogeneity,
            'timestamp': datetime.now().isoformat(),
            'base_dir': self.base_dir,
        }

        # Always: summary.json
        summary: Dict[str, Any] = {
            'metadata': self.metadata,
            'predictions': self.predictions,
            'confidence_intervals': self.confidence_intervals,
        }
        if self._has_ground_truth and self.metrics:
            summary['metrics_summary'] = {
                'overall': self.metrics.get('overall', {}),
                'correlation_per_trait': self.metrics.get('correlation_per_trait', []),
                'r2_per_trait': self.metrics.get('r2_per_trait', []),
            }

        summary_path = os.path.join(self.base_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        # Scenario A only: metrics.json 
        if self._has_ground_truth:
            if self.metrics:
                metrics_path = os.path.join(self.base_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
                logger.info(f"Saved metrics to {metrics_path}")

    def export_predictions_csv(
        self,
        sample_index=None,
        trait_names: Optional[List[str]] = None,
    ) -> None:
        """Export predictions to a human-readable CSV.

        Columns per trait:
            <Trait>_Pred, <Trait>_Std, and (scenario A only) <Trait>_True
        """
        if 'pred_mean' not in self._tensors:
            logger.warning("No predictions stored – nothing to export.")
            return

        pred_mean = self._tensors['pred_mean']
        pred_std = self._tensors.get('pred_std')
        y_true = self._tensors.get('y_true')  # absent in scenario B

        if trait_names is None:
            trait_names = [f"Trait_{i}" for i in range(self.n_traits)]

        df = pd.DataFrame(index=sample_index)

        for i, t in enumerate(trait_names):
            df[f"{t}_Pred"] = pred_mean[:, i]
            if pred_std is not None:
                df[f"{t}_Std"] = pred_std[:, i]
            if self._has_ground_truth and y_true is not None:
                df[f"{t}_True"] = y_true[:, i]

        csv_path = os.path.join(self.base_dir, "predicted_phenotypes.csv")
        df.to_csv(csv_path)
        logger.info(f"Saved predictions to: {csv_path}")

    def _compute_coverage(
        self, y_true: Tensor, ci_lower: Tensor, ci_upper: Tensor,
    ) -> Dict[str, Any]:
        in_ci = (y_true >= ci_lower) & (y_true <= ci_upper)
        return {
            'overall': float(in_ci.float().mean().item()),
            'per_trait': [
                float(in_ci[:, p].float().mean().item())
                for p in range(y_true.shape[1])
            ],
        }

    @staticmethod
    def _tensor_summary(t: Tensor) -> Dict[str, Any]:
        return {
            'shape': list(t.shape),
            'mean': float(t.mean().item()),
            'std': float(t.std().item()),
            'min': float(t.min().item()),
            'max': float(t.max().item()),
            'mean_per_trait': [float(x) for x in t.mean(dim=0).tolist()],
            'std_per_trait': [float(x) for x in t.std(dim=0).tolist()],
        }

    @staticmethod
    def _to_numpy(t: Tensor) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    @staticmethod
    def _to_list(t: Tensor) -> List[float]:
        if isinstance(t, torch.Tensor):
            return [float(x) for x in t.detach().cpu().tolist()]
        return list(t)

    def get_results_dict(self) -> Dict[str, Any]:
        """Return results as a dictionary (backward compatibility)."""
        def _maybe_tensor(key):
            if key in self._tensors:
                return torch.from_numpy(self._tensors[key])
            return None

        return {
            'method': self.method,
            'mean': _maybe_tensor('pred_mean'),
            'variance': _maybe_tensor('pred_var'),
            'std': _maybe_tensor('pred_std'),
            'ci_lower': _maybe_tensor('ci_lower'),
            'ci_upper': _maybe_tensor('ci_upper'),
            'y_true': _maybe_tensor('y_true'),
            'metrics': self.metrics,
            'confidence_intervals': self.confidence_intervals,
            'metadata': self.metadata,
        }