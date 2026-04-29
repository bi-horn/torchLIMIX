import os
import io
import csv
import json
import time
import logging
import torch
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def _val(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.view(-1).tolist()
    elif isinstance(x, list):
        return [_val(v) for v in x]
    return x if x is not None else None


def _to_numpy_1d(val, size, fill=np.nan):
    """Convert tensor / scalar / list / None → numpy 1-D float64 array of length *size*."""
    if val is None:
        return np.full(size, fill, dtype=np.float64)
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu()
        if val.numel() == 1:
            return np.full(size, val.item(), dtype=np.float64)
        return val.numpy().astype(np.float64).ravel()[:size]
    if isinstance(val, (int, float)):
        return np.full(size, val, dtype=np.float64)
    return np.asarray(val, dtype=np.float64).ravel()[:size]


def _convert_matrix(matrix):
    """Convert matrix to flattened numpy array."""
    if matrix is None:
        return None
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy().flatten()
    if isinstance(matrix, np.ndarray):
        return matrix.flatten()
    return np.array(matrix).flatten()


def _tensor_to_columns(val, size):
    """Convert a per-SNP tensor to a numpy array.

    Returns None if val is None, 1D array if scalar per SNP,
    2D array if vector per SNP.
    """
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        arr = val.detach().cpu().numpy()
    else:
        arr = np.asarray(val, dtype=np.float64)

    if arr.ndim == 1:
        return arr[:size].astype(np.float64)
    arr = arr.reshape(size, -1).astype(np.float64)
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _add_columns_from_tensor(columns: dict, name: str, tensor, size: int):
    """Add one or more columns to *columns* dict from a tensor."""
    arr = _tensor_to_columns(tensor, size)
    if arr is None:
        return
    if arr.ndim == 1:
        columns[name] = arr
    else:
        for j in range(arr.shape[1]):
            columns[f"{name}_{j}"] = arr[:, j]


class StoreResults:
    """GWAS result storage with vectorised I/O.

    Key features:
    - Null model (beta0) stored once via save_null_model().
    - Per-SNP betas, likelihoods, and PVE stored as vectorised arrays.
    - Supports parquet (default for simulations) and CSV (default for real data).
    """

    def __init__(
        self,
        output_dir: str,
        uid: str,
        rep_idx: Optional[int] = None,
        eta: Optional[float] = None,
        corr_bounds: Optional[int] = None,
        simulation_info: Optional[Dict[str, Any]] = None,
        rank: Optional[int] = None,
        test_type: Optional[str] = None,
        correction_metadata: Optional[Dict[str, Any]] = None,
        phenotype_data: Optional[Dict[str, Any]] = None,
        use_parquet: Optional[bool] = None,
    ):
        self.uid = uid
        self.test_type = test_type
        self.has_h2 = test_type in ["specific_vs_common", "any_vs_common", None]
        self.simulation_info = simulation_info or {}
        self.corr_bounds = corr_bounds
        self.rep_idx = rep_idx
        self.eta = eta
        self.rank = rank
        self.correction_metadata = correction_metadata or {}
        self.phenotype_data = phenotype_data or {}

        if use_parquet is not None:
            self.use_parquet = use_parquet
        else:
            self.use_parquet = rep_idx is not None

        # Simulation metadata
        self.ncausal = self.simulation_info.get('ncausal', None)
        self.heterogeneity_context_indices = self.simulation_info.get('heterogeneity_context_indices', [])
        self.use_heterogeneity = self.simulation_info.get('use_heterogeneity', False)
        self.rescaling_common_indices = self.simulation_info.get('rescaling_common_indices', [])
        self.n_traits = self.simulation_info.get(
            'n_traits',
            len(self.heterogeneity_context_indices) if self.heterogeneity_context_indices else 2,
        )

        # In-memory caches
        self._likelihood_data: Optional[np.ndarray] = None
        self._beta_df: Optional[pd.DataFrame] = None
        self._pve_df: Optional[pd.DataFrame] = None
        self._null_model: Optional[Dict[str, Any]] = None
        self.covariance_results = {}

        # Output paths
        self.base_dir = self._build_base_dir(output_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        ext = ".parquet" if self.use_parquet else ".csv"
        self.likelihood_path = os.path.join(self.base_dir, f"log_likelihoods{ext}")
        self.beta_path = os.path.join(self.base_dir, f"beta_results{ext}")
        self.pve_path = os.path.join(self.base_dir, f"pve_results{ext}")
        self.null_model_path = os.path.join(self.base_dir, "null_model.csv")
        self.null_model_txt_path = os.path.join(self.base_dir, "null_model.txt")
        self.sim_params_csv_path = os.path.join(self.base_dir, "sim_params.csv")
        self.optimization_stats_csv_path = os.path.join(self.base_dir, "optimization_stats.csv")
        self.covariance_csv_path = os.path.join(self.base_dir, "covariances.csv")

        # Headers
        self._set_headers()
        if simulation_info:
            self.sim_params_headers = self._build_sim_params_headers()

        # Initialise files
        if not self.use_parquet:
            self._initialize_csv()
        if simulation_info:
            self._initialize_sim_params_csv()
        self._initialize_covariance_csv()

        self._save_data_preprocessing_info()
        self._save_phenotype_data()

    # Path / header setup 
    def _build_base_dir(self, output_dir: str) -> str:
        if self.rep_idx is not None:
            if self.use_heterogeneity and self.corr_bounds is not None:
                sim_folder = f"corr{self.corr_bounds:01d}"
            elif not self.use_heterogeneity and self.eta is not None:
                eta_str = f"{self.eta:.2f}"
                sim_folder = f"eta{eta_str}"
            else:
                sim_folder = "simulation"
            return os.path.join(output_dir, sim_folder, f"rep{self.rep_idx:04d}")
        return output_dir

    def _set_headers(self):
        if self.has_h2:
            self.headers = [
                "snp_index", "lml0", "lml1", "lml2",
                "lrt10", "df10", "pv10",
                "lrt20", "df20", "pv20",
                "lrt21", "df21", "pv21",
                "scale_H0",
            ]
            self.pve_headers = ["pve1", "pve2"]
            self.beta_base_names = ["beta1", "beta1_se", "beta2", "beta2_se"]
        else:
            self.headers = [
                "snp_index", "lml0", "lml1",
                "lrt10", "df10", "pv10",
                "scale_H0",
            ]
            self.pve_headers = ["pve1"]
            self.beta_base_names = ["beta1", "beta1_se"]

    def _build_sim_params_headers(self):
        if self.use_heterogeneity:
            ctx = [f"context{i}_indices" for i in range(self.n_traits)]
            base = (["rep_idx", "use_heterogeneity", "corr_bounds", "n_traits"]
                    if self.rep_idx is not None
                    else ["use_heterogeneity", "n_traits"])
            return base + ctx + ["ncausal", "rank"]
        else:
            if self.rep_idx is not None:
                return ["rep_idx", "use_heterogeneity", "rescaling_common_indices", "eta", "rank"]
            return ["use_heterogeneity", "rescaling_common_indices", "eta", "rank"]

    # File initialisation 
    def _initialize_csv(self):
        if not os.path.exists(self.likelihood_path):
            with open(self.likelihood_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.headers)

    def _initialize_covariance_csv(self):
        if not os.path.exists(self.covariance_csv_path):
            hdrs = (["rep_idx", "matrix_size", "C0_flat", "C1_flat"]
                    if self.rep_idx is not None
                    else ["matrix_size", "C0_flat", "C1_flat"])
            with open(self.covariance_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(hdrs)

    def _initialize_sim_params_csv(self):
        if os.path.exists(self.sim_params_csv_path):
            return
        try:
            with open(self.sim_params_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.sim_params_headers)
                writer.writerow(self._build_sim_params_row())
        except Exception as e:
            logger.error(f"Failed to create sim_params CSV: {e}")

    def _build_sim_params_row(self):
        rank_val = int(self.rank) if self.rank is not None else 0
        if self.use_heterogeneity:
            context_indices = self._extract_context_indices()
            base = ([self.rep_idx, self.use_heterogeneity, self.corr_bounds, self.n_traits]
                    if self.rep_idx is not None
                    else [self.use_heterogeneity, self.n_traits])
            return base + context_indices + [self.ncausal, rank_val]
        else:
            eta = self.simulation_info.get('eta', None)
            rescaling = str(self.rescaling_common_indices) if self.rescaling_common_indices else "None"
            if self.rep_idx is not None:
                return [self.rep_idx, self.use_heterogeneity, rescaling, eta, rank_val]
            return [self.use_heterogeneity, rescaling, eta, rank_val]

    def _extract_context_indices(self):
        context_indices = []
        global_context = self.simulation_info.get('global_context_indices', {})
        for trait_i in range(self.n_traits):
            val = "None"
            if trait_i < len(self.heterogeneity_context_indices):
                ctx = self.heterogeneity_context_indices[trait_i]
                if ctx is not None:
                    if isinstance(ctx, list) and len(ctx) > 0:
                        ctx = [int(x) if hasattr(x, 'item') else x for x in ctx]
                    elif hasattr(ctx, 'item'):
                        ctx = [int(ctx.item())]
                    val = str(ctx)
            if val == "None" and trait_i in global_context:
                fb = global_context[trait_i]
                val = str(fb) if fb is not None else "None"
            context_indices.append(val)
        return context_indices

    # Covariance storage 
    def add_cov(self, C0, C1):
        C0_flat = _convert_matrix(C0)
        C1_flat = _convert_matrix(C1)
        if C0_flat is None or C1_flat is None:
            logger.warning("Covariance matrix is None, skipping save")
            return

        matrix_size = int(np.sqrt(len(C0_flat)))
        C0_str = ','.join(map(str, C0_flat))
        C1_str = ','.join(map(str, C1_flat))
        row = ([self.rep_idx, matrix_size, C0_str, C1_str]
               if self.rep_idx is not None
               else [matrix_size, C0_str, C1_str])
        try:
            with open(self.covariance_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.error(f"Failed to write covariance CSV: {e}")

    def get_covariance_matrices(self):
        if not os.path.exists(self.covariance_csv_path):
            return None
        try:
            df = pd.read_csv(self.covariance_csv_path)
            if len(df) == 0:
                return None
            last = df.iloc[-1]
            sz = int(last['matrix_size'])
            C0 = np.array([float(x) for x in last['C0_flat'].split(',')]).reshape(sz, sz)
            C1 = np.array([float(x) for x in last['C1_flat'].split(',')]).reshape(sz, sz)
            return {'C0': C0, 'C1': C1}
        except Exception as e:
            logger.error(f"Error reading covariance matrices: {e}")
            return None

    # Null model (stored once) 
    def save_null_model(self, beta0, beta0_se, lml0, scale_H0,
                        n_covariates: int, n_traits: int):
        """Save null model results once (constant across all SNPs)."""
        beta0_np = beta0.detach().cpu().numpy() if isinstance(beta0, torch.Tensor) else np.asarray(beta0)
        beta0_se_np = beta0_se.detach().cpu().numpy() if isinstance(beta0_se, torch.Tensor) else np.asarray(beta0_se)
        lml0_val = float(lml0.item() if isinstance(lml0, torch.Tensor) else lml0)
        scale_val = float(scale_H0.item() if isinstance(scale_H0, torch.Tensor) else scale_H0)

        cov_labels = ["intercept"] + [f"cov_{i}" for i in range(1, n_covariates)]

        self._null_model = {
            'beta0': beta0_np, 'beta0_se': beta0_se_np,
            'lml0': lml0_val, 'scale_H0': scale_val,
            'n_covariates': n_covariates, 'n_traits': n_traits,
            'covariate_labels': cov_labels,
        }

        # CSV with structured columns
        cov_labels = ["intercept"] + [f"cov_{i}" for i in range(1, n_covariates)]
        rows = []
        for i, label in enumerate(cov_labels):
            row = {"covariate": label}
            for j in range(n_traits):
                row[f"beta_trait_{j}"] = beta0_np[i, j]
                row[f"se_trait_{j}"] = beta0_se_np[i, j]
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.null_model_path, index=False)

        # Human-readable text
        trait_cols = [f"trait_{j}" for j in range(n_traits)]
        with open(self.null_model_txt_path, 'w') as f:
            f.write(f"Null Model (H0): y ~ N((A ⊗ M)a, C0 x K + C1 ⊗ I)\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"lml0     = {lml0_val:.10f}\n")
            f.write(f"scale_H0 = {scale_val:.10f}\n\n")
            header = f"{'':>12s}" + "".join(f"{t:>14s}" for t in trait_cols)
            for section, data in [("Fixed effects beta", beta0_np), ("Standard errors", beta0_se_np)]:
                f.write(f"{section} ({n_covariates} covariates x {n_traits} traits):\n")
                f.write(f"{'-' * 60}\n{header}\n")
                for i, label in enumerate(cov_labels):
                    vals = "".join(f"{data[i, j]:>14.6f}" for j in range(n_traits))
                    f.write(f"{label:>12s}{vals}\n")
                f.write("\n")

        logger.info(f"Null model saved to: {self.null_model_path}")

    def load_null_model(self) -> Optional[Dict[str, Any]]:
        if self._null_model is not None:
            return self._null_model
        if not os.path.exists(self.null_model_path):
            return None
        df = pd.read_csv(self.null_model_path)
        beta_cols = sorted([c for c in df.columns if c.startswith("beta_trait_")])
        se_cols = sorted([c for c in df.columns if c.startswith("se_trait_")])
        self._null_model = {
            'beta0': df[beta_cols].values,
            'beta0_se': df[se_cols].values,
            'lml0': None, 'scale_H0': None,
            'n_covariates': len(df), 'n_traits': len(beta_cols),
            'covariate_labels': df['covariate'].tolist(),
        }
        return self._null_model

    # Likelihood storage (vectorised) 
    def add_likelihood_result(
        self, snp_indices, lml0, lml1, lml2,
        lrt10, lrt20, lrt21,
        df10=None, df20=None, df21=None,
        scale_H0=None, scale_H1=None, scale_H2=None,
        C0=None, C1=None,
    ):
        idx = np.asarray(snp_indices, dtype=np.int64)
        size = len(idx)

        _lml0  = _to_numpy_1d(lml0, size)
        _lml1  = _to_numpy_1d(lml1, size)
        _lrt10 = _to_numpy_1d(lrt10, size)
        _df10  = _to_numpy_1d(df10, size, fill=1.0)
        _sH0   = _to_numpy_1d(scale_H0, size)
        _pv10  = self._compute_p_values(_lrt10, _df10)

        if self.has_h2:
            _lml2  = _to_numpy_1d(lml2, size)
            _lrt20 = _to_numpy_1d(lrt20, size)
            _lrt21 = _to_numpy_1d(lrt21, size)
            _df20  = _to_numpy_1d(df20, size, fill=1.0)
            _df21  = _to_numpy_1d(df21, size, fill=1.0)
            _pv20  = self._compute_p_values(_lrt20, _df20)
            _pv21  = self._compute_p_values(_lrt21, _df21)
            data = np.column_stack([
                idx, _lml0, _lml1, _lml2,
                _lrt10, _df10, _pv10,
                _lrt20, _df20, _pv20,
                _lrt21, _df21, _pv21,
                _sH0,
            ])
        else:
            data = np.column_stack([
                idx, _lml0, _lml1,
                _lrt10, _df10, _pv10,
                _sH0,
            ])

        np.nan_to_num(data, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
        self._likelihood_data = data

        C0_np = C0.detach().cpu().numpy() if isinstance(C0, torch.Tensor) else C0
        C1_np = C1.detach().cpu().numpy() if isinstance(C1, torch.Tensor) else C1
        self.covariance_results[0] = {'C0': C0_np, 'C1': C1_np}

        self._write_likelihood()

    def _write_likelihood(self):
        if self._likelihood_data is None:
            return
        if self.use_parquet:
            df = pd.DataFrame(self._likelihood_data, columns=self.headers)
            df['snp_index'] = df['snp_index'].astype(np.int32)
            df.to_parquet(self.likelihood_path, compression="zstd", index=False)
        else:
            buf = io.StringIO()
            buf.write(','.join(self.headers) + '\n')
            np.savetxt(buf, self._likelihood_data, delimiter=',', fmt='%.10g')
            with open(self.likelihood_path, 'w') as f:
                f.write(buf.getvalue())

    def likelihood_results(self):
        if self._likelihood_data is not None and len(self._likelihood_data) > 0:
            df = pd.DataFrame(self._likelihood_data, columns=self.headers)
            df['snp_index'] = df['snp_index'].astype(int)
            return df.set_index('snp_index').dropna(axis=1, how='all')
        return pd.DataFrame(columns=self.headers)

    # Per-SNP beta storage (vectorised) 
    def add_beta_result(self, snp_indices,
                        beta1=None, beta1_se=None,
                        beta2=None, beta2_se=None):
        """Vectorised per-SNP beta storage — direct tensor→numpy, single write."""
        if isinstance(snp_indices, (int, np.integer)):
            size = beta1.shape[0] if isinstance(beta1, torch.Tensor) else 1
            snp_indices = np.arange(int(snp_indices), int(snp_indices) + size, dtype=np.int32)
        else:
            snp_indices = np.asarray(snp_indices, dtype=np.int32)
        size = len(snp_indices)

        columns = {"snp_index": snp_indices}
        _add_columns_from_tensor(columns, "beta1", beta1, size)
        _add_columns_from_tensor(columns, "beta1_se", beta1_se, size)
        if self.has_h2:
            _add_columns_from_tensor(columns, "beta2", beta2, size)
            _add_columns_from_tensor(columns, "beta2_se", beta2_se, size)

        df = pd.DataFrame(columns)
        self._beta_df = df

        if self.use_parquet:
            float_cols = df.select_dtypes(include="float64").columns
            df[float_cols] = df[float_cols].astype(np.float32)
            df.to_parquet(self.beta_path, compression="zstd", index=False)
        else:
            df.to_csv(self.beta_path, index=False)

    def effectsizes(self):
        if self._beta_df is not None:
            return self._beta_df
        if os.path.exists(self.beta_path):
            if self.use_parquet:
                return pd.read_parquet(self.beta_path)
            return pd.read_csv(self.beta_path)
        return pd.DataFrame()

    # PVE storage (vectorised) 
    def add_pve_result(self, snp_indices, pve1=None, pve2=None):
        """Vectorised PVE storage — direct tensor→numpy, single write."""
        if isinstance(snp_indices, (int, np.integer)):
            ref = pve1 if pve1 is not None else pve2
            size = ref.shape[0] if ref is not None else 0
            snp_indices = np.arange(int(snp_indices), int(snp_indices) + size, dtype=np.int32)
        else:
            snp_indices = np.asarray(snp_indices, dtype=np.int32)
        size = len(snp_indices)

        columns = {"snp_index": snp_indices}
        _add_columns_from_tensor(columns, "pve1", pve1, size)
        if self.has_h2:
            _add_columns_from_tensor(columns, "pve2", pve2, size)

        df = pd.DataFrame(columns)
        self._pve_df = df

        if self.use_parquet:
            df["snp_index"] = df["snp_index"].astype(np.int32)
            float_cols = df.select_dtypes(include="float64").columns
            df[float_cols] = df[float_cols].astype(np.float32)
            df.to_parquet(self.pve_path, compression="zstd", index=False)
        else:
            df.to_csv(self.pve_path, index=False)

    def pve(self):
        if self._pve_df is not None:
            return self._pve_df
        if os.path.exists(self.pve_path):
            if self.use_parquet:
                return pd.read_parquet(self.pve_path)
            return pd.read_csv(self.pve_path)
        return pd.DataFrame()

    # Optimization metrics 
    def add_optimization_metrics(self, optimization_results):
        if optimization_results is None:
            logger.warning("No optimization results provided")
            return
        self.optimization_metrics = optimization_results.copy()
        self._save_analytical_metrics_json(optimization_results)

    def _convert_to_serializable(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(i) for i in obj]
        return obj

    def _save_analytical_metrics_json(self, opt_res):
        jsonl_path = self.optimization_stats_csv_path.replace('.csv', '.jsonl')
        record = {
            'rep_idx': int(self.rep_idx) if self.rep_idx is not None else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'analytical',
            'metrics': self._convert_to_serializable(opt_res),
        }
        try:
            with open(jsonl_path, 'w') as f:
                f.write(json.dumps(record) + '\n')
            logger.info(f"Metrics saved to: {jsonl_path}")
        except Exception as e:
            logger.error(f"Failed to save JSONL: {e}")
        self._print_optimization_status(opt_res)

    def _print_optimization_status(self, opt_res):
        logger.info(f"Optimization: LML={opt_res['lml']:.6f}, "
                     f"grad_norm={opt_res['grad_norm']:.2e}, "
                     f"iters={opt_res['iterations']}")

    def get_optimization_stats(self):
        if os.path.exists(self.optimization_stats_csv_path):
            try:
                return pd.read_csv(self.optimization_stats_csv_path)
            except Exception as e:
                logger.error(f"Error reading optimization stats: {e}")
        return pd.DataFrame()

    def get_sim_params_info(self):
        if os.path.exists(self.sim_params_csv_path):
            try:
                return pd.read_csv(self.sim_params_csv_path)
            except Exception as e:
                logger.error(f"Failed to read sim params CSV: {e}")
        return pd.DataFrame(columns=self.sim_params_headers)

    # Static helpers 

    @staticmethod
    def _compute_p_values(lrt_values, df_values):
        lrt = np.asarray(lrt_values, dtype=np.float64)
        df = np.asarray(df_values, dtype=np.float64)
        p_values = np.full_like(lrt, np.nan, dtype=np.float64)
        mask = (lrt >= 0) & (df > 0) & np.isfinite(lrt) & np.isfinite(df)
        if np.any(mask):
            p_values[mask] = stats.chi2.sf(lrt[mask], df[mask])
        return p_values

    @staticmethod
    def _extract_matrix_diagonal(matrix):
        if matrix is None:
            return None
        if isinstance(matrix, torch.Tensor):
            return (matrix.diag() if matrix.ndim == 2 else matrix).detach().cpu().numpy()
        if isinstance(matrix, np.ndarray):
            return np.diag(matrix) if matrix.ndim == 2 else matrix
        if isinstance(matrix, list) and len(matrix) > 0 and isinstance(matrix[0], (np.ndarray, torch.Tensor)):
            return StoreResults._extract_matrix_diagonal(matrix[0])
        return matrix

    # Preprocessing / phenotype save 
    def _save_data_preprocessing_info(self):
        from datetime import datetime
        path = os.path.join(self.base_dir, "data_preprocessing.json")
        info = {
            'timestamp': datetime.now().isoformat(),
            'uid': self.uid,
            'data_type': 'simulated' if self.eta is not None else 'real',
        }
        if self.eta is not None:
            info['simulation_params'] = {'eta': self.eta, 'rep_idx': self.rep_idx}
        if self.correction_metadata:
            info['corrections'] = {
                'batch_correction': self.correction_metadata.get('batch_correction', {}),
                'covariate_correction': self.correction_metadata.get('covariate_correction', {}),
                'transformations': self.correction_metadata.get('transformations', {}),
            }
        corrected = self.phenotype_data.get('corrections_applied', False)
        info['phenotype_files'] = (
            {'corrected': 'phenotypes_corrected.csv', 'uncorrected': 'phenotypes_uncorrected.csv'}
            if corrected else {'phenotypes': 'phenotypes.csv'}
        )
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved preprocessing info to: {path}")

    def _save_phenotype_data(self):
        corrected = self.phenotype_data.get('corrected', None)
        uncorrected = self.phenotype_data.get('uncorrected', None)
        applied = self.phenotype_data.get('corrections_applied', False)

        if corrected is None:
            return

        col_names = None
        if uncorrected is not None and hasattr(uncorrected, 'columns'):
            col_names = uncorrected.columns

        if not isinstance(corrected, pd.DataFrame):
            # Handle PyTorch tensors (detach from graph and move to CPU first)
            if hasattr(corrected, 'detach'):
                arr = corrected.detach().cpu().numpy()
            # Handle TensorFlow or standard tensors
            elif hasattr(corrected, 'numpy'):
                arr = corrected.numpy()
            else:
                arr = corrected # Fallback for pure numpy arrays
                
            corrected = pd.DataFrame(arr, columns=col_names)

        # 3. Ensure 'uncorrected' is also a DataFrame (just in case it's also a tensor)
        if uncorrected is not None and not isinstance(uncorrected, pd.DataFrame):
            if hasattr(uncorrected, 'detach'):
                arr_u = uncorrected.detach().cpu().numpy()
            elif hasattr(uncorrected, 'numpy'):
                arr_u = uncorrected.numpy()
            else:
                arr_u = uncorrected
            uncorrected = pd.DataFrame(arr_u, columns=col_names)

        if applied and uncorrected is not None:
            cp = os.path.join(self.base_dir, "phenotypes_corrected.csv")
            up = os.path.join(self.base_dir, "phenotypes_uncorrected.csv")
            
            # Now these will safely execute as Pandas DataFrames
            corrected.to_csv(cp, index=False) 
            uncorrected.to_csv(up, index=False)
            
            logger.info(f"Saved corrected: {cp}, uncorrected: {up}")
            self._save_phenotype_comparison(uncorrected, corrected)
        else:
            pp = os.path.join(self.base_dir, "phenotypes.csv")
            corrected.to_csv(pp, index=False)
            logger.info(f"Saved phenotypes to: {pp}")

    def _save_phenotype_comparison(self, df_before, df_after):
        path = os.path.join(self.base_dir, "phenotype_comparison.txt")
        with open(path, 'w') as f:
            f.write("=" * 70 + "\nPHENOTYPE CORRECTION COMPARISON\n" + "=" * 70 + "\n\n")
            for label, df in [("BEFORE", df_before), ("AFTER", df_after)]:
                f.write(f"{label} CORRECTIONS:\n" + "-" * 70 + "\n")
                for col in df.columns:
                    s = df[col]
                    f.write(f"{col}:\n  mean={s.mean():.6f}, std={s.std():.6f}\n"
                            f"  min={s.min():.6f}, max={s.max():.6f}\n")
                f.write("\n")
            f.write("CHANGES:\n" + "-" * 70 + "\n")
            for col in df_before.columns:
                mb, ma = df_before[col].mean(), df_after[col].mean()
                sb, sa = df_before[col].std(), df_after[col].std()
                dm, ds = ma - mb, sa - sb
                mp = (dm / mb * 100) if abs(mb) > 1e-8 else 0
                sp = (ds / sb * 100) if abs(sb) > 1e-8 else 0
                f.write(f"{col}:\n  mean delta={dm:.6f} ({mp:.2f}%)\n  std delta={ds:.6f} ({sp:.2f}%)\n")
            f.write("\n" + "=" * 70 + "\n")
        logger.info(f"Saved phenotype comparison to: {path}")

    # Display
    def _format_null_model_table(self, beta0, beta0_se, n_covariates, n_traits):
        labels = ["intercept"] + [f"cov_{i}" for i in range(1, n_covariates)]
        trait_labels = [f"trait_{j}" for j in range(n_traits)]
        max_label = max(len(l) for l in labels)
        sep = "─" * (max_label + 2 + n_traits * 14)
        header = " " * (max_label + 4) + "".join(f"{t:>13s}" for t in trait_labels)

        lines = []
        for section, data in [("Fixed effects beta", beta0), ("Standard errors", beta0_se)]:
            lines.append(f"  {section} ({n_covariates} covariates x {n_traits} traits):")
            lines.append(f"  {sep}")
            lines.append(f"  {header}")
            lines.append(f"  {sep}")
            for i, label in enumerate(labels):
                vals = "".join(f"{data[i, j]:>13.6f}" for j in range(n_traits))
                lines.append(f"  {label:>{max_label}s}  {vals}")
            lines.append(f"  {sep}")
            lines.append("")

        return "\n".join(lines)

    def display_formatted_results(self):
        start_time = time.time()

        likelihood_df = self.likelihood_results()
        beta_df = self.effectsizes()
        null_model = self.load_null_model()

        if likelihood_df.empty:
            print("No likelihood results to display.")
            return

        def get_col(df, col, default=np.nan):
            return df[col].values.astype(np.float64) if col in df.columns else np.full(len(df), default)

        def compute_stats(values):
            v = values[np.isfinite(values)]
            if len(v) == 0:
                return {k: np.nan for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']}
            return {'mean': np.mean(v), 'std': np.std(v), 'min': np.min(v),
                    '25%': np.percentile(v, 25), '50%': np.median(v),
                    '75%': np.percentile(v, 75), 'max': np.max(v)}

        def extract_beta_values(df, col_prefix):
            if df.empty:
                return np.array([])
            matching = [c for c in df.columns if c == col_prefix or c.startswith(col_prefix + "_")]
            if not matching:
                return np.array([])
            return df[matching].values.astype(np.float64).ravel()

        sample_cov = self.covariance_results.get(0, None)
        lml0_values  = get_col(likelihood_df, 'lml0')
        lml1_values  = get_col(likelihood_df, 'lml1')
        lrt10_values = get_col(likelihood_df, 'lrt10')
        df10_values  = get_col(likelihood_df, 'df10', default=1)

        print("\n" + "=" * 60)
        print(f"  TEST TYPE: {self.test_type or 'unknown'}")
        print("=" * 60)

        # H0
        print("\nHypothesis 0 (Null)")
        print("─" * 60)
        print("Y ~ 𝓝((A⊗𝙼)𝛂, C₀⊗𝙺 + C₁⊗𝙸)")
        if null_model is not None:
            print()
            print(self._format_null_model_table(
                null_model['beta0'], null_model['beta0_se'],
                null_model['n_covariates'], null_model['n_traits'],
            ))
        if sample_cov is not None:
            d0 = self._extract_matrix_diagonal(sample_cov['C0'])
            d1 = self._extract_matrix_diagonal(sample_cov['C1'])
            if d0 is not None:
                print(f"  diag(C0) = [{', '.join(f'{x:.6f}' for x in d0)}]")
            if d1 is not None:
                print(f"  diag(C1) = [{', '.join(f'{x:.6f}' for x in d1)}]")
        v = lml0_values[np.isfinite(lml0_values)]
        print(f"\n  lml0 = {np.mean(v) if len(v) > 0 else np.nan:.10f}")
        print("─" * 60)

        # H1
        print("\nHypothesis 1")
        print("─" * 60)
        print("Y ~ 𝓝((A⊗𝙼)𝛂 + (A₀⊗G)𝛃₀, s(C₀⊗𝙺 + C₁⊗𝙸))")
        print("             lml         effsizes      effsizes_se")
        print("  " + "-" * 53)
        s1  = compute_stats(lml1_values)
        sb1 = compute_stats(extract_beta_values(beta_df, 'beta1'))
        ss1 = compute_stats(extract_beta_values(beta_df, 'beta1_se'))
        for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            print(f"  {k:>4} {s1[k]:>13.3e} {sb1[k]:>13.3e} {ss1[k]:>13.3e}")

        # H2
        if self.has_h2:
            lml2_values  = get_col(likelihood_df, 'lml2')
            lrt20_values = get_col(likelihood_df, 'lrt20')
            lrt21_values = get_col(likelihood_df, 'lrt21')
            df20_values  = get_col(likelihood_df, 'df20', default=1)
            df21_values  = get_col(likelihood_df, 'df21', default=1)

            print("\nHypothesis 2")
            print("─" * 60)
            print("Y ~ 𝓝((A⊗𝙼)𝛂 + (A₀⊗G)𝛃₀ + (A₁⊗G)𝛃₁, s(C₀⊗𝙺 + C₁⊗𝙸))")
            print("             lml         effsizes      effsizes_se")
            print("  " + "-" * 53)
            s2  = compute_stats(lml2_values)
            sb2 = compute_stats(extract_beta_values(beta_df, 'beta2'))
            ss2 = compute_stats(extract_beta_values(beta_df, 'beta2_se'))
            for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                print(f"  {k:>4} {s2[k]:>13.3e} {sb2[k]:>13.3e} {ss2[k]:>13.3e}")

        # P-values
        print("\nLikelihood-ratio test p-values")
        print("─" * 60)
        pv01 = self._compute_p_values(lrt10_values, df10_values)

        if self.has_h2:
            pv02 = self._compute_p_values(lrt20_values, df20_values)
            pv12 = self._compute_p_values(lrt21_values, df21_values)
            print("          H0 vs H1      H0 vs H2      H1 vs H2")
            print("  " + "-" * 53)
            s01, s02, s12 = compute_stats(pv01), compute_stats(pv02), compute_stats(pv12)
            for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                print(f"  {k:>4} {s01[k]:>13.3e} {s02[k]:>13.3e} {s12[k]:>13.3e}")
        else:
            print("          H0 vs H1")
            print("  " + "-" * 23)
            s01 = compute_stats(pv01)
            for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                print(f"  {k:>4} {s01[k]:>13.3e}")

        print("\n" + "=" * 60)
        print(f"  GWAS analysis complete ({time.time() - start_time:.1f}s)")
        print("=" * 60)