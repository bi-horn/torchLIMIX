"""
Regress out batch effects and continuous covariates from phenotype data.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple

def _check_no_nan(df: pd.DataFrame, name: str) -> None:
    nan_counts = df.isna().sum()
    bad = nan_counts[nan_counts > 0]
    if len(bad) > 0:
        details = ", ".join(f"{col}={n}" for col, n in bad.items())
        raise ValueError(
            f"{name} contains NaN values, but imputation is not supported. "
            f"Per-column NaN counts: {details}. "
            f"Drop or impute these rows before calling."
        )
    
# Helpers
def _is_continuous(s: pd.Series) -> bool:
    """Auto-detect: numeric with many distinct values → continuous."""
    s = s.dropna()
    if not pd.api.types.is_numeric_dtype(s) or len(s) == 0:
        return False
    n_unique = s.nunique()
    return n_unique > 10 and (n_unique / len(s)) > 0.2


def _build_design(values: pd.Series, *, force_categorical: bool) -> Optional[np.ndarray]:
    """One-column → design matrix. Mean-centers if continuous, dummy-encodes if categorical.
    Returns None if the column has <2 categories or zero variance (degenerate fit)."""
    if not force_categorical and _is_continuous(values):
        x = values.to_numpy(dtype=np.float64) - values.mean()
        return x[:, None] if x.std() > 0 else None
    # categorical
    if values.nunique() < 2:
        return None
    return pd.get_dummies(values, drop_first=True, dtype=np.float64).to_numpy()


def _fit_apply_residuals(
    y: pd.Series,
    X: np.ndarray,
    train_idx: np.ndarray,
    min_train: int = 10,
) -> Tuple[Optional[np.ndarray], Dict, Optional[pd.Index]]:
    """Fit OLS on (rows ∩ train_idx), return residuals + mean for ALL valid rows.

    Returns
    -------
    corrected_values : ndarray or None 
    info             : dict of stats / skip reason
    valid_index      : pd.Index of rows the corrected_values correspond to
    """
    valid_idx = ~y.isna() & ~np.isnan(X).any(axis=1)
    y_valid   = y[valid_idx]
    X_valid   = X[valid_idx.values]

    valid_pos = y.index.get_indexer(y_valid.index)
    fit_mask  = np.isin(valid_pos, train_idx)
    n_train   = int(fit_mask.sum())

    if n_train < min_train:
        return None, {'status': 'skipped', 'reason': f'too_few_train_samples ({n_train})'}, None

    model = LinearRegression()
    model.fit(X_valid[fit_mask], y_valid.to_numpy()[fit_mask])

    predicted   = model.predict(X_valid)
    residuals   = y_valid.to_numpy() - predicted
    trait_mean  = float(y_valid.to_numpy()[fit_mask].mean())
    corrected   = residuals + trait_mean

    info = {
        'status':         'corrected',
        'n_train':        n_train,
        'n_applied':      int(valid_idx.sum()),
        'r2_train':       float(model.score(X_valid[fit_mask], y_valid.to_numpy()[fit_mask])),
        'original_mean':  trait_mean,
        'original_std':   float(y_valid.std()),
        'corrected_mean': float(corrected.mean()),
        'corrected_std':  float(corrected.std()),
    }
    return corrected, info, y_valid.index

def _write_corrected(
    target_df: pd.DataFrame,
    trait: str,
    valid_index: pd.Index,
    corrected_values: np.ndarray,
) -> None:
    """In-place update of one trait column with corrected values at the given index."""
    col = target_df[trait].copy()
    col.loc[valid_index] = corrected_values
    target_df[trait] = col


# Batch effect correction
def regress_batch_effects(
    pheno_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    per_trait: bool = False,
    train_idx: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Regress out batch effects per trait.

    Parameters
    ----------
    per_trait : bool
        False  -> one shared batch column (the first column of batch_df) for every trait.
        True   -> trait t uses batch_df[t]; traits without a matching column are skipped.
    train_idx : array-like of int, optional
        Positional indices (into pheno_df) used for fitting β. Residuals are computed
        for all valid rows. Defaults to all rows (legacy behavior).
    """
    common = pheno_df.index.intersection(batch_df.index)
    pheno_df = pheno_df.loc[common].copy()
    batch_df = batch_df.loc[common].copy()

    _check_no_nan(pheno_df, "pheno_df")
    _check_no_nan(batch_df, "batch_df")

    if train_idx is None:
        train_idx = np.arange(len(pheno_df))
    train_idx = np.asarray(train_idx, dtype=np.int64)

    mode    = 'per_trait' if per_trait else 'sample_level'
    shared  = None if per_trait else batch_df.columns[0]
    print(f"[batch] mode={mode}, n_samples={len(pheno_df)}, n_train={len(train_idx)}"
          + (f", shared_col='{shared}'" if shared else ""))

    corrected_df = pheno_df.copy()
    stats = {'mode': mode, 'n_samples': len(pheno_df), 'traits': {}}

    for trait in pheno_df.columns:
        # pick the batch column for this trait
        if per_trait:
            if trait not in batch_df.columns:
                stats['traits'][trait] = {'status': 'skipped', 'reason': 'no_batch_column'}
                continue
            batch_col = batch_df[trait]
            batch_name = trait
        else:
            batch_col = batch_df[shared]
            batch_name = shared

        X = _build_design(batch_col, force_categorical=True)
        if X is None:
            stats['traits'][trait] = {'status': 'skipped', 'reason': 'single_category_or_no_variance'}
            continue

        corrected, info, valid_index = _fit_apply_residuals(
            pheno_df[trait], X, train_idx,
        )
        if corrected is None:
            stats['traits'][trait] = info
            print(f"  {trait}: skipped ({info['reason']})")
            continue

        _write_corrected(corrected_df, trait, valid_index, corrected)
        info['batch_variable'] = batch_name
        stats['traits'][trait] = info
        print(f"  {trait}: r2_train={info['r2_train']:.3f}  "
              f"(n_train={info['n_train']}, n_applied={info['n_applied']})")

    return corrected_df, stats

# Continuous and categorical covariate correction
def regress_continuous_covariates(
    pheno_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    train_idx: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Regress out covariates per trait. Each trait gets its own β; the design
    matrix is shared (covariates centered/dummy-encoded once)."""
    common = pheno_df.index.intersection(cov_df.index)
    pheno_df = pheno_df.loc[common].copy()
    cov_df   = cov_df.loc[common].copy()

    _check_no_nan(pheno_df, "pheno_df")
    _check_no_nan(cov_df, "cov_df")

    if train_idx is None:
        train_idx = np.arange(len(pheno_df))
    train_idx = np.asarray(train_idx, dtype=np.int64)

    # build the shared design matrix once
    X_parts, cov_meta = [], {}
    for cov_name in cov_df.columns:
        col = cov_df[cov_name]
        is_cont = _is_continuous(col)
        X_part = _build_design(col, force_categorical=not is_cont)
        if X_part is None:
            cov_meta[cov_name] = {'type': 'skipped', 'reason': 'no_variance'}
            continue
        cov_meta[cov_name] = {
            'type':     'continuous' if is_cont else 'categorical',
            'n_cols':   X_part.shape[1],
        }
        X_parts.append(X_part)

    if not X_parts:
        print("[cov] no usable covariates after design build")
        return pheno_df, {'n_samples': len(pheno_df), 'traits': {}, 'covariate_stats': cov_meta}

    X_full = np.hstack(X_parts)
    print(f"[cov] design={X_full.shape}, n_samples={len(pheno_df)}, n_train={len(train_idx)}")

    corrected_df = pheno_df.copy()
    stats = {
        'n_samples':       len(pheno_df),
        'covariates':      list(cov_df.columns),
        'covariate_stats': cov_meta,
        'traits':          {},
    }

    for trait in pheno_df.columns:
        corrected, info, valid_index = _fit_apply_residuals(
            pheno_df[trait], X_full, train_idx,
        )
        if corrected is None:
            stats['traits'][trait] = info
            print(f"  {trait}: skipped ({info['reason']})")
            continue

        _write_corrected(corrected_df, trait, valid_index, corrected)
        stats['traits'][trait] = info
        print(f"  {trait}: r2_train={info['r2_train']:.3f}  "
              f"(n_train={info['n_train']}, n_applied={info['n_applied']})")

    return corrected_df, stats