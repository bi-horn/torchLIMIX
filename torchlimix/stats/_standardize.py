"""
Phenotype standardization utilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from typing import Union
import warnings

def standardize_data(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    method: str = 'z_score',
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Standardize data using z-score or inverse normal transformation.
    
    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or np.ndarray
        Input data. For DataFrames/2D arrays, each column is standardized independently.
        Must not contain any missing values.
    method : {'z_score', 'int', 'none'}, default 'z_score'
        - 'z_score': Standard normalization (mean=0, std=1)
        - 'int': Inverse normal transform (rank-based gaussianization)
        - 'none': Return data unchanged
        
    Returns
    -------
    Same type as input with standardized values.
    
    Raises
    ------
    RuntimeError
        If data contains missing (NaN) values.
    ValueError
        If method is not recognized.
    TypeError
        If data is not a supported type.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    >>> standardize_data(df, method='z_score')
    >>> standardize_data(df, method='int')
    """
    if method in [None, 'none', '']:
        return data
    
    if method not in ['z_score', 'int']:
        raise ValueError(
            f"method must be 'z_score', 'int', or 'none'. Got: '{method}'"
        )
    
    # Input validation
    if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError(
            f"Input must be DataFrame, Series, or ndarray. Got: {type(data).__name__}"
        )
    
    _check_no_missing(data)
    
    # Apply transformation
    if method == 'z_score':
        return _zscore_transform(data)
    else:  # method == 'int'
        return _int_transform(data)


def _check_no_missing(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> None:
    """
    Check that data contains no missing values.
    
    Raises
    ------
    RuntimeError
        If any NaN values are found.
    """
    if isinstance(data, pd.DataFrame):
        n_missing = data.isna().sum().sum()
        if n_missing > 0:
            # Find which columns have missing values
            missing_per_col = data.isna().sum()
            cols_with_missing = missing_per_col[missing_per_col > 0]
            raise RuntimeError(
                f"Data contains {n_missing} missing value(s) in columns: "
                f"{list(cols_with_missing.index)}. "
                f"NaN values are not valid for this analysis.\n"
                f"Please either:\n"
                f"  1. Impute missing values: df.fillna(df.mean()) or df.interpolate()\n"
                f"  2. Remove rows with missing values: df.dropna()"
            )
    
    elif isinstance(data, pd.Series):
        n_missing = data.isna().sum()
        if n_missing > 0:
            raise RuntimeError(
                f"Data contains {n_missing} missing value(s). "
                f"NaN values are not valid for this analysis.\n"
                f"Please either:\n"
                f"  1. Impute missing values: series.fillna(series.mean())\n"
                f"  2. Remove missing values: series.dropna()"
            )
    
    else:  # np.ndarray
        n_missing = np.isnan(data).sum()
        if n_missing > 0:
            if data.ndim == 2:
                # Find which columns have missing values
                missing_per_col = np.isnan(data).sum(axis=0)
                cols_with_missing = np.where(missing_per_col > 0)[0]
                raise RuntimeError(
                    f"Data contains {n_missing} missing value(s) in columns: "
                    f"{list(cols_with_missing)}. "
                    f"NaN values are not valid for this analysis.\n"
                    f"Please either:\n"
                    f"  1. Impute missing values: np.nan_to_num(arr, nan=np.nanmean(arr, axis=0))\n"
                    f"  2. Remove rows with missing values: arr[~np.isnan(arr).any(axis=1)]"
                )
            else:
                raise RuntimeError(
                    f"Data contains {n_missing} missing value(s). "
                    f"NaN values are not valid for this analysis.\n"
                    f"Please either:\n"
                    f"  1. Impute missing values: np.nan_to_num(arr, nan=np.nanmean(arr))\n"
                    f"  2. Remove missing values: arr[~np.isnan(arr)]"
                )


def _zscore_transform(
    data: Union[pd.DataFrame, pd.Series, np.ndarray]
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Z-score normalization: (x - mean) / std per column."""
    if isinstance(data, pd.DataFrame):
        return _zscore_dataframe(data)
    elif isinstance(data, pd.Series):
        return _zscore_series(data)
    else:
        return _zscore_array(data)


def _zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score transform for DataFrame."""
    result = df.copy().astype(np.float64)
    
    for col in df.columns:
        result[col] = _zscore_1d(df[col].values, col_name=str(col))
    
    return result


def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-score transform for Series."""
    values = _zscore_1d(series.values, col_name=series.name)
    return pd.Series(values, index=series.index, name=series.name)


def _zscore_array(arr: np.ndarray) -> np.ndarray:
    """Z-score transform for ndarray."""
    arr = arr.astype(np.float64)
    
    if arr.ndim == 1:
        return _zscore_1d(arr)
    
    result = arr.copy()
    for j in range(arr.shape[1]):
        result[:, j] = _zscore_1d(arr[:, j], col_name=f"column {j}")
    
    return result


def _zscore_1d(values: np.ndarray, col_name: str = None) -> np.ndarray:
    """Z-score transform for 1D array (no NaN expected)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    if n < 2:
        warnings.warn(
            f"Column {col_name}: Only {n} value(s), returning as-is.",
            UserWarning
        )
        return values.copy()
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    if std < 1e-10:
        warnings.warn(
            f"Column {col_name}: Near-zero variance (std={std:.2e}). "
            f"Setting standardized values to 0.",
            UserWarning
        )
        return np.zeros_like(values)
    
    return (values - mean) / std


def _int_transform(
    data: Union[pd.DataFrame, pd.Series, np.ndarray]
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Inverse normal transformation (rank-based gaussianization)."""
    if isinstance(data, pd.DataFrame):
        return _int_dataframe(data)
    elif isinstance(data, pd.Series):
        return _int_series(data)
    else:
        return _int_array(data)


def _int_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """INT transform for DataFrame."""
    result = df.copy().astype(np.float64)
    
    for col in df.columns:
        result[col] = _int_1d(df[col].values, col_name=str(col))
    
    return result


def _int_series(series: pd.Series) -> pd.Series:
    """INT transform for Series."""
    values = _int_1d(series.values, col_name=series.name)
    return pd.Series(values, index=series.index, name=series.name)


def _int_array(arr: np.ndarray) -> np.ndarray:
    """INT transform for ndarray."""
    arr = arr.astype(np.float64)
    
    if arr.ndim == 1:
        return _int_1d(arr)
    
    result = arr.copy()
    for j in range(arr.shape[1]):
        result[:, j] = _int_1d(arr[:, j], col_name=f"column {j}")
    
    return result


def _int_1d(values: np.ndarray, col_name: str = None) -> np.ndarray:
    """
    Inverse normal transform for 1D array (no NaN expected).
    
    Van der Waerden transformation: Φ⁻¹(r / (n + 1))
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    if n < 2:
        warnings.warn(
            f"Column {col_name}: Only {n} value(s), returning 0.",
            UserWarning
        )
        return np.zeros_like(values)
    
    ranks = rankdata(values, method='average')
    # Van der Waerden: quantiles in (0, 1)
    quantiles = ranks / (n + 1)
    quantiles = np.clip(quantiles, 1e-10, 1 - 1e-10)
    
    # Transform to standard normal quantiles
    return norm.ppf(quantiles)
