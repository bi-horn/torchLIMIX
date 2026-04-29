'Adapted from https://github.com/limix/limix/blob/master/limix/stats/_kinship.py'

import numpy as np
import pandas as pd
from tqdm import tqdm

def normalize_genotype_matrix(G, verbose=True):
    """
    Normalize a genotype matrix column-wise by subtracting the mean
    and dividing by the standard deviation, matching the standardization
    used in linear_kinship.
    
    Supports both NumPy arrays and Pandas DataFrames.
    
    Parameters
    ----------
    G : array_like or pd.DataFrame
        Samples-by-variants matrix. Can be a NumPy array or a Pandas DataFrame.
    verbose : bool, optional
        ``True`` for showing progress; ``False`` otherwise. Default to ``True``.
        
    Returns
    -------
    normalized_G : array_like or pd.DataFrame
        The column-wise normalized matrix. Returns the same type as input.
        
    Notes
    -----
    Normalization follows the approach in linear_kinship:
    
    .. math::
        𝚇ᵢⱼ = (𝙶ᵢⱼ - 𝑚ⱼ) / 𝑠ⱼ
    
    where 𝑚ⱼ is the mean and 𝑠ⱼ is the population standard deviation (ddof=0).
    NaNs are replaced with column mean before standardization.
    
    Examples
    --------
    >>> from numpy.random import RandomState
    >>> random = RandomState(1)
    >>> X = random.randn(4, 100)
    >>> normalized_X = normalize_genotype_matrix(X, verbose=False)
    >>> print(f"Mean: {np.mean(normalized_X, axis=0)[:5]}")
    >>> print(f"Std: {np.std(normalized_X, axis=0, ddof=0)[:5]}")
    """
    # Determine if input is a DataFrame
    is_dataframe = isinstance(G, pd.DataFrame)
    
    # Convert to NumPy array if necessary
    if is_dataframe:
        columns = G.columns
        index = G.index
        G = G.to_numpy()
    
    # Get shape and initialize normalized output
    (n, p) = G.shape
    normalized_G = np.empty((n, p), dtype=float)
    
    # Determine chunks for processing
    chunks = get_chunks(G)
    
    # Iterate over chunks
    start = 0
    for chunk in tqdm(chunks, desc="Normalization", disable=not verbose):
        end = start + chunk
        g = np.asarray(G[:, start:end], dtype=float)
        
        m = np.nanmean(g, axis=0)
        
        g = np.where(np.isnan(g), m, g)
    
        g = g - m

        s = np.std(g, axis=0, ddof=0)
        
        # Handle constant columns (avoid division by zero)
        s[s == 0] = 1.0
        
        g = g / s
        
        normalized_G[:, start:end] = g
        
        start = end

    if is_dataframe:
        normalized_G = pd.DataFrame(normalized_G, columns=columns, index=index)
    
    return normalized_G


def get_chunks(G):
    """
    Determine chunk sizes for processing.
    """
    siz = G.shape[1] // 100
    sizl = G.shape[1] - siz * 100
    chunks = [siz] * 100
    if sizl > 0:
        chunks += [sizl]
    return tuple(chunks)