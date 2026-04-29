#!/usr/bin/env python3
"""
Regress out batch effects and continuous covariates from phenotype data

Key principles:
- Covariates: Mean-centered once, then applied uniformly to ALL phenotypes
- Batch effects: Can be sample-level (one batch per sample) or trait-level (one batch per trait)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple, List
import torch


def detect_variable_type(
    series: pd.Series,
    force_categorical: bool = False,
    force_continuous: bool = False
) -> str:
    """
    Automatically detect if a variable is categorical or continuous
    """
    if force_categorical:
        return 'categorical'
    if force_continuous:
        return 'continuous'
    
    non_na = series.dropna()
    
    if len(non_na) == 0:
        return 'categorical'
    
    is_numeric = pd.api.types.is_numeric_dtype(non_na)
    
    if not is_numeric:
        return 'categorical'
    
    n_unique = non_na.nunique()
    n_samples = len(non_na)
    
    # Heuristic: if >20% unique values and >10 unique values, treat as continuous
    if n_unique > 10 and (n_unique / n_samples) > 0.2:
        return 'continuous'
    else:
        return 'categorical'


def regress_batch_effects(
    pheno_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    per_trait: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Regress out batch effects from phenotype data.
    
    Standard approach for multi-trait:
    - If per_trait=False (default): One batch variable for ALL traits
      (e.g., all traits measured in the same chamber/plate)
    - If per_trait=True: Different batch variable per trait
      (e.g., trait1 measured in batch_trait1, trait2 in batch_trait2)
    
    Args:
        pheno_df: DataFrame with phenotypes (MultiIndex: FID, IID)
        batch_df: DataFrame with batch variables (MultiIndex: FID, IID)
        per_trait: If False, use first batch column for all traits
                  If True, match batch columns to trait names
    
    Returns:
        corrected_df: DataFrame with batch-corrected phenotypes
        stats: Dictionary with correction statistics
    """
    print("\n" + "=" * 70)
    print("BATCH EFFECT CORRECTION")
    print("=" * 70)
    
    # Match samples
    common_samples = pheno_df.index.intersection(batch_df.index)
    pheno_df = pheno_df.loc[common_samples].copy()
    batch_df = batch_df.loc[common_samples].copy()
    
    print(f"\nSamples after matching: {len(common_samples)}")
    print(f"Phenotype traits: {list(pheno_df.columns)}")
    print(f"Batch variables: {list(batch_df.columns)}")
    
    # Determine batch mode
    if per_trait:
        print("\nMode: Per-trait batch correction")
        print("  (each trait has its own batch variable)")
        mode = 'per_trait'
    else:
        print("\nMode: Sample-level batch correction")
        print("  (same batch variable applied to all traits)")
        mode = 'sample_level'
        batch_var = batch_df.columns[0]
        print(f"  Using batch variable: '{batch_var}'")
    
    # Initialize
    corrected_df = pheno_df.copy()
    stats = {
        'n_samples': len(pheno_df),
        'mode': mode,
        'traits': {}
    }
    
    print("\nCorrecting batch effects...")
    print("-" * 70)
    
    for trait in pheno_df.columns:
        y = pheno_df[trait].copy()
        
        # Determine which batch variable to use
        if mode == 'sample_level':
            batch_values = batch_df[batch_var]
            batch_var_name = batch_var
        else:  # per_trait
            if trait in batch_df.columns:
                batch_values = batch_df[trait]
                batch_var_name = trait
            else:
                print(f"  {trait}: No batch variable found, skipping")
                stats['traits'][trait] = {'status': 'skipped', 'reason': 'no_batch_variable'}
                continue
        
        # Remove NaN
        valid_idx = ~batch_values.isna() & ~y.isna()
        y_valid = y[valid_idx]
        batch_valid = batch_values[valid_idx]
        
        if len(y_valid) < 10:
            print(f"  {trait}: Skipped (too few samples: {len(y_valid)})")
            stats['traits'][trait] = {'status': 'skipped', 'reason': 'too_few_samples'}
            continue
        
        # Detect variable type (should be categorical for batch)
        var_type = detect_variable_type(batch_valid, force_categorical=True)
        
        # Get unique categories
        unique_vals = batch_valid.unique()
        n_categories = len(unique_vals)
        
        if n_categories < 2:
            print(f"  {trait}: Skipped (only {n_categories} category)")
            stats['traits'][trait] = {'status': 'skipped', 'reason': 'single_category'}
            continue
        
        # Store original mean
        original_mean = y_valid.mean()
        original_std = y_valid.std()
        
        # Create dummy variables
        X = pd.get_dummies(batch_valid, drop_first=True).values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y_valid.values)
        
        # Get residuals and add back mean
        predicted = model.predict(X)
        residuals = y_valid.values - predicted
        corrected_values = residuals + original_mean
        
        # Update corrected dataframe
        corrected = y.copy()
        corrected.loc[y_valid.index] = corrected_values
        corrected_df[trait] = corrected
        
        # Statistics
        variance_explained = model.score(X, y_valid.values)
        
        stats['traits'][trait] = {
            'status': 'corrected',
            'batch_variable': batch_var_name,
            'n_categories': n_categories,
            'categories': list(unique_vals),
            'n_samples': len(y_valid),
            'variance_explained': variance_explained,
            'original_mean': original_mean,
            'original_std': original_std,
            'corrected_mean': corrected_values.mean(),
            'corrected_std': corrected_values.std()
        }
        
        print(f"  {trait}: Corrected ({n_categories} categories, R²={variance_explained:.3f})")
    
    print("-" * 70)
    print("Batch correction complete")
    print("=" * 70)
    
    return corrected_df, stats


def regress_continuous_covariates(
    pheno_df: pd.DataFrame,
    cov_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Regress out continuous covariates from phenotype data.
    
    Standard multi-trait approach:
    1. Mean-center all covariates ONCE (for consistency across traits)
    2. Fit SEPARATE regression model for EACH trait
    3. Each trait gets trait-specific covariate effects
    
    This ensures:
    - Covariates are standardized consistently
    - Each trait has its own covariate relationships
    - Standard approach used in GWAS pipelines
    
    Args:
        pheno_df: DataFrame with phenotypes (MultiIndex: FID, IID)
        cov_df: DataFrame with covariates (MultiIndex: FID, IID)
    
    Returns:
        corrected_df: DataFrame with covariate-corrected phenotypes
        stats: Dictionary with correction statistics
    """
    print("\n" + "=" * 70)
    print("COVARIATE CORRECTION")
    print("=" * 70)
    
    # Match samples
    common_samples = pheno_df.index.intersection(cov_df.index)
    pheno_df = pheno_df.loc[common_samples].copy()
    cov_df = cov_df.loc[common_samples].copy()
    
    print(f"\nSamples after matching: {len(common_samples)}")
    print(f"Phenotype traits: {list(pheno_df.columns)}")
    print(f"Covariates: {list(cov_df.columns)}")
    
    # Initialize
    corrected_df = pheno_df.copy()
    stats = {
        'n_samples': len(pheno_df),
        'n_traits': len(pheno_df.columns),
        'covariates': list(cov_df.columns),
        'traits': {},
        'covariate_stats': {}
    }
    
    print("\n" + "-" * 70)
    print("STEP 1: Mean-centering covariates (for consistency)")
    print("-" * 70)
    
    # Mean-center all covariates ONCE
    cov_centered = cov_df.copy()
    X_parts = []
    
    for cov_name in cov_df.columns:
        cov_values = cov_df[cov_name]
        
        # Detect type
        var_type = detect_variable_type(cov_values)
        
        if var_type == 'continuous':
            # Mean-center
            cov_mean = cov_values.mean()
            cov_std = cov_values.std()
            cov_centered[cov_name] = cov_values - cov_mean
            
            stats['covariate_stats'][cov_name] = {
                'type': 'continuous',
                'mean': cov_mean,
                'std': cov_std,
                'min': cov_values.min(),
                'max': cov_values.max()
            }
            
            print(f"  {cov_name}: continuous, mean-centered (μ={cov_mean:.3f}, σ={cov_std:.3f})")
            X_parts.append(cov_centered[cov_name].to_frame())
        else:
            # Categorical - create dummy variables
            n_cat = cov_values.nunique()
            print(f"  {cov_name}: categorical ({n_cat} categories)")
            
            stats['covariate_stats'][cov_name] = {
                'type': 'categorical',
                'n_categories': n_cat,
                'categories': list(cov_values.unique())
            }
            
            # Use original values for dummy creation
            X_dummy = pd.get_dummies(cov_values, drop_first=True)
            X_parts.append(X_dummy)
    
    # Build design matrix (same for all traits, but coefficients will differ)
    X_full = pd.concat(X_parts, axis=1)
    
    print(f"\nDesign matrix shape: {X_full.shape}")
    print(f"  ({X_full.shape[0]} samples × {X_full.shape[1]} covariate predictors)")
    
    print("\n" + "-" * 70)
    print("STEP 2: Fitting separate regression for EACH trait")
    print("-" * 70)
    print("(Each trait gets its own covariate coefficients)")
    print("-" * 70)
    
    # Fit SEPARATE model for EACH trait
    for trait in pheno_df.columns:
        y = pheno_df[trait].copy()
        
        # Remove NaN
        valid_idx = ~y.isna() & X_full.notna().all(axis=1)
        y_valid = y[valid_idx]
        X_valid = X_full[valid_idx].values
        
        if len(y_valid) < 10:
            print(f"  {trait}: Skipped (too few samples: {len(y_valid)})")
            stats['traits'][trait] = {'status': 'skipped', 'reason': 'too_few_samples'}
            continue
        
        # Store original statistics
        original_mean = y_valid.mean()
        original_std = y_valid.std()
        
        # Fit TRAIT-SPECIFIC model
        model = LinearRegression()
        model.fit(X_valid, y_valid.values)
        
        # Get residuals and add back original mean
        predicted = model.predict(X_valid)
        residuals = y_valid.values - predicted
        corrected_values = residuals + original_mean
        
        # Update corrected dataframe
        corrected = y.copy()
        corrected.loc[y_valid.index] = corrected_values
        corrected_df[trait] = corrected
        
        # Statistics
        variance_explained = model.score(X_valid, y_valid.values)
        
        # Store coefficients for reporting
        coef_dict = {}
        for i, col_name in enumerate(X_full.columns):
            coef_dict[col_name] = model.coef_[i]
        
        stats['traits'][trait] = {
            'status': 'corrected',
            'n_samples': len(y_valid),
            'variance_explained': variance_explained,
            'original_mean': original_mean,
            'original_std': original_std,
            'corrected_mean': corrected_values.mean(),
            'corrected_std': corrected_values.std(),
            'coefficients': coef_dict,
            'intercept': model.intercept_
        }
        
        print(f"  {trait}: R²={variance_explained:.3f} "
              f"(trait-specific coefficients fitted)")
    
    print("-" * 70)
    print("Covariate correction complete")
    print("  → Each trait corrected with trait-specific covariate effects")
    print("=" * 70)
    
    return corrected_df, stats

def save_correction_report(stats: Dict, output_path: str):
    """Save detailed correction statistics to a file"""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PHENOTYPE CORRECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Batch correction
        if 'batch' in stats:
            f.write("BATCH CORRECTION\n")
            f.write("-" * 70 + "\n")
            batch_stats = stats['batch']
            f.write(f"Mode: {batch_stats['mode']}\n")
            f.write(f"Samples: {batch_stats['n_samples']}\n\n")
            
            for trait, trait_stats in batch_stats['traits'].items():
                f.write(f"{trait}:\n")
                if trait_stats['status'] == 'corrected':
                    f.write(f"  Status: Corrected\n")
                    f.write(f"  Batch variable: {trait_stats['batch_variable']}\n")
                    f.write(f"  Categories: {trait_stats['n_categories']} "
                           f"{trait_stats['categories']}\n")
                    f.write(f"  Variance explained: {trait_stats['variance_explained']:.1%}\n")
                    f.write(f"  Original: mean={trait_stats['original_mean']:.4f}, "
                           f"sd={trait_stats['original_std']:.4f}\n")
                    f.write(f"  Corrected: mean={trait_stats['corrected_mean']:.4f}, "
                           f"sd={trait_stats['corrected_std']:.4f}\n")
                else:
                    f.write(f"  Status: Skipped ({trait_stats['reason']})\n")
                f.write("\n")
        
        # Covariate correction
        if 'covariate' in stats:
            f.write("\n" + "=" * 70 + "\n")
            f.write("COVARIATE CORRECTION\n")
            f.write("-" * 70 + "\n")
            cov_stats = stats['covariate']
            f.write(f"Samples: {cov_stats['n_samples']}\n")
            f.write(f"Traits: {cov_stats['n_traits']}\n")
            f.write(f"Covariates: {', '.join(cov_stats['covariates'])}\n\n")
            
            f.write("Covariate statistics (mean-centering applied):\n")
            for cov_name, cov_stat in cov_stats['covariate_stats'].items():
                if cov_stat['type'] == 'continuous':
                    f.write(f"  {cov_name}: continuous, μ={cov_stat['mean']:.3f}, "
                           f"σ={cov_stat['std']:.3f}\n")
                else:
                    f.write(f"  {cov_name}: categorical, {cov_stat['n_categories']} categories\n")
            
            f.write("\nPer-trait regression results:\n")
            f.write("(Each trait has trait-specific covariate coefficients)\n\n")
            
            for trait, trait_stats in cov_stats['traits'].items():
                if trait_stats['status'] == 'corrected':
                    f.write(f"{trait}:\n")
                    f.write(f"  Variance explained: R²={trait_stats['variance_explained']:.3f}\n")
                    f.write(f"  Intercept: {trait_stats['intercept']:.4f}\n")
                    f.write(f"  Coefficients:\n")
                    for cov_name, coef in trait_stats['coefficients'].items():
                        f.write(f"    {cov_name}: {coef:.4f}\n")
                    f.write(f"  Original: mean={trait_stats['original_mean']:.4f}, "
                           f"sd={trait_stats['original_std']:.4f}\n")
                    f.write(f"  Corrected: mean={trait_stats['corrected_mean']:.4f}, "
                           f"sd={trait_stats['corrected_std']:.4f}\n")
                else:
                    f.write(f"{trait}: Skipped ({trait_stats['reason']})\n")
                f.write("\n")
    
    print(f"\nSaved correction report to: {output_path}")