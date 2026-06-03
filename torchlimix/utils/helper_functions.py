import torch
from torch import Tensor
import numpy as np
import pandas as pd
from .data_loader import get_data_multitask_snp
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import tempfile, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_snp(
    config: Dict,
    *,
    geno_path: Optional[str] = None,
    pheno_path: Optional[str] = None,
    annot_path: Optional[str] = None,
    batch_path: Optional[str] = None,
    cov_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    predict_geno_path: Optional[str] = None,  
    predict_cov_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """
    Retrieve SNP dataset from config.
    
    Args:
        config: Full configuration dictionary
        geno_path: Path to genotype file (overrides config)
        pheno_path: Path to phenotype file (overrides config)
        annot_path: Path to annotation file (overrides config)
        batch_path: Path to batch file (overrides config)
        cov_path: Path to covariate file (overrides config)
        predict_geno_path: Path to external genotypes for prediction
        verbose: Print progress messages
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, data_meta)
    """
    # Extract data configuration
    data_config = config.get("data_param", {}).copy()
    
    # Pop items that are passed separately
    dset = data_config.pop('dset', 'custom')
    data_path_config = data_config.pop('data_path_config', None)
    
    # Handle vardec flag from analysis type
    if config.get("analysis", "").lower() == "vardec":
        data_config['vardec'] = True
    
    # Log configuration summary
    if verbose:
        _log_data_config(dset, data_config, 
                         geno_path, pheno_path, annot_path, batch_path, cov_path, predict_cov_path)
    
    # Create loaders
    try:
        loaders, data_meta = get_data_multitask_snp(
            dset=dset,
            data_path_config=data_path_config,
            pheno_path=pheno_path,
            geno_path=geno_path,
            annot_path=annot_path,
            batch_path=batch_path,
            cov_path=cov_path,
            output_dir=output_dir,
            predict_geno_path=predict_geno_path,   
            predict_cov_path=predict_cov_path, 
            verbose=verbose,
            **data_config
        )
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise
    
    # Extract individual loaders
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')
    
    if train_loader is None:
        raise RuntimeError("Failed to create training data loader")
    
    if verbose:
        _log_loader_summary(train_loader, val_loader, test_loader, data_meta, data_config)
    
    return train_loader, val_loader, test_loader, data_meta
 


def _log_data_config(dset, data_config, geno_path, pheno_path, annot_path, batch_path, cov_path, predict_cov_path):
    """Log data configuration summary."""
    logger.info("=" * 60)
    logger.info("DATA CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dset}")
    logger.info(f"Simulated: {data_config.get('simulated', False)}")
    logger.info(f"Transform: {data_config.get('transformation_method', 'none')}")
    logger.info(f"Split: train={data_config.get('train_pct', 1.0)}, val={data_config.get('val_pct', 0.0)}")
    
    paths = [('geno', geno_path), ('pheno', pheno_path), ('annot', annot_path),
             ('batch', batch_path), ('cov', cov_path), ('pred_cov', predict_cov_path)]
    path_str = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in paths)
    logger.info(f"Paths: {path_str}")
    
    # Simulation params
    if data_config.get('simulated'):
        sim_str = ", ".join(
            f"{k}={data_config[k]}" 
            for k in ['eta', 'rep_idx', 'num_samples', 'ncausal'] 
            if k in data_config
        )
        logger.info(f"Simulation: {sim_str}")
    
    logger.info("=" * 60)


def _log_loader_summary(train_loader, val_loader, test_loader, data_meta, data_config):
    """Log loader creation summary."""
    try:
        batch = next(iter(train_loader))
        shapes = [tuple(b.shape) if hasattr(b, 'shape') else len(b) for b in batch]
        logger.info(f"Batch shapes: {shapes}")
    except StopIteration:
        logger.warning("Train loader is empty!")
    
    if data_config.get('simulated') and data_meta.get('simulation_info') is None:
        logger.warning("simulated=True but simulation_info is None")

 
def _create_mmap_results(
    n_snps: int,
    n_covariates: int,
    n_traits: int,
    a1_cols: int,
    tmp_dir: Optional[str] = None,
) -> Dict[str, np.memmap]:
    """
    Create memory-mapped arrays for scan results.
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="scan_results_")
 
    shapes = {
        "lml":           (n_snps,),
        "scale":         (n_snps,),
        "effsizes0":     (n_snps, n_covariates, n_traits),
        "effsizes0_se":  (n_snps, n_covariates, n_traits),
        "effsizes1":     (n_snps, 1, a1_cols),
        "effsizes1_se":  (n_snps, 1, a1_cols),
        "pve":           (n_snps, a1_cols),
    }
 
    mmaps = {}
    for name, shape in shapes.items():
        path = os.path.join(tmp_dir, f"{name}.dat")
        mmaps[name] = np.memmap(path, dtype=np.float64, mode="w+", shape=shape)
 
    mmaps["_tmp_dir"] = tmp_dir
    return mmaps
 
 
def _mmap_results_to_tensors(mmaps: Dict[str, np.memmap]) -> Dict[str, Tensor]:
    """Convert mmap results back to CPU tensors (zero-copy where possible)."""
    out = {}
    for key, arr in mmaps.items():
        if key.startswith("_"):
            continue
        out[key] = torch.from_numpy(np.array(arr))
    return out