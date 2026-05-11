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


def stack_data_from_loaders(tl, vl, ttl=None, index=0, device=None):
    """
    More memory-efficient version that processes one loader at a time.
    
    Args:
        tl: Training DataLoader
        vl: Validation DataLoader  
        ttl: Test DataLoader (optional)
        index: Which element of the batch to extract (0=X_snp/kinship, 1=X_qs, 2=Y)
        device: Device to move tensors to
        
    Returns:
        Stacked tensor from all loaders
    """
    
    def extract_from_loader(loader, loader_name):
        """Extract data from a single loader"""
        if loader is None:
            return None
            
        logger.debug(f"Processing {loader_name} loader (index={index})...")
        batch_data = []
        
        try:
            for batch_idx, batch in enumerate(loader):
                if isinstance(batch, (list, tuple)) and len(batch) > index:
                    data = batch[index]
                    
                    # Move to device if specified
                    if device is not None:
                        data = data.to(device)
                    
                    batch_data.append(data)
                    
                    # Periodic memory cleanup for large datasets
                    if batch_idx % 2 == 0 and batch_idx > 0:
                        torch.cuda.empty_cache() if device == "cuda" else None
                #else:
                #    logger.warning(f"{loader_name} batch {batch_idx} missing index {index}, skipping...")
            
            if batch_data:
                stacked = torch.cat(batch_data, dim=0)
                logger.debug(f"{loader_name} stacked shape: {stacked.shape}")
                return stacked
            else:
                logger.warning(f"No valid data found in {loader_name} loader")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {loader_name} loader: {e}")
            return None
    
    # Extract data from each loader
    loaders_data = []
    
    for loader, name in [(tl, "training"), (vl, "validation"), (ttl, "test")]:
        loader_data = extract_from_loader(loader, name)
        if loader_data is not None:
            loaders_data.append(loader_data)
    
    # Concatenate all loader data
    if loaders_data:
        try:
            final_stacked = torch.cat(loaders_data, dim=0)
            #logger.info(f"Successfully stacked data from {len(loaders_data)} loaders. Final shape: {final_stacked.shape}")
            
            # Clean up intermediate tensors
            del loaders_data
            torch.cuda.empty_cache() if device == "cuda" else None
            
            return final_stacked
        except Exception as e:
            logger.error(f"Error concatenating loader data: {e}")
            return None
    else:
        logger.error(f"No data found to stack for index {index}")
        return None

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
    
    This is a convenience wrapper around get_data_multitask_snp that:
    - Extracts parameters from config['data_param']
    - Returns loaders as separate variables (legacy interface)
    
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
    # Batch shapes
    try:
        batch = next(iter(train_loader))
        shapes = [tuple(b.shape) if hasattr(b, 'shape') else len(b) for b in batch]
        logger.info(f"Batch shapes: {shapes}")
    except StopIteration:
        logger.warning("Train loader is empty!")
    
    # Simulation info check
    if data_config.get('simulated') and data_meta.get('simulation_info') is None:
        logger.warning("simulated=True but simulation_info is None")

def load_and_process_covariates(cov_path, device=None):
    """
    Load and process covariate data from various file formats.
    
    Args:
        cov_path (str): Path to covariate file
        device (torch.device): Device to place tensors on
        
    Returns:
        tuple: (covariate_matrix, sample_ids, covariate_names)
            - covariate_matrix: torch.Tensor of shape (n_samples, n_covariates)
            - sample_ids: list of sample IDs (fid_iid combined)
            - covariate_names: list of covariate column names
    """
    if cov_path is None:
        return None, None, None
        
    cov_path = Path(cov_path)
    if not cov_path.exists():
        raise FileNotFoundError(f"Covariate file not found: {cov_path}")
    
    # Determine file format and separator
    file_ext = cov_path.suffix.lower()
    if file_ext in ['.csv']:
        sep = ','
    elif file_ext in ['.tsv']:
        sep = '\t'
    else:
        # Try to auto-detect separator by examining the first line
        with open(cov_path, 'r') as f:
            first_line = f.readline().strip()
            
            # Count different separators
            tab_count = first_line.count('\t')
            comma_count = first_line.count(',')
            space_count = len(first_line.split()) - 1  # spaces between words
            
            # Choose separator based on which gives most columns
            if tab_count > 0 and tab_count >= comma_count and tab_count >= space_count:
                sep = '\t'
            elif comma_count > 0 and comma_count >= space_count:
                sep = ','
            elif space_count > 0:
                sep = r'\s+'  # Use regex for multiple spaces/whitespace
            else:
                sep = None  # Let pandas auto-detect
    
    logger.info(f"Loading covariate file: {cov_path}")
    
    has_header = detect_header(cov_path, sep)
    
    if has_header:
        # Load with header
        if sep == r'\s+':
            df = pd.read_csv(cov_path, sep=sep, header=0, engine='python')
        else:
            df = pd.read_csv(cov_path, sep=sep, header=0)
        logger.info(f"Loaded covariates with header. Shape: {df.shape}")
        logger.info(f"Column names: {list(df.columns)}")
        
        # Check if first two columns are fid and iid (case insensitive)
        first_two_cols = [col.lower() for col in df.columns[:2]]
        expected_cols = ['fid', 'iid']
        
        if first_two_cols == expected_cols:
            logger.info("Found expected 'fid' and 'iid' columns as first two columns")
            id_cols = df.columns[:2].tolist()
        else:
            logger.warning(f"First two columns are {df.columns[:2].tolist()}, not ['fid', 'iid']. Using them as ID columns anyway.")
            id_cols = df.columns[:2].tolist()
            
        # Create sample IDs by combining first two columns
        sample_ids = df[id_cols[0]].astype(str) + "_" + df[id_cols[1]].astype(str)
        
        # Get covariate columns (everything after first two)
        covariate_cols = df.columns[2:].tolist()
        
    else:
        # Load without header
        if sep == r'\s+':
            df = pd.read_csv(cov_path, sep=sep, header=None, engine='python')
        else:
            df = pd.read_csv(cov_path, sep=sep, header=None)
        logger.info(f"Loaded covariates without header. Shape: {df.shape}")
        
        # Verify we have at least 3 columns (fid, iid, + at least 1 covariate)
        if df.shape[1] < 3:
            raise ValueError(f"Covariate file must have at least 3 columns (fid, iid, covariates), but found {df.shape[1]} columns")
        
        # Use first two columns as IDs
        sample_ids = df.iloc[:, 0].astype(str) + "_" + df.iloc[:, 1].astype(str)
        
        # Get covariate columns (everything after first two)
        covariate_cols = list(range(2, df.shape[1]))
        
    if len(covariate_cols) == 0:
        logger.warning("No covariate columns found after ID columns. Returning None.")
        return None, sample_ids.tolist(), []
    
    # Extract covariate data
    covariate_data = df.iloc[:, 2:] if not has_header else df[covariate_cols]
    
    # Process different data types
    processed_covariates = process_covariate_datatypes(covariate_data)
    
    # Convert to tensor
    covariate_matrix = torch.tensor(processed_covariates, dtype=torch.float32)
    if device is not None:
        covariate_matrix = covariate_matrix.to(device)
    
    covariate_names = covariate_cols if has_header else [f"cov_{i}" for i in range(len(covariate_cols))]
    
    logger.info(f"Processed covariates: {covariate_matrix.shape[0]} samples, {covariate_matrix.shape[1]} covariates")
    logger.info(f"Covariate names: {covariate_names}")
    
    return covariate_matrix, sample_ids.tolist(), covariate_names


def detect_header(file_path, sep):
    """
    Detect if the file has a header by analyzing the first few rows.
    
    Args:
        file_path (Path): Path to the file
        sep (str): Separator used in the file
        
    Returns:
        bool: True if header is detected, False otherwise
    """
    try:
        # Handle regex separator for space-separated files
        read_kwargs = {'engine': 'python'} if sep == r'\s+' else {}
        
        # Read first few rows to analyze
        df_with_header = pd.read_csv(file_path, sep=sep, header=0, nrows=5, **read_kwargs)
        df_without_header = pd.read_csv(file_path, sep=sep, header=None, nrows=5, **read_kwargs)
        
        # Check if first row looks like column names (contains non-numeric strings)
        first_row = df_without_header.iloc[0]
        
        # If first two columns contain strings that could be IDs, likely has header
        first_two_are_strings = all(isinstance(val, str) or pd.isna(val) for val in first_row[:2])
        
        # Check if remaining columns in first row are mixed types (suggesting header)
        remaining_cols = first_row[2:] if len(first_row) > 2 else []
        has_mixed_types = len(remaining_cols) > 0 and any(isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit() 
                                                         for val in remaining_cols if pd.notna(val))
        
        # Additional check: see if column names look reasonable
        if len(df_with_header.columns) >= 2:
            first_two_cols = [str(col).lower() for col in df_with_header.columns[:2]]
            likely_id_names = any(name in ['fid', 'iid', 'id', 'sample', 'individual'] 
                                for col in first_two_cols for name in [col])
            
            if likely_id_names:
                return True
        
        # For numeric data like PCs, if first row is all numeric, probably no header
        if all(pd.to_numeric(val, errors='coerce') is not pd.NA for val in first_row):
            return False
        
        return has_mixed_types
        
    except Exception as e:
        logger.warning(f"Error detecting header: {e}. Assuming no header.")
        return False


def process_covariate_datatypes(covariate_data):
    """
    Process covariate data, handling different data types appropriately.
    
    Args:
        covariate_data (pd.DataFrame): Raw covariate data
        
    Returns:
        np.ndarray: Processed covariate matrix
    """
    processed_data = []
    
    for col in covariate_data.columns:
        col_data = covariate_data[col]
        
        # Handle missing values
        if col_data.isna().any():
            logger.warning(f"Column '{col}' contains missing values. Filling with column mean for numeric or mode for categorical.")
        
        # Try to convert to numeric first
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        
        if numeric_data.notna().sum() > 0.8 * len(col_data):  # If >80% can be converted to numeric
            # Treat as numeric
            if numeric_data.isna().any():
                numeric_data = numeric_data.fillna(numeric_data.mean())
            processed_data.append(numeric_data.values)
            logger.info(f"Column '{col}': treated as numeric")
            
        else:
            # Treat as categorical
            unique_vals = col_data.unique()
            unique_vals = unique_vals[pd.notna(unique_vals)]  # Remove NaN
            
            if len(unique_vals) == 2:
                # Binary categorical - encode as 0/1
                col_data_filled = col_data.fillna(col_data.mode().iloc[0] if not col_data.mode().empty else unique_vals[0])
                encoded = pd.Categorical(col_data_filled).codes.astype(float)
                processed_data.append(encoded)
                logger.info(f"Column '{col}': treated as binary categorical ({unique_vals})")
                
            elif len(unique_vals) <= 10:  # Small number of categories
                # One-hot encode
                col_data_filled = col_data.fillna(col_data.mode().iloc[0] if not col_data.mode().empty else unique_vals[0])
                dummies = pd.get_dummies(col_data_filled, prefix=str(col), drop_first=True)
                for dummy_col in dummies.columns:
                    processed_data.append(dummies[dummy_col].astype(float).values)
                logger.info(f"Column '{col}': one-hot encoded into {len(dummies.columns)} columns")
                
            else:
                # Too many categories - treat as numeric if possible, otherwise skip
                try:
                    # Try label encoding
                    col_data_filled = col_data.fillna(col_data.mode().iloc[0] if not col_data.mode().empty else unique_vals[0])
                    encoded = pd.Categorical(col_data_filled).codes.astype(float)
                    processed_data.append(encoded)
                    logger.info(f"Column '{col}': label encoded ({len(unique_vals)} categories)")
                except Exception as e:
                    logger.warning(f"Column '{col}': skipped due to processing error: {e}")
                    continue
    
    if not processed_data:
        raise ValueError("No valid covariate columns could be processed")
    
    # Stack all processed columns
    result = np.column_stack(processed_data)
    logger.info(f"Final processed covariate matrix shape: {result.shape}")
    
    return result



 
def _create_mmap_results(
    n_snps: int,
    n_covariates: int,
    n_traits: int,
    a1_cols: int,
    tmp_dir: Optional[str] = None,
) -> Dict[str, np.memmap]:
    """
    Create memory-mapped arrays for scan results.
 
    These live on disk and only page into RAM on access,
    so peak RSS stays bounded by the chunk size.
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