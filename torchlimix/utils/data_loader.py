import json
import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
from pathlib import Path
from importlib import resources
import numpy as np
import pandas as pd
import torch
import warnings
import gc
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from torchlimix.stats._normalize_geno import normalize_genotype_matrix
from torchlimix.stats._standardize import standardize_data
from torchlimix.utils._genotype_preparation import prepare_kinship_pipeline_torch
from torchlimix.utils.regress_effects import (
    regress_batch_effects,
    regress_continuous_covariates
)
from torchlimix.utils.simulator import PhenoSimulator
from torchlimix.utils.vardec_simulator import VarDecSimulator
torch.set_default_dtype(torch.float64)

_EXCEL_EXTENSIONS = {'.xlsx', '.xls'}
_DELIMITED_EXTENSIONS = {'.csv', '.tsv', '.txt'}
_SUPPORTED_EXTENSIONS = _EXCEL_EXTENSIONS | _DELIMITED_EXTENSIONS

@dataclass
class DataPaths:
    """File paths for dataset loading."""
    pheno: Optional[str] = None
    geno: Optional[str] = None
    annot: Optional[str] = None
    batch: Optional[str] = None
    cov: Optional[str] = None
    output_dir: Optional[str] = None
    config: Optional[str] = None  # JSON config for standard datasets
    
    def validate(self, dset: str) -> None:
        """Validate that required paths are provided."""
        if self.config is None and (self.pheno is None or self.geno is None):
            raise ValueError(
                "Either provide config path with dset, "
                "or provide pheno and geno paths directly"
            )
        if self.config is not None and dset is None:
            raise ValueError("dset must be specified when using config")

@dataclass
class SplitConfig:
    """Train/val/test split configuration."""
    train_pct: float = 0.7
    val_pct: float = 0.15
    seed: int = 42
    
    @property
    def test_pct(self) -> float:
        return 1.0 - self.train_pct - self.val_pct

@dataclass
class SimulationConfig:
    """Parameters for phenotype simulation."""
    enabled: bool = False
    num_samples: int = 250
    eta: float = 0.5
    ncausal: int = 1
    corr_bounds: float = 0.0
    use_heterogeneity: bool = False
    num_tasks: int = 2
    reference_trait: Optional[str] = None
    rep_idx: Optional[int] = 0
    vardec: bool = False
    vardec_scenario: int = 0

@dataclass
class CorrectionConfig:
    """Phenotype correction settings."""
    regress_batch: bool = True
    regress_covariates: bool = True
    per_trait_batch: bool = True # every trait gets its own batch column
    transformation: str = "none"  # "none", "int", "z_score"
    
    def validate(self) -> None:
        valid_transforms = {"none", "int", "z_score"}
        if self.transformation not in valid_transforms:
            raise ValueError(f"transformation must be one of {valid_transforms}")

def get_package_root() -> Path:
    """Get the root directory of the torchLIMIX package."""
    return Path(resources.files("torchlimix")).parent

def resolve_path_placeholders(path: str) -> str:
    """Replace ${PACKAGE_ROOT} placeholder with actual package root path."""
    if path is None:
        return None
    
    package_root = str(get_package_root())
    resolved = path.replace("${PACKAGE_ROOT}", package_root)
    resolved = os.path.expanduser(resolved)
    
    return resolved

def _detect_delimiter(path: str) -> str:
    """
    Auto-detect the column delimiter of a text file.

    Looks into the first line for the most common separator among
    tab, comma, semicolon, and space. The file extension is used
    only to break ties:
      .csv  → comma preferred on ties
      .tsv  → tab preferred on ties
      .txt  → tab preferred on ties
    """
    with open(path, 'r') as f:
        first_line = f.readline()

    ext = Path(path).suffix.lower()

    candidates = [
        ('\t', first_line.count('\t')),
        (',',  first_line.count(',')),
        (';',  first_line.count(';')),
        (' ',  first_line.count(' ')),
    ]

    # Extension-based tiebreaker: add 1 to the preferred delimiter
    preferred = {'csv': ',', '.csv': ',', '.tsv': '\t', '.txt': '\t'}
    bias_char = preferred.get(ext, '\t')
    candidates = [(d, n + (1 if d == bias_char else 0)) for d, n in candidates]

    best_delim, best_count = max(candidates, key=lambda x: x[1])

    if best_count == 0:
        return '\t'

    return best_delim

def _read_tabular_file(
    path: str,
    *,
    header: Optional[bool] = None,
    index_col: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Read a tabular file into a DataFrame, supporting:
      .csv  – comma-separated (or auto-detected delimiter)
      .tsv  – tab-separated
      .txt  – auto-detected delimiter
      .xlsx / .xls – Excel (first sheet)

    Parameters
    ----------
    path : str
        File path.
    header : bool or None
        If None the function auto-detects whether the first row is a
        header by checking if the first cell looks like a known ID column
        name (fid, f.id, familyid, sample, id, iid, …).
    index_col : int, str, list, or None
        Passed through to pandas read functions. Default None (no index).

    Returns
    -------
    pd.DataFrame
    """
    ext = Path(path).suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}' for {path}. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    if ext in _EXCEL_EXTENSIONS:
        df = pd.read_excel(path, sheet_name=0, header=0, index_col=index_col)
        return df
    
    delim = _detect_delimiter(path)

    # Auto-detect header
    if header is None:
        with open(path, 'r') as f:
            first_line = f.readline().strip()
        first_token = first_line.split(delim)[0].strip().lower()
        _header_markers = {
            'fid', 'f.id', 'familyid', 'sample', 'id', 'iid',
            'chrom', 'chr', 'chromosome', 'pos', 'position', 'snp',
        }
        header_row = 0 if first_token in _header_markers else None
    else:
        header_row = 0 if header else None

    df = pd.read_csv(path, sep=delim, header=header_row, index_col=index_col)
    return df


# Shared genotype loaders (used by both training and prediction paths)
def _load_geno_plink(path: str) -> Tuple[np.ndarray, pd.MultiIndex, pd.DataFrame]:
    """PLINK .bed/.bim/.fam triplet.

    Returns
    -------
    genotypes : np.ndarray, shape (n_samples, n_snps)
    sample_index : pd.MultiIndex  (fid, iid)
    bim : pd.DataFrame            SNP annotation from the .bim file
    """
    from pandas_plink import read_plink

    stem = str(Path(path).with_suffix(''))
    bim, fam, bed = read_plink(stem)
    geno = bed.compute().T  # (n_samples, n_snps)

    fid = fam.fid.astype(int).values
    iid = fam.iid.astype(int).values
    index = pd.MultiIndex.from_arrays([fid, iid], names=['fid', 'iid'])

    return np.asarray(geno, dtype=np.float64), index, bim


def _load_geno_hdf5(path: str) -> Tuple[np.ndarray, pd.MultiIndex, None]:
    """HDF5 file with datasets 'genotypes' and optionally 'fid'/'iid'."""
    import h5py

    with h5py.File(path, 'r') as f:
        if 'genotypes' not in f:
            raise KeyError(
                f"HDF5 file must contain a 'genotypes' dataset. "
                f"Found: {list(f.keys())}"
            )
        geno = f['genotypes'][:]

        if 'fid' in f and 'iid' in f:
            fid = f['fid'][:].astype(int)
            iid = f['iid'][:].astype(int)
        else:
            fid = np.arange(1, geno.shape[0] + 1)
            iid = fid.copy()

    index = pd.MultiIndex.from_arrays([fid, iid], names=['fid', 'iid'])
    return geno.astype(np.float64), index, None


def _load_geno_delimited(path: str) -> Tuple[np.ndarray, pd.MultiIndex, None]:
    """Delimited text (.csv/.tsv/.txt) with fid+iid columns."""
    df = _read_tabular_file(path)
    df = MultitaskDatasetSNP._set_fid_iid_index(df)
    return df.values.astype(np.float64), df.index, None


def _load_geno_npz(path: str) -> Tuple[np.ndarray, pd.MultiIndex, None]:
    """NumPy .npz archive (prediction genotypes only).

    Expected keys: 'genotypes', and optionally 'fid'/'iid'.
    """
    data = np.load(path, allow_pickle=False)

    if 'genotypes' not in data:
        raise KeyError(
            f"NPZ file must contain a 'genotypes' array. "
            f"Found keys: {list(data.keys())}"
        )

    geno = data['genotypes']

    if 'fid' in data and 'iid' in data:
        fid = data['fid'].astype(int)
        iid = data['iid'].astype(int)
    else:
        fid = np.arange(1, geno.shape[0] + 1)
        iid = fid.copy()
        warnings.warn(
            "NPZ file has no 'fid'/'iid' arrays — auto-generating IDs.",
            stacklevel=3,
        )

    index = pd.MultiIndex.from_arrays([fid, iid], names=['fid', 'iid'])
    return geno, index, None


def _load_genotype_file(
    path: str,
    *,
    allow_npz: bool = False,
) -> Tuple[np.ndarray, pd.MultiIndex, Optional[pd.DataFrame]]:
    """Unified genotype file loader.

    Supported formats
    -----------------
    .bed             PLINK binary  (reads .bed/.bim/.fam triplet)
    .h5 / .hdf5      HDF5          (datasets: 'genotypes', 'fid', 'iid')
    .csv/.tsv/.txt   Delimited text (fid, iid + SNP columns)
    .npz             NumPy archive (prediction only, when *allow_npz=True*)

    Returns
    -------
    genotypes : np.ndarray, shape (n_samples, n_snps)
    sample_index : pd.MultiIndex  (fid, iid)
    bim : pd.DataFrame or None    SNP annotation (only from PLINK)
    """
    p = Path(path)
    ext = p.suffix.lower()

    # PLINK: explicit .bed extension, or extensionless prefix with triplet
    is_plink = (
        ext == '.bed'
        or (ext == '' and all(p.with_suffix(s).exists()
                              for s in ['.bed', '.bim', '.fam']))
    )
    if is_plink:
        return _load_geno_plink(path)

    if ext in ('.h5', '.hdf5'):
        return _load_geno_hdf5(path)

    if ext in ('.csv', '.tsv', '.txt'):
        return _load_geno_delimited(path)

    if ext == '.npz':
        if not allow_npz:
            raise ValueError(
                "NPZ format is only supported for prediction genotypes, "
                "not for training genotypes."
            )
        return _load_geno_npz(path)

    supported = ['.bed', '.h5', '.hdf5', '.csv', '.tsv', '.txt']
    if allow_npz:
        supported.append('.npz')
    raise ValueError(
        f"Unsupported genotype format '{ext}'. Supported: {sorted(supported)}"
    )

class MultitaskDatasetSNP:
    """
    Multi-task SNP dataset with phenotype/genotype loading, corrections, and QS decomposition.
    """
    
    # Standard dataset configurations
    STANDARD_DATASETS = {"thaliana_horton", "thaliana_1001"}
    
    def __init__(
        self,
        dset: str,
        split: str = "train",
        *,
        pheno_path: Optional[str] = None,
        geno_path: Optional[str] = None,
        train_pct: float = 0.7,
        seed: int = 42,
        verbose: bool = True,
        # Configuration objects (override individual args if provided)
        paths: Optional[DataPaths] = None,
        split_config: Optional[SplitConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
        correction_config: Optional[CorrectionConfig] = None,
        **kwargs
    ):
        """
        Initialize MultitaskDatasetSNP.
        
        Args:
            dset: Dataset name ('thaliana_horton', 'thaliana_1001', or custom)
            split: Data split ('train', 'val', 'test')
            pheno_path: Direct path to phenotype file
            geno_path: Direct path to genotype file  
            train_pct: Training set proportion (convenience arg)
            seed: Random seed (convenience arg)
            root: Root path for annotation file output. Needed for plotting later.
            verbose: Print progress messages
            paths: DataPaths configuration object
            split_config: SplitConfig object
            sim_config: SimulationConfig object
            correction_config: CorrectionConfig object
            **kwargs: Legacy parameter support
        """
        self.dset = dset
        self.split = split
        self.verbose = verbose
        
        # Build configuration objects from args or use provided
        self._init_configs(
            paths=paths,
            split_config=split_config,
            sim_config=sim_config,
            correction_config=correction_config,
            pheno_path=pheno_path,
            geno_path=geno_path,
            train_pct=train_pct,
            seed=seed,
            **kwargs
        )
        
        self._rng = np.random.RandomState(seed)
        
        # Initialize state
        self._init_state()
        
        # Load data pipeline
        self._load_pipeline()
    
    def _init_configs(
        self,
        paths: Optional[DataPaths],
        split_config: Optional[SplitConfig],
        sim_config: Optional[SimulationConfig],
        correction_config: Optional[CorrectionConfig],
        **kwargs
    ) -> None:
        """Build configuration objects from various input sources."""

        # DataPaths
        if paths is not None:
            self.paths = paths
        else:
            self.paths = DataPaths(
                pheno=kwargs.get('pheno_path'),
                geno=kwargs.get('geno_path'),
                annot=kwargs.get('annot_path'),
                batch=kwargs.get('batch_path'),
                cov=kwargs.get('cov_path'),
                output_dir=kwargs.get('output_dir'),
                config=kwargs.get('data_path_config'),
            )
        
        # SplitConfig
        if split_config is not None:
            self.split_config = split_config
        else:
            self.split_config = SplitConfig(
                train_pct=kwargs.get('train_pct', 0.7),
                val_pct=kwargs.get('val_pct', 0.15),
                seed=kwargs.get('seed', 42),
            )
        
        # SimulationConfig
        if sim_config is not None:
            self.sim_config = sim_config
        else:
            self.sim_config = SimulationConfig(
                enabled=kwargs.get('simulated', False),
                num_samples=kwargs.get('num_samples', 250),
                eta=kwargs.get('eta', 0.5),
                ncausal=kwargs.get('ncausal', 1),
                corr_bounds=kwargs.get('corr_bounds', 0),
                use_heterogeneity=kwargs.get('use_heterogeneity', False),
                num_tasks=kwargs.get('num_tasks', 2),
                reference_trait=kwargs.get('reference_trait'),
                rep_idx=kwargs.get('rep_idx', 0),
                vardec=kwargs.get('vardec', False),
                vardec_scenario=kwargs.get('vardec_scenario', 0),
            )
        
        # CorrectionConfig
        if correction_config is not None:
            self.correction_config = correction_config
        else:
            self.correction_config = CorrectionConfig(
                regress_batch=kwargs.get('regress_out_batch_effects', True),
                regress_covariates=kwargs.get('regress_out_covariates', True),
                per_trait_batch=kwargs.get('per_trait_batch', True),
                transformation=kwargs.get('transformation_method', 'none'),
            )
        
        # Validate
        self.paths.validate(self.dset)
        self.correction_config.validate()
    
    def _init_state(self) -> None:
        """Initialize internal state variables."""
        self._df = None
        self._info = None
        self._G_norm = None
        self._G_scaling = None
        self.trait_variances = None
        self.trait_covariances = None
        self.covariate_matrix = None
        self.covariate_names = None
        self.correction_metadata = self._empty_correction_metadata()
    
    def _empty_correction_metadata(self) -> Dict:
        """Create empty correction metadata structure."""
        return {
            'batch_correction': {'applied': False, 'file': None},
            'covariate_correction': {'applied': False, 'file': None},
            'transformations': {'method': self.correction_config.transformation, 'applied': False}
        }
    
    def _log(self, msg: str) -> None:
        """Conditional logging."""
        if self.verbose:
            print(msg)
    
    def _load_pipeline(self) -> None:
        import gc

        self._resolve_paths()
        self.gen_data_standard, self.snp_positions = self._load_genotype()
        self.df = self._load_phenotype()          # raw phenotype, no transforms yet
        self._align_samples()

        self.total_samples = len(self.df)
        self._create_splits()
        train_idx = np.asarray(self.split_indices['train'], dtype=np.int64)

        self._apply_transformations(train_idx)    # fits on train, applies to all
        self._compute_phenotype_stats()          

        if (self.correction_config.regress_batch or
            self.correction_config.regress_covariates or
            self.paths.cov):
            self._apply_corrections(train_idx)

        self._compute_qs()
        self._set_current_split()

        if hasattr(self, 'simulated_data_class'):
            self.simulated_data_class = None
        if hasattr(self, 'Xr'):
            self.Xr = None
        if hasattr(self, 'global_indices'):
            self.global_indices = None

        gc.collect()

    def _resolve_paths(self) -> None:
        """Resolve data file paths from config or direct paths."""
        
        # Resolve direct paths if provided
        if self.paths.geno:
            self.paths.geno = resolve_path_placeholders(self.paths.geno)
            self._log(f"[INFO] Using direct paths")
            return
        
        if not self.paths.config:
            raise ValueError("No paths configured")
        
        # Resolve the config path itself first
        config_path = resolve_path_placeholders(self.paths.config)
        
        # Load and resolve all paths in config
        with open(config_path, 'r') as f:
            config_text = f.read()
        
        # Replace placeholder in the entire config file
        config_text = config_text.replace("${PACKAGE_ROOT}", str(get_package_root()))
        config = json.loads(config_text)
        
        # Map dataset names to config keys (just the geno key now)
        path_mappings = {
            "thaliana_horton": "thaliana_horton_geno_path",
            "thaliana_1001": "thaliana_1001_geno_path",
        }
        
        if self.dset in path_mappings:
            geno_key = path_mappings[self.dset]
            self.paths.geno = config[geno_key]
            if self.dset == "thaliana_horton" and not self.paths.batch:
                self.paths.batch = config.get("thaliana_horton_batch_path")
        else:
            self.paths.geno = config.get(f"{self.dset}_geno_path")
        
        self._log(f"[INFO] Resolved paths from config: {config_path}")
    
    # Genotype loading
    def _load_genotype_source(self) -> Tuple[pd.DataFrame, Any]:
        """Load genotype from any supported file format.

        Uses the shared ``_load_genotype_file`` dispatcher which supports
        PLINK (.bed), HDF5 (.h5/.hdf5) and delimited text (.csv/.tsv/.txt).
        """
        geno_arr, index, bim = _load_genotype_file(self.paths.geno)
        gen_data = pd.DataFrame(geno_arr, index=index)

        # Fill NaN with column means (no-op when there are none)
        if gen_data.isna().any().any():
            gen_data = gen_data.fillna(gen_data.mean())

        return gen_data, bim
    
    def _load_genotype(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load and normalize genotype data."""

        # Load from source
        gen_data, bim = self._load_genotype_source()
        snp_positions = self._load_snp_positions(bim)

        self._log(f"[INFO] Loaded genotype: {gen_data.shape}")

        if self.sim_config.enabled and self.dset in ["thaliana_horton", "thaliana_1001"]:
            gen_data, snp_positions = self._subsample_for_simulation(
                gen_data, snp_positions
            )
            self._log(f"[INFO] Subsampled to: {gen_data.shape}")

        # Store per-SNP stats for prediction normalisation
        self._geno_raw_mean = gen_data.mean(axis=0)
        self._geno_raw_std  = gen_data.std(axis=0, ddof=0)
        self._geno_raw_std[self._geno_raw_std == 0] = 1.0

        gen_data_standard = normalize_genotype_matrix(gen_data)

        self._log(f"[INFO] Normalized genotype: mean={gen_data_standard.values.mean():.4f}, "
                  f"std={gen_data_standard.values.std():.4f}")

        return gen_data_standard, snp_positions 

    def normalize_new_genotypes(self, new_geno: pd.DataFrame) -> torch.Tensor:
        """Normalise external genotypes using *training* per-SNP statistics.
    
        The transformation applied is:
            X_norm = (X_raw - train_mean) / train_std
    
        followed by the same G_stable scaling (division by G_norm) that the
        training genotypes received in _compute_qs.
    
        Parameters
        ----------
        new_geno : pd.DataFrame
            Raw genotype matrix, shape (n_new_samples, n_snps).
            Column order must match the training genotype (same SNP set).
    
        Returns
        -------
        torch.Tensor
            Float64 tensor, shape (n_new_samples, n_snps), normalised and
            G_stable-scaled — ready to feed into the model.
    
        Raises
        ------
        RuntimeError
            If training stats were not stored (dataset not loaded properly).
        ValueError
            If the number of SNPs does not match the training data.
        """
        if not hasattr(self, '_geno_raw_mean') or self._geno_raw_mean is None:
            raise RuntimeError(
                "Training normalization stats not available. "
                "Ensure the dataset was loaded before calling this method."
            )
    
        n_train_snps = len(self._geno_raw_mean)
        n_new_snps = new_geno.shape[1]
        if n_new_snps != n_train_snps:
            raise ValueError(
                f"SNP count mismatch: training has {n_train_snps} SNPs, "
                f"prediction genotype has {n_new_snps}."
            )
    
        normed = (new_geno.values - self._geno_raw_mean.values) / self._geno_raw_std.values
        normed = np.nan_to_num(normed, nan=0.0)
        return torch.as_tensor(normed, dtype=torch.float64)
        
    def _subsample_for_simulation(
        self,
        gen_data: pd.DataFrame,
        snp_positions: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Subsample or stack individuals for simulation."""
        n_available = len(gen_data)
        n_target = self.sim_config.num_samples

        if n_available == n_target:
            return gen_data, snp_positions

        gen_data_sorted = gen_data.sort_index()

        if n_available > n_target:
            indices = np.sort(self._rng.choice(n_available, size=n_target, replace=False))
            sampled_ids = gen_data_sorted.index[indices]
            self._log(f"[INFO] Subsampled {n_target} from {n_available} individuals")
            return gen_data_sorted.loc[sampled_ids], snp_positions

        # Upsample by stacking (n_target > n_available)
        full_copies = n_target // n_available
        remainder = n_target % n_available

        chunks = [gen_data_sorted] * full_copies
        if remainder > 0:
            extra_idx = np.sort(self._rng.choice(n_available, size=remainder, replace=False))
            chunks.append(gen_data_sorted.iloc[extra_idx])

        stacked = pd.concat(chunks, ignore_index=False)

        new_fids = np.arange(1, len(stacked) + 1)
        stacked.index = pd.MultiIndex.from_arrays(
            [new_fids, new_fids.copy()], names=['fid', 'iid']
        )

        self._log(
            f"[INFO] Stacked genotypes: {n_available} -> {len(stacked)} "
            f"({full_copies} full copies + {remainder} extra samples)"
        )
        return stacked, snp_positions

    def _load_phenotype(self) -> pd.DataFrame:
        """Load phenotype data with caching."""
        
        # Load from source
        self._df = self._load_phenotype_source()
        
        return self._df
    
    def _load_phenotype_source(self) -> pd.DataFrame:
        """Load phenotype from source files."""
        if self.sim_config.enabled:
            return self._generate_simulated_phenotype()
        
        if self.dset in ["thaliana_horton", "thaliana_korte"]:
            df = _read_tabular_file(self.paths.pheno)
            df = self._set_fid_iid_index(df)
            return df
        
        return self._load_phenotype_custom()
    
    def _load_phenotype_custom(self) -> pd.DataFrame:
        """
        Load custom-format phenotype data.

        Supports .csv, .tsv, .txt (auto-detected delimiter) and
        .xlsx / .xls (first sheet).

        Expected layout:
            fid  iid  trait_1  trait_2  ...
        If the first two columns are not named fid/iid the function
        assumes the first two columns *are* fid and iid.
        """
        df = _read_tabular_file(self.paths.pheno)

        df = self._set_fid_iid_index(df)

        # Rename trait columns to generic names if they are still numeric
        trait_cols = [c for c in df.columns if c not in ('fid', 'iid')]
        rename_map = {}
        for i, c in enumerate(trait_cols):
            if isinstance(c, int) or str(c).isdigit():
                rename_map[c] = f'phenotype_{i}'
        if rename_map:
            df = df.rename(columns=rename_map)

        return df.fillna(df.mean())

    def _load_auxiliary_file(
        self,
        path: Optional[str],
        data_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Load a batch or covariate file with flexible format detection.

        Supports .csv, .tsv, .txt (auto-detected delimiter) and
        .xlsx / .xls (first sheet).
        """
        if not path or not os.path.exists(path):
            return None

        try:
            df = _read_tabular_file(path)
        except Exception as e:
            self._log(f"[WARNING] Failed to read {data_type} file {path}: {e}")
            return None

        df = self._set_fid_iid_index(df)

        # Rename non-id columns to generic names when they are numeric
        other_cols = [c for c in df.columns if c not in ('fid', 'iid')]
        rename_map = {}
        for i, c in enumerate(other_cols):
            if isinstance(c, int) or str(c).isdigit():
                rename_map[c] = f'var_{i}'
        if rename_map:
            df = df.rename(columns=rename_map)

        # Match with phenotype
        common = self.df.index.intersection(df.index)
        return df.loc[common].reindex(self.df.index)

    def _try_load_annotation(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Try loading an annotation file.

        Supports .csv, .tsv, .txt and .xlsx / .xls.
        """
        if not path or not os.path.exists(path):
            return None
        try:
            df = _read_tabular_file(path)
            return self._standardize_snp_columns(df)
        except Exception as e:
            self._log(f"[WARNING] Failed to load annotation from {path}: {e}")
            return None

    @staticmethod
    def _set_fid_iid_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure *df* has a (fid, iid) MultiIndex.

        Handles three cases:
          1. Columns named 'fid' and 'iid' already present.
          2. No recognised ID columns → assume the first two columns
             are fid and iid.
          3. The index is already a MultiIndex with the right names.
        """
        # Already indexed correctly
        if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ['fid', 'iid']:
            return df

        # Normalise column names to lowercase for matching
        lower_cols = {str(c).lower(): c for c in df.columns}

        if 'fid' in lower_cols and 'iid' in lower_cols:
            fid_col = lower_cols['fid']
            iid_col = lower_cols['iid']
        else:
            # Assume first two columns are fid, iid
            fid_col = df.columns[0]
            iid_col = df.columns[1]

        df = df.rename(columns={fid_col: 'fid', iid_col: 'iid'})
        df['fid'] = df['fid'].astype(int)
        df['iid'] = df['iid'].astype(int)
        df = df.set_index(['fid', 'iid'])
        return df

    def _generate_simulated_phenotype(self) -> pd.DataFrame:
        """Generate simulated phenotype data."""

        cfg = self.sim_config  
        import gc

        # Compute kinship once with safe chunk size
        num_samples = self.gen_data_standard.shape[0]
        dynamic_chunk = 100000 if num_samples <= 1000 else 1000

        G_tensor = torch.as_tensor(self.gen_data_standard.values, dtype=torch.float64)
        
        K, QS, G_stable = prepare_kinship_pipeline_torch(
            G=G_tensor, debug=False, chunk_size=dynamic_chunk
        )
        del G_tensor

        self._cached_QS = QS
        self._cached_G_stable = G_stable

        K_np = K.numpy()
        del K
        
        if not cfg.vardec:
            chrom, pos = self.extract_chrom_pos()

            self.simulated_data_class = PhenoSimulator(
                dset=self.dset,
                X=self.gen_data_standard,
                P=cfg.num_tasks,
                eta=cfg.eta,
                rep_idx=cfg.rep_idx,
                chrom=chrom,
                pos=pos,
                reference_trait=cfg.reference_trait,
                precomputed_kinship=K_np, 
            )

            self.Xr, _, self.global_indices = self.simulated_data_class.getRegion(size=None)

            df, info = self.simulated_data_class.genPheno(
                self.Xr,
                ncausal=cfg.ncausal,
                use_heterogeneity=cfg.use_heterogeneity,
                corr_bounds=cfg.corr_bounds,
                global_indices=self.global_indices
            )

        else:
            self.simulated_data_class = VarDecSimulator(
                X=self.gen_data_standard,
                P=cfg.num_tasks,
                rep_idx=cfg.rep_idx,
                precomputed_kinship=K_np, 
            )

            df, info = self.simulated_data_class.genPheno(
                scenario=cfg.vardec_scenario
            )

        self._info = info
        print(f"[INFO] Generated simulation_info with keys: {list(self._info.keys())}")

        del K_np
        gc.collect()

        if len(df) == len(self.gen_data_standard):
            df.index = self.gen_data_standard.index

        return df

    def _apply_transformations(self, train_idx: np.ndarray) -> None:
        method = self.correction_config.transformation
        if method in [None, 'none', '']:
            self._log("[INFO] No transformation applied")
            return
        if method not in ['z_score', 'int']:
            raise ValueError(
                f"Unknown transformation: '{method}'. Valid options: 'none', 'z_score', 'int'"
            )

        n_missing = self._df.isna().sum().sum()
        if n_missing > 0:
            per_col = self._df.isna().sum()
            bad = per_col[per_col > 0]
            details = ", ".join(f"{col}={n}" for col, n in bad.items())
            raise ValueError(
                f"Phenotype data contains {n_missing} missing value(s). "
                f"NaN imputation is not supported. "
                f"Per-column NaN counts: {details}. "
                f"Please drop rows with NaN values or impute them before loading."
            )

        self._log(f"[INFO] Applying transformation: {method} (fit on train rows only)")
        if method == 'z_score':
            train_rows = self._df.iloc[train_idx]
            mu  = train_rows.mean()
            sig = train_rows.std().replace(0, 1.0)
            self._df = (self._df - mu) / sig
        else:  # 'int'
            for name, idx in self.split_indices.items():
                if idx:
                    rows = self._df.iloc[idx]
                    self._df.iloc[idx] = standardize_data(rows, method='int').values

        self.correction_metadata['transformations']['applied'] = True
        if self.verbose:
            self._log_transformation_stats()
    
    def _log_transformation_stats(self) -> None:
        """Log statistics after transformation."""
        means = self._df.mean()
        stds = self._df.std()
        
        self._log(f"[INFO] Post-transformation statistics:")
        self._log(f"  Means: [{', '.join(f'{m:.4f}' for m in means)}]")
        self._log(f"  Stds:  [{', '.join(f'{s:.4f}' for s in stds)}]")

    def _compute_phenotype_stats(self) -> None:
        """Compute phenotype statistics."""
        if self.trait_variances is not None:
            return
        
        self.trait_variances = self._df.var().values
        self.trait_covariances = self._df.cov().values
        self.target_variances = np.diag(self.trait_covariances)
        self.target_covariances = self.trait_covariances.copy()
        self.trait_correlations = self._df.corr().values
    
    def _align_samples(self) -> None:
        """Align phenotype and genotype sample indices."""
        if isinstance(self.df.index, pd.MultiIndex):
            self._align_multiindex()
        else:
            self._align_simple_index()
    
    def _align_multiindex(self) -> None:
        """Align MultiIndex samples deterministically."""
        df_tuples = {(str(f), str(i)) for f, i in self.df.index}
        gen_tuples = {(str(f), str(i)) for f, i in self.gen_data_standard.index}
        
        common = df_tuples & gen_tuples
        if not common:
            raise ValueError("No common samples between phenotype and genotype!")
        
        common_sorted = sorted(
            [(int(f), int(i)) for f, i in common],
            key=lambda x: (x[0], x[1])
        )
        
        df_map = {(str(f), str(i)): (f, i) for f, i in self.df.index}
        gen_map = {(str(f), str(i)): (f, i) for f, i in self.gen_data_standard.index}
        
        common_str = [(str(f), str(i)) for f, i in common_sorted]
        df_idx = [df_map[t] for t in common_str]
        gen_idx = [gen_map[t] for t in common_str]
        
        self.df = self.df.loc[df_idx]
        self.gen_data_standard = self.gen_data_standard.loc[gen_idx]
        
        self._log(f"[INFO] Aligned {len(common)} samples")
    
    def _align_simple_index(self) -> None:
        """Align simple index samples."""
        common = self.df.index.intersection(self.gen_data_standard.index).sort_values()
        if len(common) == 0:
            raise ValueError("No common samples!")
        
        self.df = self.df.loc[common]
        self.gen_data_standard = self.gen_data_standard.loc[common]

    def _apply_corrections(self, train_idx: np.ndarray) -> None:
        will_correct = (
            (self.correction_config.regress_batch and self.paths.batch) or
            (self.correction_config.regress_covariates and self.paths.cov)
        )
        self.df_uncorrected = self.df.copy() if will_correct else None

        if self.correction_config.regress_batch and self.paths.batch:
            batch_df = self._load_auxiliary_file(self.paths.batch, "batch")
            if batch_df is not None:
                self.df, _ = regress_batch_effects(
                    self.df, batch_df,
                    per_trait=self.correction_config.per_trait_batch,
                    train_idx=train_idx,
                )
                self.correction_metadata['batch_correction']['applied'] = True

        if self.paths.cov:
            cov_df = self._load_auxiliary_file(self.paths.cov, "covariate")
            if cov_df is not None:
                if self.correction_config.regress_covariates:
                    self.df, _ = regress_continuous_covariates(
                        self.df, cov_df, train_idx=train_idx,
                    )
                    self.correction_metadata['covariate_correction']['applied'] = True
                else:
                    self.covariate_matrix = torch.as_tensor(cov_df.values, dtype=torch.float64)
                    self.covariate_names  = list(cov_df.columns)
        
    def _load_covariates_for_model(self) -> None:
        """Load covariates as fixed effects."""
        cov_df = self._load_auxiliary_file(self.paths.cov, "covariate")
        if cov_df is None:
            return
        
        self.covariate_matrix = torch.as_tensor(cov_df.values, dtype=torch.float64)
        self.covariate_names = list(cov_df.columns)
    
    def _setup_splits_and_qs(self) -> None:
        """Create data splits and compute QS decomposition."""
        self._create_splits()
        self._compute_qs()
        self._set_current_split()

    def _create_splits(self) -> None:
        n = len(self.df)

        if self.split_config.train_pct >= 1.0:
            self.split_indices = {"train": list(range(n)), "val": [], "test": []}
            self._log(f"[INFO] Using all {n} samples for training")
            return

        self._create_random_splits()

    def _create_random_splits(self) -> None:
        """Fallback to random splits."""
        n = len(self.df)
        indices = self._rng.permutation(n)

        train_end = int(self.split_config.train_pct * n)
        val_end = int((self.split_config.train_pct + self.split_config.val_pct) * n)

        self.split_indices = {
            "train": indices[:train_end].tolist(),
            "val": indices[train_end:val_end].tolist(),
            "test": indices[val_end:].tolist(),
        }

    def _compute_qs(self) -> None:
        """Compute QS decomposition on full population."""
        import gc
        
        if hasattr(self, '_cached_QS') and self._cached_QS is not None:
            if self.verbose:
                print("[INFO] Using precomputed QS decomposition from simulation step.")
            
            QS = self._cached_QS
            G_stable = self._cached_G_stable
            
            del self._cached_QS
            del self._cached_G_stable
            gc.collect()
            
        else:
            # Compute normally if running real data (no simulator)
            num_samples = self.gen_data_standard.shape[0]
            dynamic_chunk = 100000 if num_samples <= 1000 else 1000

            if self.verbose:
                print(f"[INFO] N={num_samples}, setting QS chunk_size to {dynamic_chunk}")

            G_tensor = torch.as_tensor(
                self.gen_data_standard.values,
                dtype=torch.float64
            )

            _, QS, G_stable = prepare_kinship_pipeline_torch(
                G=G_tensor, 
                debug=False,
                chunk_size=dynamic_chunk
            )
            del G_tensor
            gc.collect()

        (Q0, Q1), S0 = QS
        del Q0, Q1, S0, QS
        gc.collect()
        
        self._G_norm = max(G_stable.min().item(), G_stable.max().item())
        self.G_stable = G_stable / self._G_norm
        
        del G_stable
        gc.collect()

        self._G_scaling = {
            'G_norm': self._G_norm,
            'scale_to_original_C0': 1.0 / (self._G_norm ** 2),
        }
        
    def _set_current_split(self) -> None:
        import gc
        import numpy as np

        if not hasattr(self, 'gen_data_tensor_full') or self.gen_data_tensor_full is None:
            np_arr = self.gen_data_standard.to_numpy(dtype=np.float64, copy=False)
            self.gen_data_tensor_full = torch.from_numpy(np_arr)
            self.gen_data_standard = None  # free the DataFrame
            del np_arr

        if not hasattr(self, 'df_tensor_full') or self.df_tensor_full is None:
            np_df = self.df.to_numpy(dtype=np.float64, copy=False)
            self.df_tensor_full = torch.from_numpy(np_df)
            self._df = None
            del np_df

        gc.collect()

        indices = self.split_indices[self.split]

        if len(indices) == self.gen_data_tensor_full.shape[0]:
            self.gen_data_tensor = self.gen_data_tensor_full
            self.data_tensor = self.df_tensor_full
        else:
            idx = torch.as_tensor(indices, dtype=torch.long)
            self.gen_data_tensor = self.gen_data_tensor_full[idx]
            self.data_tensor = self.df_tensor_full[idx]
            del idx
            gc.collect()

        self.current_split_indices = indices

    def _load_snp_positions(self, bim: Optional[Any]) -> Optional[pd.DataFrame]:
        """
        Load SNP position annotations, checking sources in priority order:
        1. Explicit annotation file (self.paths.annot)
        2. Cached annotation at {output_dir}/snp_annotation.csv
        3. Constructed from BIM data
        """
        df = self._try_load_annotation(self.paths.annot)
        if df is not None:
            self._log(f"[INFO] Loaded SNP positions from annotation file: {self.paths.annot}")
            return df

        cached_path = self._annotation_cache_path()
        df = self._try_load_annotation(cached_path)
        if df is not None:
            self._log(f"[INFO] Loaded SNP positions from {cached_path}")
            return df

        df = self._snp_positions_from_bim(bim)
        if df is not None:
            self._log(f"[INFO] Constructed SNP positions from BIM data ({len(df)} variants)")
            self._cache_annotation(df)
            return df

        self._log("[WARNING] No SNP position data available")
        return None

    def _annotation_cache_path(self) -> Optional[str]:
        """Return the cached annotation path, or None if root is unset."""
        if self.paths.output_dir is None:
            return None
        return os.path.join(self.paths.output_dir, f"snp_annotation.csv")

    def _snp_positions_from_bim(self, bim: Optional[Any]) -> Optional[pd.DataFrame]:
        """Extract chrom/pos DataFrame from a BIM object."""
        if bim is None or not isinstance(bim, pd.DataFrame):
            return None

        if 'chrom' in bim.columns and 'pos' in bim.columns:
            return bim[['chrom', 'pos']].copy()

        if len(bim.columns) >= 4:
            return pd.DataFrame(
                {'chrom': bim.iloc[:, 0], 'pos': bim.iloc[:, 3]},
                index=bim.iloc[:, 1] if len(bim.columns) > 1 else bim.index,
            )

        return None

    def _cache_annotation(self, df: pd.DataFrame) -> None:
        """Persist annotation DataFrame to root if available."""
        cache_path = self._annotation_cache_path()
        if cache_path is None:
            return
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            self._log(f"[INFO] Cached annotation to {cache_path}")
        except Exception as e:
            self._log(f"[WARNING] Failed to cache annotation: {e}")
    
    def _standardize_snp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure SNP DataFrame has 'chrom' and 'pos' columns."""
        if df is None:
            return None
        
        if 'chrom' in df.columns and 'pos' in df.columns:
            return df
        
        if df.shape[1] >= 2:
            df.columns = ['chrom', 'pos'] + [f'col_{i}' for i in range(2, df.shape[1])]
        elif df.shape[1] == 1:
            df.columns = ['pos']
            df['chrom'] = 1
        
        return df
    
    def extract_chrom_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract chromosome and position arrays."""
        if self.snp_positions is None:
            return np.array([]), np.array([])
        
        df = self.snp_positions
        
        if 'chrom' in df.columns and 'pos' in df.columns:
            return df['chrom'].to_numpy(), df['pos'].to_numpy()
        
        if isinstance(df.index, pd.MultiIndex):
            return df.index.get_level_values(0).to_numpy(), df.index.get_level_values(1).to_numpy()
        
        if df.shape[1] >= 2:
            return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
        
        return df.index.to_numpy(), df.iloc[:, 0].to_numpy()
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        self._df = value
    
    @property
    def simulation_info(self) -> Optional[Dict]:
        return self._info
    
    @property
    def G_norm(self) -> Optional[float]:
        return self._G_norm
    
    @G_norm.setter
    def G_norm(self, value: float) -> None:
        self._G_norm = value
    
    @property
    def G_scaling(self) -> Dict:
        return self._G_scaling or {'scale_to_original_C0': 1.0}
    
    @G_scaling.setter
    def G_scaling(self, value: Dict) -> None:
        self._G_scaling = value
    
    def __len__(self) -> int:
        if hasattr(self, 'data_tensor'):
            return len(self.data_tensor)
        return len(self.current_split_indices)
    
    def __getitem__(self, idx: int) -> Tuple:
        abs_idx = self.current_split_indices[idx]
        
        pheno = self.data_tensor[idx]
        geno = self.gen_data_tensor[idx]
        geno_qs = self.G_stable[abs_idx]
        
        if self.covariate_matrix is not None:
            cov = self.covariate_matrix[abs_idx]
            return (geno, geno_qs, pheno, cov, abs_idx)
        
        return (geno, geno_qs, pheno, abs_idx)

@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    batch_size: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = False


def get_data_multitask_snp(
    dset: str,
    *,
    data_path_config: Optional[str] = None,
    pheno_path: Optional[str] = None,
    geno_path: Optional[str] = None,
    annot_path: Optional[str] = None,
    batch_path: Optional[str] = None,
    cov_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    predict_geno_path: Optional[str] = None,
    predict_cov_path:  Optional[str] = None,
    loader_config: Optional[DataLoaderConfig] = None,
    verbose: bool = True,
    **config_params
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """
    Create DataLoaders for multitask SNP data.
    
    Loads data once, then creates split-specific views.
    """
    if loader_config is None:
        loader_config = DataLoaderConfig(
            batch_size=config_params.pop('batch_size', None),
            num_workers=config_params.pop('num_workers', 0),
            pin_memory=config_params.pop('pin_memory', False),
        )
    
    train_pct = config_params.get('train_pct', 0.8)
    val_pct = config_params.get('val_pct', 0.2)

    if predict_geno_path is not None:
        if verbose:
            print("[INFO] External prediction genotypes provided — "
                  "forcing train_pct=1.0, val_pct=0.0")
        config_params['train_pct'] = 1.0
        config_params['val_pct']   = 0.0
        train_pct = 1.0
        val_pct   = 0.0

    if train_pct + val_pct > 1.0:
        raise ValueError("train_pct + val_pct must not exceed 1.0")
    
    if verbose:
        _log_config(dset, data_path_config, pheno_path, geno_path, 
                    annot_path, batch_path, cov_path, config_params)
    
    dataset_kwargs = dict(
        dset=dset,
        pheno_path=pheno_path,
        geno_path=geno_path,
        annot_path=annot_path,
        batch_path=batch_path,
        cov_path=cov_path,
        output_dir=output_dir,
        data_path_config=data_path_config,
        verbose=verbose,
        **config_params
    )
    
    if verbose:
        print("[INFO] Loading data...")
    
    master_dataset = MultitaskDatasetSNP(split="train", **dataset_kwargs)

    loaders = {}
    
    train_dataset = SplitView(master_dataset, "train")
    loaders['train'] = _make_loader(train_dataset, loader_config)
    
    if val_pct > 0 and len(master_dataset.split_indices.get('val', [])) > 0:
        val_dataset = SplitView(master_dataset, "val")
        loaders['val'] = _make_loader(val_dataset, loader_config)
        if verbose:
            print(f"[INFO] Validation split: {len(val_dataset)} samples")
    
    if train_pct + val_pct < 1.0 and len(master_dataset.split_indices.get('test', [])) > 0:
        test_dataset = SplitView(master_dataset, "test")
        loaders['test'] = _make_loader(test_dataset, loader_config)
        if verbose:
            print(f"[INFO] Test split: {len(test_dataset)} samples")

    if predict_geno_path is not None:
        pred_view = PredictionSplitView(master_dataset, predict_geno_path, predict_cov_path)
        loaders['test'] = _FullBatchLoader(pred_view)
        if verbose:
            print(f"[INFO] Prediction split (external genotypes): "
                  f"{len(pred_view)} new samples, "
                  f"{pred_view.gen_data_tensor.shape[1]} SNPs")
                
    data_meta = _build_metadata(master_dataset, loaders['train'], config_params, 
                                 pheno_path, geno_path, annot_path, batch_path, 
                                 cov_path, data_path_config, dset, verbose)

    if predict_geno_path is not None:
        data_meta['prediction_sample_index'] = pred_view.sample_index
        data_meta['predict_geno_path'] = predict_geno_path
        data_meta['predict_cov_path']        = predict_cov_path

    if verbose:
        print(f"\n[INFO] Created {len(loaders)} dataloader(s): {list(loaders.keys())}")
        for name, loader in loaders.items():
            print(f"  {name}: {len(loader.dataset)} samples")
    
    return loaders, data_meta


class SplitView:
    def __init__(self, master: MultitaskDatasetSNP, split: str):
        self.master = master
        self.split = split
        self.indices = master.split_indices[split]
        self._has_covariates = master.covariate_matrix is not None

        n_split = len(self.indices)
        n_total = master.df_tensor_full.shape[0]

        if n_split == 0:
            self.data_tensor = torch.empty(
                (0, master.df_tensor_full.shape[1]), dtype=torch.float64
            )
            self.gen_data_tensor = torch.empty(
                (0, master.gen_data_tensor_full.shape[1]), dtype=torch.float64
            )
            self.geno_qs_tensor = torch.empty(
                (0, master.G_stable.shape[1]), dtype=torch.float64
            )
            self.abs_indices = torch.empty(0, dtype=torch.long)
            if self._has_covariates:
                self.cov_tensor = torch.empty(
                    (0, master.covariate_matrix.shape[1]), dtype=torch.float64
                )
        elif n_split == n_total:
            self.data_tensor = master.df_tensor_full
            self.gen_data_tensor = master.gen_data_tensor_full
            self.geno_qs_tensor = master.G_stable
            self.abs_indices = torch.arange(n_total, dtype=torch.long)
            if self._has_covariates:
                self.cov_tensor = master.covariate_matrix
        else:
            idx = torch.as_tensor(self.indices, dtype=torch.long)
            self.data_tensor = master.df_tensor_full[idx]
            self.gen_data_tensor = master.gen_data_tensor_full[idx]
            self.geno_qs_tensor = master.G_stable[idx]
            self.abs_indices = idx
            if self._has_covariates:
                self.cov_tensor = master.covariate_matrix[idx]

    def get_full_batch(self):
        """Return the entire split as a single pre-stacked tuple. O(1)."""
        if self._has_covariates:
            return (
                self.gen_data_tensor,
                self.geno_qs_tensor,
                self.data_tensor,
                self.cov_tensor,
                self.abs_indices,
            )
        return (
            self.gen_data_tensor,
            self.geno_qs_tensor,
            self.data_tensor,
            self.abs_indices,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Per-sample access — only used when mini-batching."""
        pheno = self.data_tensor[idx]
        geno = self.gen_data_tensor[idx]
        geno_qs = self.geno_qs_tensor[idx]

        if self._has_covariates:
            cov = self.cov_tensor[idx]
            return (geno, geno_qs, pheno, cov, self.abs_indices[idx])

        return (geno, geno_qs, pheno, self.abs_indices[idx])

    def __getattr__(self, name):
        return getattr(self.master, name)
 
class _FullBatchLoader:
    """Drop-in replacement for DataLoader when using full-batch mode.

    Iterating yields exactly one item: the pre-stacked full batch.
    Exposes .dataset so downstream metadata code still works.
    """

    def __init__(self, dataset: SplitView):
        self.dataset = dataset
        self._batch = dataset.get_full_batch()

    def __iter__(self):
        yield self._batch

    def __len__(self):
        return 1
    
def _make_loader(dataset: SplitView, config: DataLoaderConfig) -> DataLoader:
    """Create a DataLoader, or a zero-overhead full-batch wrapper."""
    batch_size = len(dataset) if config.batch_size in (None, 0) else config.batch_size

    if batch_size >= len(dataset):
        return _FullBatchLoader(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

def _build_metadata(
    train_dataset,
    train_loader: DataLoader,
    config_params: Dict,
    pheno_path, geno_path, annot_path, batch_path, cov_path,
    data_path_config, dset, verbose,
) -> Dict[str, Any]:
    """
    Extract metadata without keeping full DataFrames alive.
    """
    try:
        sample_batch = next(iter(train_loader))
        geno_full = sample_batch[0]
        pheno = sample_batch[2]
 
        total_snps = geno_full.shape[1] if geno_full.ndim > 1 else 1
        num_tasks = pheno.shape[1] if pheno.ndim > 1 else 1
 
        if verbose:
            print(f"\n[INFO] Inferred dimensions: SNPs={total_snps}, Tasks={num_tasks}")
    except StopIteration:
        raise ValueError("Train loader is empty")
 
    correction_meta = getattr(train_dataset, 'correction_metadata', {
        'batch_correction':  {'applied': False},
        'covariate_correction': {'applied': False},
        'transformations':   {'applied': False},
    })
 
    corrections_applied = (
        correction_meta.get('batch_correction', {}).get('applied', False)
        or correction_meta.get('covariate_correction', {}).get('applied', False)
    )
 
    return {
        'total_snps': total_snps,
        'total_samples': train_dataset.total_samples,
        'num_tasks': num_tasks,
        'snp_info_df': train_dataset.snp_positions,
        'trait_variances': train_dataset.trait_variances,
        'trait_covariances': train_dataset.trait_covariances,
        'G_norm': train_dataset.G_norm,
        'G_scaling': train_dataset.G_scaling,
        'simulation_info': train_dataset.simulation_info,
        'rep_idx': config_params.get('rep_idx'),
        'eta': config_params.get('eta'),
        'use_heterogeneity': config_params.get('use_heterogeneity', False),
        'corr_bounds': config_params.get('corr_bounds'),
        'vardec_scenario': config_params.get('vardec_scenario'),
        'correction_metadata': correction_meta,
        'phenotype_data': {
            'corrections_applied': corrections_applied,
            'n_samples': len(train_dataset), 
            'n_traits': num_tasks,
        },
        'dataset_info': {
            'dset': dset,
            'pheno_path': pheno_path,
            'geno_path': geno_path,
            'annot_path': annot_path,
            'batch_path': batch_path,
            'cov_path': cov_path,
            'data_path_config': data_path_config,
        },
    }

def _log_config(dset, data_path_config, pheno_path, geno_path, 
                annot_path, batch_path, cov_path, config_params):
    """Print configuration summary."""
    print("\n" + "="*70)
    print("DATA LOADER CONFIGURATION")
    print("="*70)
    print(f"Dataset: {dset}")
    print(f"Config: {data_path_config or 'direct paths'}")
    
    paths = [('Phenotype', pheno_path), ('Genotype', geno_path),
             ('Annotation', annot_path), ('Batch', batch_path), ('Covariate', cov_path)]
    
    if any(p for _, p in paths):
        print("\nFile paths:")
        for name, path in paths:
            if path:
                print(f"  {name}: {path}")
    
    if config_params.get('simulated'):
        print("\nSimulation:")
        for key in ['rep_idx', 'eta', 'use_heterogeneity', 'num_samples', 'num_tasks']:
            if key in config_params:
                print(f"  {key}: {config_params[key]}")
    
    print("="*70 + "\n")



def load_prediction_genotypes(
    path: str,
) -> Tuple[np.ndarray, pd.MultiIndex]:
    """Load prediction genotypes from any supported format.

    Parameters
    ----------
    path : str
        File path.  The extension determines the reader:
          .npz         → NumPy archive (keys: 'genotypes', 'fid', 'iid')
          .bed         → PLINK binary  (reads .bed/.bim/.fam triplet)
          .h5 / .hdf5  → HDF5          (datasets: 'genotypes', 'fid', 'iid')
          .csv/.tsv/.txt → delimited text (fid, iid + SNP columns)

    Returns
    -------
    genotypes : np.ndarray, shape (n_samples, n_snps), float64
        Raw (un-normalised) additive dosage matrix.
    sample_index : pd.MultiIndex
        (fid, iid) index for output alignment.
    """
    ext = Path(path).suffix.lower()

    if ext in ('.csv', '.tsv', '.txt'):
        warnings.warn(
            f"Loading prediction genotypes from '{ext}' is slow for wide "
            f"matrices. Consider converting to .npz or PLINK .bed format "
            f"for 10-100x faster loading.",
            stacklevel=2,
        )

    geno, index, _ = _load_genotype_file(path, allow_npz=True)

    # Ensure float64, fill NaN with column means
    geno = geno.astype(np.float64)
    nan_mask = np.isnan(geno)
    if nan_mask.any():
        col_means = np.nanmean(geno, axis=0)
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        for c in nan_cols:
            geno[nan_mask[:, c], c] = col_means[c]

    return geno, index

class PredictionSplitView:
    """Read-only view over external genotypes for phenotype prediction.
 
    Produces the same tuple layout as SplitView:
        (geno, geno_qs, pheno_placeholder, abs_index)
    """
    def __init__(
        self,
        master: MultitaskDatasetSNP,
        predict_geno_path: str,
        predict_cov_path: Optional[str] = None,
    ):
        self.master = master

        geno_values, self.sample_index = load_prediction_genotypes(predict_geno_path)
        geno_df = pd.DataFrame(geno_values, columns=range(geno_values.shape[1]))
        self.gen_data_tensor = master.normalize_new_genotypes(geno_df)
        self.geno_qs_tensor = (
            self.gen_data_tensor / master._G_norm
            if master._G_norm not in (None, 0)
            else self.gen_data_tensor.clone()
        )

        n_new   = self.gen_data_tensor.shape[0]
        n_tasks = master.df_tensor_full.shape[1]
        self.data_tensor = torch.full((n_new, n_tasks), float('nan'), dtype=torch.float64)
        self.abs_indices = torch.arange(n_new, dtype=torch.long)

        # Optional external covariates — must match training column order/count
        self._has_covariates = False
        self.cov_tensor = None
        if predict_cov_path is not None:
            if master.covariate_matrix is None:
                raise ValueError(
                    "predict_cov_path provided but training had no covariates; "
                    "ignore predict_cov_path or retrain with covariates."
                )
            cov_df = _read_tabular_file(predict_cov_path)
            cov_df = master._set_fid_iid_index(cov_df)
            cov_df = cov_df.loc[cov_df.index.intersection(self.sample_index).union(cov_df.index)]
            # Align to prediction sample order
            cov_df = cov_df.reindex(self.sample_index)
            if cov_df.isna().any().any():
                raise ValueError("Some prediction samples are missing covariate values.")
            n_train_cov = master.covariate_matrix.shape[1]
            if cov_df.shape[1] != n_train_cov:
                raise ValueError(
                    f"Covariate column mismatch: training has {n_train_cov}, "
                    f"prediction file has {cov_df.shape[1]}."
                )
            self.cov_tensor = torch.as_tensor(cov_df.values, dtype=torch.float64)
            self._has_covariates = True

        self.indices = list(range(n_new))
        self.split = "predict"
 
    def get_full_batch(self):
        return (
            self.gen_data_tensor,
            self.geno_qs_tensor,
            self.data_tensor,
            self.abs_indices,
        )
 
    def __len__(self):
        return self.gen_data_tensor.shape[0]
 
    def __getitem__(self, idx):
        return (
            self.gen_data_tensor[idx],
            self.geno_qs_tensor[idx],
            self.data_tensor[idx],
            self.abs_indices[idx],
        )
 
    def __getattr__(self, name):
        return getattr(self.master, name)