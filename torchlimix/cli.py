import yaml
import argparse
import traceback
import copy
import torch
from typing import Dict, Any
from torchlimix.run_backend import run_main
import logging
from pathlib import Path
from importlib import resources
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_default_config_path():
    return resources.files("torchlimix.configs") / "hyperparameters.yaml"

def get_package_root() -> Path:
    """Get the root directory of the torchLIMIX package."""
    # This gets the parent of the torchlimix package directory
    return Path(resources.files("torchlimix")).parent

def resolve_path_placeholders(path: str) -> str:
    """Replace ${PACKAGE_ROOT} placeholder with actual package root path."""
    if path is None:
        return None
    
    package_root = str(get_package_root())
    resolved = path.replace("${PACKAGE_ROOT}", package_root)
    resolved = os.path.expanduser(resolved)
    return resolved

def parse_args():
    """Parse command line arguments for likelihood analysis."""
    parser = argparse.ArgumentParser(description="Multivariate GWAS with TorchLIMIX")
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to the config file"
    )

    # Data overrides (these go to data_param)
    parser.add_argument("--dset", type=str, help="Dataset name. Your results will be stored under output_directory/dset/anylsis_type")
    
    parser.add_argument("--transformation_method", type=str, choices=["int", "z_score", "none", None],
                        help="Transformation method: 'int' (inverse normal transform), 'z_score', 'none', or None if the data is already standardized upon preprocessing.")
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS) #help="Random seed for data operations (overrides data_param.seed)")
    parser.add_argument("--num_workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, help="Default is 1 (GPU/CUDA). Set to 0 to use the CPU instead. If CUDA is not available, the program will automatically fall back to the CPU.")

    # Data loading parameters
    parser.add_argument("--train_pct", type=float, 
                        help="Percentage of data for training (0.0-1.0, overrides data_param.train_pct)")
    parser.add_argument("--val_pct", type=float, 
                        help="Percentage of data for validation (0.0-1.0, overrides data_param.val_pct)")
    parser.add_argument("--batch_size", type=int, help=argparse.SUPPRESS) #"Batch size for data loading. Use 0 for no batching (overrides data_param.batch_size)")
    parser.add_argument("--data_path_config", type=str, help=argparse.SUPPRESS) #"Path to data configuration JSON file (overrides data_param.data_path_config)")

    # Simulation parameters (these go to data_param)
    parser.add_argument("--simulated", action="store_true", default=None, help=argparse.SUPPRESS) #"Use simulated data (overrides data_param.simulated)")
    parser.add_argument("--rep_idx", type=int, help=argparse.SUPPRESS) #"Single replication index (overrides data_param.rep_idx)")
    parser.add_argument("--n_reps", type=int, default=None, help=argparse.SUPPRESS) #"Number of replications to run in batch mode")
    parser.add_argument("--start_rep_idx", type=int, default=0, help=argparse.SUPPRESS) #"Starting replication index for batch mode (default: 0)")
    parser.add_argument("--eta", type=float, help=argparse.SUPPRESS) #"Proportionality factor for effect sizes across traits (overrides data_param.eta)"
    parser.add_argument("--corr_bounds", type=int, help=argparse.SUPPRESS) #"Correlation bounds index (0-9) for heterogeneity model (overrides data_param.corr_bounds)"
    parser.add_argument("--use_heterogeneity", action="store_true", default=None, help=argparse.SUPPRESS) #"Use heterogeneity model for simulation (overrides data_param.use_heterogeneity)")
    parser.add_argument("--vardec_scenario", type=int, help=argparse.SUPPRESS) #"Variance decomposition scenario ID (overrides data_param.vardec_scenario)")
    parser.add_argument("--num_samples", type=int, help=argparse.SUPPRESS) #"Number of samples for simulation (overrides data_param.num_samples)")
    parser.add_argument("--num_tasks", type=int, help=argparse.SUPPRESS) #"Number of traits/tasks (overrides data_param.num_tasks)")
    parser.add_argument("--ncausal", type=int, help=argparse.SUPPRESS) #"Number of causal SNPs per trait (overrides data_param.ncausal)")
    parser.add_argument("--reference_trait", type=int, help=argparse.SUPPRESS) #"Reference trait index for rescaling effect (overrides data_param.reference_trait)")

    # Data path arguments 
    parser.add_argument(
        "--pheno_path", type=str, default=None,
        help=(
            "Path to phenotype file. Supported formats: .csv, .tsv, .txt "
            "(auto-detected delimiter). Expected columns: fid, iid, followed "
            "by one or more trait columns."
        ),
    )
    parser.add_argument(
        "--geno_path", type=str, default=None,
        help=(
            "Path to genotype file. Supported formats: PLINK binary "
            "(.bed/.bim/.fam, provide path without extension), "
            "HDF5 (.h5, .hdf5), or delimited text (.csv, .tsv, .txt)."
        ),
    )
    parser.add_argument("--annot_path", type=str, default=None, 
                        help="Path to SNP annotation file")
    parser.add_argument("--cov_path", type=str, default=None, 
                        help="Path to covariates file")
    parser.add_argument("--batch_path", type=str, default=None, 
                        help="Path to batch file (tab-delimited, first two columns: fid, iid)")
    parser.add_argument(
        "--predict_geno_path", type=str, default=None,
        help=(
            "Path to external genotype file for prediction. "
            "When provided, the model trains on 100%% of the original "
            "dataset and predicts phenotypes for these new samples. "
            "Supported formats: PLINK (.bed), HDF5 (.h5, .hdf5), "
            "delimited text (.csv, .tsv, .txt), or NumPy archive (.npz). "
            "SNP count must match the training genotypes."
        ),
    )  
    # Covariate handling: mutually exclusive group
    cov_group = parser.add_mutually_exclusive_group()
    cov_group.add_argument("--regress_out_covariates", action="store_true", default=None,
                          help="Regress out covariates before analysis (default behavior if cov_path provided)")
    cov_group.add_argument("--include_covariates_in_model", action="store_true", default=None,
                          help="Include covariates in the model M matrix instead of regressing them out")
    
    # Batch and covariate correction options
    parser.add_argument("--regress_out_batch_effects", action="store_true", default=None,
                        help="Apply batch correction to phenotypes (overrides data_param.regress_out_batch_effects)")
    parser.add_argument("--save_correction_stats", action="store_true", default=None,
                        help="Save correction statistics to file (overrides data_param.save_correction_stats)")
        
    # Top-level analysis parameters
    parser.add_argument("--test_type", choices=["common", "any", "specific", "any_vs_common", "specific_vs_common"], type=str, 
                        help="Type of test to run (overrides config)")
    parser.add_argument("--pheno_idx", type=int, 
                        help="Only relevant for specific effect test (here you test if the effect is specific to a certain phenotype/environment 0<=idx<=(number of phenotypes-1))")
    parser.add_argument("--analysis", type=str, choices=["gwas", "vardec", "prediction"], 
                        help="Type of analysis. Choose between gwas, and vardec (Variance Decomposition) and prediction.")
    parser.add_argument("--rank", type=int, default=None, 
                        help="Rank parameter for model. Choose rank equal to the number of phenotypes (default).")
    
    # Output
    parser.add_argument("--output_directory", type=str, 
                        help="Output directory for results (overrides config)")

    parser.add_argument("--verbose", type=lambda x: x.lower() in ('true', '1', 'yes'), 
                    default=None, metavar="BOOL",
                    help="Set verbose output (true/false). Default: true")
        
    args = parser.parse_args()
    
    # Resolve placeholders in the config path
    args.config = resolve_path_placeholders(args.config)
    
    return args

def validate_args(args, config: Dict[str, Any]) -> None:
    """Validate command line arguments and provide helpful warnings."""
    
    # Ensure data_param exists in config
    if "data_param" not in config:
        config["data_param"] = {}
    
    data_param = config["data_param"]  # Now this is a reference to the actual dict in config
    simulated = data_param.get("simulated", False)
    
    # Auto-disable simulation if real data paths are provided
    if args.pheno_path or args.geno_path:
        if simulated:
            logger.warning("="*60)
            logger.warning("AUTO-CONFIGURATION: Real data paths provided")
            logger.warning("  Setting simulated=False automatically")
            logger.warning("="*60)
            data_param["simulated"] = False  # Now this modifies config["data_param"]["simulated"]
            simulated = False
            logger.warning(f"  Verified: config['data_param']['simulated'] = {config['data_param']['simulated']}")
    
    # Validation 1: Check required paths for non-simulated data
    if not simulated:
        missing_paths = []
        if not args.pheno_path:
            missing_paths.append("--pheno_path")
        if not args.geno_path:
            missing_paths.append("--geno_path")
        
        if missing_paths:
            logger.error("="*60)
            logger.error("CONFIGURATION ERROR: Missing required data paths")
            logger.error("="*60)
            logger.error(f"When using real data (simulated=False), you must provide:")
            for path in missing_paths:
                logger.error(f"  {path}")
            logger.error("")
            logger.error("Example usage:")
            logger.error("  python script.py --pheno_path /path/to/phenotypes.txt \\")
            logger.error("                   --geno_path /path/to/genotypes.bed")
            logger.error("="*60)
            raise ValueError(f"Missing required arguments: {', '.join(missing_paths)}")
    
    # Validation 2: Covariate handling - mutually exclusive options
    if args.cov_path:
        regress_out = data_param.get("regress_out_covariates", False)
        include_in_model = data_param.get("include_covariates_in_model", False)
        
        # Check if both are True (mutually exclusive)
        if regress_out and include_in_model:
            logger.error("="*60)
            logger.error("CONFIGURATION ERROR: Mutually exclusive covariate options")
            logger.error("="*60)
            logger.error("Both regress_out_covariates and include_covariates_in_model are set to True.")
            logger.error("These options are mutually exclusive. Choose ONE:")
            logger.error("")
            logger.error("Option 1 - Regress out covariates (default):")
            logger.error("  --regress_out_covariates")
            logger.error("  (covariates are removed from phenotypes before analysis)")
            logger.error("")
            logger.error("Option 2 - Include covariates in model:")
            logger.error("  --include_covariates_in_model")
            logger.error("  (covariates are included in the fixed effects matrix M)")
            logger.error("="*60)
            raise ValueError(
                "Cannot set both regress_out_covariates=True and include_covariates_in_model=True. "
                "These are mutually exclusive options."
            )
        
        # Check if both are False (covariates provided but not used)
        if not regress_out and not include_in_model:
            logger.error("="*60)
            logger.error("CONFIGURATION ERROR: Covariates provided but not used")
            logger.error("="*60)
            logger.error("Covariate file provided but both handling options are False:")
            logger.error(f"  Covariate file: {args.cov_path}")
            logger.error(f"  regress_out_covariates: {regress_out}")
            logger.error(f"  include_covariates_in_model: {include_in_model}")
            logger.error("")
            logger.error("You must choose how to handle covariates. Choose ONE:")
            logger.error("")
            logger.error("Option 1 - Regress out covariates (recommended):")
            logger.error("  Set regress_out_covariates: True in config")
            logger.error("  Or use: --regress_out_covariates")
            logger.error("")
            logger.error("Option 2 - Include covariates in model:")
            logger.error("  Set include_covariates_in_model: True in config")
            logger.error("  Or use: --include_covariates_in_model")
            logger.error("="*60)
            raise ValueError(
                "Covariate file provided but neither regress_out_covariates nor "
                "include_covariates_in_model is set to True. "
                "You must specify how to handle covariates."
            )
        
        # Log the chosen covariate handling method
        logger.info("="*60)
        logger.info("COVARIATE HANDLING:")
        logger.info(f"  Covariate file provided: {args.cov_path}")
        
        if include_in_model:
            logger.info(f"  Method: Including covariates in model M matrix")
            logger.info(f"  regress_out_covariates: False")
            logger.info(f"  include_covariates_in_model: True")
        else:
            logger.info(f"  Method: Regressing out covariates before analysis")
            logger.info(f"  regress_out_covariates: True")
            logger.info(f"  include_covariates_in_model: False")
        
        logger.info("="*60)
    
    # Warning 3: Batch correction
    if args.batch_path:
        logger.warning("="*60)
        logger.warning("BATCH CORRECTION ACTIVATED:")
        logger.warning(f"  Batch file provided: {args.batch_path}")
        logger.warning(f"  regress_out_batch_effects will be set to True automatically")
        logger.warning("  Batch effects will be regressed out from phenotypes")
        logger.warning("="*60)
    
    # Warning 4: Conflicting correction settings
    if args.batch_path and not data_param.get("regress_out_batch_effects", False):
        logger.warning("")
        logger.warning("Note: batch_path provided but regress_out_batch_effects is False in config.")
        logger.warning("Setting regress_out_batch_effects to True automatically.")
        logger.warning("")

    # Validation: predict_geno_path requires prediction analysis
    if args.predict_geno_path:
        analysis = config.get("analysis", "gwas")
        if analysis != "prediction":
            logger.warning("="*60)
            logger.warning("AUTO-CONFIGURATION: --predict_geno_path provided")
            logger.warning("  Setting analysis='prediction' automatically")
            logger.warning("="*60)
            config["analysis"] = "prediction"
 
        if not args.pheno_path or not args.geno_path:
            raise ValueError(
                "--predict_geno_path requires --pheno_path and --geno_path "
                "(training data to fit the model on)"
            )
        
def load_config(config_path: str, args) -> Dict[str, Any]:
    """Load config from file and apply command line overrides."""
    
    # Load YAML config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    logger.info(f"Loaded config from: {config_path}")

    if "data_param" in config and "transformation_method" in config["data_param"]:
        tm = config["data_param"]["transformation_method"]
        if isinstance(tm, str) and tm.lower() == "none":
            config["data_param"]["transformation_method"] = None

    # Top-level overrides (parameters that belong at the root level)
    top_level_overrides = {
        "test_type": args.test_type,
        "analysis": args.analysis,
        "rank": args.rank,
        "output_directory": args.output_directory,
        "pheno_idx": args.pheno_idx,
        "device": args.device,
        "verbose": args.verbose
    }

    for key, value in top_level_overrides.items():
        if value is not None:
            config[key] = value
            logger.info(f"Override: {key} = {value}")

    # Nested data_param overrides
    data_param = config.get("data_param", {})

    # Data parameter overrides (parameters that belong in data_param)
    data_param_overrides = {
        "dset": args.dset,
        "transformation_method": args.transformation_method,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "train_pct": args.train_pct,
        "val_pct": args.val_pct,
        "batch_size": args.batch_size,
        "data_path_config": args.data_path_config,
        "rep_idx": args.rep_idx,
        "eta": args.eta,
        "corr_bounds": args.corr_bounds,
        "vardec_scenario": args.vardec_scenario,
        "num_samples": args.num_samples,
        "num_tasks": args.num_tasks,
        "ncausal": args.ncausal,
        "reference_trait": args.reference_trait,
    }

    # After setting data_param values, add:
    config["eta"] = data_param.get("eta")
    config["corr_bounds"] = data_param.get("corr_bounds")
    config["use_heterogeneity"] = data_param.get("use_heterogeneity", False)
    config["vardec_scenario"] = data_param.get("vardec_scenario")
    config["rep_idx"] = data_param.get("rep_idx")

    for key, value in data_param_overrides.items():
        if value is not None:
            data_param[key] = value
            logger.info(f"Override: data_param.{key} = {value}")
    
    if args.simulated is not None:
        data_param["simulated"] = True
        logger.info(f"Override: data_param.simulated = True")
    
    if args.use_heterogeneity is not None:
        data_param["use_heterogeneity"] = True
        logger.info(f"Override: data_param.use_heterogeneity = True")
    
    # Handle covariate method (mutually exclusive)
    if args.regress_out_covariates is not None:
        data_param["regress_out_covariates"] = True
        data_param["include_covariates_in_model"] = False
        logger.info(f"Override: data_param.regress_out_covariates = True")
        logger.info(f"Override: data_param.include_covariates_in_model = False")
    
    if args.include_covariates_in_model is not None:
        data_param["include_covariates_in_model"] = True
        data_param["regress_out_covariates"] = False
        logger.info(f"Override: data_param.include_covariates_in_model = True")
        logger.info(f"Override: data_param.regress_out_covariates = False")
    
    # Auto-enable batch correction if batch_path is provided
    if args.batch_path and not data_param.get("regress_out_batch_effects", False):
        data_param["regress_out_batch_effects"] = True
        logger.info(f"Auto-enabled: data_param.regress_out_batch_effects = True (batch_path provided)")
    
    if args.regress_out_batch_effects is not None:
        data_param["regress_out_batch_effects"] = True
        logger.info(f"Override: data_param.regress_out_batch_effects = True")
    
    if args.save_correction_stats is not None:
        data_param["save_correction_stats"] = True
        logger.info(f"Override: data_param.save_correction_stats = True")

    # Validate prediction analysis has test data
    analysis = config.get("analysis")
    if analysis == "prediction":
        train_pct = args.train_pct if args.train_pct is not None else data_param.get("train_pct", 0.0)
        val_pct = args.val_pct if args.val_pct is not None else data_param.get("val_pct", 0.0)
        test_pct = 1.0 - train_pct - val_pct
        
        if test_pct <= 0.0:
            data_param["train_pct"] = 0.9
            data_param["val_pct"] = 0.0
            logger.warning("=" * 60)
            logger.warning("AUTO-CONFIG: Prediction requires test data")
            logger.warning("  Current: train_pct=%.2f, val_pct=%.2f, test_pct=%.2f", 
                        train_pct, val_pct, test_pct)
            logger.warning("  Overriding to: train_pct=0.9, val_pct=0.0, test_pct=0.1")
            logger.warning("  (use_gpytorch=False)")
            logger.warning("=" * 60)

    config["data_param"] = data_param

    return config

def _resolve_scenario(config: Dict[str, Any]) -> tuple:
    """Return (scenario_name, param_name, param_value) based on current config."""
    analysis = config.get("analysis", "").lower()
    data_param = config.get("data_param", {})
    use_het = data_param.get("use_heterogeneity", False)

    if analysis == "gwas" and not use_het:
        return "GWAS", "eta", data_param.get("eta")
    elif analysis == "gwas" and use_het:
        return "GWAS", "corr_bounds", data_param.get("corr_bounds")
    elif analysis == "vardec":
        return "Variance Decomposition", "vardec_scenario", data_param.get("vardec_scenario")
    return "Unknown", "unknown", None

def run_single_rep(config, args, rep_idx=None):
    run_config = copy.deepcopy(config)
    
    if rep_idx is not None:
        run_config["data_param"]["rep_idx"] = rep_idx

    scenario_name, param_name, param_value = _resolve_scenario(run_config)
    current_rep = run_config["data_param"].get("rep_idx", "N/A")

    logger.info("="*60)
    logger.info(f"STARTING REP {current_rep}")
    logger.info(f"  Scenario : {scenario_name}")
    logger.info(f"  {param_name}: {param_value}")
    logger.info(f"  Hypothesis test : {run_config.get('test_type', 'default')}")
    logger.info(f"  Rank            : {run_config.get('rank', 'default')}")
    logger.info(f"  Standardization : {run_config.get('data_param', {}).get('transformation_method', 'int')}")
    logger.info(f"  Output dir      : {run_config.get('output_directory', 'default')}")

    data_param = run_config.get("data_param", {})
    if data_param.get("simulated", False):
        logger.info("  [simulated] rep_idx          : %s", data_param.get("rep_idx", "not set"))
        logger.info("  [simulated] eta              : %s", data_param.get("eta", "not set"))
        logger.info("  [simulated] corr_bounds      : %s", data_param.get("corr_bounds", "not set"))
        logger.info("  [simulated] use_heterogeneity: %s", data_param.get("use_heterogeneity", False))
        logger.info("  [simulated] vardec_scenario  : %s", data_param.get("vardec_scenario", "not set"))
        logger.info("  [simulated] num_samples      : %s", data_param.get("num_samples", "not set"))
        logger.info("  [simulated] num_tasks        : %s", data_param.get("num_tasks", "not set"))
        logger.info("  [simulated] ncausal          : %s", data_param.get("ncausal", "not set"))
    else:
        logger.info("  Phenotype file  : %s", args.pheno_path)
        logger.info("  Genotype file   : %s", args.geno_path)
        if args.annot_path:
            logger.info("  Annotation file : %s", args.annot_path)
        if args.cov_path:
            logger.info("  Covariate file  : %s", args.cov_path)
        if args.batch_path:
            logger.info("  Batch file      : %s", args.batch_path)
    logger.info("="*60)

    # Clear CUDA cache between reps to keep GPU memory tidy
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        results, uid = run_main(
            run_config,
            pheno_path=args.pheno_path,
            geno_path=args.geno_path,
            annot_path=args.annot_path,
            batch_path=args.batch_path,
            cov_path=args.cov_path,
            predict_geno_path=getattr(args, 'predict_geno_path', None),
        )

        if results is None or uid is None:
            logger.error("Analysis failed for rep_idx=%s — no results returned", current_rep)
            raise RuntimeError(f"Analysis failed for rep_idx={current_rep} — no results returned")

        logger.info("Rep %s completed successfully. UID: %s", current_rep, uid)

        if run_config.get("analysis") == "gwas":
            print("\n Log-Likelihood Results DataFrame:")
            print(results.likelihood_results())
            print("\n Beta Estimates DataFrame:")
            print(results.effectsizes())
            print("\n UID:", uid)
            results.display_formatted_results()

            logger.info("="*50)
            logger.info(f"Analysis UID       : {uid}")
            logger.info(f"Output directory   : {results.base_dir}")
            logger.info("="*50)

        return results, uid

    except Exception as e:
        logger.error("Rep %s failed: %s", current_rep, e)
        traceback.print_exc()
        raise

def run_simulation_batch(config: Dict[str, Any], args) -> None:
    """
    Run *n_reps* consecutive repetitions starting at *start_rep_idx*.

    Each rep gets its own deep-copied config with data_param.rep_idx stamped in,
    so the iterations are fully independent.
    """
    n_reps         = args.n_reps
    start_rep_idx  = args.start_rep_idx

    scenario_name, param_name, param_value = _resolve_scenario(config)

    logger.info("="*60)
    logger.info("BATCH SIMULATION STARTING")
    logger.info(f"  Scenario       : {scenario_name}")
    logger.info(f"  {param_name}   : {param_value}")
    logger.info(f"  n_reps         : {n_reps}")
    logger.info(f"  start_rep_idx  : {start_rep_idx}")
    logger.info(f"  Rep index range: {start_rep_idx} to {start_rep_idx + n_reps - 1}")
    logger.info("="*60)

    for rep_offset in range(n_reps):
        rep_idx = start_rep_idx + rep_offset
        logger.info("--- Batch progress: rep %d/%d  (rep_idx=%d) ---",
                    rep_offset + 1, n_reps, rep_idx)

        run_single_rep(config, args, rep_idx=rep_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Rep %d: GPU memory freed (allocated: %.1f MB, reserved: %.1f MB)",
                         rep_idx,
                         torch.cuda.memory_allocated() / 1e6,
                         torch.cuda.memory_reserved() / 1e6)
    logger.info("="*60)
    logger.info("BATCH SIMULATION COMPLETE — all %d reps finished", n_reps)
    logger.info("="*60)


def main():
    """Entry point for the CLI package."""
    args = parse_args()
    
    try:
        # Load and validate configuration
        config = load_config(args.config, args)
        
        # Validate arguments with helpful warnings
        validate_args(args, config)

        if args.n_reps is not None:
            # Batch mode: ignore --rep_idx (start_rep_idx is the starting point)
            if args.rep_idx is not None:
                logger.warning(
                    "--rep_idx was provided together with --n_reps. "
                    "In batch mode --rep_idx is ignored; the range is "
                    "[start_rep_idx, start_rep_idx + n_reps).  "
                    "Current start_rep_idx=%d", args.start_rep_idx
                )
            run_simulation_batch(config, args)
        else:
            # Single-rep mode: use --rep_idx if given, otherwise whatever is in config
            rep_idx = args.rep_idx  # may be None → run_single_rep will just use config value
            run_single_rep(config, args, rep_idx=rep_idx)
        
        logger.info("Script completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

