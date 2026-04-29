import uuid
import os
import logging
import random
import time
from tqdm import tqdm
import gc
import numpy as np
from typing import Dict
import torch
from torchlimix.utils.helper_functions import get_data_snp, stack_data_from_loaders
from torchlimix.result_factory._store_results import StoreResults
from torchlimix.result_factory._store_vardec_results import VarDecResults
from torchlimix.result_factory._store_prediction_results import PredictionResultStore
from torchlimix.lmm._kron_sum import Kron2SumTorch
from torchlimix.lmm._kron_pred import MultiTraitLMMPredict
from torchlimix.vardec._vardec import VarDecMultiTrait
torch.set_default_dtype(torch.float64)
torch.set_num_interop_threads(1)

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_output_directory(config):
    base_output_dir = os.path.expanduser(
        config.get("output_directory", "./torchlimix_results")
    )
    if not config["data_param"]["simulated"]:
        dataset_name = config["data_param"]["dset"]
        output_dir = os.path.join(base_output_dir, dataset_name)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def _create_design_matrices(p, test_type, config, device):
    design_matrices = {}
    
    if test_type == "common":
        design_matrices['A1'] = torch.ones((p, 1), device=device, dtype=torch.double)
    elif test_type == "any":
        design_matrices['A1'] = torch.eye(p, device=device, dtype=torch.double)
    elif test_type == "specific":
        trait_idx = config.get("pheno_idx", 0)
        e = torch.zeros((p, 1), device=device, dtype=torch.double)
        e[trait_idx, 0] = 1.0
        design_matrices['A1'] = e
    elif test_type == "specific_vs_common":
        design_matrices['A0'] = torch.ones((p, 1), device=device, dtype=torch.double)
        trait_idx = config.get("pheno_idx", 0)
        A1 = torch.zeros((p, 2), device=device, dtype=torch.double)
        A1[:, 0] = 1.0
        A1[trait_idx, 1] = 1.0
        design_matrices['A1'] = A1
    elif test_type == "any_vs_common":
        design_matrices['A0'] = torch.ones((p, 1), device=device, dtype=torch.double)
        design_matrices['A1'] = torch.eye(p, device=device, dtype=torch.double)
    
    return design_matrices

def _get_test_config(test_type: str, p: int) -> Dict:
    """Scan order and degrees of freedom per test type."""
    configs = {
        "common":             {"scans": ["A1"],                        "dfs": {"10": 1}},
        "any":                {"scans": ["A1"],                        "dfs": {"10": p}},
        "specific":           {"scans": ["A1"],                        "dfs": {"10": 2}},
        "specific_vs_common": {"scans": ["A0", "A1"],  "dfs": {"10": 1, "20": 2, "21": 1}},
        "any_vs_common":      {"scans": ["A0", "A1"],  "dfs": {"10": 1, "20": p, "21": p - 1}},
    }
    if test_type not in configs:
        raise ValueError(f"Unknown test_type: {test_type}")
    return configs[test_type]

def _run_scan(
    scanner, 
    A1: torch.Tensor, 
    X: torch.Tensor, 
    chunk_size: int, 
    cache_clear_interval: int,
    progress_callback: callable = None,
):
    """
    Run scanner with optimal method selection.
    
    Args:
        scanner: KronFastScannerTorch instance
        A1: Design matrix
        X: SNP data
        chunk_size,
        use_streams: Use CUDA streams if available
        progress_callback: Optional progress callback
    """
    if torch.cuda.is_available() and X.is_cuda:
        return scanner.scan_batched_gpu(A1, X, chunk_size, cache_clear_interval, progress_callback=progress_callback)
    else:
        A1_cpu = A1.detach().double().cpu()
        X_cpu = X.detach().double().cpu()
        return scanner.scan_batched_cpu(A1_cpu, X_cpu, progress_callback=progress_callback)


def process_snps(
    num_snps: int,
    chunk_size: int,
    cache_clear_interval: int,
    X_snp_all: torch.Tensor,
    scanner,
    C0,
    C1,
    config: dict,
    device: str,
    results,
    p: int,
    test_type: str,
    show_progress: bool = True,
):
    from torchlimix.stats._lrt_values import lrt_values
    start_time = time.time()

    design_matrices = _create_design_matrices(p, test_type, config, device)
    test_config = _get_test_config(test_type, p)
    n_scans = len(test_config["scans"])

    lml0 = scanner.null_lml.cpu() if isinstance(scanner.null_lml, torch.Tensor) else scanner.null_lml
    scale_H0 = scanner.null_scale.item() if hasattr(scanner.null_scale, 'item') else scanner.null_scale
    beta0 = scanner.null_beta.cpu().reshape(scanner._ntraits, scanner._ncovariates).T if isinstance(scanner.null_beta, torch.Tensor) else scanner.null_beta
    beta0_se = scanner.null_beta_se.cpu().reshape(scanner._ntraits, scanner._ncovariates).T if isinstance(scanner.null_beta_se, torch.Tensor) else scanner.null_beta_se

    results.save_null_model(
        beta0=beta0, beta0_se=beta0_se,
        lml0=lml0, scale_H0=scale_H0,
        n_covariates=scanner._ncovariates,
        n_traits=scanner._ntraits,
    )

    last_update = [0]
    pbar = tqdm(
        total=num_snps * n_scans,
        desc=f"Processing SNPs ({test_type})",
        unit="SNP",
        ncols=100,
        disable=not show_progress,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    def progress_callback(n_processed, n_total):
        delta = n_processed - last_update[0]
        if delta > 0:
            pbar.update(delta)
            last_update[0] = n_processed


    scan_keys = test_config["scans"]

    # Scan 0 
    lml1, scale_H1, beta1, beta1_se = None, None, None, None
    if len(scan_keys) >= 1:
        last_update[0] = 0
        scan0 = _run_scan(
            scanner, design_matrices[scan_keys[0]], X_snp_all,
            chunk_size, cache_clear_interval,
            progress_callback=progress_callback,
        )
        lml1     = scan0["lml"]       # already CPU (from mmap refactor)
        scale_H1 = scan0["scale"]
        beta1    = scan0["effsizes1"]
        beta1_se = scan0["effsizes1_se"]
        del scan0                      # free the rest (effsizes0, pve, etc.)

    # Scan 1
    lml2, scale_H2, beta2, beta2_se = None, None, None, None
    if len(scan_keys) >= 2:
        last_update[0] = 0
        scan1 = _run_scan(
            scanner, design_matrices[scan_keys[1]], X_snp_all,
            chunk_size, cache_clear_interval,
            progress_callback=progress_callback,
        )
        lml2     = scan1["lml"]
        scale_H2 = scan1["scale"]
        beta2    = scan1["effsizes1"]
        beta2_se = scan1["effsizes1_se"]
        del scan1

    pbar.close()

    del design_matrices
    torch.cuda.empty_cache()

    scan_elapsed = time.time() - start_time
    logger.info(f"GPU scans done in {scan_elapsed:.1f}s")

    dfs = test_config["dfs"]
    all_indices = np.arange(num_snps, dtype=np.int64)

    lrt10 = lrt_values(lml0, lml1).squeeze() if lml1 is not None else None
    lrt20 = lrt_values(lml0, lml2).squeeze() if lml2 is not None else None
    lrt21 = lrt_values(lml1, lml2).squeeze() if lml2 is not None else None

    store_start = time.time()
    results.add_likelihood_result(
        snp_indices=all_indices,
        lml0=lml0, lml1=lml1, lml2=lml2,
        lrt10=lrt10, lrt20=lrt20, lrt21=lrt21,
        df10=dfs.get("10"), df20=dfs.get("20"), df21=dfs.get("21"),
        scale_H0=scale_H0, scale_H1=scale_H1, scale_H2=scale_H2,
        C0=C0, C1=C1,
    )

    del lrt10, lrt20, lrt21
    del lml1, lml2, scale_H1, scale_H2

    results.add_beta_result(
        snp_indices=all_indices,
        beta1=beta1, beta1_se=beta1_se,
        beta2=beta2, beta2_se=beta2_se,
    )

    del beta1, beta1_se, beta2, beta2_se

    store_elapsed = time.time() - store_start
    logger.info(f"Result storage: {store_elapsed:.1f}s")

    elapsed = time.time() - start_time
    snps_per_second = num_snps / elapsed if elapsed > 0 else 0
    logger.info(
        f"SNP processing complete: {num_snps:,} SNPs in "
        f"{elapsed:.1f}s ({snps_per_second:.0f} SNPs/sec)"
    )

    return elapsed, snps_per_second

def run_mt_lmm(
    config, 
    analysis_type,
    train_args, 
    tl, 
    vl, 
    ttl, 
    data_meta,
    uid,
    output_dir
):
    """
    Run multi-trait LMM analysis (GWAS, variance decomposition, or prediction).
    """
    X_snp_all = None
    G_stable = None
    scanner = None

    device = train_args["device"]

    # Extract all parameters from config
    data_param = config.get("data_param", {})
    rank_config = config.get("rank")
    simulated = data_param.get("simulated", False)
    
    if simulated:
        rep_idx = data_param.get("rep_idx", None)
        eta = data_param.get("eta", None)
        corr_bounds = data_param.get("corr_bounds", None)
        vardec_scenario = data_param.get("vardec_scenario", None)
        use_heterogeneity = data_param.get("use_heterogeneity", False)
        
        logger.info("="*70)
        logger.info("ANALYSIS PARAMETERS (SIMULATED):")
        logger.info("="*70)
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Rank: {rank_config}")
        logger.info(f"  rep_idx: {rep_idx}")
        logger.info(f"  eta: {eta}")
        logger.info("="*70 + "\n")
    else:
        rep_idx = None
        eta = None
        corr_bounds = None
        vardec_scenario = None
        use_heterogeneity = False
        
        logger.info("="*70)
        logger.info("ANALYSIS PARAMETERS:")
        logger.info("="*70)
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Rank: {rank_config}")
        logger.info("="*70 + "\n")
    
    master = tl.dataset.master
    correction_metadata = master.correction_metadata  # Assuming this points to the "corrections" block in your JSON
    
    # Check if ANY correction method was applied
    any_correction_applied = (
        correction_metadata.get('batch_correction', {}).get('applied', False) or
        correction_metadata.get('covariate_correction', {}).get('applied', False) or
        correction_metadata.get('transformations', {}).get('applied', False)
    )

    phenotype_data = {
        'corrected': master.df_tensor_full,
        'uncorrected': getattr(master, 'df_uncorrected', None),
        'corrections_applied': any_correction_applied
    }
    
    # Log what corrections were applied
    if correction_metadata:
        logger.info("DATA PREPROCESSING APPLIED:")
        if correction_metadata.get('batch_correction', {}).get('applied'):
            logger.info("  Batch correction applied")
        if correction_metadata.get('covariate_correction', {}).get('applied'):
            logger.info("  Covariates regressed out")
        elif correction_metadata.get('covariate_correction', {}).get('included_in_model'):
            logger.info("  Covariates included in model (not regressed out)")
        if correction_metadata.get('transformations', {}).get('applied'):
            method = correction_metadata['transformations'].get('method', 'unknown')
            logger.info(f"  Transformation: {method}")
        logger.info("")

    # Initialize results storage based on analysis type
    if analysis_type == "vardec":
        logger.info("Initializing VarDecResults storage...")
        results = VarDecResults(
            config=config,
            output_dir=output_dir,
            scenario_id=vardec_scenario,
            rep_idx=rep_idx,
            uid=uid,
            rank=rank_config,
            correction_metadata=correction_metadata,
            phenotype_data=phenotype_data  
        )  
    
    elif analysis_type == "gwas":
        logger.info("Initializing StoreResults storage...")
        results = StoreResults(
            output_dir=output_dir, 
            uid=uid, 
            corr_bounds=corr_bounds, 
            rep_idx=rep_idx, 
            eta=eta, 
            simulation_info=data_meta.get("simulation_info", None),
            rank=rank_config,
            test_type=config['test_type'],
            correction_metadata=correction_metadata,
            phenotype_data=phenotype_data,
        )
    
    elif analysis_type == "prediction":
        logger.info("Initializing PredictionResultStore...")
    
        sample_batch = next(iter(tl))
        n_traits = sample_batch[2].shape[1] if len(sample_batch) > 2 else data_meta.get('num_tasks', 1)
        
        results = PredictionResultStore(
            output_dir=output_dir,
            uid=uid,
            method="Analytical_BLUP",
            n_traits=n_traits,
            rep_idx=rep_idx,
            eta=eta,
            corr_bounds=corr_bounds,
            use_heterogeneity=use_heterogeneity,
            simulation_info=data_meta.get("simulation_info", None),
            correction_metadata=correction_metadata,
            phenotype_data=phenotype_data,
        )
    else:
        raise ValueError(f"Unknown analysis type: '{analysis_type}'. Must be 'gwas', 'vardec', or 'prediction'")
    
    # Stack data based on analysis type
    if analysis_type == "prediction":
        # Prediction needs separate train/val and test data
        logger.info("Stacking data for prediction...")
        
        if ttl is None:
            raise ValueError("Prediction requires test data loader (ttl)")
        
        dtype = torch.float64
        
        # Train + validation data
        G_train = stack_data_from_loaders(tl=tl, vl=vl, ttl=None, index=1, device=device).to(dtype=dtype)
        Y_train = stack_data_from_loaders(tl=tl, vl=vl, ttl=None, index=2, device=device).to(dtype=dtype)
        n_train, p = Y_train.shape
        
        # Test data
        G_test = stack_data_from_loaders(tl=None, vl=None, ttl=ttl, index=1, device=device).to(dtype=dtype)
        Y_test = stack_data_from_loaders(tl=None, vl=None, ttl=ttl, index=2, device=device).to(dtype=dtype)
        n_test = G_test.shape[0]
        
        logger.info(f"Train+Val: {n_train} samples, Test: {n_test} samples, Traits: {p}")
        
        # For prediction, we don't need SNP data or full stacking
        n = n_train
        
    else:
        logger.info("Extracting data directly from master dataset...")
        import gc

        master = tl.dataset.master
        train_idx = torch.as_tensor(
            master.split_indices['train'], dtype=torch.long
        )

        # Move genotype (SNP) data to device, then release the CPU original
        X_snp_all = master.gen_data_tensor_full[train_idx].to(device)
        master.gen_data_tensor_full = None
        gc.collect()
        logger.info(f"X_snp_all on {device}: {X_snp_all.shape}")

        # Move G_stable to device, release CPU original
        G_stable = master.G_stable[train_idx].to(device)
        master.G_stable = None
        gc.collect()
        logger.info(f"G_stable on {device}: {G_stable.shape}")

        # Move phenotype to device, release CPU original
        Y_stacked = master.df_tensor_full[train_idx].to(device)
        master.df_tensor_full = None
        gc.collect()

        n, p = Y_stacked.shape
        logger.info(f"Data shape: {n} samples × {p} traits")

        if master.covariate_matrix is not None:
            covariates = master.covariate_matrix[train_idx].to(device)
        else:
            covariates = None

        # Free the SplitView copies that the DataLoader was holding
        for loader in (tl, vl, ttl):
            if loader is not None and hasattr(loader, 'dataset'):
                ds = loader.dataset
                for attr in ('data_tensor', 'gen_data_tensor'):
                    if hasattr(ds, attr):
                        setattr(ds, attr, None)
        gc.collect()

        del train_idx
    
    # Extract covariates if present (for GWAS/vardec)
    if analysis_type != "prediction":
        # Debug info from already-stacked tensors (no extra batch load)
        #logger.info(f"DEBUG: X_snp_all={X_snp_all.shape}, G_stable={G_stable.shape}, "
        #             f"Y_stacked={Y_stacked.shape}, covariates={'None' if covariates is None else covariates.shape}")
        #logger.info(f"DEBUG: data_param regress_out_covariates = {data_param.get('regress_out_covariates')}")
        #logger.info(f"DEBUG: data_param include_covariates_in_model = {data_param.get('include_covariates_in_model')}")

        A = torch.eye(p, device=device)
        M = torch.ones((n, 1), device=device)

        if covariates is not None:
            logger.info(f"Adding {covariates.shape[1]} covariates to fixed effects matrix M")
            M = torch.cat([M, covariates], dim=1)
            logger.info(f"Updated M shape: {M.shape}")
        else:
            logger.info("Using intercept-only fixed effects matrix M")
    
    # Validate and set rank
    rank = rank_config if isinstance(rank_config, int) and 1 <= rank_config <= p else (
        logger.warning(f"Invalid rank '{rank_config}' provided. Using p={p}.") or p
    )
    
    if rank > p:
        raise ValueError(
            f"Rank ({rank}) cannot exceed the number of traits ({p}). "
            f"Please set rank ≤ {p} (use {p} for full rank)."
        )
    logger.info(f"Selected rank: {rank}")
    
    # VARIANCE DECOMPOSITION
    if analysis_type == "vardec":
        logger.info("="*70)
        logger.info("RUNNING VARIANCE DECOMPOSITION ANALYSIS")
        logger.info("="*70)
        
        logger.info("Initializing Kron model...")
        kron_model = Kron2SumTorch(
            Y=Y_stacked, 
            A=A, 
            X=M,
            G=G_stable.to(device),
            data_meta=data_meta,  
            rank=rank, 
            device=device, 
            restricted=True, 
            config=config
        )

        logger.info("Fitting null model...")
        optimization_results = kron_model.fit()
        results.add_optimization_metrics(optimization_results)
        
        C0 = kron_model.C0
        C1 = kron_model.C1
        logger.info(f"Extracted covariances: C0 trace={torch.trace(C0):.4f}, C1 trace={torch.trace(C1):.4f}")
        
        results.add_fitted_covariances(C0, C1)
        
        logger.info("Performing variance decomposition...")
        vardec = VarDecMultiTrait(C0, C1, nsamples=n)

        try:
            variance_results = vardec.get_results()
            #vardec._print_results()  
        except Exception as e:
            logger.error("Variance decomposition raised an exception: %s", e)
            import traceback
            traceback.print_exc()
            variance_results = None

        if variance_results is None or variance_results.get('failed', False):
            logger.warning("Variance decomposition failed - NaN values will be stored")

        if simulated:
            if data_meta['simulation_info'] is not None:
                results.add_ground_truth(data_meta['simulation_info'])
                logger.info(f"Simulation info stored for scenario {vardec_scenario}")
            else:
                logger.warning("No simulation_info found in data_meta for vardec analysis")        
        results.add_vardec_results(variance_results)
        results._print_summary()
        results.save()
        
        del vardec, kron_model
    
    # GWAS
    elif analysis_type == "gwas":
        logger.info("="*70)
        logger.info("RUNNING GWAS ANALYSIS")
        logger.info("="*70)
        
        gwas_failed = False
        kron_model = None
        scanner = None
        
        num_snps = data_meta['total_snps']
        test_type = config.get("test_type", "common").lower()
        logger.info(f"Running {test_type} effect test")
        
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

            device_idx       = torch.cuda.current_device()
            total_memory     = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_idx)
            reserved_memory  = torch.cuda.memory_reserved(device_idx)
            free_memory_mb   = (total_memory - reserved_memory) / 1e6

            n_samples    = data_meta['total_samples']
            n_traits     = data_meta['num_tasks']
            n_covariates = M.shape[1]
            a1_cols           = n_traits if test_type in ["any", "any_vs_common"] else 2
            total_matrix_size = n_traits * n_covariates + a1_cols
            chol_dim = n_samples * n_traits

            # Per-SNP memory estimate
            # The cholesky_solve no longer broadcasts Z_chol per SNP
            # (reshape trick: single solve, constant Z_chol).
            # Remaining per-SNP costs:
            #   1. Data vectors:            n_samples              (×2)
            #   2. XtX-style matrices:      total_matrix_size²     (×4)
            #   3. Trait covariance ops:     n_traits²             (×6)
            #   4. Intermediate vectors:     total_matrix_size     (×4)
            #   5. Reshape solve RHS/result: chol_dim × total_matrix_size (×2)
            SAFETY_FACTOR = 2.0
            bytes_per_snp = int((
                n_samples * 8 * 2 +
                total_matrix_size ** 2 * 8 * 4 +
                n_traits ** 2 * 8 * 6 +
                total_matrix_size * 8 * 4 +
                chol_dim * total_matrix_size * 8 * 2
            ) * SAFETY_FACTOR)
            memory_per_snp_mb = bytes_per_snp / 1e6

            gpu_name    = torch.cuda.get_device_name(device_idx).upper()
            total_gb    = total_memory / (1024 ** 3)
            high_end_gpus = ("A100", "H100", "H200", "A6000", "L40", "GH200")
            is_high_end   = any(tag in gpu_name for tag in high_end_gpus)

            if is_high_end and total_gb > 60:
                mem_fraction = (0.55 if n_samples < 2000 else
                                0.45 if n_samples < 10000 else 0.35)
                hard_cap     = (2048 if n_samples < 500 else
                                1024 if n_samples < 2000 else
                                512  if n_samples < 10000 else 256)
                alignment    = 64
            elif is_high_end:
                mem_fraction = (0.40 if n_samples < 2000 else
                                0.30 if n_samples < 10000 else 0.20)
                hard_cap     = (1024 if n_samples < 500 else
                                512  if n_samples < 2000 else
                                256  if n_samples < 10000 else 128)
                alignment    = 64
            else:
                mem_fraction = (0.50 if n_samples < 500 else
                                0.35 if n_samples < 2000 else
                                0.25 if n_samples < 10000 else 0.15)
                hard_cap     = (512  if n_samples < 500 else
                                256  if n_samples < 2000 else
                                128  if n_samples < 10000 else 64)
                alignment    = 32

            optimal_chunk = int(free_memory_mb * mem_fraction / memory_per_snp_mb)
            chunk_size    = max(
                (min(optimal_chunk, hard_cap, num_snps) // alignment) * alignment,
                alignment,
            )

            # Dry-run with the actual dominant allocation pattern
            while chunk_size > alignment:
                try:
                    t1 = torch.empty(
                        chunk_size, total_matrix_size, total_matrix_size,
                        device=device, dtype=torch.float64,
                    )
                    del t1
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    old = chunk_size
                    chunk_size = max((chunk_size // 2 // alignment) * alignment, alignment)
                    logger.warning(f"Dry-run OOM at chunk_size={old}, reducing to {chunk_size}")
                    torch.cuda.empty_cache()

            cache_clear_interval = 8 if is_high_end else 4

            logger.info(f"GPU: {gpu_name} ({total_gb:.0f} GB, "
                        f"{'high-end' if is_high_end else 'standard'} profile)")
            logger.info(f"Post-init GPU memory: {free_memory_mb:.0f} MB truly free "
                        f"(reserved={reserved_memory/1e6:.0f} MB, "
                        f"allocated={allocated_memory/1e6:.0f} MB)")
            logger.info(f"Cholesky dimension: {chol_dim} "
                        f"(n_samples={n_samples} x n_traits={n_traits})")
            logger.info(f"Memory per SNP: {memory_per_snp_mb:.3f} MB  |  "
                        f"Fraction: {mem_fraction}  |  Hard cap: {hard_cap}  |  "
                        f"Alignment: {alignment}  |  Safety: {SAFETY_FACTOR}x")
            logger.info(f"Auto-determined chunk size: {chunk_size}")
        else:
            chunk_size           = min(200, num_snps)
            cache_clear_interval = 4

        logger.info(f"Using chunk size: {chunk_size}")

        try:
            logger.info("Initializing Kron model...")
            kron_model = Kron2SumTorch(
                Y=Y_stacked, 
                A=A, 
                X=M,
                G=G_stable.to(device),
                data_meta=data_meta,  
                rank=rank, 
                device=device, 
                restricted=False, 
                config=config
            )
            optimization_results = kron_model.fit()
            results.add_optimization_metrics(optimization_results)  
            
            results.add_cov(kron_model.C0, kron_model.C1)
            C0 = kron_model.C0
            C1 = kron_model.C1
            logger.info(f"Extracted covariances: C0 trace={torch.trace(C0):.4f}, C1 trace={torch.trace(C1):.4f}")
    
            logger.info("Initializing fast scanner for SNP analysis...")
            scanner = kron_model.get_fast_scanner()    
            logger.info(f"Scanner initialized. Null scale: {scanner.null_scale.item():.6f}")

            C0 = kron_model.C0.detach().clone()
            C1 = kron_model.C1.detach().clone()       
            process_snps(
                num_snps=num_snps,
                chunk_size=chunk_size,
                cache_clear_interval=cache_clear_interval,
                X_snp_all=X_snp_all,
                scanner=scanner,
                C0=C0,
                C1=C1,
                config=config,
                device=device,
                results=results,
                p=p,
                test_type=test_type,
                show_progress=True,  
            )
        except Exception as e:
            logger.error(f"Error during GWAS analysis: {e}")
            import traceback
            traceback.print_exc()
            gwas_failed = True
        
        if gwas_failed:
            logger.warning("GWAS analysis failed - NaN values will be stored")
        else:
            logger.info("GWAS analysis completed successfully") 
    
    # PREDICTION
    elif analysis_type == "prediction":
        logger.info("="*70)
        logger.info("RUNNING PREDICTION ANALYSIS")
        logger.info("="*70)
        
        # Setup model matrices for training
        A = torch.eye(p, dtype=dtype, device=device)
        M = torch.ones((n_train, 1), dtype=dtype, device=device)
        
        logger.info(f"Fitting Kron model (rank={rank})...")
        kron_model = Kron2SumTorch(
            Y=Y_train,
            A=A,
            X=M,
            G=G_train,
            data_meta=data_meta,
            rank=rank,
            device=device,
            restricted=False,
            config=config
        )
        kron_model.fit()
        
        C0 = kron_model.C0.to(dtype=dtype, device=device)
        C1 = kron_model.C1.to(dtype=dtype, device=device)
        logger.info(f"C0 trace: {torch.trace(C0):.4f}, C1 trace: {torch.trace(C1):.4f}")
        
        # Get beta
        if hasattr(kron_model, 'beta') and kron_model.beta is not None:
            beta = kron_model.beta.to(dtype=dtype, device=device)
        elif hasattr(kron_model, '_beta') and kron_model._beta is not None:
            beta = kron_model._beta.to(dtype=dtype, device=device)
        else:
            beta = torch.zeros(1, p, dtype=dtype, device=device)
        
        if beta.ndim == 1:
            beta = beta.reshape(-1, p)
        
        # Compute kinship matrices at the same scale the model was fitted on
        logger.info("Computing kinship matrices...")
        train_dataset = tl.dataset
        test_dataset  = ttl.dataset

        # z-scored genotypes: (n, S) with all SNPs
        G_train_z = train_dataset.gen_data_tensor.to(device)   # (n_train, S)
        G_test_z  = test_dataset.gen_data_tensor.to(device)    # (n_test,  S)

        S_snps = G_train_z.shape[1]
        scale  = S_snps * (master._G_norm ** 2)

        #logger.info(f"G_train_z: {G_train_z.shape}, G_test_z: {G_test_z.shape}")
        #logger.info(f"Scale factor: S={S_snps} × G_norm²={master._G_norm**2:.4f} = {scale:.4f}")

        K_train      = (G_train_z @ G_train_z.T) / scale
        K_test_train = (G_test_z  @ G_train_z.T)  / scale
        K_test_test  = (G_test_z  @ G_test_z.T)   / scale

        del G_train_z, G_test_z
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        X_train = torch.ones((n_train, 1), dtype=dtype, device=device)
        X_test = torch.ones((n_test, 1), dtype=dtype, device=device)
        
        # Initialize predictor
        logger.info("Running BLUP prediction...")
        predictor = MultiTraitLMMPredict(
            Y_train=Y_train,
            beta=beta,
            C0=C0,
            C1=C1,
            K_train=K_train,
            X_train=X_train,
            device=device,
            dtype=dtype
        )
        
        pred_result = predictor.predict(
            X_test=X_test,
            K_test_train=K_test_train,
            K_test_test=K_test_test,
            return_variance=True
        )
        
        pred_mean = pred_result['mean']
        pred_var = pred_result['variance']

        # Detect scenario: ground truth available?
        has_ground_truth = not torch.isnan(Y_test).all()

        # Store predictions (y_true=None for external prediction)
        results.store_predictions(
            pred_mean=pred_mean,
            pred_var=pred_var,
            C0=C0,
            C1=C1,
            y_true=Y_test if has_ground_truth else None,
        )

        # Export human-readable CSV (always)
        sample_idx = data_meta.get('prediction_sample_index')
        results.export_predictions_csv(sample_index=sample_idx)

        if has_ground_truth:
            # Scenario A: internal test split — compute & log metrics
            logger.info("Ground truth available. Computing prediction metrics...")
            metrics = results.compute_metrics(pred_mean=pred_mean, y_true=Y_test)

            logger.info(f"  Overall MSE:  {metrics['overall']['mse']:.6f}")
            logger.info(f"  Overall MAE:  {metrics['overall']['mae']:.6f}")
            logger.info(f"  Mean Corr:    {metrics['overall']['correlation_mean']:.4f}")
            logger.info(f"  Mean R²:      {metrics['overall']['r2_mean']:.4f}")

            if results.confidence_intervals.get('coverage'):
                cov = results.confidence_intervals['coverage']
                logger.info(f"  CI Coverage (95%): {cov['overall']:.4f}")
        else:
            # Scenario B: external prediction — nothing more to compute
            logger.info("External prediction mode (no ground truth).")
            logger.info("Predictions and uncertainty saved to predicted_phenotypes.csv")

        results.save()

        del predictor, kron_model, K_train, K_test_train, K_test_test
        del G_train, G_test, Y_train, Y_test

    logger.info("Cleaning up memory...")
    
    if analysis_type != "prediction":
        del X_snp_all, Y_stacked, G_stable
        if covariates is not None:
            del covariates

    del tl, vl, ttl
    torch.cuda.empty_cache()
    
    logger.info("="*70)
    logger.info(f"{analysis_type.upper()} ANALYSIS COMPLETE")
    logger.info("="*70)
    
    return results


def run_main(config, 
             pheno_path=None, 
             geno_path=None, 
             annot_path=None, 
             batch_path=None, 
             cov_path=None,
             predict_geno_path=None
             ):
    """
    Run the main analysis pipeline.
    
    Args:
        config: Configuration dictionary with all analysis parameters
        pheno_path: Path to phenotype file (not stored in config)
        geno_path: Path to genotype file (not stored in config)
        annot_path: Path to annotation file (not stored in config)
        batch_path: Path to batch file (not stored in config)
        cov_path: Path to covariate file (not stored in config)
    
    Note: All other parameters are read from config['data_param'].
    File paths are passed separately because they're data-specific.
    """
    
    # Set random seed for reproducibility (before any data generation or computation)
    seed = config.get("data_param", {}).get("seed", 
           config.get("data_param", {}).get("rep_idx", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Determine analysis type
    analysis_type = config.get("analysis", "gwas").lower()
    if analysis_type not in ["gwas", "vardec", "prediction"]:
        raise ValueError(f"Unknown analysis type: '{analysis_type}'. Valid: 'gwas', 'vardec', 'prediction'")
    logger.info(f"Analysis type: {analysis_type}")

    output_dir = setup_output_directory(config)
    try:
        # Setup device and UID
        device, uid = setup(config)
        train_args = {"config": config, "device": device, "uid": uid}
        verbose = config.get("verbose", False)
        # Load data
        logger.info("Loading data...")
        tl, vl, ttl, data_meta = get_data_snp(
            config,
            pheno_path=pheno_path,
            geno_path=geno_path,
            annot_path=annot_path,
            batch_path=batch_path,
            cov_path=cov_path,
            output_dir=output_dir,
            predict_geno_path=predict_geno_path,
            verbose=verbose,
        )
        
        # Validate data loading
        if tl is None:
            raise RuntimeError("Failed to load training data")
        
        # Run Analysis based on type
        results = run_mt_lmm(
            config=config,
            analysis_type=analysis_type,
            train_args=train_args,
            tl=tl,
            vl=vl,
            ttl=ttl,
            data_meta=data_meta,
            uid=uid,
            output_dir=output_dir
        )
            
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        raise
       
    gc.collect()
    torch.cuda.empty_cache()
    
    return results, uid

def setup(config):
    uid = str(uuid.uuid4())
    if config is None:
        raise ValueError("Config cannot be None.")

    if "num_threads" in config:
        n_threads = config["num_threads"]
    else:
        n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK",
                        os.environ.get("OMP_NUM_THREADS", "4")))

    torch.set_num_threads(n_threads)
    logger.info(f"Torch threads: {n_threads} intra-op, 1 inter-op")

    logger.info(f"Running with UID: {uid}")
    device = get_device(config)
    logger.info(f"Using device: {device}")

    return device, uid

def get_device(config):
    """
    Determine the device ('cuda', 'mps', or 'cpu') based on configuration and availability.
    """
    if torch.cuda.is_available() and config.get("gpus", 0) > 0:
        device = "cuda"  # Use CUDA if available and requested
    else:
        device = "cpu"  # Default to CPU if no GPU is available
    # Inform user if CUDA was selected but is unavailable
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    return device




