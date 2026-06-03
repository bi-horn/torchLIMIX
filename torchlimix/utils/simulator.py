import numpy as np
import pandas as pd
import torch
from torchlimix.utils._genotype_preparation import prepare_kinship_pipeline
torch.set_default_dtype(torch.float64)

class PhenoSimulator:
    """
    Phenotype Simulator for GxE effects simulation incorporating omega-gamma reasoning.
    """

    def __init__(self, dset, X, P=2, eta=1.0, rep_idx=None, chrom=None, pos=None, reference_trait=0, precomputed_kinship=None):
        """
        Initialize simulator

        Parameters:
            X: Genotype matrix (N x S)
            P: Number of traits/contexts
            eta: Proportionality factor for rescaling effects across contexts
            rep_idx: Replication index for reproducibility
            chrom: Chromosome array (S,)
            pos: Position array (S,)
        """
        self.dset = dset
        self.X = X
        self.chrom = chrom
        self.pos = pos
        self.N = X.shape[0]
        self.num_snps_total = X.shape[1]
        self.P = P
        self.reference_trait = reference_trait
        self.eta = eta
        if precomputed_kinship is not None:
            self.XX = precomputed_kinship
        else:
            self.XX = self._compute_kinship_matrix()
        
        # Dataset-specific parameter configurations
        self.DATASET_PARAMS = {
            'thaliana_horton': {
                'v_s': 0.15,
                'v_bg': 0.50,
                'omega': 0.60,
                'gamma': 0.40,
                'description': 'Arabidopsis thaliana_horton - moderate QTL effects, strong population structure'
            },
            'thaliana_1001': {
                'v_s': 0.15,
                'v_bg': 0.50,
                'omega': 0.60,
                'gamma': 0.40,
                'description': 'Arabidopsis thaliana_horton - moderate QTL effects, strong population structure'
            }
        }

        # Validate dataset parameter
        if dset not in self.DATASET_PARAMS:
            raise ValueError(f"Unknown dataset '{dset}'. Supported datasets: {list(self.DATASET_PARAMS.keys())}")

        # Set default parameters for the dataset
        self.default_params = self.DATASET_PARAMS[dset]
        
        print(f"[INFO] Initialized simulator for {dset}")
        print(f"[INFO] {self.default_params['description']}")
        print(f"[INFO] Default parameters - v_s: {self.default_params['v_s']}, "
              f"v_bg: {self.default_params['v_bg']}, omega: {self.default_params['omega']}, "
              f"gamma: {self.default_params['gamma']}")
        
        self._initialize_rngs(rep_idx)

    def _initialize_rngs(self, rep_idx):
        """
        Initialize all random number generators with consistent seeding strategy.
        SNP and region selection operations use the same rep_idx-dependent seed,
        while other components get different fixed seeds.
        """
        # Base seed 
        if rep_idx is None:
            base_seed = 100
            print(f"[INFO] Using test base_seed: {base_seed}")
        else:
            base_seed = 100 + rep_idx
            print(f"[INFO] Using replicate-based base_seed: {base_seed} (rep_idx: {rep_idx})")

        # SNP and region selection 
        self.rng_snp_selection = np.random.default_rng(seed=42)    # For SNP selection
        self.rng_region_selection = np.random.default_rng(seed=42) # For region selection (same as SNP)

        # Different fixed seeds for other components (consistent across replicates)
        self.rng_effect_sizes = np.random.default_rng(seed=base_seed)
        self.rng_background = np.random.default_rng(seed=base_seed + 100)
        self.rng_hidden = np.random.default_rng(seed=base_seed + 200)
        self.rng_noise = np.random.default_rng(seed=base_seed + 300)

        self.rng = self.rng_snp_selection
        self.rng_snp = self.rng_effect_sizes

    def get_default_params(self):
        """Get default parameters for the current dataset."""
        return self.default_params.copy()

    def _compute_kinship_matrix(self):
        """
        Compute kinship matrix K with PSD guarantee.
        Uses the same prepare_kinship_pipeline as original simulator.
        """
        dynamic_chunk = 100000 if self.N <= 1000 else 10000
        K, *_ = prepare_kinship_pipeline(
            G=self.X.values,
            chunk_size=dynamic_chunk,  
            debug=False
        )
        return K
    
    def setEta(self, eta):
        """Set the rescaling factor for the second trait"""
        self.eta = eta

    def selectRnd(self, n_sel, n_all, shape=None, rng=None):
        """
        Reproducibly select n_sel elements from n_all using specified RNG.

        Parameters:
            n_sel: Number of True values
            n_all: Total elements
            shape: Optional shape for reshaping output (e.g. (S, P))
            rng: Random number generator to use (defaults to self.rng_snp_selection)

        Returns:
            A boolean mask with n_sel True values.
        """
        if rng is None:
            rng = self.rng_snp_selection 
            
        mask = np.zeros(n_all, dtype=bool)
        mask[:n_sel] = True
        rng.shuffle(mask)
        if shape is not None:
            return mask.reshape(shape)
        return mask

    def getRegion(self, size=None, min_nSNPs=1, chrom_i=None, pos_min=None, pos_max=None):
        """
        Parameters:
            size: Region size in base pairs
            min_nSNPs: Minimum number of SNPs required in the region
            chrom_i: Chromosome to filter (int)
            pos_min: Lower bound on position
            pos_max: Upper bound on position

        Returns:
            Xr: Genotype submatrix for the region
            region: Tuple (chromosome, pos_start, pos_end)
            indices: Indices of selected SNPs
        """
        REGION_SIZES = {
            'human': 30000,
            'thaliana_horton': 15000,
            'thaliana_1001': 15000,
            'yeast': 10000,
        }
        # Use dataset-specific default region size if not specified
        if size is None:
            if hasattr(self, 'dset') and self.dset in REGION_SIZES:
                size = REGION_SIZES[self.dset]
            else:
                size = 12000  

        chrom = np.asarray(self.chrom)
        pos = np.asarray(self.pos)

        if chrom_i is None:
            chrom_i = self.rng_region_selection.choice(np.unique(chrom))

        mask = chrom == chrom_i
        if pos_min is not None:
            mask &= pos > pos_min
        if pos_max is not None:
            mask &= pos < pos_max

        chrom_pos = np.where(mask)[0]
        if len(chrom_pos) < min_nSNPs:
            raise ValueError(f"Not enough SNPs on chromosome {chrom_i} after filtering.")

        positions = pos[chrom_pos]
        start_idx = self.rng_region_selection.choice(chrom_pos)

        pos_start = pos[start_idx]
        pos_end = pos_start + size
        region_mask = (chrom == chrom_i) & (pos >= pos_start) & (pos <= pos_end)
        indices = np.where(region_mask)[0]

        if len(indices) < min_nSNPs:
            raise ValueError("Region does not contain enough SNPs.")

        print(f"[INFO] Selected region: Chr {chrom_i}, pos {pos_start}-{pos_end}, {len(indices)} SNPs")
        print(f"[INFO] Selected SNP indices: {indices}")

        Xr = self.X.iloc[:, indices] if isinstance(self.X, pd.DataFrame) else self.X[:, indices]
        return Xr, (chrom_i, pos_start, pos_end), indices

    def gen_rescaling_GxE(self, X, ncausal, v_s=0.05, return_snp_info=True, global_indices=None):
        """
        Generate rescaling GxE effects with consistent RNG usage.
        """
        # If v_s is 0, return zero matrix
        if v_s <= 0:
            S = np.zeros((X.shape[0], self.P))
            snp_info = {"common_indices_rescaling": [], "eta": self.eta} if return_snp_info else None
            return S, snp_info

        print(f"[INFO] Generating rescaling GxE effects with v_s = {v_s}")
        print(f"[INFO] Rescaling factor η = {self.eta}")

        S_region = X.shape[1]  # Number of SNPs in region

        # Check if we have enough SNPs
        ncausal = min(ncausal, S_region)
        if ncausal == 0:
            print("[WARN] No causal SNPs available")
            S = np.zeros((X.shape[0], self.P))
            snp_info = {"common_indices_rescaling": [], "eta": self.eta} if return_snp_info else None
            return S, snp_info

        print(f"[INFO] Using {ncausal} causal SNPs out of {S_region} available")

        causal_mask = self.selectRnd(ncausal, S_region, rng=self.rng_snp_selection)
        causal_indices = np.where(causal_mask)[0]

        # Extract genotypes for causal SNPs
        if isinstance(X, pd.DataFrame):
            G = X.iloc[:, causal_indices].values
        else:
            G = X[:, causal_indices]

        b = self.gen_binormal(ncausal, strategy="iid_binary", rng=self.rng_effect_sizes)

        # Generate rescaling effects: S = G · b · [1, η]
        s_base = G @ b  # Base genetic effect (N,)

        # Create effect matrix with rescaling pattern
        if self.P == 1:
            # Single trait case
            S_raw = s_base.reshape(-1, 1)
        elif self.P == 2:
            # Two trait case with rescaling
            s1 = s_base  # First trait gets base effect
            s2 = s_base * self.eta  # Second trait gets rescaled effect
            S_raw = np.column_stack([s1, s2])
        else:
            reference_trait = self.reference_trait  

            S_raw = np.zeros((len(s_base), self.P))
            S_raw[:, reference_trait] = s_base 
            for p in range(self.P):
                if p != reference_trait:
                    S_raw[:, p] = s_base * self.eta  # All other traits rescaled by η

        print(f"[INFO] Raw effect matrix S_raw shape: {S_raw.shape}")
        print(f"[INFO] Raw effects var per trait: {np.var(S_raw, axis=0)}")
        print(f"[INFO] Raw effects var[vec(S_raw)]: {np.var(S_raw.flatten()):.6f}")

        # Scale to target variance 
        current_var_vec = np.var(S_raw.flatten())

        if current_var_vec > 0:
            scaling_factor = np.sqrt(v_s / current_var_vec)
            S_region = S_raw * scaling_factor
        else:
            S_region = S_raw
            scaling_factor = 1.0

        # Verify final variances
        final_var_vec = np.var(S_region.flatten())
        final_var_per_trait = np.var(S_region, axis=0)

        print(f"[INFO] Final effects var[vec(S)]: {final_var_vec:.6f} (target: {v_s:.6f})")
        print(f"[INFO] Final effects var per trait: {final_var_per_trait}")

        # Package SNP info
        snp_info = None
        if return_snp_info:
            if global_indices is not None:
                global_causal_indices = global_indices[causal_indices].tolist()
            else:
                global_causal_indices = causal_indices.tolist()

            snp_info = {
                # Core rescaling model info
                "model_type": "rescaling_GxE",
                "common_indices_rescaling": global_causal_indices,
                "eta": self.eta,
                "ncausal": ncausal,

                # Effect details
                "effect_sizes": b.tolist(),
                "scaling_factor": scaling_factor,
                "causal_indices_local": causal_indices.tolist(),
                "causal_indices_global": global_causal_indices,

                # Variance accounting
                "target_variance": v_s,
                "achieved_variance": final_var_vec,
                "variance_per_trait": final_var_per_trait.tolist(),

                # Model verification
                "rescaling_pattern": [1.0] + [self.eta ** p for p in range(1, self.P)],

                "snp_indices_global": global_causal_indices,
            }
            print(f"[INFO] Selected causal SNPs (global indices): {global_causal_indices}")

        return S_region, snp_info

    def gen_binormal(self, size, std=0.1, strategy="iid_binary", rng=None):
        """
        Generate effect sizes with specified RNG.

        Parameters:
        size: Number of effect sizes to generate
        std: Standard deviation for noise (only used in some strategies)
        strategy: Strategy for generating effect sizes
        rng: Random number generator to use (defaults to self.rng_effect_sizes)

        Returns:
        effects: Array of effect sizes
        """
        if rng is None:
            rng = self.rng_effect_sizes 
            
        if strategy == "iid_binary":
            # Generate clean {-1, +1} effects
            signs = 2 * (rng.random(size) > 0.5) - 1
            return signs.astype(float)

        elif strategy == "binary_with_noise":
            # Binary effects with small noise
            signs = 2 * (rng.random(size) > 0.5) - 1
            noise = rng.normal(0.0, std, size)
            return signs + noise

        elif strategy == "normal":
            # Normal distribution effects
            effects = rng.normal(0.0, 1.0, size)
            return effects

        else:
            # Default to original method for backward compatibility
            signs = 2 * (rng.random(size) > 0.5) - 1
            noise = rng.normal(0.0, std, size)
            return signs + noise
    
 
    def _sample_ld_constrained_snps(self, X_context, sc, ld_threshold, seed):
        """Return sc SNP indices from X_context with all pairwise r² < ld_threshold,
        or None if the LD-pruned pool is too small."""
        rng = np.random.default_rng(seed)
        n = X_context.shape[1]
        if sc == 1:
            return [int(rng.integers(n))]

        # Pairwise r²; NaN (from constant cols) treated as fully redundant.
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.corrcoef(X_context, rowvar=False) ** 2
        r2 = np.where(np.isfinite(r2), r2, 1.0)
        np.fill_diagonal(r2, 1.0)

        # PLINK-style greedy prune in random order, then sample sc from the pool
        keep = np.zeros(n, dtype=bool)
        for idx in rng.permutation(n):
            if not keep.any() or np.all(r2[idx, keep] < ld_threshold):
                keep[idx] = True
        pool = np.flatnonzero(keep)
        if pool.size < sc:
            return None
        return rng.choice(pool, size=sc, replace=False).tolist()

    def gen_heterogeneous_effects(self, X, v_s=0.05, corr_bounds_idx=5,
                                return_snp_info=True, global_indices=None,
                                ncausal=2, ld_threshold=0.1,
                                max_attempts=1000, use_per_trait_scaling=True):
        """General-GxC heterogeneous effects.

        Sample ncausal LD-pruned causal SNPs per context (all pairwise r² <
        ld_threshold within each context), draw signs b_c ~ {-1,+1}, and reject
        (SNP, sign) draws until corr(S:,i, S:,j) ∈ (rho_min, rho_max) for all i<j.

        Raises ValueError if the LD constraint cannot be satisfied (either upfront
        or after exhausting max_attempts).
        """
        CORRELATION_BOUNDS = [(-1.0,-0.8),(-0.8,-0.6),(-0.6,-0.4),(-0.4,-0.2),(-0.2, 0.0),
                            ( 0.0, 0.2),( 0.2, 0.4),( 0.4, 0.6),( 0.6, 0.8),( 0.8, 1.0)]

        # Degenerate case.
        if v_s <= 0:
            S = np.zeros((X.shape[0], self.P))
            info = ({"heterogeneity_context_indices": [[] for _ in range(self.P)]}
                    if return_snp_info else None)
            return S, info

        # Normalize input.
        if isinstance(X, pd.DataFrame):
            X_values, column_names = X.values, X.columns.tolist()
        else:
            X_values, column_names = X, None
        n_samples, n_snps = X_values.shape

        if self.P >= 2:
            rho_min, rho_max = CORRELATION_BOUNDS[corr_bounds_idx]
        else:
            rho_min = rho_max = None

        print(f"[INFO] General-GxC: P={self.P}, ncausal/context={ncausal}, "
            f"total={self.P * ncausal}, r²<{ld_threshold}, per-trait={use_per_trait_scaling}")
        if self.P >= 2:
            print(f"[INFO] Correlation bound: ({rho_min:.2f}, {rho_max:.2f})")

        if self.P * ncausal > n_snps:
            raise ValueError(f"Need ≥{self.P * ncausal} SNPs; region has {n_snps}.")
        
        if self._sample_ld_constrained_snps(X_values, ncausal, ld_threshold, seed=0) is None:
            raise ValueError(
                f"LD constraint r²<{ld_threshold} infeasible: cannot find "
                f"{ncausal} LD-pruned SNPs in this region ({n_snps} SNPs). "
                f"Loosen ld_threshold, enlarge the region, or reduce ncausal."
            )

        # Rejection loop over (SNP choice, signs); count LD vs correlation failures.
        best_config, best_violation = None, float("inf")
        ld_failures = 0

        for attempt in range(max_attempts):
            context_snps, context_effects, pgs_cols = {}, {}, []
            ok = True
            used = set()
            for c in range(self.P):
                avail = [i for i in range(n_snps) if i not in used]
                if len(avail) < ncausal:
                    avail = list(range(n_snps))
                picks = self._sample_ld_constrained_snps(
                    X_values[:, avail], ncausal, ld_threshold,
                    seed=attempt * self.P + c,
                )
                if picks is None:
                    ld_failures += 1
                    ok = False
                    break
                picks = [avail[i] for i in picks]
                b = 2 * (self.rng_effect_sizes.random(ncausal) > 0.5) - 1
                context_snps[c] = picks
                context_effects[c] = b
                pgs_cols.append(X_values[:, picks] @ b)
                used.update(picks)
            if not ok:
                continue

            if self.P < 2:
                best_config = dict(context_snps=context_snps,
                                context_effects=context_effects,
                                correlation=None, pairwise_corrs=None,
                                attempt=attempt + 1)
                break

            pgs = np.column_stack(pgs_cols)
            corr = np.corrcoef(pgs, rowvar=False)
            pairwise = corr[np.triu_indices(self.P, k=1)]
            below = np.maximum(rho_min - pairwise, 0)
            above = np.maximum(pairwise - rho_max, 0)
            violation = float((below + above).max())

            if violation == 0:
                best_config = dict(context_snps=context_snps,
                                context_effects=context_effects,
                                correlation=corr, pairwise_corrs=pairwise,
                                attempt=attempt + 1)
                print(f"[SUCCESS] Valid config after {attempt+1} attempts "
                    f"(LD failures: {ld_failures}); "
                    f"pairwise ∈ [{pairwise.min():.3f}, {pairwise.max():.3f}]")
                break

            if violation < best_violation:
                best_violation = violation
                best_config = dict(context_snps=context_snps,
                                context_effects=context_effects,
                                correlation=corr, pairwise_corrs=pairwise,
                                attempt=attempt + 1)
        else:
            # Loop exhausted without an exact-fit config. Distinguish causes.
            if best_config is None:
                # Every attempt failed at the LD stage (no signs ever drawn).
                raise ValueError(
                    f"LD constraint r²<{ld_threshold} failed on every one of "
                    f"{max_attempts} attempts ({ld_failures} LD failures). "
                    f"This region is too LD-dense for {ncausal} causal SNPs "
                    f"per context. Loosen ld_threshold or enlarge the region."
                )
            # LD succeeded sometimes; correlation constraint is what couldn't be met.
            print(f"[WARN] Using best-effort config (correlation violation="
                f"{best_violation:.4f}; LD failures during search: "
                f"{ld_failures}/{max_attempts})")

        # Build effect matrix S[:,c] = G_c @ b_c.
        S_raw = np.column_stack([
            X_values[:, best_config["context_snps"][c]] @ best_config["context_effects"][c]
            for c in range(self.P)
        ])

        # Scale to target variance.
        if use_per_trait_scaling:
            S_scaled = np.zeros_like(S_raw)
            for c in range(self.P):
                v = S_raw[:, c].var()
                S_scaled[:, c] = S_raw[:, c] * np.sqrt(v_s / v) if v > 0 else S_raw[:, c]
            print(f"[INFO] Per-trait scaling: each trait variance = {v_s:.4f}")
        else:
            v = S_raw.flatten().var()
            S_scaled = S_raw * np.sqrt(v_s / v) if v > 0 else S_raw.copy()
            print(f"[INFO] Vectorized scaling: var[vec(S)] = {v_s:.4f}")

        print(f"[INFO] Per-trait variances: {[float(S_scaled[:, c].var()) for c in range(self.P)]}")
        if self.P >= 2:
            print(f"[INFO] Final corr(S:,0, S:,1): "
                f"{np.corrcoef(S_scaled[:, 0], S_scaled[:, 1])[0, 1]:.4f}")

        # Build snp_info (schema preserved).
        snp_info = None
        if return_snp_info:
            snp_info = self._create_heterogeneity_snp_info(
                best_config["context_snps"], best_config["context_effects"],
                best_config.get("correlation"), best_config.get("pairwise_corrs"),
                (rho_min, rho_max) if self.P >= 2 else None,
                best_config["attempt"], global_indices, v_s, S_scaled,
                column_names, ncausal, use_per_trait_scaling, ld_threshold,
            )
        return S_scaled, snp_info

    def _create_heterogeneity_snp_info(self, context_snps, context_effects,
                                    correlation_matrix, pairwise_corrs,
                                    corr_bounds, attempts, global_indices,
                                    v_s, S_scaled, column_names, ncausal,
                                    use_per_trait_scaling, ld_threshold):
        """Build snp_info dict for heterogeneous-effects simulation. Schema unchanged."""

        # Local → global index mapping.
        if global_indices is not None:
            global_context_indices = {c: [global_indices[i] for i in idx]
                                    for c, idx in context_snps.items()}
        else:
            global_context_indices = context_snps
        het_global = [global_context_indices.get(i, []) for i in range(self.P)]

        # Correlation satisfaction across all pairs.
        correlation_satisfied = None
        if pairwise_corrs is not None and corr_bounds is not None:
            rho_min, rho_max = corr_bounds
            correlation_satisfied = bool(np.all((rho_min < pairwise_corrs)
                                                & (pairwise_corrs < rho_max)))

        # Pairwise summary stats.
        pairwise_summary = None
        if pairwise_corrs is not None and len(pairwise_corrs) > 0:
            pairwise_summary = {
                "min":      float(np.min(pairwise_corrs)),
                "max":      float(np.max(pairwise_corrs)),
                "mean":     float(np.mean(pairwise_corrs)),
                "median":   float(np.median(pairwise_corrs)),
                "std":      float(np.std(pairwise_corrs)),
                "mean_abs": float(np.mean(np.abs(pairwise_corrs))),
                "n_pairs_in_range": (int(np.sum((corr_bounds[0] < pairwise_corrs)
                                                & (pairwise_corrs < corr_bounds[1])))
                                    if corr_bounds is not None else None),
                "n_pairs_total": int(len(pairwise_corrs)),
            }

        snp_info = {
            "model_type": "general_gxc",
            "total_causal_variants": self.P * ncausal,
            "ld_threshold": float(ld_threshold),
            "per_trait_scaling": use_per_trait_scaling,

            "context_snp_indices": global_context_indices,
            "context_effect_sizes": {k: v.tolist() for k, v in context_effects.items()},
            "n_traits": self.P,

            "heterogeneity_context_indices": het_global,
            "ncausal": ncausal,
            "local_context_indices": dict(context_snps),
            "heterogeneity_context_indices_local": [context_snps.get(i, []) for i in range(self.P)],

            "achieved_correlation_matrix": (correlation_matrix.tolist()
                                            if correlation_matrix is not None else None),
            "achieved_pairwise_correlations": (pairwise_corrs.tolist()
                                            if pairwise_corrs is not None else None),
            "pairwise_correlation_summary": pairwise_summary,
            "correlation_bounds": corr_bounds,
            "correlation_satisfied_all_pairs": correlation_satisfied,
            "achieved_correlation": (float(correlation_matrix[0, 1])
                                    if correlation_matrix is not None else None),

            "target_variance": v_s,
            "final_variance_vec": float(np.var(S_scaled.flatten())),
            "per_trait_variances": [float(np.var(S_scaled[:, i])) for i in range(self.P)],

            "attempts_used": attempts,
            "global_context_indices": global_context_indices,
        }

        if column_names is not None:
            snp_info["context_snp_names"] = {c: [column_names[i] for i in context_snps[c]]
                                            for c in context_snps}
        if self.P >= 2:
            snp_info["context1_indices"] = het_global[0]
            snp_info["context2_indices"] = het_global[1]
        return snp_info

    def gen_background_effects(self, v_bg, omega, use_XX=True):
        """
        Generate background genetic effects

        G = G^(s) + G^(i)
        G^(s) ~ MN(0, R, a_G a_G^T)  where a_G = √τ_{G,s}, τ_{G,s} ~ Uniform(0,1)
        G^(i) ~ MN(0, R, diag(c_G^2))  where c_G = √τ_{G,i}, τ_{G,i} ~ Uniform(0,1)

        Variance allocation 
        var[vec(G^(s))] = ω * v_bg
        var[vec(G^(i))] = (1-ω) * v_bg

        Parameters:
        v_bg: Total variance explained by background effects
        omega: Fraction of shared signal
        use_XX: Whether to use kinship matrix as R

        Returns:
        G_shared: Shared background component G^(s) 
        G_indep: Independent background component G^(i)
        """
        # If v_bg is 0, return zero matrices
        if v_bg <= 0:
            G_shared = np.zeros((self.N, self.P))
            G_indep = np.zeros((self.N, self.P))
            return G_shared, G_indep

        # Compute or use covariance matrix R
        if use_XX:
            if self.XX is None:
                raise ValueError("Kinship matrix (XX) not available")
            R = self.XX
        else:
            # Compute genetic relatedness matrix from random subset of SNPs
            S = self.X.shape[1]
            ncausal = int(0.05 * S)  # Use 5% of SNPs
            causal_mask = self.selectRnd(ncausal, S)
            X_causal = self.X.iloc[:, causal_mask] if isinstance(self.X, pd.DataFrame) else self.X[:, causal_mask]

            # Standardize SNP matrix
            X_std = (X_causal - X_causal.mean(axis=0)) / X_causal.std(axis=0)
            R = (X_std @ X_std.T) / ncausal

        # Stabilize covariance matrix R
        R_stable = R + 1e-6 * np.eye(self.N)

        # Cholesky decomposition of R
        try:
            L = np.linalg.cholesky(R_stable)
        except np.linalg.LinAlgError:
            print("[WARN] Covariance matrix R not PD, using SVD fallback.")
            U, s, _ = np.linalg.svd(R_stable)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-10)))

        # Sample structural parameters (scalars)
        tau_G_s = self.rng_background.uniform(0, 1)  # τ_{G,s} ~ Uniform(0,1)
        tau_G_i = self.rng_background.uniform(0, 1)  # τ_{G,i} ~ Uniform(0,1)

        a_G = np.sqrt(tau_G_s)  # a_G = √τ_{G,s}
        c_G = np.sqrt(tau_G_i)  # c_G = √τ_{G,i}
        
        # Target variances 
        target_var_shared = omega * v_bg        # var[vec(G^(s))] = ω * v_bg
        target_var_indep = (1 - omega) * v_bg   # var[vec(G^(i))] = (1-ω) * v_bg

        # Generate shared component: G^(s) ~ MN(0, R, a_G a_G^T)
        if target_var_shared > 0:
            z_shared = self.rng_background.standard_normal(self.N)
            spatial_shared = L @ z_shared  # Apply spatial correlation R
            G_shared_raw = np.outer(spatial_shared, a_G * np.ones(self.P))  # Apply trait correlation a_G a_G^T

            # Scale to achieve target variance 
            current_var_shared = np.var(G_shared_raw.flatten())  # var[vec(G^(s))]
            if current_var_shared > 0:
                scale_shared = np.sqrt(target_var_shared / current_var_shared)
                G_shared = G_shared_raw * scale_shared
            else:
                G_shared = G_shared_raw
        else:
            G_shared = np.zeros((self.N, self.P))

        # Generate independent component: G^(i) ~ MN(0, R, diag(c_G^2))
        if target_var_indep > 0:
            Z_indep = self.rng_background.standard_normal((self.N, self.P))
            # Apply scalar c_G uniformly to all traits (creates diagonal covariance with equal variances)
            G_indep_raw = L @ (Z_indep * c_G)  # c_G broadcasts as same value to all traits

            # Scale to achieve target variance
            current_var_indep = np.var(G_indep_raw.flatten())  # var[vec(G^(i))]
            if current_var_indep > 0:
                scale_indep = np.sqrt(target_var_indep / current_var_indep)
                G_indep = G_indep_raw * scale_indep
            else:
                G_indep = G_indep_raw
        else:
            G_indep = np.zeros((self.N, self.P))

        # Verify final variances
        achieved_var_shared = np.var(G_shared.flatten())
        achieved_var_indep = np.var(G_indep.flatten())
        total_var_achieved = achieved_var_shared + achieved_var_indep

        print(f"[INFO] Achieved background variances:")
        print(f"  var[vec(G^(s))]: {achieved_var_shared:.4f} (target: {target_var_shared:.4f})")
        print(f"  var[vec(G^(i))]: {achieved_var_indep:.4f} (target: {target_var_indep:.4f})")
        print(f"  Total: {total_var_achieved:.4f} (target: {v_bg:.4f})")

        return G_shared, G_indep

    def gen_hidden_effects(self, v_s, v_bg, omega, gamma, n_hidden=10):
        """
        Generate hidden confounding effects 

        H = H^(s) + H^(i)
        H^(s) ~ MN(0, MM^T, a_H a_H^T)  where a_H = √τ_{H,s}, τ_{H,s} ~ Uniform(0,1)
        H^(i) ~ MN(0, MM^T, diag(c_H^2))  where c_H = √τ_{H,i}, τ_{H,i} ~ Uniform(0,1)

        Variance allocation
        var[vec(H^(s))] = ω * γ * (1 - v_bg - v_s)
        var[vec(H^(i))] = (1-ω) * γ * (1 - v_bg - v_s)

        Parameters:
        v_s: Variance explained by regional effects
        v_bg: Variance explained by background effects  
        omega: Fraction of shared signal 
        gamma: Fraction of residual variance that is non-iid 
        n_hidden: Number of hidden confounders

        Returns:
        H_shared: Shared hidden component H^(s)
        H_indep: Independent hidden component H^(i)
        """
        # Calculate residual variance
        v_residual = 1.0 - v_bg - v_s

        # Target variances 
        target_var_shared = omega * gamma * v_residual      # var[vec(H^(s))]
        target_var_indep = (1 - omega) * gamma * v_residual # var[vec(H^(i))]

        print(f"[INFO] Hidden effects variance allocation:")
        print(f"  Residual variance: {v_residual:.4f}")
        print(f"  Target var[vec(H^(s))]: {target_var_shared:.4f}")
        print(f"  Target var[vec(H^(i))]: {target_var_indep:.4f}")

        # If no hidden variance, return zeros
        if target_var_shared <= 0 and target_var_indep <= 0:
            H_shared = np.zeros((self.N, self.P))
            H_indep = np.zeros((self.N, self.P))
            return H_shared, H_indep

        # Generate M ~ N(0,1) for covariance structure 
        M = self.rng_hidden.standard_normal((self.N, n_hidden))
        MM_T = M @ M.T

        # Stabilize MM^T matrix
        MM_T_stable = MM_T + 1e-6 * np.eye(self.N)

        # Cholesky decomposition of MM^T
        try:
            L = np.linalg.cholesky(MM_T_stable)
        except np.linalg.LinAlgError:
            print("[WARN] MM^T not positive definite; using SVD fallback")
            U, s, _ = np.linalg.svd(MM_T_stable)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-10)))

        # Sample structural parameters (scalars)
        tau_H_s = self.rng_hidden.uniform(0, 1)  # τ_{H,s} ~ Uniform(0,1)
        tau_H_i = self.rng_hidden.uniform(0, 1)  # τ_{H,i} ~ Uniform(0,1)

        a_H = np.sqrt(tau_H_s)  # a_H = √τ_{H,s}
        c_H = np.sqrt(tau_H_i)  # c_H = √τ_{H,i}

        # Generate shared component: H^(s) ~ MN(0, MM^T, a_H a_H^T)
        if target_var_shared > 0:
            z_shared = self.rng_hidden.standard_normal(self.N)
            spatial_shared = L @ z_shared  # Apply spatial correlation MM^T
            H_shared_raw = np.outer(spatial_shared, a_H * np.ones(self.P))  # Apply trait correlation a_H a_H^T

            # Scale to achieve target variance
            current_var_shared = np.var(H_shared_raw.flatten())  # var[vec(H^(s))]
            if current_var_shared > 0:
                scale_shared = np.sqrt(target_var_shared / current_var_shared)
                H_shared = H_shared_raw * scale_shared
            else:
                H_shared = H_shared_raw
        else:
            H_shared = np.zeros((self.N, self.P))

        # Generate independent component: H^(i) ~ MN(0, MM^T, diag(c_H^2))
        if target_var_indep > 0:
            Z_indep = self.rng_hidden.standard_normal((self.N, self.P))
            # Apply scalar c_H uniformly to all traits (creates diagonal covariance with equal variances)
            H_indep_raw = L @ (Z_indep * c_H)  # c_H broadcasts as same value to all traits

            # Scale to achieve target variance 
            current_var_indep = np.var(H_indep_raw.flatten())  # var[vec(H^(i))]
            if current_var_indep > 0:
                scale_indep = np.sqrt(target_var_indep / current_var_indep)
                H_indep = H_indep_raw * scale_indep
            else:
                H_indep = H_indep_raw
        else:
            H_indep = np.zeros((self.N, self.P))

        # Verify final variances
        achieved_var_shared = np.var(H_shared.flatten())
        achieved_var_indep = np.var(H_indep.flatten())
        total_var_achieved = achieved_var_shared + achieved_var_indep

        print(f"[INFO] Achieved hidden variances:")
        print(f"  var[vec(H^(s))]: {achieved_var_shared:.4f} (target: {target_var_shared:.4f})")
        print(f"  var[vec(H^(i))]: {achieved_var_indep:.4f} (target: {target_var_indep:.4f})")
        print(f"  Total: {total_var_achieved:.4f}")

        return H_shared, H_indep

    def gen_noise_iid(self, v_s, v_bg, gamma):
        """
        Generate independent residual noise

        Variance allocation:
        var[vec(Ψ)] = (1-γ) * (1 - v_bg - v_s)

        Parameters:
        v_s: Variance explained by regional effects
        v_bg: Variance explained by background effects
        gamma: Fraction of residual variance that is non-iid

        Returns:
        Psi_indep: Independent noise component
        """
        # Calculate target variance 
        v_residual = 1.0 - v_bg - v_s
        target_var_noise = (1 - gamma) * v_residual  # var[vec(Ψ)]

        print(f"[INFO] Noise variance allocation:")
        print(f"  Target var[vec(Ψ)]: {target_var_noise:.4f}")

        # If no noise variance, return zeros
        if target_var_noise <= 0:
            return np.zeros((self.N, self.P))

        # Generate independent noise
        Psi_indep_raw = self.rng_noise.standard_normal((self.N, self.P))

        # Scale to achieve target variance
        current_var = np.var(Psi_indep_raw.flatten())  # var[vec(Ψ)]
        if current_var > 0:
            scale = np.sqrt(target_var_noise / current_var)
            Psi_indep = Psi_indep_raw * scale
        else:
            Psi_indep = Psi_indep_raw

        # Verify final variance
        achieved_var = np.var(Psi_indep.flatten())
        print(f"[INFO] Achieved noise variance: var[vec(Ψ)] = {achieved_var:.4f} (target: {target_var_noise:.4f})")

        return Psi_indep

    def genPheno(
        self, 
        Xr,
        # Region effects parameters
        v_s=None,                      # Total variance explained by region
        ncausal=1,                   # Number of causal SNPs in region
        use_heterogeneity=False,      # Whether to use heterogeneity model
        corr_bounds=0,                # Zero-indexed correlation bounds for heterogeneity

        # Heterogeneity-specific parameters
        ld_threshold=0.05,             # LD threshold for heterogeneity model
        max_attempts=100,             # Max attempts for heterogeneity sampling

        # Global parameters
        v_bg=None,                     # Background genetic variance 
        omega=None,                    # Fraction of shared signal 
        gamma=None,                     # Fraction of residual variance from hidden factors  

        use_XX=True,                  # Use kinship matrix for background
        n_hidden=10,                  # Number of hidden factors
        return_snp_info=True,         # Whether to return SNP info
        global_indices=None,          # Global indices for SNPs
    ):
        """
        Generate phenotype incorporating omega-gamma reasoning with proper component balancing.

        Supports both single-model and mixed-model simulation:
        - use_heterogeneity=False: Pure rescaling model 
        - use_heterogeneity=True: General-GxC heterogeneity effects with LD constraints

        Parameters:
            Xr: Region-specific genotype matrix
            v_s: Variance explained by region effects 
            ncausal: Number of causal SNPs in region (for single model) or total budget (for mixed)
            use_heterogeneity: Whether to include heterogeneity effects instead of rescaling
            corr_bounds: Zero-indexed correlation bounds for heterogeneity model (0-9)
            rescaling_fraction: Fraction of v_s allocated to rescaling effects (when use_heterogeneity=True)
            ncausal: SNPs for rescaling (defaults to ncausal//2 when mixed)
            ncausal: SNPs per context for heterogeneity
            ld_threshold: LD threshold for heterogeneity model (r² < threshold)
            max_attempts: Maximum attempts for heterogeneity sampling
            v_bg: Background genetic variance
            omega: Fraction of shared signal across contexts
            gamma: Fraction of residual variance from hidden factors
            use_XX: Whether to use kinship matrix for background
            n_hidden: Number of hidden factors
            return_snp_info: Whether to return SNP info
            global_indices: Global indices for SNPs

        Returns:
            Y: Phenotype DataFrame
            info: Dictionary with variance components and SNP information
        """

        # Apply dataset-specific defaults for None parameters
        params = self.get_default_params()

        if v_s is None:
            v_s = params['v_s']
        if v_bg is None:
            v_bg = params['v_bg']
        if omega is None:
            omega = params['omega']
        if gamma is None:
            gamma = params['gamma']

        print(f"[INFO] Using parameters for {self.dset}: v_s={v_s}, v_bg={v_bg}, omega={omega}, gamma={gamma}")

        if use_heterogeneity:
            print(f'[INFO] Generating phenotype with heterogeneity effects')
            print(f'[INFO] Correlation bounds: {corr_bounds}, LD threshold: {ld_threshold}')
            print(f'[INFO] SNPs per context: {ncausal}')
        else:
            print('[INFO] Generating phenotype with rescaling effects.')

        print('SNP index: ', global_indices)

        # Handle SNP count limits
        num_snps_in_Xr = Xr.shape[1]
        if ncausal > num_snps_in_Xr:
            print(f"[WARN] ncausal ({ncausal}) > available SNPs ({num_snps_in_Xr}); reducing.")
            ncausal = num_snps_in_Xr

        # Generate region effects based on model type
        if use_heterogeneity:
            # General-GxC heterogeneity model with LD constraints
            S_region, snp_info = self.gen_heterogeneous_effects(
                X=Xr,
                v_s=v_s,
                corr_bounds_idx=corr_bounds,  
                return_snp_info=return_snp_info,
                global_indices=global_indices,
                ncausal=ncausal,  
                ld_threshold=ld_threshold,
                max_attempts=max_attempts
            )
        else:
            # Rescaling-only effects
            S_region, snp_info = self.gen_rescaling_GxE(
                X=Xr,
                ncausal=ncausal,
                v_s=v_s,
                return_snp_info=return_snp_info,
                global_indices=global_indices
            )
        # Background genetic effects 
        G_shared, G_indep = self.gen_background_effects(
            v_bg=v_bg, 
            omega=omega, 
            use_XX=use_XX
        )

        # Calculate residual variance (1 - v_s - v_bg)
        v_residual = max(0, 1.0 - v_s - v_bg)

        # Hidden confounders - updated function signature
        H_shared, H_indep = self.gen_hidden_effects(
            v_s=v_s,
            v_bg=v_bg,
            omega=omega, 
            gamma=gamma, 
            n_hidden=n_hidden
        )

        # Residual noise - updated function signature
        Psi_indep = self.gen_noise_iid(
            v_s=v_s,
            v_bg=v_bg,
            gamma=gamma
        )

        # Apply consistent balancing strategy
        Y = S_region + G_shared + G_indep + H_shared + H_indep + Psi_indep

        if Y.ndim > 2:
            Y = Y.squeeze()  # Remove singleton dimensions
            print(f"[FIX] Squeezed Y to {Y.shape}")

        # Convert to DataFrame
        trait_cols = [f"trait_{i+1}" for i in range(self.P)]

        # Ensure DataFrame has correct shape
        Y_df = pd.DataFrame(Y, columns=trait_cols)
        assert Y_df.shape[1] == self.P, f"Expected {self.P} traits, got {Y_df.shape[1]}"


        if isinstance(Xr, pd.DataFrame):
            # If Xr is a DataFrame, use its index
            Y_df = pd.DataFrame(Y, index=Xr.index, columns=trait_cols)
            print("[INFO] Created simulated phenotype with same index structure as genotype data")
            print(f"[DEBUG] Index type: {type(Y_df.index)}")
            print(f"[DEBUG] First 5 indices: {list(Y_df.index)[:5]}")
        else:
            if hasattr(self.X, 'index'):
                Y_df = pd.DataFrame(Y, index=self.X.index[:len(Y)], columns=trait_cols)
                print("[INFO] Created simulated phenotype with index from self.X")
            else:
                Y_df = pd.DataFrame(Y, columns=trait_cols)
                print("[WARNING] Could not preserve original index structure")

        print("\n" + "="*60)
        print("VARIANCE BREAKDOWN BEFORE STANDARDIZATION")
        print("="*60)

        # Individual component variances per trait
        print("\nComponent Variances Per Trait (Before Standardization):")
        components = [
            ("Region (S)", S_region),
            ("Shared Background (G_s)", G_shared),
            ("Indep Background (G_i)", G_indep),
            ("Shared Hidden (H_s)", H_shared),
            ("Indep Hidden (H_i)", H_indep),
            ("Indep Noise (Ψ)", Psi_indep)
        ]

        component_vars_per_trait = {}
        for label, component in components:
            var_by_trait = np.var(component, axis=0)
            component_vars_per_trait[label] = var_by_trait
            formatted_vars = ", ".join([f"{v:.6f}" for v in var_by_trait])
            print(f"  {label:25s}: [{formatted_vars}]")

        # Total phenotype variance per trait (before standardization)
        Y_var_per_trait = np.var(Y, axis=0)
        print(f"\n  {'Total Phenotype':25s}: [{', '.join([f'{v:.6f}' for v in Y_var_per_trait])}]")

        # Vectorized variances
        print(f"\nVectorized Variances var[vec(·)] (Before Standardization):")
        for label, component in components:
            vec_var = np.var(component.flatten())
            print(f"  var[vec({label.split('(')[0].strip()})] = {vec_var:.6f}")

        total_vec_var = np.var(Y.flatten())
        print(f"  var[vec(Total Phenotype)] = {total_vec_var:.6f}")

        # Variance fractions (what fraction each component contributes)
        print(f"\nVariance Fractions (Before Standardization):")
        for label, component in components:
            vec_var = np.var(component.flatten())
            fraction = vec_var / total_vec_var if total_vec_var > 0 else 0
            print(f"  {label.split('(')[0].strip():15s}: {fraction:.4f} ({fraction*100:.1f}%)")

        # Calculate achieved hidden and noise variances
        v_hidden_achieved = np.var((H_shared + H_indep).flatten())
        v_noise_achieved = np.var(Psi_indep.flatten())

        print(f"  Achieved var[vec(H)] = {v_hidden_achieved:.6f}")
        print(f"  Achieved var[vec(Ψ)] = {v_noise_achieved:.6f}")

        total_achieved = np.var(S_region.flatten()) + np.var((G_shared + G_indep).flatten()) + v_hidden_achieved + v_noise_achieved
        print(f"  Total achieved variance = {total_achieved:.6f}")
        print(f"  Total phenotype variance = {total_vec_var:.6f}")
        print(f"  Difference = {abs(total_vec_var - total_achieved):.6f}")

        # Compile comprehensive information dictionary
        info = {
            # Effect matrices
            'S_region': S_region,
            'G_shared': G_shared,
            'G_indep': G_indep,
            'H_shared': H_shared,
            'H_indep': H_indep,
            'Psi_indep': Psi_indep,

            # Variance parameters
            'v_s': v_s,
            'v_bg': v_bg,
            'v_residual': v_residual,
            'omega': omega,
            'gamma': gamma,

            # Model-specific parameters
            'use_heterogeneity': use_heterogeneity,
            'corr_bounds': corr_bounds,
            'ld_threshold': ld_threshold,
            'max_attempts': max_attempts,

            # Variance tracking (before standardization)
            'component_variances_per_trait': component_vars_per_trait,
            'total_variance_per_trait': Y_var_per_trait.tolist(),
            'total_vectorized_variance': total_vec_var,
        }

        if use_heterogeneity:
            # Add heterogeneity model specific info
            info.update({
                'ncausal': ncausal,
                'n_traits': self.P,
                'heterogeneity_context_indices': snp_info.get('heterogeneity_context_indices', [None] * self.P) if snp_info else [None] * self.P,
                'total_causal_snps': self.P * ncausal,
                'global_context_indices': snp_info.get('global_context_indices', {}) if snp_info else {},
                'local_context_indices': snp_info.get('local_context_indices', {}) if snp_info else {},

                # Correlation diagnostics — needed for realized-correlation plotting
                'achieved_correlation_matrix': snp_info.get('achieved_correlation_matrix') if snp_info else None,
                'achieved_pairwise_correlations': snp_info.get('achieved_pairwise_correlations') if snp_info else None,
                'pairwise_correlation_summary': snp_info.get('pairwise_correlation_summary') if snp_info else None,
                'correlation_satisfied_all_pairs': snp_info.get('correlation_satisfied_all_pairs') if snp_info else None,
                'correlation_bounds_range': snp_info.get('correlation_bounds') if snp_info else None,
                'achieved_correlation_01': snp_info.get('achieved_correlation') if snp_info else None,  # backwards compat
                'attempts_used': snp_info.get('attempts_used') if snp_info else None,
                'ld_threshold': snp_info.get('ld_threshold', ld_threshold) if snp_info else ld_threshold,
            })

            if snp_info:
                heterogeneity_indices = snp_info.get('heterogeneity_context_indices', [None] * self.P)
                for trait_i in range(self.P):
                    context_key = f'context{trait_i}_indices'
                    info[context_key] = heterogeneity_indices[trait_i] if trait_i < len(heterogeneity_indices) else []
        else:
            # Add rescaling model specific info 
            info.update({
                'rescaling_common_indices': snp_info.get('common_indices_rescaling', []) if snp_info else [],
                'eta': self.eta,
                'ncausal': ncausal,
                'n_traits': self.P,  # Add number of traits for StoreResults
            })

        return Y_df, info