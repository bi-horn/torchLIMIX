import numpy as np
import pandas as pd
import torch
from torchlimix.utils._genotype_preparation import prepare_kinship_pipeline


class VarDecSimulator:
    """
    Variance Decomposition Simulator
    
    Y = G + Het + Noise
    
    Each component is sampled from Matrix-Normal and scaled to target per-trait
    variance. Total phenotypic variance per trait = target_total_variance.
    
    TRAIT COVARIANCE STRUCTURES (trace = P):
    
    C_G = var_G × 1×1ᵀ       (rank-1, uniform loadings)
    C_het = var_het × I_P    (diagonal, independent)
    C_noise = var_noise × I_P
    
    Where var_G + var_het + var_noise = target_total_variance (per trait).
    
    ==========================================================================
    DECOMPOSITION RECOVERY
    ==========================================================================
    
    C0 = C_G + C_het (genetic covariance)
    C1 = C_noise
    
    Shared (λ₁):        First eigenvalue = P × var_G + var_het
    Heterogeneity:      Σᵢ₌₂ λᵢ = (P-1) × var_het
    Noise:              trace(C1) = P × var_noise
    
    BIAS: Diagonal heterogeneity contributes var_het to λ₁, causing:
      - Shared overestimated by var_het
      - Heterogeneity underestimated by var_het
    
    """
    VARIANCE_SCENARIOS = {
        0: dict(name="morphological",
                persistent_prop=0.60,
                heterogeneity_prop=0.15,
                noise_prop=0.25,
                note="Stable morphology"),

        1: dict(name="yield_favorable",
                persistent_prop=0.42,
                heterogeneity_prop=0.23,
                noise_prop=0.35,
                note="Yield, good environments"),

        2: dict(name="yield_diverse",
                persistent_prop=0.28,
                heterogeneity_prop=0.27,
                noise_prop=0.45,
                note="MET yield"),

        3: dict(name="stress_response",
                persistent_prop=0.16,
                heterogeneity_prop=0.29,
                noise_prop=0.55,
                note="Abiotic/biotic stress"),
    }
    
    def __init__(self, X, P=4, rep_idx=None, precomputed_kinship=None):
        """
        Initialize simulator.
        
        Parameters
        --
        X : np.ndarray or pd.DataFrame, shape (N, S)
            Genotype matrix
        P : int
            Number of environments/traits
        rep_idx : int, optional
            Replicate index for reproducibility
        """
        self.X = X
        self.N = X.shape[0]
        self.num_snps = X.shape[1]
        self.P = P
        
        self._init_rng(rep_idx)
        
        print(f"[INFO] Computing kinship matrix from {self.num_snps} SNPs...")
        if precomputed_kinship is not None:
            self.K = precomputed_kinship
        else:
            self.K = self._compute_kinship_matrix()
        
        print(f"[INFO] VarDecSimulator initialized:")
        print(f"  - N samples: {self.N}")
        print(f"  - P environments: {self.P}")
        print(f"  - Kinship trace: {np.trace(self.K):.2f}")

    def _init_rng(self, rep_idx):
        """Initialize RNG with seed."""
        seed = 120 if rep_idx is None else 100 + rep_idx
        self.rng = np.random.default_rng(seed)
        print(f"[INFO] RNG initialized with seed: {seed}")

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
    
    def _safe_cholesky(self, M, eps=1e-8):
        """Safe Cholesky decomposition with fallback."""
        M_stable = M + eps * np.eye(M.shape[0])
        try:
            return np.linalg.cholesky(M_stable)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(M_stable)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(np.sqrt(eigvals))

    def _sample_kronecker(self, K, C):
        """Sample from MN(0, K, C) i.e., vec(Y) ~ N(0, C ⊗ K)."""
        L_sample = self._safe_cholesky(K)
        L_trait = self._safe_cholesky(C, eps=1e-10)
        Z = self.rng.standard_normal((self.N, self.P))
        return L_sample @ Z @ L_trait.T

    def _sample_and_scale(self, K, C_unit, target_var_per_trait, name="", debug=False):
        """
        Sample from Kronecker and scale to target per-trait variance.
        
        Parameters
        --
        K : array (N, N)
            Sample covariance (kinship)
        C_unit : array (P, P)
            Unit trait covariance structure (trace = P)
        target_var_per_trait : float
            Target variance per trait (not total!)
        """
        if target_var_per_trait <= 0:
            if debug:
                print(f"    {name}: target=0, returning zeros")
            return np.zeros((self.N, self.P))
        
        raw = self._sample_kronecker(K, C_unit)
        
        # Current per-trait variance (mean across traits)
        current_var_per_trait = np.mean(np.var(raw, axis=0))
        
        if current_var_per_trait > 1e-10:
            scale = np.sqrt(target_var_per_trait / current_var_per_trait)
            scaled = raw * scale
        else:
            scaled = raw
            scale = 1.0
        
        achieved_var_per_trait = np.mean(np.var(scaled, axis=0))
        achieved_var_vec = np.var(scaled.flatten())
        
        if debug:
            print(f"    {name}:")
            print(f"      target (per-trait): {target_var_per_trait:.4f}")
            print(f"      achieved (per-trait mean): {achieved_var_per_trait:.4f}")
            print(f"      achieved var[vec]: {achieved_var_vec:.4f}")
            print(f"      per-trait variances: {np.var(scaled, axis=0).round(4)}")
        
        return scaled

    def simulate(
            self,
            persistent_prop=0.40,
            heterogeneity_prop=0.20,
            noise_prop=0.40,
            target_total_variance=1.0,
            scenario_id=None,
            debug=False,
        ):
            """
            Simulate phenotypes.

            Parameters
            ----------
            persistent_prop : float
                Proportion of per-trait variance from shared genetic effects
            heterogeneity_prop : float
                Proportion of per-trait variance from environment-specific genetic effects
            noise_prop : float
                Proportion of per-trait variance from noise
            target_total_variance : float
                Target total variance PER TRAIT
            scenario_id : int, optional
                Scenario identifier
            debug : bool
                Print debug information
            """
            total_prop = persistent_prop + heterogeneity_prop + noise_prop
            if not np.isclose(total_prop, 1.0):
                raise ValueError(f"Proportions must sum to 1.0, got {total_prop:.4f}")

            # Per-trait target variances
            var_G = persistent_prop * target_total_variance
            var_het = heterogeneity_prop * target_total_variance
            var_noise = noise_prop * target_total_variance

            if debug:
                print(f"\n{'='*70}")
                print("VARIANCE DECOMPOSITION SIMULATION")
                print(f"{'='*70}")
                if scenario_id is not None:
                    print(f"  Scenario: {scenario_id}")
                print(f"\n  TARGET VARIANCES (per-trait):")
                print(f"    Persistent (G):     {var_G:.4f} ({persistent_prop:.1%})")
                print(f"    Heterogeneity:      {var_het:.4f} ({heterogeneity_prop:.1%})")
                print(f"    Noise:              {var_noise:.4f} ({noise_prop:.1%})")
                print(f"    Total per trait:    {target_total_variance:.4f}")

            # Unit structures: trace = P, meaning each trait has variance = 1 in unit scale
            C_G_unit = np.ones((self.P, self.P))   # Rank-1 uniform: trace = P
            C_het_unit = np.eye(self.P)            # Diagonal: trace = P
            C_noise_unit = np.eye(self.P)          # Diagonal: trace = P

            # Scaled covariance matrices
            C_G = var_G * C_G_unit           # trace = P × var_G
            C_het = var_het * C_het_unit     # trace = P × var_het
            C_noise = var_noise * C_noise_unit

            C0 = C_G + C_het                 # Genetic covariance
            C1 = C_noise                     # Noise covariance

            if debug:
                print(f"\n  COVARIANCE STRUCTURE:")
                print(f"    trace(C_G)     = {np.trace(C_G):.4f} = P × {var_G:.4f}")
                print(f"    trace(C_het)   = {np.trace(C_het):.4f} = P × {var_het:.4f}")
                print(f"    trace(C_noise) = {np.trace(C_noise):.4f} = P × {var_noise:.4f}")
                print(f"    ─────────────────────────────")
                print(f"    trace(total)   = {np.trace(C0) + np.trace(C1):.4f} = P = {self.P}")

                print(f"\n    Per-trait variances (diagonal):")
                print(f"      C_G diag:     {np.diag(C_G).round(4)}")
                print(f"      C_het diag:   {np.diag(C_het).round(4)}")
                print(f"      C_noise diag: {np.diag(C_noise).round(4)}")
                print(f"      Total diag:   {(np.diag(C0) + np.diag(C1)).round(4)}")

            if debug:
                print(f"\n  SAMPLING COMPONENTS:")

            G = self._sample_and_scale(self.K, C_G_unit, var_G, "G (persistent)", debug)
            Het = self._sample_and_scale(self.K, C_het_unit, var_het, "Het", debug)
            Noise = self._sample_and_scale(np.eye(self.N), C_noise_unit, var_noise, "Noise", debug)

            Y = G + Het + Noise

            # Empirical Variances
            var_G_per_trait = np.var(G, axis=0)
            var_het_per_trait = np.var(Het, axis=0)
            var_noise_per_trait = np.var(Noise, axis=0)
            var_Y_per_trait = np.var(Y, axis=0)

            var_G_achieved = np.mean(var_G_per_trait)
            var_het_achieved = np.mean(var_het_per_trait)
            var_noise_achieved = np.mean(var_noise_per_trait)
            var_Y_achieved = np.mean(var_Y_per_trait)

            if debug:
                print(f"\n  ACHIEVED VARIANCES (per-trait mean):")
                print(f"    G:     {var_G_achieved:.4f} (target: {var_G:.4f})")
                print(f"    Het:   {var_het_achieved:.4f} (target: {var_het:.4f})")
                print(f"    Noise: {var_noise_achieved:.4f} (target: {var_noise:.4f})")
                print(f"    Y:     {var_Y_achieved:.4f} (target: {target_total_variance:.4f})")

            if debug:
                cross_G_Het = np.mean([np.cov(G[:,j], Het[:,j])[0,1] for j in range(self.P)])
                cross_G_Noise = np.mean([np.cov(G[:,j], Noise[:,j])[0,1] for j in range(self.P)])
                cross_Het_Noise = np.mean([np.cov(Het[:,j], Noise[:,j])[0,1] for j in range(self.P)])

                print(f"\n  CROSS-COVARIANCES (should be ≈ 0):")
                print(f"    cov(G, Het):     {cross_G_Het:.6f}")
                print(f"    cov(G, Noise):   {cross_G_Noise:.6f}")
                print(f"    cov(Het, Noise): {cross_Het_Noise:.6f}")

            # Eigenstructure (for diagnostics)
            eigenvals_C0 = np.linalg.eigvalsh(C0)[::-1]
            eigenvecs_C0 = np.linalg.eigh(C0)[1][:, ::-1]
            v1 = eigenvecs_C0[:, 0]

            # Expected eigenvalues for rank-1 + diagonal
            lambda1_expected = self.P * var_G + var_het
            lambda_rest_expected = var_het

            if debug:
                print(f"\n  EIGENSTRUCTURE OF C0:")
                print(f"    Eigenvalues: {eigenvals_C0.round(4)}")
                print(f"    Expected:    λ₁ = P×var_G + var_het = {lambda1_expected:.4f}")
                print(f"                 λ₂...λ_P = var_het = {lambda_rest_expected:.4f}")
                print(f"    v₁ = {v1.round(4)}")
                uniform = np.ones(self.P) / np.sqrt(self.P)
                alignment = abs(np.dot(v1, uniform))
                print(f"    v₁ alignment with uniform: {alignment:.4f}")

            # Off-diagonal mean extracts pure shared signal
            off_diag_mask = ~np.eye(self.P, dtype=bool)
            mean_off_diag_C0 = np.mean(C0[off_diag_mask])

            # Block Model decomposition
            var_shared_decomposed = self.P * mean_off_diag_C0
            var_het_decomposed = max(0, np.trace(C0) - var_shared_decomposed)
            var_noise_decomposed = np.trace(C1)
            var_total_decomposed = var_shared_decomposed + var_het_decomposed + var_noise_decomposed

            # Also compute eigenvalue-based (biased) for comparison
            var_shared_eigen = eigenvals_C0[0]
            var_het_eigen = max(0, np.sum(eigenvals_C0[1:]))

            # True total variances
            var_G_true_total = self.P * var_G
            var_het_true_total = self.P * var_het
            var_noise_true_total = self.P * var_noise

            if debug:
                print(f"\n  DECOMPOSITION COMPARISON:")
                print(f"    {'Method':<20} {'Shared':<10} {'Het':<10} {'Noise':<10}")
                print(f"    {'-'*50}")
                print(f"    {'Off-diag (unbiased)':<20} {var_shared_decomposed:<10.4f} {var_het_decomposed:<10.4f} {var_noise_decomposed:<10.4f}")
                print(f"    {'Eigenvalue (biased)':<20} {var_shared_eigen:<10.4f} {var_het_eigen:<10.4f} {var_noise_decomposed:<10.4f}")
                print(f"    {'True':<20} {var_G_true_total:<10.4f} {var_het_true_total:<10.4f} {var_noise_true_total:<10.4f}")

                print(f"\n  OFF-DIAGONAL DECOMPOSITION DETAIL:")
                print(f"    mean(off-diag C0) = {mean_off_diag_C0:.4f} (should ≈ var_G = {var_G:.4f})")
                print(f"    Shared = P × mean(off-diag) = {self.P} × {mean_off_diag_C0:.4f} = {var_shared_decomposed:.4f}")
                print(f"    Het = trace(C0) - Shared = {np.trace(C0):.4f} - {var_shared_decomposed:.4f} = {var_het_decomposed:.4f}")
                # Proportions
                print(f"\n  PROPORTIONS:")
                print(f"    {'Component':<15} {'Decomposed':<12} {'True':<12} {'Match':<8}")
                print(f"    {'-'*47}")
                prop_shared = var_shared_decomposed / var_total_decomposed
                prop_het = var_het_decomposed / var_total_decomposed
                prop_noise = var_noise_decomposed / var_total_decomposed
                print(f"    {'Shared':<15} {prop_shared:>11.1%} {persistent_prop:>11.1%}")
                print(f"    {'Heterogeneity':<15} {prop_het:>11.1%} {heterogeneity_prop:>11.1%}")
                print(f"    {'Noise':<15} {prop_noise:>11.1%} {noise_prop:>11.1%}")

                print(f"{'='*70}\n")

            trait_cols = [f"trait_{i+1}" for i in range(self.P)]
            if isinstance(self.X, pd.DataFrame):
                Y_df = pd.DataFrame(Y, index=self.X.index, columns=trait_cols)
            else:
                Y_df = pd.DataFrame(Y, columns=trait_cols)

            ground_truth = {
                # Input parameters 
                'persistent_prop': persistent_prop,
                'heterogeneity_prop': heterogeneity_prop,
                'noise_prop': noise_prop,
                'target_total_variance': target_total_variance,

                # Target variances (per-trait) 
                'var_G': var_G,
                'var_het': var_het,
                'var_noise': var_noise,

                # True total variances (= P × per-trait) 
                'var_G_true_total': var_G_true_total,
                'var_het_true_total': var_het_true_total,
                'var_noise_true_total': var_noise_true_total,

                # Achieved variances (empirical) 
                'var_G_achieved': var_G_achieved,
                'var_het_achieved': var_het_achieved,
                'var_noise_achieved': var_noise_achieved,
                'var_Y_achieved': var_Y_achieved,

                # Per-trait variances
                'var_G_per_trait': var_G_per_trait,
                'var_het_per_trait': var_het_per_trait,
                'var_noise_per_trait': var_noise_per_trait,
                'var_Y_per_trait': var_Y_per_trait,

                # Designed covariance matrices 
                'C_G': C_G,
                'C_het': C_het,
                'C0': C0,
                'C1': C1,
                'C_G_unit': C_G_unit,
                'C_het_unit': C_het_unit,

                # Eigenstructure 
                'eigenvalues_C0': eigenvals_C0,
                'eigenvectors_C0': eigenvecs_C0,
                'v1': v1,
                'mean_off_diag_C0': mean_off_diag_C0,

                # Off-diagonal decomposition (unbiased)
                'var_shared_decomposed': var_shared_decomposed,
                'var_het_decomposed': var_het_decomposed,
                'var_noise_decomposed': var_noise_decomposed,
                'var_total_decomposed': var_total_decomposed,

                'prop_shared_decomposed': var_shared_decomposed / var_total_decomposed,
                'prop_het_decomposed': var_het_decomposed / var_total_decomposed,
                'prop_noise_decomposed': var_noise_decomposed / var_total_decomposed,

                # Eigenvalue decomposition (biased, for comparison)
                'var_shared_eigen': var_shared_eigen,
                'var_het_eigen': var_het_eigen,

                # Components
                'G': G,
                'Het': Het,
                'Noise': Noise,

                'scenario_id': scenario_id,
                'n_samples': self.N,
                'n_traits': self.P,
            }

            return Y_df, ground_truth

    def genPheno(self, scenario=None, **kwargs):
        """Generate phenotypes from scenario or custom parameters."""
        if scenario is not None:
            if scenario not in self.VARIANCE_SCENARIOS:
                raise ValueError(f"Unknown scenario {scenario}.")
            
            config = self.VARIANCE_SCENARIOS[scenario]
            print(f"\n[SCENARIO {scenario}] {config['name']}")
            print(f"  {config['note']}")
            
            params = {
                'persistent_prop': config['persistent_prop'],
                'heterogeneity_prop': config['heterogeneity_prop'],
                'noise_prop': config['noise_prop'],
                'scenario_id': scenario
            }
            params.update(kwargs)
            
            return self.simulate(**params)
        else:
            return self.simulate(**kwargs)

    @classmethod
    def list_scenarios(cls):
        """Print available scenarios."""
        print(f"\n{'='*75}")
        print("AVAILABLE SCENARIOS")
        print(f"{'='*75}")
        print(f"{'ID':<3} {'Name':<20} {'G':<8} {'Het':<8} {'Noise':<8} {'Note'}")
        print(f"{'-'*75}")
        for idx, s in cls.VARIANCE_SCENARIOS.items():
            print(f"{idx:<3} {s['name']:<20} {s['persistent_prop']:<8.1%} "
                  f"{s['heterogeneity_prop']:<8.1%} {s['noise_prop']:<8.1%} {s['note']}")
        print(f"{'='*75}")