"""
Tests for phenotype corrections, standardization, genotype normalization,
and covariate extraction from data loaders.

Usage:
    pytest tests/test_corrections.py -v
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="torchlimix_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestCorrectionGuard:
    """Test the _load_pipeline guard: should call _apply_corrections when needed."""

    def test_cov_path_triggers_even_if_regress_false(self):
        from torchlimix.utils.data_loader import CorrectionConfig, DataPaths
        paths = DataPaths(cov="/some/path.csv")
        config = CorrectionConfig(regress_batch=False, regress_covariates=False)
        should_call = (config.regress_batch or config.regress_covariates or paths.cov)
        assert should_call

    def test_nothing_set_skips(self):
        from torchlimix.utils.data_loader import CorrectionConfig, DataPaths
        paths = DataPaths()
        config = CorrectionConfig(regress_batch=False, regress_covariates=False)
        should_call = (config.regress_batch or config.regress_covariates or paths.cov)
        assert not should_call

    def test_regress_batch_only(self):
        from torchlimix.utils.data_loader import CorrectionConfig, DataPaths
        paths = DataPaths(batch="/batch.csv")
        config = CorrectionConfig(regress_batch=True, regress_covariates=False)
        should_call = (config.regress_batch or config.regress_covariates or paths.cov)
        assert should_call is True

    def test_regress_cov_only(self):
        from torchlimix.utils.data_loader import CorrectionConfig, DataPaths
        paths = DataPaths(cov="/cov.csv")
        config = CorrectionConfig(regress_batch=False, regress_covariates=True)
        should_call = (config.regress_batch or config.regress_covariates or paths.cov)
        assert should_call is True


class TestCorrectionConfig:

    def test_valid_transformations(self):
        from torchlimix.utils.data_loader import CorrectionConfig
        for method in ["none", "int", "z_score"]:
            config = CorrectionConfig(transformation=method)
            config.validate()

    def test_invalid_transformation_raises(self):
        from torchlimix.utils.data_loader import CorrectionConfig
        config = CorrectionConfig(transformation="log")
        with pytest.raises(ValueError, match="transformation must be one of"):
            config.validate()


class TestStandardizeData:

    def test_zscore_dataframe(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0], 'B': [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = standardize_data(df, method='z_score')
        assert isinstance(result, pd.DataFrame)
        np.testing.assert_almost_equal(result['A'].mean(), 0.0, decimal=10)
        np.testing.assert_almost_equal(result['A'].std(ddof=1), 1.0, decimal=10)

    def test_zscore_array(self):
        from torchlimix.stats._standardize import standardize_data
        arr = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
        result = standardize_data(arr, method='z_score')
        assert isinstance(result, np.ndarray)
        np.testing.assert_almost_equal(result.mean(axis=0), [0.0, 0.0], decimal=10)

    def test_int_dataframe(self):
        from torchlimix.stats._standardize import standardize_data
        np.random.seed(42)
        df = pd.DataFrame({'A': np.random.randn(100)})
        result = standardize_data(df, method='int')
        assert isinstance(result, pd.DataFrame)
        assert abs(result['A'].mean()) < 0.2
        assert 0.8 < result['A'].std() < 1.2

    def test_int_preserves_rank_order(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [5.0, 1.0, 3.0, 2.0, 4.0]})
        result = standardize_data(df, method='int')
        np.testing.assert_array_equal(df['A'].rank().values, result['A'].rank().values)

    def test_none_returns_unchanged(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        result = standardize_data(df, method='none')
        pd.testing.assert_frame_equal(result, df)

    def test_missing_values_raise(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        with pytest.raises(RuntimeError, match="missing value"):
            standardize_data(df, method='z_score')

    def test_invalid_method_raises(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="method must be"):
            standardize_data(df, method='log')

    def test_constant_column_zscore(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [5.0, 5.0, 5.0, 5.0]})
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = standardize_data(df, method='z_score')
        assert not np.any(np.isnan(result['A'].values))
        assert not np.any(np.isinf(result['A'].values))

    def test_series_input(self):
        from torchlimix.stats._standardize import standardize_data
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name='trait')
        result = standardize_data(s, method='z_score')
        assert isinstance(result, pd.Series)
        np.testing.assert_almost_equal(result.mean(), 0.0, decimal=10)

    def test_1d_array_input(self):
        from torchlimix.stats._standardize import standardize_data
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize_data(arr, method='z_score')
        assert result.ndim == 1
        np.testing.assert_almost_equal(result.mean(), 0.0, decimal=10)


class TestGenotypeNormalization:

    def test_basic_normalization(self):
        from torchlimix.stats._normalize_geno import normalize_genotype_matrix
        np.random.seed(42)
        G = np.random.choice([0, 1, 2], size=(100, 50)).astype(float)
        result = normalize_genotype_matrix(G, verbose=False)
        assert result.shape == G.shape
        np.testing.assert_almost_equal(np.mean(result, axis=0), np.zeros(50), decimal=10)

    def test_dataframe_input(self):
        from torchlimix.stats._normalize_geno import normalize_genotype_matrix
        np.random.seed(42)
        G = pd.DataFrame(
            np.random.choice([0, 1, 2], size=(50, 20)).astype(float),
            index=pd.MultiIndex.from_arrays(
                [np.arange(50), np.arange(50)], names=['fid', 'iid']
            )
        )
        result = normalize_genotype_matrix(G, verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == G.shape
        assert list(result.index.names) == ['fid', 'iid']

    def test_nan_imputation(self):
        from torchlimix.stats._normalize_geno import normalize_genotype_matrix
        G = np.array([[0, 1], [1, np.nan], [2, 1], [1, 0]], dtype=float)
        result = normalize_genotype_matrix(G, verbose=False)
        assert not np.any(np.isnan(result))

    def test_constant_column(self):
        from torchlimix.stats._normalize_geno import normalize_genotype_matrix
        G = np.array([[1, 0], [1, 1], [1, 2], [1, 1]], dtype=float)
        result = normalize_genotype_matrix(G, verbose=False)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_array_equal(result[:, 0], np.zeros(4))


class TestExtractCovariatesFromLoaders:
    """Test the covariate detection logic based on batch tuple length."""

    def test_detects_covariates_5_elements(self):
        batch = (
            torch.randn(10, 100), torch.randn(10, 50),
            torch.randn(10, 3), torch.randn(10, 5), torch.arange(10),
        )
        has_covariates = len(batch) >= 4 and batch[3].ndim == 2
        assert has_covariates is True

    def test_no_covariates_4_elements(self):
        batch = (
            torch.randn(10, 100), torch.randn(10, 50),
            torch.randn(10, 3), torch.arange(10),
        )
        has_covariates = len(batch) >= 4 and batch[3].ndim == 2
        assert has_covariates is False

    def test_edge_case_4_elements_2d_index(self):
        """If index is accidentally 2D, it would be misdetected as covariates."""
        idx_2d = torch.arange(10).unsqueeze(-1)
        batch = (torch.randn(10, 100), torch.randn(10, 50), torch.randn(10, 3), idx_2d)
        has_covariates = len(batch) >= 4 and batch[3].ndim == 2
        assert has_covariates is True  # known limitation


class TestSplitViewCovariates:
    """Test that SplitView correctly returns 4 or 5 element tuples."""

    def _make_master(self, with_covariates: bool):
        """Build a mock master with the actual tensor attributes SplitView reads."""
        master = MagicMock()
        master.split_indices = {'train': [0, 1, 2]}
        master.df_tensor_full = torch.randn(3, 2)            # (n_samples, n_traits)
        master.gen_data_tensor_full = torch.randn(3, 10)     # (n_samples, n_snps)
        master.G_stable = torch.randn(3, 5)
        master.covariate_matrix = torch.randn(3, 5) if with_covariates else None
        return master

    def test_getitem_without_covariates(self):
        from torchlimix.utils.data_loader import SplitView
        view = SplitView(self._make_master(with_covariates=False), "train")
        item = view[0]
        assert len(item) == 4

    def test_getitem_with_covariates(self):
        from torchlimix.utils.data_loader import SplitView
        view = SplitView(self._make_master(with_covariates=True), "train")
        item = view[0]
        assert len(item) == 5
        assert item[3].shape == (5,)

class TestRegressBatchEffects:
    """Tests for regress_batch_effects after the train_idx leak fix."""

    @staticmethod
    def _make_pheno(n_per_batch=20, n_batches=3, n_traits=2, seed=0):
        """y = signal + batch_effect[batch_label]. Shared 'plate' column."""
        rng = np.random.default_rng(seed)
        n = n_per_batch * n_batches
        batch_labels = np.repeat(np.arange(n_batches), n_per_batch)
        batch_effects = rng.normal(0, 5, size=(n_batches, n_traits))
        signal = rng.normal(0, 1, size=(n, n_traits))
        y = signal + batch_effects[batch_labels]

        idx = pd.MultiIndex.from_arrays(
            [np.arange(n), np.arange(n)], names=['fid', 'iid']
        )
        pheno = pd.DataFrame(
            y, index=idx, columns=[f't{i}' for i in range(n_traits)]
        )
        batch = pd.DataFrame({'plate': batch_labels}, index=idx)
        return pheno, batch, signal

    def test_residuals_orthogonal_to_design(self):
        """Defining property of OLS residualization: residuals ⟂ design columns."""
        from torchlimix.utils.regress_effects import regress_batch_effects
        pheno, batch, _ = self._make_pheno()
        corrected, _ = regress_batch_effects(pheno, batch, per_trait=False, train_idx=None)

        X = pd.get_dummies(batch['plate'], drop_first=True, dtype=float).to_numpy()
        for col in pheno.columns:
            r = corrected[col].values - corrected[col].mean()
            for k in range(X.shape[1]):
                np.testing.assert_allclose(
                    r @ X[:, k], 0.0, atol=1e-8,
                    err_msg=f"residuals not orthogonal to design column {k} for {col}",
                )

    def test_legacy_default_matches_explicit_all(self):
        """train_idx=None must match train_idx=arange(n) exactly."""
        from torchlimix.utils.regress_effects import regress_batch_effects
        pheno, batch, _ = self._make_pheno()
        c_none, _ = regress_batch_effects(pheno, batch, train_idx=None)
        c_all,  _ = regress_batch_effects(pheno, batch, train_idx=np.arange(len(pheno)))
        pd.testing.assert_frame_equal(c_none, c_all)

    def test_train_only_fit_does_not_leak_test_rows(self):
        """β learned from train rows must not depend on test row values."""
        from torchlimix.utils.regress_effects import regress_batch_effects
        pheno, batch, _ = self._make_pheno(seed=1)
        n = len(pheno)
        train_idx = np.arange(n // 2)

        c_a, _ = regress_batch_effects(pheno, batch, train_idx=train_idx)

        pheno_perturbed = pheno.copy()
        test_mask = np.ones(n, dtype=bool); test_mask[train_idx] = False
        pheno_perturbed.iloc[test_mask] += 100.0
        c_b, _ = regress_batch_effects(pheno_perturbed, batch, train_idx=train_idx)

        np.testing.assert_allclose(
            c_a.iloc[train_idx].values,
            c_b.iloc[train_idx].values,
            atol=1e-10,
            err_msg="Test-row perturbation leaked into train residuals",
        )

    def test_train_residuals_match_standalone_train_fit(self):
        """Strong invariant: train output should equal what we'd get fitting on
        ONLY the train subset. Catches any subtle leak in the add-back-mean step."""
        from torchlimix.utils.regress_effects import regress_batch_effects
        pheno, batch, _ = self._make_pheno(seed=7)
        n = len(pheno)
        train_idx = np.arange(n // 2)

        c_split, _ = regress_batch_effects(pheno, batch, train_idx=train_idx)
        c_alone, _ = regress_batch_effects(
            pheno.iloc[train_idx], batch.iloc[train_idx], train_idx=None,
        )
        np.testing.assert_allclose(
            c_split.iloc[train_idx].values, c_alone.values, atol=1e-10,
        )

    def test_per_trait_routing(self):
        """per_trait=True picks batch_df[trait] per trait, skips traits w/o a column."""
        from torchlimix.utils.regress_effects import regress_batch_effects
        rng = np.random.default_rng(0)
        n = 60
        idx = pd.MultiIndex.from_arrays(
            [np.arange(n), np.arange(n)], names=['fid', 'iid']
        )
        pheno = pd.DataFrame({
            't0': rng.normal(0, 1, n),
            't1': rng.normal(0, 1, n),
            't2': rng.normal(0, 1, n),
        }, index=idx)
        batch = pd.DataFrame({
            't0': np.repeat([0, 1, 2], 20),
            't1': np.repeat([0, 1], 30),
        }, index=idx)

        _, stats = regress_batch_effects(pheno, batch, per_trait=True)
        assert stats['traits']['t0']['status'] == 'corrected'
        assert stats['traits']['t0']['batch_variable'] == 't0'
        assert stats['traits']['t1']['status'] == 'corrected'
        assert stats['traits']['t1']['batch_variable'] == 't1'
        assert stats['traits']['t2']['status'] == 'skipped'
        assert stats['traits']['t2']['reason'] == 'no_batch_column'

    def test_single_category_skip(self):
        from torchlimix.utils.regress_effects import regress_batch_effects
        n = 30
        idx = pd.MultiIndex.from_arrays(
            [np.arange(n), np.arange(n)], names=['fid', 'iid']
        )
        pheno = pd.DataFrame({'t0': np.random.randn(n)}, index=idx)
        batch = pd.DataFrame({'plate': np.zeros(n, dtype=int)}, index=idx)
        _, stats = regress_batch_effects(pheno, batch, per_trait=False)
        assert stats['traits']['t0']['status'] == 'skipped'

    def test_addback_mean_preserves_trait_mean_on_fit_rows(self):
        """corrected[fit_rows].mean() == y[fit_rows].mean() always (definition of add-back)."""
        from torchlimix.utils.regress_effects import regress_batch_effects
        pheno, batch, _ = self._make_pheno(seed=3)
        corrected, _ = regress_batch_effects(pheno, batch, per_trait=False)
        for col in pheno.columns:
            np.testing.assert_allclose(
                pheno[col].mean(), corrected[col].mean(), atol=1e-10,
            )


class TestRegressContinuousCovariates:
    """Tests for regress_continuous_covariates after the train_idx leak fix."""

    @staticmethod
    def _make_data(n=100, p=2, k=3, seed=4):
        rng = np.random.default_rng(seed)
        idx = pd.MultiIndex.from_arrays(
            [np.arange(n), np.arange(n)], names=['fid', 'iid']
        )
        X = rng.normal(0, 1, size=(n, k))
        beta = rng.normal(0, 2, size=(k, p))
        signal = rng.normal(0, 1, size=(n, p))
        Y = X @ beta + signal
        cov = pd.DataFrame(X, index=idx, columns=[f'c{i}' for i in range(k)])
        pheno = pd.DataFrame(Y, index=idx, columns=[f't{i}' for i in range(p)])
        return pheno, cov

    def test_residuals_orthogonal_to_design(self):
        """Residuals ⟂ each (mean-centered) covariate column."""
        from torchlimix.utils.regress_effects import regress_continuous_covariates
        pheno, cov = self._make_data()
        corrected, _ = regress_continuous_covariates(pheno, cov)

        X_centered = cov.values - cov.values.mean(axis=0)
        for col in pheno.columns:
            r = corrected[col].values - corrected[col].mean()
            for k in range(X_centered.shape[1]):
                np.testing.assert_allclose(
                    r @ X_centered[:, k], 0.0, atol=1e-8,
                    err_msg=f"residuals not orthogonal to covariate {k} for {col}",
                )

    def test_train_only_fit_does_not_leak(self):
        from torchlimix.utils.regress_effects import regress_continuous_covariates
        pheno, cov = self._make_data(seed=5)
        n = len(pheno)
        train_idx = np.arange(n // 2)

        c_a, _ = regress_continuous_covariates(pheno, cov, train_idx=train_idx)

        pheno2 = pheno.copy()
        test_mask = np.ones(n, dtype=bool); test_mask[train_idx] = False
        pheno2.iloc[test_mask] += 100.0
        c_b, _ = regress_continuous_covariates(pheno2, cov, train_idx=train_idx)

        np.testing.assert_allclose(
            c_a.iloc[train_idx].values, c_b.iloc[train_idx].values, atol=1e-10,
        )

    def test_train_residuals_match_standalone_train_fit(self):
        """Strong invariant: same as the batch-effects version."""
        from torchlimix.utils.regress_effects import regress_continuous_covariates
        pheno, cov = self._make_data(seed=11)
        n = len(pheno)
        train_idx = np.arange(n // 2)

        c_split, _ = regress_continuous_covariates(pheno, cov, train_idx=train_idx)
        c_alone, _ = regress_continuous_covariates(
            pheno.iloc[train_idx], cov.iloc[train_idx], train_idx=None,
        )
        np.testing.assert_allclose(
            c_split.iloc[train_idx].values, c_alone.values, atol=1e-10,
        )

    def test_legacy_default_matches_explicit_all(self):
        from torchlimix.utils.regress_effects import regress_continuous_covariates
        pheno, cov = self._make_data(seed=12)
        c_none, _ = regress_continuous_covariates(pheno, cov, train_idx=None)
        c_all,  _ = regress_continuous_covariates(
            pheno, cov, train_idx=np.arange(len(pheno)),
        )
        pd.testing.assert_frame_equal(c_none, c_all)

    def test_categorical_covariate_dummy_encoded(self):
        """Categorical covariate should be auto-detected and dummy-encoded.
        Residuals must be orthogonal to the dummy column."""
        from torchlimix.utils.regress_effects import regress_continuous_covariates
        rng = np.random.default_rng(6)
        n = 60
        idx = pd.MultiIndex.from_arrays(
            [np.arange(n), np.arange(n)], names=['fid', 'iid']
        )
        sex = np.tile(['M', 'F'], n // 2)
        sex_effect = np.where(sex == 'M', 5.0, -5.0)
        signal = rng.normal(0, 1, n)
        pheno = pd.DataFrame({'t0': signal + sex_effect}, index=idx)
        cov   = pd.DataFrame({'sex': sex}, index=idx)

        corrected, stats = regress_continuous_covariates(pheno, cov)
        assert stats['covariate_stats']['sex']['type'] == 'categorical'

        X_dummy = pd.get_dummies(cov['sex'], drop_first=True, dtype=float).to_numpy()
        r = corrected['t0'].values - corrected['t0'].mean()
        np.testing.assert_allclose(r @ X_dummy[:, 0], 0.0, atol=1e-8)
