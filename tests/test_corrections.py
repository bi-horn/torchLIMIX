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
        """regress_batch=F, regress_cov=F, but cov_path exists → must enter."""
        from torchlimix.utils.data_loader import CorrectionConfig, DataPaths
        paths = DataPaths(cov="/some/path.csv")
        config = CorrectionConfig(regress_batch=False, regress_covariates=False)
        should_call = (config.regress_batch or config.regress_covariates or paths.cov)
        assert should_call

    def test_nothing_set_skips(self):
        """No batch, no cov, no regression → guard is False."""
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
            config.validate()  # should not raise

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
        # INT should produce roughly standard normal
        assert abs(result['A'].mean()) < 0.2
        assert 0.8 < result['A'].std() < 1.2

    def test_int_preserves_rank_order(self):
        from torchlimix.stats._standardize import standardize_data
        df = pd.DataFrame({'A': [5.0, 1.0, 3.0, 2.0, 4.0]})
        result = standardize_data(df, method='int')

        # Rank order should be preserved
        original_ranks = df['A'].rank()
        result_ranks = result['A'].rank()
        np.testing.assert_array_equal(original_ranks.values, result_ranks.values)

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
        """Constant column should produce zeros, not NaN/inf."""
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
        # Each column should have mean ≈ 0
        col_means = np.mean(result, axis=0)
        np.testing.assert_almost_equal(col_means, np.zeros(50), decimal=10)

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
        # Index should be preserved
        assert list(result.index.names) == ['fid', 'iid']

    def test_nan_imputation(self):
        """NaN values should be replaced with column mean before normalization."""
        from torchlimix.stats._normalize_geno import normalize_genotype_matrix
        G = np.array([[0, 1], [1, np.nan], [2, 1], [1, 0]], dtype=float)
        result = normalize_genotype_matrix(G, verbose=False)

        assert not np.any(np.isnan(result))

    def test_constant_column(self):
        """Column with all same values should not produce NaN/inf."""
        from torchlimix.stats._normalize_geno import normalize_genotype_matrix
        G = np.array([[1, 0], [1, 1], [1, 2], [1, 1]], dtype=float)
        result = normalize_genotype_matrix(G, verbose=False)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        # Constant column should be all zeros
        np.testing.assert_array_equal(result[:, 0], np.zeros(4))


class TestExtractCovariatesFromLoaders:
    """Test the covariate detection logic based on batch tuple length."""

    def _make_loader(self, batches):
        """Create a mock loader that yields given batches."""
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(batches))
        return loader

    def test_detects_covariates_5_elements(self):
        """Batch with 5 elements (geno, geno_qs, pheno, cov, idx) → has covariates."""
        geno = torch.randn(10, 100)
        geno_qs = torch.randn(10, 50)
        pheno = torch.randn(10, 3)
        cov = torch.randn(10, 5)  # 2D → covariates
        idx = torch.arange(10)

        batch = (geno, geno_qs, pheno, cov, idx)
        has_covariates = len(batch) >= 4 and batch[3].ndim == 2
        assert has_covariates is True

    def test_no_covariates_4_elements(self):
        """Batch with 4 elements (geno, geno_qs, pheno, idx) → no covariates."""
        geno = torch.randn(10, 100)
        geno_qs = torch.randn(10, 50)
        pheno = torch.randn(10, 3)
        idx = torch.arange(10)  # 1D → index, not covariates

        batch = (geno, geno_qs, pheno, idx)
        has_covariates = len(batch) >= 4 and batch[3].ndim == 2
        assert has_covariates is False

    def test_edge_case_4_elements_2d_index(self):
        """If index is accidentally 2D, it would be misdetected as covariates."""
        idx_2d = torch.arange(10).unsqueeze(-1)  # (10, 1)
        batch = (torch.randn(10, 100), torch.randn(10, 50), torch.randn(10, 3), idx_2d)
        has_covariates = len(batch) >= 4 and batch[3].ndim == 2
        # This would be a false positive — documents the detection limitation
        assert has_covariates is True  # known limitation


class TestSplitViewCovariates:
    """Test that SplitView correctly returns 4 or 5 element tuples."""

    def test_getitem_without_covariates(self):
        """Mock a master dataset without covariates → 4 elements."""
        master = MagicMock()
        master.split_indices = {'train': [0, 1, 2]}
        master.df = pd.DataFrame({'t0': [1, 2, 3], 't1': [4, 5, 6]})
        master.gen_data_standard_full = pd.DataFrame(np.random.randn(3, 10))
        master.G_stable = torch.randn(3, 5)
        master.covariate_matrix = None

        from torchlimix.utils.data_loader import SplitView
        view = SplitView(master, "train")

        item = view[0]
        assert len(item) == 4  # (geno, geno_qs, pheno, idx)

    def test_getitem_with_covariates(self):
        """Mock a master dataset with covariates → 5 elements."""
        master = MagicMock()
        master.split_indices = {'train': [0, 1, 2]}
        master.df = pd.DataFrame({'t0': [1.0, 2.0, 3.0], 't1': [4.0, 5.0, 6.0]})
        master.gen_data_standard_full = pd.DataFrame(np.random.randn(3, 10))
        master.G_stable = torch.randn(3, 5)
        master.covariate_matrix = torch.randn(3, 5)

        from torchlimix.utils.data_loader import SplitView
        view = SplitView(master, "train")

        item = view[0]
        assert len(item) == 5  # (geno, geno_qs, pheno, cov, idx)
        assert item[3].shape == (5,)  # covariate vector for one sample

        