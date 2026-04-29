"""
Tests for GWAS StoreResults: null model, per-SNP betas, likelihoods, display.

Usage:
    pytest tests/test_result_storage.py -v
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="torchlimix_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_store(tmp_dir, **kwargs):
    from torchlimix.result_factory._store_results import StoreResults
    defaults = dict(
        output_dir=tmp_dir, uid="test_uid",
        test_type="any_vs_common", use_parquet=False,
    )
    defaults.update(kwargs)
    return StoreResults(**defaults)


class TestNullModel:

    def test_save_and_load(self, tmp_dir):
        store = _make_store(tmp_dir)
        beta0 = torch.randn(4, 3)
        beta0_se = torch.abs(torch.randn(4, 3))

        store.save_null_model(beta0, beta0_se, -355.0, 1.0, 4, 3)

        assert os.path.exists(store.null_model_path)
        assert os.path.exists(store.null_model_txt_path)

        loaded = store.load_null_model()
        assert loaded['n_covariates'] == 4
        assert loaded['n_traits'] == 3
        np.testing.assert_array_almost_equal(loaded['beta0'], beta0.numpy(), decimal=5)
        np.testing.assert_array_almost_equal(loaded['beta0_se'], beta0_se.numpy(), decimal=5)

    def test_load_from_fresh_instance(self, tmp_dir):
        store1 = _make_store(tmp_dir)
        beta0 = torch.randn(2, 3)
        beta0_se = torch.abs(torch.randn(2, 3))
        store1.save_null_model(beta0, beta0_se, -100.0, 1.0, 2, 3)

        store2 = _make_store(tmp_dir)
        store2._null_model = None
        loaded = store2.load_null_model()
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded['beta0'], beta0.numpy(), decimal=5)

    def test_covariate_labels(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.save_null_model(torch.randn(3, 2), torch.randn(3, 2), -200.0, 1.0, 3, 2)
        loaded = store.load_null_model()
        assert loaded['covariate_labels'] == ['intercept', 'cov_1', 'cov_2']

    def test_null_model_txt_readable(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.save_null_model(torch.randn(2, 3), torch.randn(2, 3), -300.0, 0.99, 2, 3)
        with open(store.null_model_txt_path, 'r') as f:
            content = f.read()
        assert "lml0" in content
        assert "scale_H0" in content
        assert "intercept" in content

    def test_missing_null_model_returns_none(self, tmp_dir):
        store = _make_store(tmp_dir)
        store._null_model = None
        # Don't save anything
        if os.path.exists(store.null_model_path):
            os.remove(store.null_model_path)
        assert store.load_null_model() is None



class TestBetaStorage:

    def test_vector_betas(self, tmp_dir):
        store = _make_store(tmp_dir)
        n = 100
        store.add_beta_result(
            np.arange(n),
            beta1=torch.randn(n, 3), beta1_se=torch.abs(torch.randn(n, 3)),
            beta2=torch.randn(n, 3), beta2_se=torch.abs(torch.randn(n, 3)),
        )
        df = store.effectsizes()
        assert len(df) == n
        beta1_cols = [c for c in df.columns if c.startswith('beta1_') and 'se' not in c]
        assert len(beta1_cols) == 3

    def test_scalar_betas(self, tmp_dir):
        store = _make_store(tmp_dir, test_type="common")
        n = 50
        store.add_beta_result(
            np.arange(n),
            beta1=torch.randn(n), beta1_se=torch.abs(torch.randn(n)),
        )
        df = store.effectsizes()
        assert len(df) == n
        assert 'beta1' in df.columns

    def test_no_beta0_in_per_snp(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.add_beta_result(
            np.arange(10),
            beta1=torch.randn(10, 3), beta1_se=torch.randn(10, 3),
            beta2=torch.randn(10, 3), beta2_se=torch.randn(10, 3),
        )
        df = store.effectsizes()
        beta0_cols = [c for c in df.columns if 'beta0' in c]
        assert len(beta0_cols) == 0

    def test_parquet_output(self, tmp_dir):
        store = _make_store(tmp_dir, use_parquet=True)
        store.add_beta_result(
            np.arange(20),
            beta1=torch.randn(20), beta1_se=torch.randn(20),
            beta2=torch.randn(20), beta2_se=torch.randn(20),
        )
        assert store.beta_path.endswith('.parquet')
        df = pd.read_parquet(store.beta_path)
        assert len(df) == 20

    def test_effectsizes_disk_fallback(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.add_beta_result(
            np.arange(30),
            beta1=torch.randn(30), beta1_se=torch.randn(30),
            beta2=torch.randn(30), beta2_se=torch.randn(30),
        )
        store._beta_df = None  # clear cache
        df = store.effectsizes()
        assert len(df) == 30

    def test_integer_snp_index_input(self, tmp_dir):
        store = _make_store(tmp_dir, test_type="common")
        store.add_beta_result(
            0,  # single int
            beta1=torch.randn(50), beta1_se=torch.randn(50),
        )
        df = store.effectsizes()
        assert len(df) == 50


class TestLikelihoodStorage:

    def test_basic(self, tmp_dir):
        store = _make_store(tmp_dir)
        n = 100
        store.add_likelihood_result(
            np.arange(n), -350.0,
            torch.randn(n).double() - 350,
            torch.randn(n).double() - 349,
            torch.abs(torch.randn(n)).double(),
            torch.abs(torch.randn(n)).double(),
            torch.abs(torch.randn(n)).double(),
            df10=1, df20=3, df21=2, scale_H0=1.0,
            C0=torch.eye(3), C1=torch.eye(3) * 0.5,
        )
        assert os.path.exists(store.likelihood_path)
        df = store.likelihood_results()
        assert len(df) == n
        assert 'pv10' in df.columns

    def test_covariance_stored_once(self, tmp_dir):
        store = _make_store(tmp_dir)
        C0 = torch.eye(3) * 2
        C1 = torch.eye(3) * 0.5
        store.add_likelihood_result(
            np.arange(10), -350.0,
            torch.randn(10).double(), torch.randn(10).double(),
            torch.randn(10).double(), torch.randn(10).double(),
            torch.randn(10).double(),
            C0=C0, C1=C1,
        )
        assert 0 in store.covariance_results
        np.testing.assert_array_almost_equal(
            store.covariance_results[0]['C0'], C0.numpy()
        )

    def test_no_h2_headers(self, tmp_dir):
        store = _make_store(tmp_dir, test_type="common")
        assert 'lml2' not in store.headers
        assert 'pv20' not in store.headers

class TestPValues:

    def test_basic_values(self, tmp_dir):
        store = _make_store(tmp_dir)
        lrt = np.array([0.0, 1.0, 5.0, 10.0])
        df = np.array([1.0, 1.0, 2.0, 3.0])
        pv = store._compute_p_values(lrt, df)

        assert pv[0] == pytest.approx(1.0, abs=1e-6)
        assert all(0 < p < 1 for p in pv[1:])

    def test_nan_and_negative(self, tmp_dir):
        store = _make_store(tmp_dir)
        lrt = np.array([np.nan, -1.0])
        df = np.array([1.0, 1.0])
        pv = store._compute_p_values(lrt, df)
        assert all(np.isnan(pv))


class TestTensorHelpers:

    def test_none_returns_none(self):
        from torchlimix.result_factory._store_results import _tensor_to_columns
        assert _tensor_to_columns(None, 10) is None

    def test_1d(self):
        from torchlimix.result_factory._store_results import _tensor_to_columns
        arr = _tensor_to_columns(torch.randn(50), 50)
        assert arr.shape == (50,)

    def test_2d_single_col(self):
        from torchlimix.result_factory._store_results import _tensor_to_columns
        arr = _tensor_to_columns(torch.randn(50, 1), 50)
        assert arr.shape == (50,)

    def test_2d_multi_col(self):
        from torchlimix.result_factory._store_results import _tensor_to_columns
        arr = _tensor_to_columns(torch.randn(50, 3), 50)
        assert arr.shape == (50, 3)

    def test_add_columns_scalar(self):
        from torchlimix.result_factory._store_results import _add_columns_from_tensor
        cols = {}
        _add_columns_from_tensor(cols, "beta1", torch.randn(20), 20)
        assert "beta1" in cols

    def test_add_columns_vector(self):
        from torchlimix.result_factory._store_results import _add_columns_from_tensor
        cols = {}
        _add_columns_from_tensor(cols, "beta1", torch.randn(20, 3), 20)
        assert "beta1_0" in cols and "beta1_1" in cols and "beta1_2" in cols
        assert "beta1" not in cols

    def test_add_columns_none(self):
        from torchlimix.result_factory._store_results import _add_columns_from_tensor
        cols = {}
        _add_columns_from_tensor(cols, "x", None, 10)
        assert len(cols) == 0

class TestExtractBetaValues:
    """Verify that prefix matching separates beta from beta_se columns."""

    def test_expanded_columns_no_se_leakage(self):
        """beta1 prefix should NOT match beta1_se columns."""
        df = pd.DataFrame({
            'snp_index': [0, 1],
            'beta1_0': [0.1, 0.2], 'beta1_1': [0.3, 0.4],
            'beta1_se_0': [0.01, 0.02], 'beta1_se_1': [0.03, 0.04],
        })
        # Correct prefix matching (excludes 'se')
        matching = [c for c in df.columns
                    if (c == 'beta1' or c.startswith('beta1_'))
                    and 'se' not in c]
        assert len(matching) == 2
        assert 'beta1_se_0' not in matching

    def test_scalar_column_match(self):
        df = pd.DataFrame({'snp_index': [0], 'beta1': [0.5], 'beta1_se': [0.01]})
        matching = [c for c in df.columns
                    if (c == 'beta1' or c.startswith('beta1_'))
                    and 'se' not in c]
        assert matching == ['beta1']


class TestDisplayFormatting:

    def test_null_model_table(self, tmp_dir):
        store = _make_store(tmp_dir)
        beta0 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        beta0_se = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
        text = store._format_null_model_table(beta0, beta0_se, 3, 2)

        assert "intercept" in text
        assert "cov_1" in text
        assert "cov_2" in text
        assert "trait_0" in text
        assert "0.100000" in text
        assert "Standard errors" in text