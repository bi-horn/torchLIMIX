"""
Tests for PredictionResultStore: two prediction scenarios, metrics, CI,
save/load behaviour, and external genotype loading.

Scenario A: internal train/test split (ground truth available)
Scenario B: external prediction genotypes (no ground truth)

Usage:
    pytest tests/test_prediction_results.py -v
"""

import os
import json
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
import h5py


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="torchlimix_test_pred_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def pred_tensors():
    """Reusable prediction + ground-truth tensors (n=20, p=3)."""
    torch.manual_seed(0)
    n, p = 20, 3
    return dict(
        pred_mean=torch.randn(n, p, dtype=torch.float64),
        pred_var=torch.abs(torch.randn(n, p, dtype=torch.float64)) + 0.01,
        y_true=torch.randn(n, p, dtype=torch.float64),
        C0=torch.eye(p, dtype=torch.float64) * 0.4,
        C1=torch.eye(p, dtype=torch.float64) * 0.6,
    )


@pytest.fixture
def h5_geno_path(tmp_dir):
    """Create a small HDF5 genotype file for loader tests."""
    n_samples, n_snps = 5, 100
    rng = np.random.RandomState(42)

    geno = rng.randint(0, 3, size=(n_samples, n_snps)).astype(np.float64)
    fid = np.arange(90001, 90001 + n_samples)
    iid = fid.copy()

    path = os.path.join(tmp_dir, "test_genotypes.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("genotypes", data=geno)
        f.create_dataset("fid", data=fid)
        f.create_dataset("iid", data=iid)
    return path


@pytest.fixture
def npz_geno_path(tmp_dir):
    """Create a small NPZ genotype file for loader tests."""
    n_samples, n_snps = 5, 100
    rng = np.random.RandomState(42)

    geno = rng.randint(0, 3, size=(n_samples, n_snps)).astype(np.float64)
    fid = np.arange(90001, 90001 + n_samples)
    iid = fid.copy()

    path = os.path.join(tmp_dir, "test_genotypes.npz")
    np.savez(path, genotypes=geno, fid=fid, iid=iid)
    return path

def _make_store(tmp_dir, **kwargs):
    from torchlimix.result_factory._store_prediction_results import (
        PredictionResultStore,
    )

    defaults = dict(
        output_dir=tmp_dir,
        uid="test_pred",
        method="Analytical_BLUP",
        n_traits=3,
    )
    defaults.update(kwargs)
    return PredictionResultStore(**defaults)


def _path(store, filename):
    """Construct an output path inside the store's base_dir."""
    return os.path.join(store.base_dir, filename)

class TestDirectoryStructure:

    def test_real_data_uses_output_dir(self, tmp_dir):
        store = _make_store(tmp_dir)
        assert store.base_dir == tmp_dir

    def test_simulation_eta_folder(self, tmp_dir):
        store = _make_store(tmp_dir, rep_idx=3, eta=0.5)
        assert "eta0.50" in store.base_dir
        assert "rep0003" in store.base_dir

    def test_heterogeneity_corr_folder(self, tmp_dir):
        store = _make_store(
            tmp_dir, rep_idx=1, corr_bounds=2, use_heterogeneity=True
        )
        assert "corr2" in store.base_dir
        assert "rep0001" in store.base_dir

    def test_preprocessing_json_created_at_init(self, tmp_dir):
        store = _make_store(tmp_dir)
        assert os.path.exists(_path(store, "data_preprocessing.json"))

class TestScenarioA:
    """Internal train/test split with known y_true."""

    def test_store_sets_ground_truth_flag(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        assert store._has_ground_truth is True

    def test_predictions_summary_includes_y_true(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        assert "y_true_summary" in store.predictions
        assert store.predictions["has_ground_truth"] is True

    def test_ci_coverage_computed(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        assert "coverage" in store.confidence_intervals
        cov = store.confidence_intervals["coverage"]
        assert 0.0 <= cov["overall"] <= 1.0
        assert len(cov["per_trait"]) == 3

    def test_covariance_info(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            C0=pred_tensors["C0"],
            C1=pred_tensors["C1"],
            y_true=pred_tensors["y_true"],
        )
        cov = store.predictions["covariance"]
        assert cov["C0_trace"] == pytest.approx(1.2)
        assert len(cov["heritability_per_trait"]) == 3

    def test_save_creates_all_files(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        store.compute_metrics(pred_tensors["pred_mean"], pred_tensors["y_true"])
        store.save()

        assert os.path.exists(_path(store, "summary.json"))
        assert os.path.exists(_path(store, "metrics.json"))
        assert os.path.exists(_path(store, "tensors.npz"))

    def test_metrics_json_has_overall(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        store.compute_metrics(pred_tensors["pred_mean"], pred_tensors["y_true"])
        store.save()

        with open(_path(store, "metrics.json")) as f:
            data = json.load(f)
        assert "overall" in data
        assert "per_trait" in data

    def test_summary_json_includes_metrics(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        store.compute_metrics(pred_tensors["pred_mean"], pred_tensors["y_true"])
        store.save()

        with open(_path(store, "summary.json")) as f:
            data = json.load(f)
        assert "metrics_summary" in data
        assert "overall" in data["metrics_summary"]

    def test_csv_includes_true_column(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        store.export_predictions_csv()

        df = pd.read_csv(_path(store, "predicted_phenotypes.csv"), index_col=0)
        assert "Trait_0_Pred" in df.columns
        assert "Trait_0_Std" in df.columns
        assert "Trait_0_True" in df.columns

class TestScenarioB:
    """External genotype file, y_true is None."""

    def test_store_without_y_true(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        assert store._has_ground_truth is False

    def test_store_with_all_nan_y_true(self, tmp_dir):
        store = _make_store(tmp_dir)
        y_nan = torch.full((10, 3), float("nan"))
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
            y_true=y_nan,
        )
        assert store._has_ground_truth is False

    def test_no_y_true_summary(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        assert "y_true_summary" not in store.predictions
        assert store.predictions["has_ground_truth"] is False

    def test_no_coverage(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        assert "coverage" not in store.confidence_intervals

    def test_ci_width_still_computed(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        assert "ci_width_mean" in store.confidence_intervals
        assert store.confidence_intervals["ci_width_mean"] > 0

    def test_save_skips_metrics_and_tensors(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        store.save(save_tensors=True)  # flag ignored for scenario B

        assert os.path.exists(_path(store, "summary.json"))
        assert not os.path.exists(_path(store, "metrics.json"))

    def test_summary_json_no_metrics_section(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        store.save()

        with open(_path(store, "summary.json")) as f:
            data = json.load(f)
        assert "metrics_summary" not in data
        assert data["metadata"]["has_ground_truth"] is False

    def test_csv_no_true_column(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )
        store.export_predictions_csv()

        df = pd.read_csv(_path(store, "predicted_phenotypes.csv"), index_col=0)
        assert "Trait_0_Pred" in df.columns
        assert "Trait_0_Std" in df.columns
        assert "Trait_0_True" not in df.columns

    def test_csv_with_sample_index(self, tmp_dir):
        n = 5
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(n, 3),
            torch.abs(torch.randn(n, 3)) + 0.01,
        )
        fid = np.arange(90001, 90001 + n)
        idx = pd.MultiIndex.from_arrays(
            [fid, fid.copy()], names=["fid", "iid"]
        )
        store.export_predictions_csv(sample_index=idx)

        df = pd.read_csv(_path(store, "predicted_phenotypes.csv"))
        assert "fid" in df.columns
        assert df["fid"].iloc[0] == 90001
class TestMetrics:

    def test_perfect_prediction(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=2)
        y = torch.randn(50, 2)

        metrics = store.compute_metrics(y.clone(), y)

        assert metrics["overall"]["mse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["overall"]["correlation_mean"] == pytest.approx(
            1.0, abs=1e-6
        )
        assert metrics["overall"]["r2_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_random_prediction_low_correlation(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=3)
        torch.manual_seed(42)
        y = torch.randn(100, 3)
        p = torch.randn(100, 3)

        metrics = store.compute_metrics(p, y)

        assert metrics["n_test"] == 100
        assert metrics["n_traits"] == 3
        assert metrics["overall"]["mse"] > 0
        assert abs(metrics["overall"]["correlation_mean"]) < 0.5

    def test_numpy_input_accepted(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=2)
        rng = np.random.RandomState(0)
        y = rng.randn(30, 2)
        p = y + rng.randn(30, 2) * 0.1

        metrics = store.compute_metrics(p, y)
        assert metrics["overall"]["correlation_mean"] > 0.8

    def test_per_trait_keys(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=3)
        y = torch.randn(50, 3)
        p = y + torch.randn(50, 3) * 0.5

        metrics = store.compute_metrics(p, y)

        for key in ["mse", "mae", "rmse", "nrmse", "correlation", "r2"]:
            assert key in metrics["per_trait"]["trait_0"]

        assert len(metrics["mse_per_trait"]) == 3
        assert len(metrics["correlation_per_trait"]) == 3
        assert len(metrics["r2_per_trait"]) == 3

class TestCoverage:

    def test_full_coverage(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=2)
        y = torch.zeros(20, 2)
        lo = torch.full((20, 2), -10.0)
        hi = torch.full((20, 2), 10.0)

        cov = store._compute_coverage(y, lo, hi)
        assert cov["overall"] == 1.0

    def test_zero_coverage(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=2)
        y = torch.full((20, 2), 100.0)
        lo = torch.zeros(20, 2)
        hi = torch.ones(20, 2)

        cov = store._compute_coverage(y, lo, hi)
        assert cov["overall"] == 0.0

    def test_per_trait_coverage(self, tmp_dir):
        store = _make_store(tmp_dir, n_traits=2)
        y = torch.tensor([[0.5, 100.0]] * 10)
        lo = torch.zeros(10, 2)
        hi = torch.ones(10, 2)

        cov = store._compute_coverage(y, lo, hi)
        assert cov["per_trait"][0] == 1.0  # 0.5 in [0, 1]
        assert cov["per_trait"][1] == 0.0  # 100 not in [0, 1]

class TestGetResultsDict:

    def test_scenario_a_returns_tensors(self, tmp_dir, pred_tensors):
        store = _make_store(tmp_dir)
        store.store_predictions(
            pred_tensors["pred_mean"],
            pred_tensors["pred_var"],
            y_true=pred_tensors["y_true"],
        )
        store.compute_metrics(pred_tensors["pred_mean"], pred_tensors["y_true"])

        result = store.get_results_dict()
        assert isinstance(result["mean"], torch.Tensor)
        assert result["mean"].shape == (20, 3)
        assert isinstance(result["y_true"], torch.Tensor)
        assert "metrics" in result

    def test_scenario_b_y_true_is_none(self, tmp_dir):
        store = _make_store(tmp_dir)
        store.store_predictions(
            torch.randn(10, 3),
            torch.abs(torch.randn(10, 3)) + 0.01,
        )

        result = store.get_results_dict()
        assert isinstance(result["mean"], torch.Tensor)
        assert result["y_true"] is None
        assert result["metrics"] == {}

class TestGenotypeLoading:
    """Test that external genotype files load correctly via
    load_prediction_genotypes / _load_genotype_file."""

    def test_load_hdf5(self, h5_geno_path):
        from torchlimix.utils.data_loader import load_prediction_genotypes

        geno, index = load_prediction_genotypes(h5_geno_path)

        assert geno.shape == (5, 100)
        assert geno.dtype == np.float64
        assert len(index) == 5
        assert list(index.names) == ["fid", "iid"]
        assert index[0] == (90001, 90001)

    def test_load_npz(self, npz_geno_path):
        from torchlimix.utils.data_loader import load_prediction_genotypes

        geno, index = load_prediction_genotypes(npz_geno_path)

        assert geno.shape == (5, 100)
        assert geno.dtype == np.float64
        assert index[0] == (90001, 90001)

    def test_hdf5_no_ids_autogenerated(self, tmp_dir):
        """HDF5 without fid/iid should auto-generate 1-based IDs."""
        from torchlimix.utils.data_loader import load_prediction_genotypes

        path = os.path.join(tmp_dir, "no_ids.h5")
        rng = np.random.RandomState(0)
        with h5py.File(path, "w") as f:
            f.create_dataset("genotypes", data=rng.randn(3, 50))

        geno, index = load_prediction_genotypes(path)
        assert geno.shape == (3, 50)
        assert index[0] == (1, 1)
        assert index[2] == (3, 3)

    def test_npz_no_ids_autogenerated(self, tmp_dir):
        from torchlimix.utils.data_loader import load_prediction_genotypes

        path = os.path.join(tmp_dir, "no_ids.npz")
        rng = np.random.RandomState(0)
        np.savez(path, genotypes=rng.randn(4, 60))

        with pytest.warns(UserWarning, match="auto-generating IDs"):
            geno, index = load_prediction_genotypes(path)
        assert geno.shape == (4, 60)
        assert index[0] == (1, 1)

    def test_nan_imputation(self, tmp_dir):
        from torchlimix.utils.data_loader import load_prediction_genotypes

        geno_arr = np.array(
            [[0.0, 1.0, 2.0], [1.0, np.nan, 0.0], [2.0, 1.0, np.nan]]
        )
        path = os.path.join(tmp_dir, "with_nan.npz")
        np.savez(path, genotypes=geno_arr)

        with pytest.warns(UserWarning, match="auto-generating IDs"):
            geno, _ = load_prediction_genotypes(path)
        assert not np.isnan(geno).any()
        assert geno[1, 1] == pytest.approx(1.0)
        assert geno[2, 2] == pytest.approx(1.0)

    def test_csv_genotype_loading(self, tmp_dir):
        from torchlimix.utils.data_loader import load_prediction_genotypes

        df = pd.DataFrame({
            "fid": [1, 2, 3], "iid": [1, 2, 3],
            "snp_0": [0.0, 1.0, 2.0], "snp_1": [1.0, 0.0, 1.0],
        })
        path = os.path.join(tmp_dir, "geno.csv")
        df.to_csv(path, index=False)

        with pytest.warns(UserWarning, match="slow for wide"):
            geno, index = load_prediction_genotypes(path)
        assert geno.shape == (3, 2)
        assert index[0] == (1, 1)

    def test_package_h5_file_loadable(self):
        """Smoke test: load the bundled test genotype file if it exists."""
        from torchlimix.utils.data_loader import load_prediction_genotypes
        from importlib import resources

        try:
            pkg_data = resources.files("torchlimix") / "data"
            h5_path = str(pkg_data / "test_prediction_genotypes.h5")
        except (TypeError, FileNotFoundError):
            pytest.skip("Package data directory not found")

        if not os.path.exists(h5_path):
            pytest.skip(f"Test file not present: {h5_path}")

        geno, index = load_prediction_genotypes(h5_path)
        assert geno.ndim == 2
        assert geno.dtype == np.float64
        assert len(index) == geno.shape[0]
        assert list(index.names) == ["fid", "iid"]


class TestGenotypeFileDispatcher:
    """Test the unified _load_genotype_file dispatcher."""

    def test_npz_rejected_for_training(self, npz_geno_path):
        from torchlimix.utils.data_loader import _load_genotype_file

        with pytest.raises(ValueError, match="prediction genotypes"):
            _load_genotype_file(npz_geno_path, allow_npz=False)

    def test_npz_allowed_for_prediction(self, npz_geno_path):
        from torchlimix.utils.data_loader import _load_genotype_file

        geno, index, bim = _load_genotype_file(
            npz_geno_path, allow_npz=True
        )
        assert geno.shape == (5, 100)
        assert bim is None

    def test_hdf5_returns_none_bim(self, h5_geno_path):
        from torchlimix.utils.data_loader import _load_genotype_file

        geno, index, bim = _load_genotype_file(h5_geno_path)
        assert bim is None
        assert geno.shape[0] == len(index)