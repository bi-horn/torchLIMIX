"""
Tests for VarDecResults: ground truth, covariances, vardec output, error computation.

Usage:
    pytest tests/test_vardec_results.py -v
"""

import os
import json
import pickle
import tempfile
import shutil
import pytest
import numpy as np
import torch


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="torchlimix_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def base_config():
    return {"data_param": {"dset": "test_dataset"}, "verbose": False}


def _make_vardec(tmp_dir, config, **kwargs):
    from torchlimix.result_factory._store_vardec_results import VarDecResults
    defaults = dict(
        config=config, output_dir=tmp_dir,
        scenario_id=0, rep_idx=0, uid="test", rank=3,
    )
    defaults.update(kwargs)
    return defaults.pop('config'), VarDecResults(config=config, **defaults)

class TestVarDecDirectoryStructure:

    def test_simulation_dir(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config, scenario_id=2, rep_idx=5)
        assert "vardec2" in str(vdr.base_dir)
        assert "rep0005" in str(vdr.base_dir)
        assert os.path.isdir(vdr.base_dir)

    def test_real_data_dir(self, tmp_dir, base_config):
        from torchlimix.result_factory._store_vardec_results import VarDecResults
        vdr = VarDecResults(
            config=base_config, output_dir=tmp_dir,
            rep_idx=None, scenario_id=None, uid="real_test", rank=3,
        )
        assert str(vdr.base_dir) == tmp_dir

    def test_scenario_required_with_rep_idx(self, tmp_dir, base_config):
        from torchlimix.result_factory._store_vardec_results import VarDecResults
        with pytest.raises(ValueError, match="scenario_id required"):
            VarDecResults(
                config=base_config, output_dir=tmp_dir,
                scenario_id=None, rep_idx=0, uid="test", rank=3,
            )

class TestGroundTruth:

    def _sample_gt(self, P=3):
        return {
            "scenario_id": 0, "n_samples": 200, "n_traits": P,
            "persistent_prop": 0.3, "heterogeneity_prop": 0.2, "noise_prop": 0.5,
            "var_G": 0.3, "var_het": 0.2, "var_noise": 0.5,
            "var_G_achieved": 0.28, "var_het_achieved": 0.22, "var_noise_achieved": 0.50,
            "var_Y_achieved": 1.0,
            "var_G_per_trait": [0.28] * P,
            "var_het_per_trait": [0.22] * P,
            "var_noise_per_trait": [0.50] * P,
            "var_Y_per_trait": [1.0] * P,
            "C_G": np.eye(P).tolist(), "C_het": np.eye(P).tolist(),
            "C0": (np.eye(P) * 0.5).tolist(), "C1": (np.eye(P) * 0.5).tolist(),
        }

    def test_store_ground_truth(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        gt = self._sample_gt()
        vdr.add_ground_truth(gt)

        assert vdr.ground_truth is not None
        assert vdr.ground_truth["n_traits"] == 3
        assert vdr.ground_truth["shared_prop"] == 0.3

    def test_achieved_proportions_computed(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        gt = self._sample_gt()
        vdr.add_ground_truth(gt)

        total = 0.28 + 0.22 + 0.50
        expected_prop = 0.28 / total
        assert abs(vdr.ground_truth["prop_shared_achieved"] - expected_prop) < 1e-6

class TestFittedCovariances:

    def test_store_numpy(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        C0 = np.eye(3) * 0.4
        C1 = np.eye(3) * 0.6
        vdr.add_fitted_covariances(C0, C1)

        assert vdr.fitted_covariances is not None
        assert vdr.fitted_covariances["success"] is True
        assert abs(vdr.fitted_covariances["C0_trace"] - 1.2) < 1e-10

    def test_store_torch(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        C0 = torch.eye(3) * 0.4
        C1 = torch.eye(3) * 0.6
        vdr.add_fitted_covariances(C0, C1)

        assert vdr.fitted_covariances["success"] is True
        assert len(vdr.fitted_covariances["C0_eigenvals"]) == 3


class TestVarDecOutput:

    def _sample_vardec(self, P=3):
        return {
            "failed": False,
            "overall": {
                "var_shared": 0.30, "var_het": 0.20, "var_noise": 0.50,
                "var_total": 1.0,
                "pct_shared": 30.0, "pct_het": 20.0, "pct_noise": 50.0,
                "trace_C0": 0.5, "trace_C1": 0.5, "mean_offdiag_C0": 0.01,
            },
            "per_trait": {
                "h2": [0.5] * P, "h2_mean": 0.5, "h2_median": 0.5,
                "h2_min": 0.5, "h2_max": 0.5,
                "genetic": [0.5] * P, "noise": [0.5] * P, "total": [1.0] * P,
            },
            "rg": {
                "matrix": np.eye(P).tolist(),
                "mean": 0.0, "median": 0.0, "min": 0.0, "max": 1.0,
            },
        }

    def test_store_results(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        vdr.add_vardec_results(self._sample_vardec())

        assert vdr.vardec_results is not None
        assert vdr.vardec_results["var_shared"] == 0.30
        assert vdr.vardec_results["h2_mean"] == 0.5

    def test_failed_results(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        vdr.add_vardec_results({"failed": True})

        assert vdr.vardec_results["failed"] is True
        assert np.isnan(vdr.vardec_results["var_shared"])

    def test_none_results(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        vdr.add_vardec_results(None)

        assert vdr.vardec_results["failed"] is True

    def test_errors_computed_with_ground_truth(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        gt = {
            "n_traits": 3,
            "persistent_prop": 0.3, "heterogeneity_prop": 0.2, "noise_prop": 0.5,
            "var_G": 0.3, "var_het": 0.2, "var_noise": 0.5,
            "var_G_achieved": 0.30, "var_het_achieved": 0.20, "var_noise_achieved": 0.50,
            "var_Y_achieved": 1.0,
            "var_G_per_trait": [0.30, 0.30, 0.30],
            "var_het_per_trait": [0.20, 0.20, 0.20],
            "var_noise_per_trait": [0.50, 0.50, 0.50],
            "var_Y_per_trait": [1.0, 1.0, 1.0],
            "C_G": np.eye(3).tolist(), "C_het": np.eye(3).tolist(),
            "C0": (np.eye(3) * 0.5).tolist(), "C1": (np.eye(3) * 0.5).tolist(),
        }
        vdr.add_ground_truth(gt)
        vdr.add_vardec_results(self._sample_vardec())

        assert "errors" in vdr.vardec_results
        errors = vdr.vardec_results["errors"]
        assert "mae" in errors
        assert "rmse" in errors

class TestSaveLoad:

    def test_save_creates_files(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)
        vdr.add_fitted_covariances(np.eye(3), np.eye(3))
        vdr.add_vardec_results({
            "failed": False,
            "overall": {"var_shared": 0.3, "var_het": 0.2, "var_noise": 0.5,
                        "var_total": 1.0, "pct_shared": 30, "pct_het": 20,
                        "pct_noise": 50, "trace_C0": 0.5, "trace_C1": 0.5,
                        "mean_offdiag_C0": 0.0},
            "per_trait": {"h2": [0.5], "h2_mean": 0.5, "h2_median": 0.5,
                          "h2_min": 0.5, "h2_max": 0.5,
                          "genetic": [0.5], "noise": [0.5], "total": [1.0]},
            "rg": {"matrix": [[1.0]], "mean": 0.0, "median": 0.0,
                   "min": 0.0, "max": 1.0},
        })
        vdr.save()

        assert os.path.exists(os.path.join(vdr.base_dir, "vardec_results.json"))
        assert os.path.exists(os.path.join(vdr.base_dir, "covariances.pkl"))

    def test_load_roundtrip(self, tmp_dir, base_config):
        _, vdr = _make_vardec(tmp_dir, base_config)

        gt = {
            "n_traits": 2, "scenario_id": 0, "n_samples": 100,
            "persistent_prop": 0.5, "heterogeneity_prop": 0.0, "noise_prop": 0.5,
            "var_G": 0.5, "var_het": 0.0, "var_noise": 0.5,
            "var_G_achieved": 0.48, "var_het_achieved": 0.02, "var_noise_achieved": 0.50,
            "var_Y_achieved": 1.0,
            "var_G_per_trait": [0.48, 0.48], "var_het_per_trait": [0.02, 0.02],
            "var_noise_per_trait": [0.50, 0.50], "var_Y_per_trait": [1.0, 1.0],
            "C_G": None, "C_het": None, "C0": None, "C1": None,
        }
        vdr.add_ground_truth(gt)
        vdr.save()

        from torchlimix.result_factory._store_vardec_results import VarDecResults
        loaded = VarDecResults.load(tmp_dir, scenario_id=0, rep_idx=0)
        assert loaded.ground_truth is not None
        assert loaded.ground_truth["n_traits"] == 2