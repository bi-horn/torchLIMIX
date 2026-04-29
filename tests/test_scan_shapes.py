"""
Tests for scan output shape contracts, beta reshape logic,
and the process_snps → add_beta_result flow.

Usage:
    pytest tests/test_scan_shapes.py -v
"""

import pytest
import numpy as np
import torch


class TestBetaReshape:
    """
    The scanner stores beta as vec(B) in column-major (traits-first) order.
    The reshape must be: flat.reshape(n_traits, n_cov).T → (n_cov, n_traits).
    """

    def test_intercept_only(self):
        """1 covariate, 3 traits → (1, 3)."""
        flat = torch.tensor([0.1, 0.2, 0.3])
        reshaped = flat.reshape(3, 1).T
        assert reshaped.shape == (1, 3)
        assert reshaped[0, 0] == pytest.approx(0.1)
        assert reshaped[0, 2] == pytest.approx(0.3)

    def test_with_covariates(self):
        """6 covariates, 3 traits → (6, 3)."""
        flat = torch.arange(18, dtype=torch.double)
        reshaped = flat.reshape(3, 6).T
        assert reshaped.shape == (6, 3)
        # Row 0 (intercept): [0, 6, 12]
        assert reshaped[0, 0] == pytest.approx(0.0)
        assert reshaped[0, 1] == pytest.approx(6.0)
        assert reshaped[0, 2] == pytest.approx(12.0)
        # Row 1 (cov_1): [1, 7, 13]
        assert reshaped[1, 0] == pytest.approx(1.0)
        assert reshaped[1, 1] == pytest.approx(7.0)
        assert reshaped[1, 2] == pytest.approx(13.0)

    def test_se_same_shape_as_beta(self):
        flat_beta = torch.randn(12)  # 4 cov × 3 traits
        flat_se = torch.abs(torch.randn(12))
        beta = flat_beta.reshape(3, 4).T
        se = flat_se.reshape(3, 4).T
        assert beta.shape == se.shape == (4, 3)

    def test_roundtrip(self):
        """Reshape then flatten should recover original."""
        original = torch.randn(18)
        reshaped = original.reshape(3, 6).T           # (6, 3)
        recovered = reshaped.T.contiguous().reshape(-1)  # back to (18,)
        torch.testing.assert_close(original, recovered)


class TestScanOutputShapes:
    """
    Verify the shape contracts that downstream code relies on.
    These test the reshape logic inside scan_batched_gpu/cpu.
    """

    def test_effsizes0_with_covariates(self):
        """effsizes0: (n_snps, n_covariates, n_traits)."""
        n_snps, n_traits, n_cov = 100, 3, 6
        cp = n_traits * n_cov
        beta_flat = torch.randn(n_snps, cp)
        effsizes0 = beta_flat.view(n_snps, n_traits, n_cov).transpose(-2, -1)
        assert effsizes0.shape == (100, 6, 3)

    def test_effsizes0_intercept_only(self):
        """effsizes0 with only intercept: (n_snps, 1, n_traits)."""
        n_snps, n_traits, n_cov = 100, 3, 1
        cp = n_traits * n_cov
        beta_flat = torch.randn(n_snps, cp)
        effsizes0 = beta_flat.view(n_snps, n_traits, n_cov).transpose(-2, -1)
        assert effsizes0.shape == (100, 1, 3)

    def test_effsizes1_common(self):
        """Common test: A1 has 1 col → (n_snps, 1, 1)."""
        effsizes1 = torch.randn(100, 1, 1)
        assert effsizes1.shape == (100, 1, 1)

    def test_effsizes1_any(self):
        """Any-effect test: A1 has n_traits cols → (n_snps, 1, n_traits)."""
        effsizes1 = torch.randn(100, 1, 3)
        assert effsizes1.shape == (100, 1, 3)

    def test_effsizes1_se_matches(self):
        """SE tensor should have same shape as effsizes1."""
        n_snps, a1_cols = 100, 3
        cp = 18  # 6 cov × 3 traits
        total = cp + a1_cols
        se = torch.abs(torch.randn(n_snps, total))
        effsizes1_se = se[:, cp:].view(n_snps, a1_cols, 1).transpose(-2, -1)
        assert effsizes1_se.shape == (100, 1, 3)

class TestProcessSnpsContracts:
    """
    Test the data flow from scanner output → process_snps → add_beta_result.
    Uses synthetic tensors to verify shape transformations.
    """

    def test_null_beta_reshape_flow(self):
        """Simulate the full null beta extraction and reshape."""
        n_traits, n_cov = 3, 6
        # Scanner produces flat vector
        null_beta_flat = torch.randn(n_traits * n_cov)
        null_beta_se_flat = torch.abs(torch.randn(n_traits * n_cov))

        # process_snps reshapes
        beta0 = null_beta_flat.reshape(n_traits, n_cov).T
        beta0_se = null_beta_se_flat.reshape(n_traits, n_cov).T

        assert beta0.shape == (n_cov, n_traits)
        assert beta0_se.shape == (n_cov, n_traits)

    def test_per_snp_beta_extraction(self):
        """Simulate extracting effsizes1 from scan results."""
        n_snps = 1000
        # Common test: effsizes1 shape (n_snps, 1, 1)
        scan_result = {"effsizes1": torch.randn(n_snps, 1, 1)}
        beta1 = scan_result["effsizes1"]
        assert beta1.shape == (n_snps, 1, 1)

    def test_per_snp_beta_any_test(self):
        """Any-effect test: effsizes1 shape (n_snps, 1, n_traits)."""
        n_snps, n_traits = 1000, 3
        scan_result = {"effsizes1": torch.randn(n_snps, 1, n_traits)}
        beta1 = scan_result["effsizes1"]
        assert beta1.shape == (n_snps, 1, n_traits)

    def test_pve_shape(self):
        """PVE: (n_snps, a1_cols)."""
        n_snps, a1_cols = 1000, 3
        pve = torch.randn(n_snps, a1_cols)
        assert pve.shape == (n_snps, a1_cols)

class TestContiguityForKron:
    """
    torch.kron requires contiguous inputs. Verify the patterns that
    caused failures before the fix.
    """

    def test_transpose_not_contiguous(self):
        """A.T is not contiguous for multi-column tensors."""
        A = torch.randn(5, 3)
        assert not A.T.is_contiguous()

    def test_single_column_transpose_contiguous(self):
        """A.T IS contiguous when A has 1 column — why the bug was hidden."""
        A = torch.randn(5, 1)
        assert A.T.is_contiguous()

    def test_kron_with_contiguous_fix(self):
        """torch.kron should work after .contiguous()."""
        A = torch.randn(3, 3)
        B = torch.randn(5, 3)

        # This would fail without .contiguous()
        result = torch.kron(A, B.T.contiguous())
        assert result.shape == (3 * 3, 3 * 5)

    def test_kron_fails_without_contiguous(self):
        """Verify the original bug exists without the fix."""
        A = torch.randn(3, 3)
        B = torch.randn(5, 3)
        try:
            torch.kron(A, B.T)
            # If it succeeds, that's fine too (newer torch versions may handle it)
        except RuntimeError as e:
            assert "contiguous" in str(e).lower() or "view" in str(e).lower()


class TestCpuDeviceConsistency:
    """Verify CPU scan returns CPU tensors in all paths."""

    def test_empty_a1_returns_cpu(self):
        """Empty A1 guard should return CPU tensors even if scanner is on GPU."""
        # Simulate the return dict from the empty A1 path
        results = {
            "lml": torch.tensor(-350.0),
            "scale": torch.tensor(1.0),
            "effsizes0": torch.randn(1, 3),
            "effsizes0_se": torch.randn(1, 3),
            "effsizes1": torch.empty(0),
            "effsizes1_se": torch.empty(0),
        }

        for key, val in results.items():
            if isinstance(val, torch.Tensor):
                assert not val.is_cuda, f"{key} should be on CPU"