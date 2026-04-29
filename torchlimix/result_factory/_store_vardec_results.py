import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import torch
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class VarDecResults:
    """
    Storage for variance decomposition results.

    Handles both simulation (with ground truth + error computation) and
    real data workflows. Expects vardec output from VarDecMultiTrait.get_results()
    which returns nested dict with keys: overall, per_trait, rg, failed.

    Directory layout:
        simulation:  output_dir/vardec{scenario_id}/rep{rep_idx:04d}/
        real data:   output_dir/vardec_results/
    """

    _W = 62  # column width for formatted output

    def __init__(self, config, output_dir, scenario_id=None, rep_idx=None,
                 uid=None, rank=None, correction_metadata=None, phenotype_data=None):
        self.config = config
        self.scenario_id = scenario_id
        self.rep_idx = rep_idx
        self.uid = uid
        self.rank = rank
        self.verbose = config.get("verbose", False)
        self.correction_metadata = correction_metadata or {}
        self.phenotype_data = phenotype_data or {}

        if rep_idx is not None:
            if scenario_id is None:
                raise ValueError("scenario_id required when rep_idx is provided")
            base_dir = os.path.join(output_dir, f"vardec{scenario_id}", f"rep{rep_idx:04d}")
        else:
            base_dir = output_dir

        self.base_dir = Path(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        self.ground_truth = None
        self.fitted_covariances = None
        self.vardec_results = None
        self.optimization_metrics = None

        # save preprocessing info for real data only
        if self.rep_idx is None:
            self._save_data_preprocessing_info()
            self._save_phenotype_data()

        if self.verbose:
            tag = (f"scenario {scenario_id}, rep {rep_idx:04d}"
                   if rep_idx is not None else f"uid {uid}")
            print(f"  [VarDecResults] Initialized for {tag}, rank {rank}")
            print(f"    Output → {self.base_dir}")

    def _path(self, filename):
        return os.path.join(self.base_dir, filename)

    @property
    def _paths(self):
        return {
            "ground_truth_json": self._path("ground_truth.json"),
            "ground_truth_pkl": self._path("ground_truth.pkl"),
            "covariances_pkl": self._path("covariances.pkl"),
            "covariances_json": self._path("covariances.json"),
            "vardec_json": self._path("vardec_results.json"),
            "optimization": self._path("optimization_stats.jsonl"),
        }

    def add_ground_truth(self, gt):
        """
        Store ground truth from simulation.

        Parameters
        ----------
        gt : dict
            Ground truth from simulator, expected keys include persistent_prop,
            heterogeneity_prop, noise_prop, var_G, var_het, var_noise,
            and per-trait variants.
        """
        P = gt.get("n_traits", 4)

        self.ground_truth = {
            "scenario_id": gt.get("scenario_id"),
            "timestamp": datetime.now().isoformat(),
            "rank": self.rank,
            "n_samples": gt.get("n_samples"),
            "n_traits": P,

            # target proportions
            "shared_prop": gt.get("persistent_prop", 0),
            "heterogeneity_prop": gt.get("heterogeneity_prop", 0),
            "noise_prop": gt.get("noise_prop", 0),

            # target variances (per-trait mean)
            "var_G": gt.get("var_G", 0),
            "var_het": gt.get("var_het", 0),
            "var_noise": gt.get("var_noise", 0),

            # achieved (empirical)
            "var_G_achieved": gt.get("var_G_achieved", 0),
            "var_het_achieved": gt.get("var_het_achieved", 0),
            "var_noise_achieved": gt.get("var_noise_achieved", 0),
            "var_Y_achieved": gt.get("var_Y_achieved", 0),

            # per-trait
            "var_G_per_trait": self._to_list(gt.get("var_G_per_trait")),
            "var_het_per_trait": self._to_list(gt.get("var_het_per_trait")),
            "var_noise_per_trait": self._to_list(gt.get("var_noise_per_trait")),
            "var_Y_per_trait": self._to_list(gt.get("var_Y_per_trait")),

            # covariance matrices
            "C_G": self._to_list(gt.get("C_G")),
            "C_het": self._to_list(gt.get("C_het")),
            "C0": self._to_list(gt.get("C0")),
            "C1": self._to_list(gt.get("C1")),
        }

        # achieved proportions
        vt = (self.ground_truth["var_G_achieved"]
              + self.ground_truth["var_het_achieved"]
              + self.ground_truth["var_noise_achieved"])
        self.ground_truth["var_total_achieved"] = vt

        if vt > 0:
            self.ground_truth["prop_shared_achieved"] = self.ground_truth["var_G_achieved"] / vt
            self.ground_truth["prop_het_achieved"] = self.ground_truth["var_het_achieved"] / vt
            self.ground_truth["prop_noise_achieved"] = self.ground_truth["var_noise_achieved"] / vt
        else:
            self.ground_truth["prop_shared_achieved"] = 0
            self.ground_truth["prop_het_achieved"] = 0
            self.ground_truth["prop_noise_achieved"] = 0

        if self.verbose:
            print(f"  Ground truth stored (P = {P})")

    def add_fitted_covariances(self, C0, C1):
        """Store fitted C0, C1 from the Kronecker model."""
        C0 = self._as_numpy(C0)
        C1 = self._as_numpy(C1)

        self.fitted_covariances = {
            "C0": C0,
            "C1": C1,
            "success": True,
            "errors": [],
        }

        try:
            self.fitted_covariances["C0_trace"] = float(np.trace(C0))
            self.fitted_covariances["C1_trace"] = float(np.trace(C1))
            self.fitted_covariances["C0_eigenvals"] = np.linalg.eigvalsh(C0)[::-1].tolist()
            self.fitted_covariances["C1_eigenvals"] = np.linalg.eigvalsh(C1)[::-1].tolist()
        except Exception as e:
            self.fitted_covariances["success"] = False
            self.fitted_covariances["errors"].append(str(e))

        if self.verbose:
            tr0 = self.fitted_covariances.get("C0_trace", "?")
            tr1 = self.fitted_covariances.get("C1_trace", "?")
            print(f"  Fitted covariances stored: tr(C0) = {tr0:.4f}, tr(C1) = {tr1:.4f}")

    def add_vardec_results(self, vardec_results, use_achieved=True):
        """
        Store variance decomposition output from VarDecMultiTrait.get_results().

        Parameters
        ----------
        vardec_results : dict
            Output of VarDecMultiTrait.get_results().
        use_achieved : bool
            If ground truth available, compare against achieved proportions.
        """
        if vardec_results is None or vardec_results.get("failed", False):
            self._store_failed()
            return

        overall = vardec_results.get("overall", {})
        per_trait = vardec_results.get("per_trait", {})
        rg = vardec_results.get("rg", {})

        self.vardec_results = {
            "dataset": self.config.get("data_param", {}).get("dset", "unknown"),
            "rank": self.rank,
            "timestamp": datetime.now().isoformat(),

            # overall block model
            "var_shared": overall.get("var_shared"),
            "var_het": overall.get("var_het"),
            "var_noise": overall.get("var_noise"),
            "var_total": overall.get("var_total"),
            "pct_shared": overall.get("pct_shared"),
            "pct_het": overall.get("pct_het"),
            "pct_noise": overall.get("pct_noise"),

            # diagnostics
            "trace_C0": overall.get("trace_C0"),
            "trace_C1": overall.get("trace_C1"),
            "mean_offdiag_C0": overall.get("mean_offdiag_C0"),

            # per-trait heritability
            "h2_per_trait": self._to_list(per_trait.get("h2")),
            "h2_mean": self._safe_float(per_trait.get("h2_mean")),
            "h2_median": self._safe_float(per_trait.get("h2_median")),
            "h2_min": self._safe_float(per_trait.get("h2_min")),
            "h2_max": self._safe_float(per_trait.get("h2_max")),

            # per-trait variances
            "genetic_per_trait": self._to_list(per_trait.get("genetic")),
            "noise_per_trait": self._to_list(per_trait.get("noise")),
            "total_per_trait": self._to_list(per_trait.get("total")),

            # genetic correlations
            "rg_matrix": self._to_list(rg.get("matrix")),
            "rg_mean": self._safe_float(rg.get("mean")),
            "rg_median": self._safe_float(rg.get("median")),
            "rg_min": self._safe_float(rg.get("min")),
            "rg_max": self._safe_float(rg.get("max")),
        }

        if self.scenario_id is not None:
            self.vardec_results["scenario_id"] = self.scenario_id
        if self.rep_idx is not None:
            self.vardec_results["rep_idx"] = self.rep_idx

        # compute errors against ground truth
        if self.ground_truth is not None:
            self.vardec_results["errors"] = self._compute_errors(use_achieved)

    def _store_failed(self):
        self.vardec_results = {
            "failed": True,
            "rank": self.rank,
            "scenario_id": self.scenario_id,
            "rep_idx": self.rep_idx,
            "var_shared": np.nan,
            "var_het": np.nan,
            "var_noise": np.nan,
            "h2_mean": np.nan,
            "rg_mean": np.nan,
            "error_message": "variance decomposition failed",
        }
        if self.verbose:
            print("  [VarDecResults] Decomposition failed (NaN values stored).")

    def _compute_errors(self, use_achieved=True) -> Dict:
        """
        Compute errors between recovered and ground truth variances.

        Returns
        -------
        dict
            Overall errors, per-trait h² errors, per-trait variance errors,
            genetic correlation errors, aggregate metrics, diagnostics.
        """
        gt = self.ground_truth
        vr = self.vardec_results
        P = gt.get("n_traits", 4)

        # reference variances
        if use_achieved:
            ref_type = "achieved"
            var_G_pt = np.array(gt["var_G_per_trait"])
            var_het_pt = np.array(gt["var_het_per_trait"])
            var_noise_pt = np.array(gt["var_noise_per_trait"])

            var_G_ref = np.sum(var_G_pt)
            var_het_ref = np.sum(var_het_pt)
            var_noise_ref = np.sum(var_noise_pt)

            cv_G = self._cv(var_G_pt)
            cv_het = self._cv(var_het_pt)
            cv_noise = self._cv(var_noise_pt)
        else:
            ref_type = "target"
            var_G_ref = gt["var_G"] * P
            var_het_ref = gt["var_het"] * P
            var_noise_ref = gt["var_noise"] * P

            var_G_pt = np.full(P, gt["var_G"])
            var_het_pt = np.full(P, gt["var_het"])
            var_noise_pt = np.full(P, gt["var_noise"])
            cv_G = cv_het = cv_noise = 0.0

        var_total_ref = var_G_ref + var_het_ref + var_noise_ref
        pct_G_ref, pct_het_ref, pct_noise_ref = self._pcts(
            var_G_ref, var_het_ref, var_noise_ref, var_total_ref
        )

        # recovered
        var_shared_rec = vr["var_shared"]
        var_het_rec = vr["var_het"]
        var_noise_rec = vr["var_noise"]
        var_total_rec = vr["var_total"]
        pct_shared_rec = vr["pct_shared"]
        pct_het_rec = vr["pct_het"]
        pct_noise_rec = vr["pct_noise"]

        # overall errors
        err_shared = var_shared_rec - var_G_ref
        err_het = var_het_rec - var_het_ref
        err_noise = var_noise_rec - var_noise_ref

        err_pct_shared = pct_shared_rec - pct_G_ref
        err_pct_het = pct_het_rec - pct_het_ref
        err_pct_noise = pct_noise_rec - pct_noise_ref

        rel_err_shared = self._safe_div(err_shared, var_G_ref)
        rel_err_het = self._safe_div(err_het, var_het_ref)
        rel_err_noise = self._safe_div(err_noise, var_noise_ref)

        errors_pct = np.array([err_pct_shared, err_pct_het, err_pct_noise])
        mae = float(np.mean(np.abs(errors_pct)))
        rmse = float(np.sqrt(np.mean(errors_pct ** 2)))
        max_abs_err = float(np.max(np.abs(errors_pct)))

        result = {
            "reference_type": ref_type,
            "n_traits": P,

            # reference
            "var_G_ref": float(var_G_ref),
            "var_het_ref": float(var_het_ref),
            "var_noise_ref": float(var_noise_ref),
            "var_total_ref": float(var_total_ref),
            "pct_G_ref": float(pct_G_ref),
            "pct_het_ref": float(pct_het_ref),
            "pct_noise_ref": float(pct_noise_ref),

            # recovered
            "var_shared_rec": float(var_shared_rec),
            "var_het_rec": float(var_het_rec),
            "var_noise_rec": float(var_noise_rec),
            "var_total_rec": float(var_total_rec),
            "pct_shared_rec": float(pct_shared_rec),
            "pct_het_rec": float(pct_het_rec),
            "pct_noise_rec": float(pct_noise_rec),

            # overall errors
            "err_shared": float(err_shared),
            "err_het": float(err_het),
            "err_noise": float(err_noise),
            "err_pct_shared": float(err_pct_shared),
            "err_pct_het": float(err_pct_het),
            "err_pct_noise": float(err_pct_noise),
            "rel_err_shared": float(rel_err_shared),
            "rel_err_het": float(rel_err_het),
            "rel_err_noise": float(rel_err_noise),

            # aggregates
            "mae": mae,
            "rmse": rmse,
            "max_abs_err": max_abs_err,

            # heterogeneity diagnostics
            "cv_G": float(cv_G),
            "cv_het": float(cv_het),
            "cv_noise": float(cv_noise),
            "high_heterogeneity": cv_het > 0.2 or cv_noise > 0.2,
            "large_error": max_abs_err > 5.0,
        }

        # per-trait heritability errors
        h2_rec = vr.get("h2_per_trait")
        if h2_rec is not None:
            h2_rec = np.array(h2_rec)
            genetic_ref_pt = var_G_pt + var_het_pt  # diag(C0) = shared + het per trait
            noise_ref_pt = var_noise_pt
            total_ref_pt = genetic_ref_pt + noise_ref_pt
            h2_ref = self._safe_divide_arr(genetic_ref_pt, total_ref_pt)

            h2_err = h2_rec - h2_ref
            result["h2_ref_per_trait"] = h2_ref.tolist()
            result["h2_rec_per_trait"] = h2_rec.tolist()
            result["h2_err_per_trait"] = h2_err.tolist()
            result["h2_mae"] = float(np.mean(np.abs(h2_err)))
            result["h2_rmse"] = float(np.sqrt(np.mean(h2_err ** 2)))
            result["h2_max_abs_err"] = float(np.max(np.abs(h2_err)))
            result["h2_mean_ref"] = float(np.mean(h2_ref))
            result["h2_mean_rec"] = float(np.mean(h2_rec))
            result["h2_mean_err"] = float(np.mean(h2_rec) - np.mean(h2_ref))

        # per-trait variance errors
        genetic_rec = vr.get("genetic_per_trait")
        noise_rec = vr.get("noise_per_trait")
        if genetic_rec is not None and noise_rec is not None:
            genetic_rec = np.array(genetic_rec)
            noise_rec = np.array(noise_rec)
            genetic_ref = var_G_pt + var_het_pt
            noise_ref = var_noise_pt

            genetic_err = genetic_rec - genetic_ref
            noise_err = noise_rec - noise_ref

            result["genetic_ref_per_trait"] = genetic_ref.tolist()
            result["genetic_rec_per_trait"] = genetic_rec.tolist()
            result["genetic_err_per_trait"] = genetic_err.tolist()
            result["genetic_mae"] = float(np.mean(np.abs(genetic_err)))
            result["genetic_rmse"] = float(np.sqrt(np.mean(genetic_err ** 2)))
            result["genetic_max_abs_err"] = float(np.max(np.abs(genetic_err)))
            result["genetic_mean_ref"] = float(np.mean(genetic_ref))
            result["genetic_mean_rec"] = float(np.mean(genetic_rec))
            result["genetic_mean_err"] = float(np.mean(genetic_rec) - np.mean(genetic_ref))

            result["noise_ref_per_trait"] = noise_ref.tolist()
            result["noise_rec_per_trait"] = noise_rec.tolist()
            result["noise_err_per_trait"] = noise_err.tolist()
            result["noise_pt_mae"] = float(np.mean(np.abs(noise_err)))
            result["noise_pt_rmse"] = float(np.sqrt(np.mean(noise_err ** 2)))
            result["noise_pt_max_abs_err"] = float(np.max(np.abs(noise_err)))
            result["noise_pt_mean_ref"] = float(np.mean(noise_ref))
            result["noise_pt_mean_rec"] = float(np.mean(noise_rec))
            result["noise_pt_mean_err"] = float(np.mean(noise_rec) - np.mean(noise_ref))

        # genetic correlation errors
        rg_rec = vr.get("rg_matrix")
        gt_C0 = gt.get("C0")
        if rg_rec is not None and gt_C0 is not None:
            rg_rec = np.array(rg_rec)
            gt_C0 = np.array(gt_C0)
            rg_ref = self._compute_rg(gt_C0)

            if rg_ref is not None:
                triu = np.triu_indices(P, k=1)
                rg_err = rg_rec - rg_ref
                offdiag_err = rg_err[triu]

                result["rg_ref_matrix"] = rg_ref.tolist()
                result["rg_err_matrix"] = rg_err.tolist()
                result["rg_mae"] = float(np.mean(np.abs(offdiag_err)))
                result["rg_rmse"] = float(np.sqrt(np.mean(offdiag_err ** 2)))
                result["rg_max_abs_err"] = float(np.max(np.abs(offdiag_err)))
                result["rg_mean_ref"] = float(np.mean(rg_ref[triu]))
                result["rg_mean_rec"] = float(np.mean(rg_rec[triu]))
                result["rg_mean_err"] = float(np.mean(rg_rec[triu]) - np.mean(rg_ref[triu]))

        return result
    
    def add_optimization_metrics(self, opt_results, model_h0=None):
        if opt_results is None:
            return

        self.optimization_metrics = opt_results.copy()

        record = {
            "rep_idx": int(self.rep_idx) if self.rep_idx is not None else 0,
            "scenario_id": int(self.scenario_id) if self.scenario_id is not None else 0,
            "rank": self.rank,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self._to_serializable(opt_results),
        }

        try:
            with open(self._paths["optimization"], "w") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to save optimization metrics: {e}")

    def save(self):
        paths = self._paths

        if self.ground_truth is not None:
            self._write_json(paths["ground_truth_json"], self.ground_truth)
            self._write_pickle(paths["ground_truth_pkl"], self.ground_truth)

        if self.fitted_covariances is not None:
            self._write_pickle(paths["covariances_pkl"], self.fitted_covariances)
            self._write_json(paths["covariances_json"], self.fitted_covariances, warn_on_fail=True)

        if self.vardec_results is not None:
            self._write_json(paths["vardec_json"], self.vardec_results, warn_on_fail=True)

        if self.verbose:
            print(f"  Results saved → {self.base_dir}")

    @classmethod
    def load(cls, output_dir, scenario_id=None, rep_idx=None, uid=None):
        obj = cls.__new__(cls)
        obj.output_dir = Path(output_dir)
        obj.scenario_id = scenario_id
        obj.rep_idx = rep_idx
        obj.uid = uid
        obj.rank = None
        obj.verbose = False

        if rep_idx is not None:
            base_dir = os.path.join(output_dir, f"vardec{scenario_id}", f"rep{rep_idx:04d}")
        else:
            base_dir = os.path.join(output_dir, uid)

        obj.base_dir = Path(base_dir)
        obj.ground_truth = None
        obj.fitted_covariances = None
        obj.vardec_results = None
        obj.optimization_metrics = None

        gt_pkl = os.path.join(base_dir, "ground_truth.pkl")
        if os.path.exists(gt_pkl):
            with open(gt_pkl, "rb") as f:
                obj.ground_truth = pickle.load(f)
                obj.rank = obj.ground_truth.get("rank")

        cov_pkl = os.path.join(base_dir, "covariances.pkl")
        if os.path.exists(cov_pkl):
            with open(cov_pkl, "rb") as f:
                obj.fitted_covariances = pickle.load(f)

        vardec_json = os.path.join(base_dir, "vardec_results.json")
        if os.path.exists(vardec_json):
            with open(vardec_json, "r") as f:
                obj.vardec_results = json.load(f)
                if obj.rank is None:
                    obj.rank = obj.vardec_results.get("rank")

        return obj

    def _print_summary(self):
        vr = self.vardec_results
        if vr is None or vr.get("failed"):
            print("  [VarDecResults] Decomposition failed.")
            return

        W = self._W
        has_errors = "errors" in vr
        errors = vr.get("errors", {})

        print()
        print(f"  {'Variance Decomposition Results':^{W}}")
        print(f"  {'═' * W}")

        # Overall Block Model
        if not has_errors:
            print(f"  TABLE I: Block-Model Components")
            print(f"  {'─' * W}")
            print(f"  {'Component':<20} {'Variance':>14} {'Proportion':>12}")
            print(f"  {'─' * W}")
            for key, label in [("var_shared", "Shared"),
                               ("var_het", "Heterogeneity"),
                               ("var_noise", "Noise")]:
                v = vr[key]
                pct = vr[f"pct_{key.split('_', 1)[1]}"]
                print(f"  {label:<20} {v:>14.4f} {pct:>11.1f}%")
            print(f"  {'─' * W}")
            print(f"  {'Total':<20} {vr['var_total']:>14.4f} {'100.0%':>12}")
        else:
            ref_type = errors.get("reference_type", "achieved")
            print(f"  TABLE I: Block-Model Recovery (vs. {ref_type} ground truth)")
            print(f"  {'─' * W}")
            print(f"  {'Component':<20} {'Reference':>12} {'Recovered':>12} {'Error':>12}")
            print(f"  {'─' * W}")
            for label, ref_k, rec_k, err_k in [
                ("Shared",        "var_G_ref",     "var_shared_rec", "err_shared"),
                ("Heterogeneity", "var_het_ref",   "var_het_rec",    "err_het"),
                ("Noise",         "var_noise_ref", "var_noise_rec",  "err_noise"),
            ]:
                print(f"  {label:<20} {errors[ref_k]:>12.4f} "
                      f"{errors[rec_k]:>12.4f} {errors[err_k]:>+12.4f}")
            print(f"  {'─' * W}")
            print(f"  {'Total':<20} {errors['var_total_ref']:>12.4f} "
                  f"{errors['var_total_rec']:>12.4f}")

            # proportions sub-table
            print(f"\n  TABLE II: Proportions (%)")
            print(f"  {'─' * W}")
            print(f"  {'Component':<20} {'Reference':>12} {'Recovered':>12} {'Error':>12}")
            print(f"  {'─' * W}")
            for label, ref_k, rec_k, err_k in [
                ("Shared",        "pct_G_ref",   "pct_shared_rec", "err_pct_shared"),
                ("Heterogeneity", "pct_het_ref", "pct_het_rec",    "err_pct_het"),
                ("Noise",         "pct_noise_ref", "pct_noise_rec", "err_pct_noise"),
            ]:
                print(f"  {label:<20} {errors[ref_k]:>11.1f}% "
                      f"{errors[rec_k]:>11.1f}% {errors[err_k]:>+11.1f}%")

        # Per-Trait Heritability
        tbl = "III" if has_errors else "II"
        print(f"\n  TABLE {tbl}: Per-Trait Marker-Based Heritability (h²)")
        print(f"  {'─' * W}")
        h2_list = vr.get("h2_per_trait")
        gen_list = vr.get("genetic_per_trait")
        noise_list = vr.get("noise_per_trait")

        if gen_list is not None and noise_list is not None and h2_list is not None:
            if has_errors and "h2_err_per_trait" in errors:
                print(f"  {'Trait':<14} {'Genetic':>9} {'Noise':>9} {'h²':>8} {'Err':>8}")
            else:
                print(f"  {'Trait':<14} {'Genetic':>9} {'Noise':>9} {'h²':>8}")
            print(f"  {'─' * W}")
            for j, (g, n, h) in enumerate(zip(gen_list, noise_list, h2_list)):
                line = f"  Trait_{j:<8} {g:>9.4f} {n:>9.4f} {h:>8.3f}"
                if has_errors and "h2_err_per_trait" in errors:
                    line += f" {errors['h2_err_per_trait'][j]:>+8.3f}"
                print(line)
        elif h2_list is not None:
            print(f"  h² = [{', '.join(f'{h:.3f}' for h in h2_list)}]")

        h2m = vr.get("h2_mean")
        if h2m is not None:
            print(f"  {'─' * W}")
            line = (f"  Summary: Mean = {h2m:.3f},  "
                    f"Median = {vr.get('h2_median', 0):.3f},  "
                    f"Range = [{vr.get('h2_min', 0):.3f}, {vr.get('h2_max', 0):.3f}]")
            if has_errors and "h2_mae" in errors:
                line += f",  MAE = {errors['h2_mae']:.3f}"
            print(line)

        # Genetic Correlations
        tbl = "IV" if has_errors else "III"
        print(f"\n  TABLE {tbl}: Genetic Correlations")
        print(f"  {'─' * W}")
        rg_mean = vr.get("rg_mean")
        if rg_mean is not None:
            line = (f"  Mean = {rg_mean:.3f},  "
                    f"Median = {vr.get('rg_median', 0):.3f},  "
                    f"Range = [{vr.get('rg_min', 0):.3f}, {vr.get('rg_max', 0):.3f}]")
            if has_errors and "rg_mae" in errors:
                line += f",  MAE = {errors['rg_mae']:.3f}"
            print(line)
            rg_mat = vr.get("rg_matrix")
            if rg_mat is not None and len(rg_mat) <= 8:
                print()
                for row in rg_mat:
                    print("    " + "  ".join(f"{v:>6.3f}" for v in row))
        else:
            print("  Cannot compute (zero genetic variance for ≥1 trait).")

        if has_errors:
            print(f"\n  {'Error Summary':^{W}}")
            print(f"  {'═' * W}")
            print(f"  {'Metric':<18} {'MAE':>10} {'RMSE':>10} {'Max |Err|':>10}")
            print(f"  {'─' * W}")
            print(f"  {'Overall (pp)':<18} {errors.get('mae', 0):>10.3f} "
                  f"{errors.get('rmse', 0):>10.3f} {errors.get('max_abs_err', 0):>10.3f}")
            if "h2_mae" in errors:
                print(f"  {'Heritability':<18} {errors['h2_mae']:>10.3f} "
                      f"{errors['h2_rmse']:>10.3f} {errors['h2_max_abs_err']:>10.3f}")
            if "rg_mae" in errors:
                print(f"  {'Genet. Corr.':<18} {errors['rg_mae']:>10.3f} "
                      f"{errors['rg_rmse']:>10.3f} {errors['rg_max_abs_err']:>10.3f}")
            print(f"  {'─' * W}")
            print(f"  Final: MAE = {errors.get('mae', 0):.2f} pp,  "
                  f"RMSE = {errors.get('rmse', 0):.2f} pp")

        print(f"  {'═' * W}")
        print()

    def _write_json(self, path, data, warn_on_fail=False):
        try:
            with open(path, "w") as f:
                json.dump(self._prepare_json(data), f, indent=2)
        except Exception as e:
            if warn_on_fail:
                import warnings
                warnings.warn(f"Failed to write {path}: {e}")
            else:
                raise

    def _write_pickle(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def _as_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _to_list(obj):
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (list, tuple)):
            return list(obj)
        return obj

    @staticmethod
    def _safe_float(val):
        if val is None:
            return None
        try:
            f = float(val)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_div(num, den, eps=1e-10):
        return num / den if abs(den) > eps else 0.0

    @staticmethod
    def _safe_divide_arr(num, den, eps=1e-10):
        return np.where(np.abs(den) > eps, num / den, 0.0)

    @staticmethod
    def _cv(arr):
        m = np.mean(arr)
        return float(np.std(arr) / (m + 1e-10)) if abs(m) > 1e-10 else 0.0

    @staticmethod
    def _pcts(a, b, c, total):
        if total > 1e-10:
            return 100 * a / total, 100 * b / total, 100 * c / total
        return 0.0, 0.0, 0.0

    @staticmethod
    def _compute_rg(C0):
        diag = np.diag(C0)
        if not np.all(diag > 1e-10):
            return None
        return C0 / np.sqrt(np.outer(diag, diag))

    def _to_serializable(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            return None if (np.isnan(f) or np.isinf(f)) else f
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        return obj

    def _prepare_json(self, data):
        if isinstance(data, dict):
            return {k: self._prepare_json(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._prepare_json(v) for v in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, (np.integer, np.floating)):
            f = float(data)
            return None if (np.isnan(f) or np.isinf(f)) else f
        if isinstance(data, np.bool_):
            return bool(data)
        if hasattr(data, "item"):
            return data.item()
        return data

    # Data preprocessing (real data only)
    def _save_data_preprocessing_info(self):
        info = {
            "timestamp": datetime.now().isoformat(),
            "uid": self.uid,
        }

        if self.correction_metadata:
            info["corrections"] = {
                "batch_correction": self.correction_metadata.get("batch_correction", {}),
                "covariate_correction": self.correction_metadata.get("covariate_correction", {}),
                "transformations": self.correction_metadata.get("transformations", {}),
            }

        corrections_applied = self.phenotype_data.get("corrections_applied", False)
        if corrections_applied:
            info["phenotype_files"] = {
                "corrected": "phenotypes_corrected.csv",
                "uncorrected": "phenotypes_uncorrected.csv",
            }
        else:
            info["phenotype_files"] = {"phenotypes": "phenotypes.csv"}

        path = self._path("data_preprocessing.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

        if self.verbose:
            print(f"  Saved preprocessing info → {path}")

    def _save_phenotype_data(self):
        corrected = self.phenotype_data.get('corrected', None)
        uncorrected = self.phenotype_data.get('uncorrected', None)
        applied = self.phenotype_data.get('corrections_applied', False)

        if corrected is None:
            return

        col_names = None
        if uncorrected is not None and hasattr(uncorrected, 'columns'):
            col_names = uncorrected.columns

        if not isinstance(corrected, pd.DataFrame):
            # Handle PyTorch tensors (detach from graph and move to CPU first)
            if hasattr(corrected, 'detach'):
                arr = corrected.detach().cpu().numpy()
            # Handle TensorFlow or standard tensors
            elif hasattr(corrected, 'numpy'):
                arr = corrected.numpy()
            else:
                arr = corrected 
                
            corrected = pd.DataFrame(arr, columns=col_names)

        if uncorrected is not None and not isinstance(uncorrected, pd.DataFrame):
            if hasattr(uncorrected, 'detach'):
                arr_u = uncorrected.detach().cpu().numpy()
            elif hasattr(uncorrected, 'numpy'):
                arr_u = uncorrected.numpy()
            else:
                arr_u = uncorrected
            uncorrected = pd.DataFrame(arr_u, columns=col_names)

        if applied and uncorrected is not None:
            cp = os.path.join(self.base_dir, "phenotypes_corrected.csv")
            up = os.path.join(self.base_dir, "phenotypes_uncorrected.csv")
            
            corrected.to_csv(cp, index=False) 
            uncorrected.to_csv(up, index=False)
            
            logger.info(f"Saved corrected: {cp}, uncorrected: {up}")
            self._save_phenotype_comparison(uncorrected, corrected)
        else:
            pp = os.path.join(self.base_dir, "phenotypes.csv")
            corrected.to_csv(pp, index=False)
            logger.info(f"Saved phenotypes to: {pp}")

    def _save_phenotype_comparison(self, before, after):
        path = self._path("phenotype_comparison.txt")
        with open(path, "w") as f:
            for tag, df in [("Before Corrections", before), ("After Corrections", after)]:
                f.write(f"{tag}\n{'─' * 50}\n")
                for col in df.columns:
                    m, s = df[col].mean(), df[col].std()
                    lo, hi = df[col].min(), df[col].max()
                    f.write(f"  {col}: mean = {m:.6f}, std = {s:.6f}, "
                            f"min = {lo:.6f}, max = {hi:.6f}\n")
                f.write("\n")

            f.write(f"Changes\n{'─' * 50}\n")
            for col in before.columns:
                dm = after[col].mean() - before[col].mean()
                ds = after[col].std() - before[col].std()
                mb = before[col].mean()
                sb = before[col].std()
                dm_pct = (dm / mb * 100) if abs(mb) > 1e-8 else 0
                ds_pct = (ds / sb * 100) if abs(sb) > 1e-8 else 0
                f.write(f"  {col}: Δmean = {dm:.6f} ({dm_pct:.2f}%), "
                        f"Δstd = {ds:.6f} ({ds_pct:.2f}%)\n")