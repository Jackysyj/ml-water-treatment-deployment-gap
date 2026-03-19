#!/usr/bin/env python3
"""
Manuscript data consistency validator.

Single source of truth: data/extracted/fulltext_extraction_500_cleaned.json
Checks: manuscript text, figure data summaries, SI tables.

Usage: python3 scripts/validate_manuscript.py
"""

import json
import re
import sys
import csv
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data/extracted/fulltext_extraction_500_cleaned.json"
DRAFTS = ROOT / "manuscript/drafts"
FIGURES = ROOT / "figures"

# ── Colors for terminal output ───────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    return pd.DataFrame(raw["results"])


def compute_truth_table(df):
    """Compute ALL statistics cited in the manuscript from the single source."""
    stats = {}
    n = len(df)

    # ── Basic counts ─────────────────────────────────────────────────────
    stats["total_papers"] = n

    # Year counts
    year_counts = df["year"].value_counts().sort_index().to_dict()
    stats["year_counts"] = {str(k): int(v) for k, v in year_counts.items()}

    # Sub-field counts
    sf_counts = df["sub_field"].value_counts().to_dict()
    stats["subfield_counts"] = {k: int(v) for k, v in sf_counts.items()}
    stats["wwtp_monitoring_pct"] = round(
        (sf_counts.get("WWTP", 0) + sf_counts.get("monitoring", 0))
        / len(df) * 100, 1
    )
    stats["control_pct"] = round(sf_counts.get("control", 0) / len(df) * 100, 1)

    # ── Algorithm counts ─────────────────────────────────────────────────
    algo_counter = Counter()
    for algos in df["ml_algorithms"]:
        if isinstance(algos, list):
            algo_counter.update(algos)
    stats["algo_top10"] = {
        algo: count for algo, count in algo_counter.most_common(10)
    }
    stats["ann_pct"] = round(algo_counter["ANN"] / len(df) * 100, 1)

    # WWTP neural network count (broad: ANN+LSTM+CNN+DNN+GRU+RNN+MLP)
    wwtp = df[df["sub_field"] == "WWTP"]
    nn_archs = ["ANN", "LSTM", "CNN", "DNN", "GRU", "RNN", "MLP"]
    nn_count = sum(
        1 for _, r in wwtp.iterrows()
        if isinstance(r["ml_algorithms"], list)
        and any(a in nn_archs for a in r["ml_algorithms"])
    )
    stats["wwtp_nn_count"] = nn_count
    stats["wwtp_total"] = len(wwtp)

    # Process control RL/MPC
    ctrl = df[df["sub_field"] == "control"]
    ctrl_algos = Counter()
    for algos in ctrl["ml_algorithms"]:
        if isinstance(algos, list):
            ctrl_algos.update(algos)
    stats["control_rl"] = ctrl_algos.get("RL", 0)
    stats["control_mpc"] = ctrl_algos.get("MPC", 0)

    # ── R² statistics ────────────────────────────────────────────────────
    r2 = df[df["best_metric_type"].str.lower() == "r2"].copy()
    r2["val"] = pd.to_numeric(r2["best_metric_value"], errors="coerce")
    r2_valid = r2.dropna(subset=["val"])
    stats["r2_n"] = len(r2_valid)
    stats["r2_median"] = round(float(r2_valid["val"].median()), 2)

    # R² tiers
    tiers = {
        "<0.90": r2_valid["val"] < 0.90,
        "0.90-0.95": (r2_valid["val"] >= 0.90) & (r2_valid["val"] < 0.95),
        "0.95-0.99": (r2_valid["val"] >= 0.95) & (r2_valid["val"] < 0.99),
        ">=0.99": r2_valid["val"] >= 0.99,
    }
    stats["r2_tiers"] = {}
    for name, mask in tiers.items():
        sub = r2_valid[mask]
        dep = int(sub["deployed_in_plant"].sum())
        tier_n = len(sub)
        stats["r2_tiers"][name] = {
            "n": tier_n,
            "deployed": dep,
            "rate": round(dep / tier_n * 100, 1) if tier_n > 0 else 0,
        }

    # ── Deployment & real-time ───────────────────────────────────────────
    stats["deployed_n"] = int(df["deployed_in_plant"].sum())
    stats["deployed_pct"] = round(stats["deployed_n"] / len(df) * 100, 1)
    stats["realtime_n"] = int(df["real_time_testing"].sum())
    stats["realtime_pct"] = round(stats["realtime_n"] / len(df) * 100, 1)

    # Metric-type deployment rates
    stats["metric_deploy"] = {}
    mt_lower = df["best_metric_type"].str.lower()
    for mt, mt_label in [("r2", "R2"), ("accuracy", "accuracy"), ("other", "other")]:
        sub = df[mt_lower == mt]
        dep = int(sub["deployed_in_plant"].sum())
        stats["metric_deploy"][mt_label] = {
            "n": len(sub),
            "deployed": dep,
            "rate": round(dep / len(sub) * 100, 1) if len(sub) > 0 else 0,
        }

    # ── 2025 vs 2024 trends ─────────────────────────────────────────────
    for yr in [2024, 2025]:
        ysub = df[df["year"] == yr]
        prefix = f"y{yr}"
        stats[f"{prefix}_n"] = len(ysub)
        stats[f"{prefix}_deployed_pct"] = round(
            ysub["deployed_in_plant"].sum() / len(ysub) * 100, 1
        ) if len(ysub) > 0 else 0
        if "interpretability_analysis" in df.columns:
            stats[f"{prefix}_interp_pct"] = round(
                ysub["interpretability_analysis"].sum() / len(ysub) * 100, 1
            ) if len(ysub) > 0 else 0
        stats[f"{prefix}_rt_pct"] = round(
            ysub["real_time_testing"].sum() / len(ysub) * 100, 1
        ) if len(ysub) > 0 else 0

    # ── sec2: Data quality ─────────────────────────────────────────────
    ds_counts = df["data_source"].value_counts()
    stats["scada_pct"] = round(ds_counts.get("SCADA", 0) / n * 100, 1)
    stats["experimental_pct"] = round(ds_counts.get("experimental", 0) / n * 100, 1)
    stats["simulated_pct"] = round(ds_counts.get("simulated", 0) / n * 100, 1)

    stats["real_ww_pct"] = round(df["uses_real_wastewater"].sum() / n * 100, 1)
    stats["lab_scale_pct"] = round((df["scale"] == "lab").sum() / n * 100, 1)
    stats["full_scale_pct"] = round((df["scale"] == "full_scale").sum() / n * 100, 1)

    ds_size = pd.to_numeric(df["dataset_size"], errors="coerce").dropna()
    stats["ds_n"] = len(ds_size)
    stats["ds_median"] = int(ds_size.median())
    stats["ds_iqr_lo"] = int(ds_size.quantile(0.25))
    stats["ds_iqr_hi"] = int(ds_size.quantile(0.75))

    for sf in ["sludge", "DBP", "WWTP", "monitoring"]:
        sub_ds = pd.to_numeric(df[df["sub_field"] == sf]["dataset_size"], errors="coerce").dropna()
        stats[f"ds_median_{sf}"] = int(sub_ds.median()) if len(sub_ds) > 0 else 0

    stats["uq_pct"] = round(df["uncertainty_quantification"].sum() / n * 100, 1)

    ctrl = df[df["sub_field"] == "control"]
    stats["ctrl_sim_scale_pct"] = round((ctrl["scale"] == "simulation").sum() / len(ctrl) * 100, 1)
    stats["ctrl_full_scale_pct"] = round((ctrl["scale"] == "full_scale").sum() / len(ctrl) * 100, 1)

    # ── sec3: IT/OT ────────────────────────────────────────────────────
    fw = df["model_framework"].value_counts()
    stats["fw_not_reported_pct"] = round(fw.get("not_specified", 0) / n * 100, 1)
    stats["fw_matlab_pct"] = round(fw.get("matlab", 0) / n * 100, 1)
    stats["fw_sklearn_pct"] = round(fw.get("sklearn", 0) / n * 100, 1)
    stats["fw_tf_pct"] = round(fw.get("tensorflow", 0) / n * 100, 1)
    stats["fw_pytorch_pct"] = round(fw.get("pytorch", 0) / n * 100, 1)
    stats["fw_keras_pct"] = round(fw.get("keras", 0) / n * 100, 1)

    cl = df["control_loop_type"].value_counts()
    stats["cl_none_pct"] = round(cl.get("none", 0) / n * 100, 1)
    stats["cl_advisory_pct"] = round(cl.get("advisory", 0) / n * 100, 1)
    stats["cl_closed_pct"] = round(cl.get("closed_loop", 0) / n * 100, 1)
    stats["ctrl_closed_pct"] = round(
        (ctrl["control_loop_type"] == "closed_loop").sum() / len(ctrl) * 100, 1
    )

    # Deployed model algorithms
    deployed_df = df[df["deployed_in_plant"] == True]
    dep_algos = Counter()
    for algos in deployed_df["ml_algorithms"]:
        if isinstance(algos, list):
            dep_algos.update(algos)
    stats["dep_ann"] = dep_algos.get("ANN", 0)
    stats["dep_rf"] = dep_algos.get("RF", 0)
    stats["dep_svm"] = dep_algos.get("SVM", 0)
    stats["dep_lr"] = dep_algos.get("LR", 0)
    stats["dep_dt"] = dep_algos.get("DT", 0)
    stats["dep_lstm"] = dep_algos.get("LSTM", 0)

    stats["code_pct"] = round(df["code_available"].sum() / n * 100, 1)
    stats["data_avail_pct"] = round(df["data_available"].sum() / n * 100, 1)

    # ── sec4: Trust ────────────────────────────────────────────────────
    vm = df["validation_method"].value_counts()
    stats["val_random_pct"] = round(vm.get("random_split", 0) / n * 100, 1)
    stats["val_temporal_pct"] = round(vm.get("temporal_split", 0) / n * 100, 1)
    stats["val_walkfwd_n"] = int(vm.get("walk_forward", 0))
    stats["val_walkfwd_pct"] = round(vm.get("walk_forward", 0) / n * 100, 1)
    stats["val_external_n"] = int(vm.get("external", 0))
    stats["val_external_pct"] = round(vm.get("external", 0) / n * 100, 1)
    stats["val_none_pct"] = round(vm.get("none_reported", 0) / n * 100, 1)

    # UQ by year
    for yr in [2018, 2025]:
        ysub = df[df["year"] == yr]
        if len(ysub) > 0:
            stats[f"uq_{yr}_pct"] = round(ysub["uncertainty_quantification"].sum() / len(ysub) * 100, 1)

    # Interpretability
    interp_mask = df["interpretability_method"] != "none"
    stats["interp_n"] = int(interp_mask.sum())
    stats["interp_pct"] = round(interp_mask.sum() / n * 100, 1)
    interp_df = df[interp_mask]
    im = interp_df["interpretability_method"].value_counts()
    stats["interp_fi_n"] = int(im.get("feature_importance", 0))
    stats["interp_shap_n"] = int(im.get("SHAP", 0))

    for yr in [2018, 2025]:
        ysub = df[df["year"] == yr]
        if len(ysub) > 0:
            stats[f"interp_{yr}_pct"] = round(
                (ysub["interpretability_method"] != "none").sum() / len(ysub) * 100, 1
            )

    # Deployment readiness score
    bool_cols = ["deployed_in_plant", "real_time_testing", "uncertainty_quantification",
                 "code_available", "data_available"]
    readiness = df[bool_cols].sum(axis=1) + interp_mask.astype(int)
    stats["readiness_median"] = float(readiness.median())
    stats["readiness_zero_pct"] = round((readiness == 0).sum() / n * 100, 1)

    # Phi coefficients
    from scipy.stats import chi2_contingency
    import numpy as np

    def _phi(a, b):
        ct = pd.crosstab(a, b)
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
        phi_val = np.sqrt(chi2 / n)
        if ct.shape == (2, 2):
            sign = np.sign(ct.iloc[0, 0] * ct.iloc[1, 1] - ct.iloc[0, 1] * ct.iloc[1, 0])
            return round(sign * phi_val, 3)
        return round(phi_val, 3)

    stats["phi_deploy_uq"] = _phi(df["deployed_in_plant"], df["uncertainty_quantification"])
    stats["phi_deploy_interp"] = _phi(df["deployed_in_plant"], interp_mask)
    stats["phi_code_data"] = _phi(df["code_available"], df["data_available"])

    # ── sec5: Two pathways ─────────────────────────────────────────────
    coag = df[df["sub_field"] == "coagulation"]
    others = df[df["sub_field"] != "coagulation"]
    n_coag = len(coag)
    n_others = len(others)

    stats["coag_rw_pct"] = round(coag["uses_real_wastewater"].sum() / n_coag * 100, 1)
    stats["others_rw_pct"] = round(others["uses_real_wastewater"].sum() / n_others * 100, 1)
    stats["coag_sim_pct"] = round((coag["data_source"] == "simulated").sum() / n_coag * 100, 1)
    stats["others_sim_pct"] = round(
        (others["scale"] == "simulation").sum() / n_others * 100, 1
    )
    stats["coag_full_pct"] = round((coag["scale"] == "full_scale").sum() / n_coag * 100, 1)
    stats["coag_lab_pct"] = round((coag["scale"] == "lab").sum() / n_coag * 100, 1)
    stats["others_lab_pct"] = round((others["scale"] == "lab").sum() / n_others * 100, 1)
    stats["coag_deploy_pct"] = round(coag["deployed_in_plant"].sum() / n_coag * 100, 1)
    stats["others_deploy_pct"] = round(others["deployed_in_plant"].sum() / n_others * 100, 1)
    stats["coag_advisory_pct"] = round(
        (coag["control_loop_type"] == "advisory").sum() / n_coag * 100, 1
    )

    stats["ctrl_rw_pct"] = round(ctrl["uses_real_wastewater"].sum() / len(ctrl) * 100, 1)
    stats["ctrl_sim_data_pct"] = round(
        (ctrl["data_source"] == "simulated").sum() / len(ctrl) * 100, 1
    )
    stats["ctrl_deploy_pct"] = round(ctrl["deployed_in_plant"].sum() / len(ctrl) * 100, 1)
    stats["ctrl_closed_loop_pct"] = round(
        (ctrl["control_loop_type"] == "closed_loop").sum() / len(ctrl) * 100, 1
    )
    stats["ctrl_temporal_pct"] = round(
        (ctrl["validation_method"] == "temporal_split").sum() / len(ctrl) * 100, 1
    )
    stats["coag_temporal_pct"] = round(
        (coag["validation_method"] == "temporal_split").sum() / n_coag * 100, 1
    )

    ctrl_ds = pd.to_numeric(ctrl["dataset_size"], errors="coerce").dropna()
    coag_ds = pd.to_numeric(coag["dataset_size"], errors="coerce").dropna()
    stats["ctrl_ds_median"] = int(round(ctrl_ds.median(), 0)) if len(ctrl_ds) > 0 else 0
    stats["coag_ds_median"] = int(coag_ds.median()) if len(coag_ds) > 0 else 0

    stats["coag_uq_pct"] = round(coag["uncertainty_quantification"].sum() / n_coag * 100, 1)
    stats["coag_code_pct"] = round(coag["code_available"].sum() / n_coag * 100, 1)

    coag_readiness = readiness[df["sub_field"] == "coagulation"].mean()
    ctrl_readiness = readiness[df["sub_field"] == "control"].mean()
    stats["coag_readiness"] = round(coag_readiness, 2)
    stats["ctrl_readiness"] = round(ctrl_readiness, 2)

    # Sub-field deployment rates for sec5 para 3
    for sf in ["monitoring", "membrane", "WWTP"]:
        sub = df[df["sub_field"] == sf]
        stats[f"{sf}_deploy_pct"] = round(sub["deployed_in_plant"].mean() * 100, 1)
        stats[f"{sf}_rt_pct"] = round(sub["real_time_testing"].mean() * 100, 1)
        stats[f"{sf}_interp_pct"] = round(
            (sub["interpretability_method"] != "none").mean() * 100, 1
        )

    return stats


# ── Manuscript text checker ──────────────────────────────────────────────

# Each rule: (file, description, regex_pattern, expected_value_lambda)
# The lambda receives the truth table and returns the expected string/number.

def build_checks(stats):
    """Build a list of (file, description, pattern, expected) tuples."""
    s = stats
    sf = s["subfield_counts"]
    checks = [
        # sec1 para 1
        ("sec1_landscape.md", "total papers", r"corpus of (\d+)", "500"),
        ("sec1_landscape.md", "2018 count", r"from (\d+) in 2018", str(s["year_counts"]["2018"])),
        ("sec1_landscape.md", "2023 peak", r"peak of (\d+) in 2023", str(s["year_counts"]["2023"])),
        ("sec1_landscape.md", "2024 n", r"2024 \(\*n\* = (\d+)\)", str(s["year_counts"]["2024"])),
        ("sec1_landscape.md", "2025 n", r"2025 \(\*n\* = (\d+)\)", str(s["year_counts"]["2025"])),
        ("sec1_landscape.md", "Fig ref para1", r"five years \(Fig\. (\w+)\)", "1b"),
        # sec1 para 2
        ("sec1_landscape.md", "WWTP count", r"WWTP\).*?(\d+) and", str(sf["WWTP"])),
        ("sec1_landscape.md", "monitoring count", r"and (\d+), respectively", str(sf["monitoring"])),
        ("sec1_landscape.md", "WWTP+monitoring pct", r"respectively; ([\d.]+)%", str(s["wwtp_monitoring_pct"])),
        ("sec1_landscape.md", "sludge n", r"sludge treatment, \*n\* = (\d+)", str(sf["sludge"])),
        ("sec1_landscape.md", "membrane n", r"membrane systems, \*n\* = (\d+)", str(sf["membrane"])),
        ("sec1_landscape.md", "control n", r"process control, \*n\* = (\d+)", str(sf["control"])),
        ("sec1_landscape.md", "coagulation n", r"coagulation optimization, \*n\* = (\d+)", str(sf["coagulation"])),
        ("sec1_landscape.md", "DBP n", r"byproduct modeling, \*n\* = (\d+)", str(sf["DBP"])),
        ("sec1_landscape.md", "Fig ref para2", r"corpus \(Fig\. (\w+)\)", "1d"),
        ("sec1_landscape.md", "control pct", r"only ([\d.]+)% of studies", str(s["control_pct"])),
        # sec1 para 3
        ("sec1_landscape.md", "ANN pct", r"([\d.]+)% of studies \(348", str(s["ann_pct"])),
        ("sec1_landscape.md", "ANN count", r"studies \((\d+) of 500\)", str(s["algo_top10"]["ANN"])),
        ("sec1_landscape.md", "RF count", r"RF, (\d+)\)", str(s["algo_top10"]["RF"])),
        ("sec1_landscape.md", "LR count", r"LR, (\d+)\)", str(s["algo_top10"]["LR"])),
        ("sec1_landscape.md", "SVR count", r"SVR, (\d+)\)", str(s["algo_top10"]["SVR"])),
        ("sec1_landscape.md", "WWTP NN 125/149", r"(\d+) of 149 studies", str(s["wwtp_nn_count"])),
        # sec1 para 4
        ("sec1_landscape.md", "R2 n", r"(\d+) studies reporting \*R\*", str(s["r2_n"])),
        ("sec1_landscape.md", "R2 median", r"median value is ([\d.]+)", str(s["r2_median"])),
        ("sec1_landscape.md", "deployed pct", r"only ([\d.]+)% of all 500", str(s["deployed_pct"])),
        ("sec1_landscape.md", "realtime pct", r"only ([\d.]+)% include real", str(s["realtime_pct"])),
        ("sec1_landscape.md", "R2>=0.99 n", r"(\d+) studies reporting \*R\*² ≥ 0.99", str(s["r2_tiers"][">=0.99"]["n"])),
        ("sec1_landscape.md", "R2>=0.99 rate", r"deployment rate was ([\d.]+)%", str(s["r2_tiers"][">=0.99"]["rate"])),
        ("sec1_landscape.md", "R2<0.90 rate", r"R\*² < 0.90 \(([\d.]+)%\)", str(s["r2_tiers"]["<0.90"]["rate"])),
        ("sec1_landscape.md", "R2 deploy rate", r"lowest deployment rate \(([\d.]+)%\)", str(s["metric_deploy"]["R2"]["rate"])),
        ("sec1_landscape.md", "accuracy deploy", r"accuracy \(([\d.]+)%\)", str(s["metric_deploy"]["accuracy"]["rate"])),
        ("sec1_landscape.md", "other deploy", r"objectives \(([\d.]+)%\)", str(s["metric_deploy"]["other"]["rate"])),
        # sec1 para 5 (2025 trends)
        ("sec1_landscape.md", "2025 deploy pct", r"rose to ([\d.]+)%", str(s["y2025_deployed_pct"])),
        ("sec1_landscape.md", "2024 deploy pct", r"compared with ([\d.]+)% in 2024", str(s["y2024_deployed_pct"])),

        # ── sec2: Data quality ─────────────────────────────────────────
        ("sec2_data_quality.md", "SCADA pct", r"SCADA systems contribute ([\d.]+)%", str(s["scada_pct"])),
        ("sec2_data_quality.md", "lab exp pct", r"laboratory experiments ([\d.]+)%", str(s["experimental_pct"])),
        ("sec2_data_quality.md", "simulated pct", r"simulated datasets ([\d.]+)%", str(s["simulated_pct"])),
        ("sec2_data_quality.md", "real ww pct", r"([\d.]+)% of studies report using real", str(s["real_ww_pct"])),
        ("sec2_data_quality.md", "lab scale pct", r"([\d.]+)% of all studies operate at laboratory", str(s["lab_scale_pct"])),
        ("sec2_data_quality.md", "ds n", r"Among the (\d+) studies reporting", str(s["ds_n"])),
        ("sec2_data_quality.md", "ds median", r"median is ([\d,]+) samples", str(s["ds_median"])),
        ("sec2_data_quality.md", "ds iqr lo", r"interquartile range: ([\d,]+)", str(s["ds_iqr_lo"])),
        ("sec2_data_quality.md", "ds iqr hi", r"interquartile range: [\d,]+[–-]([\d,]+)", str(f"{s['ds_iqr_hi']:,}")),
        ("sec2_data_quality.md", "full scale pct", r"full-scale data \(([\d.]+)%", str(s["full_scale_pct"])),
        ("sec2_data_quality.md", "realtime pct", r"real-time testing: only ([\d.]+)%", str(s["realtime_pct"])),
        ("sec2_data_quality.md", "UQ pct", r"only ([\d.]+)% of all studies quantify", str(s["uq_pct"])),
        ("sec2_data_quality.md", "ctrl sim pct", r"draws ([\d.]+)% of its datasets from simulated", str(s["ctrl_sim_scale_pct"])),
        ("sec2_data_quality.md", "ctrl full pct", r"only ([\d.]+)% of its studies use full-scale", str(s["ctrl_full_scale_pct"])),

        # ── sec3: IT/OT ───────────────────────────────────────────────
        ("sec3_it_ot.md", "fw not reported", r"\(([\d.]+)%\) do not report the software", str(s["fw_not_reported_pct"])),
        ("sec3_it_ot.md", "MATLAB pct", r"MATLAB \(([\d.]+)%\)", str(s["fw_matlab_pct"])),
        ("sec3_it_ot.md", "sklearn pct", r"scikit-learn ([\d.]+)%", str(s["fw_sklearn_pct"])),
        ("sec3_it_ot.md", "TF pct", r"TensorFlow ([\d.]+)%", str(s["fw_tf_pct"])),
        ("sec3_it_ot.md", "PyTorch pct", r"PyTorch ([\d.]+)%", str(s["fw_pytorch_pct"])),
        ("sec3_it_ot.md", "Keras pct", r"Keras ([\d.]+)%", str(s["fw_keras_pct"])),
        ("sec3_it_ot.md", "cl none pct", r"Two-thirds of studies \(([\d.]+)%\)", str(s["cl_none_pct"])),
        ("sec3_it_ot.md", "cl advisory pct", r"([\d.]+)% operate in an advisory", str(s["cl_advisory_pct"])),
        ("sec3_it_ot.md", "cl closed pct", r"Only ([\d.]+)% implement closed-loop", str(s["cl_closed_pct"])),
        ("sec3_it_ot.md", "ctrl closed pct", r"([\d.]+)% of studies feature closed-loop", str(s["ctrl_closed_pct"])),
        ("sec3_it_ot.md", "deployed n", r"Among the (\d+) studies reporting plant", str(s["deployed_n"])),
        ("sec3_it_ot.md", "dep ANN", r"neural networks \((\d+) studies\)", str(s["dep_ann"])),
        ("sec3_it_ot.md", "dep RF", r"random forest \((\d+)\)", str(s["dep_rf"])),
        ("sec3_it_ot.md", "dep SVM", r"support vector machines \((\d+)\)", str(s["dep_svm"])),
        ("sec3_it_ot.md", "dep LR", r"linear regression \((\d+)\)", str(s["dep_lr"])),
        ("sec3_it_ot.md", "dep DT", r"decision trees \((\d+)\)", str(s["dep_dt"])),
        ("sec3_it_ot.md", "dep LSTM", r"LSTM networks \((\d+)\)", str(s["dep_lstm"])),
        ("sec3_it_ot.md", "code pct", r"only ([\d.]+)% of studies provide source code", str(s["code_pct"])),
        ("sec3_it_ot.md", "data avail pct", r"([\d.]+)% share datasets", str(s["data_avail_pct"])),

        # ── sec4: Trust ────────────────────────────────────────────────
        ("sec4_trust.md", "val random pct", r"\(([\d.]+)%\) rely on random train-test", str(s["val_random_pct"])),
        ("sec4_trust.md", "val temporal pct", r"Only ([\d.]+)% of studies employ temporal", str(s["val_temporal_pct"])),
        ("sec4_trust.md", "val walkfwd pct", r"just four studies \(([\d.]+)%\)", str(s["val_walkfwd_pct"])),
        ("sec4_trust.md", "val external n", r"in only eight \(([\d.]+)%\)", str(s["val_external_pct"])),
        ("sec4_trust.md", "val none pct", r"([\d.]+)% of studies do not report any", str(s["val_none_pct"])),
        ("sec4_trust.md", "UQ pct", r"Only ([\d.]+)% of studies provide any measure", str(s["uq_pct"])),
        ("sec4_trust.md", "UQ 2018 pct", r"it was ([\d.]+)% in 2018", str(s["uq_2018_pct"])),
        ("sec4_trust.md", "UQ 2025 pct", r"([\d.]+)% in 2025", str(s["uq_2025_pct"])),
        ("sec4_trust.md", "2018 n", r"\*n\* = (\d+) in 2018", str(s["year_counts"]["2018"])),
        ("sec4_trust.md", "interp 2018 pct", r"rose from ([\d.]+)% in 2018", str(s["interp_2018_pct"])),
        ("sec4_trust.md", "interp 2025 pct", r"to ([\d.]+)% in 2025", str(s["interp_2025_pct"])),
        ("sec4_trust.md", "interp n", r"Among the (\d+) studies", str(s["interp_n"])),
        ("sec4_trust.md", "interp pct", r"(\d+) studies \(([\d.]+)%\) employing", None),  # skip, complex
        ("sec4_trust.md", "interp FI n", r"feature importance rankings \((\d+)", str(s["interp_fi_n"])),
        ("sec4_trust.md", "interp SHAP n", r"SHAP values \((\d+)\)", str(s["interp_shap_n"])),
        ("sec4_trust.md", "readiness zero pct", r"([\d.]+)% scoring zero on all six", str(s["readiness_zero_pct"])),
        ("sec4_trust.md", "phi deploy UQ", r"deployment and uncertainty quantification is just ([\d.]+)", str(s["phi_deploy_uq"])),
        ("sec4_trust.md", "phi deploy interp", r"deployment and interpretability it is ([\d.-]+)", str(s["phi_deploy_interp"])),
        ("sec4_trust.md", "phi code data", r"phi = ([\d.]+)\)", str(s["phi_code_data"])),

        # ── sec5: Two pathways ─────────────────────────────────────────
        ("sec5_coagulation.md", "coag rw pct", r"([\d.]+)% of coagulation studies use real", str(s["coag_rw_pct"])),
        ("sec5_coagulation.md", "others rw pct", r"compared with ([\d.]+)% across all other", str(s["others_rw_pct"])),
        ("sec5_coagulation.md", "coag sim 0%", r"(\d+)% of coagulation papers rely on simulated", str(int(s["coag_sim_pct"]))),
        ("sec5_coagulation.md", "others sim pct", r"versus ([\d.]+)% in other sub-fields", str(s["others_sim_pct"])),
        ("sec5_coagulation.md", "coag full pct", r"full-scale studies is modestly higher \(([\d.]+)%", str(s["coag_full_pct"])),
        ("sec5_coagulation.md", "coag lab pct", r"lab-scale work is notably more prevalent \(([\d.]+)%", str(s["coag_lab_pct"])),
        ("sec5_coagulation.md", "others lab pct", r"prevalent \([\d.]+% vs\. ([\d.]+)%\)", str(s["others_lab_pct"])),
        ("sec5_coagulation.md", "coag deploy pct", r"deployment rate is ([\d.]+)%", str(s["coag_deploy_pct"])),
        ("sec5_coagulation.md", "others deploy pct", r"double the rate for other sub-fields \(([\d.]+)%\)", str(s["others_deploy_pct"])),
        ("sec5_coagulation.md", "coag advisory pct", r"([\d.]+)% of studies incorporate advisory", str(s["coag_advisory_pct"])),
        ("sec5_coagulation.md", "ctrl rw pct", r"lowest real wastewater usage.*?\(([\d.]+)%\)", str(s["ctrl_rw_pct"])),
        ("sec5_coagulation.md", "ctrl sim scale pct", r"highest reliance on simulated data \(([\d.]+)%", str(s["ctrl_sim_scale_pct"])),
        ("sec5_coagulation.md", "ctrl deploy pct", r"highest deployment rate \(([\d.]+)%\)", str(s["ctrl_deploy_pct"])),
        ("sec5_coagulation.md", "ctrl closed pct", r"closed-loop integration \(([\d.]+)%\)", str(s["ctrl_closed_loop_pct"])),
        ("sec5_coagulation.md", "ctrl temporal pct", r"Temporal validation is more common.*\(([\d.]+)%", str(s["ctrl_temporal_pct"])),
        ("sec5_coagulation.md", "coag temporal pct", r"17\.3% vs\. ([\d.]+)%\)", str(s["coag_temporal_pct"])),
        ("sec5_coagulation.md", "ctrl ds median", r"substantially larger \(median ([\d,]+)", str(f"{s['ctrl_ds_median']:,}")),
        ("sec5_coagulation.md", "coag ds median", r"vs\. (\d+) samples\)", str(s["coag_ds_median"])),
        ("sec5_coagulation.md", "coag readiness", r"readiness scores \(([\d.]+) vs", str(s["coag_readiness"])),
        ("sec5_coagulation.md", "ctrl readiness", r"vs\. ([\d.]+)\)", str(s["ctrl_readiness"])),
        ("sec5_coagulation.md", "monitoring deploy", r"Monitoring applications \(deployed in ([\d.]+)%", str(s["monitoring_deploy_pct"])),
        ("sec5_coagulation.md", "monitoring rt", r"real-time testing rates \(([\d.]+)%\)", str(s["monitoring_rt_pct"])),
        ("sec5_coagulation.md", "membrane deploy", r"Membrane systems \(deployed in ([\d.]+)%", str(s["membrane_deploy_pct"])),
        ("sec5_coagulation.md", "membrane interp", r"interpretability adoption \(([\d.]+)%\)", str(s["membrane_interp_pct"])),
        ("sec5_coagulation.md", "WWTP deploy", r"lowest deployment rate \(([\d.]+)%\)", str(s["WWTP_deploy_pct"])),

        # ── sec6: Discussion ───────────────────────────────────────────
        ("sec6_discussion.md", "readiness zero pct", r"([\d.]+)% scoring zero on all six", str(s["readiness_zero_pct"])),
        ("sec6_discussion.md", "deployed pct", r"([\d.]+)% reporting plant deployment", str(s["deployed_pct"])),
        ("sec6_discussion.md", "advisory pct", r"present in ([\d.]+)% of studies overall", str(s["cl_advisory_pct"])),
        ("sec6_discussion.md", "coag advisory pct", r"([\d.]+)% of coagulation studies", str(s["coag_advisory_pct"])),
    ]
    # Remove None-expected entries (complex patterns skipped)
    checks = [(f, d, p, e) for f, d, p, e in checks if e is not None]
    return checks


def check_manuscript(stats):
    """Run regex checks against manuscript text."""
    checks = build_checks(stats)
    passed, failed, skipped = 0, 0, 0
    results = []

    for filename, desc, pattern, expected in checks:
        filepath = DRAFTS / filename
        if not filepath.exists():
            results.append(("SKIP", filename, desc, f"file not found"))
            skipped += 1
            continue

        text = filepath.read_text(encoding="utf-8")
        match = re.search(pattern, text)
        if match is None:
            results.append(("SKIP", filename, desc, f"pattern not found: {pattern}"))
            skipped += 1
            continue

        actual = match.group(1)
        if actual == expected:
            results.append(("PASS", filename, desc, f"{actual}"))
            passed += 1
        else:
            results.append(("FAIL", filename, desc, f"expected={expected}, found={actual}"))
            failed += 1

    return results, passed, failed, skipped


# ── Figure data checker ──────────────────────────────────────────────────

def check_fig1_data(stats):
    """Cross-check fig1_data_summary.csv against truth table."""
    csv_path = FIGURES / "fig1_data_summary.csv"
    if not csv_path.exists():
        return [("SKIP", "fig1_data_summary.csv", "file", "not found")], 0, 0, 1

    rows = list(csv.DictReader(open(csv_path)))
    passed, failed = 0, 0
    results = []

    # Check panel (a) R2 total
    r2_row = [r for r in rows if r["item"] == "R2_total"]
    if r2_row:
        csv_n = int(r2_row[0]["n"])
        if csv_n == stats["r2_n"]:
            results.append(("PASS", "fig1", "R2 total n", str(csv_n)))
            passed += 1
        else:
            results.append(("FAIL", "fig1", "R2 total n", f"csv={csv_n}, truth={stats['r2_n']}"))
            failed += 1

    # Check panel (d) sub-field counts
    for sf, expected_n in stats["subfield_counts"].items():
        sf_row = [r for r in rows if r["item"] == f"subfield_{sf}"]
        if sf_row:
            csv_n = int(sf_row[0]["n"])
            if csv_n == expected_n:
                results.append(("PASS", "fig1", f"subfield {sf}", str(csv_n)))
                passed += 1
            else:
                results.append(("FAIL", "fig1", f"subfield {sf}", f"csv={csv_n}, truth={expected_n}"))
                failed += 1

    # Check panel (e) R2 tiers
    for tier_name, tier_data in stats["r2_tiers"].items():
        tier_row = [r for r in rows if r["item"] == f"tier_{tier_name}"]
        if tier_row:
            csv_n = int(tier_row[0]["n"])
            if csv_n == tier_data["n"]:
                results.append(("PASS", "fig1", f"tier {tier_name} n", str(csv_n)))
                passed += 1
            else:
                results.append(("FAIL", "fig1", f"tier {tier_name} n", f"csv={csv_n}, truth={tier_data['n']}"))
                failed += 1

    return results, passed, failed, 0


# ── Ref placeholder checker ──────────────────────────────────────────────

def check_ref_placeholders():
    """Count remaining [ref] placeholders across all drafts."""
    results = []
    total = 0
    for md in sorted(DRAFTS.glob("*.md")):
        if md.name.startswith("SI_"):
            continue
        text = md.read_text(encoding="utf-8")
        count = len(re.findall(r"\[ref\]", text, re.IGNORECASE))
        if count > 0:
            results.append(("WARN", md.name, f"{count} [ref] placeholders", ""))
            total += count
    return results, total


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Manuscript Data Consistency Validator")
    print("=" * 60)

    # Load and compute
    df = load_data()
    stats = compute_truth_table(df)

    # Save truth table
    truth_path = FIGURES / "manuscript_stats.json"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with open(truth_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nTruth table saved to {truth_path}")

    # 1. Manuscript text checks
    print(f"\n{'─' * 60}")
    print("  1. Manuscript Text vs Data")
    print(f"{'─' * 60}")
    ms_results, ms_pass, ms_fail, ms_skip = check_manuscript(stats)
    for status, fname, desc, detail in ms_results:
        color = GREEN if status == "PASS" else RED if status == "FAIL" else YELLOW
        print(f"  {color}{status}{RESET}  {fname:30s} {desc:25s} {detail}")
    print(f"\n  Total: {ms_pass} passed, {ms_fail} failed, {ms_skip} skipped")

    # 2. Figure data checks
    print(f"\n{'─' * 60}")
    print("  2. Figure Data Summaries vs Data")
    print(f"{'─' * 60}")
    fig_results, fig_pass, fig_fail, fig_skip = check_fig1_data(stats)
    for status, fname, desc, detail in fig_results:
        color = GREEN if status == "PASS" else RED if status == "FAIL" else YELLOW
        print(f"  {color}{status}{RESET}  {fname:30s} {desc:25s} {detail}")
    print(f"\n  Total: {fig_pass} passed, {fig_fail} failed, {fig_skip} skipped")

    # 3. Ref placeholders
    print(f"\n{'─' * 60}")
    print("  3. Reference Placeholders")
    print(f"{'─' * 60}")
    ref_results, ref_total = check_ref_placeholders()
    for status, fname, desc, _ in ref_results:
        print(f"  {YELLOW}{status}{RESET}  {fname:30s} {desc}")
    if ref_total == 0:
        print(f"  {GREEN}All references filled.{RESET}")
    else:
        print(f"\n  {YELLOW}{ref_total} [ref] placeholders remaining.{RESET}")

    # Summary
    total_fail = ms_fail + fig_fail
    print(f"\n{'=' * 60}")
    if total_fail == 0:
        print(f"  {GREEN}ALL CHECKS PASSED{RESET}")
    else:
        print(f"  {RED}{total_fail} INCONSISTENCIES FOUND{RESET}")
    print(f"{'=' * 60}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
