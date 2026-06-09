#!/usr/bin/env python3
"""
Compute all statistical tests for the manuscript.

Outputs:
  - figures/statistical_tests.json (machine-readable)
  - stdout summary (human-readable)

Tests:
  1. Phi coefficients + χ² p-values (sec4)
  2. Wilson score 95% CI for key proportions (sec1)
  3. Fisher's exact test: R² tier vs deployment (sec1)
  4. Firth penalized logistic regression: deployment predictors (sec4/SI)
  5. Fisher's exact + χ²: coagulation comparisons (sec5)
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm

warnings.filterwarnings("ignore")

from paths import ANALYSIS_CORPUS_JSON, FIGURES_DIR

DATA_PATH = ANALYSIS_CORPUS_JSON
OUT_PATH = FIGURES_DIR / "statistical_tests.json"


# ── Data loading ──────────────────────────────────────────────────────────

def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["results"])
    # Restrict to the declared 2018-2025 corpus window. The raw extraction
    # contained 15 out-of-range records (2015-2017 x13, 2026 x2, 2 of them
    # flagged deployed); keeping them is inconsistent with the stated window
    # and inflates the deployment numerator, so they are dropped here. This
    # makes n=423 the single source of truth for all downstream stats/figures.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= 2018) & (df["year"] <= 2025)].reset_index(drop=True)
    # Boolean columns
    bool_cols = [
        "deployed_in_plant", "real_time_testing", "uncertainty_quantification",
        "code_available", "data_available", "uses_real_wastewater"
    ]
    for c in bool_cols:
        df[c] = df[c].astype(bool).astype(int)
    # Interpretability flag
    df["interpretability_flag"] = df["interpretability_method"].apply(
        lambda x: 0 if x in (None, "none", "None", "") else 1
    )
    # Temporal validation flag
    df["temporal_validation"] = df["validation_method"].apply(
        lambda x: 1 if isinstance(x, str) and "temporal" in x.lower() else 0
    )
    return df


# ── Wilson score interval ─────────────────────────────────────────────────

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


# ── Phi coefficient + χ² test ─────────────────────────────────────────────

def phi_test(df, var1, var2):
    """Compute phi coefficient and χ² p-value for two binary variables."""
    ct = pd.crosstab(df[var1], df[var2])
    # Ensure 2x2
    for val in [0, 1]:
        if val not in ct.index:
            ct.loc[val] = 0
        if val not in ct.columns:
            ct[val] = 0
    ct = ct.loc[[0, 1], [0, 1]]
    n = ct.values.sum()
    chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
    phi = np.sqrt(chi2 / n)
    # Sign: positive if concordant
    ad = ct.iloc[0, 0] * ct.iloc[1, 1]
    bc = ct.iloc[0, 1] * ct.iloc[1, 0]
    if ad < bc:
        phi = -phi
    return {"phi": round(phi, 3), "chi2": round(chi2, 3), "p": round(p, 4),
            "n": int(n), "contingency_table": ct.values.tolist()}


# ── Firth penalized logistic regression ───────────────────────────────────

def firth_logistic(X, y, max_iter=200, tol=1e-6):
    """
    Firth's penalized logistic regression via modified score equations.

    Adds Jeffreys prior (half the log-determinant of Fisher information)
    to the log-likelihood, reducing small-sample bias.

    Returns: coefficients, standard errors, p-values, confidence intervals
    """
    n, p = X.shape

    def neg_penalized_ll(beta):
        xb = X @ beta
        xb = np.clip(xb, -500, 500)
        prob = 1 / (1 + np.exp(-xb))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        # Log-likelihood
        ll = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        # Penalty: 0.5 * log(det(Fisher info))
        W = np.diag(prob * (1 - prob))
        fisher = X.T @ W @ X
        sign, logdet = np.linalg.slogdet(fisher)
        if sign <= 0:
            penalty = -1e10
        else:
            penalty = 0.5 * logdet
        return -(ll + penalty)

    # Initialize with zeros
    beta0 = np.zeros(p)

    result = minimize(neg_penalized_ll, beta0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol})
    beta_hat = result.x

    # Standard errors from inverse Fisher information
    xb = X @ beta_hat
    xb = np.clip(xb, -500, 500)
    prob = 1 / (1 + np.exp(-xb))
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    W = np.diag(prob * (1 - prob))
    fisher = X.T @ W @ X
    try:
        cov = np.linalg.inv(fisher)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    # Wald test
    z = beta_hat / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z)))

    # 95% CI for OR
    or_vals = np.exp(beta_hat)
    or_lower = np.exp(beta_hat - 1.96 * se)
    or_upper = np.exp(beta_hat + 1.96 * se)

    return {
        "coefficients": beta_hat,
        "se": se,
        "z": z,
        "p_values": p_values,
        "OR": or_vals,
        "OR_lower": or_lower,
        "OR_upper": or_upper
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    results = {}

    # ── 1. Phi coefficients + p-values ────────────────────────────────────
    print("=" * 60)
    print("1. PHI COEFFICIENTS + χ² P-VALUES")
    print("=" * 60)

    phi_pairs = [
        ("deployed_in_plant", "uncertainty_quantification", "deployed_vs_UQ"),
        ("deployed_in_plant", "interpretability_flag", "deployed_vs_interp"),
        ("code_available", "data_available", "code_vs_data"),
        ("deployed_in_plant", "code_available", "deployed_vs_code"),
        ("deployed_in_plant", "data_available", "deployed_vs_data"),
    ]

    results["phi_tests"] = {}
    for var1, var2, label in phi_pairs:
        r = phi_test(df, var1, var2)
        results["phi_tests"][label] = r
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"  {label:30s}  phi={r['phi']:+.3f}  χ²={r['chi2']:.3f}  p={r['p']:.4f}  {sig}")

    # ── 2. Wilson score CIs ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("2. WILSON SCORE 95% CIs FOR KEY PROPORTIONS")
    print("=" * 60)

    n_total = len(df)
    key_proportions = [
        ("deployed", df["deployed_in_plant"].sum(), n_total),
        ("real_time", df["real_time_testing"].sum(), n_total),
        ("UQ", df["uncertainty_quantification"].sum(), n_total),
        ("real_ww", df["uses_real_wastewater"].sum(), n_total),
        ("code", df["code_available"].sum(), n_total),
        ("interpretability", df["interpretability_flag"].sum(), n_total),
    ]

    results["wilson_ci"] = {}
    for label, k, n in key_proportions:
        ci_lo, ci_hi = wilson_ci(int(k), int(n))
        pct = 100 * k / n
        results["wilson_ci"][label] = {
            "k": int(k), "n": int(n),
            "pct": round(pct, 1),
            "ci_lower": round(100 * ci_lo, 1),
            "ci_upper": round(100 * ci_hi, 1)
        }
        print(f"  {label:20s}  {int(k)}/{int(n)} = {pct:.1f}%  95% CI: [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%]")

    # ── 3. Fisher's exact: R² tiers vs deployment ─────────────────────────
    print(f"\n{'=' * 60}")
    print("3. FISHER'S EXACT: R² TIERS VS DEPLOYMENT")
    print("=" * 60)

    r2_studies = df[df["best_metric_type"].apply(
        lambda x: isinstance(x, str) and x.lower() == "r2"
    )].copy()
    r2_studies["r2_val"] = pd.to_numeric(r2_studies["best_metric_value"], errors="coerce")
    r2_valid = r2_studies.dropna(subset=["r2_val"])

    # Tier: >=0.99 vs <0.90
    high = r2_valid[r2_valid["r2_val"] >= 0.99]
    low = r2_valid[r2_valid["r2_val"] < 0.90]

    ct_r2 = np.array([
        [high["deployed_in_plant"].sum(), len(high) - high["deployed_in_plant"].sum()],
        [low["deployed_in_plant"].sum(), len(low) - low["deployed_in_plant"].sum()]
    ])
    or_r2, p_r2 = stats.fisher_exact(ct_r2)

    results["r2_tier_fisher"] = {
        "high_deployed": int(ct_r2[0, 0]), "high_total": int(ct_r2[0].sum()),
        "low_deployed": int(ct_r2[1, 0]), "low_total": int(ct_r2[1].sum()),
        "high_pct": round(100 * ct_r2[0, 0] / ct_r2[0].sum(), 1),
        "low_pct": round(100 * ct_r2[1, 0] / ct_r2[1].sum(), 1),
        "odds_ratio": round(or_r2, 3),
        "p": round(p_r2, 4)
    }
    print(f"  R²≥0.99: {ct_r2[0,0]}/{ct_r2[0].sum():.0f} = {100*ct_r2[0,0]/ct_r2[0].sum():.1f}%")
    print(f"  R²<0.90: {ct_r2[1,0]}/{ct_r2[1].sum():.0f} = {100*ct_r2[1,0]/ct_r2[1].sum():.1f}%")
    print(f"  OR = {or_r2:.3f}, p = {p_r2:.4f}")

    # ── 4. Firth penalized logistic regression ────────────────────────────
    print(f"\n{'=' * 60}")
    print("4. FIRTH PENALIZED LOGISTIC REGRESSION")
    print("=" * 60)

    # Prepare variables
    y = df["deployed_in_plant"].values.astype(float)

    # Model: deployed ~ UQ + interpretability + temporal_validation
    # (3 predictors for 22 events, within 1:7 rule)
    var_names = ["intercept", "UQ", "interpretability", "temporal_validation"]
    X = np.column_stack([
        np.ones(len(df)),
        df["uncertainty_quantification"].values,
        df["interpretability_flag"].values,
        df["temporal_validation"].values,
    ])

    firth_result = firth_logistic(X, y)

    results["firth_regression"] = {"model1_unadjusted": []}
    print("\n  Model 1 (unadjusted): deployed ~ UQ + interpretability + temporal_validation")
    print(f"  {'Variable':25s} {'OR':>8s} {'95% CI':>16s} {'p':>8s}")
    print(f"  {'-'*60}")
    for i, name in enumerate(var_names):
        row = {
            "variable": name,
            "OR": round(firth_result["OR"][i], 3),
            "OR_lower": round(firth_result["OR_lower"][i], 3),
            "OR_upper": round(firth_result["OR_upper"][i], 3),
            "p": round(firth_result["p_values"][i], 4),
            "coef": round(firth_result["coefficients"][i], 4),
            "se": round(firth_result["se"][i], 4)
        }
        results["firth_regression"]["model1_unadjusted"].append(row)
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {name:25s} {row['OR']:8.3f} [{row['OR_lower']:.3f}, {row['OR_upper']:.3f}] {row['p']:8.4f} {sig}")

    # Model 2: add sub_field (control vs coagulation vs others)
    # Binary: is_control, is_coagulation (reference = all others)
    df["is_control"] = (df["sub_field"] == "control").astype(int)
    df["is_coagulation"] = (df["sub_field"] == "coagulation").astype(int)

    var_names2 = ["intercept", "UQ", "interpretability", "temporal_validation",
                  "is_control", "is_coagulation"]
    X2 = np.column_stack([
        np.ones(len(df)),
        df["uncertainty_quantification"].values,
        df["interpretability_flag"].values,
        df["temporal_validation"].values,
        df["is_control"].values,
        df["is_coagulation"].values,
    ])

    # 5 predictors for 22 events (~1:4.4), borderline but acceptable for Firth
    firth_result2 = firth_logistic(X2, y)

    results["firth_regression"]["model2_adjusted"] = []
    print(f"\n  Model 2 (adjusted): + is_control + is_coagulation")
    print(f"  {'Variable':25s} {'OR':>8s} {'95% CI':>16s} {'p':>8s}")
    print(f"  {'-'*60}")
    for i, name in enumerate(var_names2):
        row = {
            "variable": name,
            "OR": round(firth_result2["OR"][i], 3),
            "OR_lower": round(firth_result2["OR_lower"][i], 3),
            "OR_upper": round(firth_result2["OR_upper"][i], 3),
            "p": round(firth_result2["p_values"][i], 4),
            "coef": round(firth_result2["coefficients"][i], 4),
            "se": round(firth_result2["se"][i], 4)
        }
        results["firth_regression"]["model2_adjusted"].append(row)
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {name:25s} {row['OR']:8.3f} [{row['OR_lower']:.3f}, {row['OR_upper']:.3f}] {row['p']:8.4f} {sig}")

    # ── 5. Coagulation comparisons (already in sec5) ──────────────────────
    print(f"\n{'=' * 60}")
    print("5. COAGULATION COMPARISONS (sec5)")
    print("=" * 60)

    coag = df[df["sub_field"] == "coagulation"]
    others = df[df["sub_field"] != "coagulation"]

    # Real WW: Fisher's exact
    ct_rw = np.array([
        [coag["uses_real_wastewater"].sum(), len(coag) - coag["uses_real_wastewater"].sum()],
        [others["uses_real_wastewater"].sum(), len(others) - others["uses_real_wastewater"].sum()]
    ])
    or_rw, p_rw = stats.fisher_exact(ct_rw)
    print(f"  Real WW (coag vs others): OR={or_rw:.3f}, p={p_rw:.4f}")
    results["coag_real_ww_fisher"] = {"OR": round(or_rw, 3), "p": round(p_rw, 4)}

    # Simulation: Fisher's exact (zero cell in coagulation)
    coag_sim = coag["data_source"].apply(lambda x: "simul" in str(x).lower()).sum()
    others_sim = others["data_source"].apply(lambda x: "simul" in str(x).lower()).sum()
    ct_sim = np.array([
        [coag_sim, len(coag) - coag_sim],
        [others_sim, len(others) - others_sim]
    ])
    or_sim, p_sim = stats.fisher_exact(ct_sim)
    print(f"  Simulation (coag vs others): OR={or_sim:.3f}, p={p_sim:.4f}")
    results["coag_sim_fisher"] = {"OR": round(or_sim, 3), "p": round(p_sim, 4),
                                   "coag_sim": int(coag_sim), "others_sim": int(others_sim),
                                   "others_pct": round(100 * others_sim / len(others), 1)}

    # Control sub-field: closed-loop Fisher's exact (ctrl vs others)
    ctrl = df[df["sub_field"] == "control"]
    others_ctrl = df[df["sub_field"] != "control"]
    ctrl_cl = ctrl["control_loop_type"].apply(lambda x: "closed" in str(x).lower()).sum()
    others_cl = others_ctrl["control_loop_type"].apply(lambda x: "closed" in str(x).lower()).sum()
    ct_cl = np.array([
        [ctrl_cl, len(ctrl) - ctrl_cl],
        [others_cl, len(others_ctrl) - others_cl]
    ])
    or_cl, p_cl = stats.fisher_exact(ct_cl)
    print(f"  Closed-loop (ctrl vs others): OR={or_cl:.3f}, p={p_cl:.4f}")
    results["ctrl_closedloop_fisher"] = {
        "OR": round(or_cl, 3), "p": round(p_cl, 4),
        "ctrl_pct": round(100 * ctrl_cl / len(ctrl), 1),
        "others_pct": round(100 * others_cl / len(others_ctrl), 1)
    }

    # Deploy: ctrl vs others (Fisher's exact)
    ctrl_dep = ctrl["deployed_in_plant"].sum()
    others_dep = others_ctrl["deployed_in_plant"].sum()
    ct_dep_ctrl = np.array([
        [ctrl_dep, len(ctrl) - ctrl_dep],
        [others_dep, len(others_ctrl) - others_dep]
    ])
    or_dep_ctrl, p_dep_ctrl = stats.fisher_exact(ct_dep_ctrl)
    print(f"  Deploy (ctrl vs others): OR={or_dep_ctrl:.3f}, p={p_dep_ctrl:.4f}")
    results["ctrl_deploy_fisher"] = {
        "OR": round(or_dep_ctrl, 3), "p": round(p_dep_ctrl, 4),
        "ctrl_pct": round(100 * ctrl_dep / len(ctrl), 1),
        "others_pct": round(100 * others_dep / len(others_ctrl), 1)
    }

    # Deploy: coag vs others (Fisher's exact)
    coag_dep = coag["deployed_in_plant"].sum()
    others_dep_coag = others["deployed_in_plant"].sum()
    ct_dep_coag = np.array([
        [coag_dep, len(coag) - coag_dep],
        [others_dep_coag, len(others) - others_dep_coag]
    ])
    or_dep_coag, p_dep_coag = stats.fisher_exact(ct_dep_coag)
    print(f"  Deploy (coag vs others): OR={or_dep_coag:.3f}, p={p_dep_coag:.4f}")
    results["coag_deploy_fisher"] = {
        "OR": round(or_dep_coag, 3), "p": round(p_dep_coag, 4),
        "coag_pct": round(100 * coag_dep / len(coag), 1),
        "others_pct": round(100 * others_dep_coag / len(others), 1)
    }

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
