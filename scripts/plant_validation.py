#!/usr/bin/env python3
"""
Operational validation on the RAW (non-imputed) wastewater treatment plant
dataset, using the real sampling timestamps.

Rationale for using the raw file (data/plant_private/raw_wwtp_exports/wsc.xlsx):
  * It carries a genuine timestamp column (采样时间), so chronological / walk-
    forward evaluation rests on real time, not an assumed row order.
  * It has NO missing cells and NO post-hoc normalisation, so reported numbers
    reflect the plant's actual values, not an imputation/normalisation artefact.
  * Provenance check: the "filled/normalised" file (wsc（填补后）.xlsx) contains a
    TP(Out) target that does NOT exist in the raw export. TP(Out) is therefore a
    constructed column and is excluded from the primary operational test; core
    compliance targets here are COD/NH3-N/TN/TSS.

Three evaluation regimes are reported side by side so the manuscript can be
explicit about what each one means:
  * random_cv            -> retrospective shuffled K-fold (the publication-style
                            evaluation this paper argues is optimistic)
  * chronological_holdout-> date-based last-block holdout (near-future)
  * walk_forward         -> date-based expanding-window forecast (true operation)

Duplicate timestamps (762 rows share a timestamp = multiple grab samples at the
same logged time) are handled by splitting on DATE, never on row index, so the
same timestamp can never straddle train and test.

All derived manuscript numbers are written to figures/plant_stats.json.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from paths import FIGURES_DIR, PLANT_PRIVATE_DIR, ROOT

RAW_PATH = PLANT_PRIVATE_DIR / "wsc.xlsx"
IMPUTED_PATH = PLANT_PRIVATE_DIR / "wsc（填补后）.xlsx"
IMPUTED_SHEET = "Sheet1（规范后）"

OUT_JSON = FIGURES_DIR / "plant_stats.json"
OUT_CSV = FIGURES_DIR / "plant_validation_results.csv"
OUT_IMPORTANCE = FIGURES_DIR / "plant_feature_importance.csv"
OUT_CONFORMAL = FIGURES_DIR / "plant_conformal_results.csv"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_FRACTION = 0.2   # fraction of DAYS held out as the chronological test block
CAL_FRACTION = 0.2    # fraction of DAYS used as the conformal calibration block

# Raw Chinese column -> canonical English name used throughout the manuscript.
RAW_RENAME = {
    "采样时间": "time",
    "COD含量（重铬酸钾法）": "COD(In)",
    "氨氮（快速法）": "NH3-N(In)",
    "PH值": "pH(In)",
    "总氮": "TN(In)",
    "石油类": "TPH(In)",
    "色度": "chromaticity(In)",
    "悬浮物": "TSS(In)",
    "硝基苯": "NB(In)",
    "氰化物": "CN(In)",
    "硫化物（比色法）": "SF(In)",
    "总磷": "TP(In)",
    "挥发酚（溴化容量法）": "VP(In)",
    "甲醛": "FCOH(In)",
    "CODout": "COD(Out)",
    "总氮out": "TN(Out)",
    "氨氮out": "NH3-N(Out)",
    "PH值out": "pH(Out)",
    "石油类out": "TPH(Out)",
    "悬浮物out": "TSS(Out)",
    "挥发酚out": "VP(Out)",
    "硫化物out": "SF(Out)",
    "氰化物out": "CN(Out)",
}

FEATURE_COLS = [
    "COD(In)", "NH3-N(In)", "pH(In)", "TN(In)", "TPH(In)", "chromaticity(In)",
    "TSS(In)", "NB(In)", "CN(In)", "SF(In)", "TP(In)", "VP(In)", "FCOH(In)",
]

# TP(Out) is intentionally absent: it does not exist in the raw export.
TARGET_GROUPS = {
    "core": ["COD(Out)", "NH3-N(Out)", "TN(Out)", "TSS(Out)"],
    "auxiliary": ["pH(Out)", "TPH(Out)", "VP(Out)", "SF(Out)", "CN(Out)"],
}
TARGETS = TARGET_GROUPS["core"] + TARGET_GROUPS["auxiliary"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(target, group, split, model_name, fold, y_true, y_pred):
    return {
        "target": target,
        "group": group,
        "split": split,
        "model": model_name,
        "fold": fold,
        "n_test": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
    }


def get_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RF": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HGB": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE,
        ),
    }


def target_group(target):
    for group, targets in TARGET_GROUPS.items():
        if target in targets:
            return group
    raise KeyError(target)


def load_raw_plant_data():
    """Load the raw export, parse timestamps, sort by time, keep modelled cols."""
    df = pd.read_excel(RAW_PATH)
    df = df.rename(columns=RAW_RENAME)
    if "time" not in df.columns:
        raise ValueError("Timestamp column '采样时间' not found in raw file.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    keep = ["time"] + FEATURE_COLS + TARGETS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in raw file: {missing}")
    df = df[keep].copy()
    for col in FEATURE_COLS + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)
    df["date"] = df["time"].dt.date
    return df


def date_blocks(df, n_blocks):
    """Split unique sorted DATES into n_blocks contiguous groups.

    Returns a list of sets of dates, ordered in time.
    """
    dates = sorted(df["date"].unique())
    edges = np.linspace(0, len(dates), n_blocks + 1).astype(int)
    return [set(dates[edges[i]:edges[i + 1]]) for i in range(n_blocks)]


def chronological_holdout_masks(df):
    """Date-based last-block holdout: last TEST_FRACTION of DAYS = test."""
    dates = sorted(df["date"].unique())
    cut = int(len(dates) * (1 - TEST_FRACTION))
    test_dates = set(dates[cut:])
    test_mask = df["date"].isin(test_dates).values
    return ~test_mask, test_mask


def random_cv_results(df, target, group, models):
    """Retrospective shuffled K-fold (publication-style optimism)."""
    X = df[FEATURE_COLS]
    y = df[target]
    rows = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for model_name, model in models.items():
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            est = clone(model)
            est.fit(X.iloc[tr], y.iloc[tr])
            pred = est.predict(X.iloc[te])
            rows.append(metric_row(target, group, "random_cv", model_name, fold,
                                   y.iloc[te], pred))
    return rows


def walk_forward_results(df, target, group, models):
    """Date-based expanding window: fold k trains on blocks[0..k], tests block k+1."""
    blocks = date_blocks(df, N_SPLITS + 1)
    X = df[FEATURE_COLS]
    y = df[target]
    rows = []
    for model_name, model in models.items():
        for fold in range(1, N_SPLITS + 1):
            train_dates = set().union(*blocks[:fold])
            test_dates = blocks[fold]
            tr = df["date"].isin(train_dates).values
            te = df["date"].isin(test_dates).values
            if tr.sum() == 0 or te.sum() == 0:
                continue
            est = clone(model)
            est.fit(X[tr], y[tr])
            pred = est.predict(X[te])
            rows.append(metric_row(target, group, "walk_forward", model_name, fold,
                                   y[te], pred))
    return rows


def holdout_results(df, target, group, models):
    """Random vs date-based chronological single holdout + baselines + null."""
    X = df[FEATURE_COLS]
    y = df[target]
    rows = []

    # date-based chronological holdout
    tr_mask, te_mask = chronological_holdout_masks(df)
    X_tr_t, X_te_t = X[tr_mask], X[te_mask]
    y_tr_t, y_te_t = y[tr_mask], y[te_mask]

    # random holdout of the SAME test size (row-level shuffle)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(X))
    n_test = int(te_mask.sum())
    te_r = np.zeros(len(X), dtype=bool)
    te_r[idx[:n_test]] = True
    X_tr_r, X_te_r = X[~te_r], X[te_r]
    y_tr_r, y_te_r = y[~te_r], y[te_r]

    for model_name, model in models.items():
        est = clone(model)
        est.fit(X_tr_r, y_tr_r)
        rows.append(metric_row(target, group, "random_holdout", model_name, 1,
                               y_te_r, est.predict(X_te_r)))
        est = clone(model)
        est.fit(X_tr_t, y_tr_t)
        rows.append(metric_row(target, group, "chronological_holdout", model_name, 1,
                               y_te_t, est.predict(X_te_t)))

    dummy = DummyRegressor(strategy="mean").fit(X_tr_t, y_tr_t)
    rows.append(metric_row(target, group, "chronological_holdout", "MeanBaseline", 1,
                           y_te_t, dummy.predict(X_te_t)))

    shuffled = pd.Series(rng.permutation(y_tr_t.values), index=y_tr_t.index)
    est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                        random_state=RANDOM_STATE)
    est.fit(X_tr_t, shuffled)
    rows.append(metric_row(target, group, "null_shuffled_y", "HGB", 1,
                           y_te_t, est.predict(X_te_t)))
    return rows


def summarize_best_cv(results_df):
    cv = results_df[results_df["split"] == "random_cv"].copy()
    by_model = (
        cv.groupby(["target", "group", "model"], as_index=False)
        .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
             mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"),
             n_folds=("fold", "count"))
        .sort_values(["target", "r2_mean"], ascending=[True, False])
    )
    return by_model.groupby("target", as_index=False).head(1).reset_index(drop=True)


def walk_forward_summary(results_df):
    wf = results_df[results_df["split"] == "walk_forward"].copy()
    # best model per target by mean walk-forward R2
    by_model = (
        wf.groupby(["target", "group", "model"], as_index=False)
        .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"))
        .sort_values(["target", "r2_mean"], ascending=[True, False])
    )
    return by_model.groupby("target", as_index=False).head(1).reset_index(drop=True)


def feature_shift(df):
    tr_mask, te_mask = chronological_holdout_masks(df)
    rows = []
    for col in FEATURE_COLS:
        early = df[col][tr_mask]
        late = df[col][te_mask]
        pooled = np.sqrt((early.var(ddof=1) + late.var(ddof=1)) / 2)
        smd = 0.0 if pooled == 0 else (late.mean() - early.mean()) / pooled
        rows.append({
            "feature": col,
            "early_mean": float(early.mean()),
            "late_mean": float(late.mean()),
            "smd": float(smd),
            "ks_statistic": float(ks_2samp(early, late).statistic),
        })
    return pd.DataFrame(rows).sort_values("smd", key=lambda s: s.abs(), ascending=False)


def feature_importance(df, best_df):
    X = df[FEATURE_COLS]
    tr_mask, te_mask = chronological_holdout_masks(df)
    X_tr, X_te = X[tr_mask], X[te_mask]
    rows = []
    models = get_models()
    for _, best in best_df.iterrows():
        target, model_name = best["target"], best["model"]
        y = df[target]
        y_tr, y_te = y[tr_mask], y[te_mask]
        est = clone(models[model_name]).fit(X_tr, y_tr)
        result = permutation_importance(est, X_te, y_te, n_repeats=10,
                                        random_state=RANDOM_STATE, scoring="r2", n_jobs=-1)
        for feat, mean, sd in zip(FEATURE_COLS, result.importances_mean, result.importances_std):
            rows.append({
                "target": target, "group": target_group(target), "model": model_name,
                "feature": feat, "importance_mean": float(mean), "importance_sd": float(sd),
            })
    return pd.DataFrame(rows)


def conformal_intervals(df, best_df):
    """Split-conformal with date-based train/cal/test segments."""
    X = df[FEATURE_COLS]
    dates = sorted(df["date"].unique())
    train_end = int(len(dates) * (1 - TEST_FRACTION - CAL_FRACTION))
    cal_end = int(len(dates) * (1 - TEST_FRACTION))
    train_dates = set(dates[:train_end])
    cal_dates = set(dates[train_end:cal_end])
    test_dates = set(dates[cal_end:])
    tr = df["date"].isin(train_dates).values
    ca = df["date"].isin(cal_dates).values
    te = df["date"].isin(test_dates).values

    rows = []
    models = get_models()
    for _, best in best_df.iterrows():
        target, model_name = best["target"], best["model"]
        y = df[target]
        est = clone(models[model_name]).fit(X[tr], y[tr])
        cal_pred = est.predict(X[ca])
        test_pred = est.predict(X[te])
        q = float(np.quantile(np.abs(y[ca].values - cal_pred), 0.9, method="higher"))
        lower, upper = test_pred - q, test_pred + q
        coverage = float(np.mean((y[te].values >= lower) & (y[te].values <= upper)))
        rows.append({
            "target": target, "group": target_group(target), "model": model_name,
            "nominal_coverage": 0.9, "empirical_coverage": coverage,
            "mean_interval_width": float(np.mean(upper - lower)),
            "calibration_n": int(ca.sum()), "test_n": int(te.sum()),
            "absolute_error_quantile": q,
        })
    return pd.DataFrame(rows)


def imputed_sensitivity():
    """Re-run random_cv best-R2 on the imputed/normalised file for the SHARED
    targets, so the manuscript can show the headline is not a raw-file artefact."""
    try:
        imp = pd.read_excel(IMPUTED_PATH, sheet_name=IMPUTED_SHEET)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    shared = [t for t in TARGETS if t in imp.columns]
    feats = [c for c in FEATURE_COLS if c in imp.columns]
    for c in feats + shared:
        imp[c] = pd.to_numeric(imp[c], errors="coerce")
    imp = imp.dropna(subset=feats + shared)
    X = imp[feats]
    models = get_models()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for t in shared:
        y = imp[t]
        best = -np.inf
        for _, model in models.items():
            scores = []
            for tr, te in kf.split(X):
                est = clone(model).fit(X.iloc[tr], y.iloc[tr])
                scores.append(r2_score(y.iloc[te], est.predict(X.iloc[te])))
            best = max(best, float(np.mean(scores)))
        out[t] = round(best, 3)
    return {"available": True, "best_random_cv_r2": out}


def group_mean(best_df, group, col="r2_mean"):
    sub = best_df[best_df["group"] == group]
    return round(float(sub[col].mean()), 3)


def build_stats(df, results, best_cv, wf_best, shift, conformal, imputed_sens):
    best_records = []
    for _, row in best_cv.sort_values(["group", "target"]).iterrows():
        best_records.append({
            "target": row["target"], "group": row["group"], "best_model": row["model"],
            "r2_mean": round(float(row["r2_mean"]), 3), "r2_sd": round(float(row["r2_sd"]), 3),
            "mae_mean": round(float(row["mae_mean"]), 3), "rmse_mean": round(float(row["rmse_mean"]), 3),
        })

    wf_records = []
    wf_map = {r["target"]: r for _, r in wf_best.iterrows()}
    for t in TARGETS:
        if t in wf_map:
            r = wf_map[t]
            wf_records.append({
                "target": t, "group": target_group(t), "best_model": r["model"],
                "r2_mean": round(float(r["r2_mean"]), 3), "r2_sd": round(float(r["r2_sd"]), 3),
            })

    group_summary = {}
    for group in ["core", "auxiliary"]:
        sub = best_cv[best_cv["group"] == group]
        wf_sub = wf_best[wf_best["group"] == group]
        group_summary[group] = {
            "targets": TARGET_GROUPS[group],
            "random_cv_mean_best_r2": round(float(sub["r2_mean"].mean()), 3),
            "random_cv_median_best_r2": round(float(sub["r2_mean"].median()), 3),
            "random_cv_min_best_r2": round(float(sub["r2_mean"].min()), 3),
            "random_cv_max_best_r2": round(float(sub["r2_mean"].max()), 3),
            "walk_forward_mean_best_r2": round(float(wf_sub["r2_mean"].mean()), 3),
        }

    # random vs chronological holdout drop
    holdout = results[results["split"].isin(["random_holdout", "chronological_holdout"])]
    hb = (holdout[holdout["model"] != "MeanBaseline"]
          .sort_values(["target", "split", "r2"], ascending=[True, True, False])
          .groupby(["target", "split"], as_index=False).head(1))
    pivot = hb.pivot(index="target", columns="split", values="r2")
    temporal_drop = {}
    for t, row in pivot.iterrows():
        if "random_holdout" in row and "chronological_holdout" in row:
            temporal_drop[t] = round(float(row["random_holdout"] - row["chronological_holdout"]), 3)

    null_rows = results[results["split"] == "null_shuffled_y"]
    null_r2 = {r["target"]: round(float(r["r2"]), 3) for _, r in null_rows.iterrows()}

    conf_summary = {}
    for group in ["core", "auxiliary"]:
        sub = conformal[conformal["group"] == group]
        conf_summary[group] = {
            "mean_coverage": round(float(sub["empirical_coverage"].mean()), 3),
            "mean_interval_width": round(float(sub["mean_interval_width"].mean()), 3),
        }

    n_dates = df["date"].nunique()
    dup_ts = int(df["time"].duplicated(keep=False).sum())

    return {
        "metadata": {
            "data_file": str(RAW_PATH.relative_to(ROOT)),
            "data_source": "raw export with genuine sampling timestamps (no imputation, no normalisation)",
            "n_samples": int(len(df)),
            "n_features": len(FEATURE_COLS),
            "n_unique_days": int(n_dates),
            "time_min": str(df["time"].min()),
            "time_max": str(df["time"].max()),
            "feature_columns": FEATURE_COLS,
            "targets": TARGETS,
            "tp_out_note": "TP(Out) excluded: it is absent from the raw export (present only in the filled/normalised file as a constructed column).",
            "duplicate_timestamp_rows": dup_ts,
            "split_policy": "All temporal splits are DATE-based so duplicated timestamps never straddle train/test.",
            "missing_data_note": "Raw export has no missing cells; no imputation was applied in this analysis.",
        },
        "regime_definitions": {
            "random_cv": "Retrospective shuffled 5-fold CV (publication-style; argued to be optimistic).",
            "chronological_holdout": "Date-based last-20%-of-days holdout (near-future).",
            "walk_forward": "Date-based expanding-window forecast over 6 ordered date blocks (true operational extrapolation).",
        },
        "best_random_cv": best_records,
        "best_walk_forward": wf_records,
        "group_summary": group_summary,
        "headline": {
            "core_random_cv_mean_r2": group_mean(best_cv, "core"),
            "auxiliary_random_cv_mean_r2": group_mean(best_cv, "auxiliary"),
            "core_walk_forward_mean_r2": round(float(wf_best[wf_best["group"] == "core"]["r2_mean"].mean()), 3),
        },
        "temporal_holdout_r2_drop_random_minus_chronological": temporal_drop,
        "null_shuffled_y_r2": null_r2,
        "top_feature_shift": shift.head(8).round(3).to_dict(orient="records"),
        "conformal_summary": conf_summary,
        "imputed_sensitivity": imputed_sens,
    }


def main():
    df = load_raw_plant_data()
    models = get_models()

    rows = []
    for target in TARGETS:
        group = target_group(target)
        rows.extend(random_cv_results(df, target, group, models))
        rows.extend(walk_forward_results(df, target, group, models))
        rows.extend(holdout_results(df, target, group, models))

    results = pd.DataFrame(rows)
    best_cv = summarize_best_cv(results)
    wf_best = walk_forward_summary(results)
    shift = feature_shift(df)
    importance = feature_importance(df, best_cv)
    conformal = conformal_intervals(df, best_cv)
    imputed_sens = imputed_sensitivity()
    stats = build_stats(df, results, best_cv, wf_best, shift, conformal, imputed_sens)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    importance.to_csv(OUT_IMPORTANCE, index=False)
    conformal.to_csv(OUT_CONFORMAL, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_IMPORTANCE}")
    print(f"Saved: {OUT_CONFORMAL}")
    print(f"Saved: {OUT_JSON}")
    print("\n=== HEADLINE ===")
    print(json.dumps(stats["headline"], indent=2, ensure_ascii=False))
    print("\n=== GROUP SUMMARY ===")
    print(json.dumps(stats["group_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
