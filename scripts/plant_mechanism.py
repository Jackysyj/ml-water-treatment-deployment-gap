#!/usr/bin/env python3
"""
Six-plant sparse-sensing mechanism analysis for supplementary diagnostics.

Where plant_panel.py establishes WHAT (influent monitoring poorly predicts
compliance-critical effluent across six plants), this script establishes WHY,
with quantities computed identically across all six plants:

  (a) Influent non-stationarity: standardized mean difference (SMD) of every
      influent variable between the early and late half of each plant's record.
      Distribution drift is why retrospective shuffled CV is optimistic and
      walk-forward forecasting collapses.
  (b) Signal is genuine but bounded: best-model core-compliance R2 under random
      5-fold CV vs a shuffled-target null. Real > null confirms the models
      capture true signal; the small magnitude confirms the signal is limited.
  (c) Narrow reliance on measured influent: permutation importance for the one
      core target present at every plant (effluent COD), over the shared influent
      panel (COD, NH3-N, TN, TP, TSS, pH). Shows the model leans on a few
      influent proxies, insufficient for effluent governed by unobserved state.

Plants are ANONYMISED (Plant 1..6). Derived outputs only; raw records stay under
utility confidentiality.

Outputs:
  figures/plant_mechanism_stats.json
  figures/plant_mechanism_per_feature.csv
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from paths import FIGURES_DIR, PLANT_PRIVATE_DIR

DATA = PLANT_PRIVATE_DIR
OUT_JSON = FIGURES_DIR / "plant_mechanism_stats.json"
OUT_CSV = FIGURES_DIR / "plant_mechanism_per_feature.csv"

RANDOM_STATE = 42
N_SPLITS = 5

# Shared influent panel present at all six plants -> canonical labels.
SHARED_IN = {
    "COD含量（重铬酸钾法）": "COD",
    "氨氮（快速法）": "NH3-N",
    "总氮": "TN",
    "总磷": "TP",
    "悬浮物": "TSS",
    "PH值": "pH",
}
CORE_OUT = {"CODout": "COD", "氨氮out": "NH3-N", "总氮out": "TN",
            "总磷out": "TP", "悬浮物out": "TSS"}
REP_TARGET = "CODout"  # present at every plant

PLANTS = [
    ("wsc.xlsx", "Plant 1"),
    ("wsc1.xlsx", "Plant 2"),
    ("wsc2.xlsx", "Plant 3"),
    ("wsc3.xlsx", "Plant 4"),
    ("wsc4.xlsx", "Plant 5"),
    ("wsc5.xlsx", "Plant 6"),
]


def get_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RF": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                    random_state=RANDOM_STATE, n_jobs=-1),
        "HGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             random_state=RANDOM_STATE),
    }


def load_plant(fname):
    df = pd.read_excel(DATA / fname)
    df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def best_random_cv(X, y, shuffle_y=False):
    kf = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    yv = y.values.copy()
    if shuffle_y:
        rng = np.random.RandomState(RANDOM_STATE)
        yv = rng.permutation(yv)
    best = -np.inf
    for m in get_models().values():
        sc = []
        for tr, te in kf.split(X):
            mdl = clone(m).fit(X.iloc[tr], yv[tr])
            sc.append(r2_score(yv[te], mdl.predict(X.iloc[te])))
        best = max(best, float(np.mean(sc)))
    return best


def best_model_for(X, y):
    kf = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best, best_m = -np.inf, None
    for name, m in get_models().items():
        sc = [r2_score(y.iloc[te], clone(m).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))
              for tr, te in kf.split(X)]
        if np.mean(sc) > best:
            best, best_m = float(np.mean(sc)), name
    return best_m


def analyse_plant(fname, label):
    df = load_plant(fname)
    pred = [c for c in df.columns if c != "time" and not c.endswith("out")]
    outs = {k: v for k, v in CORE_OUT.items() if k in df.columns}
    for c in pred + list(outs):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sub = df.dropna(subset=pred + list(outs)).reset_index(drop=True)
    sub["date"] = sub["time"].dt.date
    X = sub[pred]

    # ── (a) influent non-stationarity: early vs late half SMD ──────────────
    uniq = sorted(sub["date"].unique())
    cut = uniq[len(uniq) // 2]
    early = sub[sub["date"] < cut]
    late = sub[sub["date"] >= cut]

    def smd_of(col):
        e, l = early[col], late[col]
        sd = np.sqrt((e.var(ddof=1) + l.var(ddof=1)) / 2)
        return float((l.mean() - e.mean()) / sd) if sd > 0 else 0.0

    smds = np.array([abs(smd_of(c)) for c in pred])
    mean_abs_smd = float(np.mean(smds))
    frac_drift = float(np.mean(smds >= 0.5))
    n_drift = int(np.sum(smds >= 0.5))
    # per-shared-feature |SMD| for the cross-plant heatmap (rectangular 6x6)
    shared_smd = {SHARED_IN[c]: round(abs(smd_of(c)), 3)
                  for c in SHARED_IN if c in sub.columns}

    # ── (b) real vs shuffled-null best core R2 (mean across core targets) ──
    real, null = [], []
    for oc in outs:
        y = sub[oc]
        real.append(best_random_cv(X, y, shuffle_y=False))
        null.append(best_random_cv(X, y, shuffle_y=True))
    real_mean = float(np.mean(real))
    null_mean = float(np.mean(null))

    # ── (c) permutation importance for effluent COD over shared influent ───
    shared_cols = [c for c in SHARED_IN if c in sub.columns]
    yc = sub[REP_TARGET]
    best_name = best_model_for(X[shared_cols], yc)
    Xtr, Xte, ytr, yte = train_test_split(
        X[shared_cols], yc, test_size=0.25, random_state=RANDOM_STATE)
    mdl = clone(get_models()[best_name]).fit(Xtr, ytr)
    pi = permutation_importance(mdl, Xte, yte, n_repeats=20,
                                random_state=RANDOM_STATE, scoring="r2")
    imp = np.clip(pi["importances_mean"], 0, None)
    share = imp / imp.sum() if imp.sum() > 0 else np.zeros_like(imp)
    imp_share = {SHARED_IN[c]: round(float(s), 3) for c, s in zip(shared_cols, share)}
    top_feat = max(imp_share, key=lambda k: imp_share[k]) if imp_share else None

    return {
        "plant": label,
        "n_complete": int(len(sub)),
        "n_predictors": len(pred),
        "mean_abs_smd": round(mean_abs_smd, 3),
        "frac_features_drift": round(frac_drift, 3),
        "n_features_drift": n_drift,
        "shared_smd": shared_smd,
        "real_core_r2": round(real_mean, 3),
        "null_core_r2": round(null_mean, 3),
        "cod_best_model": best_name,
        "cod_importance_share": imp_share,
        "cod_top_feature": top_feat,
        "cod_top_share": round(float(max(imp_share.values())), 3) if imp_share else None,
    }


def main():
    per_plant = []
    feat_rows = []
    for fname, label in PLANTS:
        s = analyse_plant(fname, label)
        per_plant.append(s)
        for feat in s["shared_smd"]:
            feat_rows.append({
                "plant": label,
                "feature": feat,
                "smd_abs": s["shared_smd"][feat],
                "importance_share": s["cod_importance_share"].get(feat),
            })
        print(f"{label}: mean|SMD|={s['mean_abs_smd']} ({s['n_features_drift']} drift) "
              f"real={s['real_core_r2']} null={s['null_core_r2']} "
              f"COD top={s['cod_top_feature']} ({s['cod_top_share']})")

    out = {
        "n_plants": len(per_plant),
        "shared_influent_panel": list(SHARED_IN.values()),
        "representative_target": "effluent COD",
        "notes": ("(a) influent |SMD| early vs late half; (b) best core R2 vs "
                  "shuffled-target null, 5-fold CV; (c) permutation importance "
                  "for effluent COD over shared influent panel. Plants anonymised."),
        "plants": per_plant,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    pd.DataFrame(feat_rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
