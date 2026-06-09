#!/usr/bin/env python3
"""
Corpus robustness checks for the npj Clean Water Perspective.

The script quantifies how headline deployment indicators change after common
sensitivity filters and provides a conservative keyword screen for hybrid or
physics-informed modelling.
"""

import json
import re

import pandas as pd


from paths import CONVERTED_FULLTEXT_DIR, FIGURES_DIR, ROOT, SOURCE_POOL_JSON

DATA_PATH = SOURCE_POOL_JSON
CONVERTED = CONVERTED_FULLTEXT_DIR
OUT_JSON = FIGURES_DIR / "corpus_robustness_stats.json"
OUT_SUBSETS = FIGURES_DIR / "corpus_robustness_subsets.csv"


def _fulltext_keys():
    keys = set()
    for d in CONVERTED.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^\d+_(.+)$", d.name)
        if m:
            keys.add(m.group(1).lower())
    return keys
OUT_FLAGS = FIGURES_DIR / "piml_keyword_flags.csv"

CONFERENCE_PATTERN = re.compile(
    r"conference|proceeding|symposium|workshop|ieee international|acm|iop conf|epic series",
    flags=re.IGNORECASE,
)

HIGH_IMPACT_JOURNALS = {
    "Water Research",
    "Environmental Science & Technology",
    "ACS ES&T Water",
    "ACS ES&T Engineering",
    "npj Clean Water",
    "Nature Water",
    "Journal of Hazardous Materials",
    "Science of The Total Environment",
    "Science of the Total Environment",
    "Environment International",
    "Journal of Environmental Management",
    "Journal of Water Process Engineering",
    "Water Research X",
}

CORE_WATER_JOURNALS = HIGH_IMPACT_JOURNALS | {
    "Water Environment Research",
    "Water Science and Technology",
    "Journal of Hydroinformatics",
    "Water",
    "AQUA — Water Infrastructure, Ecosystems and Society",
    "Water Supply",
    "Water Reuse",
    "Membranes",
}

PIML_PATTERN = re.compile(
    r"physics[- ]informed|physically informed|mechanistic|hybrid|grey[- ]box|"
    r"digital twin|benchmark simulation model|BSM1|BSM2|activated sludge model|"
    r"\bASM1\b|\bASM2\b|\bASM3\b|mass balance|first[- ]principles|"
    r"model predictive control|MPC|reinforcement learning",
    flags=re.IGNORECASE,
)


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["results"])
    for col in [
        "deployed_in_plant",
        "real_time_testing",
        "uncertainty_quantification",
        "code_available",
        "data_available",
        "uses_real_wastewater",
    ]:
        df[col] = df[col].astype(bool)
    df["interpretability_flag"] = df["interpretability_method"].apply(
        lambda x: False if x in (None, "", "none", "None") else True
    )
    df["is_conference_like"] = df["journal"].fillna("").str.contains(CONFERENCE_PATTERN)
    df["is_article"] = df.get("crossref_type", "").fillna("") == "journal-article"
    df["is_2018_2025"] = df["year"].between(2018, 2025)
    df["is_high_impact_journal"] = df["journal"].isin(HIGH_IMPACT_JOURNALS)
    df["is_core_water_journal"] = df["journal"].isin(CORE_WATER_JOURNALS)
    ft = _fulltext_keys()
    df["has_fulltext"] = df["doi"].fillna("").str.lower().str.replace("/", "_", regex=False).isin(ft)
    text = (
        df["paper_title"].fillna("")
        + " "
        + df["target_variable"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
        + " "
        + df["ml_algorithms"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
        + " "
        + df["best_algorithm"].fillna("").astype(str)
    )
    df["piml_hybrid_keyword_flag"] = text.str.contains(PIML_PATTERN)
    return df


def pct(series):
    return round(float(series.mean() * 100), 1) if len(series) else 0.0


def subset_summary(name, df):
    r2 = df[df["best_metric_type"].fillna("").str.lower() == "r2"].copy()
    r2["best_metric_value"] = pd.to_numeric(r2["best_metric_value"], errors="coerce")
    r2 = r2.dropna(subset=["best_metric_value"])
    return {
        "subset": name,
        "n": int(len(df)),
        "deployed_n": int(df["deployed_in_plant"].sum()),
        "deployed_pct": pct(df["deployed_in_plant"]),
        "real_time_pct": pct(df["real_time_testing"]),
        "temporal_external_walkforward_pct": pct(
            df["validation_method"].fillna("").str.contains("temporal|external|walk_forward", case=False)
        ),
        "uq_pct": pct(df["uncertainty_quantification"]),
        "code_pct": pct(df["code_available"]),
        "data_pct": pct(df["data_available"]),
        "interpretability_pct": pct(df["interpretability_flag"]),
        "r2_n": int(len(r2)),
        "r2_median": round(float(r2["best_metric_value"].median()), 3) if len(r2) else None,
        "piml_hybrid_keyword_n": int(df["piml_hybrid_keyword_flag"].sum()),
        "piml_hybrid_keyword_pct": pct(df["piml_hybrid_keyword_flag"]),
    }


def main():
    df = load_data()
    in_window = df["is_2018_2025"]
    fulltext = df["has_fulltext"]
    analysis = df[in_window & fulltext]   # the n=423 analysis corpus
    subsets = {
        "full_500": df,
        "year_2018_2025": df[in_window],
        "year_2018_2025_fulltext": analysis,
        "analysis_article_only": analysis[analysis["is_article"]],
        "analysis_conference_excluded": analysis[~analysis["is_conference_like"]],
        "analysis_high_impact_journal": analysis[analysis["is_high_impact_journal"]],
        "analysis_core_water_journal": analysis[analysis["is_core_water_journal"]],
    }
    subset_rows = [subset_summary(name, sub.copy()) for name, sub in subsets.items()]
    subset_df = pd.DataFrame(subset_rows)

    year_outliers = df[~df["is_2018_2025"]][
        ["doi", "paper_title", "year", "journal", "sub_field", "deployed_in_plant"]
    ].sort_values(["year", "journal"])

    flags = df[df["piml_hybrid_keyword_flag"]][
        ["doi", "paper_title", "year", "journal", "sub_field", "best_algorithm", "ml_algorithms"]
    ].sort_values(["sub_field", "year"])

    stats = {
        "metadata": {
            "data_file": str(DATA_PATH.relative_to(ROOT)),
            "total_records": int(len(df)),
            "note": "PIML/hybrid counts are conservative keyword-screen counts and require manual confirmation before causal claims.",
        },
        "year_out_of_scope": {
            "n": int(len(year_outliers)),
            "deployed_n": int(year_outliers["deployed_in_plant"].sum()),
            "records": year_outliers.to_dict(orient="records"),
        },
        "conference_like": {
            "n": int(df["is_conference_like"].sum()),
            "pct": pct(df["is_conference_like"]),
        },
        "subset_summaries": subset_rows,
        "journal_composition": {
            "high_impact_n": int(df["is_high_impact_journal"].sum()),
            "high_impact_pct": pct(df["is_high_impact_journal"]),
            "core_water_n": int(df["is_core_water_journal"].sum()),
            "core_water_pct": pct(df["is_core_water_journal"]),
        },
        "piml_hybrid_keyword": {
            "n": int(df["piml_hybrid_keyword_flag"].sum()),
            "pct": pct(df["piml_hybrid_keyword_flag"]),
            "by_subfield": df.groupby("sub_field")["piml_hybrid_keyword_flag"].agg(["sum", "count"]).assign(
                pct=lambda x: (x["sum"] / x["count"] * 100).round(1)
            ).reset_index().to_dict(orient="records"),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(OUT_SUBSETS, index=False)
    flags.to_csv(OUT_FLAGS, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved: {OUT_SUBSETS}")
    print(f"Saved: {OUT_FLAGS}")
    print(f"Saved: {OUT_JSON}")
    print(subset_df.to_string(index=False))


if __name__ == "__main__":
    main()
