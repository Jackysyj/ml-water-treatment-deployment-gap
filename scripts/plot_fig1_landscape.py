#!/usr/bin/env python3
"""
Fig. 1: Published evidence map for the npj Clean Water Perspective.

The figure condenses the former literature landscape and deployment-practice
figures into one Perspective-style evidence map.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (  # noqa: E402
    DPI,
    FONT_SIZES,
    HEATMAP_CMAP,
    MACARON_COLORS,
    SUBFIELD_COLORS,
    SUBFIELD_ORDER,
    apply_plot_style,
    save_figure,
)
from paths import ANALYSIS_CORPUS_JSON, FIGURES_DIR  # noqa: E402


DATA_PATH = ANALYSIS_CORPUS_JSON
STATS_PATH = FIGURES_DIR / "manuscript_stats.json"
OUTPUT_PATH = FIGURES_DIR / "fig1_landscape.png"
SUMMARY_PATH = FIGURES_DIR / "fig1_data_summary.csv"

TOP_ALGOS = ["ANN", "RF", "LR", "SVR", "LSTM", "XGBoost", "SVM", "DT", "CNN", "RL"]
R2_TIERS = ["<0.90", "0.90-0.95", "0.95-0.99", ">=0.99"]


def panel(ax, label: str):
    ax.text(
        -0.08,
        1.12,
        f"({label})",
        transform=ax.transAxes,
        fontsize=FONT_SIZES["title"] + 1,
        fontweight="bold",
        va="top",
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center * 100, max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def main():
    raw = json.load(open(DATA_PATH, encoding="utf-8"))
    df = pd.DataFrame(raw["results"])
    ms = json.load(open(STATS_PATH, encoding="utf-8"))

    df_r2 = df[df["best_metric_type"].str.lower() == "r2"].copy()
    df_r2["best_metric_value"] = pd.to_numeric(df_r2["best_metric_value"], errors="coerce")
    df_r2 = df_r2.dropna(subset=["best_metric_value"])

    year_sf = (
        df[df["year"] >= 2018]
        .groupby(["year", "sub_field"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SUBFIELD_ORDER, fill_value=0)
    )

    algo_rows = []
    for _, row in df.iterrows():
        algos = row.get("ml_algorithms")
        if isinstance(algos, list):
            for algo in algos:
                algo_rows.append({"sub_field": row["sub_field"], "algorithm": algo})
    algo_df = pd.DataFrame(algo_rows)
    algo_heat = (
        algo_df[algo_df["algorithm"].isin(TOP_ALGOS)]
        .groupby(["sub_field", "algorithm"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SUBFIELD_ORDER, fill_value=0)
        .reindex(columns=TOP_ALGOS, fill_value=0)
    )
    sf_counts = df["sub_field"].value_counts().reindex(SUBFIELD_ORDER)
    algo_heat_pct = algo_heat.div(sf_counts, axis=0) * 100

    practices = [
        ("Plant deployment", ms["deployed_pct"], ms["deployed_n"]),
        ("Real-time testing", ms["realtime_pct"], round(ms["realtime_pct"] * ms["total_papers"] / 100)),
        ("Future-facing\nvalidation", 18.2, 77),
        ("Uncertainty\nquantification", ms["uq_pct"], round(ms["uq_pct"] * ms["total_papers"] / 100)),
        ("Public code", ms["code_pct"], round(ms["code_pct"] * ms["total_papers"] / 100)),
        ("Public data", ms["data_avail_pct"], round(ms["data_avail_pct"] * ms["total_papers"] / 100)),
    ]

    bins = [0, 0.90, 0.95, 0.99, 1.01]
    labels = ["<0.90", "0.90-0.95", "0.95-0.99", ">=0.99"]
    df_r2["r2_tier"] = pd.cut(df_r2["best_metric_value"], bins=bins, labels=labels, right=False)
    tier = (
        df_r2.groupby("r2_tier", observed=False)
        .agg(n=("deployed_in_plant", "count"), deployed=("deployed_in_plant", "sum"))
        .reset_index()
    )
    tier["rate"] = tier["deployed"] / tier["n"] * 100
    tier["ci_lower"] = 0.0
    tier["ci_upper"] = 0.0
    for idx, row in tier.iterrows():
        _, lo, hi = wilson_ci(int(row["deployed"]), int(row["n"]))
        tier.at[idx, "ci_lower"] = lo
        tier.at[idx, "ci_upper"] = hi

    apply_plot_style()
    fig = plt.figure(figsize=(7.2, 8.0), dpi=DPI)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 1.0], hspace=0.48, wspace=0.36)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    violin_data = []
    violin_positions = []
    violin_colours = []
    violin_labels = []
    for i, sf in enumerate(SUBFIELD_ORDER):
        vals = df_r2[df_r2["sub_field"] == sf]["best_metric_value"].values
        if len(vals):
            violin_data.append(vals)
            violin_positions.append(i)
            violin_colours.append(SUBFIELD_COLORS[sf])
            violin_labels.append(f"{sf}\n(n={len(vals)})")
    parts = ax_a.violinplot(
        violin_data,
        positions=violin_positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc, colour in zip(parts["bodies"], violin_colours):
        pc.set_facecolor(colour)
        pc.set_edgecolor("white")
        pc.set_alpha(0.72)
        pc.set_linewidth(0.5)
    rng = np.random.default_rng(42)
    for vals, pos, colour in zip(violin_data, violin_positions, violin_colours):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax_a.scatter(pos + jitter, vals, s=8, color=colour, alpha=0.55, edgecolors="white", linewidths=0.25)
        ax_a.hlines(np.median(vals), pos - 0.25, pos + 0.25, colors="black", linewidths=1.3)
    ax_a.set_xticks(violin_positions)
    ax_a.set_xticklabels(violin_labels)
    ax_a.set_ylabel("Best reported R²")
    ax_a.set_ylim(0, 1.05)
    ax_a.text(
        0.98,
        0.06,
        f"median R² = {ms['r2_median']:.2f}; n = {ms['r2_n']}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZES["annotation"] + 1,
        fontweight="bold",
    )
    panel(ax_a, "a")

    bottom = np.zeros(len(year_sf))
    years = year_sf.index.values
    for sf in SUBFIELD_ORDER:
        vals = year_sf[sf].values
        ax_b.bar(years, vals, bottom=bottom, color=SUBFIELD_COLORS[sf], width=0.72, edgecolor="white", linewidth=0.25, label=sf)
        bottom += vals
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Number of studies")
    ax_b.set_xticks(years)
    ax_b.set_xticklabels(years, rotation=45, ha="right")
    ax_b.legend(frameon=False, fontsize=FONT_SIZES["legend"], ncol=2, loc="upper left", handlelength=1.0, columnspacing=0.6)
    panel(ax_b, "b")

    im = ax_c.imshow(algo_heat_pct.values, cmap=HEATMAP_CMAP, aspect="auto", vmin=0, vmax=80)
    ax_c.set_xticks(range(len(TOP_ALGOS)))
    ax_c.set_xticklabels(TOP_ALGOS, rotation=45, ha="right")
    ax_c.set_yticks(range(len(SUBFIELD_ORDER)))
    ax_c.set_yticklabels(SUBFIELD_ORDER)
    for i in range(len(SUBFIELD_ORDER)):
        for j in range(len(TOP_ALGOS)):
            val = algo_heat_pct.values[i, j]
            if val > 0:
                ax_c.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="white" if val > 40 else "#333333")
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.82, pad=0.02)
    cbar.set_label("% of studies")
    cbar.ax.tick_params(labelsize=FONT_SIZES["annotation"])
    panel(ax_c, "c")

    practice_labels = [p[0] for p in practices][::-1]
    practice_pcts = [p[1] for p in practices][::-1]
    practice_ns = [p[2] for p in practices][::-1]
    colours = [
        MACARON_COLORS["baby_blue"],
        MACARON_COLORS["sky_blue"],
        MACARON_COLORS["mint"],
        MACARON_COLORS["lavender"],
        MACARON_COLORS["peach"],
        MACARON_COLORS["sakura"],
    ][::-1]
    y = np.arange(len(practice_labels))
    ax_d.barh(y, practice_pcts, color=colours, edgecolor="white", linewidth=0.4)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(practice_labels)
    ax_d.set_xlabel("Studies reporting practice (%)")
    ax_d.set_xlim(0, 22)
    for yi, pct, n in zip(y, practice_pcts, practice_ns):
        ax_d.text(pct + 0.5, yi, f"{pct:.1f}%\n(n={n})", va="center", fontsize=FONT_SIZES["annotation"])
    panel(ax_d, "d")

    x = np.arange(len(tier))
    yerr_lower = tier["rate"].values - tier["ci_lower"].values
    yerr_upper = tier["ci_upper"].values - tier["rate"].values
    ax_e.bar(x, tier["rate"].values, color=MACARON_COLORS["baby_blue"], edgecolor="white", linewidth=0.4, width=0.62)
    ax_e.errorbar(x, tier["rate"].values, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)
    ax_e.axhline(ms["deployed_pct"], color="#CC3311", linestyle="--", linewidth=0.8)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(tier["r2_tier"].astype(str).values)
    ax_e.set_xlabel("Reported R² tier")
    ax_e.set_ylabel("Deployment rate (%)")
    ax_e.set_ylim(0, max(15, float(tier["ci_upper"].max()) + 2))
    for xi, row in tier.iterrows():
        ax_e.text(
            xi,
            row["rate"] + 0.35,
            f'{row["rate"]:.1f}%\n({int(row["deployed"])}/{int(row["n"])})',
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"],
        )
    ax_e.text(0.03, 0.96, "Descriptive only:\n5 deployed events", transform=ax_e.transAxes, ha="left", va="top", fontsize=FONT_SIZES["annotation"])
    panel(ax_e, "e")

    summary_rows = []
    for sf in SUBFIELD_ORDER:
        vals = df_r2[df_r2["sub_field"] == sf]["best_metric_value"]
        summary_rows.append({"panel": "a", "key": f"R2_{sf}", "value": f"{vals.median():.3f}" if len(vals) else "NA", "n": len(vals), "deployed": ""})
    summary_rows.append({"panel": "a", "key": "R2_total", "value": f"{df_r2['best_metric_value'].median():.3f}", "n": len(df_r2), "deployed": ""})
    for label, pct, n in practices:
        summary_rows.append({"panel": "d", "key": label.replace("\n", " "), "value": f"{pct:.1f}", "n": n, "deployed": ""})
    for _, row in tier.iterrows():
        summary_rows.append({
            "panel": "e",
            "key": f"tier_{row['r2_tier']}",
            "value": f"{row['rate']:.1f}",
            "n": int(row["n"]),
            "deployed": int(row["deployed"]),
        })
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["panel", "key", "value", "n", "deployed"])
        writer.writeheader()
        writer.writerows(summary_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUTPUT_PATH)
    print(f"Data summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
