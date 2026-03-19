#!/usr/bin/env python3
"""
Fig 1: The ML landscape in water treatment.

Layout:
  (a) R² violin plot by sub-field (full width, top)
  (b) Annual publication trends (stacked bar)
  (c) Algorithm usage heatmap (sub-field × algorithm)
  (d) Sub-field distribution (horizontal bar)
  (e) R² tier vs deployment rate

Data source: data/extracted/fulltext_extraction_500_cleaned.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (
    apply_plot_style, save_figure, SUBFIELD_COLORS, SUBFIELD_ORDER,
    MACARON_COLORS, FONT_SIZES, LINE_WIDTHS, DPI,
    FIGURE_SIZE_FULL
)

# ── Data loading ──────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data/extracted/fulltext_extraction_500_cleaned.json"
OUTPUT_PATH = Path(__file__).parent.parent / "figures/fig1_landscape.png"

with open(DATA_PATH) as f:
    raw = json.load(f)
df = pd.DataFrame(raw["results"])

# Use all 500 papers (no year filter) for consistency with manuscript
print(f"Total papers: {len(df)}")

# ── Panel data preparation ────────────────────────────────────────────────

# (a) R² violin: papers with R² metric
df_r2 = df[df["best_metric_type"] == "R2"].copy()
df_r2["best_metric_value"] = pd.to_numeric(df_r2["best_metric_value"], errors="coerce")
df_r2 = df_r2.dropna(subset=["best_metric_value"])
print(f"Papers with R²: {len(df_r2)}")

# (b) Annual trends by sub-field
year_sf = df.groupby(["year", "sub_field"]).size().unstack(fill_value=0)
year_sf = year_sf.reindex(columns=SUBFIELD_ORDER, fill_value=0)

# (c) Algorithm heatmap
# Explode ml_algorithms list, count per sub-field
algo_rows = []
for _, row in df.iterrows():
    algos = row["ml_algorithms"]
    if isinstance(algos, list):
        for a in algos:
            algo_rows.append({"sub_field": row["sub_field"], "algorithm": a})
algo_df = pd.DataFrame(algo_rows)

# Top algorithms (by total count)
TOP_ALGOS = ["ANN", "RF", "LR", "SVR", "LSTM", "XGBoost", "SVM", "DT", "CNN", "RL"]
algo_df_top = algo_df[algo_df["algorithm"].isin(TOP_ALGOS)]
algo_heat = algo_df_top.groupby(["sub_field", "algorithm"]).size().unstack(fill_value=0)
algo_heat = algo_heat.reindex(index=SUBFIELD_ORDER, fill_value=0)
algo_heat = algo_heat.reindex(columns=TOP_ALGOS, fill_value=0)
# Normalize by sub-field total papers
sf_counts = df["sub_field"].value_counts()
algo_heat_pct = algo_heat.div(sf_counts[algo_heat.index], axis=0) * 100

# (d) Sub-field distribution
sf_dist = df["sub_field"].value_counts().reindex(SUBFIELD_ORDER)

# (e) R² tier vs deployment rate
bins = [0, 0.90, 0.95, 0.99, 1.01]
labels = ["<0.90", "0.90–0.95", "0.95–0.99", "≥0.99"]
df_r2["r2_tier"] = pd.cut(df_r2["best_metric_value"], bins=bins, labels=labels, right=False)
tier_deploy = df_r2.groupby("r2_tier", observed=False).agg(
    n=("deployed_in_plant", "count"),
    deployed=("deployed_in_plant", "sum")
).reset_index()
tier_deploy["rate"] = tier_deploy["deployed"] / tier_deploy["n"] * 100

# ── Figure creation ───────────────────────────────────────────────────────

apply_plot_style()

fig = plt.figure(figsize=(7.2, 9.0), dpi=DPI)
gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[1, 1, 1],
    hspace=0.35,
    wspace=0.35
)

# ── (a) Violin plot — top row, full width ─────────────────────────────────

ax_a = fig.add_subplot(gs[0, :])

violin_data = []
violin_positions = []
violin_colors = []
violin_labels = []

for i, sf in enumerate(SUBFIELD_ORDER):
    vals = df_r2[df_r2["sub_field"] == sf]["best_metric_value"].values
    if len(vals) > 0:
        violin_data.append(vals)
        violin_positions.append(i)
        violin_colors.append(SUBFIELD_COLORS[sf])
        n = len(vals)
        med = np.median(vals)
        violin_labels.append(f"{sf}\n(n={n})")

parts = ax_a.violinplot(
    violin_data, positions=violin_positions,
    showmeans=False, showmedians=False, showextrema=False
)

for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(violin_colors[i])
    pc.set_edgecolor("white")
    pc.set_alpha(0.7)
    pc.set_linewidth(0.5)

# Overlay strip plot (jittered points)
for i, (vals, pos) in enumerate(zip(violin_data, violin_positions)):
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
    ax_a.scatter(
        pos + jitter, vals,
        s=8, color=violin_colors[i], alpha=0.5,
        edgecolors="white", linewidths=0.3, zorder=3
    )
    # Median line
    med = np.median(vals)
    ax_a.hlines(med, pos - 0.25, pos + 0.25, colors="black", linewidths=1.5, zorder=4)

ax_a.set_xticks(violin_positions)
ax_a.set_xticklabels(violin_labels, fontsize=FONT_SIZES["tick_label"])
ax_a.set_ylabel("Best reported R²", fontsize=FONT_SIZES["axis_label"])
ax_a.set_ylim(0.0, 1.05)
ax_a.axhline(y=0.95, color="#CC3311", linestyle="--", linewidth=0.8, zorder=1)
ax_a.text(6.6, 0.955, "R² = 0.95", fontsize=FONT_SIZES["annotation"],
          color="#CC3311", ha="right", va="bottom")
ax_a.set_title("a", fontsize=FONT_SIZES["title"], fontweight="bold",
               loc="left", x=-0.02)
ax_a.text(0.98, 0.05, f"n = {len(df_r2)} studies reporting R²",
          transform=ax_a.transAxes, fontsize=FONT_SIZES["annotation"],
          ha="right", va="bottom", fontweight="bold", color="black")

# ── (b) Annual trends — middle left ──────────────────────────────────────

ax_b = fig.add_subplot(gs[1, 0])

bottom = np.zeros(len(year_sf))
years = year_sf.index.values
for sf in SUBFIELD_ORDER:
    vals = year_sf[sf].values
    ax_b.bar(years, vals, bottom=bottom, color=SUBFIELD_COLORS[sf],
             label=sf, width=0.7, edgecolor="white", linewidth=0.3)
    bottom += vals

ax_b.set_xlabel("Year", fontsize=FONT_SIZES["axis_label"])
ax_b.set_ylabel("Number of studies", fontsize=FONT_SIZES["axis_label"])
ax_b.set_xticks(years)
ax_b.set_xticklabels(years, rotation=45, ha="right", fontsize=FONT_SIZES["tick_label"])
ax_b.legend(fontsize=FONT_SIZES["legend"], ncol=2, loc="upper left", framealpha=0.9,
            handlelength=1.0, handletextpad=0.3, columnspacing=0.5)
ax_b.set_title("b", fontsize=FONT_SIZES["title"], fontweight="bold",
               loc="left", x=-0.02)

# ── (c) Algorithm heatmap — middle right ─────────────────────────────────

ax_c = fig.add_subplot(gs[1, 1])

im = ax_c.imshow(algo_heat_pct.values, cmap="YlGnBu", aspect="auto",
                 vmin=0, vmax=80)

ax_c.set_xticks(range(len(TOP_ALGOS)))
ax_c.set_xticklabels(TOP_ALGOS, fontsize=FONT_SIZES["tick_label"],
                     rotation=45, ha="right")
ax_c.set_yticks(range(len(SUBFIELD_ORDER)))
ax_c.set_yticklabels(SUBFIELD_ORDER, fontsize=FONT_SIZES["tick_label"])

# Annotate cells with percentage
for i in range(len(SUBFIELD_ORDER)):
    for j in range(len(TOP_ALGOS)):
        val = algo_heat_pct.values[i, j]
        if val > 0:
            color = "white" if val > 40 else "black"
            ax_c.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=5.5, color=color)

cbar = fig.colorbar(im, ax=ax_c, shrink=0.8, pad=0.02)
cbar.set_label("% of studies", fontsize=FONT_SIZES["annotation"])
cbar.ax.tick_params(labelsize=FONT_SIZES["annotation"])
ax_c.set_title("c", fontsize=FONT_SIZES["title"], fontweight="bold",
               loc="left", x=-0.02)

# ── (d) Sub-field distribution — bottom left ─────────────────────────────

ax_d = fig.add_subplot(gs[2, 0])

colors_d = [SUBFIELD_COLORS[sf] for sf in SUBFIELD_ORDER]
bars = ax_d.barh(range(len(sf_dist)), sf_dist.values, color=colors_d,
                 edgecolor="white", linewidth=0.3)

ax_d.set_yticks(range(len(sf_dist)))
ax_d.set_yticklabels(SUBFIELD_ORDER, fontsize=FONT_SIZES["tick_label"])
ax_d.set_xlabel("Number of studies", fontsize=FONT_SIZES["axis_label"])
ax_d.invert_yaxis()

# Annotate with n values
for i, (v, sf) in enumerate(zip(sf_dist.values, SUBFIELD_ORDER)):
    ax_d.text(v + 2, i, f"n={v}", va="center", fontsize=FONT_SIZES["annotation"])

ax_d.set_title("d", fontsize=FONT_SIZES["title"], fontweight="bold",
               loc="left", x=-0.02)

# ── (e) R² tier vs deployment rate — bottom right ────────────────────────

ax_e = fig.add_subplot(gs[2, 1])

x_pos = range(len(tier_deploy))
bar_colors = [MACARON_COLORS["baby_blue"]] * len(tier_deploy)

bars_e = ax_e.bar(x_pos, tier_deploy["rate"].values, color=bar_colors,
                  width=0.6, edgecolor="white", linewidth=0.3)

ax_e.set_xticks(x_pos)
ax_e.set_xticklabels(tier_deploy["r2_tier"].values, fontsize=FONT_SIZES["tick_label"])
ax_e.set_xlabel("R² performance tier", fontsize=FONT_SIZES["axis_label"])
ax_e.set_ylabel("Deployment rate (%)", fontsize=FONT_SIZES["axis_label"])

# Annotate with n and deployed count
for i, row in tier_deploy.iterrows():
    ax_e.text(i, row["rate"] + 0.3,
             f'{row["rate"]:.1f}%\n({int(row["deployed"])}/{int(row["n"])})',
             ha="center", va="bottom", fontsize=FONT_SIZES["annotation"])

# Add corpus average line
corpus_deploy_rate = df["deployed_in_plant"].sum() / len(df) * 100
ax_e.axhline(y=corpus_deploy_rate, color="#CC3311",
             linestyle="--", linewidth=0.8)
ax_e.text(3.5, corpus_deploy_rate + 0.2, f"Corpus avg: {corpus_deploy_rate:.1f}%",
          fontsize=FONT_SIZES["annotation"], color="#CC3311",
          ha="right", va="bottom")

ax_e.set_title("e", fontsize=FONT_SIZES["title"], fontweight="bold",
               loc="left", x=-0.02)

# ── Save ──────────────────────────────────────────────────────────────────

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
save_figure(fig, OUTPUT_PATH)
print(f"Done. Saved to {OUTPUT_PATH}")
