#!/usr/bin/env python3
"""
Fig. 2: Bounded six-plant operational illustration for the Perspective.

The figure centres the original operational contribution rather than comparing
plant performance with literature-reported R2 values. It shows the data panel,
feedforward influent-only ML, simple operating baselines and an autoregressive
augmented scenario under the same date-based validation protocol.

Panels:
  (a) Six-plant dataset overview.
  (b) Feedforward influent-only random CV versus walk-forward by plant.
  (c) Walk-forward scenario distributions across plant-target combinations.
  (d) Per-target feedforward walk-forward R2 heatmap.
  (e) Effluent CV% per plant (min-max across core targets).
  (f) Per-target feedforward random-CV versus walk-forward R2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (  # noqa: E402
    DPI,
    FONT_SIZES,
    HEATMAP_CMAP,
    LINE_WIDTHS,
    MACARON_COLORS,
    apply_plot_style,
    save_figure,
)
from paths import FIGURES_DIR  # noqa: E402

STATS = FIGURES_DIR / "plant_panel_stats.json"
PERTGT = FIGURES_DIR / "plant_panel_per_target.csv"
OUT = FIGURES_DIR / "fig2_operational_illustration.png"
SUMMARY = FIGURES_DIR / "fig2_operational_illustration_summary.csv"

PLANT_ORDER = ["Plant 1", "Plant 2", "Plant 3", "Plant 4", "Plant 5", "Plant 6"]
PLANT_COLORS = {
    "Plant 1": MACARON_COLORS["baby_blue"],
    "Plant 2": MACARON_COLORS["sky_blue"],
    "Plant 3": MACARON_COLORS["mint"],
    "Plant 4": MACARON_COLORS["lavender"],
    "Plant 5": MACARON_COLORS["sakura"],
    "Plant 6": MACARON_COLORS["peach"],
}
SCENARIO_COLORS = {
    "Mean": "#BDBDBD",
    "Persistence": MACARON_COLORS["cream"],
    "Feedforward ML": MACARON_COLORS["baby_blue"],
    "Autoregressive ML": MACARON_COLORS["sakura"],
}
REF_GREY = "#555555"
CORE_TARGETS = ["COD", "NH3-N", "TN", "TP", "TSS"]


def panel(ax, letter: str):
    ax.set_title(f"({letter})", fontsize=FONT_SIZES["title"], fontweight="bold", loc="left")


def short_name(plant: str) -> str:
    return plant.replace("Plant ", "P")


def main():
    apply_plot_style()
    stats = json.load(open(STATS, encoding="utf-8"))
    df = pd.read_csv(PERTGT)
    plants = stats["plants"]
    by = {p["plant"]: p for p in plants}
    order = sorted(PLANT_ORDER, key=lambda p: -by[p]["core_random_cv_mean"])
    short = [short_name(p) for p in order]
    x = np.arange(len(order))

    fig = plt.figure(figsize=(7.2, 9.0), dpi=DPI)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.62, wspace=0.36)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    # (a) Dataset overview.
    records = [by[p]["n_complete"] for p in order]
    predictors = [by[p]["n_predictors"] for p in order]
    bars = ax_a.bar(x, records, color=[PLANT_COLORS[p] for p in order],
                    edgecolor="white", linewidth=0.6)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short)
    ax_a.set_ylabel("Complete records", fontsize=FONT_SIZES["axis_label"])
    ax_a.set_ylim(0, max(records) * 1.26)
    for b, n_pred in zip(bars, predictors):
        ax_a.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(records) * 0.035,
            f"{n_pred} vars",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"] - 1,
        )
    ax_a.text(0.98, 0.94, "6 plants\n6,006 records",
              transform=ax_a.transAxes, ha="right", va="top",
              fontsize=FONT_SIZES["annotation"], fontweight="bold")
    panel(ax_a, "a")

    # (b) Feedforward influent-only random CV versus walk-forward by plant.
    cv = [by[p]["core_random_cv_mean"] for p in order]
    wf = [by[p]["core_walk_forward_mean"] for p in order]
    w = 0.38
    ax_b.bar(x - w / 2, cv, w, color=[PLANT_COLORS[p] for p in order],
             edgecolor="white", linewidth=0.5, label="Random CV")
    ax_b.bar(x + w / 2, wf, w, color=MACARON_COLORS["cream"],
             edgecolor="#B8A642", linewidth=0.5, hatch="///", label="Walk-forward")
    ax_b.axhline(0, color="#999999", linewidth=0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(short)
    ax_b.set_ylabel("Feedforward influent-only R²", fontsize=FONT_SIZES["axis_label"])
    ax_b.set_ylim(min(-0.15, min(wf) - 0.04), max(0.55, max(cv) + 0.07))
    ax_b.legend(frameon=False, fontsize=FONT_SIZES["legend"], loc="upper right")
    panel(ax_b, "b")

    # (c) Scenario distributions under walk-forward.
    scenario_cols = [
        ("Mean", "walk_forward_mean_baseline_r2"),
        ("Persistence", "walk_forward_persistence_r2"),
        ("Feedforward ML", "walk_forward_influent_only_r2"),
        ("Autoregressive ML", "walk_forward_influent_plus_lagged_effluent_r2"),
    ]
    positions = np.arange(len(scenario_cols))
    data = [df[col].dropna().values for _, col in scenario_cols]
    bp = ax_c.boxplot(data, positions=positions, widths=0.46, patch_artist=True,
                      showfliers=False, medianprops={"color": "black", "linewidth": 1.1})
    for patch, (label, _) in zip(bp["boxes"], scenario_cols):
        patch.set_facecolor(SCENARIO_COLORS[label])
        patch.set_edgecolor("white")
        patch.set_alpha(0.85)
    rng = np.random.default_rng(42)
    for i, (label, col) in enumerate(scenario_cols):
        vals = df[col].dropna().values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_c.scatter(
            np.repeat(i, len(vals)) + jitter,
            vals,
            s=18,
            color=SCENARIO_COLORS[label],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.85,
            zorder=3,
        )
    ax_c.axhline(0, color="#999999", linewidth=0.7)
    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(["Mean", "Persist.", "Feed-\nforward", "Influent+\nlag"], fontsize=FONT_SIZES["tick_label"])
    ax_c.set_ylabel("Walk-forward R²", fontsize=FONT_SIZES["axis_label"])
    ax_c.set_ylim(min(-1.35, np.nanmin(np.concatenate(data)) - 0.08), 0.35)
    ax_c.text(0.04, 0.94, "n = 29 plant-targets",
              transform=ax_c.transAxes, ha="left", va="top",
              fontsize=FONT_SIZES["annotation"], fontweight="bold")
    panel(ax_c, "c")

    # (d) Feedforward walk-forward R2 heatmap.
    mat = np.full((len(order), len(CORE_TARGETS)), np.nan)
    for i, plant in enumerate(order):
        sub = df[df["plant"] == plant]
        for j, target in enumerate(CORE_TARGETS):
            row = sub[sub["target"] == target]
            if len(row):
                mat[i, j] = row["walk_forward_influent_only_r2"].iloc[0]
    im = ax_d.imshow(mat, cmap=HEATMAP_CMAP, vmin=-0.15, vmax=0.20, aspect="auto")
    ax_d.set_xticks(range(len(CORE_TARGETS)))
    ax_d.set_xticklabels(CORE_TARGETS, rotation=30, ha="right")
    ax_d.set_yticks(range(len(order)))
    ax_d.set_yticklabels(short)
    for i in range(len(order)):
        for j in range(len(CORE_TARGETS)):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                ax_d.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=FONT_SIZES["annotation"] - 1,
                          color="white" if val > 0.12 else "#333333")
    cb = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cb.set_label("walk-forward R²", fontsize=FONT_SIZES["annotation"])
    cb.ax.tick_params(labelsize=FONT_SIZES["annotation"] - 1)
    panel(ax_d, "d")

    # (e) Effluent CV% per plant.
    lo = [by[p]["effluent_cv_pct_range"][0] for p in order]
    hi = [by[p]["effluent_cv_pct_range"][1] for p in order]
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    err = [[m - a for m, a in zip(mid, lo)], [b - m for b, m in zip(hi, mid)]]
    y = np.arange(len(order))
    ax_e.errorbar(mid, y, xerr=err, fmt="o", color=MACARON_COLORS["baby_blue"],
                  ecolor="#999999", capsize=3, markersize=6,
                  markeredgecolor="white", markeredgewidth=0.6)
    for i, plant in enumerate(order):
        ax_e.plot(mid[i], i, "o", color=PLANT_COLORS[plant], markersize=6,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax_e.axvline(15, color=REF_GREY, linestyle=":", linewidth=LINE_WIDTHS["secondary"])
    ax_e.text(15, len(order) - 0.3, "near-constant\nthreshold", color=REF_GREY,
              ha="left", va="top", fontsize=FONT_SIZES["annotation"] - 1)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(short)
    ax_e.set_xlabel("Effluent CV (%)", fontsize=FONT_SIZES["axis_label"])
    ax_e.set_xlim(0, max(hi) + 12)
    panel(ax_e, "e")

    # (f) Feedforward random CV versus walk-forward per target.
    for plant in order:
        sub = df[df["plant"] == plant]
        ax_f.scatter(sub["influent_only_r2"], sub["walk_forward_influent_only_r2"],
                     color=PLANT_COLORS[plant], edgecolors="white", linewidths=0.5,
                     s=34, label=short_name(plant))
    lim_min = min(df["influent_only_r2"].min(), df["walk_forward_influent_only_r2"].min(), -0.12) - 0.03
    lim_max = max(df["influent_only_r2"].max(), df["walk_forward_influent_only_r2"].max(), 0.5) + 0.03
    ax_f.plot([lim_min, lim_max], [lim_min, lim_max], color=REF_GREY,
              linestyle="--", linewidth=LINE_WIDTHS["secondary"])
    ax_f.axhline(0, color="#999999", linewidth=0.7)
    ax_f.axvline(0, color="#999999", linewidth=0.7)
    ax_f.set_xlim(lim_min, lim_max)
    ax_f.set_ylim(lim_min, lim_max)
    ax_f.set_xlabel("Random-CV R²", fontsize=FONT_SIZES["axis_label"])
    ax_f.set_ylabel("Walk-forward R²", fontsize=FONT_SIZES["axis_label"])
    ax_f.legend(frameon=False, fontsize=FONT_SIZES["legend"] - 1, ncol=2,
                loc="lower right", handletextpad=0.2, columnspacing=0.6)
    panel(ax_f, "f")

    save_figure(fig, OUT)

    rows = []
    rows.append({
        "panel": "a",
        "plant": "all",
        "records": sum(records),
        "predictors": f"{min(predictors)}-{max(predictors)}",
    })
    rows.append({
        "panel": "b",
        "plant": "best/median/worst",
        "random_cv": f"{max(cv):.3f}/{np.median(cv):.3f}/{min(cv):.3f}",
        "walk_forward": f"{max(wf):.3f}/{np.median(wf):.3f}/{min(wf):.3f}",
    })
    for label, col in scenario_cols:
        vals = df[col].dropna()
        rows.append({
            "panel": "c",
            "scenario": label,
            "median": round(float(vals.median()), 3),
            "mean": round(float(vals.mean()), 3),
            "n": int(vals.count()),
        })
    for plant in order:
        rows.append({
            "panel": "e",
            "plant": plant,
            "effluent_cv_lo": by[plant]["effluent_cv_pct_range"][0],
            "effluent_cv_hi": by[plant]["effluent_cv_pct_range"][1],
        })
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(f"Saved: {OUT}")
    print(f"Saved: {SUMMARY}")


if __name__ == "__main__":
    main()
