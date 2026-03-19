#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main analysis script for Nature Water Analysis paper.

Validates 4 core arguments with descriptive statistics and cross-tabulations.
Outputs a single JSON file as the authoritative data source for manuscript writing.

Usage:
  python scripts/analysis_main.py
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "extracted" / "fulltext_extraction_500_cleaned.json"
OUTPUT_FILE = DATA_DIR / "analysis" / "descriptive_stats.json"

SUBFIELD_ORDER = ['WWTP', 'monitoring', 'control', 'sludge', 'membrane', 'coagulation', 'DBP']

BOOL_FIELDS = [
    'real_time_testing', 'deployed_in_plant', 'uses_real_wastewater',
    'code_available', 'data_available', 'uncertainty_quantification',
]
CAT_FIELDS = [
    'data_source', 'scale', 'model_framework', 'control_loop_type',
    'validation_method', 'interpretability_method',
]


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def safe_str(v):
    if v is None:
        return "null"
    return str(v).lower().strip()


# ============================================================
# Part 1: Descriptive Statistics
# ============================================================

def part1_overview(results):
    print("\n" + "=" * 70)
    print("PART 1: DESCRIPTIVE STATISTICS")
    print("=" * 70)

    total = len(results)
    print(f"\nTotal papers: {total}")

    # Year distribution
    years = [r.get("year") for r in results if r.get("year")]
    year_counts = Counter(years)
    print(f"\nYear range: {min(years)}-{max(years)}")
    print("Year distribution:")
    by_year = {}
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]}")
        by_year[str(y)] = year_counts[y]

    # Sub-field distribution
    sf_counts = Counter(r.get("sub_field", "unknown") for r in results)
    print("\nSub-field distribution:")
    by_subfield = {}
    for sf in SUBFIELD_ORDER:
        n = sf_counts.get(sf, 0)
        print(f"  {sf:15s}: {n:4d} ({n/total*100:.1f}%)")
        by_subfield[sf] = n

    # Algorithm frequency (top 15)
    algo_counter = Counter()
    for r in results:
        algos = r.get("ml_algorithms", [])
        if isinstance(algos, list):
            for a in algos:
                algo_counter[str(a).strip()] += 1
    print("\nTop 15 algorithms:")
    top_algos = {}
    for algo, cnt in algo_counter.most_common(15):
        print(f"  {algo:25s}: {cnt:4d}")
        top_algos[algo] = cnt

    # Metric type distribution
    mt_counts = Counter(safe_str(r.get("best_metric_type")) for r in results)
    print("\nMetric type distribution:")
    metric_types = {}
    for mt, cnt in mt_counts.most_common():
        print(f"  {mt:15s}: {cnt:4d} ({cnt/total*100:.1f}%)")
        metric_types[mt] = cnt

    # Dataset size by sub-field
    print("\nDataset size by sub-field (median [IQR]):")
    ds_by_sf = {}
    for sf in SUBFIELD_ORDER:
        sizes = []
        for r in results:
            if r.get("sub_field") == sf and r.get("dataset_size") is not None:
                try:
                    sizes.append(float(r["dataset_size"]))
                except (ValueError, TypeError):
                    pass
        if sizes:
            med = statistics.median(sizes)
            q1 = sorted(sizes)[len(sizes) // 4]
            q3 = sorted(sizes)[3 * len(sizes) // 4]
            print(f"  {sf:15s}: median={med:,.0f}, IQR=[{q1:,.0f}, {q3:,.0f}], n={len(sizes)}")
            ds_by_sf[sf] = {"median": med, "q1": q1, "q3": q3, "n": len(sizes)}
        else:
            print(f"  {sf:15s}: no data")
            ds_by_sf[sf] = None

    # Scale distribution
    scale_counts = Counter(safe_str(r.get("scale")) for r in results)
    print("\nScale distribution:")
    scale_dist = {}
    for s, cnt in scale_counts.most_common():
        print(f"  {s:15s}: {cnt:4d} ({cnt/total*100:.1f}%)")
        scale_dist[s] = cnt

    return {
        "total": total,
        "year_range": [min(years), max(years)],
        "by_year": by_year,
        "by_subfield": by_subfield,
        "top_algorithms": top_algos,
        "metric_types": metric_types,
        "dataset_size_by_subfield": ds_by_sf,
        "scale_distribution": scale_dist,
    }


# ============================================================
# Part 2: Barrier 1 — Data Quality Mismatch
# ============================================================

def part2_data_quality(results):
    print("\n" + "=" * 70)
    print("PART 2: BARRIER 1 — DATA QUALITY MISMATCH")
    print("=" * 70)

    total = len(results)

    # data_source
    ds_counts = Counter(safe_str(r.get("data_source")) for r in results)
    print("\nData source:")
    data_source = {}
    for k, v in ds_counts.most_common():
        print(f"  {k:20s}: {v:4d} ({v/total*100:.1f}%)")
        data_source[k] = v

    # scale
    scale_counts = Counter(safe_str(r.get("scale")) for r in results)
    print("\nScale:")
    scale = {}
    for k, v in scale_counts.most_common():
        print(f"  {k:20s}: {v:4d} ({v/total*100:.1f}%)")
        scale[k] = v

    # uses_real_wastewater
    rw_counts = Counter(r.get("uses_real_wastewater") for r in results)
    rw_true = rw_counts.get(True, 0)
    rw_false = rw_counts.get(False, 0)
    print(f"\nUses real wastewater: True={rw_true} ({rw_true/total*100:.1f}%), False={rw_false} ({rw_false/total*100:.1f}%)")

    # dataset_size overall
    sizes = [float(r["dataset_size"]) for r in results
             if r.get("dataset_size") is not None]
    if sizes:
        med = statistics.median(sizes)
        q1 = sorted(sizes)[len(sizes) // 4]
        q3 = sorted(sizes)[3 * len(sizes) // 4]
        print(f"\nDataset size overall: median={med:,.0f}, IQR=[{q1:,.0f}, {q3:,.0f}], n={len(sizes)}")
    else:
        med, q1, q3 = None, None, None

    # Cross: scale × sub_field
    print("\nScale by sub-field:")
    scale_by_sf = {}
    for sf in SUBFIELD_ORDER:
        sf_results = [r for r in results if r.get("sub_field") == sf]
        sc = Counter(safe_str(r.get("scale")) for r in sf_results)
        n = len(sf_results)
        fs = sc.get("full_scale", 0)
        print(f"  {sf:15s}: full_scale={fs}/{n} ({fs/n*100:.1f}%)")
        scale_by_sf[sf] = dict(sc)

    return {
        "data_source": data_source,
        "scale": scale,
        "uses_real_wastewater": {"true": rw_true, "false": rw_false},
        "dataset_size": {"median": med, "q1": q1, "q3": q3, "n": len(sizes)},
        "scale_by_subfield": scale_by_sf,
    }


# ============================================================
# Part 3: Barrier 2 — IT/OT Integration Disconnect
# ============================================================

def part3_it_ot(results):
    print("\n" + "=" * 70)
    print("PART 3: BARRIER 2 — IT/OT INTEGRATION DISCONNECT")
    print("=" * 70)

    total = len(results)

    # model_framework
    mf_counts = Counter(safe_str(r.get("model_framework")) for r in results)
    print("\nModel framework:")
    model_framework = {}
    for k, v in mf_counts.most_common():
        print(f"  {k:20s}: {v:4d} ({v/total*100:.1f}%)")
        model_framework[k] = v

    # control_loop_type
    cl_counts = Counter(safe_str(r.get("control_loop_type")) for r in results)
    print("\nControl loop type:")
    control_loop = {}
    for k, v in cl_counts.most_common():
        print(f"  {k:20s}: {v:4d} ({v/total*100:.1f}%)")
        control_loop[k] = v

    # deployed_in_plant
    dp_counts = Counter(r.get("deployed_in_plant") for r in results)
    dp_true = dp_counts.get(True, 0)
    dp_false = dp_counts.get(False, 0)
    print(f"\nDeployed in plant: True={dp_true} ({dp_true/total*100:.1f}%), False={dp_false} ({dp_false/total*100:.1f}%)")

    # real_time_testing
    rt_counts = Counter(r.get("real_time_testing") for r in results)
    rt_true = rt_counts.get(True, 0)
    rt_false = rt_counts.get(False, 0)
    print(f"Real-time testing: True={rt_true} ({rt_true/total*100:.1f}%), False={rt_false} ({rt_false/total*100:.1f}%)")

    # Cross: deployed × sub_field
    print("\nDeployed by sub-field:")
    deployed_by_sf = {}
    for sf in SUBFIELD_ORDER:
        sf_results = [r for r in results if r.get("sub_field") == sf]
        dp = sum(1 for r in sf_results if r.get("deployed_in_plant") is True)
        n = len(sf_results)
        print(f"  {sf:15s}: {dp}/{n} ({dp/n*100:.1f}%)")
        deployed_by_sf[sf] = {"deployed": dp, "total": n, "pct": round(dp / n * 100, 1)}

    return {
        "model_framework": model_framework,
        "control_loop_type": control_loop,
        "deployed_in_plant": {"true": dp_true, "false": dp_false},
        "real_time_testing": {"true": rt_true, "false": rt_false},
        "deployed_by_subfield": deployed_by_sf,
    }


# ============================================================
# Part 4: Barrier 3 — Regulatory Trust Deficit
# ============================================================

def part4_trust(results):
    print("\n" + "=" * 70)
    print("PART 4: BARRIER 3 — REGULATORY TRUST DEFICIT")
    print("=" * 70)

    total = len(results)

    # uncertainty_quantification
    uq_counts = Counter(r.get("uncertainty_quantification") for r in results)
    uq_true = uq_counts.get(True, 0)
    print(f"\nUncertainty quantification: True={uq_true} ({uq_true/total*100:.1f}%)")

    # interpretability_method
    im_counts = Counter(safe_str(r.get("interpretability_method")) for r in results)
    print("\nInterpretability method:")
    interp = {}
    for k, v in im_counts.most_common():
        print(f"  {k:25s}: {v:4d} ({v/total*100:.1f}%)")
        interp[k] = v
    has_interp = total - im_counts.get("none", 0)
    print(f"  → Has any method: {has_interp} ({has_interp/total*100:.1f}%)")

    # validation_method
    vm_counts = Counter(safe_str(r.get("validation_method")) for r in results)
    print("\nValidation method:")
    validation = {}
    for k, v in vm_counts.most_common():
        print(f"  {k:25s}: {v:4d} ({v/total*100:.1f}%)")
        validation[k] = v

    # code_available
    ca_counts = Counter(r.get("code_available") for r in results)
    ca_true = ca_counts.get(True, 0)
    print(f"\nCode available: True={ca_true} ({ca_true/total*100:.1f}%)")

    # data_available
    da_counts = Counter(r.get("data_available") for r in results)
    da_true = da_counts.get(True, 0)
    print(f"Data available: True={da_true} ({da_true/total*100:.1f}%)")

    return {
        "uncertainty_quantification": {"true": uq_true, "false": total - uq_true},
        "interpretability_method": interp,
        "has_interpretability": has_interp,
        "validation_method": validation,
        "code_available": {"true": ca_true, "false": total - ca_true},
        "data_available": {"true": da_true, "false": total - da_true},
    }


# ============================================================
# Part 5: Barrier 4 — The Coagulation Exception
# ============================================================

def part5_coagulation(results):
    print("\n" + "=" * 70)
    print("PART 5: BARRIER 4 — THE COAGULATION EXCEPTION")
    print("=" * 70)

    coag = [r for r in results if r.get("sub_field") == "coagulation"]
    others = [r for r in results if r.get("sub_field") != "coagulation"]
    n_coag = len(coag)
    n_others = len(others)
    print(f"\nCoagulation: {n_coag}, Others: {n_others}")

    comparisons = {}

    # Compare boolean fields
    for field in BOOL_FIELDS:
        c_true = sum(1 for r in coag if r.get(field) is True)
        o_true = sum(1 for r in others if r.get(field) is True)
        c_pct = c_true / n_coag * 100
        o_pct = o_true / n_others * 100

        # Fisher exact test (2x2 table)
        table = [[c_true, n_coag - c_true], [o_true, n_others - o_true]]
        _, p_val = stats.fisher_exact(table)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n{field}:")
        print(f"  Coagulation: {c_true}/{n_coag} ({c_pct:.1f}%)")
        print(f"  Others:      {o_true}/{n_others} ({o_pct:.1f}%)")
        print(f"  Fisher p={p_val:.4f} {sig}")

        comparisons[field] = {
            "coagulation": {"true": c_true, "total": n_coag, "pct": round(c_pct, 1)},
            "others": {"true": o_true, "total": n_others, "pct": round(o_pct, 1)},
            "fisher_p": round(p_val, 4),
            "significant": p_val < 0.05,
        }

    # Compare categorical: control_loop_type
    for field in ["control_loop_type", "scale"]:
        c_counts = Counter(safe_str(r.get(field)) for r in coag)
        o_counts = Counter(safe_str(r.get(field)) for r in others)
        all_cats = sorted(set(list(c_counts.keys()) + list(o_counts.keys())))

        print(f"\n{field}:")
        print(f"  {'Category':20s} {'Coag':>8s} {'Others':>8s}")
        cat_comparison = {}
        for cat in all_cats:
            cv = c_counts.get(cat, 0)
            ov = o_counts.get(cat, 0)
            print(f"  {cat:20s} {cv:4d} ({cv/n_coag*100:5.1f}%) {ov:4d} ({ov/n_others*100:5.1f}%)")
            cat_comparison[cat] = {
                "coagulation": {"count": cv, "pct": round(cv / n_coag * 100, 1)},
                "others": {"count": ov, "pct": round(ov / n_others * 100, 1)},
            }

        # Chi-square test
        observed = [[c_counts.get(cat, 0) for cat in all_cats],
                     [o_counts.get(cat, 0) for cat in all_cats]]
        try:
            chi2, p_val, dof, _ = stats.chi2_contingency(observed)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  χ²={chi2:.2f}, df={dof}, p={p_val:.4f} {sig}")
        except ValueError:
            chi2, p_val, dof = None, None, None
            sig = "N/A"

        comparisons[field] = {
            "categories": cat_comparison,
            "chi2": chi2, "p": round(p_val, 4) if p_val else None, "dof": dof,
        }

    return comparisons


# ============================================================
# Part 6: Cross-tabulation (for heatmap)
# ============================================================

def part6_cross_table(results):
    print("\n" + "=" * 70)
    print("PART 6: CROSS-TABULATION (HEATMAP DATA)")
    print("=" * 70)

    # Boolean fields: % True by sub-field
    cross = {}
    print(f"\n{'Field':30s}", end="")
    for sf in SUBFIELD_ORDER:
        print(f" {sf:>10s}", end="")
    print(f" {'Overall':>10s}")
    print("-" * 110)

    for field in BOOL_FIELDS:
        row = {}
        print(f"{field:30s}", end="")
        for sf in SUBFIELD_ORDER:
            sf_results = [r for r in results if r.get("sub_field") == sf]
            n = len(sf_results)
            t = sum(1 for r in sf_results if r.get(field) is True)
            pct = t / n * 100 if n > 0 else 0
            print(f" {pct:9.1f}%", end="")
            row[sf] = round(pct, 1)
        # Overall
        total_t = sum(1 for r in results if r.get(field) is True)
        overall_pct = total_t / len(results) * 100
        print(f" {overall_pct:9.1f}%")
        row["overall"] = round(overall_pct, 1)
        cross[field] = row

    # Categorical: % of specific value by sub-field
    cat_indicators = {
        "has_interpretability": lambda r: safe_str(r.get("interpretability_method")) != "none",
        "temporal_validation": lambda r: safe_str(r.get("validation_method")) == "temporal_split",
        "closed_loop": lambda r: safe_str(r.get("control_loop_type")) == "closed_loop",
        "full_scale": lambda r: safe_str(r.get("scale")) == "full_scale",
    }

    for label, fn in cat_indicators.items():
        row = {}
        print(f"{label:30s}", end="")
        for sf in SUBFIELD_ORDER:
            sf_results = [r for r in results if r.get("sub_field") == sf]
            n = len(sf_results)
            t = sum(1 for r in sf_results if fn(r))
            pct = t / n * 100 if n > 0 else 0
            print(f" {pct:9.1f}%", end="")
            row[sf] = round(pct, 1)
        total_t = sum(1 for r in results if fn(r))
        overall_pct = total_t / len(results) * 100
        print(f" {overall_pct:9.1f}%")
        row["overall"] = round(overall_pct, 1)
        cross[label] = row

    return cross


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    results = load_data()
    print(f"Loaded {len(results)} papers\n")

    output = {}
    output["overview"] = part1_overview(results)
    output["barrier_1_data_quality"] = part2_data_quality(results)
    output["barrier_2_it_ot"] = part3_it_ot(results)
    output["barrier_3_trust"] = part4_trust(results)
    output["barrier_4_coagulation"] = part5_coagulation(results)
    output["cross_table"] = part6_cross_table(results)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE — saved to {OUTPUT_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
