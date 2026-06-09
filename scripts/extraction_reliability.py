#!/usr/bin/env python3
"""
Per-field LLM extraction accuracy / F1 against the 50-paper gold standard.

Directly answers reviewer R2.3 ("OpenAlex + LLM reliability, especially for
nuanced fields"). The point is NOT to report one aggregate accuracy, but to show
per-field performance for the final extraction configuration actually used on
the source extraction pool (Qwen3.5-Plus + v1 prompt), with special attention to the
headline-bearing field deployed_in_plant and the two nuanced fields
control_loop_type and sub_field.

Outputs:
  figures/extraction_perfield_accuracy.csv   per-field accuracy + macro-F1
  figures/extraction_confusion.json          confusion matrices for key fields
  figures/extraction_reliability_stats.json  compact numbers for manuscript/SI
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from paths import BENCHMARK_GOLD_DIR, BENCHMARK_RESULTS_DIR, FIGURES_DIR

GOLD = BENCHMARK_GOLD_DIR / "gold_standard_v2.json"
BENCH = BENCHMARK_RESULTS_DIR / "benchmark_all_450.json"

OUT_CSV = FIGURES_DIR / "extraction_perfield_accuracy.csv"
OUT_CONF = FIGURES_DIR / "extraction_confusion.json"
OUT_STATS = FIGURES_DIR / "extraction_reliability_stats.json"

FINAL_MODEL = "qwen3.5-plus"
FINAL_PROMPT = "v1"

BOOL_FIELDS = [
    "real_time_testing", "deployed_in_plant", "uses_real_wastewater",
    "code_available", "data_available", "uncertainty_quantification",
]
CAT_FIELDS = [
    "sub_field", "data_source", "best_metric_type", "validation_method",
    "interpretability_method", "model_framework", "control_loop_type", "scale",
]
LIST_FIELDS = ["ml_algorithms"]
KEY_FIELDS = ["deployed_in_plant", "control_loop_type", "sub_field"]


def norm(v):
    """Normalise a scalar field value for comparison."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        s = v.strip().lower()
        return s if s else "null"
    return str(v).lower()


def norm_list(v):
    if not isinstance(v, list):
        return set()
    return {str(x).strip().lower() for x in v if str(x).strip()}


def load_records():
    gold = {p["paper_id"]: p for p in json.load(open(GOLD))}
    bench = json.load(open(BENCH))
    pred = {}
    for r in bench:
        if r.get("_model") == FINAL_MODEL and r.get("_prompt_version") == FINAL_PROMPT:
            if r.get("_status", "success") == "success":
                pred[r["paper_id"]] = r
    ids = sorted(set(gold) & set(pred))
    return gold, pred, ids


def per_field_accuracy(gold, pred, ids):
    rows = []
    for field in BOOL_FIELDS + CAT_FIELDS:
        g_vals, p_vals, correct = [], [], 0
        for pid in ids:
            g = norm(gold[pid].get(field))
            p = norm(pred[pid].get(field))
            g_vals.append(g)
            p_vals.append(p)
            correct += int(g == p)
        n = len(ids)
        acc = correct / n if n else float("nan")
        # macro-F1 over the union of observed labels
        labels = sorted(set(g_vals) | set(p_vals))
        macro_f1 = f1_score(g_vals, p_vals, labels=labels, average="macro", zero_division=0)
        rows.append({
            "field": field,
            "type": "boolean" if field in BOOL_FIELDS else "categorical",
            "is_key_field": field in KEY_FIELDS,
            "n": n,
            "n_correct": correct,
            "accuracy": round(acc, 3),
            "macro_f1": round(float(macro_f1), 3),
            "n_labels": len(labels),
        })
    # list field: set-based F1 (mean over papers)
    for field in LIST_FIELDS:
        f1s = []
        for pid in ids:
            g = norm_list(gold[pid].get(field))
            p = norm_list(pred[pid].get(field))
            if not g and not p:
                f1s.append(1.0)
                continue
            tp = len(g & p)
            prec = tp / len(p) if p else 0.0
            rec = tp / len(g) if g else 0.0
            f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
        rows.append({
            "field": field, "type": "list", "is_key_field": False,
            "n": len(ids), "n_correct": "", "accuracy": "",
            "macro_f1": round(float(np.mean(f1s)), 3), "n_labels": "",
        })
    return pd.DataFrame(rows)


def confusion_for_key_fields(gold, pred, ids):
    out = {}
    for field in KEY_FIELDS:
        cm = defaultdict(lambda: defaultdict(int))
        for pid in ids:
            g = norm(gold[pid].get(field))
            p = norm(pred[pid].get(field))
            cm[g][p] += 1
        labels = sorted(set(cm) | {p for row in cm.values() for p in row})
        matrix = {g: {p: cm[g][p] for p in labels} for g in labels}
        # deployment is the headline field: report TP/FP/FN/TN explicitly
        extra = {}
        if field == "deployed_in_plant":
            tp = cm["true"]["true"]; fp = cm["false"]["true"]
            fn = cm["true"]["false"]; tn = cm["false"]["false"]
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            extra = {
                "true_positive": tp, "false_positive": fp,
                "false_negative": fn, "true_negative": tn,
                "precision_for_deployed": None if np.isnan(prec) else round(prec, 3),
                "recall_for_deployed": None if np.isnan(rec) else round(rec, 3),
                "n_gold_deployed": tp + fn,
            }
        out[field] = {"labels": labels, "matrix": matrix, **extra}
    return out


def main():
    gold, pred, ids = load_records()
    df = per_field_accuracy(gold, pred, ids)
    conf = confusion_for_key_fields(gold, pred, ids)

    df.to_csv(OUT_CSV, index=False)
    with open(OUT_CONF, "w") as f:
        json.dump(conf, f, indent=2, ensure_ascii=False)

    bool_df = df[df["type"] == "boolean"]
    cat_df = df[df["type"] == "categorical"]
    stats = {
        "config": {"model": FINAL_MODEL, "prompt": FINAL_PROMPT, "n_gold_papers": len(ids)},
        "boolean_mean_accuracy": round(float(bool_df["accuracy"].astype(float).mean()), 3),
        "categorical_mean_accuracy": round(float(cat_df["accuracy"].astype(float).mean()), 3),
        "categorical_mean_macro_f1": round(float(cat_df["macro_f1"].astype(float).mean()), 3),
        "list_f1_ml_algorithms": float(df[df["type"] == "list"]["macro_f1"].iloc[0]),
        "key_fields": {},
        "weakest_categorical": None,
    }
    for field in KEY_FIELDS:
        r = df[df["field"] == field].iloc[0]
        entry = {"accuracy": float(r["accuracy"]), "macro_f1": float(r["macro_f1"])}
        if field == "deployed_in_plant":
            entry.update({k: conf[field][k] for k in
                          ["true_positive", "false_positive", "false_negative",
                           "true_negative", "precision_for_deployed",
                           "recall_for_deployed", "n_gold_deployed"]})
        stats["key_fields"][field] = entry
    weakest = cat_df.sort_values("accuracy").iloc[0]
    stats["weakest_categorical"] = {
        "field": weakest["field"], "accuracy": float(weakest["accuracy"]),
        "macro_f1": float(weakest["macro_f1"]),
    }
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_CONF}")
    print(f"Saved: {OUT_STATS}")
    print("\n=== PER-FIELD (final config:", FINAL_MODEL, FINAL_PROMPT, "| n =", len(ids), ") ===")
    print(df.to_string(index=False))
    print("\n=== KEY-FIELD STATS ===")
    print(json.dumps(stats["key_fields"], indent=2, ensure_ascii=False))
    print("\nboolean mean acc:", stats["boolean_mean_accuracy"],
          "| categorical mean acc:", stats["categorical_mean_accuracy"],
          "| weakest cat:", stats["weakest_categorical"])


if __name__ == "__main__":
    main()
