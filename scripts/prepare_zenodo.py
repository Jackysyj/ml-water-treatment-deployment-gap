#!/usr/bin/env python3
"""
Prepare Zenodo upload package for the npj Clean Water Perspective.

Exports the 423-study analysis corpus and 500-record source extraction pool as
flat CSV files, copies scripts/prompts/manual-review records, and generates a
README with the data dictionary and confidentiality boundary.

Usage: python3 scripts/prepare_zenodo.py
"""

import json
import csv
import shutil

from paths import (
    ANALYSIS_CORPUS_JSON,
    BENCHMARK_PROMPT_DIR,
    BENCHMARK_PUBLIC_DIR,
    CROSSREF_METADATA_JSON,
    FIGURES_DIR,
    MANUAL_REVIEW_DIR,
    ROOT,
    SOURCE_POOL_JSON,
    ZENODO_DIR,
)

DATA_PATH = ANALYSIS_CORPUS_JSON
SOURCE_POOL_PATH = SOURCE_POOL_JSON
CROSSREF_PATH = CROSSREF_METADATA_JSON
ARCHIVE_ZENODO_DIR = ROOT / "_archived" / "2026-06-07_perspective_pivot" / "zenodo_previous_package"

# Fields to export (exclude internal fields starting with _)
EXPORT_FIELDS = [
    "doi", "paper_title", "year", "journal", "sub_field", "data_source",
    "best_algorithm", "best_metric_type", "best_metric_value",
    "validation_method", "interpretability_method", "model_framework",
    "control_loop_type", "scale", "dataset_size", "time_span",
    "real_time_testing", "deployed_in_plant", "uses_real_wastewater",
    "code_available", "data_available", "uncertainty_quantification",
    "ml_algorithms", "target_variable",
]

CROSSREF_FIELDS = [
    "doi", "paper_title", "year", "journal", "first_author", "authors",
    "publisher", "issn", "cited_by_count", "crossref_type",
]


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def flatten_list(val):
    """Convert list to semicolon-separated string."""
    if isinstance(val, list):
        return "; ".join(str(v) for v in val)
    return val


def _archive_destination(path):
    ARCHIVE_ZENODO_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_ZENODO_DIR / path.name
    if not dest.exists():
        return dest
    stem = path.stem if path.is_file() else path.name
    suffix = path.suffix if path.is_file() else ""
    i = 2
    while True:
        candidate = ARCHIVE_ZENODO_DIR / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def clean_previous_package(zenodo_dir):
    """Archive earlier generated package contents before rebuilding."""
    if not zenodo_dir.exists():
        return
    moved = 0
    for path in sorted(zenodo_dir.iterdir()):
        shutil.move(str(path), str(_archive_destination(path)))
        moved += 1
    if moved:
        print(f"  Archived {moved} previous package entries -> {ARCHIVE_ZENODO_DIR.relative_to(ROOT)}/")


def export_dataset_csv(records, out_path):
    """Export the literature dataset as flat CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for r in records:
            row = {}
            for field in EXPORT_FIELDS:
                val = r.get(field)
                row[field] = flatten_list(val)
            writer.writerow(row)
    print(f"  Exported {len(records)} records -> {out_path.name}")


def export_crossref_csv(records, out_path):
    """Export crossref metadata as flat CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CROSSREF_FIELDS)
        writer.writeheader()
        for r in records:
            row = {}
            for field in CROSSREF_FIELDS:
                val = r.get(field)
                row[field] = flatten_list(val)
            writer.writerow(row)
    print(f"  Exported {len(records)} records -> {out_path.name}")


def copy_scripts(zenodo_dir):
    """Copy analysis and plotting scripts."""
    scripts_dst = zenodo_dir / "scripts"
    scripts_dst.mkdir(exist_ok=True)

    scripts_to_copy = [
        "README.md",
        "paths.py",
        "plot_config.py",
        "plot_fig1_landscape.py",
        "plot_fig2_operational_illustration.py",
        "export_fig3_evidence_framework.sh",
        "compute_statistics.py",
        "clean_corpus.py",
        "corpus_robustness.py",
        "extraction_reliability.py",
        "verification_summary.py",
        "plant_panel.py",
        "plant_validation.py",
        "plant_mechanism.py",
        "validate_manuscript.py",
        "generate_submission_word.py",
        "prepare_zenodo.py",
    ]

    copied = 0
    for name in scripts_to_copy:
        src = ROOT / "scripts" / name
        if src.exists():
            shutil.copy2(src, scripts_dst / name)
            copied += 1
    print(f"  Copied {copied} scripts -> scripts/")


def copy_prompts(zenodo_dir):
    """Copy extraction and screening prompts."""
    prompts_dst = zenodo_dir / "prompts"
    prompts_dst.mkdir(exist_ok=True)

    prompt_dir = BENCHMARK_PROMPT_DIR
    copied = 0
    if prompt_dir.exists():
        for f in prompt_dir.iterdir():
            if f.suffix in (".md", ".txt"):
                shutil.copy2(f, prompts_dst / f.name)
                copied += 1

    # Copy screening scripts as reference
    for name in ["screen_papers.py", "screen_supplement.py"]:
        src = ROOT / "scripts" / name
        if src.exists():
            shutil.copy2(src, prompts_dst / name)
            copied += 1

    print(f"  Copied {copied} prompt/screening files -> prompts/")


def copy_benchmark(zenodo_dir):
    """Copy benchmark summary files."""
    bench_dst = zenodo_dir / "benchmark"
    bench_dst.mkdir(exist_ok=True)

    # Gold standard
    gs_src = BENCHMARK_PUBLIC_DIR / "gold_standard_v2.csv"
    if gs_src.exists():
        shutil.copy2(gs_src, bench_dst / "gold_standard_50.csv")

    # Benchmark matrix
    bm_src = BENCHMARK_PUBLIC_DIR / "table_s4_benchmark_matrix.csv"
    if bm_src.exists():
        shutil.copy2(bm_src, bench_dst / "benchmark_matrix.csv")

    # Per-field accuracy
    pf_src = BENCHMARK_PUBLIC_DIR / "table_s5_perfield_accuracy.csv"
    if pf_src.exists():
        shutil.copy2(pf_src, bench_dst / "perfield_accuracy.csv")

    # Subfield distribution (Table S7)
    s7_src = BENCHMARK_PUBLIC_DIR / "table_s7_subfield_distribution.csv"
    if s7_src.exists():
        shutil.copy2(s7_src, bench_dst / "subfield_distribution.csv")

    # Cross-model agreement (Table S9)
    s9_src = BENCHMARK_PUBLIC_DIR / "table_s9_crossmodel_agreement.csv"
    if s9_src.exists():
        shutil.copy2(s9_src, bench_dst / "crossmodel_agreement.csv")

    print(f"  Copied benchmark files -> benchmark/")


def copy_manual_review(zenodo_dir):
    """Copy manual-review records used for load-bearing extracted fields."""
    manual_dst = zenodo_dir / "manual_review"
    manual_dst.mkdir(exist_ok=True)

    copied = 0
    for src in sorted(MANUAL_REVIEW_DIR.glob("*")):
        if src.is_file() and src.suffix in (".csv", ".json", ".md"):
            shutil.copy2(src, manual_dst / src.name)
            copied += 1
    print(f"  Copied {copied} manual-review files -> manual_review/")


def copy_derived_outputs(zenodo_dir):
    """Copy figure source data and anonymised plant-derived outputs."""
    derived_dst = zenodo_dir / "derived_outputs"
    derived_dst.mkdir(exist_ok=True)

    files = [
        "manuscript_stats.json",
        "corpus_robustness_stats.json",
        "corpus_robustness_subsets.csv",
        "verification_summary.json",
        "extraction_perfield_accuracy.csv",
        "extraction_confusion.json",
        "piml_keyword_flags.csv",
        "fig1_data_summary.csv",
        "fig2_operational_illustration_summary.csv",
        "fig3_evidence_framework.drawio",
        "fig3_evidence_framework.png",
        "fig_s1_prisma.png",
        "fig_s2_subfield_distribution.png",
        "plant_panel_stats.json",
        "plant_panel_per_target.csv",
        "plant_conformal_results.csv",
        "plant_stats.json",
        "plant_validation_results.csv",
        "plant_feature_importance.csv",
        "table1_data.csv",
        "statistical_tests.json",
    ]
    copied = 0
    for name in files:
        src = FIGURES_DIR / name
        if src.exists():
            shutil.copy2(src, derived_dst / name)
            copied += 1
    print(f"  Copied {copied} derived-output files -> derived_outputs/")


def write_readme(zenodo_dir, n_records):
    """Generate README.md with data dictionary and reproduction instructions."""
    readme = f"""# Dataset: Operational Evidence Standards for Machine Learning in Wastewater Treatment

## Overview

This repository contains the complete dataset and analysis code for:

> **Operational Evidence Standards for Machine Learning in Wastewater Treatment**
> Submitted to *npj Clean Water* as a Perspective

The main analysis dataset comprises **{n_records} full-text peer-reviewed studies** (2018-2025)
applying machine learning to water and wastewater treatment, with 22 structured fields
extracted per study. The repository also includes the 500-record source extraction pool,
manual verification files for load-bearing fields, and anonymised six-plant derived
outputs used as a bounded routine-monitoring illustration. Raw plant records are not
released because they are covered by utility confidentiality agreements.

## Repository Structure

```
.
├── README.md                          # This file
├── dataset_423_analysis_corpus.csv    # Full-text analysis corpus (22 fields)
├── source_extraction_pool_500.csv      # Source extraction pool before restrictions
├── crossref_metadata.csv              # Bibliographic metadata from Crossref
├── manual_review/                      # Full-text verification of key labels
├── derived_outputs/                    # Figure source data and plant-derived outputs
├── scripts/                           # Analysis and figure-generation scripts
│   ├── plot_config.py                 # Shared plotting configuration
│   ├── plot_fig1_landscape.py         # Figure 1: published evidence map
│   ├── plot_fig2_operational_illustration.py # Figure 2: bounded six-plant illustration
│   ├── export_fig3_evidence_framework.sh # Figure 3: draw.io export helper
│   ├── validate_manuscript.py         # Data-text consistency checker
│   ├── generate_submission_word.py    # Submission DOCX/cover/highlights builder
│   ├── prepare_zenodo.py              # Public package builder
├── prompts/                           # LLM extraction prompts
│   ├── extraction_prompt_v1.md        # Zero-shot extraction prompt (best performing)
│   ├── extraction_prompt_v2.md        # Structured extraction prompt
│   ├── extraction_prompt_fewshot.md   # Few-shot extraction prompt
│   ├── screen_papers.py              # Abstract screening script
│   └── screen_supplement.py          # Supplementary screening script
└── benchmark/                         # LLM benchmark results
    ├── gold_standard_50.csv           # 50-paper gold standard (human-annotated)
    ├── benchmark_matrix.csv           # 3×3 model-prompt performance matrix
    └── perfield_accuracy.csv          # Per-field extraction accuracy
```

## Data Dictionary: `dataset_423_analysis_corpus.csv`

### Bibliographic Metadata

| Field | Type | Description |
|:------|:-----|:------------|
| `doi` | string | Digital Object Identifier |
| `paper_title` | string | Full title |
| `year` | integer | Publication year (2018-2025) |
| `journal` | string | Journal name |

### Domain Classification

| Field | Type | Allowed Values | Description |
|:------|:-----|:---------------|:------------|
| `sub_field` | categorical | WWTP, membrane, coagulation, DBP, monitoring, sludge, control | Primary application sub-field |
| `data_source` | categorical | SCADA, experimental, simulated, literature_compiled, mixed, not_specified | Origin of training data |
| `scale` | categorical | lab, pilot, full, simulation, mixed, not_specified | Study scale |

### ML Model Details

| Field | Type | Description |
|:------|:-----|:------------|
| `ml_algorithms` | list (semicolon-separated) | All ML algorithms tested |
| `best_algorithm` | string | Highest-performing algorithm |
| `best_metric_type` | categorical | R2, RMSE, MAE, accuracy, F1, AUC, MSE, MAPE, NSE, other | Primary evaluation metric |
| `best_metric_value` | float | Best reported metric value |
| `model_framework` | string | Software framework (sklearn, tensorflow, pytorch, etc.) |

### Dataset Characteristics

| Field | Type | Description |
|:------|:-----|:------------|
| `dataset_size` | integer/null | Number of data points or samples |
| `time_span` | string/null | Duration of data collection |
| `target_variable` | list (semicolon-separated) | Prediction target(s) |

### Deployment Readiness Indicators

| Field | Type | Description |
|:------|:-----|:------------|
| `validation_method` | categorical | random_split, temporal, k_fold, walk_forward, external, none, not_specified |
| `interpretability_method` | string | none, SHAP, feature_importance, LIME, attention, etc. |
| `control_loop_type` | categorical | none, advisory, closed_loop, not_applicable |
| `real_time_testing` | boolean | Model tested with live plant data |
| `deployed_in_plant` | boolean | Model deployed in operational plant |
| `uses_real_wastewater` | boolean | Uses real (not synthetic) wastewater |
| `code_available` | boolean | Source code publicly available |
| `data_available` | boolean | Dataset publicly available |
| `uncertainty_quantification` | boolean | Reports prediction uncertainty |

## Plant-Derived Outputs

Raw six-plant records are confidential. The `derived_outputs/` directory contains
anonymised outputs sufficient to reproduce the manuscript's operational benchmark,
including feedforward, persistence and influent-plus-lagged-effluent scenarios:

| File | Description |
|:-----|:------------|
| `plant_panel_stats.json` | Per-plant sample counts, predictor counts, scenario summaries, core-target performance and effluent variability |
| `plant_panel_per_target.csv` | Per-plant, per-target mean-baseline, persistence, feedforward and lagged-effluent *R*² values |
| `fig2_operational_illustration_summary.csv` | Figure 2 source summary used by the validator |
| `fig3_evidence_framework.drawio` | Draw.io source for Figure 3 |
| `fig3_evidence_framework.png` | Exported Figure 3 framework image |
| `fig_s1_prisma.png` | Supplementary PRISMA-style screening flow diagram |
| `fig_s2_subfield_distribution.png` | Supplementary gold-standard versus source-pool sub-field distribution |
| `plant_conformal_results.csv` | Split-conformal interval diagnostics for the representative plant deep-dive |
| `plant_validation_results.csv` | Representative plant model-performance table |
| `plant_feature_importance.csv` | Representative plant permutation-importance summaries |

## Reproduction Boundary

### Requirements

```bash
pip install pandas matplotlib seaborn scipy numpy
```

The copied scripts document the analysis workflow used in the full project tree.
The Zenodo package itself provides flat exported corpus tables and derived plant
outputs. Regenerating Figs. 1 and 2 requires the full project layout with
`data/literature/` and, for plant-derived outputs, confidential raw plant records
that are not released here. Figure 3 can be re-exported from
`derived_outputs/fig3_evidence_framework.drawio` if a draw.io export wrapper is
available; set `DRAWIO_EXPORTER=/path/to/export.sh` if needed.

### Full-project figure commands

```bash
python scripts/plot_fig1_landscape.py
python scripts/plot_fig2_operational_illustration.py
bash scripts/export_fig3_evidence_framework.sh
```

### Validate Manuscript Consistency

```bash
python scripts/validate_manuscript.py
```

## Extraction Methodology

1. **Screening**: 1,004 + 1,859 candidates screened via LLM-assisted abstract evaluation
2. **Extraction**: 22 fields extracted per paper using Qwen3.5-Plus (zero-shot prompt)
3. **Consensus and resolution**: Three models independently extracted the source pool where
   possible; majority-vote consensus and benchmark-winner fallback were used as a resolution
   workflow, not as an accuracy estimate
4. **Gold standard**: 50 papers manually annotated by two reviewers; best model achieved
   categorical accuracy 87.8%, boolean accuracy 97.3%, list *F1* = 0.887
5. **Manual verification**: All deployment labels in the 423-study analysis corpus and all
   temporal/walk-forward/external validation labels were checked against full text
6. **Operational benchmark**: Six anonymised operating WWTP datasets were analysed only as
   derived outputs; raw records remain confidential

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this dataset, please cite:

```
[Authors]. Operational Evidence Standards for Machine Learning in Wastewater Treatment.
npj Clean Water, [year]. DOI: [pending]
```
"""
    readme_path = zenodo_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"  Generated README.md")


def main():
    print("=" * 60)
    print("  Preparing Zenodo upload package")
    print("=" * 60)

    # Create output directory
    ZENODO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output: {ZENODO_DIR}")
    clean_previous_package(ZENODO_DIR)

    # 1. Export main dataset
    print("\n1. Exporting dataset CSV...")
    records = load_json(DATA_PATH)
    export_dataset_csv(records, ZENODO_DIR / "dataset_423_analysis_corpus.csv")

    source_records = load_json(SOURCE_POOL_PATH)
    export_dataset_csv(source_records, ZENODO_DIR / "source_extraction_pool_500.csv")

    # 2. Export crossref metadata
    print("\n2. Exporting crossref metadata...")
    crossref = load_json(CROSSREF_PATH)
    export_crossref_csv(crossref, ZENODO_DIR / "crossref_metadata.csv")

    # 3. Copy scripts
    print("\n3. Copying scripts...")
    copy_scripts(ZENODO_DIR)

    # 4. Copy prompts
    print("\n4. Copying prompts...")
    copy_prompts(ZENODO_DIR)

    # 5. Copy benchmark
    print("\n5. Copying benchmark files...")
    copy_benchmark(ZENODO_DIR)

    # 6. Copy manual review
    print("\n6. Copying manual-review files...")
    copy_manual_review(ZENODO_DIR)

    # 7. Copy derived outputs
    print("\n7. Copying derived outputs...")
    copy_derived_outputs(ZENODO_DIR)

    # 8. Generate README
    print("\n8. Generating README...")
    write_readme(ZENODO_DIR, len(records))

    # Summary
    print(f"\n{'=' * 60}")
    total_files = sum(1 for _ in ZENODO_DIR.rglob("*") if _.is_file())
    print(f"  Zenodo package ready: {total_files} files in {ZENODO_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
