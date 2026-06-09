# Dataset: Operational Evidence Standards for Machine Learning in Wastewater Treatment

## Overview

This repository contains the complete dataset and analysis code for:

> **Operational Evidence Standards for Machine Learning in Wastewater Treatment**
> Submitted to *npj Clean Water* as a Perspective

The main analysis dataset comprises **423 full-text peer-reviewed studies** (2018-2025)
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
