# Dataset: Beyond Accuracy — ML Deployment Gap in Water Treatment

## Overview

This repository contains the complete dataset and analysis code for:

> **Beyond Accuracy: A Data-Driven Analysis of the Machine Learning Deployment Gap in Water Treatment**
> Submitted to *Nature Water*

The dataset comprises **500 peer-reviewed studies** (2018-2025) applying machine learning
to water and wastewater treatment, with 22 structured fields extracted per study using a
three-model cross-validation pipeline (Qwen3.5-Plus, Claude Sonnet 4.6, Gemini 3.1 Pro).

## Repository Structure

```
.
├── README.md                          # This file
├── dataset_500_papers.csv             # Complete 500-paper dataset (22 fields)
├── crossref_metadata.csv              # Bibliographic metadata from Crossref
├── scripts/                           # Analysis and figure-generation scripts
│   ├── plot_config.py                 # Shared plotting configuration
│   ├── plot_fig1_landscape.py         # Figure 1: ML landscape
│   ├── validate_manuscript.py         # Data-text consistency checker
│   └── plot_fig1_landscape.py         # Figure 1: ML landscape
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

## Data Dictionary: `dataset_500_papers.csv`

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

## Reproduction

### Requirements

```bash
pip install pandas matplotlib seaborn scipy numpy
```

### Generate Figures

```bash
python scripts/plot_fig1_landscape.py
# Additional figure scripts (plot_fig2_barriers.py, etc.) as available
```

### Validate Manuscript Consistency

```bash
python scripts/validate_manuscript.py
```

## Extraction Methodology

1. **Screening**: 1,004 + 1,859 candidates screened via LLM-assisted abstract evaluation
2. **Extraction**: 22 fields extracted per paper using Qwen3.5-Plus (zero-shot prompt)
3. **Cross-validation**: Three models independently extracted all 500 papers; majority-vote
   consensus resolved 98.2% of values; remaining 1.8% defaulted to benchmark winner
4. **Gold standard**: 50 papers manually annotated by two reviewers; best model achieved
   categorical accuracy 87.8%, boolean accuracy 97.3%, list F1 = 0.887

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this dataset, please cite:

```
[Authors]. Beyond Accuracy: A Data-Driven Analysis of the Machine Learning
Deployment Gap in Water Treatment. Nature Water, [year]. DOI: [pending]
```
