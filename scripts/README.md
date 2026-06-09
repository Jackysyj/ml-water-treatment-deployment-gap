# Scripts - npj Clean Water Perspective

Data processing pipeline for `Operational Evidence Standards for Machine Learning in Wastewater Treatment`.

## Pipeline Overview

```text
Step 1: Literature search
  search_openalex.py and search_supplement.py

Step 2: Abstract screening
  screen_papers.py and screen_supplement.py
  -> 500-record source extraction pool after deduplication

Step 3: PDF download and conversion
  download_papers.py and mineru_batch.sh
  -> full-text Markdown conversions

Step 4: Gold standard and LLM benchmark
  sample_gold_standard_batch4.py
  supplement_gold_standard.py
  benchmark_extract.py

Step 5: Full-corpus extraction and cleaning
  fulltext_extract.py
  clean_corpus.py
  verification_summary.py
  -> 423-study full-text analysis corpus

Step 6: Bounded operational illustration
  plant_panel.py
  -> six-plant derived outputs with feedforward, persistence and lagged-effluent scenarios

Step 7: Perspective figures and submission package
  plot_fig1_landscape.py
  plot_fig2_operational_illustration.py
  export_fig3_evidence_framework.sh
  validate_manuscript.py
  generate_submission_word.py
  prepare_zenodo.py
```

All active scripts use `paths.py` as the single source of truth for project paths.

## Key Scripts

| File | Output | Description |
|:-----|:-------|:------------|
| `clean_corpus.py` | `data/literature/extracted/fulltext_extraction_clean.json` | Restricts the source pool to the 423-study full-text analysis corpus |
| `corpus_robustness.py` | `figures/corpus_robustness_subsets.csv` | Sensitivity filters for corpus indicators |
| `verification_summary.py` | `figures/verification_summary.json` | Manual verification summary for deployment and validation fields |
| `plant_panel.py` | `figures/plant_panel_stats.json`, `figures/plant_panel_per_target.csv` | Six-plant routine-monitoring scenario outputs |
| `plot_fig1_landscape.py` | `figures/fig1_landscape.png` | Figure 1 evidence map |
| `plot_fig2_operational_illustration.py` | `figures/fig2_operational_illustration.png` | Figure 2 bounded six-plant illustration |
| `export_fig3_evidence_framework.sh` | `figures/fig3_evidence_framework.png` | Figure 3 draw.io export from `figures/fig3_evidence_framework.drawio` |
| `validate_manuscript.py` | console report | Perspective format and data-text consistency validator |
| `generate_submission_word.py` | `manuscript/submission/` | Main manuscript, SI, cover letter and highlights assembly |
| `prepare_zenodo.py` | `data/zenodo/` | Public corpus, scripts, manual-review files and derived outputs |

In the Zenodo package, `export_fig3_evidence_framework.sh` can also read
`derived_outputs/fig3_evidence_framework.drawio` and write
`derived_outputs/fig3_evidence_framework.png`; set `DRAWIO_EXPORTER` if the local
draw.io export wrapper is not at the full-project default path.

## Key Data Files

| File | Description |
|:-----|:------------|
| `data/literature/extracted/fulltext_extraction_500_cleaned.json` | 500-record source extraction pool |
| `data/literature/extracted/fulltext_extraction_clean.json` | 423-study full-text analysis corpus |
| `data/literature/manual_review/deployment_verification.csv` | Full-text verification of plant-deployment labels |
| `data/literature/manual_review/validation_verification.csv` | Full-text verification of future-facing validation labels |
| `data/plant_private/raw_wwtp_exports/` | Confidential raw six-plant records used only for local derived-output generation |
| `figures/manuscript_stats.json` | Literature evidence-map truth source |
| `figures/plant_panel_stats.json` | Six-plant scenario truth source |
| `figures/plant_panel_per_target.csv` | Plant-target scenario table |

## API Configuration

Extraction and screening scripts use Qwen3.5-Plus through DashScope API:

```bash
export QWEN_API_KEY="your-key-here"
```

The generated Perspective does not require rerunning LLM extraction unless the corpus is changed.
