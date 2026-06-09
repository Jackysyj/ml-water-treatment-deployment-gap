"""Shared project paths for the npj Clean Water Perspective workflow."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
LITERATURE_DIR = DATA_DIR / "literature"
ZENODO_DIR = DATA_DIR / "zenodo"
PLANT_PRIVATE_DIR = DATA_DIR / "plant_private" / "raw_wwtp_exports"

RAW_OPENALEX_DIR = LITERATURE_DIR / "raw_openalex"
SCREENING_DIR = LITERATURE_DIR / "screening"
SOURCE_PDFS_DIR = LITERATURE_DIR / "source_pdfs"
CONVERTED_FULLTEXT_DIR = LITERATURE_DIR / "converted_fulltext"
EXTRACTED_DIR = LITERATURE_DIR / "extracted"
BENCHMARK_DIR = LITERATURE_DIR / "benchmark"
BENCHMARK_PUBLIC_DIR = BENCHMARK_DIR / "public_tables"
BENCHMARK_PROMPT_DIR = BENCHMARK_DIR / "prompt"
BENCHMARK_RESULTS_DIR = BENCHMARK_DIR / "results"
BENCHMARK_GOLD_DIR = BENCHMARK_DIR / "gold_standard"
MANUAL_REVIEW_DIR = LITERATURE_DIR / "manual_review"

FIGURES_DIR = ROOT / "figures"
DRAFTS_DIR = ROOT / "manuscript" / "drafts"
SUBMISSION_DIR = ROOT / "manuscript" / "submission"

ANALYSIS_CORPUS_JSON = EXTRACTED_DIR / "fulltext_extraction_clean.json"
SOURCE_POOL_JSON = EXTRACTED_DIR / "fulltext_extraction_500_cleaned.json"
CROSSREF_METADATA_JSON = EXTRACTED_DIR / "crossref_metadata_500.json"
CONVERTED_MAPPING_JSON = SCREENING_DIR / "converted_dir_mapping.json"
FINAL_DOI_LIST = SCREENING_DIR / "final_doi_list.txt"
PRIMARY_DOI_LIST = SCREENING_DIR / "doi_list.txt"
