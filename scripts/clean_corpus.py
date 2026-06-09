#!/usr/bin/env python3
"""
Produce the canonical analysis corpus used by ALL downstream scripts.

Two restrictions are applied to the raw extraction
(fulltext_extraction_500_cleaned.json), in order:

1. Year window. 15 records fell outside the declared 2018-2025 window
   (2015-2017 x13, 2026 x2, 2 of them flagged deployed). They are dropped as
   inconsistent with the stated corpus window.

2. Full-text only. The reliability of nuanced LLM-extracted fields (deployment
   status, validation method) depends on the model seeing the full text, not an
   abstract. Manual verification showed that every deployment label that could
   NOT be confirmed came from an abstract-only record, while all full-text
   deployment labels were confirmed. We therefore restrict the analysis corpus
   to records with a full-text Markdown conversion in
   data/literature/converted_fulltext/, matched by DOI. This makes every
   reported field auditable against full text.

The result is written to fulltext_extraction_clean.json, the single source of
truth for figures, Table 1, statistics, and the validator. Re-run after any
corpus change.
"""

import json
import re

from paths import ANALYSIS_CORPUS_JSON, CONVERTED_FULLTEXT_DIR, ROOT, SOURCE_POOL_JSON

RAW = SOURCE_POOL_JSON
CONVERTED = CONVERTED_FULLTEXT_DIR
OUT = ANALYSIS_CORPUS_JSON
YEAR_MIN, YEAR_MAX = 2018, 2025


def in_window(rec):
    try:
        y = int(rec.get("year"))
    except (TypeError, ValueError):
        return False
    return YEAR_MIN <= y <= YEAR_MAX


def fulltext_doi_keys():
    """DOI keys (slashes encoded as underscores) that have a full-text MD dir.

    Directory names are "{idx}_{doi}" where the DOI's slashes were replaced by
    underscores. We match in the encoded space (corpus DOI -> replace / with _)
    to avoid ambiguity when a DOI contains more than one slash.
    """
    keys = set()
    for d in CONVERTED.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^\d+_(.+)$", d.name)
        if m:
            keys.add(m.group(1).lower())
    return keys


def main():
    with open(RAW) as f:
        raw = json.load(f)
    records = raw["results"]
    ft_keys = fulltext_doi_keys()

    def has_fulltext(rec):
        doi = (rec.get("doi") or "").lower()
        return bool(doi) and doi.replace("/", "_") in ft_keys

    in_year = [r for r in records if in_window(r)]
    dropped_year = [r for r in records if not in_window(r)]
    kept = [r for r in in_year if has_fulltext(r)]
    dropped_abstract = [r for r in in_year if not has_fulltext(r)]

    out = dict(raw)
    out["results"] = kept
    out["_corpus_cleaning"] = {
        "year_window": [YEAR_MIN, YEAR_MAX],
        "n_raw": len(records),
        "n_in_year": len(in_year),
        "n_dropped_out_of_window": len(dropped_year),
        "dropped_years": sorted(int(r["year"]) for r in dropped_year),
        "n_fulltext_dirs": len(ft_keys),
        "n_kept_fulltext": len(kept),
        "n_dropped_abstract_only": len(dropped_abstract),
        "fulltext_restriction": (
            "Analysis restricted to records with a full-text Markdown conversion "
            "in data/literature/converted_fulltext/, matched by DOI."
        ),
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    c = out["_corpus_cleaning"]
    print(f"Raw extracted:        {c['n_raw']}")
    print(f"In 2018-2025 window:  {c['n_in_year']}  (dropped {c['n_dropped_out_of_window']} out-of-window)")
    print(f"Full-text dirs found: {c['n_fulltext_dirs']}")
    print(f"KEPT (full-text):     {c['n_kept_fulltext']}")
    print(f"Dropped abstract-only:{c['n_dropped_abstract_only']}")
    print(f"Wrote: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
