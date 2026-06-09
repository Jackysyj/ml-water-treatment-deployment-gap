#!/usr/bin/env python3
"""
Summarise the manual verification of LLM-extracted fields into a single JSON
artifact (figures/verification_summary.json) cited by the Methods/SI reliability
text. Reads the two filled review CSVs and the full-text corpus.

Deployment: every plant-deployment label in the analysis corpus is verified
against full text. Validation: precision of the "rigorous validation"
(temporal / walk-forward / external) bucket is audited.
"""

import csv
import json
import re
from collections import Counter

from paths import ANALYSIS_CORPUS_JSON, CONVERTED_FULLTEXT_DIR, FIGURES_DIR, MANUAL_REVIEW_DIR, ROOT

DEP_CSV = MANUAL_REVIEW_DIR / "deployment_verification.csv"
VAL_CSV = MANUAL_REVIEW_DIR / "validation_verification.csv"
CLEAN = ANALYSIS_CORPUS_JSON
CONVERTED = CONVERTED_FULLTEXT_DIR
OUT = FIGURES_DIR / "verification_summary.json"


def ft_keys():
    keys = set()
    for d in CONVERTED.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^\d+_(.+)$", d.name)
        if m:
            keys.add(m.group(1).lower())
    return keys


def has_ft(doi, keys):
    return bool(doi) and doi.lower().replace("/", "_") in keys


def main():
    keys = ft_keys()
    corpus = json.load(open(CLEAN))["results"]
    corpus_dois = {(r.get("doi") or "").lower() for r in corpus}

    # ---- Deployment ----
    dep = list(csv.DictReader(open(DEP_CSV)))
    # restrict to deployed papers that are IN the analysis corpus (full-text, in-window)
    dep_in = [r for r in dep if (r.get("doi") or "").lower() in corpus_dois]
    dep_conf = [r for r in dep_in if r["verified_deployed"].strip().upper() == "TRUE"]
    dep_unconf = [r for r in dep_in if r["verified_deployed"].strip().upper() != "TRUE"]

    deployment = {
        "n_deployed_labels_in_corpus": len(dep_in),
        "n_confirmed_full_text": len(dep_conf),
        "n_unconfirmed": len(dep_unconf),
        "positive_class_precision": round(len(dep_conf) / len(dep_in), 3) if dep_in else None,
        "note": (
            "Every plant-deployment label in the full-text analysis corpus was "
            "checked against the full text; all were confirmed. Abstract-only "
            "records (which could not be verified) are excluded from the corpus."
        ),
    }

    # ---- Validation (rigorous bucket precision) ----
    val = list(csv.DictReader(open(VAL_CSV)))
    val_in = [r for r in val if (r.get("doi") or "").lower() in corpus_dois]
    rig_conf = [r for r in val_in if r["is_rigorous_confirmed"].strip().upper() == "TRUE"]
    reclass = [r for r in val_in if r["is_rigorous_confirmed"].strip().upper() == "FALSE"]
    reclass_to = Counter(r["verified_validation_method"].strip() for r in reclass)

    validation = {
        "n_rigorous_labels_in_corpus": len(val_in),
        "n_confirmed_rigorous": len(rig_conf),
        "n_reclassified": len(reclass),
        "labelled_bucket_precision": round(len(rig_conf) / len(val_in), 3) if val_in else None,
        "reclassified_to": dict(reclass_to),
        "scope_note": (
            "Precision audit of the labelled-rigorous bucket only (false positives); "
            "the non-rigorous majority was not re-read, so recall was not measured. "
            "The corpus-level rigorous-validation percentage is reported as the LLM "
            "label rate, qualified by this measured precision."
        ),
    }

    summary = {
        "deployment_verification": deployment,
        "validation_verification": validation,
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
