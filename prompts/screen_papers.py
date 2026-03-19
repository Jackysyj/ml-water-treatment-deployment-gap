#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based Abstract Screening for Water Treatment Process ML Papers.

Uses Qwen3.5-Plus to score and filter papers based on abstracts.
Input: data/processed/filtered_papers.json
Output: data/processed/screening_results.json + data/final_doi_list.txt

Author: Claude Code Assistant
Date: 2026-02-28
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"

# Qwen API config
QWEN_API_KEY = "sk-a1d4ead5825a43f0911fcc2eb71ed969"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen3.5-plus"

SCORE_THRESHOLD = 6  # Papers scoring >= 6 are included

SCREENING_PROMPT = """You are screening papers for a Nature Water Perspective article on "Why Machine Learning Has Not Yet Transformed Water Treatment".

We need papers that apply ML/DL to water/wastewater TREATMENT PROCESSES (not materials discovery).

## PAPER INFO
Title: {title}
Journal: {journal}
Year: {year}
Abstract: {abstract}

## INCLUSION CRITERIA (must meet ALL):
1. Peer-reviewed original research (EXCLUDE: reviews, preprints, editorials, book chapters, conference papers)
2. Uses ML/DL for QUANTITATIVE PREDICTION or OPTIMIZATION of a water/wastewater treatment PROCESS
3. Studies one of these process-level applications:
   - WWTP: effluent quality prediction (BOD, COD, NH3-N, TSS, TN, TP, pH)
   - membrane: fouling/flux/TMP prediction, cleaning optimization
   - coagulation: coagulant/flocculant dose optimization, turbidity prediction
   - DBP: disinfection byproduct formation prediction (THMs, HAAs, NDMA)
   - monitoring: water quality monitoring, anomaly/contamination detection in distribution systems
   - sludge: sludge treatment, dewatering, anaerobic digestion, biogas prediction
   - control: real-time process control, reinforcement learning for plant operation, aeration control
4. Reports quantitative performance metrics (R², RMSE, MAE, accuracy, etc.)

## EXCLUSION CRITERIA (exclude if ANY match):
1. Pure material performance prediction (adsorption capacity, catalytic activity, material properties)
2. Material synthesis or characterization
3. Natural water body monitoring WITHOUT treatment context (river, lake, groundwater, estuary quality prediction)
4. Pure hydrology/water resource management (flood, drought, streamflow prediction)
5. Simulation-only study without ML component (e.g., pure CFD, ASM modeling)
6. Review, meta-analysis, or survey paper

## SCORING GUIDE (1-10):
- 9-10: Perfect fit. ML applied to actual plant operation with SCADA/operational data
- 7-8: Good fit. ML for treatment process with experimental or pilot-scale data
- 5-6: Acceptable. Process-level ML but with simulated data, limited scope, or weak ML component
- 3-4: Marginal. Borderline (e.g., material-process hybrid, or natural water body without treatment)
- 1-2: Poor fit. Clearly material science, natural water body, or off-topic

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown, no explanation):
{{"score": <int 1-10>, "verdict": "include" or "exclude", "sub_field": "WWTP|membrane|coagulation|DBP|monitoring|sludge|control|other", "reason": "<brief 1-sentence explanation>"}}"""


def call_qwen(title, journal, year, abstract, max_retries=3):
    """Call Qwen API for paper screening."""
    prompt = SCREENING_PROMPT.format(
        title=title or "N/A",
        journal=journal or "N/A",
        year=year or "N/A",
        abstract=abstract or "N/A",
    )

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": "You are a water treatment and ML expert. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Clean potential markdown wrapping
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if content.startswith("json"):
                content = content[4:].strip()

            result = json.loads(content)

            # Validate required fields
            assert "score" in result and isinstance(result["score"], int)
            assert "verdict" in result and result["verdict"] in ("include", "exclude")
            assert "sub_field" in result
            assert "reason" in result

            return result

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {
                "score": 0, "verdict": "error",
                "sub_field": "other",
                "reason": f"Parse error: {str(e)[:100]}",
            }
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return {
                "score": 0, "verdict": "error",
                "sub_field": "other",
                "reason": f"API error: {str(e)[:100]}",
            }


CONCURRENCY = 5  # Number of parallel API calls


def screen_one_paper(idx_paper):
    """Screen a single paper. Returns (index, result_dict)."""
    i, paper = idx_paper
    title = paper.get("title", "")
    journal = paper.get("journal", "")
    year = paper.get("publication_year", "")
    abstract = paper.get("abstract", "")

    if not (abstract or "").strip():
        return i, {
            "paper_idx": i,
            "title": title,
            "doi": paper.get("doi", ""),
            "journal": journal,
            "year": year,
            "openalex_id": paper.get("openalex_id", ""),
            "score": 0,
            "verdict": "exclude",
            "sub_field": "other",
            "reason": "No abstract available",
        }

    result = call_qwen(title, journal, year, abstract)

    if result["verdict"] != "error":
        result["verdict"] = "include" if result["score"] >= SCORE_THRESHOLD else "exclude"

    result["paper_idx"] = i
    result["title"] = title
    result["doi"] = paper.get("doi", "")
    result["journal"] = journal
    result["year"] = year
    result["openalex_id"] = paper.get("openalex_id", "")

    return i, result


def run_screening():
    """Run LLM screening on all filtered papers with concurrency."""
    input_path = PROC_DIR / "filtered_papers.json"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run search_openalex.py first.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data["papers"]
    total = len(papers)
    print(f"{'='*70}")
    print(f"LLM Screening — {total} papers")
    print(f"Model: {QWEN_MODEL} | Concurrency: {CONCURRENCY}")
    print(f"Threshold: score >= {SCORE_THRESHOLD}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    results = [None] * total
    include_count = 0
    exclude_count = 0
    error_count = 0
    done = 0

    tasks = [(i, paper) for i, paper in enumerate(papers)]

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(screen_one_paper, t): t[0] for t in tasks}

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done += 1

            if result["verdict"] == "include":
                include_count += 1
            elif result["verdict"] == "error":
                error_count += 1
            else:
                exclude_count += 1

            verdict_str = "✓" if result["verdict"] == "include" else "✗" if result["verdict"] == "exclude" else "?"
            print(f"  [{done:4d}/{total}] {verdict_str} Score={result['score']:2d} "
                  f"[{result['sub_field']:12s}] {result['title'][:45]}")

            if done % 100 == 0:
                print(f"\n  --- Progress: {done}/{total} | "
                      f"{include_count} include / {exclude_count} exclude / {error_count} error ---\n")

    # ---- Save screening results ----
    output_path = PROC_DIR / "screening_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_screened": total,
                "include": include_count,
                "exclude": exclude_count,
                "errors": error_count,
                "threshold": SCORE_THRESHOLD,
                "model": QWEN_MODEL,
                "timestamp": datetime.now().isoformat(),
            },
            "results": [r for r in results if r is not None],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Screening results saved: {output_path}")

    # ---- Generate final DOI list ----
    included = [r for r in results if r and r["verdict"] == "include"]
    doi_path = PROJECT_ROOT / "data" / "final_doi_list.txt"
    with open(doi_path, "w") as f:
        f.write(f"# Water Treatment Process ML — Final DOI List (LLM Screened)\n")
        f.write(f"# Total: {len(included)} papers\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"# Screening: {QWEN_MODEL}, threshold >= {SCORE_THRESHOLD}\n\n")
        for r in included:
            doi = r["doi"].replace("https://doi.org/", "")
            if doi:
                f.write(f"{doi}\n")
    print(f"  Final DOI list saved: {doi_path} ({len(included)} papers)")

    # ---- Analysis ----
    print(f"\n{'='*70}")
    print("SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Total screened:  {total}")
    print(f"  Included:        {include_count}")
    print(f"  Excluded:        {exclude_count}")
    print(f"  Errors:          {error_count}")

    # Score distribution
    print("\nScore distribution:")
    score_dist = defaultdict(int)
    for r in results:
        score_dist[r["score"]] += 1
    for s in range(10, -1, -1):
        if score_dist[s] > 0:
            bar = "█" * score_dist[s]
            print(f"  Score {s:2d}: {score_dist[s]:4d} {bar}")

    # Sub-field distribution (included only)
    print("\nSub-field distribution (included):")
    field_dist = defaultdict(int)
    for r in included:
        field_dist[r["sub_field"]] += 1
    for field, count in sorted(field_dist.items(), key=lambda x: -x[1]):
        print(f"  {field:15s}: {count:4d}")

    # Check coverage
    target_fields = ["WWTP", "membrane", "coagulation", "DBP", "monitoring", "sludge", "control"]
    missing = [f for f in target_fields if field_dist.get(f, 0) < 3]
    if missing:
        print(f"\n  WARNING: Low coverage in: {', '.join(missing)}")
        print(f"  Consider adding targeted searches for these sub-fields.")

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_screening()
