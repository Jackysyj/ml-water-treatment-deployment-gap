#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Screening for Supplementary Papers (DBP + Monitoring).

Reuses screening logic from screen_papers.py.
Input: data/literature/screening/supplement_filtered.json
Output: data/literature/screening/supplement_screening.json

Usage:
  python scripts/screen_supplement.py
  python scripts/screen_supplement.py --start 500
"""

import argparse
import json
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from paths import SCREENING_DIR

PROC_DIR = SCREENING_DIR
INPUT_FILE = PROC_DIR / "supplement_filtered.json"
OUTPUT_FILE = PROC_DIR / "supplement_screening.json"

QWEN_API_KEY = "sk-a1d4ead5825a43f0911fcc2eb71ed969"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-plus"

CONCURRENCY = 10
SCORE_THRESHOLD = 6

SCREENING_PROMPT = """You are screening papers for a water-treatment ML evidence study on operational validation and deployment.

We need papers that apply ML/DL to water/wastewater TREATMENT PROCESSES (not materials discovery).

## PAPER INFO
Title: {title}
Journal: {journal}
Year: {year}
Abstract: {abstract}

## INCLUSION CRITERIA (must meet ALL):
1. Peer-reviewed original research (EXCLUDE: reviews, preprints, editorials, book chapters)
2. Uses ML/DL for QUANTITATIVE PREDICTION or OPTIMIZATION of a water/wastewater treatment PROCESS
3. Studies one of these process-level applications:
   - WWTP: effluent quality prediction (BOD, COD, NH3-N, TSS, TN, TP, pH)
   - membrane: fouling/flux/TMP prediction, cleaning optimization
   - coagulation: coagulant/flocculant dose optimization, turbidity prediction
   - DBP: disinfection byproduct formation prediction (THMs, HAAs, NDMA, bromate, chlorine residual)
   - monitoring: water quality monitoring, soft sensors, anomaly/contamination detection, spectroscopy-based estimation
   - sludge: sludge treatment, dewatering, anaerobic digestion, biogas prediction
   - control: real-time process control, reinforcement learning for plant operation, aeration control
4. Reports quantitative performance metrics (R², RMSE, MAE, accuracy, etc.)

## EXCLUSION CRITERIA (exclude if ANY match):
1. Pure material performance prediction (adsorption capacity, catalytic activity)
2. Material synthesis or characterization
3. Natural water body monitoring WITHOUT treatment context (river, lake, groundwater quality prediction)
4. Pure hydrology/water resource management (flood, drought, streamflow)
5. Simulation-only study without ML component
6. Review, meta-analysis, or survey paper
7. Air quality, soil, or non-water environmental monitoring

## SCORING GUIDE (1-10):
- 9-10: Perfect fit. ML applied to actual plant operation with SCADA/operational data
- 7-8: Good fit. ML for treatment process with experimental or pilot-scale data
- 5-6: Acceptable. Process-level ML but with simulated data or limited scope
- 3-4: Marginal. Borderline (material-process hybrid, or natural water body without treatment)
- 1-2: Poor fit. Clearly material science, natural water body, or off-topic

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown, no explanation):
{{"score": <int 1-10>, "verdict": "include" or "exclude", "sub_field": "WWTP|membrane|coagulation|DBP|monitoring|sludge|control|other", "reason": "<brief 1-sentence explanation>"}}"""


def call_qwen(title, journal, year, abstract, max_retries=3):
    prompt = SCREENING_PROMPT.format(
        title=title or "N/A",
        journal=journal or "N/A",
        year=year or "N/A",
        abstract=abstract[:3000] if abstract else "N/A",
    )
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean markdown wrapping
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            result = json.loads(raw)
            return result
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"score": 0, "verdict": "error", "sub_field": "unknown", "reason": f"JSON parse error"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return {"score": 0, "verdict": "error", "sub_field": "unknown", "reason": str(e)[:100]}


def screen_one_paper(task):
    idx, paper = task
    result = call_qwen(paper["title"], paper.get("journal", ""), paper.get("year", ""), paper.get("abstract", ""))
    result["doi"] = paper["doi"]
    result["title"] = paper["title"]
    result["journal"] = paper.get("journal", "")
    result["year"] = paper.get("year", 0)
    return idx, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    papers = data["papers"]

    # Load existing results for resume
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            prev = json.load(f)
        for r in prev.get("results", []):
            if r.get("doi"):
                existing[r["doi"]] = r

    todo = [(i, p) for i, p in enumerate(papers) if p["doi"] not in existing and i >= args.start]

    print(f"{'='*60}")
    print(f"Supplement Screening: {len(papers)} total, {len(existing)} done, {len(todo)} todo")
    print(f"Model: {QWEN_MODEL} | Concurrency: {CONCURRENCY}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    results = dict(existing)
    include = sum(1 for r in existing.values() if r.get("verdict") == "include")
    exclude = sum(1 for r in existing.values() if r.get("verdict") == "exclude")
    errors = sum(1 for r in existing.values() if r.get("verdict") == "error")
    processed = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(screen_one_paper, t): t[0] for t in todo}
        for future in as_completed(futures):
            idx, result = future.result()
            doi = result["doi"]
            verdict = result.get("verdict", "error")
            score = result.get("score", 0)
            sub = result.get("sub_field", "?")

            if verdict == "include":
                include += 1
                tag = "INCLUDE"
            elif verdict == "exclude":
                exclude += 1
                tag = "EXCLUDE"
            else:
                errors += 1
                tag = "ERROR"

            processed += 1
            results[doi] = result
            print(f"  [{processed:4d}/{len(todo)}] {tag:7s} score={score:2d} {sub:12s} | {result.get('title', '')[:50]}")

            # Save every 50
            if processed % 50 == 0:
                output = {"metadata": {"timestamp": datetime.now().isoformat()}, "results": list(results.values())}
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                print(f"\n  --- Saved. include={include} exclude={exclude} errors={errors} ---\n")

    # Final save
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": QWEN_MODEL,
            "total_screened": len(results),
            "include": include,
            "exclude": exclude,
            "errors": errors,
        },
        "results": list(results.values()),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Screening complete")
    print(f"  Include: {include}")
    print(f"  Exclude: {exclude}")
    print(f"  Errors:  {errors}")
    print(f"  Output:  {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
