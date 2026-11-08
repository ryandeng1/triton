#!/usr/bin/env python3
"""Combine disjoint RMSNorm screen/confirmation reports without touching GPU state."""
import argparse
import hashlib
import json
from pathlib import Path

from rmsnorm_experiment import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    cases = []
    sources = []
    names = set()
    for path in args.reports:
        raw = path.read_bytes()
        data = json.loads(raw)
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                        "environment": data["environment"], "arguments": data["arguments"],
                        "harness_sha256": data["harness_sha256"]})
        for case in data["cases"]:
            if case["case"] in names:
                raise ValueError(f"Duplicate case {case['case']}")
            names.add(case["case"])
            cases.append(case)
    result = {"source_reports": sources, "cases": cases, "summary": summarize(cases)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
