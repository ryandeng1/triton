#!/usr/bin/env python3
"""CPU-only audit of TE machine-code equivalence and measured comparisons."""

import argparse
import hashlib
import json
import statistics
import struct
from pathlib import Path


def text_sections(path):
    """Hash ELF64 little-endian .text sections without debug/symbol-name bytes."""
    data = Path(path).read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, name_index = struct.unpack_from("<HHH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + size * i) for i in range(count)]
    name_section = sections[name_index]
    names = data[name_section[4]:name_section[4] + name_section[5]]
    code = b"".join(data[s[4]:s[4] + s[5]] for s in sections
                    if names[s[0]:].split(b"\0")[0].startswith(b".text."))
    assert code
    return {"bytes": len(code), "sha256": hashlib.sha256(code).hexdigest()}


def analyze(screen, confirm):
    audits = []
    metadata_fields = ("n_regs", "n_spills", "shared", "num_warps", "num_ctas", "num_stages",
                       "global_scratch_size", "profile_scratch_size", "launch_cooperative_grid", "launch_pdl")
    for row in screen["rows"]:
        if row["status"] != "passed":
            continue
        variants = row["variants"]
        code = {v: text_sections(a["artifacts"]["files"]["cubin"]["path"]) for v, a in variants.items()}
        metadata = {v: {f: a["artifacts"]["resources"].get(f) for f in metadata_fields}
                    for v, a in variants.items()}
        audits.append({"shape": row["shape"], "config": row["config"], "code": code,
                       "metadata": metadata,
                       "manual_contract_machine_code_equal": code["hoisted"] == code["contract"],
                       "manual_contract_metadata_equal": metadata["hoisted"] == metadata["contract"],
                       "original_contract_machine_code_equal": code["original"] == code["contract"]})
    comparisons = {}
    for key, case in confirm["summary"].items():
        comparisons[key] = {"best": case["best"],
                            "best_to_best_speedups": case["best_to_best_speedups"],
                            "best_by_config_family": case["best_by_config_family"],
                            "best_to_best_ratios_by_seed": {}}
        rows = [r for r in confirm["rows"] if r["status"] == "passed" and r["shape"] == case["shape"]]
        best = case["best"]
        originals = {r["seed"]: r["timing"]["medians_us"]["original"] for r in rows
                     if r["config"] == best["original"]["config"]}
        for v in ("hoisted", "contract"):
            ratios = {str(r["seed"]): originals[r["seed"]] / r["timing"]["medians_us"][v] for r in rows
                      if r["config"] == best[v]["config"]}
            comparisons[key]["best_to_best_ratios_by_seed"][v] = ratios
        comparisons[key]["published_best_to_best_speedups"] = {
            v: case["best_by_config_family"]["published"]["original"]["median_us"] / value["median_us"]
            for v, value in case["best_by_config_family"]["published"].items()}
    valid = [r for report in (screen, confirm) for r in report["rows"] if r["status"] == "passed"]
    return {"screen_count": len(screen["rows"]), "confirm_count": len(confirm["rows"]),
            "screen_all_passed": screen["all_passed"], "confirm_all_passed": confirm["all_passed"],
            "all_paired_outputs_byte_equal": all(r["variants_byte_equal"] for r in valid),
            "all_timing_outputs_shared": all(r["timing"]["shared_output_allocation"] for r in valid),
            "all_shared_bindings_validated": all(set(r["timing_binding_validation"]) == set(r["variants"])
                                                  for r in valid),
            "max_probability_reference_absolute_error": max(
                a["validation_before"]["prob_max_abs_error"] for r in valid for a in r["variants"].values()),
            "all_manual_contract_machine_code_equal": all(a["manual_contract_machine_code_equal"] for a in audits),
            "all_manual_contract_metadata_equal": all(a["manual_contract_metadata_equal"] for a in audits),
            "comparisons": comparisons, "code_audits": audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--confirm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.screen.read_text()), json.loads(args.confirm.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "code_audits"}, indent=2))


if __name__ == "__main__":
    main()
