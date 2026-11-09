#!/usr/bin/env python3
"""Serialize diagnostic Nsight runs after nonprofiled slowdown measurements."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
METRICS = [
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smsp__inst_executed.sum", "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
] + [f"smsp__average_warps_issue_stalled_{kind}_per_issue_active.ratio" for kind in (
    "barrier", "membar", "long_scoreboard", "short_scoreboard", "mio_throttle", "lg_throttle")]
PLAN = [(study, variant) for study, variant_list in (
    ("upsample", ("baseline", "stages2", "stages3")),
    ("logsoftmax", ("baseline", "stages2", "stages3")),
    ("rotary_w2", ("baseline", "stages2", "stages4")),
) for variant in variant_list]


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=len(PLAN))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = json.loads((args.output / "commands.json").read_text()) if (args.output / "commands.json").exists() else []
    summary = {}
    write(args.output / "plan.json", dict(metrics=METRICS, plan=PLAN[:args.limit],
        replay="application", cache_control="none", clock_control="none",
        selection="sixth launch after profiler start; five identical warm launches in every application replay",
        caution="Short kernels and multi-pass counters are diagnostic; shuffled CUDA graph timing is performance truth. Global LSU sectors do not include all async copy traffic."))
    for study, variant in PLAN[:args.limit]:
        name = f"{study}.{variant}"
        csv_path = args.output / f"{name}.csv"
        command = ["/usr/local/cuda-13.0/bin/ncu", "--replay-mode", "application",
            "--profile-from-start", "off", "--launch-skip", "5", "--launch-count", "1",
            "--cache-control", "none", "--clock-control", "none",
            "--metrics", ",".join(METRICS), "--csv", "--page", "raw",
            "--log-file", str(csv_path), sys.executable, str(HERE / "pipeline_slowdown_experiment.py"),
            "--output", str(args.output / "profile_runtime"), "--profile", "--studies", study,
            "--variant", variant]
        if not csv_path.exists():
            print("PROFILE", name, flush=True)
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            commands.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            write(args.output / "commands.json", commands)
            if result.returncode:
                raise RuntimeError(f"NCU failed: {name}: {result.stderr}; see commands.json and CSV")
        else:
            assert any(c["command"] == command and c["returncode"] == 0 for c in commands), "Only reuse successful exact command"
        raw = csv_path.read_text()
        header = raw.find('"ID",')
        assert header >= 0, raw
        rows = list(csv.DictReader(io.StringIO(raw[header:])))
        if "Metric Name" in rows[0]:
            summary[name] = {r["Metric Name"]: {"value": r["Metric Value"], "unit": r["Metric Unit"]}
                             for r in rows if r.get("Metric Name")}
        else:
            # --page raw emits a wide table: one units row then one kernel row.
            assert len(rows) == 2 and rows[1]["ID"] == "0", "Exactly one measured launch"
            summary[name] = {m: {"value": rows[1][m], "unit": rows[0][m]} for m in METRICS}
        assert len(summary[name]) >= len(METRICS)
        write(args.output / "summary.json", summary)
        print("DONE", name, summary[name]["gpu__time_duration.sum"], flush=True)


if __name__ == "__main__":
    main()
