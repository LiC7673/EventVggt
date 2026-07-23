"""Collect pure-RGB and full-method hardware JSON into a compact CSV/JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rows, complete = [], {}
    for name, path_string in (("RGB only", args.rgb), ("Full model", args.full)):
        payload = json.loads(Path(path_string).read_text(encoding="utf-8"))
        model, bench = payload["model"], payload["benchmark"]
        rows.append(
            {
                "method": name,
                "parameters_million": model["parameters_million"],
                "tflops_per_forward": bench["profiled_tflops_per_forward"],
                "memory_peak_allocated_gb": bench["peak_allocated_gb"],
                "forward_time_mean_ms": bench["latency_mean_ms"],
                "forward_time_std_ms": bench["latency_std_ms"],
                "forward_time_p50_ms": bench["latency_p50_ms"],
                "forward_time_p95_ms": bench["latency_p95_ms"],
                "views_per_second": bench["views_per_second"],
            }
        )
        complete[name] = payload

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(f"{prefix}.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )
    print(f"Saved {prefix}.csv and {prefix}.json")


if __name__ == "__main__":
    main()
