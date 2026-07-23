"""Collect individual hardware benchmark JSON files into CSV and one JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    source = Path(args.input_dir)
    files = sorted(source.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark JSON files under {source}")
    records = []
    payloads = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = payload["variant"]
        model = payload["model"]
        bench = payload["benchmark"]
        payloads[variant] = payload
        records.append(
            {
                "variant": variant,
                "parameters_million": model["parameters_million"],
                "event_extension_parameters_million": model[
                    "event_extension_parameters_million"
                ],
                "latency_mean_ms": bench["latency_mean_ms"],
                "latency_std_ms": bench["latency_std_ms"],
                "latency_p50_ms": bench["latency_p50_ms"],
                "latency_p95_ms": bench["latency_p95_ms"],
                "samples_per_second": bench["samples_per_second"],
                "views_per_second": bench["views_per_second"],
                "baseline_allocated_gb": bench["baseline_allocated_gb"],
                "incremental_inference_peak_gb": bench[
                    "incremental_inference_peak_gb"
                ],
                "peak_allocated_gb": bench["peak_allocated_gb"],
                "peak_reserved_gb": bench["peak_reserved_gb"],
            }
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(
        json.dumps(payloads, indent=2), encoding="utf-8"
    )
    with Path(f"{prefix}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {prefix}.csv and {prefix}.json")


if __name__ == "__main__":
    main()
