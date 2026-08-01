"""Merge baseline/full hardware benchmark JSON files into one compact table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _row(label: str, path: str) -> tuple[dict, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = payload["model"]
    bench = payload["benchmark"]
    row = {
        "method": label,
        "parameters_million": model["parameters_million"],
        "event_extension_parameters_million": model.get(
            "event_extension_parameters_million", 0.0
        ),
        "tflops_per_forward": bench.get("profiled_tflops_per_forward"),
        "latency_mean_ms": bench["latency_mean_ms"],
        "latency_std_ms": bench["latency_std_ms"],
        "latency_p50_ms": bench["latency_p50_ms"],
        "latency_p95_ms": bench["latency_p95_ms"],
        "samples_per_second": bench["samples_per_second"],
        "views_per_second": bench["views_per_second"],
        "baseline_allocated_gb": bench.get("baseline_allocated_gb"),
        "incremental_inference_peak_gb": bench.get(
            "incremental_inference_peak_gb"
        ),
        "peak_allocated_gb": bench.get("peak_allocated_gb"),
        "peak_reserved_gb": bench.get("peak_reserved_gb"),
    }
    return row, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    baseline_row, baseline_payload = _row("RGB baseline", args.baseline)
    extension_row, extension_payload = _row("RGB + proposed modules", args.extension)
    rows = [baseline_row, extension_row]

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(f"{prefix}.json").write_text(
        json.dumps(
            {
                "summary": rows,
                "raw": {
                    "RGB baseline": baseline_payload,
                    "RGB + proposed modules": extension_payload,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {prefix}.csv and {prefix}.json")


if __name__ == "__main__":
    main()
