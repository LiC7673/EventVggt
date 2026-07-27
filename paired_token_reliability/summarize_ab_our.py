"""Summarize three four-scene ablation CSVs by exposure."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
EXPOSURES = ("ev_0", "ev_1", "ev_2", "ev_5", "ev_10")
FILES = ("wo_geo.csv", "wo_mul.csv", "wo_re.csv")
METRICS = (
    "abs_rel", "delta1", "rmse_log", "rmse",
    "ate", "rpe_trans", "rpe_rot_deg",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=r"E:\result\eventvgg\ab_our")
    parser.add_argument("--condition", default="final_event_refined")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def number(row, key, source):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{source}: invalid {key}, scene={row.get('scene')!r}, "
            f"exposure={row.get('exposure')!r}, value={row.get(key)!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"{source}: non-finite {key}, scene={row.get('scene')!r}, "
            f"exposure={row.get('exposure')!r}"
        )
    return value


def average(rows, key, source):
    return sum(number(row, key, source) for row in rows) / len(rows)


def summarize_file(source, condition):
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        raise RuntimeError(f"empty CSV: {source}")
    required = {"scope", "scene", "exposure", "condition", *METRICS}
    missing_columns = sorted(required - set(all_rows[0]))
    if missing_columns:
        raise ValueError(f"{source}: missing columns {missing_columns}")
    rows = [
        row for row in all_rows
        if row["scope"] == "scene"
        and row["condition"] == condition
        and row["scene"] in SCENES
        and row["exposure"] in EXPOSURES
    ]
    expected = {(scene, exposure) for scene in SCENES for exposure in EXPOSURES}
    counts = {}
    for row in rows:
        key = (row["scene"], row["exposure"])
        counts[key] = counts.get(key, 0) + 1
    missing = sorted(expected - set(counts))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    if len(rows) != 20 or missing or duplicate:
        raise RuntimeError(
            f"{source}: expected exactly 20 scene rows; selected={len(rows)}, "
            f"missing={missing}, duplicate={duplicate}"
        )

    per_exposure = {}
    for exposure in EXPOSURES:
        selected = [row for row in rows if row["exposure"] == exposure]
        per_exposure[exposure] = {
            metric: average(selected, metric, source) for metric in METRICS
        }
    overall = {metric: average(rows, metric, source) for metric in METRICS}
    return {"per_exposure": per_exposure, "overall_macro": overall}


def latex_row(values, metric):
    return "        & " + " & ".join(
        f"{values[exposure][metric]:.4f}" for exposure in EXPOSURES
    ) + r" \\"


def main():
    args = parse_args()
    root = Path(args.input_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)
    results = {}
    for filename in FILES:
        source = root / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        results[source.stem] = summarize_file(source, args.condition)

    payload = {
        "input_dir": str(root.resolve()),
        "selection": {
            "scope": "scene",
            "condition": args.condition,
            "scenes": list(SCENES),
            "exposures": list(EXPOSURES),
        },
        "ablations": results,
    }
    output = (
        Path(args.output).expanduser()
        if args.output
        else root / "ab_our_four_scene_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, result in results.items():
        values = result["per_exposure"]
        overall = result["overall_macro"]
        print(f"\n[{name}] condition={args.condition}")
        print("metric       ev_0       ev_1       ev_2       ev_5      ev_10")
        for metric in METRICS:
            cells = " ".join(
                f"{values[exposure][metric]:10.6f}"
                for exposure in EXPOSURES
            )
            print(f"{metric:<11} {cells}")
        print(
            "overall: "
            + " ".join(f"{key}={overall[key]:.6f}" for key in METRICS)
        )
        print("LaTeX depth:")
        for metric in ("abs_rel", "delta1", "rmse_log"):
            print(latex_row(values, metric))
        print(
            "LaTeX pose overall: "
            f"& {overall['ate']:.4f} & {overall['rpe_trans']:.4f} "
            f"& {overall['rpe_rot_deg']:.4f} \\\\"
        )
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
