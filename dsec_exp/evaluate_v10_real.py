"""Evaluate real-data-adapted V10 on DSEC test, one sequence at a time."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from paired_token_reliability.common import torch_load
from dsec_exp.finetune_v10_real import build_model
from dsec_exp import evaluate_dsec as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--clip-stride", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--visual-batches", type=int, default=2)
    p.add_argument("--visual-views", type=int, default=4)
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--allow-unaligned-rgb", action="store_true")
    args = p.parse_args()
    checkpoint = torch_load(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(checkpoint, device); model.eval()

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "finetune_dsec_event.yaml")
    OmegaConf.set_struct(cfg, False); OmegaConf.set_struct(cfg.data, False)
    cfg.data.root = args.root; cfg.data.event_resize_bins = 5
    cfg.data.resolution = [518, 392]; cfg.data.depth_scale = float((checkpoint.get("dsec_args") or {}).get("depth_scale", 1.0))
    cfg.loss.depth_max = 80.0; cfg.batch_size = 1; cfg.num_workers = args.num_workers; cfg.pin_mem = False

    test_root = Path(args.root) / "test"
    scenes = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    if not scenes: raise RuntimeError(f"No test scenes under {test_root}")
    eval_args = SimpleNamespace(**vars(args))
    rows = [base._evaluate_scene(model, cfg, eval_args, scene, device) for scene in scenes]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for key in fields:
        if key in {"scene", "num_windows", "depth_pixels"}: continue
        values = [row.get(key, float("nan")) for row in rows]
        values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        summary[key] = float(sum(values) / len(values)) if values else float("nan")
    report = {"checkpoint": args.checkpoint, "scenes": rows, "metrics_mean": summary}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
