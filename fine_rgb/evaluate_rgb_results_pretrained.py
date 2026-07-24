"""Evaluate untouched RGB weights on externally generated ``*_results.jpg``.

Only RGB images from ``rgb_results_root`` are passed to the model.  Geometry
ground truth, masks, intrinsics and poses are read from the matching scene in
``data_root``.  No event tensor is loaded by ``PureRgbLdrDataset`` and the
collate function removes all event-related fields defensively.
"""
from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

import finetune_no_event as rgb
from fine_rgb.evaluate_rgb_four_scenes_streaming import evaluate_experiment
from fine_rgb.evaluate_rgb_pretrained_vs_finetuned import DEFAULT_SCENES
from fine_rgb.rgb_ldr_dataset import PureRgbLdrDataset


SEVEN_SCENES = (
    *DEFAULT_SCENES,
    "DH2_Socrates and Seneca_Car_Paint_Midnight",
    "Dragon_1_Car_Paint_Midnight",
    "NAPOLEON_fix_Anodized_Red",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgb-results-root",
        default="/data1/lzh/method/event/HDRev-Diff/results",
    )
    parser.add_argument("--data-root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--pretrained", default="ckpt/model.pt")
    parser.add_argument("--base-config", default="config/finetune_no_event.yaml")
    parser.add_argument(
        "--output-dir", default="exp_f/rgb_pretrained_hdrev_diff_results_scale20"
    )
    parser.add_argument("--scenes", nargs="+", default=list(SEVEN_SCENES))
    parser.add_argument("--ldr-event-ids", default="0,1,2,5,10")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--test-frame-count", type=int, default=120)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--depth-scale", type=float, default=2.0)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument("--max-visuals-per-condition", type=int, default=0)
    # Fields consumed by the shared evaluation implementation.
    parser.set_defaults(
        skip_pretrained=False,
        skip_finetuned=True,
        skip_missing_finetuned=False,
        finetuned_template="",
    )
    return parser.parse_args()


def _scene_dir(root: Path, scene: str) -> Path:
    direct = root / scene
    if direct.is_dir():
        return direct
    matches = [path for path in root.rglob(scene) if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one results directory for scene={scene!r} under {root}, "
            f"found={matches}"
        )
    return matches[0]


def _exposure_dir(scene_dir: Path, exposure: str) -> Path:
    value = str(exposure)
    value = value[3:] if value.startswith("ev_") else value
    names = ("RGB", "rgb", "ev_0") if value == "0" else (f"ev_{value}", value)
    for name in names:
        candidate = scene_dir / name
        if (candidate / "images").is_dir():
            return candidate / "images"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no RGB result folder for exposure=ev_{value} under {scene_dir}; "
        f"tried={names}"
    )


def _numbered_results(directory: Path):
    pattern = re.compile(r"^(\d+)_results\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    numbered = []
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: (item[0], item[1].name))
    if not numbered:
        raise FileNotFoundError(f"no xx_results.jpg images under {directory}")
    if len({number for number, _ in numbered}) != len(numbered):
        raise RuntimeError(f"duplicate numeric result IDs under {directory}")
    return numbered


class ExternalResultsRgbDataset(PureRgbLdrDataset):
    def __init__(self, *args, result_images, **kwargs):
        self.result_images = list(result_images)
        super().__init__(*args, **kwargs)

    def _get_views(self, idx, resolution, rng, num_views):
        views = super()._get_views(idx, resolution, rng, num_views)
        _, start_id = self.start_img_ids[idx]
        width, height = int(resolution[0]), int(resolution[1])
        for offset, view in enumerate(views):
            frame_id = int(start_id) + offset
            if frame_id >= len(self.result_images):
                raise IndexError(
                    f"external RGB has {len(self.result_images)} frames, "
                    f"but GT requested frame {frame_id}"
                )
            with Image.open(self.result_images[frame_id]) as image:
                image = image.convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                view["img"] = image.copy()
            view["external_rgb_path"] = str(self.result_images[frame_id])
        return views


def build_result_loader(args, scene: str, exposure: str):
    scene_results = _scene_dir(Path(args.rgb_results_root), scene)
    image_dir = _exposure_dir(scene_results, exposure)
    numbered = _numbered_results(image_dir)

    # Accept both 000_results.jpg and 001_results.jpg conventions.  Ordering,
    # rather than the literal ID, determines the corresponding GT frame.
    image_paths = [path for _, path in numbered]
    dataset = ExternalResultsRgbDataset(
        ROOT=args.data_root,
        num_views=args.num_views,
        split="test",
        resolution=(args.width, args.height),
        fps=args.fps,
        seed=args.seed,
        scene_names=[scene],
        initial_scene_idx=0,
        active_scene_count=1,
        test_frame_count=args.test_frame_count,
        # This only selects metadata/GT correspondence; its RGB is replaced.
        ldr_event_id="auto",
        return_normal_gt=False,
        result_images=image_paths,
    )
    if len(dataset) <= 0:
        raise RuntimeError(f"no GT test samples for scene={scene}")
    gt_frames = dataset.active_scene_data[scene]["frame_count"]
    if len(image_paths) < gt_frames:
        raise RuntimeError(
            f"scene={scene} exposure={exposure}: external images={len(image_paths)} "
            f"but GT frames={gt_frames}"
        )
    print(
        f"[external RGB] scene={scene} exposure={exposure} "
        f"images={len(image_paths)} clips={len(dataset)} dir={image_dir}",
        flush=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=rgb.rgb_multiview_collate,
    )


def main():
    args = parse_args()
    args.depth_scale = 2.0  # Required fixed protocol; never estimate from test GT.
    exposures = [
        f"ev_{item.strip().removeprefix('ev_')}"
        for item in args.ldr_event_ids.split(",")
        if item.strip()
    ]
    checkpoint = Path(args.pretrained)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # evaluate_experiment resolves make_loader from its defining module.
    import fine_rgb.evaluate_rgb_four_scenes_streaming as streaming

    streaming.make_loader = build_result_loader
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    rows = []
    results = evaluate_experiment(
        name="rgb_pretrained_no_finetune",
        checkpoint_for_exposure=lambda _exposure: checkpoint,
        exposures=exposures,
        args=args,
        config_path=Path(args.base_config),
        device=device,
        rows=rows,
    )

    import csv
    import json

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_root": str(Path(args.rgb_results_root).resolve()),
        "checkpoint": str(checkpoint),
        "rgb_only": True,
        "finetuned": False,
        "event_input": False,
        "depth_scale": 2.0,
        "scenes": list(args.scenes),
        "exposures": exposures,
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (output / "metrics.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved metrics and visualizations to {output.resolve()}", flush=True)
    gc.collect()


if __name__ == "__main__":
    main()
