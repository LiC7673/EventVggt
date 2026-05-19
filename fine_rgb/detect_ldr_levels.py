import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fine_rgb.rgb_ldr_dataset import get_rgb_ldr_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Detect available LDR/ev_xx folders for RGB-only ablations.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-names", default=None, help="Comma-separated scene names. Empty means auto.")
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=3)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--mode", choices=["common", "union"], default="common")
    parser.add_argument("--format", choices=["lines", "csv", "json"], default="lines")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_names = None
    if args.scene_names:
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]

    # Dataset constructors print stats to stdout; keep machine-readable output clean.
    with contextlib.redirect_stdout(sys.stderr):
        dataset = get_rgb_ldr_dataset(
            root=args.root,
            num_views=args.num_views,
            resolution=tuple(args.resolution),
            fps=args.fps,
            seed=0,
            scene_names=scene_names,
            initial_scene_idx=args.initial_scene_idx,
            active_scene_count=args.active_scene_count,
            split=args.split,
            test_frame_count=args.test_frame_count,
            ldr_event_id="random",
        )
    ldr_events = dataset.get_active_ldr_events(common=(args.mode == "common"))

    if args.format == "json":
        print(
            json.dumps(
                {
                    "root": args.root,
                    "mode": args.mode,
                    "active_scenes": dataset.get_active_scenes(),
                    "ldr_events": ldr_events,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.format == "csv":
        print(",".join(ldr_events))
    else:
        for ldr_event in ldr_events:
            print(ldr_event)


if __name__ == "__main__":
    main()
