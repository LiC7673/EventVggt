"""Collect TreeOBJ reflective event files into one folder.

Expected source layout:
    F:\TreeOBJ\reflective_raw\<scene_name>\esim_event\events.h5

Default output layout:
    F:\TreeOBJ\reflective_events\<scene_name>_events.h5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(r"F:\TreeOBJ\reflective_raw")
DEFAULT_OUTPUT_DIR = Path(r"F:\TreeOBJ\reflective_events")


def find_event_files(source_root: Path) -> list[tuple[str, Path]]:
    """Return (subfolder_name, events.h5 path) for each valid child folder."""
    event_files: list[tuple[str, Path]] = []

    for child in sorted(source_root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue

        event_path = child / "esim_event" / "events.h5"
        if event_path.is_file():
            event_files.append((child.name, event_path))

    return event_files


def collect_events(
    source_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    source_root = source_root.expanduser()
    output_dir = output_dir.expanduser()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")

    event_files = find_event_files(source_root)
    if not event_files:
        print(f"No events.h5 files found under: {source_root}")
        return

    print(f"Source: {source_root}")
    print(f"Output: {output_dir}")
    print(f"Found: {len(event_files)} events.h5 files")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for scene_name, event_path in event_files:
        dest_path = output_dir / f"{scene_name}_events.h5"

        if dest_path.exists() and not overwrite:
            print(f"[skip] exists: {dest_path}")
            skipped += 1
            continue

        print(f"[copy] {event_path} -> {dest_path}")
        if not dry_run:
            shutil.copy2(event_path, dest_path)
        copied += 1

    action = "Would copy" if dry_run else "Copied"
    print(f"{action}: {copied}, skipped: {skipped}, total: {len(event_files)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect F:\\TreeOBJ\\reflective_raw\\xx\\esim_event\\events.h5 files into one folder."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root folder containing xx subfolders. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to save collected files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_events(
        source_root=args.source_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
