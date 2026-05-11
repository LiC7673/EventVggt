"""
Event Files Packing Script
Collects all events.h5 files from subdirectories into a single folder.
"""

import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def collect_event_files(source_root, output_dir, preserve_structure=True):
    """Collect all events.h5 files from subdirectories.
    
    Args:
        source_root: Root directory containing subdirectories with event files
                    (e.g., F:\TreeOBJ\reflective_raw\)
        output_dir: Output directory to collect all event files
        preserve_structure: If True, create subdirectories to preserve folder structure
                           If False, rename files with subdirectory names
    """
    
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    
    # Validate source directory
    if not source_root.is_dir():
        print(f"Error: Source directory not found: {source_root}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all event files
    event_files = []
    for subdir in sorted(source_root.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Look for events.h5 in esim_event subdirectory
        event_path = subdir / "esim_event" / "events.h5"
        if event_path.is_file():
            event_files.append((subdir.name, event_path))
    
    if not event_files:
        print(f"Warning: No event files found in {source_root}")
        return
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Copy files
    successful = 0
    failed = 0
    
    for subfolder_name, event_path in tqdm(event_files, desc="Copying event files"):
        try:
            if preserve_structure:
                # Create subdirectory structure: output_dir/{subfolder_name}/events.h5
                dest_subdir = output_dir / subfolder_name
                dest_subdir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_subdir / "events.h5"
            else:
                # Rename with subfolder name: output_dir/{subfolder_name}_events.h5
                dest_path = output_dir / f"{subfolder_name}_events.h5"
            
            shutil.copy2(event_path, dest_path)
            print(f"✓ {subfolder_name}: {event_path} -> {dest_path}")
            successful += 1
            
        except Exception as e:
            print(f"✗ {subfolder_name}: {e}")
            failed += 1
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(event_files)}")


def create_symlinks(source_root, output_dir, preserve_structure=True):
    """Create symbolic links instead of copying files (faster for large files).
    
    Args:
        source_root: Root directory containing subdirectories with event files
        output_dir: Output directory for symbolic links
        preserve_structure: If True, create subdirectories to preserve folder structure
    """
    
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    
    # Validate source directory
    if not source_root.is_dir():
        print(f"Error: Source directory not found: {source_root}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all event files
    event_files = []
    for subdir in sorted(source_root.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Look for events.h5 in esim_event subdirectory
        event_path = subdir / "esim_event" / "events.h5"
        if event_path.is_file():
            event_files.append((subdir.name, event_path))
    
    if not event_files:
        print(f"Warning: No event files found in {source_root}")
        return
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Create symlinks
    successful = 0
    failed = 0
    
    for subfolder_name, event_path in tqdm(event_files, desc="Creating symlinks"):
        try:
            event_path_abs = event_path.resolve()
            
            if preserve_structure:
                # Create subdirectory structure: output_dir/{subfolder_name}/events.h5
                dest_subdir = output_dir / subfolder_name
                dest_subdir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_subdir / "events.h5"
            else:
                # Rename with subfolder name: output_dir/{subfolder_name}_events.h5
                dest_path = output_dir / f"{subfolder_name}_events.h5"
            
            # Remove existing symlink/file
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            
            # Create symlink (absolute path)
            os.symlink(event_path_abs, dest_path)
            print(f"→ {subfolder_name}: {dest_path} -> {event_path_abs}")
            successful += 1
            
        except Exception as e:
            print(f"✗ {subfolder_name}: {e}")
            failed += 1
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(event_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect all events.h5 files from subdirectories into a single folder"
    )
    parser.add_argument(
        "source_root",
        type=str,
        help="Root directory containing subdirectories with event files " +
             "(e.g., F:\\TreeOBJ\\reflective_raw\\)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./collected_events",
        help="Output directory to collect all event files (default: ./collected_events)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying files (faster for large files)"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Rename files with subdirectory names instead of preserving folder structure"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Event Files Packing Script")
    print("=" * 70)
    print(f"Source root: {args.source_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'Symlinks' if args.symlink else 'Copy'}")
    print(f"Structure: {'Flat' if args.flat else 'Preserve'}")
    print("=" * 70)
    
    if args.symlink:
        create_symlinks(
            source_root=args.source_root,
            output_dir=args.output_dir,
            preserve_structure=not args.flat
        )
    else:
        collect_event_files(
            source_root=args.source_root,
            output_dir=args.output_dir,
            preserve_structure=not args.flat
        )


if __name__ == "__main__":
    main()
