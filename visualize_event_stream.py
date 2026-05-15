"""
Event Stream Visualization Script
Visualizes event data from event.h5 files with temporal binning.
"""

import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def load_event_data(h5_path):
    """Load events from h5 file.
    
    Args:
        h5_path: Path to event.h5 file
        
    Returns:
        events: numpy array of shape (num_events, 4) with format [timestamp, x, y, polarity]
        t_min, t_max: minimum and maximum timestamps
    """
    with h5py.File(h5_path, "r") as h5_file:
        if "events" not in h5_file:
            raise ValueError(f"'events' dataset not found in {h5_path}")
        events = h5_file["events"][:]
    
    if len(events) == 0:
        raise ValueError(f"No events found in {h5_path}")
    
    t_min = events[:, 0].min()
    t_max = events[:, 0].max()
    
    return events, t_min, t_max


def get_image_resolution(events):
    """Infer image resolution from event coordinates.
    
    Args:
        events: numpy array of shape (num_events, 4)
        
    Returns:
        width, height: image dimensions
    """
    x_max = int(events[:, 1].max()) + 1
    y_max = int(events[:, 2].max()) + 1
    return x_max, y_max


def visualize_event_bin(events, bin_idx, num_bins, output_dir, width, height):
    """Visualize one temporal bin of events.
    
    Args:
        events: numpy array of shape (num_events, 4)
        bin_idx: bin index (0-indexed)
        num_bins: total number of bins
        output_dir: output directory path
        width, height: image dimensions
    """
    # Get time range for this bin
    t_min = events[:, 0].min()
    t_max = events[:, 0].max()
    t_range = t_max - t_min
    print(f"Visualizing bin {bin_idx + 1}/{num_bins} (time range: {t_min:.6f} to {t_max:.6f})...")
    # Calculate bin boundaries
    bin_start_time = t_min + (bin_idx / num_bins) * t_range
    bin_end_time = t_min + ((bin_idx + 1) / num_bins) * t_range
    
    # Filter events in this bin
    mask = (events[:, 0] >= bin_start_time) & (events[:, 0] < bin_end_time)
    bin_events = events[mask]
    
    if len(bin_events) == 0:
        print(f"  Bin {bin_idx:03d}: No events in this bin")
        # Return black image
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        # Create visualization: positive events in red, negative in blue, background black
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Clamp coordinates to valid range
        x_coords = np.clip(bin_events[:, 1].astype(int), 0, width - 1)
        y_coords = np.clip(bin_events[:, 2].astype(int), 0, height - 1)
        polarities = bin_events[:, 3].astype(float)
        
        # Positive events (polarity = 1) -> Red channel
        pos_mask = polarities > 0
        img[y_coords[pos_mask], x_coords[pos_mask], 0] = 255
        
        # Negative events (polarity = 0 or -1) -> Blue channel
        neg_mask = polarities <= 0
        img[y_coords[neg_mask], x_coords[neg_mask], 2] = 255
        
        print(f"  Bin {bin_idx:03d}: {len(bin_events):6d} events " +
              f"({pos_mask.sum():6d} positive, {neg_mask.sum():6d} negative)")
    
    # Save image
    output_path = os.path.join(output_dir, f"event_bin_{bin_idx:03d}.png")
    Image.fromarray(img).save(output_path)
    
    return img


def visualize_events(h5_path, num_frames=120, bin_size=5, output_dir="./vis_event"):
    """Main visualization function.
    
    Args:
        h5_path: Path to event.h5 file
        num_frames: Total number of frames to divide into (default: 120)
        bin_size: Events per bin in temporal dimension (default: 5)
        output_dir: Output directory for visualizations
    """
    print(f"Loading events from {h5_path}...")
    events, t_min, t_max = load_event_data(h5_path)
    print(f"  Total events: {len(events)}")
    print(f"  Time range: {t_min:.6f} to {t_max:.6f} (duration: {t_max - t_min:.6f})")
    
    # Get image dimensions
    width, height = get_image_resolution(events)
    print(f"  Inferred resolution: {width} x {height}")
    
    # Calculate number of bins
    num_bins = max(1, num_frames * bin_size)
    print(f"\nVisualization parameters:")
    print(f"  Total frames: {num_frames}")
    print(f"  Events per bin: {bin_size}")
    print(f"  Number of bins: {num_bins}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating visualizations to {output_dir}...")
    
    # Visualize each bin
    for bin_idx in range(num_bins):
        visualize_event_bin(events, bin_idx, num_bins, output_dir, width, height)
    
    print(f"\nVisualization complete! Generated {num_bins} images.")
    print(f"Output directory: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize event stream from event.h5 file with temporal binning"
    )
    parser.add_argument(
        "event_h5_path",
        type=str,
        help="Path to event.h5 file (e.g., F:\\TreeOBJ\\reflective_raw\\200720_Triton 2_Anodized_Red\\esmi_event\\event.h5)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=120,
        help="Total number of frames to divide events into (default: 120)"
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=5,
        help="Number of frame intervals per visualization bin (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vis_event",
        help="Output directory for visualization images (default: ./vis_event)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.event_h5_path):
        print(f"Error: File not found: {args.event_h5_path}")
        return
    
    try:
        visualize_events(
            h5_path=args.event_h5_path,
            num_frames=args.num_frames,
            bin_size=args.bin_size,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
