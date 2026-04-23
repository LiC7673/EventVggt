import os
import numpy as np
import cv2
from PIL import Image
import argparse

def load_depth_exr(path):
    """Load depth from .exr file using OpenCV."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not load depth from {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]  # Take first channel if multi-channel
    return depth.astype(np.float32)

def load_mask(path):
    """Load mask from image file."""
    mask_img = Image.open(path).convert('L')  # Convert to grayscale
    mask = np.array(mask_img)
    print(f"Mask path: {path}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Unique values in mask: {np.unique(mask)}")
    print(f"Min: {mask.min()}, Max: {mask.max()}")
    # Assuming mask is binary: if mostly 0 or 1, treat 0 as valid (object)
    # If mask has values like 1 for background, invert
    if mask.max() <= 1:
        # Likely 0/1 mask, assume 0 is object
        valid_mask = mask >254
    else:
        # Grayscale, assume 0 is object, 1+ is background
        valid_mask = mask>254  # Only consider pixels where mask > 254 as valid

    # Visualize regions > 254
    highlight = np.zeros((*mask.shape, 3), dtype=np.uint8)
    highlight[mask > 254] = [255, 0, 0]  # Red for >254
    highlight_img = Image.fromarray(highlight)
    vis_path = path.replace('.png', '_highlight.png').replace('.jpg', '_highlight.png')
    highlight_img.save(vis_path)
    print(f"Saved highlight image to {vis_path}")

    return valid_mask

def analyze_depth_with_mask(depth_path, mask_path):
    """Analyze depth within mask."""
    depth = load_depth_exr(depth_path)
    min_depth = np.min(depth)
    max_depth = np.max(depth)
    mean_depth = np.mean(depth)
    median_depth = np.median(depth)
    std_depth = np.std(depth)

    print(f"Depth stats for {os.path.basename(depth_path)}:")
    print(f"  Min: {min_depth:.4f}")
    print(f"  Max: {max_depth:.4f}")
    print(f"  Mean: {mean_depth:.4f}")
    print(f"  Median: {median_depth:.4f}")
    print(f"  Std: {std_depth:.4f}")
    # print(f"  Valid pixels: {masked_depth.size}")

    mask = load_mask(mask_path)
    print(mask)
    # Ensure shapes match
    if depth.shape != mask.shape:
        print(f"Shape mismatch: depth {depth.shape}, mask {mask.shape}")
        # Resize mask to match depth if necessary
        from skimage.transform import resize
        mask = resize(mask.astype(float), depth.shape, order=0) >0.1
    # masked_depth = depth

    masked_depth = depth[mask> 0.9]  # Only consider pixels where mask > 254
    # masked_depth = masked_depth[masked_depth <128]  # Filter out zero depth values
    if masked_depth.size == 0:
        print("No valid depth in mask")
        return
    print(f"Valid depth pixels in mask: {masked_depth.shape}")
    depth[depth>230] = 0
    vis_img = Image.fromarray(depth.astype(np.uint8))
 
    vis_img.save("./masked_depth.png")
    min_depth = np.min(masked_depth)
    max_depth = np.max(masked_depth)
    mean_depth = np.mean(masked_depth)
    median_depth = np.median(masked_depth)
    std_depth = np.std(masked_depth)

    print(f"Depth stats for {os.path.basename(depth_path)}:")
    print(f"  Min: {min_depth:.4f}")
    print(f"  Max: {max_depth:.4f}")
    print(f"  Mean: {mean_depth:.4f}")
    print(f"  Median: {median_depth:.4f}")
    print(f"  Std: {std_depth:.4f}")
    print(f"  Valid pixels: {masked_depth.size}")

def main():
    parser = argparse.ArgumentParser(description="Analyze depth range within mask from .exr files")
    parser.add_argument("depth_path", help="Path to .exr depth file")
    parser.add_argument("mask_path", help="Path to mask image file")
    args = parser.parse_args()

    analyze_depth_with_mask(args.depth_path, args.mask_path)

if __name__ == "__main__":
    main()