import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import glob
from collections import Counter

# --- Configuration Constants (RGB) ---
# Ship Color (Pink Triangle) - Constant
SHIP_COLOR_RGB = (255, 82, 163)

# Thresholds
BG_THRESHOLD = 15   # Tightened to distinguish off-white BG from white bullets
SHIP_THRESHOLD = 30 

# Mask Values
MASK_BACKGROUND = 0
MASK_BULLET = 1
MASK_SHIP = 2

def analyze_background_colors(image_files, num_samples=50):
    """
    Analyzes a subset of images to find the dominant background colors.
    Returns a list of RGB tuples.
    """
    print(f"Analyzing background colors from {min(len(image_files), num_samples)} samples...")
    
    samples = image_files[:]
    if len(samples) > num_samples:
        np.random.shuffle(samples)
        samples = samples[:num_samples]
    
    global_counter = Counter()
    
    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img.reshape(-1, 3)
        pixels = [tuple(p) for p in pixels]
        global_counter.update(pixels)
        
    total_pixels = sum(global_counter.values())
    background_colors = []
    
    print("Detected Background Colors:")
    for color, count in global_counter.most_common(10): # Check top 10
        percentage = (count / total_pixels) * 100
        # Background is usually significant (> 1%)
        if percentage > 1.0: 
            print(f"RGB: {color} | {percentage:.2f}% (Background)")
            background_colors.append(color)
        else:
            print(f"RGB: {color} | {percentage:.2f}% (Object/Noise)")
            
    return background_colors

def generate_masks(input_dir, output_dir, visualize=True, use_morphology=True):
    """
    Generates segmentation masks using auto-detected background subtraction.
    """
    os.makedirs(output_dir, exist_ok=True)

    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # --- Step 1: Analyze Colors ---
    bg_colors_rgb = analyze_background_colors(image_files)
    
    # --- Step 2: Generate Masks ---
    print(f"Processing {len(image_files)} images from {input_dir}...")

    bg_colors_bgr = [np.array(c[::-1], dtype=np.uint8) for c in bg_colors_rgb]
    ship_color_bgr = np.array(SHIP_COLOR_RGB[::-1], dtype=np.uint8)

    for img_path in tqdm(image_files, desc="Generating Masks"):
        try:
            img = cv2.imread(img_path)
            if img is None: continue

            # 1. Initialize Mask as BULLET (1)
            mask = np.ones(img.shape[:2], dtype=np.uint8) * MASK_BULLET

            # 2. Detect Background -> 0
            is_bg = np.zeros(img.shape[:2], dtype=bool)
            for bg_c in bg_colors_bgr:
                dist = np.linalg.norm(img - bg_c, axis=2)
                is_bg |= (dist < BG_THRESHOLD)
            mask[is_bg] = MASK_BACKGROUND

            # 3. Detect Ship -> 2
            dist_ship = np.linalg.norm(img - ship_color_bgr, axis=2)
            is_ship = (dist_ship < SHIP_THRESHOLD)
            mask[is_ship] = MASK_SHIP

            # 4. Post-processing
            if use_morphology:
                # Optional cleanup
                pass

            # Debug: Check if mask is empty
            unique_vals = np.unique(mask)
            if len(unique_vals) == 1 and unique_vals[0] == 0:
                # Only background found
                pass 

            # Save Mask (0, 1, 2)
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            out_name = f"{name}_mask{ext}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, mask)

            # Save Visualization (for human debugging)
            if visualize:
                vis_img = np.zeros_like(img)
                # 0 (BG) -> Black (0,0,0)
                # 1 (Bullet) -> Green (0, 255, 0)
                # 2 (Ship) -> Red (0, 0, 255) - BGR
                vis_img[mask == MASK_BULLET] = [0, 255, 0]
                vis_img[mask == MASK_SHIP] = [0, 0, 255]
                
                vis_path = os.path.join(output_dir, f"{name}_vis{ext}")
                cv2.imwrite(vis_path, vis_img)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Done! Masks saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="game_screenshots")
    parser.add_argument("--output", type=str, default="game_screenshots_mask")
    parser.add_argument("--no-morph", action="store_true")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization output.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_input = os.path.join(repo_root, args.input)
        if os.path.exists(alt_input):
            args.input = alt_input
        else:
            print(f"Error: Input directory '{args.input}' does not exist.")
            exit(1)

    generate_masks(args.input, args.output, visualize=not args.no_vis, use_morphology=not args.no_morph)
