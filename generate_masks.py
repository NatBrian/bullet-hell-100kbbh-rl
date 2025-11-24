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

# Default background colors (Peach game background)
DEFAULT_BG_COLORS_RGB = [
    (255, 178, 127),  # Peach
    (255, 255, 245),  # Very light
    (255, 255, 215),  # Light yellow
]

class BulletMaskGenerator:
    """
    Generates semantic segmentation masks for bullet-hell game frames.
    Can process individual frames in memory for real-time use.
    """
    
    def __init__(self, bg_colors_bgr=None):
        """
        Initialize the mask generator.
        
        Args:
            bg_colors_bgr: List of background colors in BGR format.
                          If None, uses default peach background colors.
        """
        if bg_colors_bgr is None:
            # Convert default RGB to BGR
            bg_colors_bgr = [np.array(c[::-1], dtype=np.uint8) for c in DEFAULT_BG_COLORS_RGB]
        
        self.bg_colors_bgr = bg_colors_bgr
        self.ship_color_bgr = np.array(SHIP_COLOR_RGB[::-1], dtype=np.uint8)
    
    def generate_mask(self, frame_bgr):
        """
        Generate segmentation mask from a single BGR frame.
        
        Args:
            frame_bgr: NumPy array of shape (H, W, 3) in BGR format
        
        Returns:
            mask: NumPy array of shape (H, W) with values:
                  0 = background
                  1 = bullet
                  2 = ship
        """
        # 1. Initialize mask as BULLET (1) by default
        mask = np.ones(frame_bgr.shape[:2], dtype=np.uint8) * MASK_BULLET
        
        # 2. Detect Background -> 0
        is_bg = np.zeros(frame_bgr.shape[:2], dtype=bool)
        for bg_c in self.bg_colors_bgr:
            dist = np.linalg.norm(frame_bgr - bg_c, axis=2)
            is_bg |= (dist < BG_THRESHOLD)
        mask[is_bg] = MASK_BACKGROUND
        
        # 3. Detect Ship -> 2
        dist_ship = np.linalg.norm(frame_bgr - self.ship_color_bgr, axis=2)
        is_ship = (dist_ship < SHIP_THRESHOLD)
        mask[is_ship] = MASK_SHIP
        
        return mask
    
    def get_positions(self, mask):
        """
        Extract ship and bullet positions from a mask.
        
        Args:
            mask: Segmentation mask (H, W) with values 0, 1, 2
        
        Returns:
            ship_pos: (y, x) tuple of ship center, or None if not found
            bullet_positions: List of (y, x) tuples for bullet centers
        """
        # Find ship position (centroid of all ship pixels)
        ship_pixels = np.where(mask == MASK_SHIP)
        if len(ship_pixels[0]) > 0:
            ship_y = int(np.mean(ship_pixels[0]))
            ship_x = int(np.mean(ship_pixels[1]))
            ship_pos = (ship_y, ship_x)
        else:
            ship_pos = None
        
        # Find bullet positions (use connected components for each bullet)
        bullet_mask = (mask == MASK_BULLET).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(bullet_mask)
        
        bullet_positions = []
        for label in range(1, num_labels):  # Skip 0 (background)
            bullet_pixels = np.where(labels == label)
            if len(bullet_pixels[0]) > 0:
                bullet_y = int(np.mean(bullet_pixels[0]))
                bullet_x = int(np.mean(bullet_pixels[1]))
                bullet_positions.append((bullet_y, bullet_x))
        
        return ship_pos, bullet_positions
    
    def compute_nearest_bullet_distance(self, ship_pos, bullet_positions, normalize_by=None):
        """
        Compute distance to nearest bullet, optionally normalized.
        
        Args:
            ship_pos: (y, x) tuple
            bullet_positions: List of (y, x) tuples
            normalize_by: If provided, divide distance by this value (e.g., frame diagonal)
        
        Returns:
            Distance to nearest bullet (float), or None if no bullets or no ship
        """
        if ship_pos is None or len(bullet_positions) == 0:
            return None
        
        ship_y, ship_x = ship_pos
        min_dist = float('inf')
        
        for bullet_y, bullet_x in bullet_positions:
            dist = np.sqrt((ship_y - bullet_y)**2 + (ship_x - bullet_x)**2)
            min_dist = min(min_dist, dist)
        
        if normalize_by is not None:
            min_dist = min_dist / normalize_by
        
        return min_dist

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
    This is the batch processing interface for CLI use.
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

    # Create generator with analyzed background colors
    bg_colors_bgr = [np.array(c[::-1], dtype=np.uint8) for c in bg_colors_rgb]
    generator = BulletMaskGenerator(bg_colors_bgr)

    for img_path in tqdm(image_files, desc="Generating Masks"):
        try:
            img = cv2.imread(img_path)
            if img is None: continue

            # Use the generator
            mask = generator.generate_mask(img)

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

