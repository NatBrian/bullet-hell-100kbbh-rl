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
BG_THRESHOLD = 25   # Adjusted for Manhattan distance (was 15)
SHIP_THRESHOLD = 50 # Adjusted for Manhattan distance (was 30) 

# Mask Values
MASK_BACKGROUND = 0
MASK_BULLET = 1
MASK_SHIP = 2
MASK_ENEMY = 3

# Enemy Detection Constants
ENEMY_BRIGHT_THRESHOLD = 215  # Pixel intensity to be considered part of a bright shape (border/fill)
ENEMY_CENTER_MATCH_THRESHOLD = 40 # Adjusted for Manhattan distance (was 25)

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
        
        # Store as numpy array for fast color distance checks
        self.bg_colors_bgr = np.array(bg_colors_bgr, dtype=np.uint8)
        self.ship_color_bgr = np.array(SHIP_COLOR_RGB[::-1], dtype=np.uint8)
        
        # Identify "dark" background colors (Peach) vs "light" background colors (White/Highlights)
        # This is crucial for distinguishing Hollow Enemies (Peach center) from Filled Geometry (White center)
        self.enemy_bg_colors_bgr = []
        for c in self.bg_colors_bgr:
            # Calculate grayscale intensity (BGR)
            gray = 0.114*c[0] + 0.587*c[1] + 0.299*c[2]
            if gray < ENEMY_BRIGHT_THRESHOLD: 
                self.enemy_bg_colors_bgr.append(c)
        
        if not self.enemy_bg_colors_bgr:
            # Fallback if all BGs are bright (unlikely)
            self.enemy_bg_colors_bgr = self.bg_colors_bgr
            
        # Convert to numpy array for vectorized operations
        self.enemy_bg_colors_bgr = np.array(self.enemy_bg_colors_bgr)
    
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

        # 2. Mark background (0) based on proximity to known background colors
        # Compute per-pixel Manhattan distance to each background color and take the minimum
        if self.bg_colors_bgr.size > 0:
            diff_bg = np.abs(frame_bgr.astype(np.int16)[..., None, :] - self.bg_colors_bgr[None, None, :, :])
            dist_bg = np.sum(diff_bg, axis=3)  # Shape: (H, W, num_bg_colors)
            min_dist_bg = dist_bg.min(axis=2)
            bg_pixels = min_dist_bg < BG_THRESHOLD
            mask[bg_pixels] = MASK_BACKGROUND
        
        # 2. Detect Background -> 0
        is_bg = np.zeros(frame_bgr.shape[:2], dtype=bool)
        
        # Optimization: Use Manhattan distance (sum of absolute differences) instead of Euclidean
        # This is significantly faster as it avoids square and sqrt operations
        for bg_c in self.bg_colors_bgr:
            # diff = np.abs(frame_bgr - bg_c) -> Sum over channels
            # We can use cv2.absdiff for even more speed if needed, but numpy is okay
            diff = np.abs(frame_bgr.astype(np.int16) - bg_c.astype(np.int16))
            dist = np.sum(diff, axis=2)
            is_bg |= (dist < BG_THRESHOLD)
        mask[is_bg] = MASK_BACKGROUND
        
        # 3. Detect Ship -> 2 (and Pink Bullets)
        # The ship is Pink. Some bullets are also Pink.
        # Strategy: Find all pink pixels. The LARGEST pink blob is the Ship. The rest are Bullets.
        diff_ship = np.abs(frame_bgr.astype(np.int16) - self.ship_color_bgr.astype(np.int16))
        dist_ship = np.sum(diff_ship, axis=2)
        is_pink = (dist_ship < SHIP_THRESHOLD)
        
        # Connected components on Pink pixels
        # uint8 conversion needed for connectedComponents
        pink_mask = is_pink.astype(np.uint8)
        num_pink, labels_pink, stats_pink, centroids_pink = cv2.connectedComponentsWithStats(pink_mask, connectivity=8)
        
        if num_pink > 1:
            # Find the label with the largest area (excluding background 0)
            # stats shape: (N, 5). Column 4 is Area (cv2.CC_STAT_AREA)
            areas = stats_pink[1:, cv2.CC_STAT_AREA]
            largest_label_idx = np.argmax(areas) + 1 # +1 because we skipped 0
            
            # Mark the largest blob as SHIP
            mask[labels_pink == largest_label_idx] = MASK_SHIP
            
            # Mark other pink blobs as BULLETS (override default 1, but good to be explicit)
            # Actually, mask is initialized to 1 (BULLET). 
            # So we just need to ensure we don't mark them as SHIP.
            # But wait, if we initialized to BULLET, and then set BG to 0, 
            # the pink pixels that are NOT the ship are currently 1 (BULLET) or 0 (BG)?
            # They were not detected as BG. So they are 1.
            # So we only need to set the SHIP.
            pass
        else:
            # No pink blobs found? Or only background?
            pass
        
        # 4. Detect Enemies -> 3
        # Enemies are "hollow" bright shapes (white border with a hole inside).
        # Background geometry is "filled" bright shapes (no hole).
        # Use Contour Hierarchy to distinguish: hollow shapes have child contours.
        
        # A. Create a binary mask of bright pixels (potential enemies or bg geometry)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, ENEMY_BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Mask out HUD (Bottom Left) to prevent false positives
        h, w = bright_mask.shape
        # Exclude bottom 8% and left 40%
        hud_h = int(h * 0.08)
        hud_w = int(w * 0.40)
        bright_mask[h-hud_h:, :hud_w] = 0
        
        # B. Find contours with hierarchy
        # RETR_CCOMP retrieves 2-level hierarchy (external and holes)
        # hierarchy format: [Next, Previous, First_Child, Parent]
        contours, hierarchy = cv2.findContours(bright_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None and len(contours) > 0:
            hierarchy = hierarchy[0]  # Remove extra dimension
            
            # C. Iterate through contours to find hollow shapes
            for i in range(len(contours)):
                # Check if this contour has a child (hole inside)
                # hierarchy[i][2] is First_Child
                has_hole = hierarchy[i][2] != -1
                
                # Check if this is a top-level contour (not a hole itself)
                # hierarchy[i][3] is Parent
                is_outer = hierarchy[i][3] == -1
                
                if is_outer and has_hole:
                    # This is a hollow shape (ENEMY)
                    # Fill the contour area with MASK_ENEMY
                    # Use a temporary mask to avoid overwriting SHIP
                    temp_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                    cv2.drawContours(temp_mask, contours, i, 1, -1)  # -1 fills the contour
                    
                    # Apply to main mask, preserving SHIP
                    mask[(temp_mask == 1) & (mask != MASK_SHIP)] = MASK_ENEMY
        
        return mask
    
    def get_positions(self, mask):
        """
        Extract ship and bullet positions from a mask.
        
        Args:
            mask: Segmentation mask (H, W) with values 0, 1, 2
        
        Returns:
            ship_pos: (y, x) tuple of ship center, or None if not found
            bullet_positions: List of (y, x) tuples for bullet centers
            enemy_positions: List of (y, x) tuples for enemy centers
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

        # Find enemy positions (use connected components for each enemy)
        enemy_mask = (mask == MASK_ENEMY).astype(np.uint8)
        num_labels_enemy, labels_enemy = cv2.connectedComponents(enemy_mask)
        
        enemy_positions = []
        for label in range(1, num_labels_enemy):  # Skip 0 (background)
            enemy_pixels = np.where(labels_enemy == label)
            if len(enemy_pixels[0]) > 0:
                enemy_y = int(np.mean(enemy_pixels[0]))
                enemy_x = int(np.mean(enemy_pixels[1]))
                enemy_positions.append((enemy_y, enemy_x))
        
        return ship_pos, bullet_positions, enemy_positions
    
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

    def compute_nearest_enemy_distance(self, ship_pos, enemy_positions, normalize_by=None):
        """
        Compute distance to nearest enemy, optionally normalized.
        
        Args:
            ship_pos: (y, x) tuple
            enemy_positions: List of (y, x) tuples
            normalize_by: If provided, divide distance by this value (e.g., frame diagonal)
        
        Returns:
            Distance to nearest enemy (float), or None if no enemies or no ship
        """
        if ship_pos is None or len(enemy_positions) == 0:
            return None
        
        ship_y, ship_x = ship_pos
        min_dist = float('inf')
        
        for enemy_y, enemy_x in enemy_positions:
            dist = np.sqrt((ship_y - enemy_y)**2 + (ship_x - enemy_x)**2)
            min_dist = min(min_dist, dist)
        
        if normalize_by is not None:
            min_dist = min_dist / normalize_by
        
        return min_dist

    def compute_cumulative_risk(self, ship_pos, entity_positions, normalize_by=None):
        """
        Compute cumulative risk based on inverse distance to all entities.
        Risk = Sum(1 / (distance + epsilon))
        
        Args:
            ship_pos: (y, x) tuple
            entity_positions: List of (y, x) tuples
            normalize_by: If provided, divide distances by this value
            
        Returns:
            Total risk score (float). Higher is worse.
        """
        if ship_pos is None or len(entity_positions) == 0:
            return 0.0
            
        # Convert to numpy array for vectorization
        # ship_pos is (y, x), entity_positions is list of (y, x)
        ship_arr = np.array(ship_pos)
        entities_arr = np.array(entity_positions)
        
        # Compute Euclidean distances
        # shape: (N,)
        dists = np.linalg.norm(entities_arr - ship_arr, axis=1)
        
        if normalize_by is not None:
            dists = dists / normalize_by
            
        # Compute Inverse Distance (Risk)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        risk_scores = 1.0 / (dists + epsilon)
        
        return np.sum(risk_scores)

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
                vis_img[mask == MASK_ENEMY] = [255, 0, 0] # Blue for Enemy
                
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

