"""
Enhanced mask generation module for bullet-hell games.

This module provides a `BulletMaskGenerator` class for creating semantic
segmentation masks that distinguish the player's ship, hostile bullets,
environmental geometry, and enemies from raw game frames.  The design
retains the same public interface as the original implementation but
introduces several improvements:

* **Colour‐space reasoning** – Frames are converted to HSV space to
  extract red/pink hues characteristic of bullets more reliably than
  simple RGB distance measures.  Hue thresholds wrap around the hue
  axis to handle both light pink and dark red bullets.

* **Robust ship detection** – The player's ship remains a
  constant pink colour.  It is detected by computing the Manhattan
  distance to the known ship colour and selecting the largest connected
  component.  This avoids confusing the ship with long chains of
  bullets that may also include pink hues.

* **Improved enemy extraction** – Enemies appear as bright, hollow
  shapes whose centres match the darker peach background.  The new
  implementation first threshold‑segments all bright shapes then
  iterates through their contours.  A contour is considered an enemy if
  it has a child contour (i.e. a hole) **or** if the mean interior
  luminance is sufficiently dark.  This heuristic allows detection of
  enemies even when their hole partially overlaps with other bright
  geometry.

* **Background and geometry masking** – Pixels that closely match
  pre‑analysed background colours are labelled background.  Remaining
  bright geometry (e.g. decorative diamonds) is suppressed and not
  misclassified as bullets.  Only pixels within the red/pink hue range
  are considered bullets.

The top–level functions `analyze_background_colors` and
`generate_masks` are preserved for backwards compatibility.  When run
from the command line the script will analyse background colours in a
directory of screenshots and write corresponding mask images.  The
visualisation uses green for bullets, red for the player ship and blue
for enemies.

Note: This file does not rely on any external I/O at import time and is
designed to be safe to import as a library.  All heavy processing is
performed inside methods which must be invoked explicitly.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# --- Configuration Constants (RGB) ---
# Ship colour (pink triangle) – constant across frames
SHIP_COLOR_RGB: Tuple[int, int, int] = (255, 82, 163)

# Thresholds for Manhattan distance in RGB space
BG_THRESHOLD: int = 25
SHIP_THRESHOLD: int = 50

# Mask values
MASK_BACKGROUND: int = 0
MASK_BULLET: int = 1
MASK_SHIP: int = 2
MASK_ENEMY: int = 3

# Enemy detection constants
# Pixels brighter than this grayscale value are considered part of
# bright shapes (potential enemies or geometry)
ENEMY_BRIGHT_THRESHOLD: int = 215
# For comparing interior colour of candidate enemy shapes – not used
# directly but kept for backwards compatibility
ENEMY_CENTER_MATCH_THRESHOLD: int = 40

# Default background colours (RGB) used when no dynamic analysis is
# available.  These correspond to the peach game background and
# highlight colours.  They will be converted to BGR internally.
DEFAULT_BG_COLORS_RGB: List[Tuple[int, int, int]] = [
    (255, 178, 127),  # Peach
    (255, 255, 245),  # Very light
    (255, 255, 215),  # Light yellow
]


class BulletMaskGenerator:
    """
    Generates semantic segmentation masks for bullet‑hell game frames.

    The class can be instantiated with a set of background colours in
    BGR format.  It then provides a `generate_mask` method that takes
    a single BGR frame and returns a 2‑D mask array with values:

      * 0 – background
      * 1 – bullet
      * 2 – player ship
      * 3 – enemy ship

    Methods for extracting positions and computing distances are kept
    unchanged for compatibility with downstream code.
    """

    def __init__(self, bg_colors_bgr: Optional[Iterable[np.ndarray]] = None) -> None:
        # Convert default RGB colours to BGR if no custom list was supplied
        if bg_colors_bgr is None:
            bg_colors_bgr = [np.array(c[::-1], dtype=np.uint8) for c in DEFAULT_BG_COLORS_RGB]
        # Store background colours as an array for vectorised distance checks
        self.bg_colors_bgr: np.ndarray = np.array(list(bg_colors_bgr), dtype=np.uint8)
        # Store ship colour in BGR order for fast Manhattan distance computation
        self.ship_color_bgr: np.ndarray = np.array(SHIP_COLOR_RGB[::-1], dtype=np.uint8)
        # Identify "dark" background colours based on luminance.  These are
        # used when distinguishing hollow enemies from filled geometry.
        enemy_bg: List[np.ndarray] = []
        for c in self.bg_colors_bgr:
            # Convert BGR to grayscale using ITU‑R BT.601 coefficients
            gray_val = 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
            if gray_val < ENEMY_BRIGHT_THRESHOLD:
                enemy_bg.append(c)
        if not enemy_bg:
            enemy_bg = list(self.bg_colors_bgr)
        self.enemy_bg_colors_bgr: np.ndarray = np.array(enemy_bg, dtype=np.uint8)

        # Precompute HSV ranges for bullet hues (red/pink).  Two ranges
        # are necessary because red wraps around the hue axis.  Hue values
        # span 0–179 in OpenCV.
        # The lower range captures dark reds and the upper captures pinks.
        self.bullet_hsv_ranges: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.array([0,  80,  50], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)),
            (np.array([160,  80,  50], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
        ]
        # Minimum area (in pixels) for a connected bullet cluster.  Small
        # regions below this threshold are discarded as noise.  The value
        # may be adjusted based on game resolution.
        self.bullet_min_area: int = 4
        
        # PERFORMANCE: Cache morphological kernel (avoid recreation every frame)
        self._morph_kernel: np.ndarray = np.ones((3, 3), np.uint8)
        
        # PERFORMANCE: Preallocate int16 version of ship color for broadcasting
        self._ship_color_int16: np.ndarray = self.ship_color_bgr.astype(np.int16)
        self._bg_colors_int16: np.ndarray = self.bg_colors_bgr.astype(np.int16)
        
        # Cache for reusable buffers (set in first generate_mask call)
        self._cached_height: int = 0
        self._cached_width: int = 0
        self._reusable_temp_mask: Optional[np.ndarray] = None

    def generate_mask(self, frame_bgr: np.ndarray, skip_enemy: bool = False) -> np.ndarray:
        """Generate a segmentation mask from a single BGR frame.

        The returned mask uses the integer constants defined in this module
        to label each pixel as background, bullet, ship or enemy.

        Args:
            frame_bgr: NumPy array of shape (H, W, 3) in BGR colour space.
            skip_enemy: If True, skip enemy detection for ~30% performance boost.

        Returns:
            mask: NumPy array of shape (H, W) with values in {0, 1, 2, 3}.
        """
        # Ensure input is in uint8
        frame_bgr = frame_bgr.astype(np.uint8, copy=False)
        height, width = frame_bgr.shape[:2]
        # Initialise mask with background label by default
        mask = np.full((height, width), MASK_BACKGROUND, dtype=np.uint8)

        # -----------------------------------------------------------------
        # Step 1: Identify background pixels based on Manhattan distance to
        # known background colours.  Using a vectorised formulation avoids
        # Python loops and accelerates per‑frame processing.
        # PERFORMANCE: Use cached int16 version to avoid repeated casting
        if self.bg_colors_bgr.size > 0:
            # Convert frame once to int16
            frame_int16 = frame_bgr.astype(np.int16)
            # Broadcast subtraction and take absolute value
            diff = np.abs(frame_int16[..., None, :] - self._bg_colors_int16[None, None, :, :])
            # Sum absolute differences across the colour channels
            dist = diff.sum(axis=3)
            # For each pixel take the minimum distance across all background colours
            min_dist_bg = dist.min(axis=2)
            bg_mask = min_dist_bg < BG_THRESHOLD
            # Label background pixels
            mask[bg_mask] = MASK_BACKGROUND
        else:
            bg_mask = np.zeros((height, width), dtype=bool)

        # -----------------------------------------------------------------
        # Step 2: Ship detection.  The player ship is a large pink
        # connected component.  We use Manhattan distance in RGB space to
        # select candidate pink pixels and then label the largest
        # component as the ship.
        # PERFORMANCE: Reuse frame_int16 if available, use cached ship color
        if self.bg_colors_bgr.size > 0:
            # Reuse the int16 frame from background detection
            diff_ship = np.abs(frame_int16 - self._ship_color_int16)
        else:
            diff_ship = np.abs(frame_bgr.astype(np.int16) - self._ship_color_int16)
        dist_ship = diff_ship.sum(axis=2)
        pink_mask = dist_ship < SHIP_THRESHOLD
        # Connected components on the binary pink mask.  Only use 8‑way
        # connectivity to avoid splitting diagonal pixels into separate
        # components.
        num_pink, labels_pink, stats_pink, _ = cv2.connectedComponentsWithStats(pink_mask.astype(np.uint8), connectivity=8)
        ship_mask = np.zeros((height, width), dtype=bool)
        if num_pink > 1:
            # Skip label 0 which corresponds to background
            areas = stats_pink[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            ship_mask = labels_pink == largest_label
            # Label ship pixels
            mask[ship_mask] = MASK_SHIP

        # -----------------------------------------------------------------
        # Step 3: Bullet detection using HSV hue thresholds.  We build a
        # binary mask by combining two hue ranges (0–15 and 160–179) and
        # requiring a minimum saturation and value to exclude very dark or
        # desaturated background pixels.  Morphological opening/closing
        # cleans up small speckles and connects adjacent bullet pixels.
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        bullet_mask_hsv = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in self.bullet_hsv_ranges:
            bullet_mask_hsv |= cv2.inRange(frame_hsv, lower, upper)
        # Morphological operations: opening removes small noise; closing
        # fills small holes within bullets or chains of bullets.  A small
        # kernel is used to preserve fine details of the bullet shapes.
        # PERFORMANCE: Use cached kernel
        bullet_mask_clean = cv2.morphologyEx(bullet_mask_hsv, cv2.MORPH_OPEN, self._morph_kernel)
        bullet_mask_clean = cv2.morphologyEx(bullet_mask_clean, cv2.MORPH_CLOSE, self._morph_kernel)
        bullet_mask_bool = bullet_mask_clean.astype(bool)
        # Remove any pixels previously labelled as ship or background
        # Dilate ship mask to create a buffer zone that excludes bullets near ship edges
        ship_mask_dilated = cv2.dilate(ship_mask.astype(np.uint8), self._morph_kernel, iterations=2).astype(bool)
        bullet_mask_bool &= ~ship_mask_dilated
        bullet_mask_bool &= ~bg_mask
        # Filter out tiny regions that are likely noise.  Use connected
        # components to measure area.
        bullet_mask_uint8 = bullet_mask_bool.astype(np.uint8)
        num_bullets, bullet_labels, bullet_stats, _ = cv2.connectedComponentsWithStats(bullet_mask_uint8, connectivity=8)
        final_bullet_mask = np.zeros((height, width), dtype=bool)
        for label in range(1, num_bullets):
            area = bullet_stats[label, cv2.CC_STAT_AREA]
            if area >= self.bullet_min_area:
                final_bullet_mask[bullet_labels == label] = True

        # Exclude the bottom left corner (HUD score display) from bullet detection
        # The percentages here are empirically chosen based on the typical
        # layout of the game UI.
        hud_bullet_h = int(height * 0.05)
        hud_bullet_w = int(width * 0.50)
        final_bullet_mask[height - hud_bullet_h:, :hud_bullet_w] = 0

        # Assign bullet label
        mask[final_bullet_mask] = MASK_BULLET

        # -----------------------------------------------------------------
        # Step 4: Enemy detection.  Enemies are bright shapes with a
        # darker interior.  We threshold the grayscale image to isolate
        # bright shapes and then analyse their contours with hierarchy
        # information.  A contour is considered an enemy if it has a
        # child (a hole) or if the mean grayscale value inside the
        # contour falls below the brightness threshold.  This allows
        # detection of enemies even when the hole is occluded or
        # diminished by overlapping geometry.
        # PERFORMANCE: Skip this expensive step if not needed
        if not skip_enemy:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # Threshold to obtain bright areas
            _, bright_mask = cv2.threshold(gray, ENEMY_BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
            # Exclude the HUD (bottom left area) from consideration.  The
            # percentages here are empirically chosen based on the typical
            # layout of the game UI.
            hud_h = int(height * 0.08)
            hud_w = int(width * 0.40)
            bright_mask[height - hud_h :, :hud_w] = 0
            # Close small gaps in bright shapes to ensure contours are
            # contiguous
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, self._morph_kernel)
            contours, hierarchy = cv2.findContours(bright_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None and len(contours) > 0:
                hierarchy = hierarchy[0]
                enemy_mask_temp = np.zeros((height, width), dtype=np.uint8)
                
                # PERFORMANCE: Preallocate temp buffer (reuse if same dimensions)
                if (self._reusable_temp_mask is None or 
                    self._cached_height != height or 
                    self._cached_width != width):
                    self._reusable_temp_mask = np.zeros((height, width), dtype=np.uint8)
                    self._cached_height = height
                    self._cached_width = width
                
                for idx, contour in enumerate(contours):
                    # Only consider outer contours
                    is_outer = hierarchy[idx][3] == -1
                    if not is_outer:
                        continue
                    # Determine if this contour represents an enemy
                    has_hole = hierarchy[idx][2] != -1
                    is_enemy = False
                    if has_hole:
                        is_enemy = True
                    else:
                        # If there is no explicit hole, compute the mean
                        # grayscale value inside the contour.  A darker mean
                        # suggests the presence of the peach background within a
                        # mostly white border, indicating an enemy.  To avoid
                        # heavy per‑pixel operations, we fill the contour and
                        # then use the resulting mask to index into the
                        # grayscale image.
                        # PERFORMANCE: Reuse preallocated buffer
                        temp = self._reusable_temp_mask
                        temp.fill(0)  # Clear previous data
                        cv2.drawContours(temp, contours, idx, 255, -1)
                        inside = temp.astype(bool)
                        if inside.any():
                            mean_val = float(gray[inside].mean())
                            if mean_val < ENEMY_BRIGHT_THRESHOLD:
                                is_enemy = True
                    if is_enemy:
                        # Draw the enemy contour into the temporary mask
                        cv2.drawContours(enemy_mask_temp, contours, idx, 1, -1)
                # Update the final mask, prioritising enemy labels.  Only
                # overwrite background or bullet pixels; never overwrite the
                # ship label.
                enemy_pixels = enemy_mask_temp.astype(bool)
                overwrite = enemy_pixels & (mask != MASK_SHIP)
                mask[overwrite] = MASK_ENEMY

        return mask

    # ---------------------------------------------------------------------
    # The remaining methods are unchanged relative to the original
    # implementation.  They provide convenience functions for computing
    # positions and distances.

    def get_positions(self, mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Extract ship, bullet and enemy positions from a mask.

        Args:
            mask: Segmentation mask with values 0–3 as produced by
                `generate_mask`.

        Returns:
            ship_pos: (row, col) position of the ship centre or None if
                the ship is absent.
            bullet_positions: list of (row, col) positions for the
                centres of connected bullet components.
            enemy_positions: list of (row, col) positions for the
                centres of connected enemy components.
        """
        # Ship position (centroid of all ship pixels)
        ship_pixels = np.where(mask == MASK_SHIP)
        ship_pos: Optional[Tuple[int, int]]
        if ship_pixels[0].size > 0:
            ship_y = int(np.mean(ship_pixels[0]))
            ship_x = int(np.mean(ship_pixels[1]))
            ship_pos = (ship_y, ship_x)
        else:
            ship_pos = None

        # Bullet positions (connected components)
        bullet_mask = (mask == MASK_BULLET).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(bullet_mask)
        bullet_positions: List[Tuple[int, int]] = []
        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if ys.size > 0:
                bullet_y = int(np.mean(ys))
                bullet_x = int(np.mean(xs))
                bullet_positions.append((bullet_y, bullet_x))

        # Enemy positions (connected components)
        enemy_mask = (mask == MASK_ENEMY).astype(np.uint8)
        num_labels_enemy, labels_enemy = cv2.connectedComponents(enemy_mask)
        enemy_positions: List[Tuple[int, int]] = []
        for label in range(1, num_labels_enemy):
            ys, xs = np.where(labels_enemy == label)
            if ys.size > 0:
                enemy_y = int(np.mean(ys))
                enemy_x = int(np.mean(xs))
                enemy_positions.append((enemy_y, enemy_x))

        return ship_pos, bullet_positions, enemy_positions

    def compute_nearest_bullet_distance(self, ship_pos: Optional[Tuple[int, int]], bullet_positions: List[Tuple[int, int]], normalize_by: Optional[float] = None) -> Optional[float]:
        """Compute distance to the nearest bullet.

        Args:
            ship_pos: (row, col) position of the ship.
            bullet_positions: list of (row, col) positions of bullets.
            normalize_by: optional value to normalise the distance (e.g.
                frame diagonal).

        Returns:
            The distance to the nearest bullet or None if there are no
            bullets or the ship is missing.
        """
        if ship_pos is None or not bullet_positions:
            return None
        ship_y, ship_x = ship_pos
        min_dist = float('inf')
        for bullet_y, bullet_x in bullet_positions:
            d = np.hypot(ship_y - bullet_y, ship_x - bullet_x)
            if d < min_dist:
                min_dist = d
        if normalize_by is not None:
            min_dist /= normalize_by
        return min_dist

    def compute_nearest_enemy_distance(self, ship_pos: Optional[Tuple[int, int]], enemy_positions: List[Tuple[int, int]], normalize_by: Optional[float] = None) -> Optional[float]:
        """Compute distance to the nearest enemy."""
        if ship_pos is None or not enemy_positions:
            return None
        ship_y, ship_x = ship_pos
        min_dist = float('inf')
        for enemy_y, enemy_x in enemy_positions:
            d = np.hypot(ship_y - enemy_y, ship_x - enemy_x)
            if d < min_dist:
                min_dist = d
        if normalize_by is not None:
            min_dist /= normalize_by
        return min_dist

    def compute_cumulative_risk(self, ship_pos: Optional[Tuple[int, int]], entity_positions: List[Tuple[int, int]], normalize_by: Optional[float] = None) -> float:
        """Compute cumulative risk based on inverse distance to entities."""
        if ship_pos is None or not entity_positions:
            return 0.0
        ship_arr = np.array(ship_pos)
        entities_arr = np.array(entity_positions)
        # Euclidean distances
        dists = np.linalg.norm(entities_arr - ship_arr, axis=1)
        if normalize_by is not None:
            dists = dists / normalize_by
        epsilon = 1e-6
        risk_scores = 1.0 / (dists + epsilon)
        return float(np.sum(risk_scores))


def analyze_background_colors(image_files: List[str], num_samples: int = 50) -> List[Tuple[int, int, int]]:
    """Analyse a subset of images to find dominant background colours.

    This helper samples a subset of images, flattens their pixels and
    counts the most common colours.  Colours that make up more than 1% of
    all sampled pixels are returned as likely background colours.

    Args:
        image_files: list of file paths to images.
        num_samples: number of images to sample for analysis.

    Returns:
        List of RGB colour tuples representing background colours.
    """
    print(f"Analysing background colours from {min(len(image_files), num_samples)} samples…")
    samples = image_files[:]
    if len(samples) > num_samples:
        np.random.shuffle(samples)
        samples = samples[:num_samples]
    global_counter: Counter = Counter()
    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape(-1, 3)
        global_counter.update(map(tuple, pixels))
    total_pixels = sum(global_counter.values())
    background_colors: List[Tuple[int, int, int]] = []
    
    # Safety check: if no pixels were analyzed, return default colors
    if total_pixels == 0:
        print("Warning: No valid images found for background analysis. Using default colors.")
        return list(DEFAULT_BG_COLORS_RGB)
    
    print("Detected background colours:")
    for colour, count in global_counter.most_common(10):
        percentage = (count / total_pixels) * 100
        if percentage > 1.0:
            print(f"RGB: {colour} | {percentage:.2f}% (Background)")
            background_colors.append(colour)
        else:
            print(f"RGB: {colour} | {percentage:.2f}% (Object/Noise)")
    return background_colors


def generate_masks(input_dir: str, output_dir: str, visualize: bool = True, use_morphology: bool = True) -> None:
    """Generate segmentation masks for all images in a directory.

    This function analyses background colours in the input directory,
    constructs a `BulletMaskGenerator` instance and processes each
    image.  Resulting masks and optional visualisations are written to
    the output directory.

    Args:
        input_dir: directory containing image files (PNG/JPG/JPEG).
        output_dir: directory in which to save mask images and
            optional visualisation images.
        visualize: if True, write an RGB visualisation of each mask.
        use_morphology: (kept for backwards compatibility, always uses morphology)
    """
    os.makedirs(output_dir, exist_ok=True)
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files: List[str] = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    # Step 1: analyse colours
    bg_colors_rgb = analyze_background_colors(image_files)
    # Step 2: instantiate generator
    bg_colors_bgr = [np.array(c[::-1], dtype=np.uint8) for c in bg_colors_rgb]
    generator = BulletMaskGenerator(bg_colors_bgr)
    print(f"Processing {len(image_files)} images from {input_dir}…")
    for img_path in tqdm(image_files, desc="Generating masks"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            mask = generator.generate_mask(img)
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            out_mask_path = os.path.join(output_dir, f"{name}_mask{ext}")
            cv2.imwrite(out_mask_path, mask)
            if visualize:
                # Create an RGB visualisation of the mask.  Colour mapping:
                # background → black, bullet → green, ship → red, enemy → blue
                vis = np.zeros_like(img)
                vis[mask == MASK_BULLET] = [0, 255, 0]
                vis[mask == MASK_SHIP]   = [0, 0, 255]
                vis[mask == MASK_ENEMY]  = [255, 0, 0]
                out_vis_path = os.path.join(output_dir, f"{name}_vis{ext}")
                cv2.imwrite(out_vis_path, vis)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    print(f"Done! Masks saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate segmentation masks for bullet‑hell frames.")
    parser.add_argument("--input", type=str, default="game_screenshots", help="Directory containing input images.")
    parser.add_argument("--output", type=str, default="game_screenshots_mask", help="Directory to save output masks.")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualisation output.")
    args = parser.parse_args()
    # Resolve input directory relative to module location if necessary
    if not os.path.exists(args.input):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_input = os.path.join(repo_root, args.input)
        if os.path.exists(alt_input):
            args.input = alt_input
        else:
            print(f"Error: input directory '{args.input}' does not exist.")
            raise SystemExit(1)
    generate_masks(args.input, args.output, visualize=not args.no_vis)
