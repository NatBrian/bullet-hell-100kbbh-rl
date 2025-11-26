"""
Debug script to visualize what the mask generator is detecting for ENEMY, BULLET, and SHIP
"""
import cv2
import numpy as np
from generate_masks import BulletMaskGenerator, MASK_BACKGROUND, MASK_BULLET, MASK_SHIP, MASK_ENEMY
import argparse

def visualize_mask_detection(frame_bgr, reward_params, calibration_frame_bgr=None, bg_threshold=2):
    """
    Visualize what the mask generator detects and compute the reward.
    
    Args:
        frame_bgr: Input frame in BGR format (any size)
        reward_params: Dict with reward calculation parameters
        calibration_frame_bgr: Optional calibration frame for background learning
    """
    # Store original size
    original_shape = frame_bgr.shape
    
    # CRITICAL: Match env.py processing order exactly!
    # 1. Resize to 84x84 FIRST (matching env.py line 285)
    frame_84 = cv2.resize(frame_bgr, (84, 84), interpolation=cv2.INTER_LINEAR)
    
    # 2. Create and optionally calibrate generator
    generator = BulletMaskGenerator(bg_threshold=bg_threshold)
    if calibration_frame_bgr is not None:
        calib_84 = cv2.resize(calibration_frame_bgr, (84, 84), interpolation=cv2.INTER_LINEAR)
        num_colors = generator.calibrate_from_initial_frame(calib_84, tolerance=10)
        print(f"[Debug] Calibrated with {num_colors} colors from calibration frame")
    
    # 3. THEN generate mask from the 84x84 frame (matching env.py behavior)
    mask = generator.generate_mask(frame_84)
    
    # Get positions from the mask
    ship_pos, bullet_positions, enemy_positions = generator.get_positions(mask)

    
    # Compute distances and rewards
    frame_diagonal = np.sqrt(84**2 + 84**2)
    
    # Bullet distance and reward
    bullet_dist = generator.compute_nearest_bullet_distance(
        ship_pos, bullet_positions, normalize_by=frame_diagonal
    )
    
    # Enemy distance and reward
    enemy_dist = generator.compute_nearest_enemy_distance(
        ship_pos, enemy_positions, normalize_by=frame_diagonal
    )
    
    # Compute bullet reward (matching env.py logic)
    bullet_reward_coef = reward_params['bullet_reward_coef']
    bullet_quadratic_coef = reward_params['bullet_quadratic_coef']
    bullet_density_coef = reward_params['bullet_density_coef']
    risk_clip = reward_params['risk_clip']
    
    if bullet_dist is not None:
        bullet_risk = 1.0 / (bullet_dist + 1e-6) + bullet_quadratic_coef / (bullet_dist**2 + 1e-6)
        bullet_risk = min(bullet_risk, risk_clip)
        
        # Add density penalty
        density_penalty = bullet_density_coef * len(bullet_positions)
        
        bullet_reward = -bullet_reward_coef * bullet_risk - density_penalty
    else:
        bullet_risk = 0.0
        density_penalty = 0.0
        bullet_reward = 0.0
    
    # Compute enemy reward (matching env.py logic)
    enemy_reward_coef = reward_params['enemy_reward_coef']
    enemy_quadratic_coef = reward_params['enemy_quadratic_coef']
    
    if enemy_dist is not None:
        enemy_risk = 1.0 / (enemy_dist + 1e-6) + enemy_quadratic_coef / (enemy_dist**2 + 1e-6)
        enemy_risk = min(enemy_risk, risk_clip)
        enemy_reward = -enemy_reward_coef * enemy_risk
    else:
        enemy_risk = 0.0
        enemy_reward = 0.0
    
    # Count pixels
    num_background = np.sum(mask == MASK_BACKGROUND)
    num_bullet = np.sum(mask == MASK_BULLET)
    num_ship = np.sum(mask == MASK_SHIP)
    num_enemy = np.sum(mask == MASK_ENEMY)
    
    # Create three visualization panels: Original, Mask, and Detailed
    vis_mask = np.zeros((84, 84, 3), dtype=np.uint8)
    vis_mask[mask == MASK_BACKGROUND] = [0, 0, 0]      # Black
    vis_mask[mask == MASK_BULLET] = [0, 255, 0]        # Green
    vis_mask[mask == MASK_SHIP] = [0, 0, 255]          # Red  
    vis_mask[mask == MASK_ENEMY] = [255, 0, 0]         # Blue
    
    # Create detailed visualization with annotations
    vis_detailed = vis_mask.copy()
    
    # Draw ship position (white circle)
    if ship_pos is not None:
        cv2.circle(vis_detailed, (ship_pos[1], ship_pos[0]), 3, (255, 255, 255), -1)
    
    # Draw bullet positions and distances (yellow circles, gray lines)
    nearest_bullet_idx = None
    if ship_pos is not None and len(bullet_positions) > 0:
        # Find nearest bullet
        min_dist = float('inf')
        for idx, (bullet_y, bullet_x) in enumerate(bullet_positions):
            dist = np.sqrt((ship_pos[0] - bullet_y)**2 + (ship_pos[1] - bullet_x)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_bullet_idx = idx
    
    for idx, (bullet_y, bullet_x) in enumerate(bullet_positions):
        # Draw bullet
        color = (0, 255, 255) if idx == nearest_bullet_idx else (128, 255, 128)  # Yellow for nearest, light green for others
        thickness = 2 if idx == nearest_bullet_idx else 1
        cv2.circle(vis_detailed, (bullet_x, bullet_y), 2, color, -1)
        
        if ship_pos is not None:
            # Draw line to ship
            line_color = (0, 200, 200) if idx == nearest_bullet_idx else (64, 64, 64)
            cv2.line(vis_detailed, (ship_pos[1], ship_pos[0]), (bullet_x, bullet_y), line_color, 1)
    
    # Draw enemy positions and distances (cyan circles, blue lines)
    nearest_enemy_idx = None
    if ship_pos is not None and len(enemy_positions) > 0:
        # Find nearest enemy
        min_dist = float('inf')
        for idx, (enemy_y, enemy_x) in enumerate(enemy_positions):
            dist = np.sqrt((ship_pos[0] - enemy_y)**2 + (ship_pos[1] - enemy_x)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_enemy_idx = idx
    
    for idx, (enemy_y, enemy_x) in enumerate(enemy_positions):
        # Draw enemy
        color = (255, 255, 0) if idx == nearest_enemy_idx else (255, 128, 128)  # Cyan for nearest, light blue for others
        thickness = 2 if idx == nearest_enemy_idx else 1
        cv2.circle(vis_detailed, (enemy_x, enemy_y), 2, color, -1)
        
        if ship_pos is not None:
            # Draw line to ship (using brighter colors for visibility)
            line_color = (255, 0, 255) if idx == nearest_enemy_idx else (150, 0, 150)  # Magenta for nearest, darker magenta for others
            cv2.line(vis_detailed, (ship_pos[1], ship_pos[0]), (enemy_x, enemy_y), line_color, 1)
    
    # Create comprehensive info panel (increased height to show all metrics)
    info_panel = np.zeros((450, 600, 3), dtype=np.uint8)
    y_offset = 25
    line_height = 22
    
    # Ship info
    cv2.putText(info_panel, "=== SHIP ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_offset += line_height
    cv2.putText(info_panel, f"Position: {ship_pos if ship_pos else 'NOT DETECTED'}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Pixels: {num_ship}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y_offset += line_height + 5
    
    # Bullet info
    cv2.putText(info_panel, "=== BULLETS ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height
    cv2.putText(info_panel, f"Count: {len(bullet_positions)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Pixels: {num_bullet}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_offset += line_height
    if bullet_dist is not None:
        cv2.putText(info_panel, f"Nearest dist (norm): {bullet_dist:.4f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(info_panel, f"Nearest dist: None", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Risk: {bullet_risk:.2f} (clip: {risk_clip})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Density penalty: {density_penalty:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Reward: {bullet_reward:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += line_height + 5
    
    # Enemy info
    cv2.putText(info_panel, "=== ENEMIES ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    y_offset += line_height
    cv2.putText(info_panel, f"Count: {len(enemy_positions)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Pixels: {num_enemy}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y_offset += line_height
    if enemy_dist is not None:
        cv2.putText(info_panel, f"Nearest dist (norm): {enemy_dist:.4f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(info_panel, f"Nearest dist: None", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Risk: {enemy_risk:.2f} (clip: {risk_clip})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    y_offset += line_height
    cv2.putText(info_panel, f"Reward: {enemy_reward:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += line_height + 5
    
    # Total and metadata
    total_reward = bullet_reward + enemy_reward
    cv2.putText(info_panel, f"TOTAL REWARD: {total_reward:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += line_height + 10
    cv2.putText(info_panel, f"Original: {original_shape[1]}x{original_shape[0]} -> 84x84", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Resize for display
    frame_large = cv2.resize(frame_84, (400, 400), interpolation=cv2.INTER_NEAREST)
    vis_mask_large = cv2.resize(vis_mask, (400, 400), interpolation=cv2.INTER_NEAREST)
    vis_detailed_large = cv2.resize(vis_detailed, (400, 400), interpolation=cv2.INTER_NEAREST)
    
    # Combine horizontally: Original | Mask | Detailed
    top_row = np.hstack([frame_large, vis_mask_large, vis_detailed_large])
    
    # Add info panel below (adjusted for larger panel)
    bottom_row = np.zeros((450, 1200, 3), dtype=np.uint8)
    bottom_row[:, 300:900] = info_panel
    
    display = np.vstack([top_row, bottom_row])
    
    # Print detailed console output
    print(f"\n{'='*60}")
    print(f"{'MASK DETECTION ANALYSIS':^60}")
    print(f"{'='*60}")
    print(f"\n[SHIP]")
    print(f"  Position: {ship_pos if ship_pos else 'NOT DETECTED'}")
    print(f"  Pixel count: {num_ship}")
    
    print(f"\n[BULLETS]")
    print(f"  Count: {len(bullet_positions)}")
    print(f"  Pixel count: {num_bullet}")
    print(f"  Positions: {bullet_positions if len(bullet_positions) <= 5 else f'{bullet_positions[:5]}... ({len(bullet_positions)} total)'}")
    print(f"  Nearest distance (normalized): {bullet_dist if bullet_dist else 'None'}")
    print(f"  Risk: {bullet_risk:.4f} (clip: {risk_clip})")
    print(f"  Density penalty: {density_penalty:.4f}")
    print(f"  Reward: {bullet_reward:.4f}")
    
    print(f"\n[ENEMIES]")
    print(f"  Count: {len(enemy_positions)}")
    print(f"  Pixel count: {num_enemy}")
    print(f"  Positions: {enemy_positions if len(enemy_positions) <= 5 else f'{enemy_positions[:5]}... ({len(enemy_positions)} total)'}")
    print(f"  Nearest distance (normalized): {enemy_dist if enemy_dist else 'None'}")
    print(f"  Risk: {enemy_risk:.4f} (clip: {risk_clip})")
    print(f"  Reward: {enemy_reward:.4f}")
    
    print(f"\n[TOTAL]")
    print(f"  Combined reward: {total_reward:.4f}")
    print(f"  Background pixels: {num_background}")
    
    print(f"\n[REWARD PARAMETERS]")
    print(f"  bullet_reward_coef: {bullet_reward_coef}")
    print(f"  bullet_quadratic_coef: {bullet_quadratic_coef}")
    print(f"  bullet_density_coef: {bullet_density_coef}")
    print(f"  enemy_reward_coef: {enemy_reward_coef}")
    print(f"  enemy_quadratic_coef: {enemy_quadratic_coef}")
    print(f"  risk_clip: {risk_clip}")
    print(f"{'='*60}\n")
    
    return display, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug ENEMY, BULLET, and SHIP detection")
    parser.add_argument("--image", type=str, required=True, help="Path to image file to analyze")
    parser.add_argument("--calibration-frame", type=str, default=None, help="Path to calibration frame (initial game state with ship+background only)")
    parser.add_argument("--bg-threshold", type=int, default=2, help="Background color matching threshold (default: 2)")
    
    # Reward parameters (matching env.py defaults)
    parser.add_argument("--bullet-reward-coef", type=float, default=1.0, help="Bullet reward coefficient")
    parser.add_argument("--bullet-quadratic-coef", type=float, default=0.0, help="Bullet quadratic coefficient")
    parser.add_argument("--bullet-density-coef", type=float, default=0.0, help="Bullet density coefficient")
    parser.add_argument("--enemy-reward-coef", type=float, default=1.0, help="Enemy reward coefficient")
    parser.add_argument("--enemy-quadratic-coef", type=float, default=0.0, help="Enemy quadratic coefficient")
    parser.add_argument("--risk-clip", type=float, default=4.0, help="Risk clipping value")
    
    args = parser.parse_args()
    
    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not load image from {args.image}")
        exit(1)
    
    print(f"Loaded image: {args.image}")
    print(f"Original size: {frame.shape}")
    
    # Prepare reward parameters
    reward_params = {
        'bullet_reward_coef': args.bullet_reward_coef,
        'bullet_quadratic_coef': args.bullet_quadratic_coef,
        'bullet_density_coef': args.bullet_density_coef,
        'enemy_reward_coef': args.enemy_reward_coef,
        'enemy_quadratic_coef': args.enemy_quadratic_coef,
        'risk_clip': args.risk_clip,
    }
    
    # Calibrate if calibration frame is provided
    calibration_frame = None
    if args.calibration_frame:
        calibration_frame = cv2.imread(args.calibration_frame)
        if calibration_frame is None:
            print(f"Warning: Could not load calibration frame from {args.calibration_frame}")
        else:
            print(f"Loading calibration frame: {args.calibration_frame}")
    
    # Analyze
    display, mask = visualize_mask_detection(frame, reward_params, calibration_frame, bg_threshold=args.bg_threshold)
    
    # Show
    cv2.imshow("Debug: ENEMY | BULLET | SHIP Detection", display)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save
    output_path = args.image.replace('.png', '_debug.png')
    cv2.imwrite(output_path, display)
    print(f"Saved debug visualization to: {output_path}")
