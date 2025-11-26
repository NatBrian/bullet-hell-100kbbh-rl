import gymnasium as gym
import numpy as np
import os
import cv2
import time
import win32gui
import win32ui
import win32con
import win32api
import pydirectinput
import mss
from collections import deque

# Disable fail-safe to prevent mouse corner detection from stopping training
pydirectinput.FAILSAFE = False
from gymnasium import spaces
from generate_masks import BulletMaskGenerator

# Try importing dxcam, but don't fail if it's not available (fallback to mss)
try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False

class BulletHellEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "debug", "both"], "render_fps": 30}

    def __init__(
        self,
        # ============================================================================
        # SHARED PARAMETERS (Used by both baseline and safety strategies)
        # ============================================================================
        window_title="100KBBH",
        game_path=None,
        render_mode=None,
        frame_skip=1,  # Changed from 4 to 1 for faster iteration
        stack_size=4,
        alive_thresh=150.0,  # LUMINANCE THRESHOLD FOR ALIVE
        dead_thresh=130.0,  # LUMINANCE THRESHOLD FOR DEAD
        dead_streak=3,
        action_duration=0.016,  # REACTION TIME AI PRESS KEY IN SECONDS (1 frame at 60 FPS)
        save_screenshots=0,  # Interval in ms to save screenshots (0 to disable)
        force_mss=False,  # Force usage of MSS for screen capture
        bg_threshold=2,  # Background color matching threshold
        # Reward strategy selector
        reward_strategy="baseline",  # 'baseline' (old) or 'safety' (new)
        # Distance-based reward shaping (both strategies)
        use_bullet_distance_reward=False,  # Enable bullet distance reward shaping
        bullet_reward_coef=0.01,  # Coefficient for bullet distance reward
        bullet_quadratic_coef=0.10,  # Quadratic coefficient for bullet distance
        use_enemy_distance_reward=False,  # Enable enemy distance reward shaping
        enemy_reward_coef=0.02,  # Coefficient for enemy distance reward
        enemy_quadratic_coef=0.10,  # Quadratic coefficient for enemy distance
        # Basic reward values (both strategies)
        alive_reward=4.0,  # Reward per frame survived (when bright)
        death_penalty=-20.0,  # Penalty on death
        risk_clip=10.0,  # Clip for distance-based risk
        # ============================================================================
        # BASELINE STRATEGY ONLY
        # ============================================================================
        bullet_density_coef=0.01,  # Coefficient for cumulative bullet risk (density penalty)
        # ============================================================================
        # SAFETY STRATEGY ONLY
        # ============================================================================
        # Dodging differentiation
        dodge_skill_threshold=2.0,  # Minimum cumulative risk to trigger dodge bonus
        dodge_skill_multiplier=0.15,  # Bonus per unit of cumulative risk when dodging
        graze_requires_movement=True,  # Require active movement to earn graze bonus
        graze_bonus_multiplier=0.15,  # Graze bonus as fraction of alive_reward
        # Enemy anticipation
        enemy_danger_multiplier=3.0,  # How much more dangerous enemies are vs bullets
        enemy_escape_multiplier=10.0,  # Enemy escape bonus multiplier (vs 5.0 for bullets)
    ):
        super().__init__()
        self.window_title = window_title
        self.game_path = game_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.alive_thresh = alive_thresh
        self.dead_thresh = dead_thresh
        self.dead_streak = dead_streak
        self.action_duration = action_duration
        self.save_screenshots = save_screenshots
        self.last_screenshot_time = 0
        self.reward_strategy = reward_strategy
        self.use_bullet_distance_reward = use_bullet_distance_reward
        self.bullet_reward_coef = bullet_reward_coef
        self.use_enemy_distance_reward = use_enemy_distance_reward
        self.enemy_reward_coef = enemy_reward_coef
        self.bullet_quadratic_coef = bullet_quadratic_coef
        self.bullet_density_coef = bullet_density_coef
        self.enemy_quadratic_coef = enemy_quadratic_coef
        self.alive_reward = alive_reward
        self.death_penalty = death_penalty
        self.risk_clip = risk_clip
        self.force_mss = force_mss
        self.bg_threshold = bg_threshold
        self.last_bullet_dist = None
        self.last_bullet_risk = None
        self.last_enemy_dist = None
        self.dodge_skill_threshold = dodge_skill_threshold
        self.dodge_skill_multiplier = dodge_skill_multiplier
        self.graze_requires_movement = graze_requires_movement
        self.graze_bonus_multiplier = graze_bonus_multiplier
        self.enemy_danger_multiplier = enemy_danger_multiplier
        self.enemy_escape_multiplier = enemy_escape_multiplier
        
        # Initialize bullet mask generator if needed
        if self.use_bullet_distance_reward or self.use_enemy_distance_reward or self.render_mode in ["debug", "both"]:
            self.mask_generator = BulletMaskGenerator(bg_threshold=self.bg_threshold)
        
        if self.save_screenshots > 0:
            os.makedirs("game_screenshots", exist_ok=True)

        # Action Space: 9 discrete actions
        # 0: Idle, 1: W, 2: S, 3: A, 4: D, 5: WA, 6: WD, 7: SA, 8: SD
        self.action_space = spaces.Discrete(9)
        self.key_map = {
            0: [],
            1: ['w'],
            2: ['s'],
            3: ['a'],
            4: ['d'],
            5: ['w', 'a'],
            6: ['w', 'd'],
            7: ['s', 'a'],
            8: ['s', 'd'],
        }

        # Observation Space: Stacked grayscale frames (C, H, W)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(stack_size, 84, 84), dtype=np.uint8
        )

        self.window_rect = None
        self.hwnd = None
        self.camera = None
        self.mss_sct = None
        self.frame_stack = deque(maxlen=stack_size)
        self.luminance_history = deque(maxlen=dead_streak)
        
        self.steps = 0
        self.episode_reward = 0.0
        self.last_ship_pos = None  # Track last known ship position
        self.last_action = None  # Track last action for reward shaping
        
        # Initialize capture method
        self._init_capture()
        
        # Input State Tracking
        self.pressed_keys = set()

    def _init_capture(self):
        """Initialize dxcam or mss."""
        self.use_dxcam = False
        
        if DXCAM_AVAILABLE and not self.force_mss:
            # Try multiple dxcam initialization strategies for RTX GPUs
            init_attempts = [
                {"device_idx": 0, "output_idx": 0, "output_color": "BGR"},  # Primary GPU, primary monitor
                {"device_idx": 0, "output_color": "BGR"},  # Primary GPU, auto-detect monitor
                {"output_color": "BGR"},  # Auto-detect everything
            ]
            
            for idx, kwargs in enumerate(init_attempts):
                try:
                    camera = dxcam.create(**kwargs)
                    if camera is not None:
                        self.camera = camera
                        self.use_dxcam = True
                        print(f"Using dxcam for screen capture (attempt {idx+1} succeeded).")
                        break
                    else:
                        print(f"dxcam.create() attempt {idx+1} returned None.")
                except Exception as e:
                    print(f"dxcam initialization attempt {idx+1} failed: {e}")
                    # Clean up partially initialized camera object if it exists
                    if hasattr(self, 'camera') and self.camera is not None:
                        # Just set to None, let GC handle it (suppressing errors in __del__ is hard)
                        self.camera = None
            
            if not self.use_dxcam:
                print("All dxcam initialization attempts failed. Falling back to mss.")
        elif self.force_mss:
            print("Forcing mss for screen capture (dxcam disabled by config).")
        
        if not self.use_dxcam:
            self.mss_sct = mss.mss()
            print("Using mss for screen capture.")
    
    def _get_window_rect(self):
        """Finds the game window and returns its bounding box."""
        def find_window_handle():
            hwnd = win32gui.FindWindow(None, self.window_title)
            if not hwnd:
                # Try partial match
                def callback(h, ctx):
                    if win32gui.IsWindowVisible(h):
                        title = win32gui.GetWindowText(h)
                        if self.window_title in title:
                            ctx.append(h)
                found = []
                win32gui.EnumWindows(callback, found)
                if found:
                    hwnd = found[0]
            return hwnd

        hwnd = find_window_handle()
        self.hwnd = hwnd
        
        if not hwnd and self.game_path:
            print(f"Window not found. Launching game from: {self.game_path}")
            import subprocess
            import os
            if os.path.exists(self.game_path):
                abs_game_path = os.path.abspath(self.game_path)
                subprocess.Popen(abs_game_path, cwd=os.path.dirname(abs_game_path))
                # Wait for window to appear
                for _ in range(10):
                    time.sleep(1)
                    hwnd = find_window_handle()
                    if hwnd:
                        break
            else:
                print(f"Game path does not exist: {self.game_path}")

        if not hwnd:
            raise RuntimeError(f"Window '{self.window_title}' not found!")

        # Ensure window is not minimized and is in foreground
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.2)
        
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass # Might fail if another window is actively focused, but we try.

        # Move window to top-left corner to ensure visibility and space for debug windows
        # Get current size to preserve it
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        win32gui.MoveWindow(hwnd, 0, 0, w, h, True)

        rect = win32gui.GetWindowRect(hwnd)
        # rect is (left, top, right, bottom)
        # Adjust for borders if needed, but usually raw rect is okay for visual games
        # We might want to crop the title bar. 
        # Standard Windows title bar is ~30px. Let's crop top slightly.
        client_rect = win32gui.GetClientRect(hwnd)
        client_w = client_rect[2] - client_rect[0]
        client_h = client_rect[3] - client_rect[1]
        
        # Map client to screen
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        left, top = pt
        right = left + client_w
        bottom = top + client_h
        
        return (left, top, right, bottom)

    def _capture_frame(self, return_color=False):
        """
        Captures a frame from the window.
        
        Args:
            return_color: If True, returns both grayscale and downscaled color (BGR) frames.
                         If False, returns only the grayscale frame.
        
        Returns:
            If return_color=False: grayscale frame (84, 84)
            If return_color=True: tuple of (grayscale_frame, color_frame_bgr) where color_frame_bgr is 84x84
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.window_rect is None:
                    self.window_rect = self._get_window_rect()

                left, top, right, bottom = self.window_rect
                width = right - left
                height = bottom - top
                
                if width <= 0 or height <= 0:
                    print(f"Invalid window dimensions: {width}x{height}. Retrying...")
                    self.window_rect = self._get_window_rect()
                    continue

                region = (left, top, right, bottom)
                
                frame = None
                if self.use_dxcam:
                    # Retry dxcam a few times if it returns None (no new frame)
                    for _ in range(3):
                        frame = self.camera.grab(region=region)
                        if frame is not None:
                            break
                        time.sleep(0.001)
                
                if frame is None: # Fallback to mss if dxcam failed or is not used
                    if self.mss_sct is None:
                        self.mss_sct = mss.mss()
                    monitor = {"top": top, "left": left, "width": width, "height": height}
                    img = self.mss_sct.grab(monitor)
                    frame = np.array(img)
                    if frame.size == 0:
                        raise RuntimeError("Captured empty frame from MSS")
                    frame = frame[:, :, :3] # BGRA -> BGR
                
                # Save screenshot if enabled and interval passed
                if self.save_screenshots > 0:
                    current_time = time.time() * 1000 # Convert to ms
                    if current_time - self.last_screenshot_time >= self.save_screenshots:
                        filename = f"game_screenshots/{int(current_time)}.png"
                        # frame is currently BGR (from dxcam or mss converted)
                        # cv2.imwrite expects BGR
                        cv2.imwrite(filename, frame)
                        self.last_screenshot_time = current_time

                # Downscale once and derive grayscale from the small frame
                small_frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_LINEAR)
                gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                if return_color:
                    return gray_frame, small_frame
                else:
                    return gray_frame
            
            except Exception as e:
                print(f"Capture failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(0.1)
                # Force refresh window rect
                try:
                    self.window_rect = self._get_window_rect()
                except:
                    pass
        
        raise RuntimeError("Failed to capture frame after multiple attempts.")

    def _get_obs(self):
        """Returns the stacked observation."""
        return np.array(self.frame_stack)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.episode_reward = 0.0
        self.luminance_history.clear()
        self.last_bullet_dist = None
        self.last_bullet_risk = None
        self.last_enemy_dist = None
        
        # 1. Focus window
        try:
            self.window_rect = self._get_window_rect()
        except RuntimeError as e:
            print(e)
            # Wait and retry once
            time.sleep(2)
            self.window_rect = self._get_window_rect()

        # 2. Release all keys and Press SPACE to restart
        self._release_all_keys()
        if self._is_focused():
            pydirectinput.press('space')
        
        # 3. Wait for brightness (alive)
        # We poll until mean luminance > alive_thresh
        max_retries = 50
        for _ in range(max_retries):
            frame = self._capture_frame()
            lum = np.mean(frame)
            if lum > self.alive_thresh:
                break
            time.sleep(0.1)
            if self._is_focused():
                pydirectinput.press('space') # Keep tapping space if stuck on death screen

        # 4. Calibrate mask generator (if using mask-based rewards)
        # IMPORTANT: Capture AFTER game is fully ALIVE, not during PAUSE/DEAD/END
        # Wait a small amount to ensure transition is complete and game is stable
        use_mask = self.use_bullet_distance_reward or self.use_enemy_distance_reward
        if use_mask and hasattr(self, 'mask_generator'):
            # Small delay to ensure game has fully transitioned to ALIVE state
            # and any fade-in animations are complete
            time.sleep(0.2)
            
            # Capture and verify it's still in ALIVE state (bright)
            gray_check, calibration_frame = self._capture_frame(return_color=True)
            lum_check = np.mean(gray_check)
            
            if lum_check > self.alive_thresh:
                # Game is confirmed ALIVE, safe to calibrate
                self.mask_generator.calibrate_from_initial_frame(calibration_frame, tolerance=10)
            else:
                # Luminance dropped - game went back to PAUSE/DEAD
                # Skip calibration this time, will retry next episode
                print(f"[Warning] Skipped calibration: luminance {lum_check:.1f} < {self.alive_thresh}")
        
        # 5. Fill stack
        if use_mask:
            frame_gray, frame_color = self._capture_frame(return_color=True)
            mask = self.mask_generator.generate_mask(frame_color)
            obs_frame = mask
        else:
            frame_gray = self._capture_frame()
            obs_frame = frame_gray

        for _ in range(self.stack_size):
            self.frame_stack.append(obs_frame)
        
        return self._get_obs(), {}

    def _is_focused(self):
        """Checks if the game window is currently focused."""
        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            if not fg_hwnd:
                return False
                
            # 1. Fast path: Handle matches known window
            if hasattr(self, 'hwnd') and self.hwnd and self.hwnd == fg_hwnd:
                return True
            
            # 2. Check if foreground window is actually our game (e.g. after restart)
            # This handles stale self.hwnd
            if self.window_title:
                title = win32gui.GetWindowText(fg_hwnd)
                if self.window_title in title:
                    # It IS the game, but handle changed.
                    # Update handle and force rect refresh to ensure we track it correctly
                    self.hwnd = fg_hwnd
                    self.window_rect = None 
                    return True
            
            return False
        except Exception:
            return False

    def step(self, action, q_values=None):
        """Execute action with proper frame skipping.
        
        Frame skip means:
        1. Execute the same action for N consecutive frames
        2. Accumulate rewards over all N frames
        3. Only add the LAST frame to the observation stack
        4. Terminate as soon as death is detected
        """
        total_reward = 0.0
        step_bullet_reward = 0.0
        step_enemy_reward = 0.0
        step_safety_factor = 1.0  # Default to safe
        step_total_risk = 0.0
        step_graze_bonus = 0.0
        step_dodge_skill_bonus = 0.0
        step_bullet_escape_bonus = 0.0
        step_enemy_escape_bonus = 0.0
        terminated = False
        last_frame = None  # what goes into the stack
        last_render_frame = None  # what we show in render mode (original grayscale)
        last_lum = 0.0
        
        # Execute action for frame_skip frames
        for frame_idx in range(self.frame_skip):
            # 1. Perform Action (Stateful)
            if self._is_focused():
                target_keys = set(self.key_map[action])
                if action != self.last_action:
                    # Release keys that are no longer needed
                    for k in self.pressed_keys - target_keys:
                        pydirectinput.keyUp(k)
                    
                    # Press new keys
                    for k in target_keys - self.pressed_keys:
                        pydirectinput.keyDown(k)
                    
                    self.pressed_keys = target_keys
            else:
                # Lost focus - release everything to be safe
                if self.pressed_keys:
                    self._release_all_keys()
            
            # Only sleep if we are rendering for human to see, otherwise run as fast as possible
            if self.render_mode == "human":
                time.sleep(self.action_duration)  # Idle

            # 2. Capture & Process
            use_mask = self.use_bullet_distance_reward or self.use_enemy_distance_reward or self.render_mode in ["debug", "both"]
            if use_mask:
                gray_frame, color_frame = self._capture_frame(return_color=True)
                mask = self.mask_generator.generate_mask(color_frame)
                obs_frame = mask if (self.use_bullet_distance_reward or self.use_enemy_distance_reward) else gray_frame
            else:
                gray_frame = self._capture_frame()
                mask = None
                obs_frame = gray_frame
            
            last_frame = obs_frame
            last_render_frame = gray_frame
            
            # 3. Check Death
            lum = np.mean(gray_frame)
            last_lum = lum
            self.luminance_history.append(lum)
            
            # If all recent frames are dark, we are dead
            if len(self.luminance_history) == self.dead_streak:
                if all(l < self.dead_thresh for l in self.luminance_history):
                    terminated = True
            
            # 4. Compute Reward for this frame
            if terminated:
                frame_reward = self.death_penalty
            else:
                # Base Alive Check
                is_alive = lum >= self.dead_thresh
                
                if not is_alive:
                    frame_reward = 0.0
                else:
                    if self.reward_strategy == "baseline":
                        # --- BASELINE STRATEGY (Old) ---
                        # Fixed alive reward + penalties for danger
                        frame_reward = self.alive_reward
                        if use_mask and mask is not None:
                            dist_reward, b_rew, e_rew, s_factor, t_risk, g_bonus, d_bonus, b_escape, e_escape = self._compute_rewards_baseline(mask)
                            frame_reward += dist_reward
                            step_bullet_reward += b_rew
                            step_enemy_reward += e_rew
                            step_safety_factor = s_factor  # Will be 1.0 for baseline
                            step_total_risk = t_risk  # Will be 0.0 for baseline
                            step_graze_bonus = g_bonus  # Will be 0.0 for baseline
                            step_dodge_skill_bonus = d_bonus  # Will be 0.0 for baseline
                            step_bullet_escape_bonus = b_escape  # Will be 0.0 for baseline
                            step_enemy_escape_bonus = e_escape  # Will be 0.0 for baseline
                            
                    elif self.reward_strategy == "safety":
                        # --- SAFETY STRATEGY (New) ---
                        # Alive reward is conditional on safety (scaled by risk)
                        # No direct penalties, only safety scaling + escape bonus
                        if use_mask and mask is not None:
                            frame_reward, b_rew, e_rew, s_factor, t_risk, g_bonus, d_bonus, b_escape, e_escape = self._compute_rewards_safety(mask)
                            step_bullet_reward += b_rew
                            step_enemy_reward += e_rew
                            # For safety, track the LAST (most recent) frame's metrics
                            step_safety_factor = s_factor
                            step_total_risk = t_risk
                            step_graze_bonus += g_bonus
                            step_dodge_skill_bonus += d_bonus
                            step_bullet_escape_bonus += b_escape
                            step_enemy_escape_bonus += e_escape
                        else:
                            # Fallback if mask fails but we are alive
                            frame_reward = self.alive_reward
            
            total_reward += frame_reward
            
            # Early termination on death
            if terminated:
                break
        
        # Only add the LAST frame to the stack (standard DQN frame skip)
        self.frame_stack.append(last_frame)
        
        self.episode_reward += total_reward
        self.steps += 1
        
        # Track last action for potential reward shaping and key reuse
        self.last_action = action
        
        # 5. Info
        info = {
            "luminance": last_lum,
            "episode_reward": self.episode_reward,
            "step_reward": total_reward,
           "bullet_distance_enabled": self.use_bullet_distance_reward,
            "enemy_distance_enabled": self.use_enemy_distance_reward,
            "frames_executed": frame_idx + 1,  # How many frames were actually executed
            "bullet_distance_reward": step_bullet_reward,
            "enemy_distance_reward": step_enemy_reward,
            "q_values": q_values,
            "reward_strategy": self.reward_strategy,
            # Safety strategy metrics
            "safety_factor": step_safety_factor,
            "total_risk": step_total_risk,
            "graze_bonus": step_graze_bonus,
            "dodge_skill_bonus": step_dodge_skill_bonus,
            "bullet_escape_bonus": step_bullet_escape_bonus,
            "enemy_escape_bonus": step_enemy_escape_bonus,
        }
        
        # 6. Render
        if self.render_mode == "human":
            self._render_frame(last_render_frame, info)
        elif self.render_mode == "debug":
            # For debug mode, we need the mask and color frame
            if mask is not None and 'color_frame' in locals():
                self._render_debug_frame(mask, color_frame, info)
            else:
                # Fallback if mask wasn't generated for some reason
                self._render_frame(last_render_frame, info)
        elif self.render_mode == "both":
            self._render_frame(last_render_frame, info)
            if mask is not None and 'color_frame' in locals():
                self._render_debug_frame(mask, color_frame, info)

        return self._get_obs(), total_reward, terminated, False, info
    
    def _compute_rewards_baseline(self, mask):
        """
        Baseline Reward Strategy (Original):
        - Fixed positive reward for being alive.
        - Negative penalties for being close to bullets/enemies.
        - Negative penalties for bullet density.
        
        Returns:
            tuple: (total_reward, bullet_part, enemy_part, safety_factor, total_risk, 
                   graze_bonus, dodge_skill_bonus, bullet_escape_bonus, enemy_escape_bonus)
                   Safety metrics are stub values for baseline: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        """
        try:
            # Extract positions
            ship_pos, bullet_positions, enemy_positions = self.mask_generator.get_positions(mask)
            
            # Track last known ship position for fallback
            if ship_pos is not None:
                self.last_ship_pos = ship_pos
            elif self.last_ship_pos is not None:
                # Ship detection failed, use last known position with penalty
                ship_pos = self.last_ship_pos
                return -0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                # Can't find ship at all - heavily penalize
                return -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            bullet_part = 0.0
            enemy_part = 0.0
            frame_diagonal = np.sqrt(84**2 + 84**2)

            # Bullet Reward (quadratic distance shaping)
            if self.use_bullet_distance_reward:
                # 1. Nearest Bullet (Immediate Threat)
                dist = self.mask_generator.compute_nearest_bullet_distance(
                    ship_pos, bullet_positions, normalize_by=frame_diagonal
                )
                if dist is not None:
                    # Quadratic shaping
                    risk = 1.0 / (dist + 1e-6) + self.bullet_quadratic_coef / (dist**2 + 1e-6)
                    risk = min(risk, self.risk_clip)
                    closing_pen = 0.0
                    escape_bonus = 0.0
                    if self.last_bullet_dist is not None:
                        delta = self.last_bullet_dist - dist  # positive if getting closer
                        if delta > 0:
                            closing_pen = min(delta * self.risk_clip, self.risk_clip)
                        elif delta < 0:
                            escape_bonus = min(-delta * self.risk_clip, self.risk_clip)
                    bullet_part -= self.bullet_reward_coef * (risk + closing_pen)
                    bullet_part += self.bullet_reward_coef * escape_bonus
                    self.last_bullet_dist = dist
                else:
                    self.last_bullet_dist = None

                # 2. Bullet Density (Cumulative Risk)
                if self.bullet_density_coef > 0:
                    density_risk = self.mask_generator.compute_cumulative_risk(
                        ship_pos, bullet_positions, normalize_by=frame_diagonal
                    )
                    # Clip density risk to prevent explosion when inside a bullet (though game over usually happens first)
                    # We allow a higher clip for density since it sums multiple bullets
                    density_risk = min(density_risk, self.risk_clip * 5.0) 
                    bullet_part -= self.bullet_density_coef * density_risk

            # Enemy Reward (quadratic distance shaping)
            if self.use_enemy_distance_reward:
                dist = self.mask_generator.compute_nearest_enemy_distance(
                    ship_pos, enemy_positions, normalize_by=frame_diagonal
                )
                if dist is not None:
                    # Quadratic shaping
                    risk = 1.0 / (dist + 1e-6) + self.enemy_quadratic_coef / (dist**2 + 1e-6)
                    risk = min(risk, self.risk_clip)
                    closing_pen = 0.0
                    escape_bonus = 0.0
                    if self.last_enemy_dist is not None:
                        delta = self.last_enemy_dist - dist
                        if delta > 0:
                            closing_pen = min(delta * self.risk_clip, self.risk_clip)
                        elif delta < 0:
                            escape_bonus = min(-delta * self.risk_clip, self.risk_clip)
                    enemy_part -= self.enemy_reward_coef * (risk + closing_pen)
                    enemy_part += self.enemy_reward_coef * escape_bonus
                    self.last_enemy_dist = dist
                else:
                    self.last_enemy_dist = None

            total_reward = bullet_part + enemy_part
            # Return with stub safety metrics: safety=1.0, risk=0, bonuses=0
            return total_reward, bullet_part, enemy_part, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        except Exception as e:
            print(f"Reward computation failed: {e}")
            return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _compute_rewards_safety(self, mask):
        """
        Safety Reward Strategy (Enhanced):
        - Alive reward is NOT guaranteed. It is scaled by safety.
        - Safety Factor = 1.0 / (1.0 + Risk)
        - If Risk is high (close to bullet), Safety Factor -> 0, so Reward -> 0.
        - If Risk is low (far from bullet), Safety Factor -> 1, so Reward -> Alive Reward.
        - Explicit 'Escape Bonus' for moving away from danger.
        - NO direct subtraction penalties (prevents suicide loophole).
        - Min Safety Floor: Ensures survival is always rewarded (distinguishes grazing from death).
        - Enemy Scaling: Weights enemy risk relative to bullet risk using coefficients.
        - Graze Bonus: Bonus for surviving high-risk situations (optionally requires movement).
        - Dodge Skill Bonus: Rewards surviving in high bullet density (multiple bullets nearby).
        - Enhanced Enemy Danger: Enemies are 3x more dangerous than bullets (configurable).
        - Stronger Enemy Escape: 10x multiplier for fleeing enemies vs 5x for bullets.
        - Movement-Required Graze: Only award graze bonus if agent is actively evading.
        
        Returns:
            tuple: (total_reward, bullet_part, enemy_part, safety_factor, total_risk, 
                   graze_bonus, dodge_skill_bonus, bullet_escape_bonus, enemy_escape_bonus)
        """
        try:
            # Extract positions
            ship_pos, bullet_positions, enemy_positions = self.mask_generator.get_positions(mask)
            
            # Track last known ship position
            if ship_pos is not None:
                self.last_ship_pos = ship_pos
            elif self.last_ship_pos is not None:
                ship_pos = self.last_ship_pos
                # If ship is lost, we assume high risk/low safety, but don't penalize heavily yet
                # Just give 0 reward
                return 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0
            
            frame_diagonal = np.sqrt(84**2 + 84**2)
            
            # Initialize Risks and Bonuses
            bullet_risk = 0.0
            enemy_risk = 0.0
            
            bullet_escape_bonus = 0.0
            enemy_escape_bonus = 0.0
            dodge_skill_bonus = 0.0  # NEW: Reward for surviving high bullet density

            # --- 1. Bullet Risk Analysis ---
            if self.use_bullet_distance_reward:
                # Calculate Cumulative Risk (Sum of 1/dist for ALL bullets)
                # This is much smoother than "nearest bullet" tracking
                current_bullet_risk = self.mask_generator.compute_cumulative_risk(
                    ship_pos, bullet_positions, normalize_by=frame_diagonal
                )
                
                # Also get nearest distance for max risk clipping
                nearest_dist = self.mask_generator.compute_nearest_bullet_distance(
                    ship_pos, bullet_positions, normalize_by=frame_diagonal
                )
                
                if nearest_dist is not None:
                    # Base Risk for Safety Factor (dominated by nearest bullet)
                    # We still use the quadratic formula for the "Safety Factor" itself
                    # to ensure it drops sharply when VERY close.
                    raw_risk = 1.0 / (nearest_dist + 1e-6) + self.bullet_quadratic_coef / (nearest_dist**2 + 1e-6)
                    bullet_risk = min(raw_risk, self.risk_clip)
                    
                    # Escape Bonus: Based on CUMULATIVE risk delta
                    # This handles "threading the needle" better. If you move closer to one but 
                    # further from 5 others, your cumulative risk might drop (good).
                    if self.last_bullet_risk is not None:
                        # Positive delta = Risk increased (Bad)
                        # Negative delta = Risk decreased (Good)
                        risk_delta = current_bullet_risk - self.last_bullet_risk
                        
                        if risk_delta < 0: # Risk Decreasing (Moving to safety)
                            # Scale bonus by how dangerous the situation is.
                            # Escaping high risk is worth more than escaping low risk.
                            urgency = min(current_bullet_risk, self.risk_clip) / self.risk_clip
                            # Multiplier 50.0 (increased from 5.0) to compensate for smaller magnitude
                            # of risk delta (1/dist) compared to raw distance delta.
                            bullet_escape_bonus = min(-risk_delta * urgency * 50.0, 1.0)
                    
                    self.last_bullet_risk = current_bullet_risk
                    self.last_bullet_dist = nearest_dist # Keep tracking for debug/legacy
                    
                    # Dodge Skill Bonus (Threading the Needle)
                    # Reward surviving in high density areas
                    if len(bullet_positions) > 1:
                        # If cumulative risk is high (dense)
                        if current_bullet_risk > self.dodge_skill_threshold:
                            # And we are NOT making it significantly worse (delta <= small epsilon)
                            # We don't strictly require improvement, just stability in chaos.
                            if self.last_bullet_risk is not None:
                                delta = current_bullet_risk - self.last_bullet_risk
                                if delta <= 0.1: # Allow slight fluctuations
                                    dodge_skill_bonus = min(
                                        (current_bullet_risk - self.dodge_skill_threshold) * self.dodge_skill_multiplier,
                                        2.0
                                    )
                else:
                    self.last_bullet_dist = None
                    self.last_bullet_risk = 0.0

            # --- 2. Enemy Risk Analysis ---
            if self.use_enemy_distance_reward:
                dist = self.mask_generator.compute_nearest_enemy_distance(
                    ship_pos, enemy_positions, normalize_by=frame_diagonal
                )
                
                if dist is not None:
                    raw_risk = 1.0 / (dist + 1e-6) + self.enemy_quadratic_coef / (dist**2 + 1e-6)
                    enemy_risk = min(raw_risk, self.risk_clip)
                    
                    # Stronger enemy escape bonus
                    # Enemies spawn immediate bullet waves, so fleeing is more critical
                    if self.last_enemy_dist is not None:
                        delta = self.last_enemy_dist - dist
                        if delta < 0: # Moving away
                            urgency = enemy_risk / self.risk_clip
                            # Use configurable enemy_escape_multiplier (default 10.0) instead of 5.0
                            enemy_escape_bonus = min(
                                -delta * self.risk_clip * urgency * self.enemy_escape_multiplier,
                                2.0  # Increased cap from 1.0 to 2.0 for stronger incentive
                            )
                            
                    self.last_enemy_dist = dist
                else:
                    self.last_enemy_dist = None

            # --- 3. Calculate Final Reward ---
            
            # Aggregate Risk with Enemy Danger Multiplier
            # Scale enemy risk relative to bullet risk using coefficients
            # If enemy_reward_coef is 2x bullet_reward_coef, enemy is 2x more "risky" (repulsive)
            base_coef = self.bullet_reward_coef if self.bullet_reward_coef > 0 else 0.01
            enemy_scale = (self.enemy_reward_coef / base_coef) if self.bullet_reward_coef > 0 else 1.0
            
            # Apply enemy danger multiplier to make enemies MUCH more repulsive
            # Default 3.0x means enemies contribute 3x more to total risk
            total_risk = bullet_risk + (enemy_risk * enemy_scale * self.enemy_danger_multiplier)
            
            # Safety Factor (0.0 to 1.0)
            # If total_risk is 0, safety is 1.0
            # If total_risk is high (e.g. 10), safety is ~0.1
            safety_factor = 1.0 / (1.0 + total_risk)
            
            # Min Safety Floor
            # Ensure that staying alive is ALWAYS worth something, even in extreme danger.
            # This distinguishes "Grazing/Dodging" (Alive but High Risk) from "Death" (0 Reward).
            # Without this, High Risk ~ 0 Reward, which confuses the agent.
            min_safety = 0.1
            safety_factor = max(safety_factor, min_safety)
            
            # Conditional Alive Reward
            conditional_alive_reward = self.alive_reward * safety_factor
            
            # Graze Bonus with Movement Requirement
            # Reward the agent for surviving high-risk situations.
            # This encourages the agent to be comfortable near bullets (dodging) rather than just panicking.
            # We only give this if risk is significant (e.g. > 1.0, which implies dist < ~1.0 normalized)
            graze_bonus = 0.0
            if total_risk > 1.0:
                # Bonus scales with risk, but is capped.
                graze_factor = min(total_risk, self.risk_clip) / self.risk_clip
                
                if self.graze_requires_movement:
                    # DECOUPLED: We now allow graze bonus if the agent is simply ALIVE in a high risk zone.
                    # The "Movement" requirement was too strict for perpendicular dodging.
                    # We still check if they are NOT getting significantly closer (suicide).
                    # If risk_delta is massive positive (running into bullet), maybe deny?
                    # For now, we trust the Safety Factor (which drops to 0) to handle suicide.
                    # So "Graze" is just "High Risk Survival".
                    graze_bonus = self.alive_reward * self.graze_bonus_multiplier * graze_factor
                else:
                    # Original behavior: always award regardless of movement
                    graze_bonus = self.alive_reward * self.graze_bonus_multiplier * graze_factor

            # Add All Bonuses
            total_escape_bonus = bullet_escape_bonus + enemy_escape_bonus + dodge_skill_bonus
            
            total_reward = conditional_alive_reward + total_escape_bonus + graze_bonus
            
            bullet_part = bullet_escape_bonus + dodge_skill_bonus
            enemy_part = enemy_escape_bonus
            
            # Return with detailed breakdown including safety metrics and separate escape bonuses
            # bullet_part includes dodge_skill_bonus since it's related to bullet dodging
            return total_reward, bullet_part, enemy_part, safety_factor, total_risk, graze_bonus, dodge_skill_bonus, bullet_escape_bonus, enemy_escape_bonus
        
        except Exception as e:
            print(f"Safety reward computation failed: {e}")
            return 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0

    def _render_frame(self, frame, info):
        display_src = frame
        display = cv2.resize(display_src, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.putText(display, f"Rew: {info['episode_reward']:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Step: {info['step_reward']:.3f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"B_Rew: {info['bullet_distance_reward']:.3f}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display, f"E_Rew: {info['enemy_distance_reward']:.3f}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        cv2.putText(display, f"Lum: {info['luminance']:.1f}", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show bullet distance status
        bd_status = "ON" if info['bullet_distance_enabled'] else "OFF"
        ed_status = "ON" if info['enemy_distance_enabled'] else "OFF"
        cv2.putText(display, f"BD: {bd_status} | ED: {ed_status}", (10, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Strategy indicator
        strategy_color = (255, 200, 100) if info['reward_strategy'] == 'safety' else (100, 200, 255)
        cv2.putText(display, f"Strategy: {info['reward_strategy'].upper()}", (10, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, strategy_color, 2)
        
        # Safety Factor (only for safety strategy)
        if info['reward_strategy'] == 'safety':
            safety = info['safety_factor']
            # Color code: green=safe, yellow=moderate, red=dangerous
            if safety > 0.7:
                safety_color = (0, 255, 0)  # Green
            elif safety > 0.4:
                safety_color = (0, 255, 255)  # Yellow
            else:
                safety_color = (0, 0, 255)  # Red
            
            cv2.putText(display, f"Safety: {safety:.2f}", (10, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, safety_color, 2)
        
        window_name = "Agent View"
        cv2.imshow(window_name, display)
        
        # Move window
        if self.window_rect:
            left, top, right, bottom = self.window_rect
            if self.render_mode == "both":
                # Place below the game window
                cv2.moveWindow(window_name, left, bottom)
            else:
                # Place to the right of the game window
                cv2.moveWindow(window_name, right, top)
            
        cv2.waitKey(1)
    
    def _render_debug_frame(self, mask, color_frame, info):
        """Render debug visualization showing mask detection details."""
        from generate_masks import MASK_BACKGROUND, MASK_BULLET, MASK_SHIP, MASK_ENEMY
        
        # Get positions
        ship_pos, bullet_positions, enemy_positions = self.mask_generator.get_positions(mask)
        
        # Create visualization panels
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
        
        # Draw bullet positions and find nearest
        nearest_bullet_idx = None
        if ship_pos is not None and len(bullet_positions) > 0:
            min_dist = float('inf')
            for idx, (bullet_y, bullet_x) in enumerate(bullet_positions):
                dist = np.sqrt((ship_pos[0] - bullet_y)**2 + (ship_pos[1] - bullet_x)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_bullet_idx = idx
        
        for idx, (bullet_y, bullet_x) in enumerate(bullet_positions):
            color = (0, 255, 255) if idx == nearest_bullet_idx else (128, 255, 128)
            cv2.circle(vis_detailed, (bullet_x, bullet_y), 2, color, -1)
            if ship_pos is not None:
                line_color = (0, 200, 200) if idx == nearest_bullet_idx else (64, 64, 64)
                cv2.line(vis_detailed, (ship_pos[1], ship_pos[0]), (bullet_x, bullet_y), line_color, 1)
        
        # Draw enemy positions and find nearest
        nearest_enemy_idx = None
        if ship_pos is not None and len(enemy_positions) > 0:
            min_dist = float('inf')
            for idx, (enemy_y, enemy_x) in enumerate(enemy_positions):
                dist = np.sqrt((ship_pos[0] - enemy_y)**2 + (ship_pos[1] - enemy_x)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy_idx = idx
        
        for idx, (enemy_y, enemy_x) in enumerate(enemy_positions):
            color = (255, 255, 0) if idx == nearest_enemy_idx else (255, 128, 128)
            cv2.circle(vis_detailed, (enemy_x, enemy_y), 2, color, -1)
            if ship_pos is not None:
                line_color = (255, 0, 255) if idx == nearest_enemy_idx else (150, 0, 150)
                cv2.line(vis_detailed, (ship_pos[1], ship_pos[0]), (enemy_x, enemy_y), line_color, 1)
        
        # Convert color_frame to BGR if needed and ensure it's the right size
        if color_frame.shape[:2] != (84, 84):
            color_frame = cv2.resize(color_frame, (84, 84), interpolation=cv2.INTER_LINEAR)
        
        # Create info panel
        info_panel = np.zeros((450, 600, 3), dtype=np.uint8)
        y_offset = 20
        line_height = 18
        
        # Count pixels
        num_background = np.sum(mask == MASK_BACKGROUND)
        num_bullet = np.sum(mask == MASK_BULLET)
        num_ship = np.sum(mask == MASK_SHIP)
        num_enemy = np.sum(mask == MASK_ENEMY)
        
        # Ship info
        cv2.putText(info_panel, "=== SHIP ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        y_offset += line_height
        cv2.putText(info_panel, f"Position: {ship_pos if ship_pos else 'NOT DETECTED'}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(info_panel, f"Pixels: {num_ship}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y_offset += line_height + 3
        
        # Bullet info
        cv2.putText(info_panel, "=== BULLETS ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(info_panel, f"Count: {len(bullet_positions)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(info_panel, f"Pixels: {num_bullet}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += line_height
        cv2.putText(info_panel, f"Reward: {info['bullet_distance_reward']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += line_height + 3
        
        # Enemy info
        cv2.putText(info_panel, "=== ENEMIES ===", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        y_offset += line_height
        cv2.putText(info_panel, f"Count: {len(enemy_positions)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(info_panel, f"Pixels: {num_enemy}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        y_offset += line_height
        cv2.putText(info_panel, f"Reward: {info['enemy_distance_reward']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += line_height + 3
        
        # Total reward and metadata
        cv2.putText(info_panel, f"TOTAL REWARD: {info['step_reward']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y_offset += line_height + 5
        cv2.putText(info_panel, f"Episode Reward: {info['episode_reward']:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        y_offset += line_height
        cv2.putText(info_panel, f"Luminance: {info['luminance']:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        y_offset += line_height + 5
        
        # Strategy indicator
        strategy_color = (255, 200, 100) if info['reward_strategy'] == 'safety' else (100, 200, 255)
        cv2.putText(info_panel, f"=== STRATEGY: {info['reward_strategy'].upper()} ===", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, strategy_color, 2)
        y_offset += line_height + 3
        
        # Safety metrics (show for safety strategy or if interesting for baseline)
        if info['reward_strategy'] == 'safety':
            # Safety Factor with color coding
            safety = info['safety_factor']
            if safety > 0.7:
                safety_color = (0, 255, 0)  # Green - Safe
            elif safety > 0.4:
                safety_color = (0, 255, 255)  # Yellow - Moderate
            else:
                safety_color = (0, 0, 255)  # Red - Dangerous
            
            cv2.putText(info_panel, f"Safety Factor: {safety:.3f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, safety_color, 2)
            y_offset += line_height
            
            # Total Risk
            cv2.putText(info_panel, f"Total Risk: {info['total_risk']:.3f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += line_height
            
            # Bonuses
            cv2.putText(info_panel, f"Graze Bonus: +{info['graze_bonus']:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 255), 1)
            y_offset += line_height
            
            cv2.putText(info_panel, f"Dodge Skill: +{info['dodge_skill_bonus']:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
            y_offset += line_height
            
            # Escape bonuses
            cv2.putText(info_panel, f"Bullet Escape: +{info['bullet_escape_bonus']:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            y_offset += line_height
            
            cv2.putText(info_panel, f"Enemy Escape: +{info['enemy_escape_bonus']:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
            y_offset += line_height
        
        # Resize for display (scale up from 84x84)
        frame_large = cv2.resize(color_frame, (400, 400), interpolation=cv2.INTER_NEAREST)
        vis_mask_large = cv2.resize(vis_mask, (400, 400), interpolation=cv2.INTER_NEAREST)
        vis_detailed_large = cv2.resize(vis_detailed, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        # Combine horizontally: Original | Mask | Detailed
        top_row = np.hstack([frame_large, vis_mask_large, vis_detailed_large])
        
        # Add info panel below
        bottom_row = np.zeros((450, 1200, 3), dtype=np.uint8)
        bottom_row[:, 0:600] = info_panel

        # Draw Q-values if available
        if info.get('q_values') is not None:
            q_vals = info['q_values']
            # Normalize for display
            q_min, q_max = np.min(q_vals), np.max(q_vals)
            if q_max != q_min:
                q_norm = (q_vals - q_min) / (q_max - q_min)
            else:
                q_norm = np.zeros_like(q_vals)
            
            # Bar chart parameters
            bar_width = 40
            bar_spacing = 15
            chart_x = 650
            chart_y = 400
            max_bar_height = 200
            
            # Action labels
            action_names = ["Idle", "W", "S", "A", "D", "WA", "WD", "SA", "SD"]
            
            cv2.putText(bottom_row, "Q-VALUES", (chart_x, chart_y - max_bar_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, val in enumerate(q_norm):
                x = chart_x + i * (bar_width + bar_spacing)
                h = int(val * max_bar_height)
                
                # Color based on value (Red=Low, Green=High)
                color = (0, int(255 * val), int(255 * (1 - val)))
                
                # Highlight selected action
                if i == self.last_action:
                    cv2.rectangle(bottom_row, (x - 2, chart_y - h - 2), (x + bar_width + 2, chart_y + 2), (255, 255, 255), 2)
                
                cv2.rectangle(bottom_row, (x, chart_y - h), (x + bar_width, chart_y), color, -1)
                
                # Label
                cv2.putText(bottom_row, action_names[i], (x, chart_y + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Value
                cv2.putText(bottom_row, f"{q_vals[i]:.1f}", (x, chart_y - h - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        display = np.vstack([top_row, bottom_row])
        
        window_name = "Debug: ENEMY | BULLET | SHIP Detection"
        cv2.imshow(window_name, display)
        
        # Move window to the right of the game window
        if self.window_rect:
            _, top, right, _ = self.window_rect
            cv2.moveWindow(window_name, right, top)
            
        cv2.waitKey(1)

    def _release_all_keys(self):
        """Releases all currently pressed keys."""
        for k in self.pressed_keys:
            pydirectinput.keyUp(k)
        self.pressed_keys.clear()

    def close(self):
        # Release any stuck keys
        if hasattr(self, 'pressed_keys'):
            self._release_all_keys()

        if self.render_mode in ["human", "debug"]:
            cv2.destroyAllWindows()
