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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        window_title="100KBBH",
        game_path=None,
        render_mode=None,
        frame_skip=1,  # Changed from 4 to 1 for faster iteration
        stack_size=4,
        alive_thresh=150.0, # LUMINANCE THRESHOLD FOR ALIVE
        dead_thresh=130.0, # LUMINANCE THRESHOLD FOR DEAD
        dead_streak=3,
        action_duration=0.016, # REACTION TIME AI PRESS KEY IN SECONDS (1 frame at 60 FPS)
        save_screenshots=0, # Interval in ms to save screenshots (0 to disable)
        use_bullet_distance_reward=False,  # Enable bullet distance reward shaping
        bullet_reward_coef=0.01,  # Coefficient for bullet distance reward
        use_enemy_distance_reward=False, # Enable enemy distance reward shaping
        enemy_reward_coef=0.02, # Coefficient for enemy distance reward
        bullet_quadratic_coef=0.10, # Quadratic coefficient for bullet distance
        enemy_quadratic_coef=0.10, # Quadratic coefficient for enemy distance
        alive_reward=4.0,  # Reward per frame survived (when bright)
        death_penalty=-20.0,  # Penalty on death
        risk_clip=10.0,  # Clip for distance-based risk
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
        self.use_bullet_distance_reward = use_bullet_distance_reward
        self.bullet_reward_coef = bullet_reward_coef
        self.use_enemy_distance_reward = use_enemy_distance_reward
        self.enemy_reward_coef = enemy_reward_coef
        self.bullet_quadratic_coef = bullet_quadratic_coef
        self.enemy_quadratic_coef = enemy_quadratic_coef
        self.alive_reward = alive_reward
        self.death_penalty = death_penalty
        self.risk_clip = risk_clip
        self.last_bullet_dist = None
        self.last_enemy_dist = None
        
        # Initialize bullet mask generator if needed
        if self.use_bullet_distance_reward or self.use_enemy_distance_reward:
            self.mask_generator = BulletMaskGenerator()
        
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
        if DXCAM_AVAILABLE:
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
                        try:
                            del self.camera
                        except:
                            pass
                    self.camera = None
            
            if not self.use_dxcam:
                print("All dxcam initialization attempts failed. Falling back to mss.")
        
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
            pydirectinput.press('space') # Keep tapping space if stuck on death screen

        # 4. Fill stack
        use_mask = self.use_bullet_distance_reward or self.use_enemy_distance_reward
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

    def step(self, action):
        """Execute action with proper frame skipping.
        
        Frame skip means:
        1. Execute the same action for N consecutive frames
        2. Accumulate rewards over all N frames
        3. Only add the LAST frame to the observation stack
        4. Terminate as soon as death is detected
        """
        total_reward = 0.0
        terminated = False
        last_frame = None  # what goes into the stack
        last_render_frame = None  # what we show in render mode (original grayscale)
        last_lum = 0.0
        
        # Execute action for frame_skip frames
        for frame_idx in range(self.frame_skip):
            # 1. Perform Action (Stateful)
            target_keys = set(self.key_map[action])
            if action != self.last_action:
                # Release keys that are no longer needed
                for k in self.pressed_keys - target_keys:
                    pydirectinput.keyUp(k)
                
                # Press new keys
                for k in target_keys - self.pressed_keys:
                    pydirectinput.keyDown(k)
                
                self.pressed_keys = target_keys
            
            # Only sleep if we are rendering for human to see, otherwise run as fast as possible
            if self.render_mode == "human":
                time.sleep(self.action_duration)  # Idle

            # 2. Capture & Process
            use_mask = self.use_bullet_distance_reward or self.use_enemy_distance_reward
            if use_mask:
                gray_frame, color_frame = self._capture_frame(return_color=True)
                mask = self.mask_generator.generate_mask(color_frame)
                obs_frame = mask
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
                # Avoid granting survival reward on dark frames that precede death detection
                frame_reward = 0.0 if lum < self.dead_thresh else self.alive_reward
                
                # Add distance shaping if enabled
                if use_mask and mask is not None:
                    dist_reward = self._compute_distance_rewards(mask)
                    frame_reward += dist_reward
            
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
            "frames_executed": frame_idx + 1  # How many frames were actually executed
        }
        
        # 6. Render
        if self.render_mode == "human":
            self._render_frame(last_render_frame, info)

        return self._get_obs(), total_reward, terminated, False, info
    
    def _compute_distance_rewards(self, mask):
        """
        Compute reward based on distance to nearest bullet and enemy using quadratic shaping.
        
        Args:
            mask: 84x84 segmentation mask (BGR frame already downscaled)
        
        Returns:
            Total distance reward (float)
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
                return -0.5
            else:
                # Can't find ship at all - heavily penalize
                return -1.0
            
            total_reward = 0.0
            frame_diagonal = np.sqrt(84**2 + 84**2)

            # Bullet Reward (quadratic distance shaping)
            if self.use_bullet_distance_reward:
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
                    total_reward -= self.bullet_reward_coef * (risk + closing_pen)
                    total_reward += self.bullet_reward_coef * escape_bonus
                    self.last_bullet_dist = dist
                else:
                    self.last_bullet_dist = None

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
                    total_reward -= self.enemy_reward_coef * (risk + closing_pen)
                    total_reward += self.enemy_reward_coef * escape_bonus
                    self.last_enemy_dist = dist
                else:
                    self.last_enemy_dist = None

            return total_reward
        
        except Exception as e:
            # If mask generation fails, return neutral reward
            print(f"Reward computation failed: {e}")
            return 0.0

    def _render_frame(self, frame, info):
        display_src = frame
        display = cv2.resize(display_src, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.putText(display, f"Rew: {info['episode_reward']:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Step: {info['step_reward']:.3f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Lum: {info['luminance']:.1f}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show bullet distance status
        bd_status = "ON" if info['bullet_distance_enabled'] else "OFF"
        ed_status = "ON" if info['enemy_distance_enabled'] else "OFF"
        cv2.putText(display, f"BD: {bd_status} | ED: {ed_status}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        window_name = "Agent View"
        cv2.imshow(window_name, display)
        
        # Move window to the right of the game window
        if self.window_rect:
            _, top, right, _ = self.window_rect
            cv2.moveWindow(window_name, right + 20, top)
            
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

        if self.render_mode == "human":
            cv2.destroyAllWindows()
