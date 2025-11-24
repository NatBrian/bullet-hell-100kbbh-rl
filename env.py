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
import threading
from collections import deque
from gymnasium import spaces

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
        frame_skip=4,
        stack_size=4,
        alive_thresh=150.0, # LUMINANCE THRESHOLD FOR ALIVE
        dead_thresh=130.0, # LUMINANCE THRESHOLD FOR DEAD
        dead_streak=5,
        action_duration=0.01, # REACTION TIME AI PRESS KEY IN SECONDS
        save_screenshots=0, # Interval in ms to save screenshots (0 to disable)
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
        
        # Initialize capture method
        self._init_capture()

    def _init_capture(self):
        """Initialize dxcam or mss."""
        self.use_dxcam = False
        if DXCAM_AVAILABLE:
            try:
                self.camera = dxcam.create(output_color="BGR")
                self.use_dxcam = True
                print("Using dxcam for screen capture.")
            except Exception as e:
                print(f"dxcam initialization failed: {e}. Falling back to mss.")
        
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

    def _capture_frame(self):
        """Captures a frame from the window."""
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
                    with self.mss_sct as sct:
                        monitor = {"top": top, "left": left, "width": width, "height": height}
                        img = sct.grab(monitor)
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

                # Resize and Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY) if frame.shape[2] == 4 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
                return frame
            
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
        
        # 1. Focus window
        try:
            self.window_rect = self._get_window_rect()
        except RuntimeError as e:
            print(e)
            # Wait and retry once
            time.sleep(2)
            self.window_rect = self._get_window_rect()

        # 2. Press SPACE to restart
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
        frame = self._capture_frame()
        for _ in range(self.stack_size):
            self.frame_stack.append(frame)
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Perform Action
        keys = self.key_map[action]
        if keys:
            for k in keys:
                pydirectinput.keyDown(k)
            time.sleep(self.action_duration)
            for k in keys:
                pydirectinput.keyUp(k)
        else:
            time.sleep(self.action_duration) # Idle

        # 2. Capture & Process
        # Frame skip: we already waited action_duration. 
        # If we want true frame skip, we might repeat the action or just wait.
        # For simplicity, we capture once after the action.
        frame = self._capture_frame()
        self.frame_stack.append(frame)
        
        # 3. Check Death
        lum = np.mean(frame)
        self.luminance_history.append(lum)
        
        terminated = False
        # If all recent frames are dark, we are dead
        if len(self.luminance_history) == self.dead_streak:
            if all(l < self.dead_thresh for l in self.luminance_history):
                terminated = True
        
        # 4. Reward
        # This is the only feedback the AI gets.
        # +1 for every "tick" it survives.
        # -100 if it dies (screen goes dark).
        # The AI tries to maximize this number over time.
        reward = 1.0 if not terminated else -100.0
        self.episode_reward += reward
        self.steps += 1
        
        # 5. Info
        info = {
            "luminance": lum,
            "episode_reward": self.episode_reward
        }
        
        # 6. Render
        if self.render_mode == "human":
            self._render_frame(frame, info)

        return self._get_obs(), reward, terminated, False, info

    def _render_frame(self, frame, info):
        display = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.putText(display, f"Rew: {info['episode_reward']:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Lum: {info['luminance']:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        window_name = "Agent View"
        cv2.imshow(window_name, display)
        
        # Move window to the right of the game window
        if self.window_rect:
            _, top, right, _ = self.window_rect
            cv2.moveWindow(window_name, right + 20, top)
            
        cv2.waitKey(1)

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
