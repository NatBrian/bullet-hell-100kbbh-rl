# 100KBBH Reinforcement Learning Agent

This project implements a production-ready Deep Q-Network (DQN) agent to play the "100KBBH" bullet hell game. It uses visual input (screen capture) and simulates keyboard output to control the game, learning to dodge bullets through trial and error and advanced reward shaping.

![100KBBH-demo](assets/demo.gif)

## Key Features

*   **Visual Learning**: The agent plays using only pixels. It does not access the game's internal memory.
*   **Robust Screen Capture**:
    *   Primary: **DXCam** (High FPS, GPU-accelerated) with multiple initialization strategies for RTX GPUs.
    *   Fallback: **MSS** (Compatible, CPU-based) with auto-retry logic and error handling.
*   **Advanced Reward Shaping**:
    *   **Computer Vision**: Uses `BulletMaskGenerator` to detect bullets, enemies, and player ship in real-time from 84x84 frames.
    *   **Bullet Distance Rewards**: Penalizes proximity to bullets and rewards escaping. Includes closing penalty and escape bonus based on distance changes.
    *   **Enemy Distance Rewards**: Similar distance-based shaping for enemy avoidance.
    *   **Risk Clamping**: Prevents extreme reward values with configurable clipping.
    *   **Ship Detection Fallback**: Uses last known position when detection fails temporarily.
*   **Smooth Movement**:
    *   **Stateful Input Handling**: Maintains key press state across frames for smooth, continuous movement instead of repeated tap inputs.
    *   **Action Duration**: Configurable key press duration.
*   **Smart Automation**:
    *   **Auto-Launch**: Automatically starts the game executable if it's not running.
    *   **Batch Training**: `batch_train.py` orchestrates long training sessions with automatic restarts and checkpoint management.
*   **Robust Checkpoint Management**:
    *   **Atomic Saves**: Uses temporary files and atomic renames to prevent corruption.
    *   **Automatic Cleanup**: `--keep-latest-only` mode automatically deletes old numbered checkpoints, keeping only `latest.pth` and `latest_full.pth`.
    *   **Full Resume**: Supports resuming with complete training state (policy, optimizer, epsilon, replay buffer, steps, episodes).
*   **Advanced RL Algorithms**:
    *   **Frame Stacking**: Stacks 4 consecutive frames to perceive velocity and trajectory.
    *   **Double DQN**: Reduces overestimation bias for stable learning.
    *   **Experience Replay**: Stores past experiences to learn from them multiple times.
    *   **Frame Skip**: Configurable frame skip (default: 1) with proper reward accumulation.
*   **Live Visualization**: Shows "Agent View" window displaying what the agent sees (grayscale/masked) with real-time stats (reward, luminance).

## Project Structure

*   **`env.py`**: The **Gymnasium Environment**. Handles screen capture, window management, stateful input, and reward calculation.
*   **`agent.py`**: The **Brain**. Contains the `CNNQNetwork` (PyTorch) and `DQNAgent` with Double DQN support.
*   **`train.py`**: The **Training Loop**. Orchestrates training, logging, and atomic checkpointing.
*   **`batch_train.py`**: The **Orchestrator**. Runs training in batches with automatic checkpoint cleanup and full resume support.
*   **`generate_masks.py`**: **Vision System**. The `BulletMaskGenerator` class detects game entities and computes distances for reward shaping.
*   **`eval.py`**: The **Evaluator**. Runs the agent in greedy mode to test performance.
*   **`utils.py`**: Helper functions.
*   **`requirements.txt`**: Python dependencies.

## Installation

1.  **Prerequisites**: Windows 10/11, Python 3.8+.
2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Game Executable**: Place `100KBBH-1.0.3.exe` in `assets/` or specify path with `--game_path`.

## Usage

### 1. Batch Training (Recommended)

Use `batch_train.py` for robust, long-duration training. It handles restarts, checkpoint cleanup, and disk space automatically.

```powershell
# Run 100 batches of 100 episodes each (10,000 episodes total)
# With Double DQN and automatic checkpoint cleanup enabled
python batch_train.py --batches 100 --episodes-per-batch 100 --double-dqn --keep-latest-only

# Run with visualization (slower but good for debugging)
python batch_train.py --batches 10 --episodes-per-batch 50 --double-dqn --render

# Disable automatic cleanup to keep all numbered checkpoints
python batch_train.py --batches 100 --episodes-per-batch 100 --double-dqn --no-keep-latest-only
```

**Batch Training Arguments:**
*   `--batches`: Number of training batches to run (default: 100).
*   `--episodes-per-batch`: Episodes per batch (default: 100).
*   `--double-dqn`: Enable Double DQN algorithm.
*   `--render`: Show the agent's view during training.
*   `--keep-latest-only`: Only save latest checkpoints to save disk space (default: True).
*   `--no-keep-latest-only`: Keep all numbered checkpoints.

### 2. Manual Training

You can run `train.py` directly for debugging, short runs, or custom configurations.

```powershell
# Basic Training (1000 episodes)
python train.py

# With visualization and Double DQN
python train.py --render --double_dqn --total_episodes 1000

# Resume from a full checkpoint (policy + optimizer + epsilon + steps)
python train.py --full-resume checkpoints/latest_full.pth --add-episodes 500

# Resume from policy-only checkpoint
python train.py --resume checkpoints/latest.pth --total_episodes 2000

# Enable both bullet and enemy distance rewards (enabled by default)
python train.py --double_dqn

# Disable bullet distance rewards
python train.py --no-bullet-distance-reward

# Disable enemy distance rewards
python train.py --no-enemy-distance-reward

# Custom reward coefficients
python train.py --bullet-reward-coef 0.02 --enemy-reward-coef 0.015 --risk-clip 5.0

# Save screenshots every 1000ms for debugging
python train.py --save-screenshots 1000
```

**Key Training Arguments:**
*   `--total_episodes`: Total episodes to train (default: 1000).
*   `--add-episodes`: Train for this many additional episodes beyond resume point.
*   `--resume`: Path to policy-only checkpoint.
*   `--full-resume`: Path to full checkpoint (includes optimizer, epsilon, replay buffer, steps).
*   `--keep-latest-only`: Only save `latest.pth` and `latest_full.pth` (saves disk space).
*   `--double_dqn`: Enable Double DQN algorithm.
*   `--render`: Show the agent's view window.
*   `--no-bullet-distance-reward`: Disable bullet distance reward shaping.
*   `--bullet-reward-coef`: Linear coefficient for bullet distance reward (default: 0.01).
*   `--bullet-quadratic-coef`: Quadratic coefficient for bullet distance reward (default: 0.1).
*   `--no-enemy-distance-reward`: Disable enemy distance reward shaping.
*   `--enemy-reward-coef`: Linear coefficient for enemy distance reward (default: 0.02).
*   `--enemy-quadratic-coef`: Quadratic coefficient for enemy distance reward (default: 0.1).
*   `--alive-reward`: Reward per frame survived (default: 4.0).
*   `--death-penalty`: Penalty on death (default: -20.0).
*   `--risk-clip`: Clip value for distance-based risk (default: 10.0).
*   `--save-screenshots`: Save screenshots every X ms (0 to disable).
*   `--frame_skip`: Number of frames to repeat each action (default: 1).
*   `--game_path`: Path to game executable (default: `assets/100KBBH-1.0.3.exe`).

### 3. Evaluation

Test a trained model in greedy mode (no exploration).

```powershell
# Run 10 test episodes with visualization
python eval.py run --checkpoint checkpoints/latest.pth --episodes 10 --render

# Run 100 test episodes without rendering (faster)
python eval.py run --checkpoint checkpoints/latest.pth --episodes 100
```

## How It Works

The AI operates in a continuous loop:

1.  **Observation**: Captures the game window (client area only), resizes to 84x84.
2.  **Processing**:
    *   **Frame Stacking**: Maintains a stack of the last 4 frames (84x84 each).
    *   **Reward Shaping** (optional): `BulletMaskGenerator` analyzes the BGR frame to:
        *   Detect player ship position.
        *   Detect bullets.
        *   Detect enemies.
        *   Compute normalized distances to nearest bullet and enemy.
3.  **Decision**: The CNN outputs Q-values for 9 actions:
    *   0: Idle
    *   1-4: W, S, A, D (cardinal directions)
    *   5-8: WA, WD, SA, SD (diagonals)
4.  **Action**: Stateful keyboard input via `pydirectinput`:
    *   Keys are held down continuously across frames if the same action is repeated.
    *   Only changed keys are pressed/released when switching actions.
    *   All keys are released on episode reset and environment close.
5.  **Reward**:
    *   **Base Survival**: +4.0 per frame survived (when luminance > dead_thresh).
    *   **Death Penalty**: -20.0 when death detected (luminance below threshold for 3 consecutive frames).
    *   **Bullet Distance** (optional, enabled by default):
        *   **Risk**: Penalty based on distance: `-(linear_coef * (1/dist) + quadratic_coef * (1/dist^2))`, clamped by `risk_clip`.
        *   **Closing Penalty**: Extra penalty if moving closer to bullets.
        *   **Escape Bonus**: Bonus if moving away from bullets.
    *   **Enemy Distance** (optional, enabled by default): Same structure as bullet distance.
6.  **Learning**: 
    *   Stores (state, action, reward, next_state, done) in replay buffer.
    *   Samples random minibatches for training every 4 steps (configurable).
    *   Updates target network every 1000 steps.
    *   Epsilon decays from 1.0 to 0.1 over 200,000 steps.

## Troubleshooting

*   **"Window not found"**: 
    *   Ensure the game executable path is correct (default: `assets/100KBBH-1.0.3.exe`).
    *   The environment will attempt to auto-launch if `--game_path` is provided.
    *   Check that the window title matches (default: "100KBBH").
*   **Screen capture errors**: 
    *   DXCam initialization may fail on some systems; the code automatically falls back to MSS.
    *   Temporary capture failures are automatically retried (max 3 attempts).
    *   If persistent, check GPU drivers and available display outputs.
*   **Agent not moving**: 
    *   Run terminal as **Administrator** (required for `pydirectinput` on some systems).
    *   Ensure the game window is not minimized and is visible on screen.
*   **Checkpoint save failures**:
    *   Check available disk space (minimum 500MB required).
    *   Use `--keep-latest-only` to automatically clean up old checkpoints.
    *   Corrupted checkpoints are automatically cleaned up (`.tmp` files).
*   **Training crashes/memory leaks**:
    *   Use `batch_train.py` instead of running `train.py` directly.
    *   Each batch restarts the process, clearing any accumulated memory leaks.

## Advanced Configuration

### Custom Reward Tuning

The reward system is highly configurable to balance exploration vs. safety:

```powershell
# More aggressive bullet avoidance
python train.py --bullet-reward-coef 0.05 --risk-clip 5.0

# Focus only on survival (disable distance rewards)
python train.py --no-bullet-distance-reward --no-enemy-distance-reward --alive-reward 10.0

# Encourage more exploration with lower death penalty
python train.py --death-penalty -10.0
```

### Performance Optimization

```powershell
# Faster training without rendering (headless mode)
python batch_train.py --batches 1000 --episodes-per-batch 100

# Adjust frame skip for faster iteration (increases frame skip, less granularity)
python train.py --frame_skip 2

# Reduce replay buffer size for lower memory usage
python train.py --buffer_size 20000
```

# Acknowledgements

*   **100KBBH**: The game used in this project is available at [100KBBH](https://github.com/EterDelta/100KBBH).
