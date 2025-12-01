"""
Shared training utilities for both DQN and PPO training scripts.

This module contains reusable functions for:
- Environment creation from args
- Checkpoint directory setup
- Common argument parsers
"""

import os
import argparse
from env import BulletHellEnv


def create_environment(args):
    """
    Create BulletHellEnv from parsed arguments.
    
    This function is shared between train.py and train_ppo.py to ensure
    consistent environment configuration regardless of the algorithm used.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        BulletHellEnv instance
    """
    env = BulletHellEnv(
        window_title=args.window_title,
        game_path=args.game_path,
        render_mode="both" if (args.render and args.render_debug) else (
            "debug" if args.render_debug else ("human" if args.render else None)
        ),
        frame_skip=args.frame_skip,
        stack_size=args.stack_size,
        alive_thresh=args.alive_thresh,
        dead_thresh=args.dead_thresh,
        dead_streak=args.dead_streak,
        save_screenshots=args.save_screenshots,
        reward_strategy=args.reward_strategy,
        use_bullet_distance_reward=not args.no_bullet_distance_reward,
        bullet_reward_coef=args.bullet_reward_coef,
        use_enemy_distance_reward=not args.no_enemy_distance_reward,
        enemy_reward_coef=args.enemy_reward_coef,
        bullet_quadratic_coef=args.bullet_quadratic_coef,
        bullet_density_coef=args.bullet_density_coef,
        enemy_quadratic_coef=args.enemy_quadratic_coef,
        alive_reward=args.alive_reward,
        death_penalty=args.death_penalty,
        risk_clip=args.risk_clip,
        force_mss=args.force_mss,
        bg_threshold=args.bg_threshold,
        dodge_skill_threshold=args.dodge_skill_threshold,
        dodge_skill_multiplier=args.dodge_skill_multiplier,
        graze_requires_movement=args.graze_requires_movement,
        graze_bonus_multiplier=args.graze_bonus_multiplier,
        enemy_danger_multiplier=args.enemy_danger_multiplier,
        enemy_escape_multiplier=args.enemy_escape_multiplier,
        use_mask_obs=args.use_mask_obs,
    )
    return env


def setup_directories(checkpoint_dir, log_dir, reward_strategy, algo="dqn"):
    """
    Setup checkpoint and log directories with proper naming.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        log_dir: Base log directory
        reward_strategy: "baseline" or "safety"
        algo: "dqn" or "ppo"
        
    Returns:
        tuple: (checkpoint_dir, log_dir) with proper suffixes
    """
    strategy_suffix = f"_{reward_strategy}"
    
    # Add algo prefix for PPO to keep separate from DQN
    if algo == "ppo":
        if not checkpoint_dir.endswith(strategy_suffix):
            checkpoint_dir = f"checkpoints_ppo_{reward_strategy}"
        if not log_dir.endswith(strategy_suffix):
            log_dir = f"logs_ppo_{reward_strategy}"
    else:
        # DQN: original behavior
        if not checkpoint_dir.endswith(strategy_suffix):
            checkpoint_dir = f"{checkpoint_dir}_{reward_strategy}"
        if not log_dir.endswith(strategy_suffix):
            log_dir = f"{log_dir}_{reward_strategy}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return checkpoint_dir, log_dir


def add_common_args(parser):
    """
    Add common arguments shared by both DQN and PPO training.
    
    This includes all environment and reward strategy parameters.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    # Environment params
    parser.add_argument("--window_title", type=str, default="100KBBH")
    parser.add_argument("--game_path", type=str, default="assets/100KBBH-1.0.3.exe",
                        help="Path to game executable")
    parser.add_argument("--render", action="store_true", help="Show agent view")
    parser.add_argument("--render-debug", action="store_true",
                        help="Show debug mask visualization (overrides --render)")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--stack_size", type=int, default=4)
    parser.add_argument("--alive_thresh", type=float, default=150.0)
    parser.add_argument("--dead_thresh", type=float, default=130.0)
    parser.add_argument("--dead_streak", type=int, default=3)
    parser.add_argument("--save-screenshots", type=int, default=0,
                        help="Save screenshots every X ms (0 to disable)")
    parser.add_argument("--force-mss", action="store_true",
                        help="Force usage of MSS for screen capture (bypass DXCAM)")
    parser.add_argument("--bg-threshold", type=int, default=2,
                        help="Background color matching threshold (default: 2)")
    
    # Shared hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (used by both DQN and PPO)")
    
    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=50,
                        help="Save checkpoint every N episodes/updates")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--keep-latest-only", action="store_true",
                        help="Only save latest checkpoints (saves disk space)")
    
    # Reward strategy selector
    parser.add_argument("--reward-strategy", type=str, default="baseline",
                        choices=["baseline", "safety"],
                        help="Reward strategy to use")
    
    # Distance-based reward shaping (both strategies)
    parser.add_argument("--no-bullet-distance-reward", action="store_true",
                        help="Disable bullet distance reward shaping")
    parser.add_argument("--bullet-reward-coef", type=float, default=0.01,
                        help="Coefficient for bullet distance reward")
    parser.add_argument("--bullet-quadratic-coef", type=float, default=0.1,
                        help="Quadratic coefficient for bullet distance reward")
    parser.add_argument("--no-enemy-distance-reward", action="store_true",
                        help="Disable enemy distance reward shaping")
    parser.add_argument("--enemy-reward-coef", type=float, default=0.02,
                        help="Coefficient for enemy distance reward")
    parser.add_argument("--enemy-quadratic-coef", type=float, default=0.1,
                        help="Quadratic coefficient for enemy distance reward")
    
    # Basic reward values (both strategies)
    parser.add_argument("--alive-reward", type=float, default=4.0,
                        help="Reward per frame survived when alive")
    parser.add_argument("--death-penalty", type=float, default=-20.0,
                        help="Penalty on death")
    parser.add_argument("--risk-clip", type=float, default=10.0,
                        help="Clip value for distance-based risk")
    
    # Baseline strategy only
    parser.add_argument("--bullet-density-coef", type=float, default=0.01,
                        help="Coefficient for cumulative bullet risk density penalty (baseline only)")
    
    # Safety strategy only
    parser.add_argument("--dodge-skill-threshold", type=float, default=2.0,
                        help="Cumulative risk threshold to trigger dodge skill bonus (safety only)")
    parser.add_argument("--dodge-skill-multiplier", type=float, default=0.15,
                        help="Dodge skill bonus per unit of cumulative risk (safety only)")
    parser.add_argument("--graze-requires-movement", action="store_true", default=True,
                        help="Require active movement to earn graze bonus (safety only)")
    parser.add_argument("--graze-bonus-multiplier", type=float, default=0.15,
                        help="Graze bonus as fraction of alive reward (safety only)")
    parser.add_argument("--enemy-danger-multiplier", type=float, default=3.0,
                        help="How much more dangerous enemies are vs bullets (safety only)")
    parser.add_argument("--enemy-escape-multiplier", type=float, default=10.0,
                        help="Enemy escape bonus multiplier, vs 5.0 for bullets (safety only)")
    
    # Observation settings
    parser.add_argument("--use-mask-obs", action=argparse.BooleanOptionalAction, default=True,
                        help="Use segmentation mask as observation instead of grayscale frames (default: True)")
