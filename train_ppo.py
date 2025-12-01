"""
PPO Training Script for Bullet Hell RL Agent.

This script trains a PPO agent using Stable-Baselines3.
It uses the same environment and reward configurations as train.py (DQN),
but with PPO-specific hyperparameters and training loop.
"""

import argparse
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent_ppo import PPOAgent
from env_wrapper import SB3BulletHellWrapper
from training_utils import create_environment, setup_directories, add_common_args


def train_ppo(args):
    """
    Train PPO agent.
    
    Args:
        args: Parsed command-line arguments
    """
    # Setup directories
    args.checkpoint_dir, args.log_dir = setup_directories(
        args.checkpoint_dir,
        args.log_dir,
        args.reward_strategy,
        algo="ppo"
    )
    
    print("=" * 70)
    print("PPO TRAINING")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log directory: {args.log_dir}")
    print(f"Reward strategy: {args.reward_strategy}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print("=" * 70)
    print()
    
    # Create environment
    print("Creating environment...")
    env = create_environment(args)
    
    # Wrap for SB3
    env = SB3BulletHellWrapper(env)
    print(f"Environment created: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # Create PPO agent
    print("Creating PPO agent...")
    agent = PPOAgent(env, args)
    print()
    
    # Handle resume from checkpoint
    current_timesteps = 0
    resume_path = args.resume or args.full_resume
    
    if resume_path:
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            agent = PPOAgent.load(resume_path, env)
            current_timesteps = agent.model.num_timesteps
            print(f"Resumed agent current timesteps: {current_timesteps:,}")
            print()
        else:
            print(f"Warning: Checkpoint {resume_path} not found. Starting fresh.")
            print()
    
    # Calculate save frequency in timesteps
    if args.save_freq_timesteps is not None:
        # Use explicit timestep-based save frequency
        save_freq_timesteps = args.save_freq_timesteps
        print(f"Save frequency: every {save_freq_timesteps:,} timesteps (explicit)")
    else:
        # Fall back to episode-based calculation
        # args.save_freq is in "episodes worth" - convert to timesteps
        # Rough estimate: 200 timesteps per episode
        save_freq_timesteps = args.save_freq * 200
        print(f"Save frequency: every ~{args.save_freq} episodes ({save_freq_timesteps:,} timesteps)")
    
    print("Starting training...")
    print(f"Training for additional {args.total_timesteps:,} timesteps")
    print(f"Target cumulative timesteps: {current_timesteps + args.total_timesteps:,}")
    print("=" * 70)
    print()
    
    try:
        # Train the agent
        # reset_num_timesteps=False ensures we continue counting from current_timesteps
        agent.train(
            total_timesteps=args.total_timesteps,
            save_freq=save_freq_timesteps,
            save_path=args.checkpoint_dir,
            reset_num_timesteps=False
        )
        
        # Save final model
        final_path = os.path.join(args.checkpoint_dir, "final_model")
        agent.save(final_path)
        print()
        print("=" * 70)
        print(f"Training complete! Final model saved to: {final_path}.w")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Training interrupted by user!")
        interrupt_path = os.path.join(args.checkpoint_dir, "interrupted_model")
        agent.save(interrupt_path)
        print(f"Saved interrupted model to: {interrupt_path}.zip")
        print("=" * 70)
    
    finally:
        # Clean up
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Bullet Hell RL agent using PPO"
    )
    
    # Add all common arguments (environment, rewards, etc.)
    add_common_args(parser)
    
    # PPO-SPECIFIC ARGUMENTS
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total timesteps to train (default: 100000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to PPO checkpoint (.zip) to resume from")
    parser.add_argument("--full-resume", type=str, default=None,
                        help="Alias for --resume (for consistency with DQN script)")
    parser.add_argument("--save-freq-timesteps", type=int, default=1000,
                        help="Save checkpoint every N timesteps (overrides save_freq if specified, default: None)")
    
    # PPO Hyperparameters
    parser.add_argument("--ppo-lr", type=float, default=3e-4,
                        help="PPO: Learning rate (default: 3e-4)")
    parser.add_argument("--ppo-n-steps", type=int, default=2048,
                        help="PPO: Number of steps to collect before update (default: 2048)")
    parser.add_argument("--ppo-batch-size", type=int, default=64,
                        help="PPO: Mini-batch size (default: 64)")
    parser.add_argument("--ppo-n-epochs", type=int, default=10,
                        help="PPO: Number of epochs per update (default: 10)")
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95,
                        help="PPO: GAE lambda parameter (default: 0.95)")
    parser.add_argument("--ppo-clip-range", type=float, default=0.2,
                        help="PPO: Clipping parameter (default: 0.2)")
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01,
                        help="PPO: Entropy coefficient for exploration (default: 0.01)")
    parser.add_argument("--ppo-vf-coef", type=float, default=0.5,
                        help="PPO: Value function coefficient (default: 0.5)")
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5,
                        help="PPO: Gradient clipping (default: 0.5)")
    
    args = parser.parse_args()
    train_ppo(args)
