"""
PPO Evaluation Script for Bullet Hell RL Agent.

This script evaluates trained PPO models.
"""

import argparse
import numpy as np

from agent_ppo import PPOAgent
from env_wrapper import SB3BulletHellWrapper
from training_utils import create_environment, add_common_args


def evaluate_ppo(args):
    """
    Evaluate a trained PPO agent.
    
    Args:
        args: Parsed command-line arguments
    """
    print("=" * 70)
    print("PPO EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Reward strategy: {args.reward_strategy}")
    print("=" * 70)
    print()
    
    # Create environment
    print("Creating environment...")
    env = create_environment(args)
    
    # Wrap for SB3
    env = SB3BulletHellWrapper(env)
    print(f"Environment created.")
    print()
    
    # Load PPO agent
    print(f"Loading PPO model from: {args.checkpoint}")
    agent = PPOAgent.load(args.checkpoint, env)
    print("Model loaded successfully!")
    print()
   
    # Run evaluation episodes
    print(f"Running {args.episodes} evaluation episodes...")
    print("=" * 70)
    
    rewards = []
    episode_lengths = []
    
    for i in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Get action from PPO policy (deterministic for evaluation)
            action = agent.predict(obs, deterministic=not args.stochastic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        print(f"Episode {i+1}/{args.episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Steps = {episode_steps}")
    
    # Print summary statistics
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent"
    )
    
    # Add all common arguments
    add_common_args(parser)
    
    # Evaluation-specific arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint (.zip file)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate (default: 5)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    
    args = parser.parse_args()
    evaluate_ppo(args)
