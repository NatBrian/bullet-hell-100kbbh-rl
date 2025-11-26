import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from env import BulletHellEnv
from agent import DQNAgent

from utils import generate_html_report

def generate_report(log_dir, output_file="report.html"):
    generate_html_report(log_dir, output_file)

def evaluate(args):
    env = BulletHellEnv(
        window_title=args.window_title,
        render_mode="both" if (args.render and args.render_debug) else ("debug" if args.render_debug else ("human" if args.render else None)),
        frame_skip=args.frame_skip,
        stack_size=args.stack_size,
        alive_thresh=args.alive_thresh,
        dead_thresh=args.dead_thresh,
        dead_streak=args.dead_streak,
        force_mss=args.force_mss,
        bg_threshold=args.bg_threshold,
        reward_strategy=args.reward_strategy,
        use_bullet_distance_reward=not args.no_bullet_distance_reward,
        bullet_reward_coef=args.bullet_reward_coef,
        bullet_quadratic_coef=args.bullet_quadratic_coef,
        use_enemy_distance_reward=not args.no_enemy_distance_reward,
        enemy_reward_coef=args.enemy_reward_coef,
        enemy_quadratic_coef=args.enemy_quadratic_coef,
        alive_reward=args.alive_reward,
        death_penalty=args.death_penalty,
        risk_clip=args.risk_clip,
        bullet_density_coef=args.bullet_density_coef,
        dodge_skill_threshold=args.dodge_skill_threshold,
        dodge_skill_multiplier=args.dodge_skill_multiplier,
        graze_requires_movement=args.graze_requires_movement,
        graze_bonus_multiplier=args.graze_bonus_multiplier,
        enemy_danger_multiplier=args.enemy_danger_multiplier,
        enemy_escape_multiplier=args.enemy_escape_multiplier,
    )

    agent = DQNAgent(
        input_shape=(args.stack_size, 84, 84),
        num_actions=env.action_space.n,
        device="cuda" if args.cuda else "cpu"
    )

    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    rewards = []
    for i in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if args.render or args.render_debug:
                action, q_values = agent.act(state, epsilon=0.0, return_q_values=True)
            else:
                action = agent.act(state, epsilon=0.0)
                q_values = None
                
            next_state, reward, terminated, truncated, _ = env.step(action, q_values=q_values)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            
        rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward {episode_reward}")

    print(f"Average Reward: {np.mean(rewards)}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Eval command
    eval_parser = subparsers.add_parser("run")
    
    # ============================================================================
    # SHARED PARAMETERS (Used by both baseline and safety strategies)
    # ============================================================================
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--window_title", type=str, default="100KBBH")
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.add_argument("--render-debug", action="store_true", help="Show debug mask visualization (overrides --render)")
    eval_parser.add_argument("--frame_skip", type=int, default=1)
    eval_parser.add_argument("--stack_size", type=int, default=4)
    eval_parser.add_argument("--alive_thresh", type=float, default=150.0, help="Luminance threshold for alive state")
    eval_parser.add_argument("--dead_thresh", type=float, default=130.0, help="Luminance threshold for dead state")
    eval_parser.add_argument("--dead_streak", type=int, default=3)
    eval_parser.add_argument("--force-mss", action="store_true", help="Force usage of MSS for screen capture (bypass DXCAM)")
    eval_parser.add_argument("--cuda", action="store_true")
    eval_parser.add_argument("--bg-threshold", type=int, default=2, help="Background color matching threshold (default: 2)")
    
    # Reward strategy selector
    eval_parser.add_argument("--reward-strategy", type=str, default="baseline", choices=["baseline", "safety"], help="Reward strategy to use")
    
    # Distance-based reward shaping (both strategies)
    eval_parser.add_argument("--no-bullet-distance-reward", action="store_true", help="Disable bullet distance reward shaping")
    eval_parser.add_argument("--bullet-reward-coef", type=float, default=0.01, help="Coefficient for bullet distance reward")
    eval_parser.add_argument("--bullet-quadratic-coef", type=float, default=0.1, help="Quadratic coefficient for bullet distance reward")
    eval_parser.add_argument("--no-enemy-distance-reward", action="store_true", help="Disable enemy distance reward shaping")
    eval_parser.add_argument("--enemy-reward-coef", type=float, default=0.02, help="Coefficient for enemy distance reward")
    eval_parser.add_argument("--enemy-quadratic-coef", type=float, default=0.1, help="Quadratic coefficient for enemy distance reward")
    
    # Basic reward values (both strategies)
    eval_parser.add_argument("--alive-reward", type=float, default=4.0, help="Reward per frame survived when alive")
    eval_parser.add_argument("--death-penalty", type=float, default=-20.0, help="Penalty on death")
    eval_parser.add_argument("--risk-clip", type=float, default=10.0, help="Clip value for distance-based risk")
    
    # ============================================================================
    # BASELINE STRATEGY ONLY
    # ============================================================================
    eval_parser.add_argument("--bullet-density-coef", type=float, default=0.01, help="Coefficient for cumulative bullet risk density penalty (baseline only)")
    
    # ============================================================================
    # SAFETY STRATEGY ONLY
    # ============================================================================
    # Dodging differentiation
    eval_parser.add_argument("--dodge-skill-threshold", type=float, default=2.0, help="Cumulative risk threshold to trigger dodge skill bonus (safety only)")
    eval_parser.add_argument("--dodge-skill-multiplier", type=float, default=0.15, help="Dodge skill bonus per unit of cumulative risk (safety only)")
    eval_parser.add_argument("--graze-requires-movement", action="store_true", default=True, help="Require active movement to earn graze bonus (safety only)")
    eval_parser.add_argument("--graze-bonus-multiplier", type=float, default=0.15, help="Graze bonus as fraction of alive reward (safety only)")
    # Enemy anticipation
    eval_parser.add_argument("--enemy-danger-multiplier", type=float, default=3.0, help="How much more dangerous enemies are vs bullets (safety only)")
    eval_parser.add_argument("--enemy-escape-multiplier", type=float, default=10.0, help="Enemy escape bonus multiplier, vs 5.0 for bullets (safety only)")

    # Report command
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()

    if args.command == "run":
        evaluate(args)
    elif args.command == "report":
        generate_report(args.log_dir)
    else:
        parser.print_help()
