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
        render_mode="human" if args.render else None,
        frame_skip=args.frame_skip,
        stack_size=args.stack_size,
        use_bullet_distance_reward=not args.no_bullet_distance_reward,
        bullet_reward_coef=args.bullet_reward_coef,
        use_enemy_distance_reward=not args.no_enemy_distance_reward,
        enemy_reward_coef=args.enemy_reward_coef
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
            action = agent.act(state, epsilon=0.0) # Greedy
            next_state, reward, terminated, truncated, _ = env.step(action)
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
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--window_title", type=str, default="100KBBH")
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.add_argument("--frame_skip", type=int, default=1)
    eval_parser.add_argument("--stack_size", type=int, default=4)
    eval_parser.add_argument("--cuda", action="store_true")
    # Reward shaping params (match training defaults)
    eval_parser.add_argument("--no-bullet-distance-reward", action="store_true", help="Disable bullet distance reward shaping")
    eval_parser.add_argument("--bullet-reward-coef", type=float, default=0.1, help="Coefficient for bullet distance reward")
    eval_parser.add_argument("--no-enemy-distance-reward", action="store_true", help="Disable enemy distance reward shaping")
    eval_parser.add_argument("--enemy-reward-coef", type=float, default=0.05, help="Coefficient for enemy distance reward")

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
