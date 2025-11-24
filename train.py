import argparse
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm

from env import BulletHellEnv
from agent import DQNAgent

def train(args):
    # Setup paths
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Logging
    writer = SummaryWriter(log_dir=args.log_dir)
    csv_path = os.path.join(args.log_dir, "training_log.csv")
    log_data = []

    # Environment
    env = BulletHellEnv(
        window_title=args.window_title,
        game_path=args.game_path,
        render_mode="human" if args.render else None,
        frame_skip=args.frame_skip,
        stack_size=args.stack_size,
        alive_thresh=args.alive_thresh,
        dead_thresh=args.dead_thresh,
        dead_streak=args.dead_streak
    )

    # Agent
    agent = DQNAgent(
        input_shape=(args.stack_size, 84, 84),
        num_actions=env.action_space.n,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        double_dqn=args.double_dqn
    )

    # Load checkpoint if provided
    start_episode = 0
    if args.resume:
        if os.path.exists(args.resume):
            agent.load(args.resume)
            print(f"Resumed from {args.resume}")
        else:
            print(f"Checkpoint {args.resume} not found, starting fresh.")

    # Training Loop
    epsilon = args.epsilon_start
    total_steps = 0
    
    pbar = tqdm(total=args.total_episodes)
    
    for episode in range(start_episode, args.total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        episode_steps = 0
        
        while True:
            # Select Action
            action = agent.act(state, epsilon)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn
            if total_steps > args.learning_starts and total_steps % args.train_freq == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update Target Net
                if total_steps % args.target_update_freq == 0:
                    agent.update_target_network()
                
                # Decay Epsilon
                if epsilon > args.epsilon_end:
                    epsilon -= (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_steps

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        # Logging
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Loss/Avg", avg_loss, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("Steps/Episode", episode_steps, episode)
        
        log_entry = {
            "episode": episode,
            "reward": episode_reward,
            "steps": episode_steps,
            "loss": avg_loss,
            "epsilon": epsilon,
            "total_steps": total_steps
        }
        log_data.append(log_entry)
        pd.DataFrame(log_data).to_csv(csv_path, index=False)
        
        pbar.set_description(f"Ep {episode} | Rew: {episode_reward:.1f} | Eps: {epsilon:.2f}")
        pbar.update(1)
        
        # Checkpointing
        if (episode + 1) % args.save_freq == 0:
            path = os.path.join(args.checkpoint_dir, f"checkpoint_{episode+1}.pth")
            agent.save(path)
            agent.save(os.path.join(args.checkpoint_dir, "latest.pth"))

    env.close()
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Env params
    parser.add_argument("--window_title", type=str, default="100KBBH")
    parser.add_argument("--game_path", type=str, default="100KBBH-1.0.3.exe", help="Path to game executable")
    parser.add_argument("--render", action="store_true", help="Show agent view")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--stack_size", type=int, default=4)
    parser.add_argument("--alive_thresh", type=float, default=150.0)
    parser.add_argument("--dead_thresh", type=float, default=130.0)
    parser.add_argument("--dead_streak", type=int, default=5)
    
    # Agent params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--double_dqn", action="store_true")
    
    # Training params
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=4, help="Train every N steps")
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    
    args = parser.parse_args()
    train(args)
