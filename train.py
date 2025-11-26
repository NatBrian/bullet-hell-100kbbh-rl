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

def save_full_checkpoint(path, agent, epsilon, total_steps, episode, args):
    """Save checkpoint with atomic write and error handling."""
    import shutil
    
    # Check available disk space (require at least 500MB free)
    try:
        stat = shutil.disk_usage(os.path.dirname(path) or '.')
        free_gb = stat.free / (1024**3)
        if stat.free < 500 * 1024 * 1024:  # 500MB minimum
            raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB free. Need at least 0.5GB.")
    except FileNotFoundError:
        # Directory might not exist yet, which is fine for the check if we check parent
        pass
    
    ckpt = {
        "agent": agent.get_full_state(),
        "epsilon": epsilon,
        "total_steps": total_steps,
        "episode": episode,
        "args": vars(args),
    }
    
    # Use atomic write: save to temp file, then rename
    temp_path = path + ".tmp"
    try:
        torch.save(ckpt, temp_path)
        # Atomic rename (os.replace is atomic on POSIX and Windows Python 3.3+)
        os.replace(temp_path, path)
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}") from e

def load_full_checkpoint(path, agent, epsilon_start):
    # Handle PyTorch >=2.6 weights_only default by explicitly allowing full pickled state
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    
    if "agent" in ckpt:
        agent.load_full_state(ckpt["agent"])
        epsilon = ckpt.get("epsilon", epsilon_start)
        total_steps = ckpt.get("total_steps", 0)
        start_episode = ckpt.get("episode", 0) + 1
        print(f"Loaded full checkpoint from {path} (episode {start_episode})")
    else:
        # Fallback: treat as policy-only checkpoint
        agent.load(path)
        epsilon = epsilon_start
        total_steps = 0
        start_episode = 0
        print(f"Loaded policy-only checkpoint from {path}; starting fresh epsilon/steps")
    return epsilon, total_steps, start_episode

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
        render_mode="both" if (args.render and args.render_debug) else ("debug" if args.render_debug else ("human" if args.render else None)),
        frame_skip=args.frame_skip,
        stack_size=args.stack_size,
        alive_thresh=args.alive_thresh,
        dead_thresh=args.dead_thresh,
        dead_streak=args.dead_streak,
        save_screenshots=args.save_screenshots,
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
        bg_threshold=args.bg_threshold
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
    epsilon = args.epsilon_start
    total_steps = 0
    if args.full_resume:
        if os.path.exists(args.full_resume):
            epsilon, total_steps, start_episode = load_full_checkpoint(args.full_resume, agent, args.epsilon_start)
            print(f"Resumed full checkpoint from {args.full_resume}")
        else:
            print(f"Full checkpoint {args.full_resume} not found, starting fresh.")
    elif args.resume:
        if os.path.exists(args.resume):
            agent.load(args.resume)
            print(f"Resumed from {args.resume}")
        else:
            print(f"Checkpoint {args.resume} not found, starting fresh.")

    # Training Loop
    if args.add_episodes is not None:
        end_episode = start_episode + args.add_episodes
        planned_episodes = args.add_episodes
    else:
        end_episode = args.total_episodes
        planned_episodes = end_episode - start_episode

    if planned_episodes <= 0:
        print(f"No episodes to run (start_episode={start_episode}, target_end={end_episode}). "
              f"Increase --total_episodes or pass --add-episodes to continue training.")
        env.close()
        writer.close()
        return
    
    pbar = tqdm(total=planned_episodes)
    
    for episode in range(start_episode, end_episode):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        episode_steps = 0
        
        while True:
            # Select Action
            if args.render or args.render_debug:
                action, q_values = agent.act(state, epsilon, return_q_values=True)
            else:
                action = agent.act(state, epsilon)
                q_values = None
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action, q_values=q_values)
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
            try:
                # Determine checkpoint paths
                if not args.keep_latest_only:
                    # Save numbered checkpoints (old behavior)
                    path = os.path.join(args.checkpoint_dir, f"checkpoint_{episode+1}.pth")
                    full_path = os.path.join(args.checkpoint_dir, f"checkpoint_{episode+1}_full.pth")
                else:
                    # Skip numbered checkpoints (space-saving mode)
                    path = None
                    full_path = None
                
                # Always save latest checkpoints
                latest = os.path.join(args.checkpoint_dir, "latest.pth")
                latest_full = os.path.join(args.checkpoint_dir, "latest_full.pth")
                
                # Save policy-only checkpoints
                if path:
                    agent.save(path)
                agent.save(latest)
                
                # Save full checkpoints
                if full_path:
                    save_full_checkpoint(full_path, agent, epsilon, total_steps, episode, args)
                save_full_checkpoint(latest_full, agent, epsilon, total_steps, episode, args)
                
                # Clean up old numbered checkpoints if in keep_latest_only mode
                if args.keep_latest_only:
                    from pathlib import Path
                    checkpoint_path = Path(args.checkpoint_dir)
                    deleted_count = 0
                    for old_ckpt in checkpoint_path.glob("checkpoint_*.pth*"):
                        if old_ckpt.suffix != ".tmp":
                            try:
                                old_ckpt.unlink()
                                deleted_count += 1
                            except:
                                pass
                    if deleted_count > 0:
                        print(f"\nCleaned up {deleted_count} old checkpoint(s)")
                
                print(f"\n[Saved] Saved checkpoints at episode {episode+1}")
            except Exception as e:
                print(f"\n[Warning] Failed to save checkpoint at episode {episode+1}: {e}")
                print("Training will continue, but checkpoint was not saved.")

    env.close()
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Env params
    parser.add_argument("--window_title", type=str, default="100KBBH")
    parser.add_argument("--game_path", type=str, default="assets/100KBBH-1.0.3.exe", help="Path to game executable")
    parser.add_argument("--render", action="store_true", help="Show agent view")
    parser.add_argument("--render-debug", action="store_true", help="Show debug mask visualization (overrides --render)")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--stack_size", type=int, default=4)
    parser.add_argument("--alive_thresh", type=float, default=150.0)
    parser.add_argument("--dead_thresh", type=float, default=130.0)
    parser.add_argument("--dead_streak", type=int, default=3)
    
    # Agent params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--double_dqn", action="store_true")
    
    # Training params
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--add-episodes", type=int, default=None, help="Train for this many episodes beyond the resume point (ignored when not resuming)")
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=4, help="Train every N steps")
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay_steps", type=int, default=200000, help="Epsilon decay steps (increased for better exploration)")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--full-resume", type=str, default=None, help="Path to full checkpoint (policy+optimizer+replay) to resume")
    parser.add_argument("--keep-latest-only", action="store_true", help="Only save latest.pth and latest_full.pth (saves disk space)")
    parser.add_argument("--save-screenshots", type=int, default=0, help="Save screenshots every X ms (0 to disable)")
    parser.add_argument("--alive-reward", type=float, default=4.0, help="Reward per frame survived when alive")
    parser.add_argument("--death-penalty", type=float, default=-20.0, help="Penalty on death")
    parser.add_argument("--risk-clip", type=float, default=10.0, help="Clip value for distance-based risk")
    
    # Bullet distance reward params (enabled by default)
    parser.add_argument("--no-bullet-distance-reward", action="store_true", help="Disable bullet distance reward shaping")
    parser.add_argument("--bullet-reward-coef", type=float, default=0.01, help="Coefficient for bullet distance reward")
    parser.add_argument("--bullet-quadratic-coef", type=float, default=0.1, help="Quadratic coefficient for bullet distance reward")
    parser.add_argument("--bullet-density-coef", type=float, default=0.01, help="Coefficient for cumulative bullet risk (density)")
    
    # Enemy distance reward params (enabled by default)
    parser.add_argument("--no-enemy-distance-reward", action="store_true", help="Disable enemy distance reward shaping")
    parser.add_argument("--enemy-reward-coef", type=float, default=0.02, help="Coefficient for enemy distance reward")
    parser.add_argument("--enemy-quadratic-coef", type=float, default=0.1, help="Quadratic coefficient for enemy distance reward")
    
    # Debugging
    parser.add_argument("--force-mss", action="store_true", help="Force usage of MSS for screen capture (bypass DXCAM)")
    parser.add_argument("--bg-threshold", type=int, default=2, help="Background color matching threshold (default: 2)")

    args = parser.parse_args()
    train(args)
