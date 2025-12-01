import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_checkpoint(checkpoint_path):
    """Diagnose checkpoint state and training progress."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    print("="*80)
    print(f"CHECKPOINT DIAGNOSTICS: {checkpoint_path}")
    print("="*80)
    
    # Basic info
    if "episode" in ckpt:
        print(f"\nEpisode: {ckpt['episode']}")
    if "total_steps" in ckpt:
        print(f"Total Steps: {ckpt['total_steps']}")
    if "epsilon" in ckpt:
        print(f"Epsilon: {ckpt['epsilon']:.4f}")
    
    # Args
    if "args" in ckpt:
        args = ckpt["args"]
        print(f"\nTraining Configuration:")
        print(f"  Reward Strategy: {args.get('reward_strategy', 'N/A')}")
        print(f"  Learning Rate: {args.get('lr', 'N/A')}")
        print(f"  Gamma: {args.get('gamma', 'N/A')}")
        print(f"  Buffer Size: {args.get('buffer_size', 'N/A')}")
        print(f"  Batch Size: {args.get('batch_size', 'N/A')}")
        print(f"  Frame Skip: {args.get('frame_skip', 'N/A')}")
        print(f"  Stack Size: {args.get('stack_size', 'N/A')}")
        print(f"  Double DQN: {args.get('double_dqn', 'N/A')}")
        print(f"\n  Epsilon Config:")
        print(f"    Start: {args.get('epsilon_start', 'N/A')}")
        print(f"    End: {args.get('epsilon_end', 'N/A')}")
        print(f"    Decay Steps: {args.get('epsilon_decay_steps', 'N/A')}")
        print(f"\n  Training Config:")
        print(f"    Learning Starts: {args.get('learning_starts', 'N/A')}")
        print(f"    Train Freq: {args.get('train_freq', 'N/A')} steps")
        print(f"    Target Update Freq: {args.get('target_update_freq', 'N/A')} steps")
        print(f"\n  Reward Config:")
        print(f"    Alive Reward: {args.get('alive_reward', 'N/A')}")
        print(f"    Death Penalty: {args.get('death_penalty', 'N/A')}")
        print(f"    Use Bullet Distance: {not args.get('no_bullet_distance_reward', True)}")
        print(f"    Use Enemy Distance: {not args.get('no_enemy_distance_reward', True)}")
        
        # Check if epsilon decay is reasonable
        total_steps = ckpt.get('total_steps', 0)
        epsilon_decay_steps = args.get('epsilon_decay_steps', 200000)
        epsilon_current = ckpt.get('epsilon', 1.0)
        epsilon_start = args.get('epsilon_start', 1.0)
        epsilon_end = args.get('epsilon_end', 0.1)
        
        expected_epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps)
        
        print(f"\n  Epsilon Analysis:")
        print(f"    Current: {epsilon_current:.4f}")
        print(f"    Expected (based on steps): {expected_epsilon:.4f}")
        print(f"    Decay Progress: {100 * total_steps / epsilon_decay_steps:.1f}%")
        
        if abs(epsilon_current - expected_epsilon) > 0.05:
            print(f"    [WARNING] Epsilon mismatch! Might indicate decay bug.")
    
    # Agent state
    if "agent" in ckpt:
        agent_state = ckpt["agent"]
        if "memory_state" in agent_state:
            mem_state = agent_state["memory_state"]
            print(f"\nReplay Buffer:")
            print(f"  Size: {mem_state.get('size', 'N/A')}")
            print(f"  Capacity: {mem_state.get('capacity', 'N/A')}")
            print(f"  Utilization: {100 * mem_state.get('size', 0) / max(mem_state.get('capacity', 1), 1):.1f}%")
    
    print("\n" + "="*80)


def analyze_training_logs(log_dir):
    """Analyze training logs for issues."""
    csv_path = os.path.join(log_dir, "training_log.csv")
    
    if not os.path.exists(csv_path):
        print(f"Training log not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print(f"TRAINING LOG ANALYSIS: {log_dir}")
    print("="*80)
    
    print(f"\nTotal Episodes: {len(df)}")
    print(f"Total Steps: {df['total_steps'].max() if 'total_steps' in df.columns else 'N/A'}")
    
    # Reward statistics
    print(f"\nReward Statistics:")
    print(f"  Mean: {df['reward'].mean():.2f}")
    print(f"  Std: {df['reward'].std():.2f}")
    print(f"  Min: {df['reward'].min():.2f}")
    print(f"  Max: {df['reward'].max():.2f}")
    print(f"  Last 100 Mean: {df['reward'].tail(100).mean():.2f}")
    print(f"  First 100 Mean: {df['reward'].head(100).mean():.2f}")
    
    # Episode length statistics
    if 'steps' in df.columns:
        print(f"\nEpisode Length (Steps):")
        print(f"  Mean: {df['steps'].mean():.2f}")
        print(f"  Std: {df['steps'].std():.2f}")
        print(f"  Min: {df['steps'].min()}")
        print(f"  Max: {df['steps'].max()}")
        print(f"  Last 100 Mean: {df['steps'].tail(100).mean():.2f}")
        print(f"  First 100 Mean: {df['steps'].head(100).mean():.2f}")
        
        # Convert steps to time (assuming ~60 FPS and frame_skip)
        # Default frame_skip = 1, so steps * 1 / 60 = seconds
        avg_survival_steps = df['steps'].tail(100).mean()
        avg_survival_seconds = avg_survival_steps / 60.0  # Assuming frame_skip=1
        print(f"\n  Average Survival Time (last 100 episodes): {avg_survival_seconds:.2f} seconds")
        
        # Check if there's improvement
        first_half_avg = df['steps'].head(len(df)//2).mean()
        second_half_avg = df['steps'].tail(len(df)//2).mean()
        improvement = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        print(f"\n  Improvement (first half vs second half): {improvement:+.2f}%")
        
        if abs(improvement) < 5:
            print(f"  [WARNING] Little to no improvement detected!")
    
    # Learning statistics
    if 'loss' in df.columns:
        print(f"\nLoss Statistics:")
        print(f"  Mean: {df['loss'].mean():.4f}")
        print(f"  Last 100 Mean: {df['loss'].tail(100).mean():.4f}")
    
    # Epsilon statistics
    if 'epsilon' in df.columns:
        print(f"\nExploration (Epsilon):")
        print(f"  Current: {df['epsilon'].iloc[-1]:.4f}")
        print(f"  Min Reached: {df['epsilon'].min():.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Check both strategies
    for strategy in ["baseline", "safety"]:
        checkpoint_path = f"checkpoints_{strategy}/latest_full.pth"
        log_dir = f"logs_{strategy}"
        
        if os.path.exists(checkpoint_path):
            diagnose_checkpoint(checkpoint_path)
        
        if os.path.exists(log_dir):
            analyze_training_logs(log_dir)
        
        print("\n\n")
