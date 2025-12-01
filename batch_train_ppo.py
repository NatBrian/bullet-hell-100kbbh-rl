"""
Batch training script for PPO.

Runs multiple training batches sequentially, automatically resuming from
the latest checkpoint between batches. Useful for long training sessions
with periodic cleanup and resource management.
"""

import argparse
import subprocess
import sys
import time
import gc
from pathlib import Path


def cleanup_old_checkpoints(checkpoint_dir: Path):
    """
    Delete old numbered checkpoints, enforcing the single-file policy.
    
    For PPO, we now keep:
    - final_model.zip
    - interrupted_model.zip (if exists)
    - ppo_checkpoint.zip (latest training state)
    
    This function migrates legacy numbered checkpoints (ppo_checkpoint_*.zip)
    to ppo_checkpoint.zip and deletes the rest.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    """
    if not checkpoint_dir.exists():
        return
    
    # 1. Check for the new fixed-name checkpoint
    fixed_ckpt = checkpoint_dir / "ppo_checkpoint.zip"
    
    # 2. Get all legacy numbered checkpoints
    legacy_checkpoints = sorted(
        checkpoint_dir.glob("ppo_checkpoint_*.zip"),
        key=lambda p: p.stat().st_mtime
    )
    
    if not legacy_checkpoints:
        return

    # Migration: If we have legacy checkpoints but no fixed checkpoint,
    # rename the latest legacy one to the fixed name.
    if not fixed_ckpt.exists() and legacy_checkpoints:
        latest_legacy = legacy_checkpoints[-1]
        try:
            print(f"[Cleanup] Migrating latest checkpoint {latest_legacy.name} to ppo_checkpoint.zip")
            latest_legacy.rename(fixed_ckpt)
            # Remove it from the list of legacy checkpoints to delete since it's now the fixed one
            legacy_checkpoints.pop() 
        except Exception as e:
            print(f"[Warning] Failed to rename {latest_legacy.name}: {e}")
            # If rename failed, we shouldn't delete it
            return

    # 3. Delete ALL legacy checkpoints (since we either have a fixed one now, or we just created it)
    deleted_count = 0
    deleted_size = 0
    
    for old_ckpt in legacy_checkpoints:
        try:
            size = old_ckpt.stat().st_size
            old_ckpt.unlink()
            deleted_count += 1
            deleted_size += size
        except Exception as e:
            print(f"[Warning] Failed to delete {old_ckpt.name}: {e}")
    
    if deleted_count > 0:
        size_mb = deleted_size / (1024 * 1024)
        print(f"[Cleanup] Deleted {deleted_count} old numbered checkpoint(s), freed {size_mb:.1f} MB")


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """
    Find the most recent PPO checkpoint in the directory.
    
    Looks for:
    1. final_model.zip (if training completed)
    2. ppo_checkpoint.zip (new fixed name)
    3. Most recent ppo_checkpoint_*.zip (legacy fallback)
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None
    
    # Check for final model first
    final_model = checkpoint_dir / "final_model.zip"
    if final_model.exists():
        return final_model
    
    # Check for interrupted model
    interrupted_model = checkpoint_dir / "interrupted_model.zip"
    if interrupted_model.exists():
        return interrupted_model
    
    # Check for fixed name checkpoint (new standard)
    fixed_checkpoint = checkpoint_dir / "ppo_checkpoint.zip"
    if fixed_checkpoint.exists():
        return fixed_checkpoint
    
    # Fallback: Find most recent legacy checkpoint
    checkpoints = list(checkpoint_dir.glob("ppo_checkpoint_*.zip"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    return None


def run_batch(
    batches: int,
    timesteps_per_batch: int,
    checkpoint_dir: Path,
    extra_args: list[str],
    render: bool,
    render_debug: bool,
    keep_latest_only: bool,
    force_mss: bool,
    reward_strategy: str
):
    """
    Run multiple batches of PPO training.
    
    Args:
        batches: Number of training batches to run
        timesteps_per_batch: Timesteps per batch
        checkpoint_dir: Base checkpoint directory
        extra_args: Additional arguments to pass to train_ppo.py
        render: Whether to enable agent view rendering
        render_debug: Whether to enable full debug rendering
        keep_latest_only: Whether to clean up old checkpoints
        force_mss: Force MSS screen capture
        reward_strategy: Reward strategy ("baseline" or "safety")
    """
    # Checkpoint directory is automatically named by training_utils.setup_directories
    # It will be: checkpoints_ppo_{reward_strategy}
    print(f"Batch training PPO with {batches} batches")
    print(f"Timesteps per batch: {timesteps_per_batch:,}")
    print(f"Reward strategy: {reward_strategy}")
    print(f"Checkpoint directory will be: checkpoints_ppo_{reward_strategy}")
    print("=" * 70)
    print()
    
    for i in range(batches):
        print(f"[Batch {i+1}/{batches}] Starting...")
        
        # Clean up old checkpoints to save disk space
        if keep_latest_only and i > 0:
            actual_checkpoint_dir = Path(f"checkpoints_ppo_{reward_strategy}")
            cleanup_old_checkpoints(actual_checkpoint_dir)
        
        # Find latest checkpoint to resume from
        actual_checkpoint_dir = Path(f"checkpoints_ppo_{reward_strategy}")
        resume_path = find_latest_checkpoint(actual_checkpoint_dir)
        
        # Build command
        cmd = [
            sys.executable,
            "train_ppo.py",
            "--total-timesteps",
            str(timesteps_per_batch),
            "--reward-strategy",
            reward_strategy,
        ]
        
        # Add resume if checkpoint exists
        if resume_path:
            cmd.extend(["--resume", str(resume_path)])
            print(f"[Batch {i+1}] Resuming from: {resume_path}")
        else:
            print(f"[Batch {i+1}] Starting fresh (no checkpoint found)")
        
        # Add optional flags
        if render_debug:
            cmd.append("--render-debug")
        elif render:
            cmd.append("--render")
        if force_mss:
            cmd.append("--force-mss")
        
        # Append any user-specified extra args
        cmd.extend(extra_args)
        
        # Run training
        print(f"[Batch {i+1}] Command: {' '.join(cmd)}")
        print("=" * 70)
        result = subprocess.run(cmd)
        
        # Critical: Add cleanup delay between batches
        # This allows the previous train_ppo.py process to fully release all resources
        print(f"[Batch {i+1}] Process exited with code {result.returncode}")
        
        if result.returncode != 0:
            print(f"[Batch {i+1}] Training exited with error code {result.returncode}")
            print("Stopping batch training.")
            break
        
        # Cleanup between batches (except for last batch)
        if i < batches - 1:
            print("[Cleanup] Waiting for resource cleanup...")
            time.sleep(2.0)  # 2 second delay for OS to release resources
            gc.collect()  # Force garbage collection
            print("[Cleanup] Ready for next batch")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PPO training in batches with automatic resumption"
    )
    
    parser.add_argument("--batches", type=int, default=100,
                        help="Number of batches to run (default: 100)")
    parser.add_argument("--timesteps-per-batch", type=int, default=100000,
                        help="Timesteps per batch (default: 100000)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                        help="Base checkpoint directory (default: checkpoints)")
    parser.add_argument("--render", action="store_true",
                        help="Enable agent view rendering")
    parser.add_argument("--render-debug", action="store_true",
                        help="Enable full debug rendering (mask, detections, rewards)")
    parser.add_argument("--keep-latest-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Only keep latest 3 checkpoints to save space (default: True)")
    parser.add_argument("--force-mss", action="store_true",
                        help="Force MSS screen capture (bypass DXCAM)")
    parser.add_argument("--reward-strategy", type=str, default="baseline",
                        choices=["baseline", "safety"],
                        help="Reward strategy to use (default: baseline)")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional args to forward to train_ppo.py (place after '--')")
    
    args = parser.parse_args()
    
    run_batch(
        batches=args.batches,
        timesteps_per_batch=args.timesteps_per_batch,
        checkpoint_dir=args.checkpoint_dir,
        extra_args=args.extra_args,
        render=args.render,
        render_debug=args.render_debug,
        keep_latest_only=args.keep_latest_only,
        force_mss=args.force_mss,
        reward_strategy=args.reward_strategy,
    )
