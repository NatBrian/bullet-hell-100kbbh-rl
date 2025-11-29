import argparse
import subprocess
import sys
from pathlib import Path


def cleanup_old_checkpoints(checkpoint_dir: Path):
    """Delete old numbered checkpoints, keeping only latest.pth and latest_full.pth."""
    if not checkpoint_dir.exists():
        return
    
    deleted_count = 0
    deleted_size = 0
    
    for ckpt_file in checkpoint_dir.glob("checkpoint_*.pth*"):
        # Skip .tmp files (they'll be cleaned up separately)
        if ckpt_file.suffix == ".tmp":
            continue
        
        try:
            size = ckpt_file.stat().st_size
            ckpt_file.unlink()
            deleted_count += 1
            deleted_size += size
        except Exception as e:
            print(f"[Warning] Failed to delete {ckpt_file.name}: {e}")
    
    # Also clean up any leftover .tmp files from failed saves
    for tmp_file in checkpoint_dir.glob("*.tmp"):
        try:
            tmp_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"[Warning] Failed to delete {tmp_file.name}: {e}")
    
    if deleted_count > 0:
        size_mb = deleted_size / (1024 * 1024)
        print(f"[Cleanup] Cleaned up {deleted_count} old checkpoint(s), freed {size_mb:.1f} MB")


def pick_resume_path(checkpoint_dir: Path) -> Path | None:
    """Prefer full checkpoint, otherwise policy-only; return None if nothing exists."""
    full = checkpoint_dir / "latest_full.pth"
    policy = checkpoint_dir / "latest.pth"
    if full.exists():
        return full
    if policy.exists():
        return policy
    return None


def run_batch(batches: int, episodes_per_batch: int, checkpoint_dir: Path, extra_args: list[str], use_double_dqn: bool, render: bool, keep_latest_only: bool, force_mss: bool, reward_strategy: str):
    import time
    import gc
    
    # Adjust checkpoint dir based on strategy (same logic as train.py)
    strategy_suffix = f"_{reward_strategy}"
    if not str(checkpoint_dir).endswith(strategy_suffix):
        checkpoint_dir = Path(f"{str(checkpoint_dir)}_{reward_strategy}")
    
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    for i in range(batches):
        # Clean up old checkpoints to save disk space (keep only latest.pth and latest_full.pth)
        if keep_latest_only:
            cleanup_old_checkpoints(checkpoint_dir)
        
        resume_path = pick_resume_path(checkpoint_dir)
        cmd = [
            sys.executable,
            "train.py",
            "--add-episodes",
            str(episodes_per_batch),
            "--reward-strategy",
            reward_strategy,
            "--checkpoint_dir",
            str(checkpoint_dir),
        ]
        if use_double_dqn:
            cmd.append("--double_dqn")
        if render:
            cmd.append("--render")
        if keep_latest_only:
            cmd.append("--keep-latest-only")
        if force_mss:
            cmd.append("--force-mss")

        # If we have a full checkpoint, prefer it; otherwise fall back to policy-only.
        if resume_path:
            if resume_path.name.endswith("full.pth"):
                cmd.extend(["--full-resume", str(resume_path)])
            else:
                cmd.extend(["--resume", str(resume_path)])

        # Append any user-specified extra args (e.g., shaping toggles)
        cmd.extend(extra_args)

        print(f"[Batch {i+1}/{batches}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        # Critical: Add cleanup delay between batches
        # This allows the previous train.py process to fully release all resources
        # (window handles, keyboard state, MSS/DXCam, OpenCV windows)
        print(f"[Batch {i+1}] Process exited with code {result.returncode}")
        print("[Cleanup] Waiting for resource cleanup...")
        time.sleep(2.0)  # 2 second delay for OS to release resources
        gc.collect()  # Force garbage collection
        print("[Cleanup] Ready for next batch")
        
        if result.returncode != 0:
            print(f"[Batch {i+1}] Training exited with non-zero code {result.returncode}; stopping.")
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training in batches, resuming checkpoints between runs.")
    parser.add_argument("--batches", type=int, default=100, help="Number of batches to run")
    parser.add_argument("--episodes-per-batch", type=int, default=100, help="Episodes per batch")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True, help="Use Double DQN (default: True). Use --no-double-dqn to disable.")
    parser.add_argument("--render", action="store_true", help="Render agent view")
    parser.add_argument("--keep-latest-only", action=argparse.BooleanOptionalAction, default=True, help="Only save latest checkpoints to save disk space (default: True). Use --no-keep-latest-only to disable.")
    parser.add_argument("--force-mss", action="store_true", help="Force usage of MSS for screen capture (bypass DXCAM)")
    parser.add_argument("--reward-strategy", type=str, default="baseline", choices=["baseline", "safety"], help="Reward strategy to use")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args to forward to train.py (place after '--')",
    )
    args = parser.parse_args()

    run_batch(
        batches=args.batches,
        episodes_per_batch=args.episodes_per_batch,
        checkpoint_dir=args.checkpoint_dir,
        extra_args=args.extra_args,
        use_double_dqn=args.double_dqn,
        render=args.render,
        keep_latest_only=args.keep_latest_only,
        force_mss=args.force_mss,
        reward_strategy=args.reward_strategy,
    )
