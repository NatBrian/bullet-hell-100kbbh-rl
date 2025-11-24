import argparse
import subprocess
import sys
from pathlib import Path


def pick_resume_path(checkpoint_dir: Path) -> Path | None:
    """Prefer full checkpoint, otherwise policy-only; return None if nothing exists."""
    full = checkpoint_dir / "latest_full.pth"
    policy = checkpoint_dir / "latest.pth"
    if full.exists():
        return full
    if policy.exists():
        return policy
    return None


def run_batch(batches: int, episodes_per_batch: int, checkpoint_dir: Path, extra_args: list[str], use_double_dqn: bool, render: bool):
    for i in range(batches):
        resume_path = pick_resume_path(checkpoint_dir)
        cmd = [
            sys.executable,
            "train.py",
            "--add-episodes",
            str(episodes_per_batch),
        ]
        if use_double_dqn:
            cmd.append("--double_dqn")
        if render:
            cmd.append("--render")

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
        if result.returncode != 0:
            print(f"[Batch {i+1}] Training exited with code {result.returncode}; stopping.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training in batches, resuming checkpoints between runs.")
    parser.add_argument("--batches", type=int, default=100, help="Number of batches to run")
    parser.add_argument("--episodes-per-batch", type=int, default=100, help="Episodes per batch")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--double-dqn", action="store_true", help="Enable Double DQN")
    parser.add_argument("--render", action="store_true", help="Render agent view")
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
    )
