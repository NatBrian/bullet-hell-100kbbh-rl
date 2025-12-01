"""
PPO Agent wrapper using Stable-Baselines3.

This wrapper provides a similar interface to DQNAgent, making it easy to
switch between DQN and PPO in the training/evaluation scripts.
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure


class CheckpointCallback(BaseCallback):
    """
    Callback for saving checkpoints during PPO training.
    """
    def __init__(self, save_freq, save_path, name_prefix='checkpoint', verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Use model's total timestep counter (persists across resume)
            total_timesteps = self.model.num_timesteps
            # Save to fixed filename "ppo_checkpoint" (SB3 adds .zip)
            # This overwrites the previous checkpoint, effectively "cleaning up" old ones
            path = os.path.join(self.save_path, self.name_prefix)
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path} (total timesteps: {total_timesteps})")
        return True


class PPOAgent:
    """
    PPO agent wrapper using Stable-Baselines3.
    
    Provides an interface similar to DQNAgent for easy integration
    with existing training scripts.
    """
    
    def __init__(self, env, args):
        """
        Initialize PPO agent.
        
        Args:
            env: SB3BulletHellWrapper instance (wrapped environment)
            args: Training arguments from argparse
        """
        # Extract PPO hyperparameters from args, with defaults
        ppo_lr = getattr(args, 'ppo_lr', 3e-4)
        ppo_n_steps = getattr(args, 'ppo_n_steps', 2048)
        ppo_batch_size = getattr(args, 'ppo_batch_size', 64)
        ppo_n_epochs = getattr(args, 'ppo_n_epochs', 10)
        ppo_gae_lambda = getattr(args, 'ppo_gae_lambda', 0.95)
        ppo_clip_range = getattr(args, 'ppo_clip_range', 0.2)
        ppo_ent_coef = getattr(args, 'ppo_ent_coef', 0.01)
        ppo_vf_coef = getattr(args, 'ppo_vf_coef', 0.5)
        ppo_max_grad_norm = getattr(args, 'ppo_max_grad_norm', 0.5)
        
        # Reuse gamma from DQN args (discount factor is algorithm-agnostic)
        gamma = getattr(args, 'gamma', 0.99)
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # TensorBoard logging
        tensorboard_log = getattr(args, 'log_dir', 'logs_ppo')
        
        # Create PPO model
        self.model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=ppo_lr,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=ppo_n_epochs,
            gamma=gamma,
            gae_lambda=ppo_gae_lambda,
            clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef,
            vf_coef=ppo_vf_coef,
            max_grad_norm=ppo_max_grad_norm,
            tensorboard_log=tensorboard_log,
            verbose=1,
            device=device,
            # Policy network kwargs (similar to DQN architecture)
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512],  # Policy network: 1 hidden layer with 512 units
                    vf=[512]   # Value network: 1 hidden layer with 512 units
                ),
                # CNN feature extractor will be created automatically by SB3
            )
        )
        
        print(f"Initialized PPO agent with:")
        print(f"  Learning rate: {ppo_lr}")
        print(f"  Steps per update: {ppo_n_steps}")
        print(f"  Batch size: {ppo_batch_size}")
        print(f"  Epochs per update: {ppo_n_epochs}")
        print(f"  Gamma: {gamma}")
        print(f"  Device: {device}")
    
    def train(self, total_timesteps, save_freq=None, save_path=None, reset_num_timesteps=True):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total number of environment steps to train for
            save_freq: Save checkpoint every N timesteps (optional)
            save_path: Directory to save checkpoints (optional)
            reset_num_timesteps: Whether to reset the total timestep counter (default: True).
                               Set to False when resuming training to keep cumulative count.
        """
        callbacks = []
        
        # Add checkpoint callback if specified
        if save_freq is not None and save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix='ppo_checkpoint',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Combine callbacks if any exist
        callback = CallbackList(callbacks) if callbacks else None
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            log_interval=10,  # Log every 10 updates
            reset_num_timesteps=reset_num_timesteps
        )
    
    def predict(self, state, deterministic=True):
        """
        Get action from policy.
        
        Args:
            state: Observation (numpy array)
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Action index (int)
        """
        action, _states = self.model.predict(state, deterministic=deterministic)
        return int(action)
    
    def save(self, path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint (will add .zip extension)
        """
        # SB3 automatically adds .zip extension
        self.model.save(path)
        print(f"Saved PPO model to {path}.zip")
    
    @classmethod
    def load(cls, path, env):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            env: Environment instance (wrapped)
            
        Returns:
            PPOAgent instance with loaded model
        """
        # Create a new instance without calling __init__
        instance = cls.__new__(cls)
        
        # Load the model
        instance.model = PPO.load(path, env=env)
        
        print(f"Loaded PPO model from {path}")
        return instance
