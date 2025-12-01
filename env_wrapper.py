"""
Stable-Baselines3 compatibility wrapper for BulletHellEnv.

This minimal wrapper ensures the environment works correctly with SB3's PPO.
The BulletHellEnv is already Gymnasium-compliant, so very little wrapping is needed.
"""

import gymnasium as gym
import numpy as np


class SB3BulletHellWrapper(gym.Wrapper):
    """
    Minimal wrapper to ensure BulletHellEnv works with Stable-Baselines3.
    
    The main purpose is to ensure step() signature matches SB3 expectations
    and to handle any edge cases with observation/action spaces.
    """
    
    def __init__(self, env):
        """
        Args:
            env: BulletHellEnv instance
        """
        super().__init__(env)
        
        # SB3 expects these attributes to exist
        # Our env already has them, but we ensure they're properly formatted
        assert isinstance(self.observation_space, gym.spaces.Box), \
            "Observation space must be gym.spaces.Box"
        assert isinstance(self.action_space, gym.spaces.Discrete), \
            "Action space must be gym.spaces.Discrete"
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation
            info: Info dict
        """
        # BulletHellEnv already returns (obs, info) tuple
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-8 for our 9 discrete actions)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (death)
            truncated: Whether episode was truncated (not used in our env)
            info: Additional info dict
        """
        # Pass q_values=None since PPO doesn't use Q-values
        obs, reward, terminated, truncated, info = self.env.step(action, q_values=None)
        return obs, reward, terminated, truncated, info
