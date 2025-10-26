#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from dbd_env import DBDEnv


CURRENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = CURRENT_DIR / "checkpoints"
CHECKPOINT_ROOT.mkdir(exist_ok=True)
LOG_DIR = CURRENT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class AdaptationProgressCallback(BaseCallback):
    def __init__(self, checkpoint_freq, checkpoint_root, report_freq, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_root = checkpoint_root
        self.report_freq = report_freq
        self.total_timesteps = total_timesteps
        self.best_mean_reward = -np.inf
        self.iteration = 0
        self.last_report_step = 0
        self.debug_printed = False
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_report_step >= self.report_freq:
            self._report_progress()
            self.last_report_step = self.num_timesteps
        if self.num_timesteps % self.checkpoint_freq == 0 and self.num_timesteps > 0:
            self._save_checkpoint()
        return True
    
    def _report_progress(self):
        self.iteration += 1
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            n_episodes = len(self.model.ep_info_buffer)
        else:
            mean_reward, mean_length, n_episodes = 0.0, 0.0, 0
        total_iters = self.total_timesteps // self.report_freq
        if n_episodes > 0:
            print(f"Iter {self.iteration:2d}/{total_iters} | Reward: {mean_reward:8.4f} | Episodes: {n_episodes:5d} | Steps: {self.num_timesteps:7d}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"  ★ New best: {self.best_mean_reward:.4f}")
        else:
            print(f"Iter {self.iteration:2d}/{total_iters} | Collecting samples...")
    
    def _save_checkpoint(self):
        checkpoint_path = self.checkpoint_root / f"ppo_adaptation_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        if self.verbose > 0:
            print(f"  ✓ Checkpoint: {checkpoint_path.name}")


def create_env():
    env = DBDEnv()
    return Monitor(env)


def train():
    total_timesteps = 100000
    n_steps = 1000
    checkpoint_freq = 2000
    report_freq = 1000
    env = DummyVecEnv([create_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=2000,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=10.0,
        ent_coef=1e-3,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(LOG_DIR),
        policy_kwargs={"net_arch": [128, 128], "activation_fn": nn.ReLU},
        device="auto",
    )
    
    progress_callback = AdaptationProgressCallback(
        checkpoint_freq=checkpoint_freq,
        checkpoint_root=CHECKPOINT_ROOT,
        report_freq=report_freq,
        total_timesteps=total_timesteps,
        verbose=1
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback,
        progress_bar=False,
    )
    
    final_path = CHECKPOINT_ROOT / "ppo_adaptation_final.zip"
    model.save(final_path)
    env.close()
    return model


def test_adaptation(model_path, n_episodes=10):
    """
    Test the trained adaptation model on randomized scenarios.
    
    Each episode will have randomized:
    - Polarization error: [0, 0.2] (0-20%)
    - Momentum sigma: [0.03, 0.07] ℏkL
    """
    model = PPO.load(model_path)
    env = create_env()
    
    episode_rewards = []
    episode_info = []
    
    for episode in range(n_episodes):
        obs, reset_info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Extract randomized parameters from reset_info dict
        initial_pol_error = reset_info.get('initial_polarization_error', 0.0)
        initial_momentum_sigma = reset_info.get('initial_momentum_sigma', 0.05)
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            steps += 1
        
        # Get final parameters (may have drifted)
        final_pol_error = info.get('current_polarization_error', initial_pol_error)
        final_momentum_sigma = info.get('current_momentum_sigma', initial_momentum_sigma)
        beam_score = info.get('beam_splitting_score', 0.0) if isinstance(info, dict) else 0.0
        
        episode_rewards.append(episode_reward)
        episode_info.append({
            'reward': episode_reward,
            'steps': steps,
            'initial_pol_error': initial_pol_error,
            'final_pol_error': final_pol_error,
            'initial_momentum_sigma': initial_momentum_sigma,
            'final_momentum_sigma': final_momentum_sigma,
            'beam_split': beam_score
        })
        
        print(f"Episode {episode + 1:2d}: Reward={episode_reward:7.3f} | Efficiency={beam_score:.4f} | "
              f"Pol={initial_pol_error:.3f}→{final_pol_error:.3f} | "
              f"σp={initial_momentum_sigma:.4f}→{final_momentum_sigma:.4f}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_efficiency = np.mean([ep['beam_split'] for ep in episode_info]) if episode_info else 0.0
    std_efficiency = np.std([ep['beam_split'] for ep in episode_info]) if episode_info else 0.0
    
    print(f"\n{'='*70}")
    print(f"Test Results Summary")
    print(f"{'='*70}")
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean efficiency: {mean_efficiency:.4f} ± {std_efficiency:.4f}")
    
    # Print distribution of initial and final parameters
    initial_pol_errors = [ep['initial_pol_error'] for ep in episode_info]
    final_pol_errors = [ep['final_pol_error'] for ep in episode_info]
    initial_momentum_sigmas = [ep['initial_momentum_sigma'] for ep in episode_info]
    final_momentum_sigmas = [ep['final_momentum_sigma'] for ep in episode_info]
    
    print(f"\n{'='*70}")
    print(f"Parameter Adaptation Analysis")
    print(f"{'='*70}")
    print(f"Initial Polarization Error:")
    print(f"  Mean: {np.mean(initial_pol_errors):.3f} ± {np.std(initial_pol_errors):.3f}")
    print(f"  Range: [{np.min(initial_pol_errors):.3f}, {np.max(initial_pol_errors):.3f}]")
    
    print(f"\nFinal Polarization Error (after drift):")
    print(f"  Mean: {np.mean(final_pol_errors):.3f} ± {np.std(final_pol_errors):.3f}")
    print(f"  Range: [{np.min(final_pol_errors):.3f}, {np.max(final_pol_errors):.3f}]")
    
    print(f"\nInitial Momentum Sigma (σp):")
    print(f"  Mean: {np.mean(initial_momentum_sigmas):.4f} ± {np.std(initial_momentum_sigmas):.4f}")
    print(f"  Range: [{np.min(initial_momentum_sigmas):.4f}, {np.max(initial_momentum_sigmas):.4f}]")
    
    print(f"\nFinal Momentum Sigma (after drift):")
    print(f"  Mean: {np.mean(final_momentum_sigmas):.4f} ± {np.std(final_momentum_sigmas):.4f}")
    print(f"  Range: [{np.min(final_momentum_sigmas):.4f}, {np.max(final_momentum_sigmas):.4f}]")
    
    # Calculate average drift
    pol_drift = np.mean([abs(ep['final_pol_error'] - ep['initial_pol_error']) for ep in episode_info])
    mom_drift = np.mean([abs(ep['final_momentum_sigma'] - ep['initial_momentum_sigma']) for ep in episode_info])
    
    print(f"\nAverage Parameter Drift:")
    print(f"  Polarization: {pol_drift:.4f}")
    print(f"  Momentum: {mom_drift:.5f}")
    print(f"{'='*70}")
    
    env.close()
    return mean_reward, episode_info


if __name__ == "__main__":
    train()
    test_adaptation("checkpoints/ppo_adaptation_final.zip", n_episodes=20)