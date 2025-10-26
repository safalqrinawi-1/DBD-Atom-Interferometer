#!/usr/bin/env python3
import os
import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dbd_env import DBDEnv

ENABLE_REWARD_DEBUG_WRAPPER = False
TOTAL_TIMESTEPS = 100000
N_ENVS = 4
N_STEPS = 1000
CHECKPOINT_FREQ = 2000
REPORT_FREQ = 1000

CURRENT_DIR = Path(__file__).resolve().parent
LOG_DIR = CURRENT_DIR / "logs"
CHECKPOINT_DIR = CURRENT_DIR / "checkpoints"
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


def make_debug_reward_env():
    class DebugRewardEnv(DBDEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._printed = False
        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            if not self._printed:
                print(f"[env dbg] r={float(r):+.4f} term={term} trunc={trunc} "
                      f"beam={info.get('beam_splitting_score', 0):.4f}")
            if term or trunc:
                self._printed = True
            return obs, r, term, trunc, info
    return DebugRewardEnv


class ProgressCallback(BaseCallback):
    def __init__(self, checkpoint_freq, checkpoint_root, report_freq, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_root = checkpoint_root
        self.report_freq = report_freq
        self.total_timesteps = total_timesteps
        self.best_mean_reward = -np.inf
        self.iteration = 0
        self.last_report_step = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_report_step >= self.report_freq:
            self._report_progress()
            self.last_report_step = self.num_timesteps
        if self.num_timesteps % self.checkpoint_freq == 0:
            self._save_checkpoint()
        return True
    
    def _report_progress(self):
        self.iteration += 1
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        else:
            mean_reward = 0.0
        print(f"Iter {self.iteration:2d} | Reward: {mean_reward:8.4f} | Steps: {self.num_timesteps:7d}")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            print(f"New best: {self.best_mean_reward:.4f}")
    
    def _save_checkpoint(self):
        checkpoint_path = self.checkpoint_root / f"ppo_checkpoint_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        if self.verbose > 0:
            print(f"Checkpoint: {checkpoint_path.name}")


def create_env():
    if ENABLE_REWARD_DEBUG_WRAPPER:
        DebugEnv = make_debug_reward_env()
        env = DebugEnv()
    else:
        env = DBDEnv()
    return Monitor(env)


def make_env(rank, seed=0):
    def _init():
        env = create_env()
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    use_subproc = N_ENVS > 1
    env = (
        SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        if use_subproc
        else DummyVecEnv([create_env])
    )

    import torch.nn as nn
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
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
        policy_kwargs={
            "net_arch": [128, 128],
            "activation_fn": nn.ReLU,
        },
        device="auto",
    )

    callback = ProgressCallback(
        checkpoint_freq=CHECKPOINT_FREQ,
        checkpoint_root=CHECKPOINT_DIR,
        report_freq=REPORT_FREQ,
        total_timesteps=TOTAL_TIMESTEPS,
    )

    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False,
    )

    model.save(CHECKPOINT_DIR / "ppo_final.zip")
    env.close()
    print(f"Training complete. Best reward: {callback.best_mean_reward:.4f}")
    return model


def test_trained_model(model_path, n_episodes=10):
    model = PPO.load(model_path)
    env = create_env()
    
    rewards = []
    efficiencies = []  # Track beam splitting efficiency
    
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(np.asarray(reward).sum())
        
        rewards.append(ep_reward)
        efficiencies.append(info['beam_splitting_score'])  # Store final efficiency
        print(f"Episode {i+1:2d}: Reward = {ep_reward:.3f}, Efficiency = {info['beam_splitting_score']:.4f}")
    
    env.close()
    
    mean_reward, std_reward = np.mean(rewards), np.std(rewards)
    mean_eff, std_eff = np.mean(efficiencies), np.std(efficiencies)
    
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean efficiency: {mean_eff:.4f} ± {std_eff:.4f}")
    
    return mean_reward


if __name__ == "__main__":
    model = train()
    test_trained_model(CHECKPOINT_DIR / "ppo_final.zip")

